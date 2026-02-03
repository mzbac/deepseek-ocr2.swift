import Foundation
import MLX
import Tokenizers

public final class DeepseekOCR2Generator {
    public struct GenerationResult: Sendable {
        public let tokens: [Int]
        public let text: String
        public let tokenCount: Int
    }

    private static let deepseekEndOfSentence = "<｜end▁of▁sentence｜>"

    public let model: DeepseekOCR2ForCausalLM
    public let tokenizer: any Tokenizer

    public let imageTokenId: Int
    public let stopTokenIds: Set<Int>
    private let stopTokenIdsInt32Sorted: [Int32]
    private let stopTokenIdsArray: MLXArray?

    public init(
        model: DeepseekOCR2ForCausalLM,
        tokenizer: any Tokenizer,
        imageTokenId: Int? = nil,
        stopTokenIds: Set<Int>? = nil
    ) {
        self.model = model
        self.tokenizer = tokenizer
        self.imageTokenId = imageTokenId ?? model.imageTokenId

        let eosTokenId = tokenizer.eosTokenId ?? 1
        var defaultStopTokenIds: Set<Int> = [eosTokenId]
        if let deepseekEosId = tokenizer.convertTokenToId(Self.deepseekEndOfSentence) {
            defaultStopTokenIds.insert(deepseekEosId)
        }
        let resolvedStopTokenIds = stopTokenIds ?? defaultStopTokenIds
        self.stopTokenIds = resolvedStopTokenIds
        let sorted = resolvedStopTokenIds.map { Int32($0) }.sorted()
        self.stopTokenIdsInt32Sorted = sorted
        if sorted.isEmpty {
            self.stopTokenIdsArray = nil
        } else {
            self.stopTokenIdsArray = MLXArray(sorted).asType(.int32)
        }
    }

    public func generate(
        processedImages: DeepseekOCR2ProcessedImages,
        prompt: String,
        maxNewTokens: Int = 4096,
        temperature: Float = 0.0,
        topP: Float = 1.0
    ) throws -> GenerationResult {
        let results = try generateBatch(
            processedImages: [processedImages],
            prompt: prompt,
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topP: topP
        )
        guard let first = results.first else {
            throw DeepseekOCR2Error.modelLoadFailed("Failed to generate: empty batch output.")
        }
        return first
    }

    public func generateBatch(
        processedImages: [DeepseekOCR2ProcessedImages],
        prompt: String,
        maxNewTokens: Int = 4096,
        temperature: Float = 0.0,
        topP: Float = 1.0
    ) throws -> [GenerationResult] {
        try generateBatchSharedDecode(
            processedImages: processedImages,
            prompt: prompt,
            maxNewTokens: maxNewTokens,
            temperature: temperature,
            topP: topP
        )
    }

    public func generateBatchSharedDecode(
        processedImages: [DeepseekOCR2ProcessedImages],
        prompt: String,
        maxNewTokens: Int = 4096,
        temperature: Float = 0.0,
        topP: Float = 1.0
    ) throws -> [GenerationResult] {
        guard !processedImages.isEmpty else { return [] }

        let totalBatch = processedImages.count
        if maxNewTokens <= 0 {
            return Array(repeating: GenerationResult(tokens: [], text: "", tokenCount: 0), count: totalBatch)
        }

        let (preIds, postIds) = try tokenizePromptParts(prompt: prompt)
        let preIdsInt32 = preIds.map { Int32($0) }
        let postIdsInt32 = postIds.map { Int32($0) }
        let bosTokenId = tokenizer.bosTokenId ?? 0
        let padTokenId = tokenizer.eosTokenId ?? bosTokenId

        let promptLengths: [Int] = processedImages.map { image in
            1 + preIds.count + image.numImageTokens + postIds.count
        }
        let maxPromptLen = promptLengths.max() ?? 0

        var flatInputIds: [Int32] = []
        flatInputIds.reserveCapacity(totalBatch * maxPromptLen)
        for (image, length) in zip(processedImages, promptLengths) {
            let numPads = maxPromptLen - length

            flatInputIds.append(Int32(bosTokenId))
            flatInputIds.append(contentsOf: preIdsInt32)
            flatInputIds.append(contentsOf: Array(repeating: Int32(imageTokenId), count: image.numImageTokens))
            flatInputIds.append(contentsOf: postIdsInt32)
            if numPads > 0 {
                flatInputIds.append(contentsOf: Array(repeating: Int32(padTokenId), count: numPads))
            }
        }
        let inputIdArray = MLXArray(flatInputIds).reshaped(totalBatch, maxPromptLen)

        let globalViews = concatenated(processedImages.map { $0.globalView }, axis: 0)

        let cropsPerImage = processedImages.map { $0.crops?.dim(0) ?? 0 }
        let totalCrops = cropsPerImage.reduce(0, +)
        let cropsFlat: MLXArray? = {
            guard totalCrops > 0 else { return nil }
            let crops = processedImages.compactMap { $0.crops }
            guard !crops.isEmpty else { return nil }
            return concatenated(crops, axis: 0)
        }()

        var cache = model.newCache()
        let logits = try model.forwardPrompt(
            inputIds: inputIdArray,
            globalView: globalViews,
            crops: cropsFlat,
            cropsPerImage: cropsPerImage,
            cache: cache
        )
        for layerCache in cache {
            (layerCache as? KVCacheRagged)?.setOffsets(promptLengths)
        }

        let vocabSize = logits.dim(2)
        let lastIndices = MLXArray(promptLengths.map { Int32(max($0 - 1, 0)) }).reshaped(totalBatch, 1, 1)
        let lastIndicesBroad = broadcast(lastIndices, to: [totalBatch, 1, vocabSize])
        let lastLogits = takeAlong(logits, lastIndicesBroad, axis: 1).squeezed(axis: 1)

        var batch = totalBatch
        var activeOriginalIndices = Array(0..<totalBatch)
        var results: [GenerationResult?] = Array(repeating: nil, count: totalBatch)

        var tokenBuffer = MLXArray.zeros([batch, maxNewTokens], dtype: .int32)
        var finished = MLXArray.zeros([batch], dtype: .bool)
        let eosTokenId = tokenizer.eosTokenId ?? padTokenId
        var eosTokens = broadcast(MLXArray(Int32(eosTokenId)), to: [batch])
        var stopCheckInterval = max(1, min(4, batch))

        // Prime the pipeline: compute the first token and evaluate it asynchronously.
        var currentTokens = sampleBatchToken(
            logits: lastLogits,
            temperature: temperature,
            topP: topP
        )
        asyncEval(currentTokens)

        var stepsGenerated = 0
        for step in 0..<maxNewTokens {
            var tokensToProcess = currentTokens

            tokenBuffer[0..., step] = tokensToProcess
            finished = finished .|| stopMask(tokensToProcess)
            stepsGenerated = step + 1
            if stepsGenerated >= maxNewTokens { break }

            if stepsGenerated % stopCheckInterval == 0 {
                eval(finished)
                let finishedMask = finished.asArray(Bool.self)
                if finishedMask.contains(true) {
                    for (i, didFinish) in finishedMask.enumerated() where didFinish {
                        let result = decodeTokenRow(tokenBuffer, row: i, steps: stepsGenerated)
                        results[activeOriginalIndices[i]] = result
                    }

                    let keepIndices = finishedMask.enumerated().compactMap { $0.element ? nil : $0.offset }
                    if keepIndices.isEmpty {
                        batch = 0
                        activeOriginalIndices.removeAll(keepingCapacity: true)
                        break
                    }

                    if keepIndices.count != batch {
                        activeOriginalIndices = keepIndices.map { activeOriginalIndices[$0] }
                        tokenBuffer = selectBatchRows(tokenBuffer, indices: keepIndices)
                        tokensToProcess = selectBatchRows(tokensToProcess, indices: keepIndices)

                        cache = cache.map { layerCache in
                            guard let ragged = layerCache as? KVCacheRaggedSimple else { return layerCache }
                            return ragged.selecting(batchIndices: keepIndices)
                        }

                        batch = keepIndices.count
                        finished = MLXArray.zeros([batch], dtype: .bool)
                        eosTokens = broadcast(MLXArray(Int32(eosTokenId)), to: [batch])
                        stopCheckInterval = max(1, min(4, batch))
                        currentTokens = tokensToProcess
                    }
                }
            }

            if batch == 0 {
                break
            }

            let stepLogits = model.forwardGeneration(
                inputIds: tokensToProcess.reshaped(batch, 1),
                cache: cache
            )
            currentTokens = sampleBatchToken(
                logits: stepLogits[0..., -1],
                temperature: temperature,
                topP: topP
            )
            currentTokens = which(finished, eosTokens, currentTokens)
            asyncEval(currentTokens, tokenBuffer, finished)
        }

        // Ensure any outstanding async evaluation is complete before returning.
        eval(tokenBuffer)

        if batch > 0 {
            let remaining = decodeTokenBuffer(tokenBuffer, steps: stepsGenerated)
            for (originalIndex, result) in zip(activeOriginalIndices, remaining) {
                results[originalIndex] = result
            }
        }

        return results.map {
            $0 ?? GenerationResult(tokens: [], text: "", tokenCount: 0)
        }
    }

    private func tokenizePromptParts(prompt: String) throws -> (pre: [Int], post: [Int]) {
        let imageToken = "<image>"
        var promptText = prompt
        if !promptText.contains(imageToken) {
            promptText = imageToken + "\n" + promptText
        }

        let splits = promptText.components(separatedBy: imageToken)
        guard splits.count == 2 else {
            throw DeepseekOCR2Error.shapeMismatch("Only single-image prompts are supported (expected exactly one <image> tag).")
        }

        let pre = splits[0]
        let post = splits[1]

        let preIds = tokenizer.encode(text: pre, addSpecialTokens: false)
        let postIds = tokenizer.encode(text: post, addSpecialTokens: false)
        return (preIds, postIds)
    }

    private func sampleBatchWithTemperature(logits: MLXArray, temperature: Float, topP: Float) -> MLXArray {
        precondition(logits.ndim == 2, "sampleBatchWithTemperature expects logits [batch, vocab]")

        let batch = logits.dim(0)
        let scaledLogits = logits / temperature

        if topP < 1.0 {
            let eps: Float = 1e-10
            let probs = softmax(scaledLogits, axis: -1)
            let sortedIndices = argSort(probs, axis: -1).asType(.int32)
            let sortedProbs = takeAlong(probs, sortedIndices, axis: -1)
            let cumulativeProbs = cumsum(sortedProbs, axis: -1)

            let topProbs = MLX.where(
                cumulativeProbs .> (Float(1) - topP),
                sortedProbs,
                zeros(like: sortedProbs)
            )

            let sortedToken = categorical(log(topProbs + eps), axis: -1).asType(.int32)
            let token = takeAlong(sortedIndices, sortedToken.reshaped(batch, 1), axis: -1).squeezed(axis: -1)
            return token.asType(.int32)
        }

        return categorical(scaledLogits, axis: -1).asType(.int32)
    }

    private func sampleBatchToken(logits: MLXArray, temperature: Float, topP: Float) -> MLXArray {
        if temperature <= 0 {
            return argMax(logits, axis: -1).asType(.int32)
        }
        return sampleBatchWithTemperature(logits: logits, temperature: temperature, topP: topP)
    }

    private func stopMask(_ tokens: MLXArray) -> MLXArray {
        precondition(tokens.ndim == 1, "stopMask expects tokens [batch]")

        guard let stopTokenIdsArray else {
            return MLXArray.zeros([tokens.dim(0)], dtype: .bool)
        }

        let batch = tokens.dim(0)
        let nStop = stopTokenIdsArray.dim(0)

        let tokens2d = tokens.reshaped(batch, 1)
        let stop2d = stopTokenIdsArray.reshaped(1, nStop)
        let matches = tokens2d .== stop2d
        return matches.any(axis: 1)
    }

    private func decodeTokenBuffer(_ tokenBuffer: MLXArray, steps: Int) -> [GenerationResult] {
        let batch = tokenBuffer.dim(0)
        guard steps > 0 else {
            return Array(repeating: GenerationResult(tokens: [], text: "", tokenCount: 0), count: batch)
        }

        let flatTokens = tokenBuffer[0..., 0..<steps].asArray(Int32.self)
        precondition(flatTokens.count == batch * steps, "Unexpected token buffer size")

        var results: [GenerationResult] = []
        results.reserveCapacity(batch)

        for b in 0..<batch {
            let start = b * steps
            let end = start + steps

            var tokens: [Int] = []
            tokens.reserveCapacity(min(steps, 512))

            for raw in flatTokens[start..<end] {
                let tokenId = Int(raw)
                if stopTokenIds.contains(tokenId) {
                    break
                }
                tokens.append(tokenId)
            }

            var text = tokenizer.decode(tokens: tokens)
            if text.hasSuffix(Self.deepseekEndOfSentence) {
                text = String(text.dropLast(Self.deepseekEndOfSentence.count))
                text = text.trimmingCharacters(in: .whitespacesAndNewlines)
            }
            results.append(GenerationResult(tokens: tokens, text: text, tokenCount: tokens.count))
        }

        return results
    }

    private func decodeTokenRow(_ tokenBuffer: MLXArray, row: Int, steps: Int) -> GenerationResult {
        guard steps > 0 else {
            return GenerationResult(tokens: [], text: "", tokenCount: 0)
        }

        let flat = tokenBuffer[row, 0..<steps].asArray(Int32.self)
        var tokens: [Int] = []
        tokens.reserveCapacity(min(steps, 512))

        for raw in flat {
            let tokenId = Int(raw)
            if stopTokenIds.contains(tokenId) {
                break
            }
            tokens.append(tokenId)
        }

        var text = tokenizer.decode(tokens: tokens)
        if text.hasSuffix(Self.deepseekEndOfSentence) {
            text = String(text.dropLast(Self.deepseekEndOfSentence.count))
            text = text.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return GenerationResult(tokens: tokens, text: text, tokenCount: tokens.count)
    }

    private func selectBatchRows(_ array: MLXArray, indices: [Int]) -> MLXArray {
        precondition(!indices.isEmpty, "selectBatchRows expects non-empty indices")
        precondition(array.ndim >= 1, "selectBatchRows expects rank >= 1 array")

        let newBatch = indices.count
        let indexArray = MLXArray(indices.map { Int32($0) }).asType(.int32)

        var reshaped: [Int] = [newBatch]
        if array.ndim > 1 {
            reshaped.append(contentsOf: Array(repeating: 1, count: array.ndim - 1))
        }
        let expanded = indexArray.reshaped(reshaped)

        var targetShape: [Int] = [newBatch]
        if array.ndim > 1 {
            targetShape.append(contentsOf: (1..<array.ndim).map { array.dim($0) })
        }
        let broadcasted = broadcast(expanded, to: targetShape)
        return takeAlong(array, broadcasted, axis: 0)
    }
}
