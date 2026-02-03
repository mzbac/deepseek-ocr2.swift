@testable import DeepSeekOCR2
import CoreImage
import MLX
import MLXNN
import XCTest

final class DeepSeekOCR2Tests: XCTestCase {
    private func loadSanitizedDeepseekV2Weights(
        _ model: Module,
        config: DeepseekV2Configuration,
        fromSafetensors fileURL: URL,
        filterKeys: (String) -> Bool
    ) throws {
        let weights = try WeightsLoader.loadSafetensors(
            fromSafetensorsFile: fileURL,
            filterKeys: filterKeys
        )
        let sanitized = DeepseekV2Weights.sanitizeMoEExpertWeights(weights, config: config)
        let parameters = ModuleParameters.unflattened(sanitized)
        try model.update(parameters: parameters, verify: .all)
        eval(model)
    }

    func testKVCacheSimpleBatchUpdate() {
        let cache = KVCacheSimple()

        let keys1 = MLXArray(Int32(0) ..< Int32(2 * 1 * 1 * 2))
            .asType(.float32)
            .reshaped(2, 1, 1, 2)
        let values1 = keys1 + MLXArray(Float32(100))

        let (cachedKeys1, cachedValues1) = cache.update(keys: keys1, values: values1)
        XCTAssertEqual(cachedKeys1.shape, [2, 1, 1, 2])
        XCTAssertEqual(cachedValues1.shape, [2, 1, 1, 2])

        let maxDiffKeys1 = abs(cachedKeys1 - keys1).max().item(Float.self)
        let maxDiffValues1 = abs(cachedValues1 - values1).max().item(Float.self)
        XCTAssertLessThanOrEqual(maxDiffKeys1, 0, "maxDiffKeys1=\(maxDiffKeys1)")
        XCTAssertLessThanOrEqual(maxDiffValues1, 0, "maxDiffValues1=\(maxDiffValues1)")

        let keys2 = keys1 + MLXArray(Float32(10))
        let values2 = values1 + MLXArray(Float32(10))

        let (cachedKeys2, cachedValues2) = cache.update(keys: keys2, values: values2)
        XCTAssertEqual(cachedKeys2.shape, [2, 1, 2, 2])
        XCTAssertEqual(cachedValues2.shape, [2, 1, 2, 2])

        let expectedKeys2 = concatenated([keys1, keys2], axis: 2)
        let expectedValues2 = concatenated([values1, values2], axis: 2)

        let maxDiffKeys2 = abs(cachedKeys2 - expectedKeys2).max().item(Float.self)
        let maxDiffValues2 = abs(cachedValues2 - expectedValues2).max().item(Float.self)
        XCTAssertLessThanOrEqual(maxDiffKeys2, 0, "maxDiffKeys2=\(maxDiffKeys2)")
        XCTAssertLessThanOrEqual(maxDiffValues2, 0, "maxDiffValues2=\(maxDiffValues2)")

        cache.reset()

        // Exercise the resize path (step=256) with a long prompt update.
        let promptLen = 260
        let keysLong = MLXArray(Int32(0) ..< Int32(2 * 1 * promptLen * 2))
            .asType(.float32)
            .reshaped(2, 1, promptLen, 2)
        let valuesLong = keysLong + MLXArray(Float32(1000))

        let (cachedKeysLong, cachedValuesLong) = cache.update(keys: keysLong, values: valuesLong)
        XCTAssertEqual(cachedKeysLong.shape, [2, 1, promptLen, 2])
        XCTAssertEqual(cachedValuesLong.shape, [2, 1, promptLen, 2])

        let maxDiffKeysLong = abs(cachedKeysLong - keysLong).max().item(Float.self)
        let maxDiffValuesLong = abs(cachedValuesLong - valuesLong).max().item(Float.self)
        XCTAssertLessThanOrEqual(maxDiffKeysLong, 0, "maxDiffKeysLong=\(maxDiffKeysLong)")
        XCTAssertLessThanOrEqual(maxDiffValuesLong, 0, "maxDiffValuesLong=\(maxDiffValuesLong)")

        let originalOffset = cache.offset
        let sliced = cache.slice(batchIndex: 1)
        XCTAssertEqual(sliced.offset, originalOffset)

        let keys3 = MLXArray.ones([1, 1, 1, 2], dtype: .float32)
        let values3 = keys3 + MLXArray(Float32(123))
        _ = sliced.update(keys: keys3, values: values3)
        XCTAssertEqual(cache.offset, originalOffset, "Updating sliced cache should not mutate original.")
    }

    func testImageProcessorRejectsUnsupportedSizes() {
        XCTAssertThrowsError(try DeepseekOCR2ImageProcessor(baseSize: 512))
        XCTAssertThrowsError(try DeepseekOCR2ImageProcessor(localSize: 512))
    }

    func testImageProcessorRejectsEmptyImage() throws {
        let processor = try DeepseekOCR2ImageProcessor(cropMode: false)
        let empty = CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 1)).cropped(to: .zero)
        XCTAssertThrowsError(try processor.process(empty))
    }

    func testDeepseekV2TinyParity() throws {
        guard let configURL = Bundle.module.url(forResource: "config", withExtension: "json") else {
            XCTFail("Missing config.json resource")
            return
        }
        guard let weightsURL = Bundle.module.url(forResource: "weights", withExtension: "safetensors") else {
            XCTFail("Missing weights.safetensors resource")
            return
        }
        guard let parityURL = Bundle.module.url(forResource: "parity", withExtension: "safetensors") else {
            XCTFail("Missing parity.safetensors resource")
            return
        }

        let config = try DeepseekV2Configuration.load(from: configURL)
        let model = DeepseekV2ForCausalLM(config: config)
        try loadSanitizedDeepseekV2Weights(
            model,
            config: config,
            fromSafetensors: weightsURL,
            filterKeys: { $0.hasPrefix("model.") || $0.hasPrefix("lm_head.") }
        )

        let parity = try MLX.loadArrays(url: parityURL)
        guard let inputIds = parity["input_ids"] else {
            XCTFail("Missing input_ids in parity.safetensors")
            return
        }
        guard let expected = parity["expected_logits"] else {
            XCTFail("Missing expected_logits in parity.safetensors")
            return
        }

        let actual = model(inputIds.asType(.int32)).asType(.float32)
        let expectedFloat = expected.asType(.float32)
        let maxDiff = abs(actual - expectedFloat).max().item(Float.self)

        XCTAssertLessThanOrEqual(maxDiff, 5e-4, "maxDiff=\(maxDiff)")
    }

    func testDeepseekV2TinyBatchCacheDeterminism() throws {
        guard let configURL = Bundle.module.url(forResource: "config", withExtension: "json") else {
            XCTFail("Missing config.json resource")
            return
        }
        guard let weightsURL = Bundle.module.url(forResource: "weights", withExtension: "safetensors") else {
            XCTFail("Missing weights.safetensors resource")
            return
        }
        guard let parityURL = Bundle.module.url(forResource: "parity", withExtension: "safetensors") else {
            XCTFail("Missing parity.safetensors resource")
            return
        }

        let config = try DeepseekV2Configuration.load(from: configURL)
        let model = DeepseekV2ForCausalLM(config: config)
        try loadSanitizedDeepseekV2Weights(
            model,
            config: config,
            fromSafetensors: weightsURL,
            filterKeys: { $0.hasPrefix("model.") || $0.hasPrefix("lm_head.") }
        )

        let parity = try MLX.loadArrays(url: parityURL)
        guard let inputIdsSingle = parity["input_ids"] else {
            XCTFail("Missing input_ids in parity.safetensors")
            return
        }

        let inputIds = inputIdsSingle.asType(.int32)
        let inputIdsBatch = concatenated([inputIds, inputIds], axis: 0)

        let cache: [KVCache] = (0..<config.numHiddenLayers).map { _ in KVCacheSimple() }
        var logits = model(inputIdsBatch, cache: cache)

        for _ in 0..<8 {
            let lastLogits = logits[0..., -1]
            let nextTokens = argMax(lastLogits, axis: -1).asArray(Int32.self)
            XCTAssertEqual(nextTokens.count, 2)
            XCTAssertEqual(nextTokens[0], nextTokens[1], "Batch elements diverged in greedy next-token selection.")

            let next = MLXArray(nextTokens).reshaped(2, 1)
            logits = model(next, cache: cache)
        }
    }

    func testDeepseekV2BatchCacheGQAManualAttentionDoesNotCrash() {
        let config = DeepseekV2Configuration(
            vocabSize: 32,
            hiddenSize: 32,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            maxPositionEmbeddings: 64,
            attentionBias: false,
            attentionDropout: 0,
            useMLA: false
        )

        let model = DeepseekV2ForCausalLM(config: config)
        let cache: [KVCache] = (0..<config.numHiddenLayers).map { _ in KVCacheSimple() }

        // batch>1, seqLen==1, cache!=nil triggers the deterministic manual attention path.
        let inputIds = MLXArray([Int32(1), Int32(2)]).reshaped(2, 1)
        let logits1 = model(inputIds, cache: cache)
        XCTAssertEqual(logits1.shape, [2, 1, config.vocabSize])

        let nextIds = MLXArray([Int32(3), Int32(4)]).reshaped(2, 1)
        let logits2 = model(nextIds, cache: cache)
        XCTAssertEqual(logits2.shape, [2, 1, config.vocabSize])

        for layerCache in cache {
            XCTAssertEqual(layerCache.offset, 2)
        }
    }

    func testDeepseekV2RaggedBatchCacheMatchesSingle() {
        MLXRandom.seed(0)

        let tol: Float = 1e-5

        let config = DeepseekV2Configuration(
            vocabSize: 64,
            hiddenSize: 32,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 2,
            maxPositionEmbeddings: 64,
            attentionBias: false,
            attentionDropout: 0,
            useMLA: false
        )

        let model = DeepseekV2ForCausalLM(config: config)

        let a: [Int32] = [1, 2, 3, 4]
        let b: [Int32] = [1, 2, 3, 4, 5, 6]
        let lengths = [a.count, b.count]
        let maxLen = lengths.max() ?? 0
        XCTAssertGreaterThan(maxLen, 0)

        let pad: Int32 = 0
        let aPadded = a + Array(repeating: pad, count: max(0, maxLen - a.count))
        let bPadded = b + Array(repeating: pad, count: max(0, maxLen - b.count))
        let batchInput = MLXArray(aPadded + bPadded).reshaped(2, maxLen)

        let ragged: [KVCache] = (0..<config.numHiddenLayers).map { _ in KVCacheRaggedSimple() }
        var logitsBatch = model(batchInput, cache: ragged).asType(.float32)
        for cache in ragged {
            (cache as? KVCacheRagged)?.setOffsets(lengths)
        }

        let singleA: [KVCache] = (0..<config.numHiddenLayers).map { _ in KVCacheSimple() }
        let singleB: [KVCache] = (0..<config.numHiddenLayers).map { _ in KVCacheSimple() }
        var logitsA = model(MLXArray(a).reshaped(1, a.count), cache: singleA).asType(.float32)
        var logitsB = model(MLXArray(b).reshaped(1, b.count), cache: singleB).asType(.float32)

        let lastIndices = MLXArray(lengths.map { Int32($0 - 1) }).reshaped(2, 1, 1)
        let lastIndicesBroad = broadcast(lastIndices, to: [2, 1, config.vocabSize])
        let lastBatch = takeAlong(logitsBatch, lastIndicesBroad, axis: 1).squeezed(axis: 1)

        let lastA = logitsA[0, a.count - 1]
        let lastB = logitsB[0, b.count - 1]

        XCTAssertLessThanOrEqual(abs(lastBatch[0] - lastA).max().item(Float.self), tol)
        XCTAssertLessThanOrEqual(abs(lastBatch[1] - lastB).max().item(Float.self), tol)

        for _ in 0..<3 {
            let nextTokens = argMax(lastBatch, axis: -1).asArray(Int32.self)
            XCTAssertEqual(nextTokens.count, 2)

            let nextBatch = MLXArray(nextTokens).reshaped(2, 1)
            logitsBatch = model(nextBatch, cache: ragged).asType(.float32)

            logitsA = model(MLXArray([nextTokens[0]]).reshaped(1, 1), cache: singleA).asType(.float32)
            logitsB = model(MLXArray([nextTokens[1]]).reshaped(1, 1), cache: singleB).asType(.float32)

            XCTAssertLessThanOrEqual(abs(logitsBatch[0, 0] - logitsA[0, 0]).max().item(Float.self), tol)
            XCTAssertLessThanOrEqual(abs(logitsBatch[1, 0] - logitsB[0, 0]).max().item(Float.self), tol)
        }
    }

    func testGatherMMBatchedDeterminism() {
        MLXRandom.seed(0)

        let batch = 2
        let seqLen = 1
        let inputDims = 32
        let outputDims = 16
        let numExperts = 8
        let topK = 4

        let token = MLXRandom.normal([1, seqLen, inputDims], dtype: .float32)
        let x = concatenated([token, token], axis: 0)

        let weight = MLXRandom.normal([numExperts, outputDims, inputDims], dtype: .float32)
        let weightT = weight.swappedAxes(-1, -2)

        let baseIdx: [Int32] = [1, 3, 5, 7]
        let indices = MLXArray(baseIdx + baseIdx).reshaped(batch, seqLen, topK)

        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let y = MLX.gatherMM(expanded, weightT, rhsIndices: indices, sortedIndices: false)
        eval(y)

        // identical inputs/indices must produce identical outputs across batch
        let maxDiff = abs(y[0] - y[1]).max().item(Float.self)
        XCTAssertLessThanOrEqual(maxDiff, 0, "gatherMM batch elements differ (maxDiff=\(maxDiff))")
    }

    func testDeepseekOCR2TinyImageInjectionParity() throws {
        guard let configURL = Bundle.module.url(forResource: "config", withExtension: "json") else {
            XCTFail("Missing config.json resource")
            return
        }
        guard let weightsURL = Bundle.module.url(forResource: "weights", withExtension: "safetensors") else {
            XCTFail("Missing weights.safetensors resource")
            return
        }
        guard let parityURL = Bundle.module.url(forResource: "parity_inject", withExtension: "safetensors") else {
            XCTFail("Missing parity_inject.safetensors resource")
            return
        }

        let config = try DeepseekV2Configuration.load(from: configURL)
        let model = DeepseekOCR2InjectedForCausalLM(config: config)
        try loadSanitizedDeepseekV2Weights(
            model,
            config: config,
            fromSafetensors: weightsURL,
            filterKeys: { $0.hasPrefix("model.") || $0.hasPrefix("lm_head.") }
        )

        let parity = try MLX.loadArrays(url: parityURL)
        guard let inputIds = parity["input_ids"] else {
            XCTFail("Missing input_ids in parity_inject.safetensors")
            return
        }
        guard let imageEmbeds = parity["image_embeds"] else {
            XCTFail("Missing image_embeds in parity_inject.safetensors")
            return
        }
        guard let imageSeqMask = parity["image_seq_mask"] else {
            XCTFail("Missing image_seq_mask in parity_inject.safetensors")
            return
        }
        guard let expected = parity["expected_logits"] else {
            XCTFail("Missing expected_logits in parity_inject.safetensors")
            return
        }

        let actual = try model.forward(
            inputIds: inputIds.asType(.int32),
            imageEmbeds: imageEmbeds,
            imageSeqMask: imageSeqMask
        )
        .asType(.float32)
        let expectedFloat = expected.asType(.float32)
        let maxDiff = abs(actual - expectedFloat).max().item(Float.self)

        XCTAssertLessThanOrEqual(maxDiff, 5e-4, "maxDiff=\(maxDiff)")
    }
}
