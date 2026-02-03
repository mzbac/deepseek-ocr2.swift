import Foundation
import MLX
import MLXNN

final class DeepseekOCR2Projector: Module, UnaryLayer {
    @ModuleInfo(key: "layers") var layers: Linear

    init(inputDim: Int, outputDim: Int) {
        self._layers.wrappedValue = Linear(inputDim, outputDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers(x)
    }
}

public final class DeepseekOCR2Model: Module {
    public let config: DeepseekV2Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    @ModuleInfo(key: "sam_model") var samModel: SamImageEncoderViT
    @ModuleInfo(key: "qwen2_model") var qwen2Model: Qwen2Decoder2Encoder
    @ModuleInfo(key: "projector") var projector: DeepseekOCR2Projector

    @ParameterInfo(key: "view_seperator") var viewSeperator: MLXArray

    init(config: DeepseekV2Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize
        )
        self.layers = (0..<config.numHiddenLayers).map { DeepseekV2DecoderLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)

        self._samModel.wrappedValue = SamImageEncoderViT()
        self._qwen2Model.wrappedValue = Qwen2Decoder2Encoder()
        self._projector.wrappedValue = DeepseekOCR2Projector(inputDim: 896, outputDim: config.hiddenSize)

        let embedStd = Float(1.0 / sqrt(Double(config.hiddenSize)))
        self._viewSeperator.wrappedValue = MLXRandom.normal([config.hiddenSize]) * embedStd

        super.init()
    }

    func embeddings(for inputIds: MLXArray) -> MLXArray {
        embedTokens(inputIds)
    }

    func forwardLanguage(inputsEmbeds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = inputsEmbeds
        let seqLen = h.dim(1)
        let offset = cache?.first?.offset ?? 0

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if seqLen == 1 {
            mask = .none
        } else if offset == 0 {
            mask = .causal
        } else {
            mask = .array(createCausalMask(n: seqLen, offset: offset))
        }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }

    func encodeImage(globalView: MLXArray, crops: MLXArray?) throws -> MLXArray {
        // globalView: [B, 1024, 1024, 3]
        // crops (optional): [B * P, 768, 768, 3] (or [P, ...] when B=1)
        let batch = globalView.dim(0)

        let globalVision = samModel(globalView)
        try validateVisionTokens(globalVision, expectedNQuery: 256, context: "globalView")
        let globalEncoded = qwen2Model(globalVision)
        let globalProjected = projector(globalEncoded)

        let viewSep = broadcast(
            viewSeperator.reshaped(1, 1, config.hiddenSize),
            to: [batch, 1, config.hiddenSize]
        )

        if let crops {
            let totalCrops = crops.dim(0)
            guard batch > 0 else {
                throw DeepseekOCR2Error.shapeMismatch("batch must be > 0")
            }
            guard totalCrops % batch == 0 else {
                throw DeepseekOCR2Error.shapeMismatch(
                    "crops batch dimension (\(totalCrops)) must be divisible by globalView batch (\(batch))"
                )
            }
            let cropsPerImage = totalCrops / batch

            let localVision = samModel(crops)
            try validateVisionTokens(localVision, expectedNQuery: 144, context: "crops")
            let localEncoded = qwen2Model(localVision)
            let localProjected = projector(localEncoded)
            let localTokens = localProjected.dim(1)
            let localFlat = localProjected.reshaped(batch, cropsPerImage * localTokens, config.hiddenSize)

            return concatenated([localFlat, globalProjected, viewSep], axis: 1)
        }

        return concatenated([globalProjected, viewSep], axis: 1)
    }

    func encodeImage(globalView: MLXArray, crops: MLXArray?, cropsPerImage: [Int]) throws -> MLXArray {
        // globalView: [B, 1024, 1024, 3]
        // crops (optional): [sum(P_i), 768, 768, 3]
        let batch = globalView.dim(0)
        guard cropsPerImage.count == batch else {
            throw DeepseekOCR2Error.shapeMismatch(
                "cropsPerImage count (\(cropsPerImage.count)) must equal batch (\(batch))"
            )
        }

        let globalVision = samModel(globalView)
        try validateVisionTokens(globalVision, expectedNQuery: 256, context: "globalView")
        let globalEncoded = qwen2Model(globalVision)
        let globalProjected = projector(globalEncoded)

        let viewSep = broadcast(
            viewSeperator.reshaped(1, 1, config.hiddenSize),
            to: [batch, 1, config.hiddenSize]
        )

        let totalCrops = crops?.dim(0) ?? 0
        if totalCrops == 0 {
            return concatenated([globalProjected, viewSep], axis: 1)
        }

        guard let crops else {
            throw DeepseekOCR2Error.shapeMismatch("crops must be non-nil when totalCrops > 0")
        }

        let expected = cropsPerImage.reduce(0, +)
        guard expected == totalCrops else {
            throw DeepseekOCR2Error.shapeMismatch(
                "crops dim(0) (\(totalCrops)) must equal sum(cropsPerImage) (\(expected))"
            )
        }

        let localVision = samModel(crops)
        try validateVisionTokens(localVision, expectedNQuery: 144, context: "crops")
        let localEncoded = qwen2Model(localVision)
        let localProjected = projector(localEncoded)
        let localTokens = localProjected.dim(1)

        var perImage: [MLXArray] = []
        perImage.reserveCapacity(batch)

        var maxTokens = 0
        var cropStart = 0
        for b in 0..<batch {
            let cropCount = cropsPerImage[b]

            let global = globalProjected[b, 0..., 0...].reshaped(1, globalProjected.dim(1), config.hiddenSize)
            let sep = viewSep[b, 0..., 0...].reshaped(1, 1, config.hiddenSize)

            if cropCount > 0 {
                let cropEnd = cropStart + cropCount
                let localSlice = localProjected[cropStart..<cropEnd, 0..., 0...]
                let localFlat = localSlice.reshaped(1, cropCount * localTokens, config.hiddenSize)
                let combined = concatenated([localFlat, global, sep], axis: 1)
                maxTokens = max(maxTokens, combined.dim(1))
                perImage.append(combined)
                cropStart = cropEnd
            } else {
                let combined = concatenated([global, sep], axis: 1)
                maxTokens = max(maxTokens, combined.dim(1))
                perImage.append(combined)
            }
        }

        let paddedPerImage = perImage.map { features in
            let tokens = features.dim(1)
            guard tokens < maxTokens else { return features }
            return padded(features, widths: [[0, 0], [0, maxTokens - tokens], [0, 0]])
        }

        return concatenated(paddedPerImage, axis: 0)
    }

    private func validateVisionTokens(_ vision: MLXArray, expectedNQuery: Int, context: String) throws {
        guard vision.ndim == 4 else {
            throw DeepseekOCR2Error.shapeMismatch("\(context) vision features must be rank-4 [B, H, W, C]")
        }

        let h = vision.dim(1)
        let w = vision.dim(2)
        let c = vision.dim(3)
        let nQuery = h * w

        guard nQuery == expectedNQuery else {
            throw DeepseekOCR2Error.shapeMismatch(
                "\(context) produced unexpected patch grid: H=\(h) W=\(w) (nQuery=\(nQuery)), expected nQuery=\(expectedNQuery)"
            )
        }
        guard c == 896 else {
            throw DeepseekOCR2Error.shapeMismatch("\(context) vision features must have channel dimension 896 (got C=\(c))")
        }
    }

    func mergeInputIdsWithImageFeatures(inputIds: MLXArray, imageFeatures: MLXArray, imageTokenId: Int) throws -> MLXArray
    {
        let inputsEmbeds = embeddings(for: inputIds)
        let imageTokenMask = inputIds .== MLXArray(Int32(imageTokenId))
        return try mergeEmbeddings(inputsEmbeds: inputsEmbeds, imageFeatures: imageFeatures, mask: imageTokenMask)
    }

    private func mergeEmbeddings(inputsEmbeds: MLXArray, imageFeatures: MLXArray, mask: MLXArray) throws -> MLXArray {
        guard inputsEmbeds.ndim == 3 else {
            throw DeepseekOCR2Error.shapeMismatch("inputsEmbeds must be rank-3 [batch, seq, hidden]")
        }
        guard imageFeatures.ndim == 3 else {
            throw DeepseekOCR2Error.shapeMismatch("imageFeatures must be rank-3 [batch, tokens, hidden]")
        }
        guard mask.ndim == 2 else {
            throw DeepseekOCR2Error.shapeMismatch("mask must be rank-2 [batch, seq]")
        }

        let batch = inputsEmbeds.dim(0)
        let seqLen = inputsEmbeds.dim(1)
        let numImageTokens = imageFeatures.dim(1)
        let hidden = inputsEmbeds.dim(2)

        guard imageFeatures.dim(0) == batch, imageFeatures.dim(2) == hidden else {
            throw DeepseekOCR2Error.shapeMismatch(
                "imageFeatures must be [\(batch), \(numImageTokens), \(hidden)] (got [\(imageFeatures.dim(0)), \(numImageTokens), \(imageFeatures.dim(2))])"
            )
        }
        guard mask.dim(0) == batch, mask.dim(1) == seqLen else {
            throw DeepseekOCR2Error.shapeMismatch(
                "mask must be [\(batch), \(seqLen)] (got [\(mask.dim(0)), \(mask.dim(1))])"
            )
        }

        if numImageTokens == 0 {
            return inputsEmbeds
        }

        // `mask` contains a contiguous run of `true` values (the image token placeholder).
        // Avoid `argMax(mask)` because tie-breaking across multiple 1s is not guaranteed on Metal/MPS,
        // which can cause batch elements to inject image features at different offsets.
        let positions = broadcast(
            MLXArray(Int32(0) ..< Int32(seqLen)).reshaped(1, seqLen),
            to: [batch, seqLen]
        )
        let sentinel = MLXArray(Int32(seqLen))
        let maskedPositions = which(mask, positions, sentinel)
        let firstTrueIdxs = maskedPositions.min(axis: 1).asType(.int32)  // [B]

        // Scatter image features into a `[B, seqLen, hidden]` tensor at per-sequence offsets.
        let imageOffsets = MLXArray(Int32(0) ..< Int32(numImageTokens)).reshaped(1, numImageTokens)  // [1, T]
        let tokenPositions = firstTrueIdxs.reshaped(batch, 1) + imageOffsets  // [B, T]
        let tokenPositionsBroad = broadcast(
            tokenPositions.reshaped(batch, numImageTokens, 1),
            to: [batch, numImageTokens, hidden]
        )

        let aligned = putAlong(
            MLXArray.zeros([batch, seqLen, hidden], dtype: imageFeatures.dtype),
            tokenPositionsBroad,
            values: imageFeatures,
            axis: 1
        )

        let expandedMask = mask.expandedDimensions(axis: -1)
        return which(expandedMask, aligned.asType(inputsEmbeds.dtype), inputsEmbeds)
    }
}

public final class DeepseekOCR2ForCausalLM: Module {
    @ModuleInfo(key: "model") public var model: DeepseekOCR2Model
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    public let imageTokenId: Int = 128815

    public init(config: DeepseekV2Configuration) {
        self._model.wrappedValue = DeepseekOCR2Model(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
        super.init()
    }

    public func forwardPrompt(
        inputIds: MLXArray,
        globalView: MLXArray,
        crops: MLXArray?,
        cropsPerImage: [Int]? = nil,
        cache: [KVCache]? = nil
    ) throws -> MLXArray {
        let imageFeatures: MLXArray
        if let cropsPerImage {
            imageFeatures = try model.encodeImage(globalView: globalView, crops: crops, cropsPerImage: cropsPerImage)
        } else {
            imageFeatures = try model.encodeImage(globalView: globalView, crops: crops)
        }
        let inputsEmbeds = try model.mergeInputIdsWithImageFeatures(
            inputIds: inputIds,
            imageFeatures: imageFeatures,
            imageTokenId: imageTokenId
        )
        let hiddenStates = model.forwardLanguage(inputsEmbeds: inputsEmbeds, cache: cache)
        return lmHead(hiddenStates)
    }

    public func forwardGeneration(inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let embeds = model.embeddings(for: inputIds)
        let hiddenStates = model.forwardLanguage(inputsEmbeds: embeds, cache: cache)
        return lmHead(hiddenStates)
    }

    public func newCache() -> [KVCache] {
        (0..<model.config.numHiddenLayers).map { _ in KVCacheRaggedSimple() }
    }
}

extension DeepseekOCR2ForCausalLM {
    public static func hasQuantization(at directory: URL) -> Bool {
        let manifestURL = directory.appendingPathComponent("quantization.json")
        return FileManager.default.fileExists(atPath: manifestURL.path)
    }

    public static func load(
        from directory: URL,
        dtype: DType? = nil,
        keepMoEGateWeightsFloat32: Bool = true
    ) throws -> DeepseekOCR2ForCausalLM {
        let configURL = directory.appendingPathComponent("config.json")
        let config = try DeepseekV2Configuration.load(from: configURL)
        try validate(config)
        let model = DeepseekOCR2ForCausalLM(config: config)

        guard let enumerator = FileManager.default.enumerator(
            at: directory,
            includingPropertiesForKeys: nil
        ) else {
            throw DeepseekOCR2Error.modelLoadFailed("Failed to enumerate model directory: \(directory.path)")
        }

        var safetensorFiles: [URL] = []
        for case let url as URL in enumerator {
            guard url.pathExtension == "safetensors" else { continue }
            if isGitLFSPointer(url) {
                throw DeepseekOCR2Error.modelLoadFailed(
                    """
                    Found a Git LFS pointer instead of real weights: \(url.lastPathComponent).
                    Download weights via Hugging Face snapshot (recommended) or run `git lfs pull`.
                    """
                )
            }
            safetensorFiles.append(url)
        }

        guard !safetensorFiles.isEmpty else {
            throw DeepseekOCR2Error.modelLoadFailed("No .safetensors files found under: \(directory.path)")
        }

        var weights: [String: MLXArray] = [:]
        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        if FileManager.default.fileExists(atPath: indexURL.path) {
            struct SafetensorsIndex: Decodable {
                var weightMap: [String: String]

                enum CodingKeys: String, CodingKey {
                    case weightMap = "weight_map"
                }
            }

            if let data = try? Data(contentsOf: indexURL),
               let index = try? JSONDecoder().decode(SafetensorsIndex.self, from: data)
            {
                weights.reserveCapacity(index.weightMap.count)
            }
        }
        for file in safetensorFiles.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let arrays = try MLX.loadArrays(url: file)
            for (key, value) in arrays {
                weights[key] = value
            }
        }

        let manifestURL = directory.appendingPathComponent("quantization.json")
        let quantManifest: DeepSeekQuantizationManifest?
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            do {
                quantManifest = try DeepSeekQuantizationManifest.load(from: manifestURL)
            } catch {
                throw DeepseekOCR2Error.modelLoadFailed(
                    "Failed to load quantization manifest (\(manifestURL.lastPathComponent)): \(error)"
                )
            }
        } else {
            quantManifest = nil
        }

        var sanitized = sanitize(weights, config: config)
        if let quantManifest {
            do {
                try DeepSeekQuantizationApplier.apply(to: model, manifest: quantManifest, weights: sanitized)
            } catch {
                throw DeepseekOCR2Error.modelLoadFailed("Failed to apply quantization: \(error)")
            }
        }
        if let dtype {
            sanitized = castFloatingWeights(
                sanitized,
                to: dtype,
                keepMoEGateWeightsFloat32: keepMoEGateWeightsFloat32
            )
        }
        let parameters = ModuleParameters.unflattened(sanitized)
        try model.update(parameters: parameters, verify: [.all])
        try checkedEval(model)
        return model
    }

    private static func validate(_ config: DeepseekV2Configuration) throws {
        var issues: [String] = []

        if config.useMLA {
            issues.append("use_mla=true is not supported (expected use_mla=false).")
        }
        if config.hiddenSize % config.numAttentionHeads != 0 {
            issues.append(
                "hidden_size (\(config.hiddenSize)) must be divisible by num_attention_heads (\(config.numAttentionHeads))."
            )
        }

        let supportedScoringFuncs: Set<String> = ["softmax", "sigmoid"]
        if !supportedScoringFuncs.contains(config.scoringFunc) {
            issues.append(
                "Unsupported scoring_func='\(config.scoringFunc)'. Supported: \(supportedScoringFuncs.sorted().joined(separator: ", "))."
            )
        }

        let supportedTopkMethods: Set<String> = ["greedy", "group_limited_greedy"]
        if !supportedTopkMethods.contains(config.topkMethod) {
            issues.append(
                "Unsupported topk_method='\(config.topkMethod)'. Supported: \(supportedTopkMethods.sorted().joined(separator: ", "))."
            )
        }

        if config.nRoutedExperts != nil, config.numExpertsPerTok == nil {
            issues.append("num_experts_per_tok is required when n_routed_experts is set.")
        }

        if config.topkMethod == "group_limited_greedy" {
            guard let nGroup = config.nGroup, let topkGroup = config.topkGroup else {
                issues.append("topk_method='group_limited_greedy' requires n_group and topk_group.")
                throw DeepseekOCR2Error.modelLoadFailed("Invalid config.json:\n- " + issues.joined(separator: "\n- "))
            }

            if let nExperts = config.nRoutedExperts, nExperts % nGroup != 0 {
                issues.append("n_routed_experts (\(nExperts)) must be divisible by n_group (\(nGroup)).")
            }
            if topkGroup > nGroup {
                issues.append("topk_group (\(topkGroup)) must be <= n_group (\(nGroup)).")
            }
        }

        guard issues.isEmpty else {
            throw DeepseekOCR2Error.modelLoadFailed("Invalid config.json:\n- " + issues.joined(separator: "\n- "))
        }
    }

    private static func sanitize(_ weights: [String: MLXArray], config: DeepseekV2Configuration) -> [String: MLXArray] {
        var out = weights

        for (key, value) in weights {
            guard key.hasPrefix("model.sam_model"),
                  key.hasSuffix(".weight"),
                  value.ndim == 4
            else { continue }

            out[key] = maybeTransposeTorchConvWeight(value)
        }

        out = DeepseekV2Weights.sanitizeMoEExpertWeights(out, config: config)
        return out
    }

    private static func castFloatingWeights(
        _ weights: [String: MLXArray],
        to dtype: DType,
        keepMoEGateWeightsFloat32: Bool
    ) -> [String: MLXArray] {
        let keepGateFloat32 = keepMoEGateWeightsFloat32 && dtype != .float32
        var out: [String: MLXArray] = [:]
        out.reserveCapacity(weights.count)

        for (key, value) in weights {
            guard value.dtype.isFloatingPoint else {
                out[key] = value
                continue
            }

            if keepGateFloat32, key.hasSuffix(".mlp.gate.weight") {
                out[key] = value.asType(.float32)
            } else if key.hasSuffix(".scales") || key.hasSuffix(".biases") {
                out[key] = value.asType(.float32)
            } else {
                out[key] = value.asType(dtype)
            }
        }

        return out
    }

    private static func isGitLFSPointer(_ url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }

        let data = handle.readData(ofLength: 256)
        guard let header = String(data: data, encoding: .utf8) else { return false }
        return header.contains("git-lfs.github.com/spec/v1")
    }

    private static func maybeTransposeTorchConvWeight(_ weight: MLXArray) -> MLXArray {
        guard weight.ndim == 4 else { return weight }

        let d1 = weight.dim(1)
        let d3 = weight.dim(3)

        // MLX Conv2d uses NHWC inputs and stores weights as [out, kH, kW, in].
        // HF PyTorch weights are [out, in, kH, kW].
        if d1 > 16, d3 <= 16 {
            return weight.transposed(0, 2, 3, 1)
        }
        if d1 <= 16, d3 > 16 {
            return weight
        }

        // Ambiguous small shapes (e.g. patch_embed: [out, 3, 16, 16] vs [out, 16, 16, 3]).
        if d1 == 3 {
            return weight.transposed(0, 2, 3, 1)
        }

        return weight
    }
}
