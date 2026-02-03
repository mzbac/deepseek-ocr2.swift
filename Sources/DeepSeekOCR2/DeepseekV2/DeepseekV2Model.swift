import Foundation
import MLX
import MLXNN

final class DeepseekV2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class DeepseekV2MoEGate: Module {
    private let topK: Int
    private let nExperts: Int
    private let routedScalingFactor: Float
    private let scoringFunc: String
    private let topkMethod: String
    private let nGroup: Int?
    private let topkGroup: Int?
    private let normTopkProb: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(config: DeepseekV2Configuration) {
        precondition(config.numExpertsPerTok != nil, "num_experts_per_tok is required for MoE")
        precondition(config.nRoutedExperts != nil, "n_routed_experts is required for MoE")

        self.topK = config.numExpertsPerTok!
        self.nExperts = config.nRoutedExperts!
        self.routedScalingFactor = config.routedScalingFactor
        self.scoringFunc = config.scoringFunc
        self.topkMethod = config.topkMethod
        self.nGroup = config.nGroup
        self.topkGroup = config.topkGroup
        self.normTopkProb = config.normTopkProb

        self._weight.wrappedValue = MLXRandom.uniform(
            low: -0.01, high: 0.01, [nExperts, config.hiddenSize], dtype: .float32
        )
        super.init()
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> (MLXArray, MLXArray) {
        let hiddenSize = hiddenStates.dim(-1)
        let flat = hiddenStates.reshaped([-1, hiddenSize]).asType(.float32)
        let logits = matmul(flat, weight.asType(.float32).transposed(1, 0))

        let scores: MLXArray
        switch scoringFunc {
        case "softmax":
            scores = softmax(logits, axis: -1, precise: true)
        case "sigmoid":
            scores = sigmoid(logits)
        default:
            fatalError("Unsupported scoring_func: \(scoringFunc)")
        }

        let topkIdx: MLXArray
        switch topkMethod {
        case "greedy":
            // argPartition returns indices for the kth *smallest* values; use -scores for top-k.
            let part = argPartition(-scores, kth: topK - 1, axis: -1)
            topkIdx = part[.ellipsis, 0 ..< topK]
        case "group_limited_greedy":
            guard let nGroup, let topkGroup else {
                fatalError("group_limited_greedy requires n_group and topk_group")
            }

            let grouped = scores.reshaped([-1, nGroup, nExperts / nGroup])
            let groupScores = grouped.max(axis: -1)

            let groupPart = argPartition(-groupScores, kth: topkGroup - 1, axis: -1)
            let groupIdx = groupPart[.ellipsis, 0 ..< topkGroup]

            // Build a [nTokens, nGroup] mask with 1s in selected groups.
            let nTokens = groupScores.dim(0)
            let rowIdx = broadcast(
                MLXArray(0 ..< nTokens).reshaped([nTokens, 1]),
                to: [nTokens, topkGroup]
            )
            let ones = MLXArray.ones([nTokens, topkGroup], dtype: .float32)
            let groupMask = MLXArray.zeros([nTokens, nGroup], dtype: .float32).at[rowIdx, groupIdx].add(ones)

            // Expand to [nTokens, nExperts] (same layout as python)
            let scoreMask = broadcast(groupMask.reshaped([nTokens, nGroup, 1]), to: [nTokens, nGroup, nExperts / nGroup])
                .reshaped([nTokens, nExperts])

            let zero: Float = 0
            let masked = which(scoreMask .> zero, scores, zeros(like: scores))
            let part = argPartition(-masked, kth: topK - 1, axis: -1)
            topkIdx = part[.ellipsis, 0 ..< topK]
        default:
            fatalError("Unsupported topk_method: \(topkMethod)")
        }

        var topkWeight = takeAlong(scores, topkIdx, axis: -1)

        if topK > 1 && normTopkProb {
            let eps: Float = 1e-20
            let denom = topkWeight.sum(axis: -1, keepDims: true) + eps
            topkWeight = (topkWeight / denom) * routedScalingFactor
        } else {
            topkWeight = topkWeight * routedScalingFactor
        }

        return (topkIdx, topkWeight)
    }
}

final class DeepseekV2MoE: Module, UnaryLayer {
    private let numExpertsPerTok: Int

    @ModuleInfo(key: "experts") var experts: SwitchGLU
    @ModuleInfo(key: "gate") var gate: DeepseekV2MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV2MLP?

    init(config: DeepseekV2Configuration) {
        precondition(config.nRoutedExperts != nil, "n_routed_experts is required for MoE")
        precondition(config.numExpertsPerTok != nil, "num_experts_per_tok is required for MoE")

        self.numExpertsPerTok = config.numExpertsPerTok!
        self._experts.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.nRoutedExperts!,
            bias: false
        )

        self._gate.wrappedValue = DeepseekV2MoEGate(config: config)

        if let nSharedExperts = config.nSharedExperts {
            let intermediateSize = config.moeIntermediateSize * nSharedExperts
            self._sharedExperts.wrappedValue = DeepseekV2MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: intermediateSize
            )
        } else {
            self._sharedExperts.wrappedValue = nil
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let identity = x
        let (topkIdx, topkWeight) = gate(x)
        let batch = x.dim(0)
        let seqLen = x.dim(1)

        let idx = topkIdx.reshaped([batch, seqLen, numExpertsPerTok])
        let weights = topkWeight.reshaped([batch, seqLen, numExpertsPerTok]).asType(.float32)

        let choices = experts(x, idx).asType(.float32)
        var y = (choices * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(x.dtype)

        if let sharedExperts {
            y = y + sharedExperts(identity)
        }
        return y
    }
}

final class DeepseekV2Attention: Module {
    private let heads: Int
    private let kvHeads: Int
    private let headDim: Int
    private let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    private let rope: RoPE

    init(config: DeepseekV2Configuration) {
        precondition(config.useMLA == false, "Only use_mla=false (MHA) is supported for now.")
        precondition(config.hiddenSize % config.numAttentionHeads == 0, "hidden_size must be divisible by num_attention_heads")

        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDim = config.hiddenSize / config.numAttentionHeads
        self.scale = pow(Float(headDim), -0.5)

        self._wq.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: config.attentionBias)
        self._wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        self._wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: config.attentionBias)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta, scale: 1.0)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (batch, seqLen) = (x.dim(0), x.dim(1))

        var q = wq(x).reshaped(batch, seqLen, heads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(batch, seqLen, kvHeads, headDim).transposed(0, 2, 1, 3)
        var v = wv(x).reshaped(batch, seqLen, kvHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        let raggedOffsets: MLXArray? = {
            guard seqLen == 1, let raggedCache = cache as? KVCacheRagged else { return nil }
            return raggedCache.offsets
        }()
        if let raggedOffsets, seqLen == 1 {
            q = rope(q, offset: raggedOffsets)
            k = rope(k, offset: raggedOffsets)
        } else if batch > 1, seqLen == 1 {
            // Workaround: On Metal, applying RoPE at a non-zero cache offset can produce
            // per-batch position skew for incremental decode. Using the array-offset
            // overload (per-sequence offsets) keeps batched decode deterministic.
            let offsetVector = broadcast(MLXArray(Int32(offset)), to: [batch])
            q = rope(q, offset: offsetVector)
            k = rope(k, offset: offsetVector)
        } else {
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)
        }

        let attn: MLXArray
        if let cache {
            (k, v) = cache.update(keys: k, values: v)
            let resolvedMask: MLXFast.ScaledDotProductAttentionMaskMode
            if seqLen == 1,
                let raggedCache = cache as? KVCacheRagged,
                let raggedOffsets = raggedCache.offsets
            {
                resolvedMask = .array(createRaggedDecodeMask(lengths: raggedOffsets, maxLength: k.dim(2)))
            } else {
                resolvedMask = mask
            }
            attn = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: resolvedMask
            )
        } else {
            attn = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: mask
            )
        }

        let output = attn.transposed(0, 2, 1, 3).reshaped(batch, seqLen, heads * headDim)
        return wo(output)
    }
}

final class DeepseekV2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: DeepseekV2Attention
    var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(config: DeepseekV2Configuration, layerIdx: Int) {
        self._selfAttn.wrappedValue = DeepseekV2Attention(config: config)

        if config.nRoutedExperts != nil,
            layerIdx >= config.firstKDenseReplace,
            layerIdx % config.moeLayerFreq == 0
        {
            self.mlp = DeepseekV2MoE(config: config)
        } else {
            self.mlp = DeepseekV2MLP(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public final class DeepseekV2ModelInner: Module {
    public let config: DeepseekV2Configuration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    var layers: [DeepseekV2DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    public init(config: DeepseekV2Configuration) {
        self.config = config
        self._embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { DeepseekV2DecoderLayer(config: config, layerIdx: $0) }
        self._norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func embeddings(for inputIds: MLXArray) -> MLXArray {
        embedTokens(inputIds)
    }

    public func forward(inputsEmbeds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
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

    public func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        forward(inputsEmbeds: embeddings(for: inputIds), cache: cache)
    }
}

public final class DeepseekV2ForCausalLM: Module {
    @ModuleInfo(key: "model") public var model: DeepseekV2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(config: DeepseekV2Configuration) {
        precondition(config.useMLA == false, "Only use_mla=false (MHA) is supported for now.")
        self._model.wrappedValue = DeepseekV2ModelInner(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    public func callAsFunction(_ inputIds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(model(inputIds, cache: cache))
    }

    public func forward(inputsEmbeds: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        lmHead(model.forward(inputsEmbeds: inputsEmbeds, cache: cache))
    }
}
