import Foundation
import MLX
import MLXNN

final class Qwen2EncoderAttention: Module {
    private let heads: Int
    private let kvHeads: Int
    private let headDim: Int
    private let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    private let rope: RoPE

    init(hiddenSize: Int, numHeads: Int, numKeyValueHeads: Int, ropeTheta: Float = 1_000_000.0) {
        precondition(hiddenSize % numHeads == 0, "hiddenSize must be divisible by numHeads")
        self.heads = numHeads
        self.kvHeads = numKeyValueHeads
        self.headDim = hiddenSize / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))

        self._wq.wrappedValue = Linear(hiddenSize, heads * headDim, bias: true)
        self._wk.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: true)
        self._wv.wrappedValue = Linear(hiddenSize, kvHeads * headDim, bias: true)
        self._wo.wrappedValue = Linear(heads * headDim, hiddenSize, bias: false)

        self.rope = RoPE(dimensions: headDim, traditional: false, base: ropeTheta, scale: 1.0)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let (batch, seqLen) = (x.dim(0), x.dim(1))

        var q = wq(x).reshaped(batch, seqLen, heads, headDim).transposed(0, 2, 1, 3)
        var k = wk(x).reshaped(batch, seqLen, kvHeads, headDim).transposed(0, 2, 1, 3)
        let v = wv(x).reshaped(batch, seqLen, kvHeads, headDim).transposed(0, 2, 1, 3)

        q = rope(q)
        k = rope(k)

        let attn = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: mask
        )

        let merged = attn.transposed(0, 2, 1, 3).reshaped(batch, seqLen, heads * headDim)
        return wo(merged)
    }
}

final class Qwen2EncoderMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        self._gate.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._up.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        self._down.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

final class Qwen2EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen2EncoderAttention
    @ModuleInfo(key: "mlp") var mlp: Qwen2EncoderMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(hiddenSize: Int, intermediateSize: Int, numHeads: Int, numKeyValueHeads: Int) {
        self._selfAttn.wrappedValue = Qwen2EncoderAttention(
            hiddenSize: hiddenSize,
            numHeads: numHeads,
            numKeyValueHeads: numKeyValueHeads
        )
        self._mlp.wrappedValue = Qwen2EncoderMLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize)
        self._inputLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: 1e-6)
        self._postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

final class Qwen2EncoderModelInner: Module {
    var layers: [Qwen2EncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(hiddenSize: Int, intermediateSize: Int, numLayers: Int, numHeads: Int, numKeyValueHeads: Int) {
        self.layers = (0..<numLayers).map { _ in
            Qwen2EncoderLayer(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                numHeads: numHeads,
                numKeyValueHeads: numKeyValueHeads
            )
        }
        self._norm.wrappedValue = RMSNorm(dimensions: hiddenSize, eps: 1e-6)
        super.init()
    }

    func callAsFunction(_ inputsEmbeds: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        var h = inputsEmbeds
        for layer in layers {
            h = layer(h, mask: mask)
        }
        return norm(h)
    }
}

final class Qwen2EncoderDecoder: Module {
    @ModuleInfo(key: "model") var model: Qwen2EncoderModelInner

    init(hiddenSize: Int, intermediateSize: Int, numLayers: Int, numHeads: Int, numKeyValueHeads: Int) {
        self._model.wrappedValue = Qwen2EncoderModelInner(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numLayers: numLayers,
            numHeads: numHeads,
            numKeyValueHeads: numKeyValueHeads
        )
        super.init()
    }

    func callAsFunction(_ inputsEmbeds: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        model(inputsEmbeds, mask: mask)
    }
}

final class Qwen2Decoder2Encoder: Module, UnaryLayer {
    @ModuleInfo(key: "model") var model: Qwen2EncoderDecoder
    @ModuleInfo(key: "query_768") var query768: Embedding
    @ModuleInfo(key: "query_1024") var query1024: Embedding
    nonisolated(unsafe) private static let causalFlowMask144 = makeCausalFlowMask(nQuery: 144)
    nonisolated(unsafe) private static let causalFlowMask256 = makeCausalFlowMask(nQuery: 256)

    override init() {
        // Matches `build_qwen2_decoder_as_encoder` in deepencoderv2.py.
        let hiddenSize = 896
        let intermediateSize = 4864
        let numLayers = 24
        let numHeads = 14
        let numKeyValueHeads = 2

        self._model.wrappedValue = Qwen2EncoderDecoder(
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            numLayers: numLayers,
            numHeads: numHeads,
            numKeyValueHeads: numKeyValueHeads
        )
        self._query768.wrappedValue = Embedding(embeddingCount: 144, dimensions: hiddenSize)
        self._query1024.wrappedValue = Embedding(embeddingCount: 256, dimensions: hiddenSize)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // x: [B, H, W, C=896]
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        precondition(channels == 896, "Qwen2Decoder2Encoder expects channel dimension 896")

        let nQuery = height * width
        let xTokens = x.reshaped(batch, nQuery, channels)

        let queryWeights: MLXArray
        let mask: MLXArray
        switch nQuery {
        case 144:
            queryWeights = query768.weight
            mask = Self.causalFlowMask144
        case 256:
            queryWeights = query1024.weight
            mask = Self.causalFlowMask256
        default:
            fatalError("Unsupported nQuery=\(nQuery). Expected 144 (768->12x12) or 256 (1024->16x16).")
        }

        let query = broadcast(queryWeights.expandedDimensions(axis: 0), to: [batch, nQuery, channels])

        let xCombined = concatenated([xTokens, query], axis: 1)

        let encoded = model(xCombined, mask: .array(mask))
        return encoded[0..., nQuery..., 0...]
    }

    private static func makeCausalFlowMask(nQuery: Int) -> MLXArray {
        let total = 2 * nQuery
        let positions = MLXArray(Int32(0) ..< Int32(total))
        let linds = positions.reshaped([total, 1])
        let rinds = positions.reshaped([1, total])

        let imageRows = linds .< MLXArray(Int32(nQuery))
        let queryRows = imageRows .== MLXArray(false)

        let imageCols = rinds .< MLXArray(Int32(nQuery))
        let queryCols = imageCols .== MLXArray(false)

        let imageImage = imageRows & imageCols
        let queryImage = queryRows & imageCols
        let queryQuery = queryRows & queryCols & (linds .>= rinds)

        return imageImage | queryImage | queryQuery
    }
}
