import MLX
import MLXNN

public enum DeepseekOCR2Error: Error, Sendable {
    case shapeMismatch(String)
    case imageLoadFailed(String)
    case imageProcessingFailed(String)
    case modelLoadFailed(String)
    case tokenizerLoadFailed(String)
}

final class DeepseekOCR2InjectedForCausalLM: Module {
    @ModuleInfo(key: "model") public var model: DeepseekV2ModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(config: DeepseekV2Configuration) {
        precondition(config.useMLA == false, "Only use_mla=false (MHA) is supported for now.")
        self._model.wrappedValue = DeepseekV2ModelInner(config: config)
        self._lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabSize, bias: false)
    }

    func callAsFunction(_ inputIds: MLXArray) -> MLXArray {
        lmHead(model(inputIds))
    }

    func forward(inputIds: MLXArray, imageEmbeds: MLXArray, imageSeqMask: MLXArray) throws -> MLXArray {
        let tokenEmbeds = model.embeddings(for: inputIds)
        let mixed = try Self.inject(imageEmbeds: imageEmbeds, mask: imageSeqMask, into: tokenEmbeds)
        return lmHead(model.forward(inputsEmbeds: mixed))
    }

    private static func inject(imageEmbeds: MLXArray, mask: MLXArray, into tokenEmbeds: MLXArray) throws -> MLXArray {
        guard tokenEmbeds.ndim == 3 else {
            throw DeepseekOCR2Error.shapeMismatch("tokenEmbeds must be rank-3 [batch, seq, hidden]")
        }
        guard mask.ndim == 2 else {
            throw DeepseekOCR2Error.shapeMismatch("imageSeqMask must be rank-2 [batch, seq]")
        }
        guard imageEmbeds.ndim == 3 else {
            throw DeepseekOCR2Error.shapeMismatch("imageEmbeds must be rank-3 [batch, seq, hidden]")
        }

        let batch = tokenEmbeds.dim(0)
        let seqLen = tokenEmbeds.dim(1)
        let hidden = tokenEmbeds.dim(2)

        guard mask.dim(0) == batch, mask.dim(1) == seqLen else {
            throw DeepseekOCR2Error.shapeMismatch(
                "imageSeqMask shape must match [\(batch), \(seqLen)] (got [\(mask.dim(0)), \(mask.dim(1))])"
            )
        }
        guard imageEmbeds.dim(0) == batch, imageEmbeds.dim(1) == seqLen, imageEmbeds.dim(2) == hidden else {
            throw DeepseekOCR2Error.shapeMismatch(
                "imageEmbeds shape must match [\(batch), \(seqLen), \(hidden)] (got [\(imageEmbeds.dim(0)), \(imageEmbeds.dim(1)), \(imageEmbeds.dim(2))])"
            )
        }

        let maskBool: MLXArray = {
            if mask.dtype == .bool {
                return mask
            }
            return mask .> 0
        }()
        let mask3d = broadcast(maskBool.reshaped([batch, seqLen, 1]), to: [batch, seqLen, hidden])
        return which(mask3d, imageEmbeds.asType(tokenEmbeds.dtype), tokenEmbeds)
    }
}
