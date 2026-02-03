import CoreImage
import Foundation
import MLX
import Tokenizers

public final class DeepseekOCR2Pipeline {
    public let model: DeepseekOCR2ForCausalLM
    public let tokenizer: any Tokenizer
    public let imageProcessor: DeepseekOCR2ImageProcessor
    public let generator: DeepseekOCR2Generator

    public convenience init(modelPath: String) async throws {
        try await self.init(modelURL: URL(fileURLWithPath: modelPath))
    }

    public init(
        modelURL: URL,
        imageProcessor: DeepseekOCR2ImageProcessor? = nil,
        dtype: DType? = nil,
        keepMoEGateWeightsFloat32: Bool = true
    ) async throws {
        self.model = try DeepseekOCR2ForCausalLM.load(
            from: modelURL,
            dtype: dtype,
            keepMoEGateWeightsFloat32: keepMoEGateWeightsFloat32
        )
        self.tokenizer = try await AutoTokenizer.from(modelFolder: modelURL)
        self.imageProcessor = try imageProcessor ?? DeepseekOCR2ImageProcessor()
        self.generator = DeepseekOCR2Generator(model: model, tokenizer: tokenizer)
    }

    public func recognize(
        imagePath: String,
        prompt: String = "<image>\n<|grounding|>Convert the document to markdown.",
        maxTokens: Int = 4096
    ) throws -> String {
        let processed = try imageProcessor.process(imageAt: imagePath)
        return try recognize(processedImages: processed, prompt: prompt, maxTokens: maxTokens)
    }

    public func recognize(
        image: CIImage,
        prompt: String = "<image>\n<|grounding|>Convert the document to markdown.",
        maxTokens: Int = 4096
    ) throws -> String {
        let processed = try imageProcessor.process(image)
        return try recognize(processedImages: processed, prompt: prompt, maxTokens: maxTokens)
    }

    public func recognize(
        processedImages: DeepseekOCR2ProcessedImages,
        prompt: String = "<image>\n<|grounding|>Convert the document to markdown.",
        maxTokens: Int = 4096
    ) throws -> String {
        let result = try generator.generate(
            processedImages: processedImages,
            prompt: prompt,
            maxNewTokens: maxTokens,
            temperature: 0.0,
            topP: 1.0
        )
        return result.text
    }
}
