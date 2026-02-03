import ArgumentParser
import DeepSeekOCR2
import Foundation
import Hub
import MLX

@main
struct DeepSeekOCR2CLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "DeepSeekOCR2CLI",
        abstract: "DeepSeek OCR 2 (mlx-swift).",
        subcommands: [Run.self, Quantize.self],
        defaultSubcommand: Run.self
    )

    static func resolveHuggingFaceHubCacheDirectory(downloadBasePath: String?) -> URL {
        func normalize(_ value: String?) -> String? {
            let trimmed = value?.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let trimmed, !trimmed.isEmpty else { return nil }
            return trimmed
        }

        if let downloadBasePath = normalize(downloadBasePath) {
            return URL(fileURLWithPath: downloadBasePath).standardizedFileURL
        }

        let env = ProcessInfo.processInfo.environment
        if let hubCache = normalize(env["HF_HUB_CACHE"]) {
            return URL(fileURLWithPath: hubCache).standardizedFileURL
        }
        if let hfHome = normalize(env["HF_HOME"]) {
            return URL(fileURLWithPath: hfHome).appendingPathComponent("hub").standardizedFileURL
        }

        let home = FileManager.default.homeDirectoryForCurrentUser
        return home.appendingPathComponent(".cache/huggingface/hub").standardizedFileURL
    }

    static func resolveLocalModelDirectory(_ path: String) throws -> URL? {
        let expanded = (path as NSString).expandingTildeInPath
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: expanded, isDirectory: &isDirectory) else {
            return nil
        }
        guard isDirectory.boolValue else {
            throw ValidationError("Expected a directory at \(expanded).")
        }
        return URL(fileURLWithPath: expanded).standardizedFileURL
    }

    struct Run: AsyncParsableCommand {
        enum ModelDType: String, ExpressibleByArgument {
            case auto
            case float32
            case float16
            case bfloat16

            var mlxDType: DType? {
                switch self {
                case .auto:
                    return nil
                case .float32:
                    return .float32
                case .float16:
                    return .float16
                case .bfloat16:
                    return .bfloat16
                }
            }
        }

        struct JSONLRecord: Codable {
            let image: String
            let text: String?
            let tokenCount: Int?
            let durationS: Double?
            let error: String?
        }

        @Option(help: "Hugging Face model id to download/use, or a local model directory (skips download).")
        var model: String = "deepseek-ai/DeepSeek-OCR-2"

        @Option(help: "Hugging Face revision (branch/tag/commit) (remote models only).")
        var revision: String = "main"

        @Option(help: "Hub download base directory (defaults to Hugging Face hub cache).")
        var downloadBase: String?

        @Option(
            name: .customLong("cache-limit"),
            help: "GPU memory cache limit in MB (default: 2048; 0 disables caching)."
        )
        var cacheLimitMB: Int = 2048

        @Option(
            name: .customLong("image"),
            parsing: .upToNextOption,
            help: "Path(s) to image file(s). Repeat or pass multiple values."
        )
        var images: [String] = []

        @Option(help: "Prompt (use <image> for the image placeholder).")
        var prompt: String = "<image>\n<|grounding|>Convert the document to markdown."

        @Option(help: "Maximum number of new tokens to generate.")
        var maxNewTokens: Int = 4096

        @Option(help: "Maximum number of images per inference batch (0 = all).")
        var batchSize: Int = 0

        @Option(help: "Temperature (0 = greedy).")
        var temperature: Float = 0.0

        @Option(help: "Top-p (nucleus) sampling.")
        var topP: Float = 1.0

        @Option(help: "Cast floating-point model weights to this dtype (auto keeps checkpoint dtype).")
        var dtype: ModelDType = .auto

        @Flag(help: "Disable dynamic tiling/crop mode.")
        var noCropMode: Bool = false

        @Flag(help: "Emit JSON Lines (one JSON object per image) instead of raw text.")
        var jsonl: Bool = false

        mutating func run() async throws {
            func eprintln(_ message: String) {
                FileHandle.standardError.write(Data((message + "\n").utf8))
            }

            guard cacheLimitMB >= 0 else {
                throw ValidationError("--cache-limit must be >= 0.")
            }
            let (bytes, overflow) = cacheLimitMB.multipliedReportingOverflow(by: 1024 * 1024)
            guard !overflow else {
                throw ValidationError("--cache-limit is too large.")
            }
            Memory.cacheLimit = bytes
            if cacheLimitMB != 2048 {
                eprintln("GPU cache limit: \(cacheLimitMB)MB")
            }

            let imageProcessor = try DeepseekOCR2ImageProcessor(cropMode: !noCropMode)

            var imagePaths = images
            if imagePaths.isEmpty {
                throw ValidationError("Provide at least one --image.")
            }

            let modelURL: URL
            if let localURL = try DeepSeekOCR2CLI.resolveLocalModelDirectory(model) {
                modelURL = localURL
            } else {
                let modelId = model
                let modelRevision = revision
                let downloadBaseURL = DeepSeekOCR2CLI.resolveHuggingFaceHubCacheDirectory(downloadBasePath: downloadBase)
                try? FileManager.default.createDirectory(at: downloadBaseURL, withIntermediateDirectories: true)

                let hub = HubApi(downloadBase: downloadBaseURL, useOfflineMode: false)
                var lastCompleted: Int64 = -1
                let globs = [
                    "*.safetensors",
                    "*.json",
                    "tokenizer.*",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ]
                modelURL = try await hub.snapshot(from: modelId, revision: modelRevision, matching: globs) { progress in
                    if progress.completedUnitCount != lastCompleted {
                        lastCompleted = progress.completedUnitCount
                        let total = max(progress.totalUnitCount, 1)
                        eprintln("Downloading \(modelId) (\(lastCompleted)/\(total) files)...")
                    }
                }
            }

            let loadStart = CFAbsoluteTimeGetCurrent()
            let targetDType = dtype.mlxDType
            if let targetDType {
                eprintln("Loading model (casting weights to \(targetDType))...")
            }
            let pipeline = try await DeepseekOCR2Pipeline(
                modelURL: modelURL,
                imageProcessor: imageProcessor,
                dtype: targetDType,
                keepMoEGateWeightsFloat32: true
            )
            let loadEnd = CFAbsoluteTimeGetCurrent()
            eprintln(String(format: "Model loaded in %.3fs", loadEnd - loadStart))

            let encoder = JSONEncoder()
            encoder.outputFormatting = [.withoutEscapingSlashes]

            var hadError = false
            struct BatchItem {
                let imagePath: String
                let processed: DeepseekOCR2ProcessedImages
                let preprocessDurationS: Double
            }

            func runBatch(_ chunk: [BatchItem]) throws {
                let genStart = CFAbsoluteTimeGetCurrent()
                do {
                    let batchProcessed = chunk.map { $0.processed }
                    let results = try pipeline.generator.generateBatch(
                        processedImages: batchProcessed,
                        prompt: prompt,
                        maxNewTokens: maxNewTokens,
                        temperature: temperature,
                        topP: topP
                    )
                    let genEnd = CFAbsoluteTimeGetCurrent()
                    let perImageGen = (genEnd - genStart) / Double(max(chunk.count, 1))

                    for (item, result) in zip(chunk, results) {
                        let duration = item.preprocessDurationS + perImageGen
                        if jsonl {
                            let record = JSONLRecord(
                                image: item.imagePath,
                                text: result.text,
                                tokenCount: result.tokenCount,
                                durationS: duration,
                                error: nil
                            )
                            let line = try String(data: encoder.encode(record), encoding: .utf8) ?? ""
                            print(line)
                        } else {
                            if imagePaths.count > 1 {
                                print("=== \(item.imagePath) ===")
                            }
                            print(result.text)
                        }
                    }
                } catch {
                    hadError = true
                    eprintln("Batch generation failed (batch=\(chunk.count)): \(error)")
                    let genEnd = CFAbsoluteTimeGetCurrent()
                    let perImageGen = (genEnd - genStart) / Double(max(chunk.count, 1))

                    for item in chunk {
                        let duration = item.preprocessDurationS + perImageGen
                        if jsonl {
                            let record = JSONLRecord(
                                image: item.imagePath,
                                text: nil,
                                tokenCount: nil,
                                durationS: duration,
                                error: String(describing: error)
                            )
                            let line = try String(data: encoder.encode(record), encoding: .utf8) ?? ""
                            print(line)
                        } else {
                            eprintln("Error processing \(item.imagePath): \(error)")
                        }
                    }
                }
            }

            let effectiveBatchSize = batchSize > 0 ? batchSize : imagePaths.count
            var chunk: [BatchItem] = []
            chunk.reserveCapacity(max(1, min(effectiveBatchSize, 32)))

            for imagePath in imagePaths {
                let preprocessStart = CFAbsoluteTimeGetCurrent()
                do {
                    let processed = try imageProcessor.process(imageAt: imagePath)
                    let preprocessEnd = CFAbsoluteTimeGetCurrent()
                    chunk.append(
                        BatchItem(
                            imagePath: imagePath,
                            processed: processed,
                            preprocessDurationS: preprocessEnd - preprocessStart
                        )
                    )
                    if chunk.count >= effectiveBatchSize {
                        try autoreleasepool {
                            try runBatch(chunk)
                        }
                        chunk.removeAll(keepingCapacity: true)
                    }
                } catch {
                    hadError = true
                    let preprocessEnd = CFAbsoluteTimeGetCurrent()
                    if jsonl {
                        let record = JSONLRecord(
                            image: imagePath,
                            text: nil,
                            tokenCount: nil,
                            durationS: preprocessEnd - preprocessStart,
                            error: String(describing: error)
                        )
                        let line = try String(data: encoder.encode(record), encoding: .utf8) ?? ""
                        print(line)
                    } else {
                        eprintln("Error preprocessing \(imagePath): \(error)")
                    }
                }
            }

            if !chunk.isEmpty {
                try autoreleasepool {
                    try runBatch(chunk)
                }
            }

            if hadError {
                throw ExitCode.failure
            }
        }
    }

    struct Quantize: AsyncParsableCommand {
        enum Mode: String, ExpressibleByArgument {
            case affine
            case mxfp4

            var specMode: DeepSeekQuantizationMode {
                switch self {
                case .affine:
                    return .affine
                case .mxfp4:
                    return .mxfp4
                }
            }
        }

        @Option(help: "Hugging Face model id to download/use, or a local model directory (skips download).")
        var model: String = "deepseek-ai/DeepSeek-OCR-2"

        @Option(help: "Hugging Face revision (branch/tag/commit) (remote models only).")
        var revision: String = "main"

        @Option(help: "Hub download base directory (defaults to Hugging Face hub cache).")
        var downloadBase: String?

        @Option(
            name: .customLong("cache-limit"),
            help: "GPU memory cache limit in MB (default: 2048; 0 disables caching)."
        )
        var cacheLimitMB: Int = 2048

        @Option(help: "Output directory for quantized weights.")
        var outputDir: String

        @Option(help: "Quantization bits per weight (4 or 8).")
        var bits: Int = 4

        @Option(help: "Quantization group size (32, 64, 128).")
        var groupSize: Int = 64

        @Option(help: "Quantization mode (affine or mxfp4).")
        var mode: Mode = .affine

        @Flag(help: "Print progress.")
        var verbose: Bool = false

        mutating func run() async throws {
            func eprintln(_ message: String) {
                FileHandle.standardError.write(Data((message + "\n").utf8))
            }

            guard cacheLimitMB >= 0 else {
                throw ValidationError("--cache-limit must be >= 0.")
            }
            let (bytes, overflow) = cacheLimitMB.multipliedReportingOverflow(by: 1024 * 1024)
            guard !overflow else {
                throw ValidationError("--cache-limit is too large.")
            }
            Memory.cacheLimit = bytes
            if cacheLimitMB != 2048 {
                eprintln("GPU cache limit: \(cacheLimitMB)MB")
            }

            let sourceURL: URL
            let resolvedModelId: String?
            let resolvedRevision: String?
            if let localURL = try DeepSeekOCR2CLI.resolveLocalModelDirectory(model) {
                sourceURL = localURL
                resolvedModelId = nil
                resolvedRevision = nil
            } else {
                let modelId = model
                let modelRevision = revision
                let downloadBaseURL = DeepSeekOCR2CLI.resolveHuggingFaceHubCacheDirectory(downloadBasePath: downloadBase)
                try? FileManager.default.createDirectory(at: downloadBaseURL, withIntermediateDirectories: true)

                let hub = HubApi(downloadBase: downloadBaseURL, useOfflineMode: false)
                var lastCompleted: Int64 = -1
                let globs = [
                    "*.safetensors",
                    "*.json",
                    "tokenizer.*",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                ]
                sourceURL = try await hub.snapshot(from: modelId, revision: modelRevision, matching: globs) { progress in
                    if progress.completedUnitCount != lastCompleted {
                        lastCompleted = progress.completedUnitCount
                        let total = max(progress.totalUnitCount, 1)
                        eprintln("Downloading \(modelId) (\(lastCompleted)/\(total) files)...")
                    }
                }
                resolvedModelId = modelId
                resolvedRevision = modelRevision
            }

            let outputURL = URL(fileURLWithPath: outputDir)
            let spec = DeepSeekQuantizationSpec(groupSize: groupSize, bits: bits, mode: mode.specMode)
            try DeepSeekQuantizer.quantizeAndSave(
                from: sourceURL,
                to: outputURL,
                spec: spec,
                modelId: resolvedModelId,
                revision: resolvedRevision,
                verbose: verbose
            )
        }
    }
}
