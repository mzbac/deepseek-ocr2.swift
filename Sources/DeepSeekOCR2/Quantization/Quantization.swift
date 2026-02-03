import Foundation
import MLX
import MLXNN

public enum DeepSeekQuantizationMode: String, Codable, Sendable {
    case affine
    case mxfp4

    public var mlxMode: QuantizationMode {
        switch self {
        case .affine:
            return .affine
        case .mxfp4:
            return .mxfp4
        }
    }
}

public struct DeepSeekQuantizationSpec: Codable, Sendable {
    public var groupSize: Int
    public var bits: Int
    public var mode: DeepSeekQuantizationMode

    public init(groupSize: Int = 64, bits: Int = 8, mode: DeepSeekQuantizationMode = .affine) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
    }

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
        case mode
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.groupSize = try container.decodeIfPresent(Int.self, forKey: .groupSize) ?? 64
        self.bits = try container.decodeIfPresent(Int.self, forKey: .bits) ?? 8
        let modeStr = try container.decodeIfPresent(String.self, forKey: .mode) ?? "affine"
        self.mode = DeepSeekQuantizationMode(rawValue: modeStr) ?? .affine
    }
}

public struct DeepSeekQuantizationManifest: Codable {
    public var modelId: String?
    public var revision: String?
    public var groupSize: Int
    public var bits: Int
    public var mode: String
    public var layers: [QuantizedLayerInfo]

    public struct QuantizedLayerInfo: Codable {
        public var name: String
        public var shape: [Int]
        public var inDim: Int
        public var outDim: Int
        public var file: String
        public var quantFile: String?
        public var groupSize: Int?
        public var bits: Int?
        public var mode: String?

        enum CodingKeys: String, CodingKey {
            case name
            case shape
            case inDim = "in_dim"
            case outDim = "out_dim"
            case file
            case quantFile = "quant_file"
            case groupSize = "group_size"
            case bits
            case mode
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelId = "model_id"
        case revision
        case groupSize = "group_size"
        case bits
        case mode
        case layers
    }

    public static func load(from url: URL) throws -> DeepSeekQuantizationManifest {
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(DeepSeekQuantizationManifest.self, from: data)
    }
}

public enum DeepSeekQuantizationError: Error, LocalizedError {
    case noSafetensorsFound(URL)
    case invalidGroupSize(Int)
    case invalidBits(Int)
    case quantizationFailed(String)
    case outputDirectoryCreationFailed(URL)

    public var errorDescription: String? {
        switch self {
        case .noSafetensorsFound(let url):
            return "No safetensors files found in \(url.path)"
        case .invalidGroupSize(let size):
            return "Invalid group size: \(size). Supported sizes: 32, 64, 128"
        case .invalidBits(let bits):
            return "Invalid bits: \(bits). Supported values: 4, 8"
        case .quantizationFailed(let reason):
            return "Quantization failed: \(reason)"
        case .outputDirectoryCreationFailed(let url):
            return "Failed to create output directory: \(url.path)"
        }
    }
}

public struct DeepSeekQuantizer {
    public static let supportedGroupSizes: Set<Int> = [32, 64, 128]
    public static let supportedBits: Set<Int> = [4, 8]

    public static func quantizeAndSave(
        from sourceURL: URL,
        to outputURL: URL,
        spec: DeepSeekQuantizationSpec,
        modelId: String? = nil,
        revision: String? = nil,
        verbose: Bool = false
    ) throws {
        guard supportedGroupSizes.contains(spec.groupSize) else {
            throw DeepSeekQuantizationError.invalidGroupSize(spec.groupSize)
        }
        guard supportedBits.contains(spec.bits) else {
            throw DeepSeekQuantizationError.invalidBits(spec.bits)
        }

        let resolvedRevision: String?
        if let revision {
            resolvedRevision = revision
        } else {
            resolvedRevision = resolveGitRevision(at: sourceURL)
        }

        let fm = FileManager.default

        do {
            try fm.createDirectory(at: outputURL, withIntermediateDirectories: true)
        } catch {
            throw DeepSeekQuantizationError.outputDirectoryCreationFailed(outputURL)
        }

        let contents = try fm.contentsOfDirectory(at: sourceURL, includingPropertiesForKeys: nil)
        let safetensorFiles = contents.filter { $0.pathExtension == "safetensors" }

        guard !safetensorFiles.isEmpty else {
            throw DeepSeekQuantizationError.noSafetensorsFound(sourceURL)
        }

        if verbose {
            print("Found \(safetensorFiles.count) safetensors file(s)")
            print("Quantization spec: bits=\(spec.bits), group_size=\(spec.groupSize), mode=\(spec.mode.rawValue)")
        }

        let maxShardBytes = 4_500_000_000

        func bytes(of array: MLXArray) -> Int {
            array.shape.reduce(1, *) * array.dtype.size
        }

        var shardTempURLs: [URL] = []
        var current: [String: MLXArray] = [:]
        current.reserveCapacity(4096)
        var currentBytes = 0

        var quantizedLayers: [DeepSeekQuantizationManifest.QuantizedLayerInfo] = []
        quantizedLayers.reserveCapacity(1024)
        var layerShardIndex: [String: Int] = [:]
        layerShardIndex.reserveCapacity(1024)

        var quantizedCount = 0
        var skippedCount = 0

        let allowlistedWeightKeys: Set<String>? = {
            let configURL = sourceURL.appendingPathComponent("config.json")
            guard FileManager.default.fileExists(atPath: configURL.path) else {
                if verbose {
                    print("Warning: config.json not found; falling back to name-based quantization filter.")
                }
                return nil
            }

            do {
                let config = try DeepseekV2Configuration.load(from: configURL)
                let model = DeepseekOCR2ForCausalLM(config: config)

                var allowlist: Set<String> = []
                allowlist.reserveCapacity(4096)

                let nExperts = config.nRoutedExperts
                let moeMarker = ".mlp.experts."

                for (path, module) in model.leafModules().flattened() {
                    if module is Linear {
                        allowlist.insert("\(path).weight")
                    } else if module is SwitchLinear, let nExperts, let range = path.range(of: moeMarker) {
                        let before = String(path[..<range.upperBound])
                        let after = String(path[range.upperBound...])
                        for e in 0..<nExperts {
                            allowlist.insert("\(before)\(e).\(after).weight")
                        }
                    }
                }

                if verbose {
                    print("Quantization allowlist: \(allowlist.count) weight tensors")
                }

                return allowlist
            } catch {
                if verbose {
                    print("Warning: failed to derive quantization allowlist from config.json: \(error)")
                    print("Falling back to name-based quantization filter.")
                }
                return nil
            }
        }()

        func flushShard() throws {
            guard !current.isEmpty else { return }
            let idx = shardTempURLs.count + 1
            let name = String(format: "model-tmp-%05d.safetensors", idx)
            let url = outputURL.appendingPathComponent(name)
            try? fm.removeItem(at: url)
            try MLX.save(arrays: current, metadata: [:], url: url)
            shardTempURLs.append(url)
            if verbose {
                print("Saved: \(name)")
            }
            current.removeAll(keepingCapacity: true)
            currentBytes = 0
        }

        for file in safetensorFiles.sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            if verbose {
                print("Loading: \(file.lastPathComponent)")
            }
            let weights = try MLX.loadArrays(url: file)
            for key in weights.keys.sorted() {
                let tensor = weights[key]!

                let isMoEGate = key.contains(".mlp.gate.")
                let isEmbedding = key.contains("embed") || key.contains("embedding")

                let shouldQuantize: Bool
                if let allowlistedWeightKeys {
                    shouldQuantize = allowlistedWeightKeys.contains(key) && !isMoEGate
                } else {
                    shouldQuantize = key.hasSuffix(".weight") && tensor.ndim == 2 && !isMoEGate && !isEmbedding
                }

                var entries: [(String, MLXArray)] = []
                entries.reserveCapacity(3)
                var quantizedBase: String?

                if shouldQuantize, key.hasSuffix(".weight"), tensor.ndim == 2 {
                    let outDim = tensor.dim(0)
                    let inDim = tensor.dim(1)

                    if inDim % spec.groupSize == 0 {
                        let base = String(key.dropLast(".weight".count))
                        quantizedBase = base

                        var f = tensor
                        if f.dtype != .float32 {
                            f = f.asType(.float32)
                        }

                        let (wq, scales, biases) = MLX.quantized(
                            f,
                            groupSize: spec.groupSize,
                            bits: spec.bits,
                            mode: spec.mode.mlxMode
                        )

                        entries.append((key, wq))
                        entries.append(("\(base).scales", scales))
                        if let biases {
                            entries.append(("\(base).biases", biases))
                        }

                        quantizedLayers.append(.init(
                            name: base,
                            shape: [outDim, inDim],
                            inDim: inDim,
                            outDim: outDim,
                            file: "",
                            quantFile: nil,
                            groupSize: spec.groupSize,
                            bits: spec.bits,
                            mode: spec.mode.rawValue
                        ))

                        quantizedCount += 1
                        if verbose, quantizedCount % 50 == 0 {
                            print("Quantized \(quantizedCount) layers...")
                        }
                    } else {
                        entries.append((key, tensor))
                        skippedCount += 1
                    }
                } else {
                    entries.append((key, tensor))
                }

                let entryBytes = entries.reduce(0) { $0 + bytes(of: $1.1) }
                if entryBytes > maxShardBytes {
                    throw DeepSeekQuantizationError.quantizationFailed(
                        "Tensor group '\(key)' too large to fit in a single shard."
                    )
                }

                if currentBytes > 0, currentBytes + entryBytes > maxShardBytes {
                    try flushShard()
                }

                let assignedShard = shardTempURLs.count + 1
                for (k, v) in entries {
                    current[k] = v
                    currentBytes += bytes(of: v)
                }

                if let quantizedBase {
                    layerShardIndex[quantizedBase] = assignedShard
                }
            }
        }

        try flushShard()

        if verbose {
            print("Quantized \(quantizedCount) linear layers")
            print("Skipped \(skippedCount) incompatible layers")
            print("Saving to \(shardTempURLs.count) shard(s)...")
        }

        let total = max(1, shardTempURLs.count)
        var shardNameByIndex: [Int: String] = [:]
        shardNameByIndex.reserveCapacity(total)

        for (i, tmpURL) in shardTempURLs.enumerated() {
            let idx = i + 1
            let finalName = String(format: "model-%05d-of-%05d.safetensors", idx, total)
            let finalURL = outputURL.appendingPathComponent(finalName)
            try? fm.removeItem(at: finalURL)
            try fm.moveItem(at: tmpURL, to: finalURL)
            shardNameByIndex[idx] = finalName
        }

        var updatedLayers = quantizedLayers
        for i in 0..<updatedLayers.count {
            if let shardIndex = layerShardIndex[updatedLayers[i].name],
               let fname = shardNameByIndex[shardIndex]
            {
                updatedLayers[i].file = fname
                updatedLayers[i].quantFile = fname
            }
        }

        let manifest = DeepSeekQuantizationManifest(
            modelId: modelId,
            revision: resolvedRevision,
            groupSize: spec.groupSize,
            bits: spec.bits,
            mode: spec.mode.rawValue,
            layers: updatedLayers
        )

        let manifestURL = outputURL.appendingPathComponent("quantization.json")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(manifest)
        try data.write(to: manifestURL)

        if verbose {
            print("Saved quantization manifest: quantization.json")
        }

        try copyAncillaryFiles(from: sourceURL, to: outputURL, verbose: verbose)

        if verbose {
            print("Quantization complete!")
            print("Output directory: \(outputURL.path)")
        }
    }

    private static func copyAncillaryFiles(from sourceURL: URL, to outputURL: URL, verbose: Bool) throws {
        let fm = FileManager.default
        let contents = try fm.contentsOfDirectory(at: sourceURL, includingPropertiesForKeys: nil)

        let copyExtensions: Set<String> = ["json", "txt", "md"]
        let skipFiles: Set<String> = ["quantization.json", "model.safetensors.index.json"]

        for file in contents {
            let name = file.lastPathComponent
            let ext = file.pathExtension.lowercased()

            if ext == "safetensors" {
                continue
            }

            if skipFiles.contains(name) {
                continue
            }

            if copyExtensions.contains(ext) {
                let destURL = outputURL.appendingPathComponent(name)
                try? fm.removeItem(at: destURL)
                try fm.copyItem(at: file, to: destURL)
                if verbose {
                    print("Copied: \(name)")
                }
            }
        }
    }

    private static func resolveGitRevision(at url: URL) -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["rev-parse", "HEAD"]
        process.currentDirectoryURL = url

        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice

        do {
            try process.run()
            process.waitUntilExit()

            if process.terminationStatus == 0 {
                let data = pipe.fileHandleForReading.readDataToEndOfFile()
                if let output = String(data: data, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines),
                    !output.isEmpty
                {
                    return output
                }
            }
        } catch {
        }

        return nil
    }
}

enum DeepSeekQuantizationApplier {
    static func apply(
        to model: Module,
        manifest: DeepSeekQuantizationManifest,
        weights: [String: MLXArray]
    ) throws {
        let defaultGroupSize = manifest.groupSize
        let defaultBits = manifest.bits
        let defaultMode = manifest.mode

        var layerInfoByName: [String: DeepSeekQuantizationManifest.QuantizedLayerInfo] = [:]
        layerInfoByName.reserveCapacity(manifest.layers.count)
        for layer in manifest.layers {
            layerInfoByName[layer.name] = layer
        }

        var updates: [(String, Module)] = []
        updates.reserveCapacity(1024)

        for (path, module) in model.leafModules().flattened() {
            guard weights["\(path).scales"] != nil else { continue }

            let layerInfo = layerInfoByName[path]
            let groupSize = layerInfo?.groupSize ?? defaultGroupSize
            let bits = layerInfo?.bits ?? defaultBits
            let modeStr = layerInfo?.mode ?? defaultMode
            let mode: QuantizationMode = modeStr == "mxfp4" ? .mxfp4 : .affine

            if module is SwitchLinear {
                guard let weight = weights["\(path).weight"],
                      let scales = weights["\(path).scales"]
                else { continue }

                let biases = weights["\(path).biases"]
                let bias = weights["\(path).bias"]
                let quantized = QuantizedSwitchLinear(
                    weight: weight,
                    scales: scales,
                    biases: biases,
                    bias: bias,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
                updates.append((path, quantized))
            } else if let _ = module as? Linear {
                guard let weight = weights["\(path).weight"],
                      let scales = weights["\(path).scales"]
                else { continue }

                let biases = weights["\(path).biases"]
                let bias = weights["\(path).bias"]
                let quantized = QuantizedLinear(
                    weight: weight,
                    bias: bias,
                    scales: scales,
                    biases: biases,
                    groupSize: groupSize,
                    bits: bits,
                    mode: mode
                )
                quantized.freeze()
                updates.append((path, quantized))
            }
        }

        guard !updates.isEmpty else { return }
        try model.update(modules: ModuleChildren.unflattened(updates), verify: .none)
    }
}
