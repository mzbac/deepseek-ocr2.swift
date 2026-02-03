import Foundation
import MLX
import MLXNN

public enum WeightsLoaderError: Error, Sendable {
    case unableToEnumerate(URL)
}

public enum WeightsLoader {
    public static func loadSafetensors(
        fromSafetensorsFile fileURL: URL,
        filterKeys: (String) -> Bool = { _ in true }
    ) throws -> [String: MLXArray] {
        let arrays = try MLX.loadArrays(url: fileURL)
        return arrays.filter { filterKeys($0.key) }
    }

    public static func loadSafetensors(
        from modelDirectory: URL,
        filterKeys: (String) -> Bool = { _ in true }
    ) throws -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        guard let enumerator = FileManager.default.enumerator(
            at: modelDirectory,
            includingPropertiesForKeys: nil
        ) else {
            throw WeightsLoaderError.unableToEnumerate(modelDirectory)
        }
        for case let url as URL in enumerator {
            guard url.pathExtension == "safetensors" else { continue }
            let arrays = try MLX.loadArrays(url: url)
            for (key, value) in arrays {
                guard filterKeys(key) else { continue }
                weights[key] = value
            }
        }
        return weights
    }

    public static func load(
        _ model: Module,
        from modelDirectory: URL,
        verify: Module.VerifyUpdate = .none,
        filterKeys: (String) -> Bool = { _ in true }
    ) throws {
        let weights = try loadSafetensors(from: modelDirectory, filterKeys: filterKeys)
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: verify)
        try checkedEval(model)
    }

    public static func load(
        _ model: Module,
        fromSafetensors fileURL: URL,
        verify: Module.VerifyUpdate = .none,
        filterKeys: (String) -> Bool = { _ in true }
    ) throws {
        let weights = try loadSafetensors(fromSafetensorsFile: fileURL, filterKeys: filterKeys)
        let parameters = ModuleParameters.unflattened(weights)
        try model.update(parameters: parameters, verify: verify)
        try checkedEval(model)
    }
}
