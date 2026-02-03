@testable import DeepSeekOCR2
import Foundation
import MLX
import MLXNN
import XCTest

final class QuantizationTests: XCTestCase {
    func testDeepSeekQuantizerWritesScales() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let sourceURL = root.appendingPathComponent("source")
        let outputURL = root.appendingPathComponent("output")
        defer { try? fm.removeItem(at: root) }

        try fm.createDirectory(at: sourceURL, withIntermediateDirectories: true)
        try fm.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let outDim = 16
        let inDim = 32
        let linWeight = MLXArray(Int32(0) ..< Int32(outDim * inDim))
            .asType(.float32)
            .reshaped(outDim, inDim)
        let embedWeight = MLXArray.ones([4, inDim], dtype: .float32)

        let safetensorsURL = sourceURL.appendingPathComponent("model.safetensors")
        try MLX.save(
            arrays: [
                "lin.weight": linWeight,
                "embed_tokens.weight": embedWeight,
            ],
            metadata: [:],
            url: safetensorsURL
        )

        let spec = DeepSeekQuantizationSpec(groupSize: 32, bits: 8, mode: .affine)
        try DeepSeekQuantizer.quantizeAndSave(from: sourceURL, to: outputURL, spec: spec)

        XCTAssertTrue(fm.fileExists(atPath: outputURL.appendingPathComponent("quantization.json").path))

        let outFiles = try fm.contentsOfDirectory(at: outputURL, includingPropertiesForKeys: nil)
        let shardFiles = outFiles.filter { $0.pathExtension == "safetensors" }
        XCTAssertFalse(shardFiles.isEmpty)

        var outWeights: [String: MLXArray] = [:]
        for file in shardFiles {
            let arrays = try MLX.loadArrays(url: file)
            for (k, v) in arrays {
                outWeights[k] = v
            }
        }

        XCTAssertNotNil(outWeights["lin.weight"])
        XCTAssertNotNil(outWeights["lin.scales"])

        if let qWeight = outWeights["lin.weight"] {
            XCTAssertFalse(qWeight.dtype.isFloatingPoint)
        }
        if let scales = outWeights["lin.scales"] {
            XCTAssertTrue(scales.dtype.isFloatingPoint)
        }
        if let biases = outWeights["lin.biases"] {
            XCTAssertTrue(biases.dtype.isFloatingPoint)
        }

        XCTAssertNotNil(outWeights["embed_tokens.weight"])
        XCTAssertNil(outWeights["embed_tokens.scales"])
    }

    func testDeepSeekQuantizerAllowlistPreventsQuantizingEmbeddingWeights() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let sourceURL = root.appendingPathComponent("source")
        let outputURL = root.appendingPathComponent("output")
        defer { try? fm.removeItem(at: root) }

        try fm.createDirectory(at: sourceURL, withIntermediateDirectories: true)
        try fm.createDirectory(at: outputURL, withIntermediateDirectories: true)

        let config = DeepseekV2Configuration(
            vocabSize: 32,
            hiddenSize: 32,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            useMLA: false
        )
        let configData = try JSONEncoder().encode(config)
        try configData.write(to: sourceURL.appendingPathComponent("config.json"))

        let queryWeight = MLXArray.zeros([256, 896], dtype: .float32)
        let lmHeadWeight = MLXArray.ones([16, 32], dtype: .float32)

        let safetensorsURL = sourceURL.appendingPathComponent("model.safetensors")
        try MLX.save(
            arrays: [
                "model.qwen2_model.query_1024.weight": queryWeight,
                "lm_head.weight": lmHeadWeight,
            ],
            metadata: [:],
            url: safetensorsURL
        )

        let spec = DeepSeekQuantizationSpec(groupSize: 32, bits: 8, mode: .affine)
        try DeepSeekQuantizer.quantizeAndSave(from: sourceURL, to: outputURL, spec: spec)

        let outFiles = try fm.contentsOfDirectory(at: outputURL, includingPropertiesForKeys: nil)
        let shardFiles = outFiles.filter { $0.pathExtension == "safetensors" }
        XCTAssertFalse(shardFiles.isEmpty)

        var outWeights: [String: MLXArray] = [:]
        for file in shardFiles {
            let arrays = try MLX.loadArrays(url: file)
            for (k, v) in arrays {
                outWeights[k] = v
            }
        }

        XCTAssertEqual(outWeights["model.qwen2_model.query_1024.weight"]?.shape, [256, 896])
        XCTAssertNil(outWeights["model.qwen2_model.query_1024.scales"])
        XCTAssertNil(outWeights["model.qwen2_model.query_1024.biases"])

        XCTAssertNotNil(outWeights["lm_head.weight"])
        XCTAssertNotNil(outWeights["lm_head.scales"])
        XCTAssertFalse(outWeights["lm_head.weight"]!.dtype.isFloatingPoint)
    }

    func testMoESanitizeStacksScalesAndBiases() {
        let config = DeepseekV2Configuration(
            vocabSize: 32,
            hiddenSize: 32,
            intermediateSize: 64,
            moeIntermediateSize: 64,
            numHiddenLayers: 1,
            numAttentionHeads: 4,
            nRoutedExperts: 2,
            numExpertsPerTok: 1,
            moeLayerFreq: 1,
            firstKDenseReplace: 0,
            useMLA: false
        )

        let prefix = "model.layers.0.mlp.experts"
        let outDim = 16
        let inDim = 32
        let scaleShape = [outDim, 1]

        var weights: [String: MLXArray] = [:]
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            for e in 0..<2 {
                weights["\(prefix).\(e).\(proj).weight"] = MLXArray.ones([outDim, inDim], dtype: .float32) * Float32(e + 1)
                weights["\(prefix).\(e).\(proj).scales"] = MLXArray.ones(scaleShape, dtype: .float32) * Float32(e + 1)
                weights["\(prefix).\(e).\(proj).biases"] = MLXArray.ones(scaleShape, dtype: .float32) * Float32(e + 1)
            }
        }

        let sanitized = DeepseekV2Weights.sanitizeMoEExpertWeights(weights, config: config)

        for proj in ["gate_proj", "up_proj", "down_proj"] {
            XCTAssertEqual(sanitized["\(prefix).\(proj).weight"]?.dim(0), 2)
            XCTAssertEqual(sanitized["\(prefix).\(proj).scales"]?.dim(0), 2)
            XCTAssertEqual(sanitized["\(prefix).\(proj).biases"]?.dim(0), 2)

            XCTAssertNil(sanitized["\(prefix).0.\(proj).weight"])
            XCTAssertNil(sanitized["\(prefix).0.\(proj).scales"])
            XCTAssertNil(sanitized["\(prefix).0.\(proj).biases"])
        }
    }

    func testQuantizationApplierSwapsLinearAndSwitchLinear() throws {
        final class TinyQuantModel: Module {
            @ModuleInfo(key: "lin") var lin: Linear
            @ModuleInfo(key: "switch") var switchLayer: Module

            init(inDim: Int, outDim: Int, experts: Int) {
                self._lin.wrappedValue = Linear(inDim, outDim, bias: false)
                self._switchLayer.wrappedValue = SwitchLinear(
                    inputDims: inDim,
                    outputDims: outDim,
                    numExperts: experts,
                    bias: false
                )
                super.init()
            }
        }

        let inDim = 32
        let outDim = 16
        let experts = 2
        let groupSize = 32
        let bits = 8

        let model = TinyQuantModel(inDim: inDim, outDim: outDim, experts: experts)

        let wLin = MLXArray(Int32(0) ..< Int32(outDim * inDim))
            .asType(.float32)
            .reshaped(outDim, inDim)
        let (wqLin, sLin, bLin) = MLX.quantized(wLin, groupSize: groupSize, bits: bits, mode: .affine)

        var wqExperts: [MLXArray] = []
        var sExperts: [MLXArray] = []
        var bExperts: [MLXArray] = []
        for e in 0..<experts {
            let base = Int32(e * outDim * inDim)
            let w = MLXArray(base ..< base + Int32(outDim * inDim))
                .asType(.float32)
                .reshaped(outDim, inDim)
            let (wq, s, b) = MLX.quantized(w, groupSize: groupSize, bits: bits, mode: .affine)
            wqExperts.append(wq)
            sExperts.append(s)
            if let b {
                bExperts.append(b)
            }
        }

        let wqSwitch = stacked(wqExperts, axis: 0)
        let sSwitch = stacked(sExperts, axis: 0)
        let bSwitch: MLXArray? = bExperts.count == experts ? stacked(bExperts, axis: 0) : nil

        var weights: [String: MLXArray] = [
            "lin.weight": wqLin,
            "lin.scales": sLin,
            "switch.weight": wqSwitch,
            "switch.scales": sSwitch,
        ]
        if let bLin {
            weights["lin.biases"] = bLin
        }
        if let bSwitch {
            weights["switch.biases"] = bSwitch
        }

        let manifest = DeepSeekQuantizationManifest(
            modelId: nil,
            revision: nil,
            groupSize: groupSize,
            bits: bits,
            mode: "affine",
            layers: []
        )

        try DeepSeekQuantizationApplier.apply(to: model, manifest: manifest, weights: weights)

        let leaf = Dictionary(uniqueKeysWithValues: model.leafModules().flattened())
        XCTAssertTrue(leaf["lin"] is QuantizedLinear)
        XCTAssertTrue(leaf["switch"] is QuantizedSwitchLinear)

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)

        let x = MLXArray.ones([1, inDim], dtype: .float32)
        let y = model.lin(x)
        XCTAssertEqual(y.shape, [1, outDim])

        guard let qs = model.switchLayer as? QuantizedSwitchLinear else {
            XCTFail("switch layer was not quantized")
            return
        }

        let xSwitch = MLX.expandedDimensions(MLXArray.ones([1, 1, inDim], dtype: .float32), axes: [-2, -3])
        let indices = MLXArray([Int32(0)]).reshaped(1, 1, 1)
        let out = qs(xSwitch, indices, sortedIndices: false)
        XCTAssertEqual(out.dim(-1), outDim)
    }
}
