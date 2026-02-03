import MLX
import MLXNN

// Adapted from mlx-swift-lm's SwitchLayers.swift (gatherMM-based expert routing).

func gatherSort(x: MLXArray, indices: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
    let m = indices.dim(-1)
    let flatIndices = indices.flattened()
    let order = argSort(flatIndices)
    let inverseOrder = argSort(order)

    let sortedX = x.flattened(start: 0, end: -3)[order.floorDivide(m)]
    let sortedIndices = flatIndices[order]
    return (sortedX, sortedIndices, inverseOrder)
}

func scatterUnsort(x: MLXArray, invOrder: MLXArray, shape: [Int]) -> MLXArray {
    let restored = x[invOrder]
    return unflatten(restored, axis: 0, shape: shape)
}

private protocol SwitchLinearProtocol {
    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool) -> MLXArray
}

extension SwitchLinear: SwitchLinearProtocol {}
extension QuantizedSwitchLinear: SwitchLinearProtocol {}

final class SwitchGLU: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Module
    @ModuleInfo(key: "up_proj") var upProj: Module
    @ModuleInfo(key: "down_proj") var downProj: Module

    private let activation: (MLXArray) -> MLXArray

    init(
        inputDims: Int,
        hiddenDims: Int,
        numExperts: Int,
        activation: @escaping (MLXArray) -> MLXArray = MLXNN.silu,
        bias: Bool = false
    ) {
        self._gateProj.wrappedValue = SwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims,
            numExperts: numExperts,
            bias: bias
        )
        self._upProj.wrappedValue = SwitchLinear(
            inputDims: inputDims,
            outputDims: hiddenDims,
            numExperts: numExperts,
            bias: bias
        )
        self._downProj.wrappedValue = SwitchLinear(
            inputDims: hiddenDims,
            outputDims: inputDims,
            numExperts: numExperts,
            bias: bias
        )
        self.activation = activation
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
        var x = MLX.expandedDimensions(x, axes: [-2, -3])

        let doSort = indices.size >= 64

        var idx = indices
        var inverseOrder = MLXArray()

        if doSort {
            (x, idx, inverseOrder) = gatherSort(x: x, indices: indices)
        }

        let xUp = callSwitchLinear(upProj, x, idx, sortedIndices: doSort)
        let xGate = callSwitchLinear(gateProj, x, idx, sortedIndices: doSort)
        x = callSwitchLinear(downProj, activation(xGate) * xUp, idx, sortedIndices: doSort)

        if doSort {
            x = scatterUnsort(x: x, invOrder: inverseOrder, shape: indices.shape)
        }

        return MLX.squeezed(x, axis: -2)
    }

    private func callSwitchLinear(
        _ module: Module,
        _ x: MLXArray,
        _ indices: MLXArray,
        sortedIndices: Bool
    ) -> MLXArray {
        if let layer = module as? SwitchLinearProtocol {
            return layer(x, indices, sortedIndices: sortedIndices)
        }
        fatalError("Unexpected SwitchLinear module type: \(type(of: module))")
    }
}

final class SwitchLinear: Module, Quantizable {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray?

    init(inputDims: Int, outputDims: Int, numExperts: Int, bias: Bool = false) {
        let scale = (Float(1.0) / Float(inputDims)).squareRoot()
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numExperts, outputDims, inputDims]
        )
        if bias {
            self._bias.wrappedValue = MLXArray.zeros([numExperts, outputDims])
        } else {
            self._bias.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        let weightT = weight.swappedAxes(-1, -2)
        var result = MLX.gatherMM(x, weightT, rhsIndices: indices, sortedIndices: sortedIndices)

        if let bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }

    func toQuantized(groupSize: Int, bits: Int, mode: QuantizationMode) -> Module {
        QuantizedSwitchLinear(self, groupSize: groupSize, bits: bits, mode: mode)
    }
}

final class QuantizedSwitchLinear: Module, Quantized {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "scales") var scales: MLXArray
    @ParameterInfo(key: "biases") var biases: MLXArray?
    @ParameterInfo(key: "bias") var bias: MLXArray?

    let groupSize: Int
    let bits: Int
    let mode: QuantizationMode

    init(
        weight: MLXArray,
        scales: MLXArray,
        biases: MLXArray?,
        bias: MLXArray?,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self._weight.wrappedValue = weight
        self._scales.wrappedValue = scales
        self._biases.wrappedValue = biases
        self._bias.wrappedValue = bias
        super.init()
        self.freeze()
    }

    convenience init(_ other: SwitchLinear, groupSize: Int = 64, bits: Int = 4, mode: QuantizationMode = .affine) {
        var f = other.weight
        if f.dtype != .float32 {
            f = f.asType(.float32)
        }
        let (wq, scales, biases) = MLX.quantized(f, groupSize: groupSize, bits: bits, mode: mode)
        self.init(
            weight: wq,
            scales: scales,
            biases: biases,
            bias: other.bias,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    func callAsFunction(_ x: MLXArray, _ indices: MLXArray, sortedIndices: Bool) -> MLXArray {
        var result = MLX.gatherQuantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            rhsIndices: indices,
            transpose: true,
            groupSize: groupSize,
            bits: bits,
            mode: mode,
            sortedIndices: sortedIndices
        )

        if let bias {
            result = result + MLX.expandedDimensions(bias[indices], axis: -2)
        }

        return result
    }
}
