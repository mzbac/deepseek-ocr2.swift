import Foundation
import MLX
import MLXNN

final class SamMLPBlock: Module, UnaryLayer {
    @ModuleInfo(key: "lin1") var lin1: Linear
    @ModuleInfo(key: "lin2") var lin2: Linear

    init(embeddingDim: Int, mlpDim: Int) {
        self._lin1.wrappedValue = Linear(embeddingDim, mlpDim)
        self._lin2.wrappedValue = Linear(mlpDim, embeddingDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lin2(gelu(lin1(x)))
    }
}

final class SamAttention: Module {
    private let numHeads: Int
    private let headDim: Int
    private let scale: Float
    private let useRelPos: Bool
    private let windowSize: Int
    private let baseImageGrid: Int

    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear

    @ParameterInfo(key: "rel_pos_h") var relPosH: MLXArray
    @ParameterInfo(key: "rel_pos_w") var relPosW: MLXArray

    private struct RelPosResampleKey: Hashable {
        let qSize: Int
        let kSize: Int
        let inputLength: Int
    }

    private struct RelPosIndexKey: Hashable {
        let qSize: Int
        let kSize: Int
    }

    private var relPosHResampleCache: [RelPosResampleKey: MLXArray] = [:]
    private var relPosWResampleCache: [RelPosResampleKey: MLXArray] = [:]
    private var relPosIndexCache: [RelPosIndexKey: MLXArray] = [:]

    init(embedDim: Int, numHeads: Int, windowSize: Int, baseImageGrid: Int, useRelPos: Bool) {
        precondition(embedDim % numHeads == 0, "embedDim must be divisible by numHeads")
        self.numHeads = numHeads
        self.headDim = embedDim / numHeads
        self.scale = 1.0 / sqrt(Float(headDim))
        self.useRelPos = useRelPos
        self.windowSize = windowSize
        self.baseImageGrid = baseImageGrid

        self._qkv.wrappedValue = Linear(embedDim, embedDim * 3, bias: true)
        self._proj.wrappedValue = Linear(embedDim, embedDim, bias: true)

        let relInputSize = windowSize > 0 ? windowSize : baseImageGrid
        self._relPosH.wrappedValue = MLXArray.zeros([2 * relInputSize - 1, headDim])
        self._relPosW.wrappedValue = MLXArray.zeros([2 * relInputSize - 1, headDim])

        super.init()
    }

    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws -> Self {
        relPosHResampleCache.removeAll(keepingCapacity: true)
        relPosWResampleCache.removeAll(keepingCapacity: true)
        relPosIndexCache.removeAll(keepingCapacity: true)
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
        let (batchSize, height, width, _) = (
            hiddenStates.dim(0),
            hiddenStates.dim(1),
            hiddenStates.dim(2),
            hiddenStates.dim(3)
        )

        var qkvOut = qkv(hiddenStates)
        qkvOut =
            qkvOut
            .reshaped(batchSize, height * width, 3, numHeads, -1)
            .transposed(2, 0, 3, 1, 4)

        let q = qkvOut[0]
        let k = qkvOut[1]
        let v = qkvOut[2]

        let attnBias: MLXArray?
        if useRelPos {
            attnBias = computeRelPosBias(queries: q, height: height, width: width)
        } else {
            attnBias = nil
        }

        let attnOutput: MLXArray
        if let attnBias {
            attnOutput = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .array(attnBias)
            )
        } else {
            attnOutput = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .none
            )
        }

        let merged =
            attnOutput
            .reshaped(batchSize, numHeads, height, width, -1)
            .transposed(0, 2, 3, 1, 4)
            .reshaped(batchSize, height, width, -1)

        return proj(merged)
    }

    private func computeRelPosBias(queries: MLXArray, height: Int, width: Int) -> MLXArray {
        // queries: [B, heads, H*W, headDim]
        // returns: [B, heads, H*W, H*W] additive bias
        let batchSize = queries.dim(0)

        let qFlat = queries.reshaped(batchSize * numHeads, height * width, headDim)
        let rQ = qFlat.reshaped(batchSize * numHeads, height, width, headDim)

        let Rh = getRelPos(qSize: height, kSize: height, relPos: relPosH, cache: &relPosHResampleCache)
        let Rw = getRelPos(qSize: width, kSize: width, relPos: relPosW, cache: &relPosWResampleCache)

        let relH = matmul(rQ, Rh.swappedAxes(-2, -1))
        var relW = matmul(rQ.transposed(0, 2, 1, 3), Rw.swappedAxes(-2, -1))
        relW = relW.transposed(0, 2, 1, 3)

        let relHBias = relH.reshaped(batchSize * numHeads, height * width, height, 1)
        let relWBias = relW.reshaped(batchSize * numHeads, height * width, 1, width)

        let combined = relHBias + relWBias
        return combined
            .reshaped(batchSize, numHeads, height * width, height, width)
            .reshaped(batchSize, numHeads, height * width, height * width)
    }

    private func getRelPos(
        qSize: Int,
        kSize: Int,
        relPos: MLXArray,
        cache: inout [RelPosResampleKey: MLXArray]
    ) -> MLXArray {
        let maxRelDist = 2 * max(qSize, kSize) - 1
        let relPosResized: MLXArray = {
            let inputLength = relPos.dim(0)
            guard inputLength != maxRelDist else { return relPos }

            let key = RelPosResampleKey(qSize: qSize, kSize: kSize, inputLength: inputLength)
            if let cached = cache[key] { return cached }

            let relPos3D = relPos.expandedDimensions(axis: 0)
            let scale = Float(maxRelDist) / Float(inputLength)
            let resized = Upsample(scaleFactor: [scale], mode: .linear(alignCorners: false))(relPos3D)
                .squeezed(axis: 0)
            cache[key] = resized
            return resized
        }()

        let indices: MLXArray = {
            let key = RelPosIndexKey(qSize: qSize, kSize: kSize)
            if let cached = relPosIndexCache[key] { return cached }

            let qScale = max(Float(kSize) / Float(qSize), Float(1))
            let kScale = max(Float(qSize) / Float(kSize), Float(1))

            let qCoords = MLXArray(Int32(0) ..< Int32(qSize)).asType(.float32) * qScale
            let kCoords = MLXArray(Int32(0) ..< Int32(kSize)).asType(.float32) * kScale

            let qCoordsExpanded = qCoords.reshaped(qSize, 1)
            let kCoordsExpanded = kCoords.reshaped(1, kSize)
            let offset = Float(kSize - 1) * kScale
            let relativeCoords = qCoordsExpanded - kCoordsExpanded + offset

            let resolved = relativeCoords.asType(.int32)
            relPosIndexCache[key] = resolved
            return resolved
        }()

        return relPosResized[indices]
    }
}

final class SamViTBlock: Module, UnaryLayer {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "attn") var attn: SamAttention
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "mlp") var mlp: SamMLPBlock

    private let windowSize: Int

    init(embedDim: Int, numHeads: Int, mlpRatio: Int, windowSize: Int, baseImageGrid: Int) {
        self.windowSize = windowSize
        self._norm1.wrappedValue = LayerNorm(dimensions: embedDim, eps: 1e-6)
        self._attn.wrappedValue = SamAttention(
            embedDim: embedDim,
            numHeads: numHeads,
            windowSize: windowSize,
            baseImageGrid: baseImageGrid,
            useRelPos: true
        )
        self._norm2.wrappedValue = LayerNorm(dimensions: embedDim, eps: 1e-6)
        self._mlp.wrappedValue = SamMLPBlock(embeddingDim: embedDim, mlpDim: embedDim * mlpRatio)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let shortcut = x
        var h = norm1(x)

        let (height, width) = (h.dim(1), h.dim(2))

        if windowSize > 0 {
            let (windows, paddingShape, batchSize) = windowPartition(h, windowSize: windowSize)
            let attended = attn(windows)
            h = windowUnpartition(
                attended,
                windowSize: windowSize,
                paddingShape: paddingShape,
                originalShape: (height, width),
                batchSize: batchSize
            )
        } else {
            h = attn(h)
        }

        h = shortcut + h
        return h + mlp(norm2(h))
    }

    private func windowPartition(_ x: MLXArray, windowSize: Int) -> (MLXArray, (Int, Int), Int) {
        let (batchSize, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))

        let padH = (windowSize - height % windowSize) % windowSize
        let padW = (windowSize - width % windowSize) % windowSize

        var h = x
        if padH > 0 || padW > 0 {
            h = padded(h, widths: [[0, 0], [0, padH], [0, padW], [0, 0]])
        }

        let padHeight = height + padH
        let padWidth = width + padW

        h = h.reshaped(batchSize, padHeight / windowSize, windowSize, padWidth / windowSize, windowSize, channels)
        h = h.transposed(0, 1, 3, 2, 4, 5)
        let windows = h.reshaped(-1, windowSize, windowSize, channels)

        return (windows, (padHeight, padWidth), batchSize)
    }

    private func windowUnpartition(
        _ windows: MLXArray,
        windowSize: Int,
        paddingShape: (Int, Int),
        originalShape: (Int, Int),
        batchSize: Int
    ) -> MLXArray {
        let (padHeight, padWidth) = paddingShape
        let (height, width) = originalShape
        let channels = windows.dim(3)

        var h = windows.reshaped(
            batchSize,
            padHeight / windowSize,
            padWidth / windowSize,
            windowSize,
            windowSize,
            channels
        )
        h = h.transposed(0, 1, 3, 2, 4, 5)
        h = h.reshaped(batchSize, padHeight, padWidth, channels)

        if padHeight > height || padWidth > width {
            h = h[0..., 0..<height, 0..<width, 0...]
        }
        return h
    }
}

final class SamPatchEmbed: Module, UnaryLayer {
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(inChannels: Int, embedDim: Int, patchSize: Int) {
        self._proj.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: embedDim,
            kernelSize: .init(patchSize),
            stride: .init(patchSize),
            padding: 0,
            bias: true
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        proj(x)
    }
}

final class SamImageEncoderViT: Module, UnaryLayer {
    @ModuleInfo(key: "patch_embed") var patchEmbed: SamPatchEmbed
    @ParameterInfo(key: "pos_embed") var posEmbed: MLXArray
    var blocks: [SamViTBlock]
    @ModuleInfo(key: "neck") var neck: [UnaryLayer]
    @ModuleInfo(key: "net_2") var net2: Conv2d
    @ModuleInfo(key: "net_3") var net3: Conv2d

    private let baseImageGrid: Int
    private struct PosEmbedKey: Hashable {
        let height: Int
        let width: Int
    }
    private var posEmbedCache: [PosEmbedKey: MLXArray] = [:]

    override init() {
        let imageSize = 1024
        let patchSize = 16
        let embedDim = 768
        let depth = 12
        let numHeads = 12
        let windowSize = 14
        let globalAttnIndexes: Set<Int> = [2, 5, 8, 11]

        let baseGrid = imageSize / patchSize
        self.baseImageGrid = baseGrid

        self._patchEmbed.wrappedValue = SamPatchEmbed(inChannels: 3, embedDim: embedDim, patchSize: patchSize)
        self._posEmbed.wrappedValue = MLXArray.zeros([1, baseGrid, baseGrid, embedDim])

        self.blocks = (0..<depth).map { idx in
            SamViTBlock(
                embedDim: embedDim,
                numHeads: numHeads,
                mlpRatio: 4,
                windowSize: globalAttnIndexes.contains(idx) ? 0 : windowSize,
                baseImageGrid: baseGrid
            )
        }

        let neckLayers: [UnaryLayer] = [
            Conv2d(
                inputChannels: embedDim,
                outputChannels: 256,
                kernelSize: 1,
                bias: false
            ),
            LayerNorm(dimensions: 256, eps: 1e-6),
            Conv2d(
                inputChannels: 256,
                outputChannels: 256,
                kernelSize: 3,
                padding: 1,
                bias: false
            ),
            LayerNorm(dimensions: 256, eps: 1e-6),
        ]
        self._neck.wrappedValue = neckLayers
        self._net2.wrappedValue = Conv2d(
            inputChannels: 256,
            outputChannels: 512,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            bias: false
        )
        self._net3.wrappedValue = Conv2d(
            inputChannels: 512,
            outputChannels: 896,
            kernelSize: 3,
            stride: 2,
            padding: 1,
            bias: false
        )

        super.init()
    }

    func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
        var h = patchEmbed(pixelValues)

        let targetH = h.dim(1)
        let targetW = h.dim(2)
        h = h + resizedPosEmbed(targetH: targetH, targetW: targetW)

        for block in blocks {
            h = block(h)
        }

        for layer in neck {
            h = layer(h)
        }
        h = net2(h)
        h = net3(h)
        return h
    }

    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String],
        modulePath: [String]
    ) throws -> Self {
        posEmbedCache.removeAll(keepingCapacity: true)
        return try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
    }

    private func resizedPosEmbed(targetH: Int, targetW: Int) -> MLXArray {
        let origH = posEmbed.dim(1)
        let origW = posEmbed.dim(2)
        if origH == targetH && origW == targetW {
            return posEmbed
        }

        let key = PosEmbedKey(height: targetH, width: targetW)
        if let cached = posEmbedCache[key] {
            return cached
        }
        let scaleH = Float(targetH) / Float(origH)
        let scaleW = Float(targetW) / Float(origW)
        let resized = Upsample(scaleFactor: [scaleH, scaleW], mode: .cubic(alignCorners: false))(posEmbed)
        posEmbedCache[key] = resized
        return resized
    }
}
