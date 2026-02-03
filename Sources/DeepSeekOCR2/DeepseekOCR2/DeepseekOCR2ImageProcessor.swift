import CoreImage
import CoreImage.CIFilterBuiltins
import Foundation
import MLX

public struct DeepseekOCR2ProcessedImages {
    public let globalView: MLXArray
    public let crops: MLXArray?
    public let cropRatio: (Int, Int)
    public let numImageTokens: Int
}

public struct DeepseekOCR2ImageProcessor {
    public static let imageMean: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)
    public static let imageStd: (CGFloat, CGFloat, CGFloat) = (0.5, 0.5, 0.5)

    private static let normalizationHalf: Float = 0.5
    private static let inv255: Float = 1.0 / 255.0

    public let baseSize: Int
    public let localSize: Int
    public let cropMode: Bool
    public let minPatches: Int
    public let maxPatches: Int

    private let context = CIContext()

    public init(
        baseSize: Int = 1024,
        localSize: Int = 768,
        cropMode: Bool = true,
        minPatches: Int = 2,
        maxPatches: Int = 6
    ) throws {
        guard baseSize == 1024 else {
            throw DeepseekOCR2Error.imageProcessingFailed(
                "DeepseekOCR2ImageProcessor currently supports baseSize=1024 only (got baseSize=\(baseSize))."
            )
        }
        guard localSize == 768 else {
            throw DeepseekOCR2Error.imageProcessingFailed(
                "DeepseekOCR2ImageProcessor currently supports localSize=768 only (got localSize=\(localSize))."
            )
        }

        self.baseSize = baseSize
        self.localSize = localSize
        self.cropMode = cropMode
        self.minPatches = minPatches
        self.maxPatches = maxPatches
    }

    public func process(imageAt url: URL) throws -> DeepseekOCR2ProcessedImages {
        guard let ciImage = CIImage(contentsOf: url) else {
            throw DeepseekOCR2Error.imageLoadFailed(url.path)
        }
        return try process(ciImage)
    }

    public func process(imageAt path: String) throws -> DeepseekOCR2ProcessedImages {
        try process(imageAt: URL(fileURLWithPath: path))
    }

    public func process(_ image: CIImage) throws -> DeepseekOCR2ProcessedImages {
        let extent = image.extent
        guard extent.width > 0, extent.height > 0 else {
            throw DeepseekOCR2Error.imageProcessingFailed(
                "Invalid image extent (width=\(extent.width), height=\(extent.height))"
            )
        }

        guard cropMode else {
            return try processNoCropMode(image)
        }
        return try processCropMode(image)
    }

    private func processNoCropMode(_ image: CIImage) throws -> DeepseekOCR2ProcessedImages {
        let globalTargetSize = CGSize(width: CGFloat(baseSize), height: CGFloat(baseSize))
        let globalPadded = padToSquare(image, targetSize: globalTargetSize)
        let globalView = try normalizeAndConvert(globalPadded)

        let numImageTokens = 256 + 1
        return .init(globalView: globalView, crops: nil, cropRatio: (1, 1), numImageTokens: numImageTokens)
    }

    private func processCropMode(_ image: CIImage) throws -> DeepseekOCR2ProcessedImages {
        let originalWidth = Int(image.extent.width)
        let originalHeight = Int(image.extent.height)

        var cropRatio: (Int, Int) = (1, 1)
        var cropImages: [CIImage] = []

        if originalWidth > localSize || originalHeight > localSize {
            cropRatio = findClosestAspectRatio(
                width: originalWidth,
                height: originalHeight,
                imageSize: localSize,
                minPatches: minPatches,
                maxPatches: maxPatches
            )

            if cropRatio.0 > 1 || cropRatio.1 > 1 {
                cropImages = try dynamicPreprocess(image, imageSize: localSize, targetRatio: cropRatio)
            }
        }

        let globalTargetSize = CGSize(width: CGFloat(baseSize), height: CGFloat(baseSize))
        let globalPadded = padToSquare(image, targetSize: globalTargetSize)
        let globalView = try normalizeAndConvert(globalPadded)

        var cropsArray: MLXArray?
        if !cropImages.isEmpty {
            let processedCrops = try cropImages.map { try normalizeAndConvert($0).squeezed(axis: 0) }
            cropsArray = stacked(processedCrops, axis: 0)
        }

        let numCrops = cropRatio.0 > 1 || cropRatio.1 > 1 ? (cropRatio.0 * cropRatio.1) : 0
        let numImageTokens = (numCrops * 144) + 256 + 1

        return .init(globalView: globalView, crops: cropsArray, cropRatio: cropRatio, numImageTokens: numImageTokens)
    }

    private func findClosestAspectRatio(
        width: Int,
        height: Int,
        imageSize: Int,
        minPatches: Int,
        maxPatches: Int
    ) -> (Int, Int) {
        let aspectRatio = Double(width) / Double(height)
        let area = width * height

        var targetRatios: [(Int, Int)] = []
        for n in minPatches...maxPatches {
            for i in 1...n {
                for j in 1...n {
                    if i * j <= maxPatches && i * j >= minPatches {
                        targetRatios.append((i, j))
                    }
                }
            }
        }
        targetRatios.sort { $0.0 * $0.1 < $1.0 * $1.1 }

        var bestRatioDiff = Double.infinity
        var bestRatio = (1, 1)

        for ratio in targetRatios {
            let targetAspectRatio = Double(ratio.0) / Double(ratio.1)
            let ratioDiff = abs(aspectRatio - targetAspectRatio)

            if ratioDiff < bestRatioDiff {
                bestRatioDiff = ratioDiff
                bestRatio = ratio
            } else if ratioDiff == bestRatioDiff {
                if Double(area) > 0.5 * Double(imageSize * imageSize * ratio.0 * ratio.1) {
                    bestRatio = ratio
                }
            }
        }

        return bestRatio
    }

    private func dynamicPreprocess(_ image: CIImage, imageSize: Int, targetRatio: (Int, Int)) throws -> [CIImage] {
        let targetWidth = imageSize * targetRatio.0
        let targetHeight = imageSize * targetRatio.1
        let blocks = targetRatio.0 * targetRatio.1

        let resizedImage = try resizeDirectly(
            image,
            to: CGSize(width: CGFloat(targetWidth), height: CGFloat(targetHeight))
        )

        var processedImages: [CIImage] = []
        for i in 0..<blocks {
            let col = i % targetRatio.0
            let row = i / targetRatio.0
            let flippedRow = targetRatio.1 - 1 - row

            let cropRect = CGRect(
                x: CGFloat(col * imageSize),
                y: CGFloat(flippedRow * imageSize),
                width: CGFloat(imageSize),
                height: CGFloat(imageSize)
            )

            let croppedImage = resizedImage.cropped(to: cropRect)
            let translatedImage = croppedImage.transformed(
                by: CGAffineTransform(translationX: -cropRect.origin.x, y: -cropRect.origin.y)
            )
            processedImages.append(translatedImage)
        }

        return processedImages
    }

    private func resizeDirectly(_ image: CIImage, to size: CGSize) throws -> CIImage {
        let scaleX = size.width / image.extent.width
        let scaleY = size.height / image.extent.height

        let filter = CIFilter.bicubicScaleTransform()
        filter.inputImage = image
        filter.scale = Float(scaleY)
        filter.aspectRatio = Float(scaleX / scaleY)
        guard let scaledImage = filter.outputImage else {
            throw DeepseekOCR2Error.imageProcessingFailed("CoreImage resize failed (bicubicScaleTransform produced nil output).")
        }

        return scaledImage.cropped(to: CGRect(origin: .zero, size: size))
    }

    private func padToSquare(_ image: CIImage, targetSize: CGSize) -> CIImage {
        let originalWidth = image.extent.width
        let originalHeight = image.extent.height

        let scaleX = targetSize.width / originalWidth
        let scaleY = targetSize.height / originalHeight
        let scale = min(scaleX, scaleY)

        let scaledWidth = originalWidth * scale
        let scaledHeight = originalHeight * scale

        let scaleTransform = CGAffineTransform(scaleX: scale, y: scale)
        var scaledImage = image.transformed(by: scaleTransform)

        let padX = (targetSize.width - scaledWidth) / 2
        let padY = (targetSize.height - scaledHeight) / 2

        let translateTransform = CGAffineTransform(translationX: padX, y: padY)
        scaledImage = scaledImage.transformed(by: translateTransform)

        let fillColor = CIColor(red: 0.5, green: 0.5, blue: 0.5, alpha: 1.0)
        let backgroundRect = CGRect(origin: .zero, size: targetSize)
        let background = CIImage(color: fillColor).cropped(to: backgroundRect)

        let composited = scaledImage.composited(over: background)

        return composited.cropped(to: backgroundRect)
    }

    private func normalizeAndConvert(_ image: CIImage) throws -> MLXArray {
        let rawArray = try asMLXArrayRaw(image)
        let half = Self.normalizationHalf
        let normalized = (rawArray - half) / half
        return normalized.asType(.bfloat16)
    }

    private func asMLXArrayRaw(_ image: CIImage) throws -> MLXArray {
        let size = image.extent.size
        let width = Int(size.width.rounded())
        let height = Int(size.height.rounded())
        guard width > 0, height > 0 else {
            throw DeepseekOCR2Error.imageProcessingFailed(
                "Invalid image dimensions (width=\(width), height=\(height))"
            )
        }

        let format = CIFormat.RGBA8
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel
        let bytesPerRow = width * bytesPerPixel

        var data = Data(count: width * height * bytesPerPixel)
        try data.withUnsafeMutableBytes { ptr in
            guard let baseAddress = ptr.baseAddress else {
                throw DeepseekOCR2Error.imageProcessingFailed("Failed to allocate image bitmap buffer.")
            }
            guard let colorSpace = CGColorSpace(name: CGColorSpace.sRGB) else {
                throw DeepseekOCR2Error.imageProcessingFailed("Failed to create sRGB color space.")
            }
            context.render(
                image,
                toBitmap: baseAddress,
                rowBytes: bytesPerRow,
                bounds: image.extent,
                format: format,
                colorSpace: colorSpace
            )
        }

        let uint8Array = MLXArray(data, [height, width, 4], type: UInt8.self)
        var array = uint8Array.asType(.float32) * Self.inv255
        array = array[0..., 0..., ..<3]
        array = array.reshaped(1, height, width, 3)

        return array
    }

    public func clearCaches() {
        context.clearCaches()
    }
}
