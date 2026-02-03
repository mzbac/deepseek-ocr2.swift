// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "DeepSeekOCR2",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .library(name: "DeepSeekOCR2", targets: ["DeepSeekOCR2"]),
        .executable(name: "DeepSeekOCR2CLI", targets: ["DeepSeekOCR2CLI"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", .upToNextMinor(from: "0.30.3")),
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            .upToNextMinor(from: "1.1.6")
        ),
        .package(url: "https://github.com/apple/swift-argument-parser", .upToNextMinor(from: "1.5.0")),
    ],
    targets: [
        .target(
            name: "DeepSeekOCR2",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .executableTarget(
            name: "DeepSeekOCR2CLI",
            dependencies: [
                "DeepSeekOCR2",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
        .testTarget(
            name: "DeepSeekOCR2Tests",
            dependencies: [
                "DeepSeekOCR2"
            ],
            resources: [
                .process("Fixtures")
            ],
            swiftSettings: [
                .enableExperimentalFeature("StrictConcurrency")
            ]
        ),
    ]
)
