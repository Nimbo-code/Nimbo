// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "NimboCore",
    platforms: [
        .macOS(.v15),
        .iOS(.v18)
    ],
    products: [
        .library(
            name: "NimboCore",
            targets: ["NimboCore"]
        )
    ],
    dependencies: [
        // Transformers with tokenizers
        .package(url: "https://github.com/huggingface/swift-transformers", branch: "main"),
        // YAML parser
        .package(url: "https://github.com/jpsim/Yams.git", from: "5.0.0"),
        // Templating (similar to Jinja)
        .package(url: "https://github.com/stencilproject/Stencil.git", from: "0.14.0")
    ],
    targets: [
        .target(
            name: "NimboCore",
            dependencies: [
                "Yams",
                .product(name: "Transformers", package: "swift-transformers"),
                "Stencil"
            ]
        )
    ]
)
