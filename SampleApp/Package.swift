// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "NimboChat",
    platforms: [
        .iOS(.v18)
    ],
    products: [
        .executable(
            name: "NimboChat",
            targets: ["NimboChat"]
        )
    ],
    dependencies: [
        .package(path: "NimboCore")
    ],
    targets: [
        .executableTarget(
            name: "NimboChat",
            dependencies: [
                .product(name: "NimboCore", package: "NimboCore")
            ],
            path: "NimboChat",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ]
        )
    ]
)
