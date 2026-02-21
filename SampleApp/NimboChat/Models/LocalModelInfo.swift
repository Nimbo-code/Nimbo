//
//  LocalModelInfo.swift
//  NimboChat
//
//  Local model information for model picker
//

import Foundation

/// Represents a locally available CoreML model
struct LocalModelInfo: Identifiable, Sendable {
    let id: String
    let name: String
    let path: URL
    let hasMetaYaml: Bool

    /// Display-friendly name derived from directory name
    var displayName: String {
        // Clean up directory name for display
        name.replacingOccurrences(of: "_", with: " ")
            .replacingOccurrences(of: "-", with: " ")
    }

    /// Check if this model directory appears valid
    var isValid: Bool {
        hasMetaYaml
    }

    init(path: URL) {
        self.path = path
        self.name = path.lastPathComponent
        self.id = path.lastPathComponent
        self.hasMetaYaml = FileManager.default.fileExists(
            atPath: path.appendingPathComponent("meta.yaml").path
        )
    }
}
