//
//  ModelManagerViewModel.swift
//  NimboChat
//
//  Model management: scan, load, unload local CoreML models
//

import Foundation
import SwiftUI
import Observation

/// ViewModel for managing local CoreML models
@Observable
@MainActor
final class ModelManagerViewModel {
    // MARK: - State

    /// All discovered local models
    var localModels: [LocalModelInfo] = []

    /// Currently loaded model ID
    var loadedModelId: String?

    /// Currently loading model ID
    var loadingModelId: String?

    /// Whether a model is currently being loaded
    var isLoadingModel: Bool = false

    /// Loading progress
    var loadingProgress: ModelLoadingProgress?

    /// Name of model being loaded (for display)
    var loadingModelName: String?

    /// Error message
    var errorMessage: String?

    /// Whether initial model scan has completed
    var hasCompletedInitialLoad: Bool = false

    /// Request to show model selection sheet
    var requestModelSelection: Bool = false

    // MARK: - Dependencies

    private let inferenceService = InferenceService.shared

    // MARK: - Initialization

    init() {
        Task {
            await scanForModels()
            hasCompletedInitialLoad = true

            // Auto-load last model if enabled
            await autoLoadLastModel()
        }
    }

    // MARK: - Model Scanning

    /// Scan Documents/Models/ directory for available models
    func scanForModels() async {
        let modelsDir = await StorageService.shared.modelsDirectory
        let fileManager = FileManager.default

        // Ensure Models directory exists
        if !fileManager.fileExists(atPath: modelsDir.path) {
            try? fileManager.createDirectory(at: modelsDir, withIntermediateDirectories: true)
        }

        var found: [LocalModelInfo] = []

        do {
            let contents = try fileManager.contentsOfDirectory(
                at: modelsDir,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            )

            for item in contents {
                let values = try? item.resourceValues(forKeys: [.isDirectoryKey])
                guard values?.isDirectory == true else { continue }

                let modelInfo = LocalModelInfo(path: item)
                if modelInfo.isValid {
                    found.append(modelInfo)
                }
            }
        } catch {
            logError("Failed to scan models directory: \(error)", category: .model)
        }

        // Also check bookmarked external paths
        let bookmarkedModels = loadBookmarkedModels()
        found.append(contentsOf: bookmarkedModels)

        localModels = found
        logInfo("Found \(found.count) local models", category: .model)
    }

    // MARK: - Model Loading

    /// Load a model by its path
    func loadModel(_ model: LocalModelInfo) async {
        guard !isLoadingModel else { return }

        isLoadingModel = true
        loadingModelId = model.id
        loadingModelName = model.displayName
        errorMessage = nil

        // Start polling loading progress
        let progressTask = Task {
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(100))
                if let progress = inferenceService.loadingProgress {
                    self.loadingProgress = progress
                }
            }
        }

        do {
            // Start security-scoped access if needed
            let didStartAccess = model.path.startAccessingSecurityScopedResource()

            try await inferenceService.loadModel(from: model.path)

            if didStartAccess {
                model.path.stopAccessingSecurityScopedResource()
            }

            loadedModelId = model.id
            loadingProgress = nil

            // Remember last model
            await StorageService.shared.saveSelectedModelId(model.id)

            logInfo("Loaded model: \(model.displayName)", category: .model)

        } catch {
            errorMessage = error.localizedDescription
            loadingProgress = nil
            logError("Failed to load model: \(error)", category: .model)
        }

        progressTask.cancel()
        isLoadingModel = false
        loadingModelId = nil
        loadingModelName = nil
    }

    /// Unload the current model
    func unloadModel() async {
        await inferenceService.unloadModel()
        loadedModelId = nil
        loadingProgress = nil
        logInfo("Model unloaded", category: .model)
    }

    // MARK: - File Picker

    /// Add a model from UIDocumentPicker result
    func addModelFromPicker(url: URL) async {
        // Start security-scoped access
        guard url.startAccessingSecurityScopedResource() else {
            errorMessage = "Cannot access the selected folder"
            return
        }

        // Save bookmark for future access
        let key = url.lastPathComponent
        await StorageService.shared.saveBookmark(for: url, key: key)

        // Add to local models
        let modelInfo = LocalModelInfo(path: url)
        if modelInfo.isValid {
            if !localModels.contains(where: { $0.id == modelInfo.id }) {
                localModels.append(modelInfo)
            }
            logInfo("Added external model: \(modelInfo.displayName)", category: .model)
        } else {
            errorMessage = "Selected folder does not contain a valid model (no meta.yaml found)"
        }

        url.stopAccessingSecurityScopedResource()
    }

    // MARK: - Auto-Load

    /// Auto-load the last used model
    func autoLoadLastModel() async {
        let autoLoad = await StorageService.shared.autoLoadLastModel
        guard autoLoad else { return }

        guard let lastModelId = await StorageService.shared.selectedModelId else { return }

        // Find the model in local models
        if let model = localModels.first(where: { $0.id == lastModelId }) {
            await loadModel(model)
        }
    }

    // MARK: - Helpers

    /// Load bookmarked external model paths
    private func loadBookmarkedModels() -> [LocalModelInfo] {
        var models: [LocalModelInfo] = []

        let defaults = UserDefaults.standard
        let bookmarkKeys = defaults.dictionaryRepresentation().keys.filter { $0.hasPrefix("bookmark_") }

        for key in bookmarkKeys {
            guard let bookmarkData = defaults.data(forKey: key) else { continue }

            do {
                var isStale = false
                let url = try URL(resolvingBookmarkData: bookmarkData, options: [], relativeTo: nil, bookmarkDataIsStale: &isStale)
                let didAccess = url.startAccessingSecurityScopedResource()
                let info = LocalModelInfo(path: url)
                if info.isValid {
                    models.append(info)
                }
                if didAccess {
                    url.stopAccessingSecurityScopedResource()
                }
            } catch {
                logWarning("Failed to resolve bookmark \(key): \(error)", category: .model)
            }
        }

        return models
    }
}
