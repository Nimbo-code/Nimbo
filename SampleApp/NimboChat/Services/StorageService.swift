//
//  StorageService.swift
//  NimboChat
//
//  Persistence for conversations and settings (iOS only)
//

import Foundation

/// Errors that can occur during storage operations
enum StorageError: LocalizedError {
    case encodingFailed
    case decodingFailed
    case fileWriteFailed(Error)
    case fileReadFailed(Error)
    case directoryCreationFailed(Error)

    var errorDescription: String? {
        switch self {
        case .encodingFailed: return "Failed to encode data"
        case .decodingFailed: return "Failed to decode data"
        case .fileWriteFailed(let error): return "Failed to write file: \(error.localizedDescription)"
        case .fileReadFailed(let error): return "Failed to read file: \(error.localizedDescription)"
        case .directoryCreationFailed(let error): return "Failed to create directory: \(error.localizedDescription)"
        }
    }
}

/// Service for persisting app data
actor StorageService {
    static let shared = StorageService()

    private let fileManager = FileManager.default
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    private init() {
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        decoder.dateDecodingStrategy = .iso8601
    }

    // MARK: - Directories

    private var documentsDirectory: URL {
        fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
    }

    private var conversationsDirectory: URL {
        documentsDirectory.appendingPathComponent("Conversations", isDirectory: true)
    }

    /// Models directory - visible in Files app with UIFileSharingEnabled
    var modelsDirectory: URL {
        documentsDirectory.appendingPathComponent("Models", isDirectory: true)
    }

    private func ensureDirectoryExists(_ url: URL) throws {
        if !fileManager.fileExists(atPath: url.path) {
            do {
                try fileManager.createDirectory(at: url, withIntermediateDirectories: true)
                logDebug("Created directory: \(url.path)", category: .storage)
            } catch {
                throw StorageError.directoryCreationFailed(error)
            }
        }
    }

    // MARK: - Conversations

    func saveConversation(_ conversation: Conversation) async throws {
        try ensureDirectoryExists(conversationsDirectory)

        let fileURL = conversationsDirectory.appendingPathComponent("\(conversation.id.uuidString).json")

        do {
            let data = try encoder.encode(conversation)
            try data.write(to: fileURL, options: .atomic)
            logDebug("Saved conversation: \(conversation.id)", category: .storage)
        } catch let error as EncodingError {
            logError("Encoding failed: \(error)", category: .storage)
            throw StorageError.encodingFailed
        } catch {
            logError("Write failed: \(error)", category: .storage)
            throw StorageError.fileWriteFailed(error)
        }
    }

    func loadConversations() async throws -> [Conversation] {
        try ensureDirectoryExists(conversationsDirectory)

        var conversations: [Conversation] = []

        do {
            let files = try fileManager.contentsOfDirectory(
                at: conversationsDirectory,
                includingPropertiesForKeys: nil
            )

            for file in files where file.pathExtension == "json" {
                do {
                    let data = try Data(contentsOf: file)
                    let conversation = try decoder.decode(Conversation.self, from: data)
                    conversations.append(conversation)
                } catch {
                    logWarning("Failed to load conversation \(file.lastPathComponent): \(error)", category: .storage)
                }
            }
        } catch {
            throw StorageError.fileReadFailed(error)
        }

        conversations.sort { $0.updatedAt > $1.updatedAt }
        logInfo("Loaded \(conversations.count) conversations", category: .storage)

        return conversations
    }

    func deleteConversation(_ id: UUID) async throws {
        let fileURL = conversationsDirectory.appendingPathComponent("\(id.uuidString).json")

        if fileManager.fileExists(atPath: fileURL.path) {
            do {
                try fileManager.removeItem(at: fileURL)
                logDebug("Deleted conversation: \(id)", category: .storage)
            } catch {
                throw StorageError.fileWriteFailed(error)
            }
        }
    }

    // MARK: - Model Files

    func modelPath(for modelId: String) -> URL {
        let cleanId = modelId.trimmingCharacters(in: .whitespacesAndNewlines)
        return modelsDirectory.appendingPathComponent(cleanId.replacingOccurrences(of: "/", with: "_"))
    }

    func isModelDownloaded(_ modelId: String) async -> Bool {
        let modelDir = modelPath(for: modelId)
        let metaYaml = modelDir.appendingPathComponent("meta.yaml")
        return fileManager.fileExists(atPath: metaYaml.path)
    }

    // MARK: - Settings

    static let defaultTemperatureValue: Float = 0.0
    static let defaultMaxTokensValue: Int = 2048
    static let defaultSystemPromptValue: String = ""
    static let defaultDebugLevelValue: Int = 0
    static let defaultRepetitionDetectionValue: Bool = false
    static let defaultAutoLoadLastModelValue: Bool = true
    static let defaultLoadLastChatValue: Bool = false

    // Sampling defaults
    static let defaultDoSampleValue: Bool = false
    static let defaultTopPValue: Float = 0.95
    static let defaultTopKValue: Int = 0
    static let defaultUseRecommendedSamplingValue: Bool = true

    var defaultTemperature: Float {
        UserDefaults.standard.object(forKey: "temperature") as? Float ?? Self.defaultTemperatureValue
    }

    var defaultMaxTokens: Int {
        UserDefaults.standard.object(forKey: "maxTokens") as? Int ?? Self.defaultMaxTokensValue
    }

    var defaultSystemPrompt: String {
        UserDefaults.standard.object(forKey: "systemPrompt") as? String ?? Self.defaultSystemPromptValue
    }

    var selectedModelId: String? {
        UserDefaults.standard.object(forKey: "selectedModelId") as? String
    }

    var autoLoadLastModel: Bool {
        UserDefaults.standard.object(forKey: "autoLoadLastModel") as? Bool ?? Self.defaultAutoLoadLastModelValue
    }

    func saveAutoLoadLastModel(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "autoLoadLastModel")
    }

    var debugLevel: Int {
        UserDefaults.standard.object(forKey: "debugLevel") as? Int ?? Self.defaultDebugLevelValue
    }

    func saveDebugLevel(_ value: Int) {
        UserDefaults.standard.set(value, forKey: "debugLevel")
    }

    var repetitionDetectionEnabled: Bool {
        UserDefaults.standard.object(forKey: "repetitionDetectionEnabled") as? Bool ?? Self.defaultRepetitionDetectionValue
    }

    func saveRepetitionDetectionEnabled(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "repetitionDetectionEnabled")
    }

    var loadLastChat: Bool {
        UserDefaults.standard.object(forKey: "loadLastChat") as? Bool ?? Self.defaultLoadLastChatValue
    }

    func saveLoadLastChat(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "loadLastChat")
    }

    // Sampling settings
    var doSample: Bool {
        UserDefaults.standard.object(forKey: "doSample") as? Bool ?? Self.defaultDoSampleValue
    }

    func saveDoSample(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "doSample")
    }

    var topP: Float {
        UserDefaults.standard.object(forKey: "topP") as? Float ?? Self.defaultTopPValue
    }

    func saveTopP(_ value: Float) {
        UserDefaults.standard.set(value, forKey: "topP")
    }

    var topK: Int {
        UserDefaults.standard.object(forKey: "topK") as? Int ?? Self.defaultTopKValue
    }

    func saveTopK(_ value: Int) {
        UserDefaults.standard.set(value, forKey: "topK")
    }

    var useRecommendedSampling: Bool {
        UserDefaults.standard.object(forKey: "useRecommendedSampling") as? Bool ?? Self.defaultUseRecommendedSamplingValue
    }

    func saveUseRecommendedSampling(_ value: Bool) {
        UserDefaults.standard.set(value, forKey: "useRecommendedSampling")
    }

    func clearLastModel() {
        UserDefaults.standard.removeObject(forKey: "selectedModelId")
    }

    func saveTemperature(_ value: Float) {
        UserDefaults.standard.set(value, forKey: "temperature")
    }

    func saveMaxTokens(_ value: Int) {
        UserDefaults.standard.set(value, forKey: "maxTokens")
    }

    func saveSystemPrompt(_ value: String) {
        UserDefaults.standard.set(value, forKey: "systemPrompt")
    }

    func saveSelectedModelId(_ value: String?) {
        UserDefaults.standard.set(value, forKey: "selectedModelId")
    }

    // Security-scoped bookmark for external model paths
    func saveBookmark(for url: URL, key: String) {
        do {
            let bookmarkData = try url.bookmarkData(
                options: .minimalBookmark,
                includingResourceValuesForKeys: nil,
                relativeTo: nil
            )
            UserDefaults.standard.set(bookmarkData, forKey: "bookmark_\(key)")
        } catch {
            logError("Failed to save bookmark: \(error)", category: .storage)
        }
    }

    func resolveBookmark(key: String) -> URL? {
        guard let bookmarkData = UserDefaults.standard.data(forKey: "bookmark_\(key)") else {
            return nil
        }
        do {
            var isStale = false
            let url = try URL(resolvingBookmarkData: bookmarkData, options: [], relativeTo: nil, bookmarkDataIsStale: &isStale)
            if isStale {
                saveBookmark(for: url, key: key)
            }
            return url
        } catch {
            logError("Failed to resolve bookmark: \(error)", category: .storage)
            return nil
        }
    }

    func resetToDefaults() {
        UserDefaults.standard.set(Self.defaultTemperatureValue, forKey: "temperature")
        UserDefaults.standard.set(Self.defaultMaxTokensValue, forKey: "maxTokens")
        UserDefaults.standard.set(Self.defaultSystemPromptValue, forKey: "systemPrompt")
        UserDefaults.standard.set(Self.defaultDebugLevelValue, forKey: "debugLevel")
        UserDefaults.standard.set(Self.defaultRepetitionDetectionValue, forKey: "repetitionDetectionEnabled")
        UserDefaults.standard.set(Self.defaultAutoLoadLastModelValue, forKey: "autoLoadLastModel")
        UserDefaults.standard.set(Self.defaultLoadLastChatValue, forKey: "loadLastChat")
        UserDefaults.standard.set(Self.defaultDoSampleValue, forKey: "doSample")
        UserDefaults.standard.set(Self.defaultTopPValue, forKey: "topP")
        UserDefaults.standard.set(Self.defaultTopKValue, forKey: "topK")
        UserDefaults.standard.set(Self.defaultUseRecommendedSamplingValue, forKey: "useRecommendedSampling")
        logInfo("Settings reset to defaults", category: .storage)
    }
}
