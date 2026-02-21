//
//  ChatViewModel.swift
//  NimboChat
//
//  ViewModel for chat functionality
//

import Foundation
import SwiftUI
import Observation

/// Main view model for chat functionality
@Observable
@MainActor
final class ChatViewModel {
    // MARK: - Published State

    var conversations: [Conversation] = []
    var currentConversation: Conversation?
    var inputText: String = ""
    var isGenerating: Bool = false
    var streamingContent: String = ""
    var errorMessage: String?
    var currentWindowShifts: Int = 0
    var currentTokensPerSecond: Double = 0
    var currentHistoryTokens: Int = 0
    var pendingScrollToBottomRequest: UUID?

    var debugLevel: Int {
        inferenceService.debugLevel
    }

    var uiUpdatesPaused: Bool = false
    private var pendingStreamingText: String = ""
    private var pendingHistoryTokens: Int?
    private var pendingWindowShifts: Int = 0

    // MARK: - Dependencies

    private let inferenceService = InferenceService.shared

    // MARK: - Initialization

    init() {
        Task {
            await loadConversations()
        }
    }

    // MARK: - Conversation Management

    func loadConversations() async {
        do {
            conversations = try await StorageService.shared.loadConversations()
            logInfo("Loaded \(conversations.count) conversations", category: .app)

            if currentConversation == nil {
                let loadLastChat = await StorageService.shared.loadLastChat
                if loadLastChat, let first = conversations.first {
                    currentConversation = first
                } else {
                    newConversation()
                }
            }
        } catch {
            logError("Failed to load conversations: \(error)", category: .storage)
            errorMessage = "Failed to load conversations"

            if currentConversation == nil {
                newConversation()
            }
        }
    }

    func newConversation() {
        let conversation = Conversation()
        conversations.insert(conversation, at: 0)
        currentConversation = conversation

        Task {
            try? await StorageService.shared.saveConversation(conversation)
        }

        logDebug("Created new conversation", category: .app)
    }

    func selectConversation(_ conversation: Conversation) {
        currentConversation = conversation
    }

    func deleteConversation(_ conversation: Conversation) {
        conversations.removeAll { $0.id == conversation.id }

        if currentConversation?.id == conversation.id {
            currentConversation = conversations.first
        }

        Task {
            try? await StorageService.shared.deleteConversation(conversation.id)
        }

        logDebug("Deleted conversation: \(conversation.id)", category: .app)
    }

    func deleteConversation(at indexSet: IndexSet) {
        for index in indexSet {
            let conversation = conversations[index]
            deleteConversation(conversation)
        }
    }

    func clearAllConversations() {
        let conversationsToDelete = conversations
        conversations.removeAll()
        currentConversation = nil

        Task {
            for conversation in conversationsToDelete {
                try? await StorageService.shared.deleteConversation(conversation.id)
            }
        }

        newConversation()

        logDebug("Cleared all conversations", category: .app)
    }

    // MARK: - Message Sending

    func sendMessage() async {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        guard !isGenerating else { return }

        if currentConversation == nil {
            newConversation()
        }

        guard var conversation = currentConversation else { return }

        inputText = ""

        let userMessage = ChatMessage.user(text)
        conversation.addMessage(userMessage)

        let assistantMessage = ChatMessage.assistant("", isComplete: false)
        conversation.addMessage(assistantMessage)

        currentConversation = conversation
        updateConversationInList(conversation)

        isGenerating = true
        streamingContent = ""
        currentWindowShifts = 0
        currentTokensPerSecond = 0
        currentHistoryTokens = 0
        errorMessage = nil

        let preScrollId = UUID()
        pendingScrollToBottomRequest = preScrollId
        await Task.yield()
        try? await Task.sleep(for: .milliseconds(400))
        if pendingScrollToBottomRequest == preScrollId {
            pendingScrollToBottomRequest = nil
        }

        do {
            let contextMessages = conversation.messages.filter {
                $0.role != .assistant || !$0.content.isEmpty
            }

            let result = try await inferenceService.generateResponse(
                for: contextMessages,
                onToken: { [weak self] token in
                    Task { @MainActor in
                        guard let self = self else { return }
                        if self.uiUpdatesPaused {
                            self.pendingStreamingText += token
                        } else {
                            self.streamingContent += token
                        }
                    }
                },
                onWindowShift: { [weak self] in
                    Task { @MainActor in
                        guard let self = self else { return }
                        if self.uiUpdatesPaused {
                            self.pendingWindowShifts += 1
                        } else {
                            self.currentWindowShifts += 1
                        }
                    }
                },
                onHistoryUpdate: { [weak self] historyTokens in
                    Task { @MainActor in
                        guard let self = self else { return }
                        if self.uiUpdatesPaused {
                            self.pendingHistoryTokens = historyTokens
                        } else {
                            self.currentHistoryTokens = historyTokens
                        }
                    }
                }
            )

            conversation.updateLastAssistantMessage(
                content: result.text,
                tokensPerSecond: result.tokensPerSecond,
                tokenCount: result.tokenCount,
                windowShifts: result.windowShifts,
                prefillTime: result.prefillTime,
                prefillTokens: result.prefillTokens,
                historyTokens: result.historyTokens,
                isComplete: true,
                wasCancelled: result.wasCancelled,
                stopReason: result.stopReason
            )

            currentConversation = conversation
            updateConversationInList(conversation)

            try? await StorageService.shared.saveConversation(conversation)

            logInfo("Message generated: \(result.tokenCount) tokens", category: .app)

        } catch {
            logError("Generation failed: \(error)", category: .inference)
            errorMessage = error.localizedDescription

            conversation.messages.removeLast()
            currentConversation = conversation
            updateConversationInList(conversation)
        }

        isGenerating = false
        streamingContent = ""
    }

    func setUIUpdatesPaused(_ paused: Bool) {
        guard paused != uiUpdatesPaused else { return }
        uiUpdatesPaused = paused

        if !paused {
            flushPendingStreamingUpdates()
        }
    }

    private func flushPendingStreamingUpdates() {
        if !pendingStreamingText.isEmpty {
            streamingContent += pendingStreamingText
            pendingStreamingText = ""
        }

        if let history = pendingHistoryTokens {
            currentHistoryTokens = history
            pendingHistoryTokens = nil
        }

        if pendingWindowShifts > 0 {
            currentWindowShifts += pendingWindowShifts
            pendingWindowShifts = 0
        }
    }

    func cancelGeneration() {
        inferenceService.cancelGeneration()
    }

    private func updateConversationInList(_ conversation: Conversation) {
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            conversations[index] = conversation
        }
    }

    // MARK: - Model Management

    var isModelLoaded: Bool {
        inferenceService.isModelLoaded
    }

    var modelLoadingProgress: ModelLoadingProgress? {
        inferenceService.loadingProgress
    }

    func loadModel(from path: URL) async throws {
        try await inferenceService.loadModel(from: path)
    }

    func unloadModel() async {
        await inferenceService.unloadModel()
    }

    // MARK: - Settings

    var temperature: Float {
        get { inferenceService.temperature }
        set { inferenceService.temperature = newValue }
    }

    var maxTokens: Int {
        get { inferenceService.maxTokens }
        set { inferenceService.maxTokens = newValue }
    }

    var systemPrompt: String {
        get { inferenceService.systemPrompt }
        set { inferenceService.systemPrompt = newValue }
    }

    func saveSettings() async {
        await StorageService.shared.saveTemperature(temperature)
        await StorageService.shared.saveMaxTokens(maxTokens)
        await StorageService.shared.saveSystemPrompt(systemPrompt)
    }
}
