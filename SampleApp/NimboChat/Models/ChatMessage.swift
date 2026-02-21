//
//  ChatMessage.swift
//  NimboChat
//
//  Message model for chat conversations
//

import Foundation

/// Role of a chat message participant
enum MessageRole: String, Codable, Sendable {
    case system
    case user
    case assistant
}

/// A single message in a chat conversation
struct ChatMessage: Identifiable, Codable, Sendable, Equatable {
    let id: UUID
    let role: MessageRole
    var content: String
    let timestamp: Date

    // Performance metrics
    var tokensPerSecond: Double?
    var tokenCount: Int?
    var windowShifts: Int?
    var prefillTime: TimeInterval?
    var prefillTokens: Int?
    var historyTokens: Int?

    // Generation state
    var isComplete: Bool
    var wasCancelled: Bool
    var stopReason: String?

    init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        timestamp: Date = Date(),
        tokensPerSecond: Double? = nil,
        tokenCount: Int? = nil,
        windowShifts: Int? = nil,
        prefillTime: TimeInterval? = nil,
        prefillTokens: Int? = nil,
        historyTokens: Int? = nil,
        isComplete: Bool = true,
        wasCancelled: Bool = false,
        stopReason: String? = nil
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
        self.tokensPerSecond = tokensPerSecond
        self.tokenCount = tokenCount
        self.windowShifts = windowShifts
        self.prefillTime = prefillTime
        self.prefillTokens = prefillTokens
        self.historyTokens = historyTokens
        self.isComplete = isComplete
        self.wasCancelled = wasCancelled
        self.stopReason = stopReason
    }

    static func user(_ content: String) -> ChatMessage {
        ChatMessage(role: .user, content: content)
    }

    static func assistant(_ content: String, isComplete: Bool = true) -> ChatMessage {
        ChatMessage(role: .assistant, content: content, isComplete: isComplete)
    }

    static func system(_ content: String) -> ChatMessage {
        ChatMessage(role: .system, content: content)
    }
}

// MARK: - Formatting Helpers

extension ChatMessage {
    var performanceStats: String? {
        var parts: [String] = []

        if let tps = tokensPerSecond {
            parts.append(String(format: "%.1f tok/s", tps))
        }

        if let count = tokenCount {
            parts.append("\(count) tokens")
        }

        if let prefillTime = prefillTime, let prefillTokens = prefillTokens, prefillTime > 0 {
            let prefillSpeed = Double(prefillTokens) / prefillTime
            parts.append(String(format: "%.1f t/s prefill", prefillSpeed))
        }

        if let ctx = historyTokens {
            parts.append("\(ctx) ctx")
        }

        if let shifts = windowShifts, shifts > 0 {
            parts.append("\(shifts) window shifts")
        }

        return parts.isEmpty ? nil : parts.joined(separator: " | ")
    }

    var formattedTime: String {
        let formatter = DateFormatter()
        formatter.timeStyle = .short
        return formatter.string(from: timestamp)
    }
}
