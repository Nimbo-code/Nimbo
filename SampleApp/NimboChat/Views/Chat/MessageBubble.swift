//
//  MessageBubble.swift
//  NimboChat
//
//  Individual message display (plain text, iOS only)
//

import SwiftUI
import UIKit

struct MessageBubble: View, Equatable {
    let message: ChatMessage

    @State private var showCopyButton = false

    private var isUser: Bool {
        message.role == .user
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            messageContent
                .onTapGesture {
                    withAnimation(.easeInOut(duration: 0.2)) {
                        showCopyButton.toggle()
                    }
                }

            // Stats (for assistant messages)
            if !isUser && message.isComplete {
                statsView
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Message Content

    private var messageContent: some View {
        HStack(alignment: .top, spacing: 12) {
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .fill(isUser ? Color.accentColor.opacity(0.9) : llmAccent)
                .frame(width: 3)

            HStack(alignment: .top, spacing: 4) {
                VStack(alignment: .leading, spacing: 8) {
                    if message.content.isEmpty && !message.isComplete {
                        ProgressView()
                            .controlSize(.small)
                    } else {
                        Text(message.content)
                            .textSelection(.enabled)
                            .lineSpacing(3)
                    }
                }

                if !message.content.isEmpty {
                    copyButton
                        .opacity(showCopyButton ? 1 : 0)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 6)
        .foregroundStyle(.primary)
    }

    // MARK: - Copy Button

    @ViewBuilder
    private var copyButton: some View {
        Button {
            UIPasteboard.general.string = message.content
            withAnimation {
                showCopyButton = false
            }
        } label: {
            Image(systemName: "doc.on.doc")
                .font(.caption)
                .padding(5)
                .background(.ultraThinMaterial, in: Circle())
        }
        .buttonStyle(.plain)
    }

    static func == (lhs: MessageBubble, rhs: MessageBubble) -> Bool {
        lhs.message == rhs.message
    }
}

private let llmAccent = Color(red: 1.0, green: 0.62, blue: 0.2)

// MARK: - Stats View

extension MessageBubble {
    @ViewBuilder
    fileprivate var statsView: some View {
        HStack(spacing: 8) {
            // Generation speed
            if let tps = message.tokensPerSecond {
                HStack(spacing: 2) {
                    Image(systemName: "gauge.medium")
                        .font(.caption2)
                    Text(String(format: "%.1f tok/s", tps))
                        .font(.caption2)
                }
                .foregroundStyle(.secondary)
            }

            // Prefill speed
            if let prefillTime = message.prefillTime, let prefillTokens = message.prefillTokens, prefillTime > 0 {
                let prefillSpeed = Double(prefillTokens) / prefillTime
                HStack(spacing: 2) {
                    Image(systemName: "arrow.right.circle")
                        .font(.caption2)
                    Text(String(format: "%.0f t/s", prefillSpeed))
                        .font(.caption2)
                }
                .foregroundStyle(.cyan)
            }

            // History tokens
            if let ctx = message.historyTokens {
                HStack(spacing: 2) {
                    Image(systemName: "text.alignleft")
                        .font(.caption2)
                    Text("\(ctx) ctx")
                        .font(.caption2)
                }
                .foregroundStyle(.green)
            }
        }

        // Window shifts
        if let shifts = message.windowShifts, shifts > 0 {
            HStack(spacing: 4) {
                Image(systemName: "arrow.left.arrow.right")
                    .font(.caption2)
                Text("\(shifts) context shifts")
                    .font(.caption2)
            }
            .foregroundStyle(.orange)
        }

        // Cancelled
        if message.wasCancelled {
            HStack(spacing: 4) {
                Image(systemName: "stop.circle")
                    .font(.caption2)
                Text("Cancelled")
                    .font(.caption2)
            }
            .foregroundStyle(.orange)
        }
    }
}
