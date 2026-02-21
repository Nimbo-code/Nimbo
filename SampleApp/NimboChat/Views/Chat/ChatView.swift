//
//  ChatView.swift
//  NimboChat
//
//  Main chat interface (iOS only)
//

import SwiftUI
import UIKit

struct ChatView: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(ModelManagerViewModel.self) private var modelManager

    @State private var scrollProxy: ScrollViewProxy?
    @State private var inputAccessoryHeight: CGFloat = 0

    private var contentBottomPadding: CGFloat {
        max(24, inputAccessoryHeight + 48)
    }

    private var visibleMessages: [ChatMessage] {
        var messages = chatVM.currentConversation?.messages.filter { $0.role != .system } ?? []
        if chatVM.isGenerating, let last = messages.last, last.role == .assistant, !last.isComplete {
            messages.removeLast()
        }
        return messages
    }

    var body: some View {
        ZStack(alignment: .bottom) {
            // Messages
            messagesView

            // No model prompt
            if modelManager.hasCompletedInitialLoad && modelManager.loadedModelId == nil && !modelManager.isLoadingModel {
                VStack(spacing: 16) {
                    Image(systemName: modelManager.errorMessage != nil ? "exclamationmark.triangle" : "cpu")
                        .font(.system(size: 40))
                        .foregroundStyle(modelManager.errorMessage != nil ? .red : .secondary)

                    if let error = modelManager.errorMessage {
                        Text("Model loading failed")
                            .font(.headline)
                            .foregroundStyle(.primary)
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .multilineTextAlignment(.center)
                            .padding(.horizontal, 32)
                    }

                    Button {
                        modelManager.requestModelSelection = true
                    } label: {
                        Label("Select Model", systemImage: "folder")
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .allowsHitTesting(true)
            }

            // Loading overlay
            if let progress = modelManager.loadingProgress, modelManager.isLoadingModel {
                ZStack {
                    Color.black.opacity(0.5)
                        .ignoresSafeArea()
                    Rectangle()
                        .fill(.thickMaterial)
                        .opacity(0.9)
                        .ignoresSafeArea()

                    ModelLoadingGauge(progress: progress, modelName: modelManager.loadingModelName)
                        .fixedSize()
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .transition(.opacity)
                .allowsHitTesting(false)
            }

            // Input bar
            VStack(spacing: 8) {
                InputBar()
                    .environment(chatVM)
            }
            .padding(.horizontal, 12)
            .padding(.bottom, 8)
            .background(
                GeometryReader { geometry in
                    Color.clear.preference(key: InputAccessoryHeightKey.self, value: geometry.size.height)
                }
            )
        }
        .navigationBarTitleDisplayMode(.inline)
        .toolbar(.hidden, for: .navigationBar)
        .errorToast(Binding(
            get: { chatVM.errorMessage },
            set: { chatVM.errorMessage = $0 }
        ))
        .onPreferenceChange(InputAccessoryHeightKey.self) { height in
            inputAccessoryHeight = height
        }
        .onChange(of: chatVM.pendingScrollToBottomRequest) { _, requestId in
            guard requestId != nil else { return }
            scrollToBottom()
        }
    }

    // MARK: - Messages View

    private var messagesView: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(spacing: 14) {
                    ForEach(visibleMessages) { message in
                        MessageBubble(message: message)
                            .id(message.id)
                    }

                    // Streaming message
                    if chatVM.isGenerating {
                        StreamingMessageView(content: chatVM.streamingContent)
                            .id("streaming")
                    }

                    // Bottom spacer
                    Color.clear
                        .frame(height: contentBottomPadding + 24)
                        .id("bottom")
                }
                .padding(.horizontal, 18)
                .padding(.top, 60) // Space for floating controls
            }
            .onAppear {
                scrollProxy = proxy
            }
            .onChange(of: chatVM.streamingContent) { _, _ in
                if chatVM.isGenerating {
                    scrollToBottom()
                }
            }
        }
        .background(chatBackground)
    }

    private func scrollToBottom() {
        withAnimation(.easeInOut(duration: 0.2)) {
            scrollProxy?.scrollTo("bottom", anchor: .bottom)
        }
    }

    private struct InputAccessoryHeightKey: PreferenceKey {
        static var defaultValue: CGFloat = 0
        static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
            value = nextValue()
        }
    }
}

// MARK: - Streaming Message View

struct StreamingMessageView: View {
    let content: String
    @State private var cursorVisible = true

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            RoundedRectangle(cornerRadius: 2, style: .continuous)
                .fill(gaugeAccent)
                .frame(width: 3)

            VStack(alignment: .leading, spacing: 0) {
                if content.isEmpty {
                    thinkingDots
                } else {
                    HStack(alignment: .bottom, spacing: 0) {
                        Text(content)
                            .textSelection(.enabled)
                            .lineSpacing(3)

                        Text("|")
                            .fontWeight(.light)
                            .opacity(cursorVisible ? 1 : 0)
                            .animation(.easeInOut(duration: 0.5).repeatForever(autoreverses: true), value: cursorVisible)
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.vertical, 6)
        .frame(maxWidth: .infinity, alignment: .leading)
        .onAppear {
            cursorVisible = true
        }
    }

    private var thinkingDots: some View {
        HStack(spacing: 4) {
            ForEach(0..<3, id: \.self) { index in
                Circle()
                    .fill(gaugeAccent)
                    .frame(width: 6, height: 6)
            }
        }
    }
}

// MARK: - Model Loading Gauge

struct ModelLoadingGauge: View {
    let progress: ModelLoadingProgress
    var modelName: String? = nil

    private var clampedProgress: Double {
        min(max(progress.percentage, 0), 1)
    }

    var body: some View {
        VStack(spacing: 10) {
            VStack(spacing: 2) {
                Text(progress.stage)
                    .font(.callout)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                if let detail = progress.detail, !detail.isEmpty, !detail.contains("/") {
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.center)
                }
            }

            VStack(spacing: 8) {
                // Progress bar
                GeometryReader { geometry in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(Color.black.opacity(0.35))

                        RoundedRectangle(cornerRadius: 8, style: .continuous)
                            .fill(
                                LinearGradient(
                                    colors: [gaugeAccent.opacity(0.5), gaugeAccent, Color.white.opacity(0.9)],
                                    startPoint: .leading,
                                    endPoint: .trailing
                                )
                            )
                            .frame(width: max(0, geometry.size.width * clampedProgress))
                    }
                }
                .frame(width: 240, height: 16)

                HStack(spacing: 10) {
                    Text("Loading...")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(gaugeAccent.opacity(0.85))
                        .textCase(.uppercase)

                    Text("\(Int(clampedProgress * 100))%")
                        .font(.caption)
                        .fontWeight(.semibold)
                        .foregroundStyle(.primary)
                        .monospacedDigit()
                }

                if let name = modelName, !name.isEmpty {
                    Text(name)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
            }
        }
        .padding(.horizontal, 22)
        .padding(.vertical, 18)
        .background(Color.white.opacity(0.06), in: RoundedRectangle(cornerRadius: 18, style: .continuous))
    }
}

// MARK: - Platform Colors

private let chatBackground = LinearGradient(
    colors: [
        Color(red: 0.06, green: 0.07, blue: 0.08),
        Color(red: 0.03, green: 0.03, blue: 0.04)
    ],
    startPoint: .topLeading,
    endPoint: .bottomTrailing
)

let gaugeAccent = Color(red: 1.0, green: 0.62, blue: 0.2)
