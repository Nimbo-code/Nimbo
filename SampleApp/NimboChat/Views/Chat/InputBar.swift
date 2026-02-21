//
//  InputBar.swift
//  NimboChat
//
//  Text input with send/stop button (iOS only, no voice)
//

import SwiftUI

struct InputBar: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(ModelManagerViewModel.self) private var modelManager

    @FocusState private var isFocused: Bool
    @State private var showLoadingToast = false
    @State private var showNoModelToast = false

    var body: some View {
        @Bindable var vm = chatVM

        ZStack(alignment: .top) {
            HStack(alignment: .center, spacing: 12) {
                // Text field
                textField

                // Send/Stop button
                sendButton
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 12)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 26, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 26, style: .continuous)
                    .stroke(Color.white.opacity(0.12), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.2), radius: 12, y: 6)

            // Toast overlay
            if showLoadingToast {
                LoadingToastView(message: "Model still loading...", icon: "hourglass")
                    .transition(.asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .opacity
                    ))
                    .offset(y: -50)
            } else if showNoModelToast {
                LoadingToastView(message: "No model loaded", icon: "cpu.fill")
                    .transition(.asymmetric(
                        insertion: .move(edge: .top).combined(with: .opacity),
                        removal: .opacity
                    ))
                    .offset(y: -50)
            }
        }
    }

    // MARK: - Text Field

    private var textField: some View {
        @Bindable var vm = chatVM

        return TextField("Message...", text: $vm.inputText, axis: .vertical)
            .textFieldStyle(.plain)
            .lineLimit(1...6)
            .focused($isFocused)
            .disabled(chatVM.isGenerating)
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .background(
                RoundedRectangle(cornerRadius: 20)
                    .fill(Color.white.opacity(0.08))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 20)
                    .stroke(Color.white.opacity(0.12), lineWidth: 1)
            )
            .onSubmit {
                sendMessage()
            }
    }

    // MARK: - Send Button

    private var sendButton: some View {
        Button {
            if chatVM.isGenerating {
                chatVM.cancelGeneration()
            } else {
                sendMessage()
            }
        } label: {
            if chatVM.isGenerating {
                // Stop button
                ZStack {
                    Circle()
                        .fill(Color.red.opacity(0.3))
                        .frame(width: 30, height: 30)
                    RoundedRectangle(cornerRadius: 3, style: .continuous)
                        .fill(Color.primary)
                        .frame(width: 10, height: 10)
                }
            } else {
                Image(systemName: "arrow.up.circle.fill")
                    .font(.system(size: 27))
                    .foregroundStyle(canSend ? Color.accentColor : Color.secondary.opacity(0.5))
                    .frame(width: 30, height: 30)
            }
        }
        .buttonStyle(.plain)
        .disabled(!canSend && !chatVM.isGenerating)
    }

    private var canSend: Bool {
        !chatVM.inputText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty &&
        !chatVM.isGenerating &&
        !modelManager.isLoadingModel &&
        modelManager.loadedModelId != nil
    }

    // MARK: - Actions

    private func sendMessage() {
        if modelManager.isLoadingModel {
            showToast(loading: true)
            return
        }

        if modelManager.loadedModelId == nil {
            showToast(loading: false)
            return
        }

        guard canSend else { return }

        Task {
            await chatVM.sendMessage()
        }

        isFocused = false
    }

    private func showToast(loading: Bool) {
        withAnimation(.easeOut(duration: 0.2)) {
            if loading {
                showLoadingToast = true
            } else {
                showNoModelToast = true
            }
        }

        Task {
            try? await Task.sleep(for: .seconds(2))
            withAnimation(.easeIn(duration: 0.3)) {
                showLoadingToast = false
                showNoModelToast = false
            }
        }
    }
}

// MARK: - Loading Toast View

private struct LoadingToastView: View {
    let message: String
    var icon: String? = nil

    var body: some View {
        HStack(spacing: 8) {
            if let icon = icon {
                Image(systemName: icon)
                    .font(.subheadline)
                    .foregroundStyle(.orange)
            } else {
                ProgressView()
                    .controlSize(.small)
            }

            Text(message)
                .font(.subheadline)
                .fontWeight(.medium)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial, in: Capsule())
        .shadow(color: .black.opacity(0.1), radius: 4, y: 2)
    }
}
