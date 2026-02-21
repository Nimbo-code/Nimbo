//
//  ContentView.swift
//  NimboChat
//
//  Root view with navigation (iPhone only)
//

import SwiftUI

struct ContentView: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(ModelManagerViewModel.self) private var modelManager

    @State private var showingModelSheet = false
    @State private var showingSettings = false
    @State private var showingConversationSheet = false
    @State private var hasCheckedInitialState = false

    var body: some View {
        NavigationStack {
            ZStack {
                ChatView()
                    .environment(chatVM)
                    .environment(modelManager)
            }
            .overlay(alignment: .top) {
                overlayControls
            }
        }
        .sheet(isPresented: $showingConversationSheet) {
            ConversationListSheet {
                showingConversationSheet = false
            }
            .environment(chatVM)
        }
        .sheet(isPresented: $showingModelSheet) {
            ModelPickerView()
                .environment(modelManager)
        }
        .sheet(isPresented: $showingSettings) {
            NavigationStack {
                SettingsView()
                    .environment(chatVM)
            }
        }
        .onChange(of: modelManager.requestModelSelection) { _, requested in
            guard requested else { return }
            showingModelSheet = true
            modelManager.requestModelSelection = false
        }
        .task {
            guard !hasCheckedInitialState else { return }
            hasCheckedInitialState = true

            let hasSelectedModelBefore = await StorageService.shared.selectedModelId != nil

            if hasSelectedModelBefore {
                return
            }

            try? await Task.sleep(for: .milliseconds(500))

            if modelManager.localModels.isEmpty && !modelManager.isLoadingModel {
                showingModelSheet = true
            }
        }
    }

    // MARK: - Floating Controls

    private var overlayControls: some View {
        HStack {
            Spacer()
            HStack(spacing: 10) {
                Button {
                    chatVM.newConversation()
                } label: {
                    Image(systemName: "plus")
                }

                Button {
                    showingConversationSheet = true
                } label: {
                    Image(systemName: "list.bullet")
                }

                Button {
                    showingModelSheet = true
                } label: {
                    ZStack(alignment: .bottomTrailing) {
                        Image(systemName: "cpu")
                        Circle()
                            .fill(modelStatusColor)
                            .frame(width: 6, height: 6)
                            .offset(x: 3, y: 3)
                    }
                }

                Button {
                    showingSettings = true
                } label: {
                    Image(systemName: "gearshape")
                }
            }
            .font(.system(size: 14, weight: .semibold))
            .foregroundStyle(.primary)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(
                Capsule()
                    .stroke(Color.white.opacity(0.12), lineWidth: 1)
            )
            .shadow(color: .black.opacity(0.25), radius: 12, y: 6)
        }
        .padding(.top, 8)
        .padding(.horizontal, 12)
    }

    private var modelStatusColor: Color {
        if modelManager.isLoadingModel {
            return .blue
        }
        if modelManager.loadedModelId != nil {
            return .green
        }
        return .orange
    }
}

// MARK: - Conversation List Sheet

private struct ConversationListSheet: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(\.dismiss) private var dismiss

    @State private var showingClearAlert = false

    let onClose: () -> Void

    var body: some View {
        NavigationStack {
            List {
                Section {
                    ForEach(chatVM.conversations) { conversation in
                        Button {
                            chatVM.selectConversation(conversation)
                            dismiss()
                            onClose()
                        } label: {
                            ConversationRow(conversation: conversation)
                        }
                        .buttonStyle(.plain)
                        .contextMenu {
                            Button(role: .destructive) {
                                chatVM.deleteConversation(conversation)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                    }
                    .onDelete { indexSet in
                        chatVM.deleteConversation(at: indexSet)
                    }
                } header: {
                    Text("Conversations")
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Chats")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                        onClose()
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    HStack(spacing: 16) {
                        Button(role: .destructive) {
                            showingClearAlert = true
                        } label: {
                            Image(systemName: "trash")
                        }
                        .disabled(chatVM.conversations.isEmpty)

                        Button {
                            chatVM.newConversation()
                            dismiss()
                            onClose()
                        } label: {
                            Image(systemName: "plus")
                        }
                    }
                }
            }
            .alert("Clear All Chats?", isPresented: $showingClearAlert) {
                Button("Cancel", role: .cancel) {}
                Button("Clear All", role: .destructive) {
                    chatVM.clearAllConversations()
                }
            } message: {
                Text("This will delete all conversations. This action cannot be undone.")
            }
        }
    }
}

// MARK: - Conversation Row

struct ConversationRow: View {
    let conversation: Conversation

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(conversation.title)
                .font(.headline)
                .lineLimit(1)

            HStack {
                if let preview = conversation.lastMessagePreview {
                    Text(preview)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                Spacer()

                Text(conversation.formattedDate)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
        }
        .padding(.vertical, 4)
    }
}
