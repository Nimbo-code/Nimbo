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
            .safeAreaInset(edge: .top) {
                topNavigationBar
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

    // MARK: - Top Navigation Bar

    private var topNavigationBar: some View {
        HStack {
            // Left: hamburger menu → conversation list
            Button {
                showingConversationSheet = true
            } label: {
                Image(systemName: "line.3.horizontal")
                    .font(.system(size: 20, weight: .medium))
                    .foregroundStyle(.primary)
                    .frame(width: 36, height: 36)
            }

            Spacer()

            // Center: title + model name
            VStack(spacing: 2) {
                Text(chatVM.currentConversation?.title ?? "New Chat")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                if let modelName = modelManager.loadedModelName {
                    Text(modelName)
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                } else if modelManager.isLoadingModel {
                    Text("Loading model...")
                        .font(.system(size: 11))
                        .foregroundStyle(.secondary)
                } else {
                    Button {
                        showingModelSheet = true
                    } label: {
                        Text("No model selected")
                            .font(.system(size: 11))
                            .foregroundStyle(Color(red: 1.0, green: 0.42, blue: 0.21))
                    }
                }
            }

            Spacer()

            // Right: settings gear
            Button {
                showingSettings = true
            } label: {
                Image(systemName: "gearshape")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.primary)
                    .frame(width: 36, height: 36)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(Color(.systemBackground))
        .overlay(alignment: .bottom) {
            Divider()
        }
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
