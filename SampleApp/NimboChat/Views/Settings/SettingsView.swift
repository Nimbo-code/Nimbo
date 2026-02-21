//
//  SettingsView.swift
//  NimboChat
//
//  App settings and configuration (iOS only)
//

import SwiftUI

struct SettingsView: View {
    @Environment(ChatViewModel.self) private var chatVM
    @Environment(\.dismiss) private var dismiss

    @State private var temperature: Float = 0.0
    @State private var maxTokens: Int = 2048
    @State private var systemPrompt: String = ""

    // Sampling settings
    @State private var doSample: Bool = false
    @State private var topP: Float = 0.95
    @State private var topK: Int = 0
    @State private var useRecommendedSampling: Bool = true

    @State private var showingLogs = false
    @State private var autoLoadLastModel = true
    @State private var debugLevel: Int = 0
    @State private var repetitionDetectionEnabled = false
    @State private var loadLastChat = false
    @State private var showingResetConfirmation = false

    var body: some View {
        Form {
            // Model settings
            modelSection

            // Generation settings
            generationSection

            // System prompt
            systemPromptSection

            // Logs
            logsSection

            // About
            aboutSection
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .confirmationAction) {
                Button("Done") {
                    saveSettings()
                    dismiss()
                }
            }
        }
        .onAppear {
            loadSettings()
        }
        .onDisappear {
            saveSettings()
        }
        .sheet(isPresented: $showingLogs) {
            LogsView()
        }
    }

    // MARK: - Model Section

    private var modelSection: some View {
        Section("Model") {
            Toggle("Auto-load last model", isOn: $autoLoadLastModel)
            Toggle("Load last chat on startup", isOn: $loadLastChat)

            Button(role: .destructive) {
                Task {
                    await StorageService.shared.clearLastModel()
                }
            } label: {
                Label("Clear remembered model", systemImage: "xmark.circle")
            }
        }
    }

    // MARK: - Generation Section

    private var generationSection: some View {
        Section("Generation") {
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Temperature")
                    Spacer()
                    Text(String(format: "%.2f", temperature))
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Slider(value: $temperature, in: 0...2, step: 0.05)
            }

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Max Tokens")
                    Spacer()
                    Text("\(maxTokens)")
                        .foregroundStyle(.secondary)
                        .monospacedDigit()
                }
                Slider(value: Binding(
                    get: { Double(maxTokens) },
                    set: { maxTokens = Int($0) }
                ), in: 64...4096, step: 64)
            }

            // Sampling
            Toggle("Enable Sampling", isOn: $doSample)

            if doSample {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Top-P")
                        Spacer()
                        Text(String(format: "%.2f", topP))
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: $topP, in: 0...1, step: 0.05)
                }

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Top-K")
                        Spacer()
                        Text("\(topK)")
                            .foregroundStyle(.secondary)
                            .monospacedDigit()
                    }
                    Slider(value: Binding(
                        get: { Double(topK) },
                        set: { topK = Int($0) }
                    ), in: 0...100, step: 1)
                }

                Toggle("Use recommended sampling", isOn: $useRecommendedSampling)
            }

            Toggle("Repetition detection", isOn: $repetitionDetectionEnabled)
        }
    }

    // MARK: - System Prompt Section

    private var systemPromptSection: some View {
        Section("System Prompt") {
            TextEditor(text: $systemPrompt)
                .frame(minHeight: 80)
                .font(.body)
        }
    }

    // MARK: - Logs Section

    private var logsSection: some View {
        Section("Debug") {
            Picker("Debug Level", selection: $debugLevel) {
                Text("Off").tag(0)
                Text("Basic").tag(1)
                Text("Verbose").tag(2)
            }

            Button {
                showingLogs = true
            } label: {
                Label("View Logs", systemImage: "doc.text")
            }
        }
    }

    // MARK: - About Section

    private var aboutSection: some View {
        Section("About") {
            HStack {
                Text("Version")
                Spacer()
                Text("\(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0")")
                    .foregroundStyle(.secondary)
            }

            Button(role: .destructive) {
                showingResetConfirmation = true
            } label: {
                Label("Reset to Defaults", systemImage: "arrow.counterclockwise")
            }
            .alert("Reset Settings?", isPresented: $showingResetConfirmation) {
                Button("Cancel", role: .cancel) {}
                Button("Reset", role: .destructive) {
                    Task {
                        await StorageService.shared.resetToDefaults()
                        loadSettings()
                    }
                }
            } message: {
                Text("This will reset all settings to their default values.")
            }
        }
    }

    // MARK: - Settings Load/Save

    private func loadSettings() {
        Task {
            temperature = await StorageService.shared.defaultTemperature
            maxTokens = await StorageService.shared.defaultMaxTokens
            systemPrompt = await StorageService.shared.defaultSystemPrompt
            autoLoadLastModel = await StorageService.shared.autoLoadLastModel
            debugLevel = await StorageService.shared.debugLevel
            repetitionDetectionEnabled = await StorageService.shared.repetitionDetectionEnabled
            loadLastChat = await StorageService.shared.loadLastChat
            doSample = await StorageService.shared.doSample
            topP = await StorageService.shared.topP
            topK = await StorageService.shared.topK
            useRecommendedSampling = await StorageService.shared.useRecommendedSampling
        }
    }

    private func saveSettings() {
        chatVM.temperature = temperature
        chatVM.maxTokens = maxTokens
        chatVM.systemPrompt = systemPrompt

        Task {
            await StorageService.shared.saveTemperature(temperature)
            await StorageService.shared.saveMaxTokens(maxTokens)
            await StorageService.shared.saveSystemPrompt(systemPrompt)
            await StorageService.shared.saveAutoLoadLastModel(autoLoadLastModel)
            await StorageService.shared.saveDebugLevel(debugLevel)
            await StorageService.shared.saveRepetitionDetectionEnabled(repetitionDetectionEnabled)
            await StorageService.shared.saveLoadLastChat(loadLastChat)
            await StorageService.shared.saveDoSample(doSample)
            await StorageService.shared.saveTopP(topP)
            await StorageService.shared.saveTopK(topK)
            await StorageService.shared.saveUseRecommendedSampling(useRecommendedSampling)
        }
    }
}

// MARK: - Logs View

struct LogsView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var logs: String = ""

    var body: some View {
        NavigationStack {
            ScrollView {
                Text(logs)
                    .font(.system(.caption, design: .monospaced))
                    .padding()
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .navigationTitle("Logs")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Close") {
                        dismiss()
                    }
                }
                ToolbarItem(placement: .primaryAction) {
                    HStack {
                        Button {
                            UIPasteboard.general.string = logs
                        } label: {
                            Image(systemName: "doc.on.doc")
                        }
                        Button {
                            AppLogger.shared.clearLogs()
                            logs = "Logs cleared."
                        } label: {
                            Image(systemName: "trash")
                        }
                    }
                }
            }
            .onAppear {
                logs = AppLogger.shared.exportLogs()
                if logs.isEmpty {
                    logs = "No logs yet."
                }
            }
        }
    }
}
