//
//  ModelPickerView.swift
//  NimboChat
//
//  Model selection and file picker integration
//

import SwiftUI
import UniformTypeIdentifiers

struct ModelPickerView: View {
    @Environment(ModelManagerViewModel.self) private var modelManager
    @Environment(\.dismiss) private var dismiss

    @State private var showingFilePicker = false

    var body: some View {
        NavigationStack {
            List {
                // Loaded model section
                if let loadedId = modelManager.loadedModelId,
                   let loadedModel = modelManager.localModels.first(where: { $0.id == loadedId }) {
                    Section("Loaded Model") {
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(loadedModel.displayName)
                                    .font(.headline)
                                Text(loadedModel.path.path)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }

                            Spacer()

                            Circle()
                                .fill(Color.green)
                                .frame(width: 10, height: 10)
                        }

                        Button(role: .destructive) {
                            Task {
                                await modelManager.unloadModel()
                            }
                        } label: {
                            Label("Unload Model", systemImage: "eject")
                        }
                    }
                }

                // Available models
                Section("Available Models") {
                    if modelManager.localModels.isEmpty {
                        VStack(spacing: 12) {
                            Image(systemName: "folder.badge.questionmark")
                                .font(.system(size: 30))
                                .foregroundStyle(.secondary)

                            Text("No models found")
                                .font(.headline)
                                .foregroundStyle(.secondary)

                            Text("Place CoreML model folders in the app's Documents/Models/ directory using the Files app, or tap \"Add Model\" to select from another location.")
                                .font(.caption)
                                .foregroundStyle(.tertiary)
                                .multilineTextAlignment(.center)
                        }
                        .frame(maxWidth: .infinity)
                        .padding(.vertical, 20)
                    } else {
                        ForEach(modelManager.localModels) { model in
                            Button {
                                Task {
                                    await modelManager.loadModel(model)
                                    dismiss()
                                }
                            } label: {
                                HStack {
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(model.displayName)
                                            .font(.body)
                                            .foregroundStyle(.primary)
                                        Text(model.path.path)
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                            .lineLimit(1)
                                    }

                                    Spacer()

                                    if model.id == modelManager.loadedModelId {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(.green)
                                    } else if model.id == modelManager.loadingModelId {
                                        ProgressView()
                                            .controlSize(.small)
                                    }
                                }
                            }
                            .disabled(modelManager.isLoadingModel)
                        }
                    }
                }

                // Add model section
                Section {
                    Button {
                        showingFilePicker = true
                    } label: {
                        Label("Add Model from Files", systemImage: "folder.badge.plus")
                    }

                    Button {
                        Task {
                            await modelManager.scanForModels()
                        }
                    } label: {
                        Label("Refresh Model List", systemImage: "arrow.clockwise")
                    }
                }

                // Error
                if let error = modelManager.errorMessage {
                    Section {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }
            }
            .listStyle(.insetGrouped)
            .navigationTitle("Models")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
            .fileImporter(
                isPresented: $showingFilePicker,
                allowedContentTypes: [.folder],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    Task {
                        await modelManager.addModelFromPicker(url: url)
                    }
                case .failure(let error):
                    logError("File picker failed: \(error)", category: .model)
                }
            }
        }
    }
}
