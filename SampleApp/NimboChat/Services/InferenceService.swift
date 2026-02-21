//
//  InferenceService.swift
//  NimboChat
//
//  Wrapper around NimboCore for model inference
//

import Foundation
import CoreML
@preconcurrency import NimboCore

/// Errors during inference
enum InferenceError: LocalizedError {
    case modelNotLoaded
    case configNotFound
    case tokenizerFailed(Error)
    case modelLoadFailed(Error)
    case generationFailed(Error)
    case cancelled

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded: return "No model is loaded"
        case .configNotFound: return "Model configuration not found"
        case .tokenizerFailed(let e): return "Tokenizer error: \(e.localizedDescription)"
        case .modelLoadFailed(let e): return "Model load error: \(e.localizedDescription)"
        case .generationFailed(let e): return "Generation error: \(e.localizedDescription)"
        case .cancelled: return "Generation was cancelled"
        }
    }
}

/// Result of token generation
struct GenerationResult: Sendable {
    let text: String
    let tokensPerSecond: Double
    let tokenCount: Int
    let windowShifts: Int
    let prefillTime: TimeInterval
    let prefillTokens: Int
    let historyTokens: Int
    let isComplete: Bool
    let wasCancelled: Bool
    let stopReason: String
}

/// Loading progress information
struct ModelLoadingProgress: Sendable {
    let percentage: Double
    let stage: String
    let detail: String?
}

/// Thread-safe container for generation statistics
final class GenerationStats: @unchecked Sendable {
    var tokenCount: Int = 0
    var windowShifts: Int = 0
    var generatedText: String = ""
}

/// Repetition detector for detecting generation loops
final class RepetitionDetector: @unchecked Sendable {
    private var tokenHistory: [Int] = []
    private let windowSize: Int
    private let ngramSize: Int
    private let threshold: Int

    init(windowSize: Int = 50, ngramSize: Int = 5, threshold: Int = 3) {
        self.windowSize = windowSize
        self.ngramSize = ngramSize
        self.threshold = threshold
    }

    func addToken(_ token: Int) {
        tokenHistory.append(token)
        if tokenHistory.count > windowSize {
            tokenHistory.removeFirst()
        }
    }

    func isRepeating() -> Bool {
        guard tokenHistory.count >= ngramSize * threshold else { return false }

        var ngramCounts: [String: Int] = [:]

        for i in 0...(tokenHistory.count - ngramSize) {
            let ngram = tokenHistory[i..<(i + ngramSize)].map(String.init).joined(separator: ",")
            ngramCounts[ngram, default: 0] += 1

            if ngramCounts[ngram]! >= threshold {
                return true
            }
        }

        return false
    }

    func reset() {
        tokenHistory.removeAll()
    }
}

/// Service for loading models and running inference
@MainActor
final class InferenceService: ObservableObject {
    static let shared = InferenceService()

    // State
    @Published private(set) var isModelLoaded = false
    @Published private(set) var isGenerating = false
    @Published private(set) var loadingProgress: ModelLoadingProgress?
    @Published private(set) var currentModelId: String?

    // Internal state
    private var config: YAMLConfig?
    private var tokenizer: Tokenizer?
    private var inferenceManager: InferenceManager?
    private var loadedModels: LoadedModels?

    // Generation control
    private var generationTask: Task<Void, Never>?
    private var shouldCancel = false
    private let repetitionDetector = RepetitionDetector()

    // Model template
    private var currentTemplate: String = "default"

    // Settings
    var temperature: Float = 0.0
    var maxTokens: Int = 512
    var systemPrompt: String = ""
    var debugLevel: Int = 0
    var repetitionDetectionEnabled: Bool = false

    // Sampling settings
    var doSample: Bool = false
    var topP: Float = 0.95
    var topK: Int = 0
    var useRecommendedSampling: Bool = true

    private(set) var modelRecommendedSampling: (doSample: Bool, temperature: Float, topP: Float, topK: Int)?
    private(set) var isArgmaxModel: Bool = false

    var modelMaxContextSize: Int {
        guard let config = config else { return 2048 }
        return config.stateLength > 0 ? config.stateLength : config.contextLength
    }

    private init() {
        Task {
            temperature = await StorageService.shared.defaultTemperature
            maxTokens = await StorageService.shared.defaultMaxTokens
            systemPrompt = await StorageService.shared.defaultSystemPrompt
            debugLevel = await StorageService.shared.debugLevel
            repetitionDetectionEnabled = await StorageService.shared.repetitionDetectionEnabled
            doSample = await StorageService.shared.doSample
            topP = await StorageService.shared.topP
            topK = await StorageService.shared.topK
            useRecommendedSampling = await StorageService.shared.useRecommendedSampling
        }
    }

    // MARK: - Model Loading

    func loadModel(from modelPath: URL) async throws {
        let metaYamlPath = modelPath.appendingPathComponent("meta.yaml").path

        guard FileManager.default.fileExists(atPath: metaYamlPath) else {
            throw InferenceError.configNotFound
        }

        logInfo("Loading model from: \(modelPath.path)", category: .model)

        await unloadModel()

        do {
            loadingProgress = ModelLoadingProgress(percentage: 0.05, stage: "Loading configuration", detail: nil)
            config = try YAMLConfig.load(from: metaYamlPath)

            guard let config = config else {
                throw InferenceError.configNotFound
            }

            loadingProgress = ModelLoadingProgress(percentage: 0.1, stage: "Loading tokenizer", detail: nil)
            let detectedTemplate = detectTemplate(from: config)
            currentTemplate = detectedTemplate
            tokenizer = try await Tokenizer(
                modelPath: modelPath.path,
                template: detectedTemplate,
                debugLevel: debugLevel
            )

            loadingProgress = ModelLoadingProgress(percentage: 0.2, stage: "Loading CoreML models", detail: nil)

            let progressDelegate = LoadingProgressDelegate { [weak self] percentage, stage, detail in
                Task { @MainActor in
                    self?.loadingProgress = ModelLoadingProgress(
                        percentage: 0.2 + percentage * 0.7,
                        stage: stage,
                        detail: detail
                    )
                }
            }

            let modelLoader = ModelLoader(progressDelegate: progressDelegate)
            loadedModels = try await modelLoader.loadModel(from: config)

            loadingProgress = ModelLoadingProgress(percentage: 0.95, stage: "Initializing inference engine", detail: nil)

            inferenceManager = try InferenceManager(
                models: loadedModels!,
                contextLength: config.contextLength,
                batchSize: config.batchSize,
                splitLMHead: config.splitLMHead,
                debugLevel: debugLevel,
                v110: config.configVersion == "0.1.1",
                argmaxInModel: config.argmaxInModel,
                slidingWindow: config.slidingWindow,
                updateMaskPrefill: config.updateMaskPrefill,
                prefillDynamicSlice: config.prefillDynamicSlice,
                modelPrefix: config.modelPrefix,
                vocabSize: config.vocabSize,
                lmHeadChunkSizes: config.lmHeadChunkSizes
            )

            currentModelId = modelPath.lastPathComponent
            isModelLoaded = true
            loadingProgress = ModelLoadingProgress(percentage: 1.0, stage: "Ready", detail: nil)

            applySamplingConfig(from: config)

            logInfo("Model loaded successfully", category: .model)

        } catch {
            loadingProgress = nil
            throw InferenceError.modelLoadFailed(error)
        }
    }

    func unloadModel() async {
        cancelGeneration()

        inferenceManager?.unload()
        inferenceManager = nil
        loadedModels = nil
        tokenizer = nil
        config = nil
        currentModelId = nil
        isModelLoaded = false
        loadingProgress = nil

        logInfo("Model unloaded", category: .model)
    }

    // MARK: - Generation

    func generateResponse(
        for messages: [ChatMessage],
        onToken: @escaping @Sendable (String) -> Void,
        onWindowShift: @escaping @Sendable () -> Void,
        onHistoryUpdate: (@Sendable (Int) -> Void)? = nil
    ) async throws -> GenerationResult {
        guard let tokenizer = tokenizer,
              let inferenceManager = inferenceManager else {
            throw InferenceError.modelNotLoaded
        }

        isGenerating = true
        shouldCancel = false
        repetitionDetector.reset()

        defer {
            isGenerating = false
        }

        let capturedTemplate = currentTemplate
        let capturedSystemPrompt = systemPrompt
        let capturedConfig = config
        let capturedTokenizerForPrep = tokenizer

        struct PreparedInput {
            let chatMessages: [NimboCore.Tokenizer.ChatMessage]
            let inputTokens: [Int]
            let contextLength: Int
            let maxContextSize: Int
        }

        let prepared = await Task.detached(priority: .userInitiated) {
            () -> PreparedInput in
            var chatMessages: [NimboCore.Tokenizer.ChatMessage] = []

            if !messages.contains(where: { $0.role == .system }) {
                let resolvedPrompt = InferenceService.resolveSystemPrompt(
                    capturedSystemPrompt,
                    template: capturedTemplate
                )
                if !resolvedPrompt.isEmpty {
                    chatMessages.append(.system(resolvedPrompt))
                }
            }

            for message in messages {
                switch message.role {
                case .system:
                    chatMessages.append(.system(message.content))
                case .user:
                    chatMessages.append(.user(message.content))
                case .assistant:
                    chatMessages.append(.assistant(message.content))
                }
            }

            let contextLength = capturedConfig?.contextLength ?? 512
            let stateLength = (capturedConfig?.stateLength ?? 0) > 0 ? (capturedConfig?.stateLength ?? contextLength) : contextLength
            let maxContextSize = stateLength - 100

            var inputTokens = capturedTokenizerForPrep.applyChatTemplate(
                input: chatMessages,
                addGenerationPrompt: true
            )

            while inputTokens.count > maxContextSize && chatMessages.count > 2 {
                let startIndex = (chatMessages.first?.role == "system") ? 1 : 0
                if chatMessages.count > startIndex + 2 {
                    chatMessages.remove(at: startIndex)
                    if chatMessages.count > startIndex {
                        chatMessages.remove(at: startIndex)
                    }
                    inputTokens = capturedTokenizerForPrep.applyChatTemplate(
                        input: chatMessages,
                        addGenerationPrompt: true
                    )
                } else {
                    break
                }
            }

            return PreparedInput(
                chatMessages: chatMessages,
                inputTokens: inputTokens,
                contextLength: contextLength,
                maxContextSize: maxContextSize
            )
        }.value

        let inputTokens = prepared.inputTokens

        logDebug("Input tokens: \(inputTokens.count)", category: .inference)

        let stats = GenerationStats()
        let startTime = Date()

        let capturedRepetitionDetector = repetitionDetector
        let capturedInferenceManager = inferenceManager
        let capturedRepetitionEnabled = repetitionDetectionEnabled
        let capturedTemperature = temperature
        let capturedMaxTokens = maxTokens
        let capturedTokenizer = tokenizer
        let inputTokenCount = inputTokens.count

        do {
            let (prefillTime, stopReason) = try await Task.detached(priority: .userInitiated) {
                () async throws -> (TimeInterval, String) in
                var pendingText = ""
                var lastEmitTime = CFAbsoluteTimeGetCurrent()
                var lastHistoryEmitTime = lastEmitTime
                let emitInterval: CFAbsoluteTime = 1.0 / 30.0
                let minChunkChars = 48
                var allTokenIds: [Int] = []
                var prevDecodedText: String = ""

                func emitIfNeeded(force: Bool = false) {
                    let now = CFAbsoluteTimeGetCurrent()
                    let shouldEmit = force || now - lastEmitTime >= emitInterval || pendingText.count >= minChunkChars

                    if shouldEmit, !pendingText.isEmpty {
                        let chunk = pendingText
                        pendingText = ""
                        lastEmitTime = now
                        onToken(chunk)
                    }

                    if let onHistoryUpdate, force || now - lastHistoryEmitTime >= emitInterval {
                        lastHistoryEmitTime = now
                        onHistoryUpdate(inputTokenCount + stats.tokenCount)
                    }
                }

                let (_, prefillTime, stopReason) = try await capturedInferenceManager.generateResponse(
                    initialTokens: inputTokens,
                    temperature: capturedTemperature,
                    maxTokens: capturedMaxTokens,
                    eosTokens: capturedTokenizer.eosTokenIds,
                    tokenizer: capturedTokenizer,
                    onToken: { token in
                        if capturedRepetitionEnabled {
                            capturedRepetitionDetector.addToken(token)
                            if capturedRepetitionDetector.isRepeating() {
                                logWarning("Repetition detected, aborting", category: .inference)
                                capturedInferenceManager.AbortGeneration(Code: 2)
                                return
                            }
                        }

                        stats.tokenCount += 1
                        allTokenIds.append(token)
                        let fullText = capturedTokenizer.decode(tokens: allTokenIds)
                        let text: String
                        if fullText.count > prevDecodedText.count {
                            text = String(fullText[fullText.index(fullText.startIndex, offsetBy: prevDecodedText.count)...])
                        } else {
                            text = capturedTokenizer.decode(tokens: [token])
                        }
                        prevDecodedText = fullText
                        stats.generatedText += text

                        pendingText += text
                        emitIfNeeded()
                    },
                    onWindowShift: {
                        stats.windowShifts += 1
                        onWindowShift()
                    }
                )
                emitIfNeeded(force: true)
                return (prefillTime, stopReason)
            }.value

            let totalTime = Date().timeIntervalSince(startTime)
            let tokensPerSecond = totalTime > 0 ? Double(stats.tokenCount) / totalTime : 0

            logInfo("Generation complete: \(stats.tokenCount) tokens, \(String(format: "%.1f", tokensPerSecond)) tok/s", category: .inference)

            let historyTokens = inputTokens.count + stats.tokenCount
            let wasCancelled = shouldCancel || stopReason.hasPrefix("abort_generation1")

            return GenerationResult(
                text: stats.generatedText,
                tokensPerSecond: tokensPerSecond,
                tokenCount: stats.tokenCount,
                windowShifts: stats.windowShifts,
                prefillTime: prefillTime,
                prefillTokens: inputTokens.count,
                historyTokens: historyTokens,
                isComplete: true,
                wasCancelled: wasCancelled,
                stopReason: stopReason
            )

        } catch {
            if shouldCancel {
                let historyTokens = inputTokens.count + stats.tokenCount
                return GenerationResult(
                    text: stats.generatedText,
                    tokensPerSecond: 0,
                    tokenCount: stats.tokenCount,
                    windowShifts: stats.windowShifts,
                    prefillTime: 0,
                    prefillTokens: inputTokens.count,
                    historyTokens: historyTokens,
                    isComplete: false,
                    wasCancelled: true,
                    stopReason: "cancelled"
                )
            }
            throw InferenceError.generationFailed(error)
        }
    }

    func cancelGeneration() {
        shouldCancel = true
        inferenceManager?.AbortGeneration(Code: 1)
        generationTask?.cancel()
        generationTask = nil
    }

    // MARK: - Sampling Configuration

    private func applySamplingConfig(from config: YAMLConfig) {
        isArgmaxModel = config.argmaxInModel
        if isArgmaxModel {
            logWarning("Model uses argmax output - sampling is unavailable, using greedy decoding", category: .model)
            modelRecommendedSampling = nil
            return
        }

        if let rec = config.recommendedSampling {
            modelRecommendedSampling = (
                doSample: rec.doSample,
                temperature: Float(rec.temperature),
                topP: Float(rec.topP),
                topK: rec.topK
            )

            if useRecommendedSampling {
                doSample = rec.doSample
                temperature = Float(rec.temperature)
                topP = Float(rec.topP)
                topK = rec.topK
            }
        } else {
            modelRecommendedSampling = nil
        }
    }

    var effectiveSamplingDescription: String {
        if isArgmaxModel {
            return "Greedy (argmax model)"
        }
        if useRecommendedSampling, let rec = modelRecommendedSampling {
            return "Model recommended: \(String(format: "%.2f", rec.temperature)) / \(String(format: "%.2f", rec.topP)) / \(rec.topK)"
        }
        if doSample && temperature > 0 {
            return "Custom: \(String(format: "%.2f", temperature)) / \(String(format: "%.2f", topP)) / \(topK)"
        }
        return "Greedy"
    }

    // MARK: - Helpers

    nonisolated private static func resolveSystemPrompt(_ prompt: String, template: String) -> String {
        let normalizedPrompt = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if normalizedPrompt.isEmpty {
            return ""
        }
        return normalizedPrompt
    }

    private func detectTemplate(from config: YAMLConfig) -> String {
        let path = config.modelPath.lowercased()

        if path.contains("gemma") {
            return path.contains("gemma3") ? "gemma3" : "gemma"
        } else if path.contains("qwen") {
            return "qwen"
        } else if path.contains("deepseek") {
            return "deepseek"
        } else if path.contains("deephermes") {
            return "deephermes"
        } else if path.contains("llama") {
            return "llama3"
        }

        return "default"
    }
}

// MARK: - Loading Progress Delegate

private final class LoadingProgressDelegate: ModelLoadingProgressDelegate, @unchecked Sendable {
    private let onProgress: (Double, String, String?) -> Void

    init(onProgress: @escaping (Double, String, String?) -> Void) {
        self.onProgress = onProgress
    }

    func loadingProgress(percentage: Double, stage: String, detail: String?) {
        onProgress(percentage, stage, detail)
    }

    func loadingCompleted(models: LoadedModels) {
        onProgress(1.0, "Complete", nil)
    }

    func loadingCancelled() {
        onProgress(0, "Cancelled", nil)
    }

    func loadingFailed(error: Error) {
        onProgress(0, "Failed", error.localizedDescription)
    }
}
