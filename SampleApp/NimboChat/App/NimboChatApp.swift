//
//  NimboChatApp.swift
//  NimboChat
//
//  On-device LLM chat powered by CoreML
//

import SwiftUI

@main
struct NimboChatApp: App {
    @State private var chatViewModel = ChatViewModel()
    @State private var modelManager = ModelManagerViewModel()

    init() {
        logInfo("=== Nimbo Chat Starting ===", category: .app)
        logInfo("App version: \(Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?") (\(Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "?"))", category: .app)
    }

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(chatViewModel)
                .environment(modelManager)
                .preferredColorScheme(.light)
        }
    }
}
