//
//  Logger.swift
//  NimboChat
//
//  Centralized logging system
//

import Foundation
import os.log

/// Log levels for categorizing messages
enum LogLevel: Int, Comparable {
    case debug = 0
    case info = 1
    case warning = 2
    case error = 3

    static func < (lhs: LogLevel, rhs: LogLevel) -> Bool {
        lhs.rawValue < rhs.rawValue
    }

    var emoji: String {
        switch self {
        case .debug: return "[DEBUG]"
        case .info: return "[INFO]"
        case .warning: return "[WARN]"
        case .error: return "[ERROR]"
        }
    }
}

/// Log categories for filtering
enum LogCategory: String, CaseIterable {
    case app = "App"
    case inference = "Inference"
    case storage = "Storage"
    case ui = "UI"
    case model = "Model"
}

/// Centralized logger with categories and levels
final class AppLogger: @unchecked Sendable {
    static let shared = AppLogger()

    private let osLog = OSLog(subsystem: "com.nimbo.chat", category: "general")
    private var minimumLevel: LogLevel = .debug
    private var enabledCategories: Set<LogCategory> = Set(LogCategory.allCases)

    private let queue = DispatchQueue(label: "com.nimbo.chat.logger")

    private(set) var recentLogs: [LogEntry] = []
    private let maxLogEntries = 500

    struct LogEntry: Identifiable, Sendable {
        let id = UUID()
        let timestamp: Date
        let level: LogLevel
        let category: LogCategory
        let message: String
        let file: String
        let line: Int

        var formattedTimestamp: String {
            let formatter = DateFormatter()
            formatter.dateFormat = "HH:mm:ss.SSS"
            return formatter.string(from: timestamp)
        }

        var formattedMessage: String {
            "[\(formattedTimestamp)] \(level.emoji) [\(category.rawValue)] \(message)"
        }
    }

    private init() {}

    func setMinimumLevel(_ level: LogLevel) {
        queue.sync { minimumLevel = level }
    }

    func setCategory(_ category: LogCategory, enabled: Bool) {
        queue.sync {
            if enabled {
                enabledCategories.insert(category)
            } else {
                enabledCategories.remove(category)
            }
        }
    }

    func log(
        _ message: String,
        level: LogLevel = .info,
        category: LogCategory = .app,
        file: String = #file,
        line: Int = #line
    ) {
        queue.async { [weak self] in
            guard let self = self else { return }

            guard level >= self.minimumLevel,
                  self.enabledCategories.contains(category) else { return }

            let entry = LogEntry(
                timestamp: Date(),
                level: level,
                category: category,
                message: message,
                file: (file as NSString).lastPathComponent,
                line: line
            )

            self.recentLogs.append(entry)
            if self.recentLogs.count > self.maxLogEntries {
                self.recentLogs.removeFirst()
            }

            let osLogType: OSLogType = {
                switch level {
                case .debug: return .debug
                case .info: return .info
                case .warning: return .default
                case .error: return .error
                }
            }()

            os_log("%{public}@", log: self.osLog, type: osLogType, entry.formattedMessage)

            #if DEBUG
            print(entry.formattedMessage)
            #endif
        }
    }

    func clearLogs() {
        queue.sync { recentLogs.removeAll() }
    }

    func exportLogs() -> String {
        queue.sync {
            recentLogs.map { $0.formattedMessage }.joined(separator: "\n")
        }
    }
}

// MARK: - Global Logging Functions

func logDebug(_ message: String, category: LogCategory = .app, file: String = #file, line: Int = #line) {
    AppLogger.shared.log(message, level: .debug, category: category, file: file, line: line)
}

func logInfo(_ message: String, category: LogCategory = .app, file: String = #file, line: Int = #line) {
    AppLogger.shared.log(message, level: .info, category: category, file: file, line: line)
}

func logWarning(_ message: String, category: LogCategory = .app, file: String = #file, line: Int = #line) {
    AppLogger.shared.log(message, level: .warning, category: category, file: file, line: line)
}

func logError(_ message: String, category: LogCategory = .app, file: String = #file, line: Int = #line) {
    AppLogger.shared.log(message, level: .error, category: category, file: file, line: line)
}
