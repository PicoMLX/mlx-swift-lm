// Copyright © 2026 Apple Inc.

import Darwin
import Foundation
import XCTest

/// A sharded Qwen 3.5 decodes through the traced layer functions and the compiled decode step
/// as through the general path. One test process can only form a group of one, which never
/// splits a weight, so this launches two ranks of `ShardedDecodeParity` over MLX's ring backend
/// on this Mac and checks what each reports.
final class Qwen35ShardedDecodeTests: XCTestCase {

    /// A port nothing listens on right now.
    private func freePort() throws -> Int {
        let descriptor = socket(AF_INET, SOCK_STREAM, 0)
        guard descriptor >= 0 else { throw POSIXError(.EBADF) }
        defer { Darwin.close(descriptor) }
        var address = sockaddr_in()
        address.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        address.sin_family = sa_family_t(AF_INET)
        var length = socklen_t(MemoryLayout<sockaddr_in>.size)
        let bound = withUnsafeMutablePointer(to: &address) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                Darwin.bind(descriptor, $0, length) == 0
                    && getsockname(descriptor, $0, &length) == 0
            }
        }
        guard bound else { throw POSIXError(.EADDRINUSE) }
        return Int(UInt16(bigEndian: address.sin_port))
    }

    func testTracedAndCompiledPathsMatchTheGeneralShardedPathOnTwoRanks() throws {
        let runner = Bundle(for: Self.self).bundleURL.deletingLastPathComponent()
            .appendingPathComponent("ShardedDecodeParity")
        XCTAssertTrue(
            FileManager.default.isExecutableFile(atPath: runner.path),
            "ShardedDecodeParity isn't built next to the test bundle")
        let hostfile = FileManager.default.temporaryDirectory
            .appendingPathComponent("sharded-decode-\(UUID().uuidString).json")
        let hosts = try [freePort(), freePort()].map { ["127.0.0.1:\($0)"] }
        try JSONSerialization.data(withJSONObject: hosts).write(to: hostfile)
        defer { try? FileManager.default.removeItem(at: hostfile) }

        let ranks = try (0 ..< 2).map { rank -> (process: Process, output: Pipe) in
            let process = Process()
            process.executableURL = runner
            var environment = ProcessInfo.processInfo.environment
            environment["MLX_RANK"] = String(rank)
            environment["MLX_HOSTFILE"] = hostfile.path
            process.environment = environment
            let output = Pipe()
            process.standardOutput = output
            try process.run()
            return (process, output)
        }
        // A rank whose partner never connects waits for good; don't let that hang the suite.
        let deadline = Date().addingTimeInterval(120)
        for rank in ranks {
            while rank.process.isRunning, Date() < deadline { Thread.sleep(forTimeInterval: 0.1) }
            if rank.process.isRunning { rank.process.terminate() }
            rank.process.waitUntilExit()
        }

        var reports: [[String: Any]] = []
        for rank in ranks {
            XCTAssertEqual(rank.process.terminationStatus, 0)
            let data = rank.output.fileHandleForReading.readDataToEndOfFile()
            reports.append(
                try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any]))
        }
        for report in reports {
            XCTAssertEqual(report["size"] as? Int, 2)
            for weights in ["float", "4-bit"] {
                let result = try XCTUnwrap(report[weights] as? [String: Any])
                let label = "rank \(report["rank"] ?? "?"), \(weights)"
                // The MLP has 128 rows whole; each rank holds half of them.
                XCTAssertEqual(result["gateRows"] as? Int, 64, label)
                XCTAssertEqual(
                    try XCTUnwrap(result["tracedLayers"] as? Double), 0, accuracy: 1e-4, label)
                XCTAssertEqual(
                    try XCTUnwrap(result["compiledStep"] as? Double), 0, accuracy: 1e-4, label)
                XCTAssertEqual(
                    try XCTUnwrap(result["acrossRanks"] as? Double), 0, accuracy: 1e-4, label)
            }
        }
        for weights in ["float", "4-bit"] {
            let tokens = reports.map { ($0[weights] as? [String: Any])?["tokens"] as? [Int] }
            XCTAssertEqual(tokens[0], tokens[1], weights)
        }
    }
}
