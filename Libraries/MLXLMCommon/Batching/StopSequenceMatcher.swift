// Copyright © 2024 Apple Inc.

import Foundation

/// A trie node for matching a multi-token stop sequence.
public struct StopSequenceTrieNode: Sendable {
    public var children: [Int: StopSequenceTrieNode] = [:]

    /// Set on terminal nodes. Reaching this node completes the sequence
    /// and transitions to `transition.next`, or terminates if that's nil.
    public var transition: Transition?

    public struct Transition: Sendable {
        public let matchedSequence: [Int]
        public let next: String?
    }

    public init() {}
}

/// Snapshot of a single row's match progress.
public struct StopSequenceMatcherState: Sendable {
    public let currentState: String?
    public let trieNode: StopSequenceTrieNode?
    public let allStates: [String: StopSequenceTrieNode]
    public let pendingMatch: [Int]

    public init(
        currentState: String?,
        trieNode: StopSequenceTrieNode?,
        allStates: [String: StopSequenceTrieNode],
        pendingMatch: [Int] = []
    ) {
        self.currentState = currentState
        self.trieNode = trieNode
        self.allStates = allStates
        self.pendingMatch = pendingMatch
    }
}

/// Per-row stop-sequence detector. Each named state holds a list of
/// `(sequence, nextState)` transitions; matching `sequence` from `state`
/// transitions to `nextState`, or terminates the row if `nextState` is nil.
///
/// A typical configuration is `["normal": [(eosTokens, nil)]]` -- one
/// terminal match on the model's EOS tokens.
public struct StopSequenceMatcher: Sendable {

    public let states: [String: StopSequenceTrieNode]
    public let initial: String

    public init(
        states: [String: [(sequence: [Int], next: String?)]],
        initial: String = "normal"
    ) {
        var compiled: [String: StopSequenceTrieNode] = [:]
        for (name, transitions) in states {
            var root = StopSequenceTrieNode()
            for (sequence, next) in transitions {
                let transition = StopSequenceTrieNode.Transition(
                    matchedSequence: sequence,
                    next: next
                )
                Self.insert(
                    into: &root,
                    sequence: sequence,
                    index: 0,
                    transition: transition
                )
            }
            compiled[name] = root
        }
        self.states = compiled
        self.initial = initial
    }

    private static func insert(
        into node: inout StopSequenceTrieNode,
        sequence: [Int],
        index: Int,
        transition: StopSequenceTrieNode.Transition
    ) {
        if index == sequence.count {
            node.transition = transition
            return
        }
        let token = sequence[index]
        var child = node.children[token] ?? StopSequenceTrieNode()
        insert(into: &child, sequence: sequence, index: index + 1, transition: transition)
        node.children[token] = child
    }

    /// An empty matcher that never matches. Rows finish only on `max_tokens`.
    public init() {
        self.states = [:]
        self.initial = "normal"
    }

    public func makeState() -> StopSequenceMatcherState {
        StopSequenceMatcherState(
            currentState: states.isEmpty ? nil : initial,
            trieNode: states[initial],
            allStates: states,
            pendingMatch: []
        )
    }

    /// Advance the state by one token. Returns the new state, the matched
    /// sequence if a terminal node was reached on this token, and the state
    /// name after the transition (nil indicates the row terminated).
    public func match(
        _ state: StopSequenceMatcherState, _ token: Int
    ) -> (
        next: StopSequenceMatcherState,
        matchedSequence: [Int]?,
        currentState: String?
    ) {
        guard state.trieNode != nil, let currentState = state.currentState,
            let root = state.allStates[currentState]
        else {
            return (state, nil, state.currentState)
        }

        func fire(_ transition: StopSequenceTrieNode.Transition) -> (
            next: StopSequenceMatcherState,
            matchedSequence: [Int]?,
            currentState: String?
        ) {
            let nextState = transition.next
            let nextNode = nextState.flatMap { state.allStates[$0] }
            return (
                StopSequenceMatcherState(
                    currentState: nextState,
                    trieNode: nextNode,
                    allStates: state.allStates,
                    pendingMatch: []
                ),
                transition.matchedSequence,
                nextState
            )
        }

        var candidate = state.pendingMatch + [token]
        while !candidate.isEmpty {
            if let child = Self.findPrefix(candidate, in: root) {
                if let transition = child.transition {
                    return fire(transition)
                }
                // The full candidate is a viable (non-terminal) prefix, but a
                // SHORTER stop may have just completed at this token and would
                // otherwise be shadowed by the longer partial match forever
                // (e.g. stops [[1,2,3],[2]] fed 1,2: pending [1,2] hides the
                // completed [2]). Scan proper suffixes for a terminal hit
                // before committing to the longer pending match.
                var suffix = Array(candidate.dropFirst())
                while !suffix.isEmpty {
                    if let node = Self.findPrefix(suffix, in: root),
                        let transition = node.transition
                    {
                        return fire(transition)
                    }
                    suffix.removeFirst()
                }
                return (
                    StopSequenceMatcherState(
                        currentState: currentState,
                        trieNode: child,
                        allStates: state.allStates,
                        pendingMatch: candidate
                    ),
                    nil,
                    currentState
                )
            }

            candidate.removeFirst()
        }

        return (
            StopSequenceMatcherState(
                currentState: currentState,
                trieNode: root,
                allStates: state.allStates,
                pendingMatch: []
            ),
            nil,
            currentState
        )
    }

    private static func findPrefix(
        _ tokens: [Int],
        in root: StopSequenceTrieNode
    ) -> StopSequenceTrieNode? {
        var node = root
        for token in tokens {
            guard let child = node.children[token] else {
                return nil
            }
            node = child
        }
        return node
    }
}
