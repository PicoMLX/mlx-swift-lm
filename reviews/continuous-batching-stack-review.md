# Continuous-Batching PR Stack — Readiness Review

**Date:** 2026-07-02 · **Scope:** PRs #18 (cb-1 caches), #19 (cb-2 engine), #20 (cb-3 prompt cache), #21 (cb-4 scheduler), #22 (cb-5 quantized), plus #17 (quantized-mask fix).
**Method:** 73-agent multi-pass review of the *union tree* (main + all 5 PRs merged locally), with every new finding adversarially verified by an independent agent. 50 findings confirmed (17 major), 8 refuted, 15 pre-existing review-thread items verified still open. Reviewed on Linux (no Metal) — code-level review, not a runtime gate.

---

## 1. Verdict

**The stack is not ready to merge as a whole, but it is much closer than the raw finding count suggests — and the architecture is sound.** The quality is strikingly uneven across the layers:

| PR | Layer | Verdict |
|---|---|---|
| #18 cb-1 caches | `BatchKVCache`, `BatchRotatingKVCache`, factory | **Nearly ready.** The hard math (left-padding, ragged prefill, rotation) is careful and has survived three review rounds; remaining confirmed findings here are minor API-hardening issues. Two open thread items remain (serialization registration; scalar-offset model gating). |
| #19 cb-2 engine | `BatchGenerationEngine`, `DecodeBatch`, `PrefillBatch` | **Close, with 3 real parity bugs.** Penalty-on-log-probs, adopted-row stop-matcher default, and adopted-row penalty seeding each make batched output diverge from single-stream output. |
| #20 cb-3 prompt cache | `LRUPromptCache` | **Functionally OK for in-memory use; not ready as shipped.** Lock-held GPU deep-copies, a topology-blind namespace (maxKVSize contamination), and ~430 LOC of premature disk persistence. |
| #21 cb-4 scheduler | `InferenceScheduler`, `EngineDriver`, `ChatSession`/`ModelContainer` | **Not ready.** CI is red (Swift 6 region-isolation error at `ChatSession.swift:660`), 25 unresolved review threads, and the deep review confirmed additional actor-reentrancy races (double engine creation, stale `.idle` clobber, cancel dead-zone) plus behavior-parity losses (stop strings, tool-call restart context loss, speculative decoding silently disabled). |
| #22 cb-5 quantized | `BatchQuantizedKVCache` | **Defer.** It is dead code in the integrated system — no reachable producer through the scheduler — and it missed a correctness fix (trim clamp) its siblings got. Keep only the `quantizedScaledDotProductAttention` mask fix (or take #17 instead). |
| #17 mask fix | `quantizedScaledDotProductAttention` | **Ready.** Merge it (or upstream's equivalent) independently; drop the duplicate from cb-5 when it lands. |

**Three process-level blockers apply to the whole stack** (§2): the stack no longer merges cleanly in order; the CI "green" on the cache PRs is illusory (the test filter names a suite that doesn't exist, so the batched-cache tests have never run on Metal); and the integrated system (union of all 5 PRs) has never been compiled anywhere.

---

## 2. Merge logistics & CI — fix these first

1. **Stack skew.** cb-1 (PR #18) gained 11 fix commits (`d8e78d0..84257ff` — trim clamps, SSM factory reject, mask logical-span fixes, …) after cb-2/cb-3/cb-5 were cut from it. All sibling branches and cb-4 still sit on the old base. Consequence: merging PR #18 then PR #19 conflicts in `KVCache.swift`; cb-3/cb-5 conflict on `swift.yml`. **The final state of the feature exists on no branch.** A local union merge succeeds with only trivial conflicts and (verified hunk-by-hunk) no silent mis-merges — but nothing has ever compiled it. → Rebase cb-2/3/4/5 onto cb-1's tip, or push the union tree as an integration branch and let CI build it.
2. **CI never ran the cache tests.** `.github/workflows/swift.yml` uses `-only-testing:MLXLMTests/BatchCacheTests` — but no suite of that name exists; the file declares `BatchKVCacheCoverageTests`, `BatchRotatingKVCacheCoverageTests`, `BatchedCacheFactoryTests`, `BatchedSSMCacheTests`, `BatchMaskingTests`. xcodebuild silently matches nothing, so the "parity gate" the PRs cite has never executed the cache math on Metal (cb-5's quantized suite names do match). Additionally, no branch's CI runs `BatchModelRegressionTests`, `BatchEngineTests`, `InferenceSchedulerTests`, or `ModelContainerBatchingTests`, and the union merge silently dropped cb-3's `PromptCacheTests` entry. → Fix the filter (or run the whole `MLXLMTests` target); these suites need no model downloads.
3. **PR #21 compile failure.** `ChatSession.swift:660:20: error: pattern that the region based isolation checker does not understand how to check` (Xcode 26.3, `-swift-version 6`). Root cause: a capture-heavy `Task {}` closure whose body reassigns an `inout` closure parameter across an `await` (`cache = try await Self.streamThroughScheduler(...)` inside `cache.update { }`), with non-Sendable state (`SendableBox`, `Cache`) woven through. Fix by restructuring, not by `@unchecked`: compute the new `Cache` into a local inside `streamThroughScheduler`'s caller, then assign after the await (or replace `update`'s inout-closure with an async `get`/`set` pair on the cache box).

---

## 3. Architecture

```
 ChatSession ──────────► ModelContainer.generate / generateBatched        (public entry)
                              │  (scheduler installed? non-VLM?)
                              ▼
 InferenceScheduler (actor) ── policy/state machine: idle→single→upgrading→batched
      │  .single: TokenIterator task        │ .batched
      │  (UpgradeFlag handshake)            ▼
      └──────────────────────────► EngineDriver (actor) ── owns engine + prompt-cache write-back
                                        │
                                   BatchGenerationEngine (non-Sendable, single-executor)
                                        │  insert / next / cancel / adoptActiveBatch
                              PrefillBatch ──generate()──► DecodeBatch (step loop, RowSamplers,
                                        │                   LogitProcessors, StopSequenceMatcher)
                                        ▼
                            [any BatchedCache] per layer   (BatchKVCache / BatchRotatingKVCache /
                                        │                    BatchQuantizedKVCache / BatchedCacheList)
                                        ▼
                            model.callAsFunction(_:cache:) ── masks via cache.makeMask(),
                                                              RoPE via cache.ropeOffset (.batch)
 LRUPromptCache (Sendable, lock-based) ◄── write-back from EngineDriver + single path;
                                            fetch on single-path admission ONLY (batched fetch unwired)
```

**How the auto-upgrade works** (your requirement #3, and it is implemented essentially as you specified):

1. *First request:* `route` sees `.idle` → builds a plain `TokenIterator` and runs the **exact single-stream decode loop** in a detached task. Per-token overhead vs. no scheduler: one uncontended lock read (the upgrade flag) + one `Task.isCancelled` check. No batch caches, no batched masks.
2. *Second request arrives:* scheduler sets `.upgrading`, deposits a `CheckedContinuation`; at its next loop iteration the single task deposits a `LiveIteratorState` snapshot (caches, pending token, budget) via SE-0430 `sending` and exits without closing its stream. Caches are wrapped **by reference** with `BatchKVCache.fromSingle` (zero copy, no re-prefill), the row is adopted into a `DecodeBatch` with its original stream continuation, the new request prefills, and `EngineDriver.drain` loops the engine.
3. *Third+ requests* join through the engine queue (batched sub-prefill, then `extend` merge). Once batched, it stays batched until the batch drains, then returns to `.idle` — exactly the "no need to overcomplicate" behavior you asked for.
4. *Unsupported requests* (SSM models, `maxKVSize`, `kvBits`) are queued FIFO and run serially — graceful, no crash.

**State ownership is clean:** scheduler actor owns routing; `EngineDriver` actor owns the non-Sendable engine and all batched caches; non-Sendable values cross isolation exactly once via `sending`. There is **zero `@unchecked Sendable`** in the stack — genuinely excellent Swift 6 craftsmanship (with the reentrancy caveats in §7).

### The API, end to end

```swift
// 1. Load context, install scheduler + shared prompt cache (opt-in).
let context = try await LLMModelFactory.shared.load(
    configuration: ModelConfiguration(id: "mlx-community/Qwen2.5-7B-Instruct-4bit"))

let scheduler = InferenceScheduler(
    configuration: .init(completionBatchSize: 32, prefillBatchSize: 8,
                         prefillStepSize: 2048, maxBatchSize: 16))
let promptCache = LRUPromptCache(maxSize: 32, maxBytes: 2 << 30)

let container = ModelContainer(context: context,
                               scheduler: scheduler,      // nil = historical path, byte-for-byte
                               promptCache: promptCache)  // REQUIRED for multi-turn perf (§5, M-9)

// 2. Fire N concurrent chats. First runs single-stream; second triggers
//    auto-upgrade; the rest join the batch. Consumers can't tell the difference.
await withTaskGroup(of: Void.self) { group in
    for prompt in incomingPrompts {
        group.addTask {
            let session = ChatSession(container)
            for try await chunk in session.streamResponse(to: prompt) { await respond(chunk) }
        }
    }
}

// 2b. Or raw, with prepared inputs:
let stream = try await container.generate(input: lmInput,
    parameters: GenerateParameters(maxTokens: 512, temperature: 0.7))
for await event in stream { if let text = event.chunk { print(text, terminator: "") } }

// 2c. Or an explicit batch:
let streams = try await container.generateBatched(
    inputs.map { GenerationRequest(input: $0, parameters: .init(maxTokens: 256)) })

// Cancellation: drop the stream. Admission control: await container.setMaxBatchSize(8)
```

One discoverability gap: **no `loadModelContainer` factory can produce a scheduler-enabled container** — `scheduler` is a `private let` settable only through the low-level `ModelContext` initializer. The headline feature is invisible from the front door.

---

## 4. Scorecard against your original goals

| Goal | Status |
|---|---|
| Non-batched single-stream, zero overhead | ✅ **Met.** `scheduler: nil` is byte-for-byte the old path; with a scheduler + 1 request, per-token overhead is one lock read. Caveat: `TokenIterator` is built *on the scheduler actor*, so a long prefill blocks admission of the second request (§6-3). |
| Continuous batching of multiple chats | ✅ Implemented (engine + scheduler), ⚠️ with the parity bugs in §5. |
| Auto-upgrade on 2nd request; stay batched to drain | ✅ **Met, and well-designed** (zero-copy cache adoption, no re-prefill). ⚠️ Upgrade-window races in §5 (M-5, M-12/13). |
| Flexible for future interleaving (BERT/embeddings/Flux/speculative) | ⚠️ **Partly.** The `EngineDriver.drain` step-boundary yield is the right seam, but request types are hard-wired to token generation, and `ChatSession` **silently disables speculative decoding** when a scheduler is installed. See §9 for the change-now list. |
| Swifty API in line with existing ones | ✅ Largely: `AsyncStream<Generation>`, cancel-by-drop, opt-in init. ⚠️ Java-y spots: 7 request-carrier structs, 5 parallel arrays in `insert`, `Batch`/`Batched` naming split, misnamed `schedulerBusy` (§8). |
| Support **all** LLM cache variants | ❌ **Not met.** In practice 2 of 7 variants batch (KVCacheSimple, RotatingKVCache keep=0 — which covers ~90% of popular checkpoints); QuantizedKVCache batches in theory only; Chunked/Mamba/ArraysCache/CacheList **serialize gracefully** (queue, no crash). Two admitted models are actively unsafe when batched: Mistral3 (llama-4 scaling) and Gemma3n (sliding-mask re-slice). Full matrix in §10. |
| No subtle bugs | ❌ 50 confirmed findings, 17 major (§5) — though none in the core cache math, which held up under hand-recomputation. |
| Performance | ⚠️ Solid skeleton (correct double-buffering, zero-copy upgrade), but the top wall-clock item is unwired prompt-cache fetch on the batched path (§6). |
| Ready for prefix-caching / MaterializedMLXArray phase | ⚠️ Good bones (protocol-typed `PromptCaching`, trie already in place, dormant `cachedKVState` seam), 4 cheap change-now items (§9). |
| Swift 6 compliant | ✅ Strict-concurrency clean *by design* (no `@unchecked`), ❌ PR21 currently doesn't compile, and 3 confirmed actor-reentrancy races (§7). |
| Not over-engineered | ⚠️ ~20% (≈1,500–1,700 LOC) is speculative or dead in-tree (§8). |

---

## 5. Confirmed bugs (adversarially verified)

All 50 are in Appendix A with failure scenarios; the 17 major ones:

**Batched output silently diverges from single-stream output (parity class):**
- **M-1** `DecodeBatch.swift:372` — repetition/presence penalties applied to **log-probs instead of raw logits**; sign-dependent penalty math inverts. Any batched request with `repetitionPenalty` samples differently than the same request single-stream.
- **M-2** `BatchGenerationEngine.swift:371` + `InferenceScheduler.swift:1275` — adopted (upgraded) rows re-seed penalty processors from **generated tokens only**; the prompt drops out of the penalty window mid-stream.
- **M-3** `SchedulerTokenHandler.swift:83/93` — **stop strings are lost** on every scheduler path: no `StopStringFilter` exists anywhere in the batched pipeline; runtime `stopStrings` (upstream #372) stream through and generation runs on. (Also flagged in open threads; confirmed twice independently.)
- **M-4** `BatchGenerationEngine.swift:385` — `makeAdoptedBatch(stateMachines: nil)` installs a **never-matching stop matcher** instead of the engine default: adopted rows ignore EOS and run to `maxTokens`.

**Scheduler state machine (races confirmed by interleaving analysis):**
- **M-5** `InferenceScheduler.swift:921` — actor-reentrancy in `ensureDriver()` (`await` before `self.driver` assignment) can mint **two engines/drivers** running concurrent decode loops on one model.
- **M-6** `InferenceScheduler.swift:1078/1082` — TOCTOU between `await driver.hasWork` and a concurrent submit: a stale `batchLoopFinished` clobbers a live `.single`/`.batched` state back to `.idle` → untracked streams, or a single iterator decoding concurrently with the engine.
- **M-7** `InferenceScheduler.swift:474` — cancellation during the `.upgrading` window is a **no-op**; with a buffering text handler the orphaned row can generate to `Int.max`.
- **M-8** `InferenceScheduler.swift:373` — `parametersAreBatchable` screens `kvBits` but not `kvScheme`: a running `kvScheme` request is **truncated mid-generation** at upgrade (failed migration → premature `.length`), and admitted ones silently lose KV quantization.

**High-level API behavior:**
- **M-9** `ModelContainer.swift:90` — scheduler **without** a `promptCache` (the default!) silently turns every multi-turn `ChatSession` into full-history re-prefill per turn — a large latency regression from enabling a feature marketed as zero-cost.
- **M-10** `GenerationRequest.swift:35` — the public `promptCache` field is **never used as a cache**, only as a write-back boolean; per-request caches are silently ignored (open thread, confirmed).
- **M-11/12** `LRUPromptCache.swift:321/237` — cache namespace ignores topology (`maxKVSize` requests can fetch unbounded caches and vice-versa — attention semantics silently change), and fetch/save run **full GPU deep-copies + eval while holding the unfair lock**, stalling every concurrent request.

**Model-level batching hazards (admitted by the factory, unsafe when ragged):**
- **M-13** `Mistral3Text.swift:229` — llama-4 attention scaling reads scalar `cache[0].offset` (padded max width): short rows get the long row's positional scaling on every layer.
- **M-14** `Gemma3nText.swift:594` — sliding-window layers re-slice the batch mask with a scalar window-capped offset; past the window the mask is displaced (one gather index OOB) → wrong KV slots.
- **M-15** `BatchQuantizedKVCache.swift:279` — `trim()` missed the min-live-row clamp its siblings got in cb-1 (`34ca30a`) — direct evidence of the stack-skew problem in §2.
- Plus **15 pre-existing thread items verified still open** (Appendix B), led by the two critical `ChatSession` ones: the compile failure, and the tool-call restart that **replaces the whole conversation with just the tool result** on the scheduler path (total context loss; the non-scheduler path is unaffected because its KV cache carries the context).

**What held up:** the core cache mathematics. Agents hand-recomputed left-padding/rotation/extract arithmetic across the `{rotated}×{over-window}×{ragged}×{prefill/decode}` matrix and confirmed all 11 cb-1 fix commits are correct; 8 plausible-sounding claims (e.g. SSM double-advance, `fromSingle` keep>0 bypass, legacy-mask nil for t==1 in reachable models) were **refuted** with line-level evidence (Appendix C).

---

## 6. Performance (ranked, LAN case: 2–8 chats, 7B–30B)

1. **Batched path never fetches from the prompt cache** (`InferenceScheduler.swift:613-616`): while ≥2 chats are active, every `ChatSession` turn re-prefills its full history — seconds per turn at exactly the moment batching matters. The cache-side machinery (mixed-depth `prepareBatched`/`finalize`) already exists; only engine wiring is missing. *Biggest wall-clock win in the whole stack.*
2. **Decode-loop dtype cast after `asyncEval`** (`DecodeBatch.swift:396`): `asType(.int32)` creates a new node that evals *behind* the already-committed next-step buffer, defeating the double-buffer ≈ every token (~10–20% at 7B). One-line fix: normalize dtype inside the async graph.
3. **Admission freezes all decoding** for the entire joining prompt's prefill (`BatchGenerationEngine.swift:237`). Interleave decode steps between prefill chunks.
4. **Greedy fast path is dead code under the scheduler** (`InferenceScheduler.swift:1032`): a non-nil sampler is always installed, forcing log-softmax + per-row Swift sampling loops (two full-vocab argsorts/row/step with top-p). Return `nil` for default params; group identical-param rows.
5. **Row-finish costs**: per-layer `.item()` syncs + full-batch gather per finish + two full KV copies for write-back, on the driver actor. Host-side `leftPadding` mirror kills the syncs.
6. Also: per-admission `extend` copies the whole batch KV (O(k²) over an admission ramp); driver admits waiters one row at a time instead of one sub-batch; prefill uses sync `eval` per chunk where the single path uses `asyncEval`; masks are rebuilt as materialized arrays even for uniform batches.

Confirmed right: one intended sync barrier per step (upstream-parity double-buffer), zero-copy upgrade adoption, quantized decode staying in `quantizedMM`, O(prompt) trie lookups.

---

## 7. Swift 6 concurrency

The discipline is unusually good: strict concurrency with **zero `@unchecked Sendable`**, no `nonisolated(unsafe)`, no lock held across `await`, and a textbook SE-0430 `sending` handshake for the upgrade snapshot. `Generation` streams carry only Sendable payloads; no `MLXArray` ever rides a stream.

The holes are exactly where the compiler can't see: **actor reentrancy** (M-5/6/7 — each function is locally correct; the races live between `await`s), one **SendableBox used to alias rather than transfer** (`attachSchedulerIfNeeded` copies the `ModelContext`, permanently splitting the model into two synchronization domains while `SerialAccessContainer` still promises exclusivity — safe for evaluated weights, but `update()`/mutating `perform` become loaded guns), and identified windows where **two threads evaluate distinct MLX graphs sharing weights/RNG concurrently** — sanctioned by pre-existing repo comments, but resting on mlx-swift runtime internals nothing pins or tests. And PR21's `ChatSession.swift:660` currently fails the region-isolation checker outright (§2-3).

---

## 8. Over-engineering

≈1,500–1,700 LOC (~20%) is speculative or dead in-tree:

- **`BatchQuantizedKVCache` (604 LOC)** — unreachable through every integrated path (no `newCache` returns a `QuantizedKVCache` probe; scheduler rejects `kvBits`; engine always built with `cacheParameters: nil`). Defer; keep the SDPA mask fix.
- **`LRUPromptCache` disk persistence (~390 LOC)** + checkpoint dual-LRU (~40) — a whole crash-safe storage subsystem with no in-repo caller. Split out.
- **Request-carrier proliferation**: 7 structs for one logical request (`GenerationRequest → ScheduledRequest → SingleRequest → SchedulerRequest → PendingSubmission → RequestRecord/AdoptedRecord`); folding `EngineDriver` into the scheduler (or making it a non-actor helper) deletes ~350–450 LOC and 3 concepts.
- **Dead/degenerate machinery**: multi-state `StopSequenceMatcher` DFA used only for single-token EOS; unused `rawToken` handler mode; `PrefillBatch.extend` whose precondition makes it uncallable; dead `mismatchedRequestCounts` case; dormant `cachedKVState` fields; `.pendingUpgrade` + queue machinery that could be "run non-batchable requests concurrently" (~120 LOC).
- **Public-surface overcommitment**: `DecodeBatch`/`PrefillBatch`/`adoptActiveBatch`/`FinishedRowCache` all `public` — lock internal seams down (or `@_spi`) before someone builds on them.

What a LAN-server author still builds by hand: bounded queues/backpressure (all three queues are unbounded), a default `maxTokens` that isn't `Int.max`, fairness, typed errors on the stream, and observability beyond `activeRequestCount`.

---

## 9. Future-readiness (your next phase)

- **Prefix caching / agentic prefill:** good bones — `PromptCaching` is protocol-typed for a future block/paged cache, the store is already a token trie with cross-request sharing, and write-back keys on full prompt+generation history. What blocks radix caching is granularity (monolithic per-leaf `[KVCache]` copies), not keying. **The biggest gap is that the batched path never fetches** (§6-1) — the next-phase feature's seam (`SchedulerRequest.cachedKVState`) exists but is dead.
- **MaterializedMLXArray:** slots in at exactly six identified boundaries (LRU `State`, `PromptCacheFetchResult.cache`, `SchedulerRequest.cachedKVState`, `FinishedRowCache.finalCache`, `LiveIteratorState.cache`, disk snapshots). Entries are *already* materialized (deep-copy + eval) at insert, so the semantic change is nil. **Change now:** give `PromptCaching` a Sendable snapshot payload type and async (or async-refined) requirements — retrofitting `async` onto a shipped sync protocol breaks every call site.
- **Speculative decoding:** the batched step API is locked to 1 token/row/step in four places (`BatchStepResult.token: Int` being the public one). The rollback primitive needs per-row trim (`trimBatched(perRow:)`) — the math already exists in `finalize()`'s roll. **Change now (cheap, breaking later):** pluralize `BatchStepResult` (or mark the surface `@_spi`), add `trimBatched(perRow:)` with a default impl, and generalize the scheduler's single path over `TokenIteratorProtocol` — which also fixes the today-bug that installing a scheduler silently disables speculative decoding at concurrency 1.
- **Embeddings/BERT/Flux interleaving:** `EngineDriver.drain`'s between-step `Task.yield()` is the right seam; extract a small `GPUWorkSource` protocol on the (still-internal) driver now, before the public surface calcifies. Don't generalize `GenerationRequest` — aux work wants a continuation, not a token handler. Note `EmbeddingModel` isn't a `LanguageModel`, so it can never enter `BatchGenerationEngine` — interleaving happens at the driver loop, not the engine.

---

## 10. Cache-variant coverage matrix

| Variant | Producers | Batched? | Behavior under concurrency |
|---|---|---|---|
| `KVCacheSimple` | ~40 LLM families (default) | ✅ `BatchKVCache` | Correct (masks, per-row RoPE, extract round-trip) — except Mistral3 scalar-offset scaling (M-13) |
| `RotatingKVCache` keep=0 | Gemma3/3n/4, GPT-OSS, Olmo3, Exaone4, AfMoE, MiMoV2, Mistral3 | ✅ `BatchRotatingKVCache` | Correct incl. wrap/extract — except Gemma3n mask re-slice (M-14) |
| `RotatingKVCache` keep>0 | any `maxKVSize` request | ❌ factory throws | Clean FIFO serialization (scheduler gates earlier on `maxKVSize`) |
| `QuantizedKVCache` | `kvBits`/`kvScheme` mid-decode | ⚠️ theory only | `kvBits` gated → serialized; **`kvScheme` leaks through the gate (M-8)**; `BatchQuantizedKVCache` has no reachable producer |
| `ChunkedKVCache` | prompt-cache deserialization only | ❌ factory throws | Serialization; but LRU admits it via subclass check → fatalError on write-back (open thread) |
| `MambaCache`/`ArraysCache` | NemotronH, Jamba, LFM2(+MoE), GraniteMoeHybrid, Qwen3Next, Qwen3.5 | ❌ factory throws (deliberate: some models ignore SSM masks) | Clean FIFO serialization; batched SSM machinery exists but is gated |
| `CacheList` hybrids | FalconH1, BaichuanM1 | ❌ (recurses into Mamba child) | Clean FIFO serialization; `BatchedCacheList` is well-built dead code |
| VLM topologies | all VLMs | ❌ `loadedAsVLM` gate | Single-path fallback; PaliGemma/Qwen2VL scalar-RoPE hazards unreachable today (gate is a fragile load-time flag — check `input.image == nil` per request instead) |

Honest summary vs. your requirement: **2 of 7 variants batch in practice** (covering the large majority of popular checkpoints); everything else degrades gracefully to serial. If "all variants" is a hard requirement, the SSM/hybrid gate is the big-ticket item — it needs model-level `createSSMMask` wiring (Jamba, LFM2, GraniteMoeHybrid, BaichuanM1), not cache work; the batched SSM cache code already exists and is tested.

---

## 11. Unusual things worth knowing

- Doc comments leak private downstream fork names (“Layr”, “batching3”, “PicoCore”) at `BatchGenerationEngine.swift:292`, `EngineDriver.swift:168,397`, `LRUPromptCache.swift:35,166` — scrub before merge.
- cb-2 quietly **changed Gemma2's single-stream attention numerics** (boolean-mask-added-to-logits → `where`/-1e9) with zero tests pinning either behavior.
- The `BatchModelRegressionTests` parity suite tests only Phi3/Gemma4/FalconH1 with 3–5-token prompts and 2 decode tokens — it skips every model with a confirmed batching bug (Gemma2, Gemma3n, Mistral3) and never crosses a sliding window; no test exercises stop strings, penalty parity, upgrade token-parity end-to-end, cancellation during `.upgrading`, or any ChatSession-with-scheduler flow.
- Batched completion info hardcodes `promptTime: 0` → `promptTokensPerSecond == +∞` in your server metrics.
- The default `maxTokens` on the batched path is `Int.max`.

---

## 12. Recommended path to merge

1. **Unblock the process** (§2): rebase the stack onto cb-1's tip (or push the union as the integration branch), fix the CI `-only-testing` filter + add the engine/scheduler/prompt-cache suites, fix `ChatSession.swift:660`.
2. **Merge #17** (or upstream's fix) standalone.
3. **Land PR #18** after the two open items (serialization registration can also be documented-as-unsupported for now).
4. **Fix the parity class in PR #19** (M-1..M-4) — these are small, local fixes — plus the dtype-cast perf fix, then land.
5. **Slim PR #20** to the in-memory cache (defer persistence), fix the two LRU majors (topology namespace, lock-held eval), then land.
6. **PR #21 needs a real hardening pass**: the three reentrancy races (M-5/6/7 — consider making driver creation synchronous and re-checking state after every await), the kvScheme gate, stop-string parity (reuse `TextToolTokenLoopHandler` instead of the parallel handler stack), the ChatSession tool-call restart, speculative-decoding gating, and either wiring `GenerationRequest.promptCache` or removing the field. Land last, with the missing tests from §11.
7. **Defer PR #22** (keep its SDPA fixes); re-introduce when the engine can actually be constructed with quantized caches — with the trim clamp fix.
8. Then the highest-value follow-up of all: **wire prompt-cache fetch into batched admission** (§6-1) — it is also your phase-2 feature.

---

*Appendices: A — all 50 confirmed findings with failure scenarios; B — 15 pre-existing review-thread items still open; C — 8 investigated-and-refuted claims.*
## Appendix A — All 50 confirmed findings (adversarially verified)

Every finding below was independently re-derived by a second agent instructed to refute it; only CONFIRMED verdicts are listed. Severities are the verifier's, not the finder's.


### MAJOR

**`Libraries/MLXLLM/Models/Gemma3nText.swift:594` — Gemma3n sliding-window layers re-slice the batch-aware mask with a scalar capped cache offset, misaligning (and indexing past) the mask once context exceeds the sliding window**

Gemma3n's topology (RotatingKVCache keep=0 + KVCacheSimple, Gemma3nText.swift:677-695) is accepted by the batched-cache factory, so it runs on the engine. Its decoder derives `pastSeenTokens = cacheArray.first??.offset` (line 817) and each sliding layer, whenever the mask is `.array`, rebuilds it: `offset = max(0, cachePosition.max() - effectiveSeqLen + 1)`, `take(updatedMask, (0..<min(effectiveSeqLen, W)) + offset, axis: -1)` (lines 584-597). Single-stream this surgery is skipped during decode because `RotatingKVCache.makeMask(n:1)` returns `.none` when maxCacheSize == windowSize (KVCache.swift:768-788), but `BatchRotatingKVCache.makeMask` ALWAYS returns `.array` (BatchRotatingKVCache.swift:1010-1087), so batched decode enters the [...]

*Failure:* Two concurrent Gemma3n requests decode past the sliding window (512/4096 tokens of accumulated context). From the wrap point onward, every sliding layer's attention mask is displaced by >= 1 column relative to the ring buffer (with one gather index out of range), so rows attend the wrong KV slots — degraded or garbage continuations that never occur on the single-stream path.

**`Libraries/MLXLLM/Models/Mistral3Text.swift:229` — Mistral3 llama-4 attention scaling reads scalar cache[0].offset: ragged batched rows get the longest row's positions (and rotating rows a window-capped position), silently distorting logits**

`Mistral3TextModelInner.callAsFunction` computes `offset = cache[0].offset` (Mistral3Text.swift:227-232) and feeds it to `getLlama4AttentionScale(start: offset, stop: offset + n, ...)` when the checkpoint config carries `rope_parameters.llama_4_scaling_beta` (lines 248-257). Mistral3Text's topology (RotatingKVCache keep=0 for sliding layers + KVCacheSimple, newCache at 353-361) is ACCEPTED by `makeBatchedCacheFactory`, so the model is admitted to continuous batching — but under batching `BatchKVCache.offset` returns the shared padded max width `_idx` (BatchKVCache.swift:43-46) and `BatchRotatingKVCache.offset` returns `min(_scalarOffset, maxCacheSize)` (BatchRotatingKVCache.swift:193-196), neither of which is the row's true position. [...]

*Failure:* Two concurrent requests on a Ministral/Mistral-3 checkpoint whose config includes llama_4_scaling_beta: request A has an 800-token prompt, request B a 10-token prompt. After joint admission, B's queries are scaled with position ~800+ instead of ~10 on every layer, every step — outputs differ measurably from the single-stream path with no error or warning; quality silently degrades until the batch empties.

**`Libraries/MLXLMCommon/Batching/BatchGenerationEngine.swift:371` — Adopted rows seed penalty processors with generated-only history, dropping the prompt from the penalty context**

makeAdoptedBatch's doc (lines 340-344) promises 'the migrated row's full tokens history seeds the processor's penalty context so penalties continue seamlessly', but the scheduler must pass generated-only history (InferenceScheduler.swift:1275 `tokens: liveGeneratedTokenIds + [liveCurrentToken]`) because the same `tokens` parameter also becomes the row's `allTokens` and the driver's write-back key assumes it excludes the prompt (EngineDriver.swift:350, 523-527). DecodeBatch then seeds each fresh processor with `priorTokens` = generated-only (line 371 here; DecodeBatch.swift:172-174), while the single stream's TokenRing held `(prompt + generated).suffix(contextSize)` (Evaluate.swift:157 prompt() with full prompt; TokenRing.loadPrompt [...]

*Failure:* Request A with repetitionPenalty=1.3 has emitted 5 tokens when request B arrives and triggers the single-to-batch upgrade: A's fresh RepetitionContext ring holds only 5 tokens (prompt absent) instead of prompt+5, so a prompt token that single-stream decoding would have penalized is no longer penalized, and A's output visibly changes mid-stream (e.g., starts echoing prompt phrases).

**`Libraries/MLXLMCommon/Batching/BatchGenerationEngine.swift:385` — makeAdoptedBatch with nil stateMachines installs a never-matching empty stop matcher instead of the engine default, and loses partial multi-token stop progress**

insert() falls back to `defaultStateMachine` (the engine's EOS matcher, line 157), but makeAdoptedBatch passes `stateMachines` straight through (line 385); DecodeBatch fills nil with `StopSequenceMatcher()` — an empty matcher that never matches (DecodeBatch.swift:158, StopSequenceMatcher.swift:94-97). An adopted row without an explicit matcher therefore ignores EOS and runs to maxTokens, asymmetric with insert()-ed rows on the same engine. Additionally, matcher states are always created fresh (`machines.map { $0.makeState() }`, DecodeBatch.swift:160), so any partial multi-token stop-sequence progress from the pre-upgrade stream is discarded. The scheduler currently masks both: it always passes defaultStopMatcher() [...]

*Failure:* Power user builds BatchGenerationEngine(eosTokens: [[eosId]]), runs a single TokenIterator, then migrates it via makeAdoptedBatch(seedTokens:..., stateMachines: nil): inserted rows stop on eosId but the adopted row streams eosId and keeps generating until maxTokens, emitting post-EOS garbage.

**`Libraries/MLXLMCommon/Batching/BatchQuantizedKVCache.swift:279` — trim() lacks the min-live-row clamp that cb-1 commit 34ca30a added to BatchKVCache and BatchRotatingKVCache**

Commit 34ca30a ('Clamp batched cache trims to the minimum live per-row length') fixed BatchKVCache.trim (BatchKVCache.swift:187-192) and BatchRotatingKVCache.trim to clamp the trim to `min(n, _idx, max(0, batchOffsets.min()))`, because subtracting the full n from every row of a ragged batch drives a shorter row's batchOffsets negative and leaves its leftPadding larger than the new _idx. BatchQuantizedKVCache.trim (lines 278-283) still does `let trimmed = min(_idx, n)` with no per-row clamp — the fix was applied only to the cb-1 siblings and never mirrored into the cb-5 class, even though the class advertises `isTrimmable == true` (line 275) so generic `trimPromptCache` consumers will call it. No engine-internal path trims a batched [...]

*Failure:* BatchQuantizedKVCache with leftPadding [4, 0], prefilled to _idx = 6 (batchOffsets [2, 6]). trim(5) returns 5, sets _idx = 1, batchOffsets = [-3, 1]. Row 0 now has leftPadding 4 > _idx 1: extract(idx: 0) at line 485 slices `padding ..< self._idx` = 4..<1 (invalid range -> MLX slice failure) and writes metaState offset `_idx - padding` = -3; decode after the trim rotates row 0 with negative RoPE positions and its mask row (`rinds >= 4` with [...]

**`Libraries/MLXLMCommon/Batching/DecodeBatch.swift:372` — Per-row penalty processors run on log-probs instead of raw logits, diverging from single-stream repetition-penalty math**

In DecodeBatch.step the whole batch is normalized first (`logprobs = sampleLogits - logSumExp(...)`, line 355) and `processors[i]?.process(logits: rowLogprobs)` (line 372) is fed the normalized slice. The single path processes RAW logits before any normalization (Evaluate.swift convertToToken: `logits = processor?.process(logits: logits)`; TopPSampler then does its own logSoftmax). RepetitionContext.process branches on sign (`selected < 0 ? *penalty : /penalty`, Evaluate.swift:388-392); log-probs are always <= 0, so the batched path always takes the multiply branch, penalizing with different magnitudes than the single stream (which divides positive logits). Additive presence/frequency penalties are shift-invariant and unaffected, but [...]

*Failure:* Request with repetitionPenalty=1.3 and temperature=0 runs batched: a context token with raw logit +2.0 is penalized to 2.0/1.3=1.54 on the single path but to logprob*1.3 on the batched path; near-tied competitors resolve differently, so the batched request emits a different token sequence than the identical single-stream request.

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:373` — parametersAreBatchable ignores kvScheme: running kvScheme request is truncated mid-generation at single->batch upgrade, and kvScheme requests admitted to the batch silently lose KV quantization**

`parametersAreBatchable` gates only `kvBits` and `maxKVSize` (InferenceScheduler.swift:372-374), but `maybeQuantizeKVCache` treats `kvScheme` as an independent quantization trigger even when `kvBits == nil` (KVCache.swift:2095-2101: "kvScheme overrides kvBits"), and `TokenIterator.step` calls it every decode step with the request's kvScheme (Evaluate.swift:699-705, fields set at 583/617). So a request with `kvScheme: "affine4"` and `kvBits: nil` passes the batchable gate. Two consequences: (a) if it is the RUNNING single request when a second request arrives, `canUpgrade` probes fresh caches via `newCache(parameters:)` (line 350-354) which for nearly all models returns plain `KVCacheSimple` (only FalconH1 reads kvScheme), so the [...]

*Failure:* Server sets `GenerateParameters(kvScheme: "affine8")` for all requests on a Llama/Qwen model with a scheduler installed. Request A streams normally; ~10s in, request B arrives. The scheduler initiates an upgrade, request A's task deposits its live (now QuantizedKVCache) state and exits, migration fails, and request A's stream is closed after one more token with stopReason .length at an arbitrary point (e.g. 180 of the requested 2000 [...]

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:474` — Stream cancellation arriving during the `.upgrading` window is dropped; recovery relies on a side channel that text-mode buffering can defer indefinitely**

cancel(requestID:) (lines 471-491) resolves a request only if state is `.single`/`.pendingUpgrade` (task.cancel), it is in `queuedRequests`, or the driver knows its token. While `state == .upgrading` (set at line 1114, lasting through the handshake await, driver creation, and adoptMigrated), the original single request is stored nowhere the cancel path can see: the switch at 473-478 yields nil, it is not queued, and `driver?.cancel(token:)` is a no-op (driver may not exist yet; the adopted row's record is registered only later at EngineDriver.swift:336). The same applies to the joining request held as a local in upgrade(). The AsyncStream onTermination fires exactly once (SchedulerTokenHandler.swift:163-167), so the cancel is [...]

*Failure:* R1 streaming; R2 arrives, upgrade begins (state=.upgrading). The consumer drops R1's stream during the window (one decode step + engine construction wide). cancel(R1) finds nothing and no-ops. R1 is adopted into the batch and the model enters a malformed tool-call emission: processToken keeps buffering and returning true, so the row generates until Int.max tokens — GPU pinned and batch slot occupied indefinitely for a request nobody is consuming.

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:921` — Actor-reentrancy race in ensureDriver() can create two BatchGenerationEngines and two EngineDrivers**

ensureDriver() (InferenceScheduler.swift:900-926) checks `if let driver { return driver }`, builds the engine + EngineDriver, and only assigns `self.driver = driver` at line 924 — AFTER an `await driver.setMaxBatchSize(...)` suspension at line 920/922 (taken whenever `hasPendingMaxBatchSize` is true, i.e. setMaxBatchSize was ever called, or `configuration.maxBatchSize != nil`). Because InferenceScheduler is a reentrant actor, a second admission path can run at that suspension, see `self.driver == nil`, and construct a second engine/driver. Concrete interleaving: R1 running single; R2 arrives → upgrade() awaits the handshake (line 1116); R3 arrives → route `.upgrading` → admitToBatch → ensureDriver creates driverB and suspends at [...]

*Failure:* Scheduler configured with `maxBatchSize` (or after any setMaxBatchSize call). R1 decoding single; R2 and R3 submitted near-simultaneously. R3's ensureDriver suspends at `await driver.setMaxBatchSize`; R2's upgrade resumes and builds a second engine. Two engines run concurrent decode loops on the same model; cancelling R3's stream is routed to the wrong driver and silently does nothing, so R3 decodes to its token budget with no consumer; [...]

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:1078` — TOCTOU between batchLoopFinished's `await driver.hasWork` and a concurrent submit leaves state=.idle while the batch loop is running**

batchLoopFinished (lines 1072-1088) reads `await driver.hasWork` and, if false, sets `state = .idle` and drains the queue. This read and the state write are separated by actor suspensions and race with a concurrent admitToBatch. Plain-FIFO interleaving that breaks it: the batch loop exits and enqueues batchLoopFinished; it suspends at the hasWork hop; a new submit R4 then runs on the scheduler (route sees `.batched` at line 554-555), sets state=.batched (line 978), and enqueues driver.submit — but the driver processes the earlier hasWork message first (false), then the submit. Back on the scheduler, R4's continuation runs startBatchLoop (new drain begins), and batchLoopFinished's continuation then stomps `state = .idle` (line 1082) [...]

*Failure:* Server under churn: batch empties at the exact moment a new request arrives. hasWork reads false before the submit lands; scheduler ends at state=.idle with the driver's drain loop actively stepping the engine. The next submit starts a single-stream TokenIterator whose forward passes run concurrently with the engine's batch forward passes for that request's entire generation; a held kvBits request also starts concurrently despite the design [...]

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:1082` — Stale batchLoopFinished callback clobbers a live .single/.pendingUpgrade state back to .idle (TOCTOU across await)**

batchLoopFinished (InferenceScheduler.swift:1072-1088) does a check-then-act across a suspension point: it reads `await driver.hasWork` (line 1078), and if false unconditionally sets `state = .idle` (line 1082) without verifying the scheduler is still in `.batched`. Because startBatchLoop (1063-1068) spawns an unstructured Task on EVERY admitToBatch/upgrade call, multiple loop-completion callbacks per batch epoch are in flight (each no-op `drain` still runs batchLoopFinished — EngineDriver.swift:415). While one such callback is suspended at `await driver.hasWork`, actor reentrancy lets a user submit run route(.idle) → startSingle, installing `state = .single(R)` (startSingle is fully synchronous, lines 590-707). The stale callback [...]

*Failure:* state=.batched, batch empties; primary loop's batchLoopFinished suspends at `await driver.hasWork` (or a churn-spawned duplicate does). A user submits request R -> route(.idle) -> startSingle -> state=.single(R). The suspended batchLoopFinished resumes with its stale hasWork=false observation and executes `state = .idle`. R keeps decoding but is untracked: dropping R's stream no longer cancels its task promptly, R is never upgradeable, and [...]

**`Libraries/MLXLMCommon/Batching/LRUPromptCache.swift:237` — Full GPU deep-copy + eval executed while holding the unfair lock on every fetch hit and for the entire cache during save**

`fetchNearestCacheResult` runs `State.fetchNearestCache` inside `state.withLockUnchecked` (lines 237-239), and every hit branch calls `LRUPromptCache.deepCopy` there (lines 950, 966, 985), which materializes and `eval()`s the whole entry (lines 760-764) — potentially hundreds of MB of KV per entry — while the os_unfair_lock is held. `diskSnapshotsAndOwnedKeys` (lines 590-598) is worse: it deep-copies and evals EVERY live entry inside one lock hold (line 618). Insert correctly materializes before taking the lock (line 262), so any concurrent insert (EngineDriver.writeBack on the decode hot path, EngineDriver.swift:528) or fetch blocks on the lock for the full GPU copy duration. Stored entries are immutable after insert (only `lastUsed` [...]

*Failure:* Agentic workload with `save(to:)` on a timer: save takes the lock and deep-copies+evals a 5 GB cache; meanwhile the driver actor finishes a row and calls insertCache, blocking its decode loop on the lock for the whole copy — every active batched request stalls for seconds. Same contention between two concurrent single-path requests: request B's prefill cannot start (its fetch blocks) while request A's multi-GB fetch copy evals.

**`Libraries/MLXLMCommon/Batching/LRUPromptCache.swift:321` — Prompt-cache namespace ignores cache topology (maxKVSize): cross-request contamination silently changes attention semantics**

Entries are keyed only by (model, salt, tokens); nothing records the cache topology the entry was built with. `canUsePromptCache` (LRUPromptCache.swift:313-322) only checks that the CURRENT request's `model.newCache(parameters:)` is compatible, and the fetch call site (InferenceScheduler.swift:618-648) hands the fetched `[KVCache]` straight to `TokenIterator(cache:)`, which bypasses `model.newCache` entirely (Evaluate.swift:608 `cache ?? model.newCache(parameters:)`). `GenerateParameters.maxKVSize` (Evaluate.swift:64) is per-request and flips newCache between `KVCacheSimple` and `RotatingKVCache(maxSize:)` (LanguageModel.swift:305-311). Write-back stores whichever topology the finished run used under the same namespace. So two [...]

*Failure:* Request A (maxKVSize=nil) completes and stores KVCacheSimple layers for its prompt. Request B arrives with the same prompt prefix but maxKVSize=512: canUsePromptCache passes (newCache would be Rotating -> compatible), fetch hits, and B decodes on unbounded KVCacheSimple caches — its sliding-window memory bound is silently dropped. Reverse order: A stored RotatingKVCache(512) entries; a later request without maxKVSize gets a shorter-prefix [...]

**`Libraries/MLXLMCommon/Batching/SchedulerTokenHandler.swift:83` — Scheduler routing loses stop-string handling: modelConfiguration.effectiveStopStrings never applied on the batched/scheduler path**

The single-stream reference path filters text through `StopStringFilter(stopStrings: modelConfiguration.effectiveStopStrings)` inside `generateTask` (Evaluate.swift:1543-1546; also 1486, 1684). The scheduler's text handler has no stop-string filter at all — `SchedulerTokenHandler.text` builds only a detokenizer + `ToolCallProcessor(format:)` (SchedulerTokenHandler.swift:80-84, 106-128) — and the engine-side `StopSequenceMatcher` is constructed exclusively from EOS token IDs (`eosSequences` → `stopTokenIds(configuration:tokenizer:)`, InferenceScheduler.swift:928-934, 876); `grep effectiveStopStrings Batching/` is empty. Since `ModelContainer.generate` transparently reroutes ALL non-VLM generation through the scheduler when one is [...]

*Failure:* A model config with `stopStrings: ["</answer>"]` (multi-token textual stop): without a scheduler generation stops at the marker and suppresses it; with a scheduler installed the same call streams '</answer>' to the user and keeps generating until EOS or maxTokens — different, wrong output from a supposedly transparent opt-in.

**`Libraries/MLXLMCommon/Batching/SchedulerTokenHandler.swift:93` — Scheduler path silently drops runtime stop strings (effectiveStopStrings) — no StopStringFilter anywhere in the batched pipeline**

The non-scheduler single path threads `context.configuration.effectiveStopStrings` into `TextToolTokenLoopHandler`/`StopStringFilter` (Evaluate.swift:1484-1488, 2145-2228; upstream #372), which holds back partial matches and trims the whole matched stop string. The scheduler path has no equivalent: `SchedulerTokenHandler.text` (line 93-170) builds only a detokenizer + tool-call processor, the engine's stop matcher is built from single-token EOS ids only (`eosSequences`, InferenceScheduler.swift:930-935 maps `stopTokenIds` → one-token sequences), and the scheduler's own single-request loop checks only `stopTokenIds`/`unknownTokenId` (InferenceScheduler.swift:789). `ModelConfiguration.effectiveStopStrings` falls back to `extraEOSTokens` [...]

*Failure:* A model configured with `stopStrings = ["\nUser:"]` (or an extraEOSTokens string that tokenizes to >1 token): via plain `generate()` the output stops at and excludes "\nUser:"; via `InferenceScheduler`/ChatSession-with-scheduler the request streams "\nUser:" and everything after it, generating until EOS or maxTokens (Int.max default → effectively runaway).

**`Libraries/MLXLMCommon/GenerationRequest.swift:35` — GenerationRequest.promptCache is never used as a cache — it is only a write-back boolean; per-request caches are silently ignored**

The public field is documented 'Optional cross-request prompt cache for prefix KV reuse', but the scheduler only tests it for nil: InferenceScheduler.swift:606 `let writeBack = scheduled.request.promptCache != nil`, then fetches/inserts exclusively through the actor's attached cache — `if let promptCache, writeBack` (line 618) and `cacheForWriteBack = writeBack ? self.promptCache : nil` (line 664); EngineDriver does the same (EngineDriver.swift:97, 273: `writeBackToPromptCache: request.promptCache != nil` with fetch/insert against the driver's own `promptCache`). The attached cache comes solely from the container (ModelContainer.attachSchedulerIfNeeded, ModelContainer.swift:270-280). So a request-supplied cache instance is never read [...]

*Failure:* Caller builds `ModelContainer(context:scheduler:)` (no container promptCache) and submits `generateBatched([GenerationRequest(input: i, promptCache: myLRUCache)])`: attach stores nil, line 618's `if let promptCache` fails → no prefix fetch, no write-back; `myLRUCache` stays empty forever and repeated shared-prefix requests re-prefill fully — the documented parameter is a silent no-op. Passing a different cache than the container's silently [...]

**`Libraries/MLXLMCommon/ModelContainer.swift:90` — Scheduler without a promptCache (the default) silently degrades every multi-turn ChatSession to full-history re-prefill per turn**

`init(context:scheduler:promptCache: nil)` defaults promptCache to nil (ModelContainer.swift:87-95). ChatSession's scheduler path deliberately abandons the session `[KVCache]` and 're-tokenizes the full conversation each turn, relying on ModelContainer.promptCache to hit the shared prefix' (ChatSession.swift:679-686). But `ModelContainer.generate` fills `GenerationRequest.promptCache` from the container (line 243), and with nil the scheduler computes `writeBack = false` (InferenceScheduler.swift:606) → no prefix fetch or write-back ever happens (lines 618, 664). Nothing warns: the obvious 'turn on batching' call — `ModelContainer(context: ctx, scheduler: InferenceScheduler())` — makes every ChatSession turn prefill the entire re- [...]

*Failure:* 20-turn chat with a ~4k-token accumulated history on `ModelContainer(context:scheduler: InferenceScheduler())`: turn 20 re-prefills ~4k tokens (seconds on a phone-class GPU) versus ~tens of tokens on the plain container — a large, silent latency regression caused by enabling a feature marketed as zero-cost.


### MINOR

**`Libraries/MLXLMCommon/Batching/BatchGenerationEngine.swift:98` — Engine validates cache topology but not model interface: prepare()/LMOutput.State are bypassed, and state-only models hit fatalError**

Init validates only the cache topology via makeBatchedCacheFactories (lines 98-100). PrefillBatch/DecodeBatch never call model.prepare(...) and use only the stateless `callAsFunction(_ inputs: MLXArray, cache:)` (PrefillBatch.swift:149, DecodeBatch.swift:330), while the single path uses prepare + the state-ful overload and threads `LMOutput.State` between steps (Evaluate.swift:660-700). The default stateless overload is `fatalError` (LanguageModel.swift:286-288), so a model that only implements the state-based interface (several VLMs, e.g. Qwen25VL whose decode depends on state seeded during prepare, Qwen25VL.swift:938/1013) crashes or silently mis-generates if handed to the engine — with no throw at init and no doc caveat on [...]

*Failure:* BatchGenerationEngine(model: someVLM) succeeds (per-layer caches are plain KVCacheSimple, topology passes); insert()+next() then crashes with fatalError("callAsFunction(inputs:cache:) not implemented") — or, for models that implement the stateless overload but need prepare-time state, silently generates garbage.

**`Libraries/MLXLMCommon/Batching/BatchGenerationEngine.swift:99` — Quantized batched caching is dead code as wired: kvBits in cacheParameters is silently ignored for every supported model**

The engine builds cache factories from `model.newCache(parameters: cacheParameters)` (BatchGenerationEngine.swift:98-99), but the default `newCache` implementation (LanguageModel.swift:299-312) honors only maxKVSize and returns KVCacheSimple regardless of kvBits — single-stream quantization happens later via maybeQuantizeKVCache in the iterator, which the engine has no equivalent of. The only in-repo model whose newCache eagerly returns QuantizedKVCache is FalconH1 (FalconH1.swift:816-820), and its CacheList(MambaCache, ...) topology is rejected by the factory (BatchedCache.swift:310-315). The scheduler additionally constructs the engine without cacheParameters (InferenceScheduler.swift:906-912) and routes kvBits requests to the [...]

*Failure:* A library user constructs `BatchGenerationEngine(model: llama, cacheParameters: GenerateParameters(kvBits: 4))` expecting 4-bit KV memory. newCache ignores kvBits, the probe is KVCacheSimple, the factory returns plain BatchKVCache rows, and the process uses full fp16 KV memory — the quantization request is silently dropped (potential OOM at the batch sizes the user planned for).

**`Libraries/MLXLMCommon/Batching/BatchGenerationEngine.swift:186` — insert() validates emptiness but not token-id range; a negative token id fatally traps during admission**

validateRequest (lines 179-204) rejects empty prompts and non-positive maxTokens but not token values. Prompt tokens are later force-converted with `UInt32($0)` in PrefillBatch.prompt (PrefillBatch.swift:134, 141) and generate (`UInt32($0.last!)`, line 195), which traps on negative values — a process-fatal crash, not a thrown error, and it fires inside next() after the request was already dequeued (admitFromQueue at line 407 removes it before prefill). For an explicit throwing API whose whole insert path is built around recoverable BatchGenerationEngineError, garbage token ids should be caught at insert time (out-of-vocab ids >= vocabSize are also unchecked and become out-of-bounds GPU gathers).

*Failure:* engine.insert(prompts: [[-1]]) succeeds; the next engine.next() call crashes the whole process with a Swift runtime trap in UInt32.init, taking down every other in-flight request in the batch.

**`Libraries/MLXLMCommon/Batching/BatchKVCache.swift:300` — extend() silently drops the receiver's rows when self has row metadata but no KV buffer and other is populated**

extend's precondition (BatchKVCache.swift:288-293) rejects a metadata-carrying-but-unprefilled `other`, but the mirror case on the receiver is not handled: when `self.keys == nil` with `self.batchSize > 0` (rows added via init(leftPadding:) or prepareBatched(leftPadding:), BatchKVCache.swift:570-579) and `other.keys != nil`, the `else if other.keys != nil` branch (lines 300-307) wholesale adopts other's keys/values/batchOffsets/leftPadding, discarding self's rows without any error. The both-empty branch (lines 295-299) explicitly concatenates metadata "so admitted rows survive", so the asymmetry looks like an oversight rather than a contract. It is unreachable through the in-repo engine (DecodeBatch's constructor always runs a priming [...]

*Failure:* Direct API use: b1 = BatchKVCache(leftPadding: [2, 3]) representing two admitted-but-unprefilled rows; b1.extend(other: populatedBatch1Row). b1 now has batchSize 1 (other's row only) while the caller's per-row bookkeeping still tracks 3 rows; a subsequent filter/extract with index 1 or 2 gathers out-of-range or wrong rows, mixing one request's KV history into another's.

**`Libraries/MLXLMCommon/Batching/BatchKVCache.swift:444` — fromSingle(_:) accepts a ChunkedKVCache (KVCacheSimple subclass) and desyncs _idx from the retained buffer**

The a6915c3 fix added ChunkedKVCache rejection to merge (BatchKVCache.swift:385-388), the factory (BatchedCache.swift:269-274), and migrateCaches (InferenceScheduler.swift:316-318 uses an exact-type match with a comment noting `as?` would wrongly accept ChunkedKVCache), but `fromSingle(_ cache: KVCacheSimple)` (line 444) still takes the base type with no precondition. ChunkedKVCache keeps an absolute `offset` after maybeTrimFront (KVCache.swift:1143-1152) while its keys buffer holds only the retained chunk, so fromSingle sets `_idx = cache.offset` (line 450) larger than `keys.dim(2)` — violating the `_idx <= keys.dim(2)` invariant every other method assumes (state slicing at line 132, makeMask offset at line 491, update's prev range [...]

*Failure:* Any future/external caller passes a front-trimmed ChunkedKVCache (offset 10000, buffer width 8192) to BatchKVCache.fromSingle: the returned cache reports offset 10000 over an 8192-wide buffer; the next update builds a mask of width 10001 against 8193 keys (SDPA shape mismatch crash) or slice-assigns past the buffer. All current in-repo callers happen to guard, so this is a latent public-API foot-gun rather than a live bug.

**`Libraries/MLXLMCommon/Batching/BatchQuantizedKVCache.swift:184` — Cache quantizes with default .affine mode but attention dequantizes with self.mode — non-affine mode silently corrupts**

initQuant (line 101) and updateQuantized (lines 184-185) call `quantized(_, groupSize:bits:)` without the `mode:` argument (MLX 0.31's quantized() takes a mode; cf. SwitchLayers.swift:298-299 which passes it), so storage is always affine-encoded. But attentionWithCacheUpdate (AttentionUtils.swift:63-65) passes `quantizedKVCache.mode` into quantizedScaledDotProductAttention, whose quantizedMM calls decode with that mode (KVCache.swift:1979-1983, 2047-2051). A cache constructed with mode != .affine therefore encodes affine and decodes with another scheme. The same asymmetry pre-exists in the single QuantizedKVCache on main (KVCache.swift:896, 990-991 — #230 threaded mode through attention but not through quantization), and no in-repo [...]

*Failure:* `BatchQuantizedKVCache(leftPadding: [0], mode: .mxfp4)` (or any future non-affine mode): updateQuantized packs K/V with affine layout, quantizedMM interprets the packed words as mxfp4 — attention output is numerical garbage for every row, with no error raised.

**`Libraries/MLXLMCommon/Batching/BatchQuantizedKVCache.swift:541` — finalize() rolls the entire step-rounded buffer instead of the ..<_idx head slice used by BatchKVCache**

BatchKVCache.finalize deliberately slices to `[..., ..<_idx, ...]` before dynamicRoll (BatchKVCache.swift:544-547), with a comment explaining the buffer is step-rounded and rolling the whole width wraps the spare tail. BatchQuantizedKVCache.finalize (lines 539-543) rolls the full-width triples. I verified by index arithmetic that this is functionally masked-safe: for a row with shift r, the wrapped tail lands in [0, r) which the simultaneous `leftPadding += padding` (line 545) masks, real data lands at [leftPad+r, _idx), and the displaced right-pad garbage lands at [_idx, _idx+r), which updateQuantized either overwrites (writes start at prev = _idx) or drops at the next grow (line 169's `..<prev` slice), and every reader (state:231, [...]

*Failure:* Ragged cached-prompt prefill of 10 tokens into a fresh cache (buffer width 256): finalize gathers 3 x [B,H,256,packed] arrays instead of 3 x [B,H,10,packed] — up to ~25x wasted roll work per layer per prefill cycle; any future reader that consumes buffer columns beyond _idx (as the rotating sibling does) would observe relocated right-pad garbage at [_idx, _idx+r).

**`Libraries/MLXLMCommon/Batching/BatchRotatingKVCache.swift:678` — extend() does not validate maxSize/keep compatibility between the two caches (merge() does)**

merge() preconditions equal maxSize and keep across sources (lines 866-874), but extend() only preconditions that a non-empty `other` is prefilled (line 679-684). If `other` has a different maxCacheSize, the merged rows silently inherit self's window: pad() aligns buffers by physical width and `self._scalarOffset = max(...)` (line 745), after which trims (`_idx - maxCacheSize + 1`) and mask spans are computed with self's window for rows built under another window — wrong sliding-window truncation with no diagnostic. Unreachable through the engine (all caches come from one factory or the same model's fromSingle), but extend/extendBatched is public API and extendBatched already runtime-type-checks, so the window check is the natural [...]

*Failure:* Caller extends a maxSize-4096 batch with a BatchRotatingKVCache built with maxSize 1024 (e.g. caches from two differently-configured engines): the 1024-window rows are subsequently trimmed/rotated with the 4096 window, retaining or masking the wrong key spans with no precondition failure.

**`Libraries/MLXLMCommon/Batching/BatchRotatingKVCache.swift:691` — extend() adopt branch silently drops self's row metadata when self is empty-with-rows and other is populated**

The both-empty branch (lines 687-690) deliberately concatenates leftPadding/batchOffsets "so admitted rows survive until their first prefill", acknowledging that a cache can carry rows (batchSize > 0) with keys == nil. But the adjacent branch (lines 691-699, self.keys == nil && other.keys != nil) overwrites self.leftPadding/batchOffsets with other's arrays, discarding any such pre-prefill rows on self with no precondition (the precondition at 679-684 only constrains `other`). Engine flows never hit it (DecodeBatch caches are always prefilled, and fromSingle on a live iterator's caches is populated), but the asymmetry contradicts the both-empty branch's stated contract and would silently shrink the batch if an empty-with-rows receiver [...]

*Failure:* cacheA = BatchRotatingKVCache(maxSize: 16, leftPadding: [0, 0]) (2 admitted rows, never prefilled); cacheA.extend(other: populatedBatch1Row) -> cacheA.batchSize becomes 1 and the two admitted rows' leftPadding/batchOffsets vanish without any precondition failure; a later per-row prefill then runs against the wrong row set.

**`Libraries/MLXLMCommon/Batching/BatchRotatingKVCache.swift:836` — extract() hardcodes step "256" in the serialized metaState and fromSingle drops the source's step**

extract() serializes the RotatingKVCache metaState as [keep, maxCacheSize, "256", offset, idx] (line 835-837), ignoring both the batch cache's `step` property (line 177) and any custom step the row's original single cache had; fromSingle (line 958-988) likewise never copies `cache.metaState[2]` into `batchCache.step`. A single RotatingKVCache created with a non-default step (the public init exposes step:, and metaState round-trips it at KVCache.swift:711/734) that is upgraded via fromSingle and later extracted comes back with step silently reset to 256, changing its buffer growth granularity. All in-repo constructions use the default 256, so this is behavioral drift only for custom callers.

*Failure:* RotatingKVCache(maxSize: 8192, step: 512) is upgraded via fromSingle, decoded batched, then extracted on finish: the returned cache's metaState says step=256, so subsequent single-stream decode grows the buffer in 256-token increments instead of 512 — no wrong tokens, but the serialized cache no longer round-trips the original configuration.

**`Libraries/MLXLMCommon/Batching/BatchRotatingKVCache.swift:1059` — makeMask builds a decode-shaped (ring-rolled) mask for width-1 prepared prefill chunks that update() routes to updateConcat (temporal layout)**

update() (line 240) routes a 1-token update with `_lengths != nil` to updateConcat, which temporalOrders/trims and returns a LINEAR buffer. But makeMask has no knowledge of `_lengths`: for n==1 with `_idx >= maxCacheSize` it sets `isRotated = true` (line 1059), decrements effectiveLeftPadding by 1 and circularly rolls the mask by `currentIdx+1` (lines 1069-1084) — decode/ring semantics. The lp arithmetic happens to coincide (mask's `_idx - ms + 0` plus the isRotated `-1` equals updateConcat's `_idx - ms + 1` trim), but the roll shifts every row's left-padding block one column to the right of the true garbage block in the temporally-ordered buffer updateConcat returns. Additionally, makeMask never accounts for the pending `dynamicRoll` [...]

*Failure:* Direct API (or future engine change) constructs BatchRotatingKVCache(maxSize: 4000, leftPadding: [p>0, 0]), prepares a right-padded prefill of widths 2048+2048+1: at the width-1 chunk, rows with positive effective padding get the pad-mask block rolled to columns [1, lp+1) while the real garbage occupies [0, lp) of the temporal buffer -> one garbage key attended and one valid key masked for that row's query; today only pad-query rows hit it [...]

**`Libraries/MLXLMCommon/Batching/BatchedCache.swift:310` — All SSM/hybrid cache topologies (9+ model families) are rejected wholesale from batching — including the five models that do thread real SSM masks — so concurrent requests silently serialize**

`makeBatchedCacheFactory` throws `unsupportedCacheTopology` for any exact `MambaCache`/`ArraysCache` (BatchedCache.swift:310-322) and recursion into `CacheList` propagates the rejection (276-287), even though full batched SSM machinery exists and is tested (ArraysCache filter/extend/extract/makeMask at KVCache.swift:1274-1391, the `ArraysCache: BatchedCache` conformance at BatchedCache.swift:170-197, and `BatchedCacheList` with masking-child delegation at 44-166). The stated reason is that SOME models hard-code a nil SSM mask (GraniteMoeHybrid.swift:20; Jamba/LFM2/BaichuanM1 pass no mask), but the blanket type-level rejection also excludes the mask-safe models the comment itself names (Qwen3Next.swift:458, Qwen35.swift:537, [...]

*Failure:* A user deploys Qwen3Next (which threads a correct SSM mask) with the scheduler and sends 8 concurrent requests expecting batched throughput; every request runs one-at-a-time behind the others (request 8 waits for 7 full generations), with no error, warning, or API signal distinguishing this from a batched model.

**`Libraries/MLXLMCommon/Batching/DecodeBatch.swift:35` — allTokens means prompt+generated for inserted rows but generated-only for adopted rows; doc says 'all produced tokens'**

BatchStepResult.allTokens is documented as 'All produced tokens for this row' (line 35), but freshly inserted rows carry the full prompt in it (PrefillBatch appends the prompt to `tokens` at PrefillBatch.swift:118-120/191-192 before handing off), while adopted rows carry generated-only history (InferenceScheduler.swift:1275). The same asymmetry applies to FinishedRowCache.allTokens, which the doc positions as the prompt-cache write-back key. EngineDriver compensates with a per-record `tokensIncludePrompt` flag and key reconstruction (EngineDriver.swift:50-54, 62-73, 522-527), but a direct engine user keying a prompt cache or counting generated tokens from engine.next(capturingFinalCaches:) output gets prompt-contaminated counts/keys [...]

*Failure:* A power user drives engine.next(capturingFinalCaches: true) directly and stores FinishedRowCache.finalCache keyed by allTokens: for an adopted row the key omits the prompt, so a later lookup for the generated-only suffix reuses KV state that actually encodes a different prompt at those positions, producing corrupt continuations.

**`Libraries/MLXLMCommon/Batching/DecodeBatch.swift:302` — extend() discards the appended batch's fallbackSampler for rows without an explicit sampler**

extend appends `other.samplers` verbatim (line 302); after the merge, nil sampler slots resolve through the RECEIVING batch's `fallbackSampler` (step(), line 374), silently switching those rows' sampling policy when the two fallbacks differ. The `fallbackIsGreedy` AND at line 311 only prevents the argMax fast path; it does not restore the intended fallback. Inside BatchGenerationEngine this is currently unobservable (every batch is built with the engine's greedy defaultSampler, lines 96, 380, 422), but DecodeBatch/extend are public API and PrefillBatch/DecodeBatch accept arbitrary fallbacks, so direct users (or a future non-greedy engine default) hit it. Already fixed on the cb-2-engine tip (commit 76114e8 binds [...]

*Failure:* A DecodeBatch built with fallbackSampler = makeRowSampler(temperature: 0.8) and per-row samplers nil is passed to adoptActiveBatch while a greedy-fallback batch is running: after extend, its rows silently decode greedily instead of sampling.

**`Libraries/MLXLMCommon/Batching/EngineDriver.swift:273` — Write-back path bypasses PromptCachePolicy / canUsePromptCache gating (enforced on fetch only)**

The fetch side is gated by `canUsePromptCache` — media-free input, kvBits nil, `defaultPromptCachePolicy == .exact` (LRUPromptCache.swift:313-322) — but the insert side is not: EngineDriver.admit sets `writeBackToPromptCache: request.promptCache != nil` (line 273) and `writeBack` inserts unconditionally (lines 502-534); the single path likewise inserts whenever a cache is attached (InferenceScheduler.swift:857-869) without re-checking policy. Quantized (kvBits) caches happen to be rejected downstream by `isCacheCompatible`, and VLMs never reach the scheduler (ModelContainer.swift:235-247), so today the only leak is the policy hook itself: a model declaring `.disabled` would still have its KV state copied, eval'd, and stored on every [...]

*Failure:* A model overrides `defaultPromptCachePolicy = .disabled` (the hook PR3 itself introduces, LanguageModel.swift:12-20). Requests with a promptCache attached still deep-copy and store full KV snapshots at every completion via EngineDriver.writeBack/runSingle write-back, consuming maxBytes budget and evicting other models' useful entries, while fetch never returns them.

**`Libraries/MLXLMCommon/Batching/EngineDriver.swift:468` — Multi-token stop sequences: prefix tokens are streamed with no holdback and never retracted; only the final matching token is suppressed**

DecodeBatch.next() emits every token immediately, including tokens the StopSequenceMatcher is holding as a pending partial match (union DecodeBatch.swift:244-254: non-finished rows get a normal BatchStepResult even when the new matcher state carries a non-empty pendingMatch). When the sequence completes, only the final token is routed to `processStopToken` (EngineDriver.swift:467-468), which decoded mode drops — the previously streamed prefix tokens of the stop sequence are never retracted, and `deliver` ignores the `matchedSequence` field that would allow trimming. Upstream single-stream semantics (StopStringFilter, Evaluate.swift:2163-2185) hold back any suffix that could begin a stop string and trim the entire match. Today the in- [...]

*Failure:* Direct engine user configures stop sequence [198, 9906] ('\nHello'): the row samples 198 → delivered and detokenized into the stream; next step samples 9906 → finishReason=.stop, token 9906 suppressed — the consumer has already received the '\n' that the equivalent single-stream stop-string filter would have withheld and trimmed.

**`Libraries/MLXLMCommon/Batching/EngineDriver.swift:482` — Batched completion info hardcodes promptTime: 0 -> promptTokensPerSecond is +infinity**

EngineDriver.deliver builds GenerateCompletionInfo with `promptTime: 0` (482), and finishUpgradedSingle does the same (InferenceScheduler.swift:1335). GenerateCompletionInfo.promptTokensPerSecond computes `Double(promptTokenCount) / promptTime` (Evaluate.swift:1990-1991), so every batched or upgrade-finalized request reports promptTokensPerSecond == +inf (promptTokenCount > 0 is guaranteed — empty prompts are rejected at insert). The single path reports the real `it.promptPrefillTime` (InferenceScheduler.swift:843), so metrics silently switch from real numbers to infinity the moment a second request causes batching. Any dashboard/UI dividing or displaying this value gets inf/NaN.

*Failure:* Client displays `info.promptTokensPerSecond` after each generation. Solo request: '412 tok/s'. Same request while another is in flight (batched path): 'inf tok/s' — and JSON-encoding the value throws (Double.infinity is not JSON-representable) in telemetry pipelines.

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:643` — Synchronous GPU prefill runs inside the scheduler actor, blocking all scheduler messages (submit/cancel/upgrade) for the full prompt-prefill duration**

startSingle constructs the TokenIterator on the actor (lines 643-648); TokenIterator.init runs the full (chunked) prompt prefill synchronously, which for a long prompt is seconds of GPU-blocking work executed on a cooperative-pool thread while the InferenceScheduler actor is held. Every queued actor message — a second submit (which is what would trigger the upgrade), a cancel from a dropped stream, setMaxBatchSize — waits behind it. This is a deliberate trade (comment lines 581-589: surface prefill errors to the caller), but it means the auto-upgrade path cannot even begin until the first request's prefill completes, and stream-cancellations are unserviceable during it. The same pattern exists on the driver: stepOnce [...]

*Failure:* iPhone app: user submits a 30k-token prompt (multi-second prefill on the scheduler actor), immediately regrets it and cancels the stream. The cancel Task cannot run until prefill finishes; a second user request submitted meanwhile also waits the full prefill before it can even be routed — UI-visible multi-second unresponsiveness of everything touching the scheduler.

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:906` — BatchQuantizedKVCache (all of cb-5) has no reachable producer through the scheduler; kvBits users additionally lose all concurrency**

The engine validates topology from `model.newCache(parameters: cacheParameters)` (BatchGenerationEngine.swift:98-100), but `ensureDriver` constructs the engine without passing cacheParameters (InferenceScheduler.swift:904-913), so the probe is always `newCache(parameters: nil)` — which for every in-repo model returns KVCacheSimple/RotatingKVCache/SSM types, never QuantizedKVCache. Requests with `kvBits` are diverted to serial single-stream by `parametersAreBatchable` (372-374, queued at 521-527/961-968), and the only newCache that emits QuantizedKVCache (FalconH1.swift:806-823, with kvScheme/kvBits + quantizedKVStart==0) wraps it in a CacheList next to a MambaCache, which the factory rejects. So the entire BatchQuantizedKVCache [...]

*Failure:* A memory-constrained deployment sets kvBits=4 on every request expecting the advertised quantized batched decoding; the scheduler quietly serializes all traffic (state .pendingUpgrade queueing), and the quantized batched cache path never executes in production.

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:934` — Batched rows never stop on tokenizer.unknownTokenId, while the scheduler's own single path does**

The scheduler's single-request loop stops with `.stop` when `token == unknownTokenId || stopTokenIds.contains(token)` (InferenceScheduler.swift:789, mirroring the historical synchronous loop). The batched engine's stop matcher is built from `eosSequences` = `stopTokenIds(...)` only (lines 930-935: configuration.eosTokenIds + tokenizer.eosTokenId + convertible extraEOSTokens); `unknownTokenId` is never added, and DecodeBatch has no other stop check besides maxTokens. The same request therefore terminates on an unk token when it happens to run single but keeps generating when batched (including after a single→batch upgrade, whose matcher is `defaultStopMatcher()` built from the same eosSequences, line 1243).

*Failure:* A degenerate/quantized model samples the unk token: solo request → stream ends immediately with `.stop`; same request while another request is active (batched) → unk is streamed as text (often decoding to garbage) and generation continues to EOS/maxTokens (Int.max default on the scheduler).

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:1032` — Greedy argMax fast path is unreachable for all scheduler workloads: rowSampler(for:) always installs a non-nil sampler (greedySampler for temperature<=0)**

DecodeBatch.step gates its batched argMax fast path on `samplers.contains { $0 != nil }` (DecodeBatch.swift:340-342 union / 375-377 cb-2 tip). But `InferenceScheduler.rowSampler(for:)` (line 1032-1040) always returns a non-nil RowSampler — `makeRowSampler` returns an explicit `greedySampler` closure for temperature<=0 (RowSamplers.swift:77-79) — and it is installed for every submitted row (lines 984, 1241) and every adopted row. So `anySampler` is always true and every decode step of a pure-greedy workload takes the slow path: a full-vocabulary `logSumExp` over [B, vocab] (DecodeBatch.swift:355 / 395-396 tip), B separate row slices, B one-row argMax kernel launches, and a `concatenated` — instead of one batched argMax (line 383/443). [...]

*Failure:* Serve 8 concurrent temperature=0 requests through InferenceScheduler on a 150k-vocab model: every decode step executes an extra [8,150k] logSumExp reduction plus 8 per-row slice+argMax kernels and a concat instead of a single [8,150k] argMax — a persistent multi-percent decode slowdown; the fast path never engages for the lifetime of the process.

**`Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:1275` — Single→batch upgrade seeds penalty processors with generated-only history, dropping prompt tokens from the penalty context window**

The upgrade path passes `tokens: liveGeneratedTokenIds + [liveCurrentToken]` (generated-only) to `makeAdoptedBatch` (line 1275). `makeAdoptedBatch` drops the seed and DecodeBatch seeds each freshly-built penalty processor with `processors[i]?.prompt(tokens[i])` (DecodeBatch.swift:172-175 union / 200-203 tip), then feeds the seed via didSample in the priming step. The single stream it replaces had called `processor.prompt(full prompt)` (Evaluate.swift:659) so its TokenRing held the last `contextSize` (default 20) tokens of prompt+generated. After an upgrade with fewer than ~20 generated tokens, the adopted row's ring contains only the generated tokens — prompt tokens that the single stream would still be penalizing are no longer [...]

*Failure:* Request with repetitionPenalty=1.3, 100-token prompt, upgraded to the batch after 5 generated tokens: single-stream would penalize the last 15 prompt tokens + 5 generated; the adopted row penalizes only 6 tokens, so the next sampled tokens differ from what the un-upgraded stream would have produced (upgrade becomes user-visible as a mid-stream behavior change).

**`Libraries/MLXLMCommon/Batching/LRUPromptCache.swift:95` — LRU bookkeeping is O(entries x prompt-length) full token-array comparisons; trie costs one node + Dictionary per token**

`CacheOrder` stores `(RootKey, [Int])` pairs in plain arrays; `remove` (lines 94-105, called by every `touch` on every hit and by superseding inserts) does `firstIndex(where: { $0.tokens == tokens })` — a linear scan where each miss compares full token arrays element-by-element. For agentic prompts (10k-100k tokens) with several entries this is O(entries x tokens) CPU on the scheduler actor per fetch, plus `Array(tokens[...])` copies for keys. The trie itself allocates one `TrieNode` class instance with its own `[Int32: TrieNode]` Dictionary per token per entry (lines 66-70, 1018-1031) — roughly 100+ bytes/token, i.e. tens of MB of pure pointer structure for a handful of long prompts, and O(prompt) dictionary hops per lookup. Hashing [...]

*Failure:* 10 cached entries of ~50k tokens each: every fetch's `touch` scans up to 10 entries comparing ~50k Ints each (~500k comparisons) inside the state lock on the InferenceScheduler actor, adding measurable per-request latency before prefill even starts.

**`Libraries/MLXLMCommon/Batching/LRUPromptCache.swift:368` — save() rewrites every live entry's full tensor file on each call, decodes all sidecars twice, and never cleans crashed saves' *.tmp files**

There is no dirty tracking: every `save(to:)` re-serializes every disk-persistable entry's complete KV tensors (lines 368-412) even if unchanged since the last save — for periodic autosave in an agent loop that is GBs of redundant writes per cycle. `removeStaleSidecars` (lines 463-465) and `enforceDiskBudget` (lines 689-691) each re-read and JSON-decode every sidecar in the directory (the budget pass constructs a new JSONDecoder per file), and sidecars embed the full token array as JSON, so a 100k-token prompt's sidecar is ~1 MB decoded up to three times per save/load cycle. Finally, a crash between writing the temp pair and the final moves leaves `entry_*.UUID.tmp.safetensors/.tmp.json` orphans that nothing ever deletes: the `defer` [...]

*Failure:* An app autosaves the cache every N turns and is force-quit occasionally: unchanged multi-GB entries are rewritten on every save (I/O + wear), and each crash strands a full-size orphan .tmp.safetensors in the directory; over weeks the directory grows unboundedly with files invisible to maxDiskBytes enforcement.

**`Libraries/MLXLMCommon/Batching/LRUPromptCache.swift:510` — load() eagerly loads every persisted tensor oldest-first, then LRU-evicts the overflow; each survivor is deep-copied twice**

`loadSynchronously` sorts sidecars ascending by lastUsed (line 508) and calls `loadPromptCache` + `insertCache` for each (lines 510-532). With more persisted entries than `maxSize` (default 10) or `maxBytes`, the oldest entries' safetensors are fully loaded and materialized only to be immediately evicted by the newer ones inserted after them (insert-time eviction, lines 1047-1058). Additionally, each loaded `[KVCache]` goes through the public `insertCache`, whose `materializedCopy` re-copies and re-evals arrays that were just freshly loaded from disk and are aliased nowhere. There is no early stop once the newest `maxSize`/`maxBytes` worth of entries has been admitted (which descending iteration plus a budget counter would allow).

*Failure:* A shared persistence directory has accumulated 60 sidecars from prior sessions; `load(from:)` on app startup reads and materializes all 60 multi-hundred-MB tensor files (minutes of I/O + GPU traffic), of which 50 are evicted before load returns, and the 10 survivors were each copied twice.

**`Libraries/MLXLMCommon/Batching/LRUPromptCache.swift:618` — GPU eval and full deep copies performed while holding OSAllocatedUnfairLock (fetch and save paths) — blocks actor executor threads for the duration of GPU work**

fetchNearestCacheResult enters `state.withLockUnchecked` (line 237) and, inside the lock, State.fetchNearestCache calls LRUPromptCache.deepCopy (lines 950, 966, 985) → materializedCopy → `eval(arrays)` (line 762): a synchronous GPU materialization of an entire multi-layer KV cache under an os_unfair_lock. The save path is worse: diskSnapshotsAndOwnedKeys (lines 588-599) deep-copies and eval()s EVERY live entry inside the lock (line 618, deliberately per the comment). os_unfair_lock is not async-aware; any concurrent locker blocks its thread outright. The intended concurrent users are the InferenceScheduler actor (fetch at InferenceScheduler.swift:623) and the EngineDriver actor (writeBack insert at EngineDriver.swift:528): whichever [...]

*Failure:* App calls promptCache.save(to:) while a batch is decoding. The persistence queue thread takes the unfair lock and evals dozens of cache entries; the EngineDriver finishes a row and calls insertCache for write-back, blocking the driver's executor thread on the lock for the entire save — token streaming for all live requests stalls, and other actors scheduled on that cooperative thread stall with it.

**`Libraries/MLXLMCommon/Batching/LRUPromptCache.swift:1033` — Byte accounting undercounts retained memory for rotating entries normalized down by trim**

When an over-covered linear RotatingKVCache is inserted, `normalizeRotating` (lines 788-804) only decrements `offset`/`idx` via `trim`; the materialized physical buffer keeps its original dim(2) rows. The stored bytes at line 1033 are computed from `$1.state`, and RotatingKVCache's state getter slices to `offset` (KVCache.swift:686-697), so `nBytes` counts only the logical rows while the entry actually retains the full pre-trim buffer (the trimmed tail's memory is held for the entry's lifetime). `maxBytes` eviction and the persisted `byteCount` therefore understate real footprint. Impact is bounded because in-tree write-backs always insert with coverage exactly equal to the key length (offset == tokens.count, no trim), so this only [...]

*Failure:* An external caller snapshots a chat session mid-stream and inserts its rotating caches keyed by the prompt only (offset 4096 > tokens.count 1024): each layer stores a 4096-row buffer but accounts 1024 rows; with several such entries, resident memory exceeds `maxBytes` by up to 4x without triggering eviction.

**`Libraries/MLXLMCommon/Batching/PrefillBatch.swift:154` — PrefillBatch.prompt forces a synchronous eval barrier after every prefill chunk, unlike the single path's asyncEval pipelining**

The chunk loop calls `eval(cache.innerState())` after every `prefillStepSize` (default 2048) chunk (PrefillBatch.swift:152-155, same at cb-2 tip), a hard CPU↔GPU sync per chunk. The single-stream default prefill uses `asyncEval(cache)` per chunk and one final `eval(cache)` (LLMModel.swift:32-41), explicitly so 'the CPU builds chunk N+1's graph while the GPU evaluates chunk N'. After fix 1321949 removed the per-chunk `advanceBatched(n)` CPU-side metadata mutation, there is no remaining ordering reason the batched loop cannot use the same asyncEval-per-chunk + single-final-eval pattern (the ragged-length masking is all in-graph). For long prompts (e.g. 16k tokens = 8 chunks) batched prefill serializes graph construction and execution at [...]

*Failure:* Admit a batch with a 16k-token prompt: 8 chunk boundaries each stall the CPU until the GPU fully drains before building the next chunk's graph; the same prompt through the single path overlaps these, so batched prefill latency is measurably worse than single prefill for identical work.

**`Libraries/MLXLMCommon/Batching/StopSequenceMatcher.swift:126` — Longest-prefix-only matching can miss a stop sequence embedded inside a longer candidate's live prefix**

match() extends `pendingMatch` with the new token and keeps the LONGEST suffix that is a valid trie path (lines 124-154), never checking whether a SHORTER suffix completes a different stop sequence. With states ["normal": [([7,8,9], nil), ([8], nil)]], the stream 7,8 leaves pendingMatch=[7,8] (a live prefix of [7,8,9]) and never reports the completed [8] stop; if the next token is not 9, the [8] stop is lost entirely and generation runs on. Proper Aho-Corasick failure links (or checking every suffix for terminals) would stop at 8. Unreachable via the scheduler (its sequences are all single-token, InferenceScheduler.swift:934, and the terminal-at-node check at line 127 handles a stop that is itself a prefix of another), but the [...]

*Failure:* engine built with eosTokens [[7,8,9],[8]]; model emits 7 then 8 then 5: no stop fires at 8 (pending [7,8] shadows it) and the reset at token 5 discards it, so the row generates past its stop token until maxTokens.

**`Libraries/MLXLMCommon/ChatSession.swift:625` — Scheduler path lacks the non-scheduler path's Task.isCancelled guard before tool dispatch**

The non-scheduler restart loop dispatches tools only `if let toolDispatch, !pendingToolCalls.isEmpty, !Task.isCancelled` (ChatSession.swift:890-892). streamThroughScheduler dispatches unconditionally at lines 625-631 and then restarts `model.generate` (line 619) even when the consuming task has been cancelled via `continuation.onTermination = { task.cancel() }` (lines 908-910). A side-effectful toolDispatch (network call, file write) can therefore execute after the consumer walked away, followed by a full prefill of a new scheduler request whose first `emit` immediately returns false.

*Failure:* Caller breaks out of `streamResponse` (stream terminated) just as a toolCall event was already delivered: the session still executes the tool (e.g. POSTs an order to an API) and burns a full-conversation prefill for a dead consumer; the non-scheduler path skips both.

**`Libraries/MLXLMCommon/KVCache.swift:2044` — quantizedScaledDotProductAttention uses non-precise softmax, diverging from Python mlx-lm (precise=True) and the fused SDPA path**

Line 2044 computes `softmax(scores, axis: -1)` in the score dtype (typically bf16/fp16). Python mlx-lm's quantized_scaled_dot_product_attention uses `mx.softmax(scores, axis=-1, precise=True)`, and the unquantized path (MLXFast.scaledDotProductAttention, AttentionUtils.swift:69) accumulates its softmax at higher precision inside the fused kernel. The omission is pre-existing on main (main KVCache.swift:1957), but PR5 makes this function the sole attention path for all quantized batched rows, so every batched quantized decode inherits bf16 softmax accumulation over the full (padded) key length.

*Failure:* Long-context quantized batched decode in bf16 (8-bit mantissa): softmax normalization over thousands of keys accumulates rounding error, measurably degrading logits versus the same request on the single-stream fused-SDPA path — a silent quality regression rather than a crash.

**`Libraries/MLXLMCommon/ModelContainer.swift:270` — Sharing one InferenceScheduler across two ModelContainers silently runs the second container's requests on the first container's model**

`attachSchedulerIfNeeded` returns early when `scheduler.isAttached` (ModelContainer.swift:271), and `InferenceScheduler.attach` is first-wins (`if self.context == nil`, InferenceScheduler.swift:116-123). `InferenceScheduler` is a public type the caller constructs and passes in (ModelContainer.swift:87-95), so nothing prevents installing the same instance into two containers; the second container's `generate` then submits into a scheduler bound to the first container's model, tokenizer, and prompt cache — no error, no assertion. A precondition (scheduler already attached to a different context → throw/fatalError) or having the container create its own scheduler from a plain `Configuration` value would remove the footgun.

*Failure:* `let s = InferenceScheduler(); let a = ModelContainer(context: llamaCtx, scheduler: s); let b = ModelContainer(context: qwenCtx, scheduler: s)`. After one generate on `a`, every `b.generate(...)` decodes with Llama weights and Llama's tokenizer while the caller believes they are talking to Qwen — silent wrong output.

**`Libraries/MLXLMCommon/ModelContainer.swift:299` — generateBatched throws .schedulerBusy for 'no scheduler installed' (misleading error semantics) and uses `try` on non-throwing attach**

`guard let scheduler else { throw BatchedGenerationError.schedulerBusy }` (ModelContainer.swift:298-300) and `InferenceScheduler.submitOne`'s unattached guard (InferenceScheduler.swift:422-424) both reuse `.schedulerBusy`, whose doc reads 'The scheduler was busy with an incompatible workload' (GenerationRequest.swift:66-67) — a caller diagnosing why explicit batching fails on a plain container gets a nonsensical error; the test even canonizes it (ModelContainerBatchingTests.swift:17-25). A dedicated `.noScheduler`/`.notAttached` case would cost nothing. Additionally `try await attachSchedulerIfNeeded(scheduler)` (ModelContainer.swift:238 and 310) applies `try` to a non-throwing async func, which produces a 'no calls to throwing [...]

*Failure:* Developer calls `container.generateBatched(requests)` on a container built via `loadContainer` (which can never install a scheduler): the thrown error says the scheduler was 'busy', sending them to debug concurrency/contention instead of the actual cause (no scheduler exists).
## Appendix B — Pre-existing review-thread items verified still open

- **[critical]** `Libraries/MLXLMCommon/ChatSession.swift:660` — PR21 Swift 6 region-isolation compile failure at `Task {` — union tree carries the identical failing pattern (diagnosis + fix)
- **[critical]** `Libraries/MLXLMCommon/ChatSession.swift:629` — Scheduler-path tool restart re-prompts with ONLY the tool result — total conversation context loss
- **[major]** `Libraries/MLXLMCommon/Batching/BatchedCache.swift:384` — No layer refuses ragged batches for scalar-offset models (PaliGemma, Qwen2VL, Mistral3, NemotronLabsDiffusion) — silent output corruption
- **[major]** `Libraries/MLXLMCommon/KVCache.swift:1641` — Batched caches unregistered in prompt-cache serialization: savePromptCache silently writes them as "KVCache", loadPromptCache then hits fatalError
- **[major]** `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:978` — Third request during .upgrading overwrites state with .batched; upgrade epoch has no sentinel, so later transitions strand streams
- **[major]** `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:373` — kvScheme-only quantization not screened by parametersAreBatchable -> upgrade truncates the first request mid-generation
- **[major]** `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:430` — ModelConfiguration.effectiveStopStrings never enforced on any scheduler path
- **[major]** `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:117` — attach() silently keeps the first ModelContext; a scheduler reused across containers generates with the wrong model
- **[major]** `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:1064` — startBatchLoop spawns a busy-spinning task chain for every submit while a drain is already running
- **[major]** `Libraries/MLXLMCommon/ChatSession.swift:625` — Scheduler path dispatches first toolCall and breaks the stream — parallel tool calls, trailing chunks and .info dropped; row cancelled mid-decode
- **[major]** `Libraries/MLXLMCommon/Batching/LRUPromptCache.swift:310` — ChunkedKVCache admitted by `is KVCacheSimple` subclass check, then fatalError in materializedCopy metaState write-back
- **[minor]** `Libraries/MLXLMCommon/Batching/PrefillBatch.swift:247` — PrefillBatch.extend is dead-by-construction: precondition requires both batches to have zero rows, so every legal call is a no-op
- **[minor]** `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:1275` — Adopted (upgraded) row's penalty processor is rebuilt from generated tokens only — prompt context lost
- **[minor]** `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:634` — Prompt-cache hit builds the iterator from the uncached remainder — penalty processors lose the cached prefix
- **[minor]** `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:934` — Batched rows never stop on tokenizer.unknownTokenId; upgrade-handoff EOS check misses it too

## Appendix C — Claims investigated and refuted (no action needed)

- `Libraries/MLXLMCommon/KVCache.swift:320` — Legacy createAttentionMask([KVCache]?) overload returns nil for t == 1, skipping batched left-padding masks on every decode step: The code-level observations in the claim are individually accurate, but the claimed failure scenario is unreachable because Qwen3VL cannot execute a single forward pass under BatchGenerationEngine at all — it crashes with fatalError before any mask is [...]
- `Libraries/MLXLMCommon/Batching/BatchRotatingKVCache.swift:962` — fromSingle accepts keep>0 sources, bypassing the factory's keep-prefix rejection; soundness relies on distant scheduler gates: The claim's code description is accurate but its failure scenario is unreachable through any supported path, and the claim itself concedes this. Verified facts: fromSingle (BatchRotatingKVCache.swift:958-962) does construct keep>0 batch caches without [...]
- `Libraries/MLXLMCommon/Batching/BatchQuantizedKVCache.swift:403` — extend() adoption branch silently discards admitted-but-unprefilled receiver rows (and stale _rightPadding is never reconciled): The claimed failure scenario is blocked at its very first step by the precondition the claim itself cites. Step 1 of the scenario — "cacheA (empty) extended with empty cacheB carrying 2 admitted rows -> receiver has batchSize 2, keys nil" — is [...]
- `Libraries/MLXLMCommon/Batching/PrefillBatch.swift:156` — Chunked prefill double-advances SSM cache lengths, corrupting recurrent state on multi-chunk ragged prompts: The double-advance mechanics are real and the cb-2 fix (1321949) is indeed missing from the union tree: PrefillBatch.prompt calls cache.advanceBatched(n) per chunk (Batching/PrefillBatch.swift:156-158); ArraysCache.advanceBatched maps to advance(n) [...]
- `Libraries/MLXLMCommon/Batching/LRUPromptCache.swift:207` — Exact-hit contract (empty remainder, all tokens 'reused') is undocumented for external callers of the public fetch API: The claim's central premise — that the exact-hit empty-remainder behavior is undocumented for external callers — is contradicted by the code. The public `fetchNearestCache` doc comment explicitly states the behavior at LRUPromptCache.swift:203-204: "1. [...]
- `Libraries/MLXLMCommon/ModelContainer.swift:273` — attachSchedulerIfNeeded uses SendableBox to DUPLICATE (not transfer) the ModelContext, silently voiding SerialAccessContainer's exclusivity guarantee: The mechanical claim is accurate: attachSchedulerIfNeeded (stack-merge Libraries/MLXLMCommon/ModelContainer.swift:266-280) copies the ModelContext struct inside `context.read` ("let copy = context; return SendableBox(copy)") and hands the copy to the [...]
- `Libraries/MLXLMCommon/Batching/InferenceScheduler.swift:552` — Design deliberately permits concurrent MLX graph evaluation from two executors; safety rests on undocumented MLX runtime thread-safety, not on the code's own serialization story: The claimed concurrency windows are real and reachable: (a) a 3rd request routed at `.upgrading` (InferenceScheduler.swift:549-552, entered at :1114 while `upgrade()` is suspended at :1116) starts EngineDriver prefill while the single task may still be [...]
- `Libraries/MLXLMCommon/Batching/PromptCaching.swift:74` — PromptCaching: Sendable protocol trades non-Sendable [KVCache] through nonisolated requirements — deep-copy safety is convention, not compiler-checked: The claim's description of the code is factually accurate but describes a design property, not a bug. PromptCaching is Sendable (PromptCaching.swift:74) and its nonisolated synchronous requirements do pass non-Sendable [KVCache] in (insertCache, lines [...]
