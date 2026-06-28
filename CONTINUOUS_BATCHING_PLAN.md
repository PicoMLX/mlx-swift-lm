# Continuous Batching PR Stack — Stabilization Plan (v2)

Supersedes the v1 plan. Changes in this revision: retarget PR1 to the rebased branch,
rebase the whole stack on current `main`, add fork CI, settle naming, verify
forward-compatibility for non-LLM modalities, and define the review-loop working process.

---

## 0. Decisions resolved in this revision

These answer the open questions raised against v1.

1. **PR #17 (finfo.min) — do not touch, do not integrate into PR1–4.**
   #17 is in review upstream (ml-explore) and depends on an Apple mlx-swift PR. The
   `Float.leastNormalMagnitude` → most-negative-finite fix lives in
   `quantizedScaledDotProductAttention`. **PR1–4 never exercise the quantized SDPA path**
   (PR1's factory rejects quantized caches), so the core stack does **not** need #17.
   The **only** consumer is the quantized-cache follow-up **PR #12**, whose batched bool-mask
   correctness (and one of its tests) depends on the fix. Resolution: assume #17 lands on
   `main` first; **PR #12 rebases on the updated `main` and drops its duplicate finfo hunk**
   rather than carrying its own copy. If #17 slips, #12 stays deferred — acceptable, it is
   already a follow-up. **Net: no CB PR carries the finfo fix.**

2. **PR1 = PR #16 (`claude/sweet-shannon-144odn`), not #8.**
   #16 explicitly supersedes #8 — it is #8 rebased on `main` and reconciled with the
   `ArraysCache` batching lifecycle that `main` grew independently (#331/#323/#230).
   All recent review activity (13 open threads, Jun 23–24) is on #16. **Retire #8 /
   `claude/determined-fermat-cwpdS`.** PR2/PR3/PR12 currently base on the old
   `determined-fermat` tip and must be re-pointed onto #16's tip (then onto `main`).

3. **CI on the fork without touching upstream** — see §2.

4. **Naming** — concrete proposals in §3 (confirm before the rename churn).

5. **Forward-compatibility for Flux / BERT / embeddings / speculative decoding** — verified
   feasible with the current layering; required seams documented in §4. We still do **not**
   build the outer coordinator in this stack.

---

## 1. Branch & merge order

Merge order (each rebased on the previous, all ultimately on current `main`):

1. **CI-enable PR → `main`** (standalone, tiny; §2). Merge first so every CB branch inherits it.
2. **PR1 — caches**: PR #16 `claude/sweet-shannon-144odn`.
3. *(optional)* **PR12 — quantized cache**: `claude/tender-darwin-5jshar`, stacked on PR1,
   rebased on `main` **after #17 lands** (§0.1). Keep as a follow-up unless the team wants
   `kvBits` batching in the first merge.
4. **PR2 — batch engine**: `claude/batched-2-engine`.
5. **PR3 — prompt cache**: `claude/batched-3-promptcache`.
6. **PR4 — scheduler + public API**: `claude/batched-4-scheduler`.

Delete the frozen `claude/batched-stack-base` integration branch; PR4 rebases directly on the
live PR1/PR2/PR3 tips.

---

## 2. Phase 0 — rebase + CI (do this before any fixes)

**2.1 Rebase the entire stack on current `main`.**
`main` is 28 commits ahead of every CB branch except #16 (4 behind). Rebase in dependency
order: PR1(#16) onto `main`, then PR2/PR3 onto PR1, then PR4 onto PR1+PR2+PR3, then PR12 onto
PR1 (and `main`, post-#17). Resolve conflicts in favor of `main`'s upstream-aligned
`ArraysCache`/`kvScheme` implementations (as #16 already did), keeping only the additive
surface the `Batching/` files require. Do the rebase **before** applying review fixes so fixes
are not re-overwritten by stale copies in PR4.

**2.2 Enable CI on the fork — upstream-safe.**
`/.github/workflows/pull_request.yml` gates both jobs (`lint`, `mac_build_and_test`) on:

```yaml
if: github.repository == 'ml-explore/mlx-swift-lm'
```

Change each to an OR clause that adds the fork **without** altering upstream behavior:

```yaml
if: github.repository == 'ml-explore/mlx-swift-lm' || github.repository == 'PicoMLX/mlx-swift-lm'
```

This is inert upstream (the first clause still matches in ml-explore), so it cannot
"accidentally change" upstream CI even if the file is ever PR'd there. Do this as the
standalone CI-enable PR to `main` (item 1.1) so all CB branches inherit it on rebase.

- The **`lint` job** runs on `ubuntu-22.04` (swift-format + pre-commit) — fork-safe and
  directly fixes the recurring "authored on Linux, couldn't run swift-format" gap. Enable it
  even if the Metal lane isn't ready.
- The **`mac_build_and_test` job** needs `runs-on: [self-hosted, macos]` with the Metal
  toolchain. **Prerequisite:** a self-hosted macOS+Metal runner registered to the PicoMLX
  fork. If none exists, that runner is a hard dependency for the build/numerics gate — flag
  to the team. (GitHub-hosted `macos-15` is a weak fallback: it builds, but heavier
  model-download/Metal parity tests are slow/unreliable there.)

> Note: CI ≠ the Codex/Gemini review bots. CI gives build+test+lint; the bots review every
> push (§7). Both are wanted on every CB PR.

---

## 3. Naming (confirm before the rename)

Rationale ties directly to §4: the **scheduler/admission/memory layer is reusable across
modalities, but the engine is autoregressive-LLM/VLM-specific.** Names should reflect that
split so a future `EmbeddingBatchExecutor` / `DiffusionBatchExecutor` can coexist without
stealing a generic name from an LLM-only type.

| Current | Recommendation | Why |
|---|---|---|
| `InferenceScheduler` (actor) | **keep** | Genuinely the generic policy/admission/memory coordinator; it can later route non-LLM batches. Generic name is correct here. |
| `BatchInferenceEngine` (class) | **`BatchGenerationEngine`** (alt: `LLMBatchEngine`) | It is autoregressive token *generation* (LLM **and** VLM). "Generation" covers VLM; "Inference" is what embeddings/Flux also do, so it over-claims the generic word. Confirm preference. |
| `EngineDriver` (actor) | **`BatchEngineActor`** (alt: `EngineExecutor`, `GenerationLoopActor`) | "Driver" is vague. This type is the actor that *exclusively owns the engine and pumps the serial decode loop*. Name it for that role. |
| `DecodeBatch` / `PrefillBatch` | keep | Clear and accurate. |
| `BatchStepResult` / `FinishedRowCache` | keep | Clear. |
| `RowSampler` / `RowProcessorSource` | keep | Clear. |
| `BatchKVCache` / `BatchRotatingKVCache` / `BatchQuantizedKVCache` / `BatchedCache` | keep | Consistent `Batch*` family. |

Do the rename in Phase 0 (right after rebase, before fixes) so review threads land on
final symbol names. **`InferenceScheduler` is the only public name that should stay generic.**

---

## 4. Forward-compatibility — verified, not built

Goal: confirm the architecture can **later** interleave Flux (image gen), BERT (tool
selection), embeddings, and speculative decoding, without building that coordinator now.

**Verdict: feasible with the current layering, provided the four seams below are preserved.**
The split is: `InferenceScheduler` (generic admission / FIFO / wired-memory / max-batch policy)
→ a **per-modality executor** (today: the autoregressive engine behind `BatchEngineActor`).

What's already in the repo:
- **Speculative decoding** exists single-stream: `SpeculativeDecoding.swift`,
  `MTPSpeculativeTokenIterator` / `SpeculativeTokenIterator` (both `TokenIteratorProtocol`),
  `MTPDrafterModel(Factory)`, Gemma4 assistant drafter. Pattern: drafter proposes a block,
  target verifies in one pass, **variable tokens accepted per round**.
- **Embeddings/BERT**: `MLXEmbedders` (`Bert`, `NomicBert`, `LFM2Bidirectional`) — bidirectional
  encoders, single forward pass, **no KV decode loop, no EOS**.
- **Flux**: not in the repo today (future/external).

Seams to preserve (verify during PR2/PR4 review; do not implement):

1. **Keep `InferenceScheduler` free of LLM-token assumptions.** Admission, FIFO queueing,
   `setMaxBatchSize`, and the wired-memory ticket lifecycle must not bake in
   "every unit emits tokens / has EOS." Embeddings and Flux admit through the same scheduler
   but run a different executor. The engine-specific logic belongs in the driver/executor,
   not the scheduler.
2. **Treat the executor as the per-modality piece behind the driver.** The autoregressive
   engine is one executor; an `EmbeddingBatchExecutor` (one forward → N vectors) or a
   `DiffusionBatchExecutor` (fixed denoising steps, no KV/tokenizer) would be siblings. Avoid
   coupling the scheduler to the concrete `BatchGenerationEngine` beyond what the autoregressive
   path needs — a thin executor protocol seam is enough (one conformer for now).
3. **Do not hard-limit token fan-out to exactly one token/row/step.** Speculative decoding
   accepts a *block* per row per step. `BatchStepResult` / `SchedulerTokenHandler` should be
   able to carry ≥1 accepted token per row per step (or emit multiple results per step), so a
   future `SpeculativeDecodeBatch` slots into the same fan-out. Flag any code that assumes
   strict 1-token-per-row-per-step as the spot to keep flexible.
4. **Keep `GenerationRequest` from over-coupling to LLM token streaming.** It already carries a
   full `LMInput` (so VLM batching is a later `PrefillBatch` extension). For embeddings/Flux a
   sibling request type is acceptable later; just don't bake LLM-only fields so deep that the
   scheduler can't admit a non-generative unit.

These are review checks on PR2/PR4, not new code. The §8 assumption ("no outer coordinator in
this stack") stands.

---

## 5. Per-PR fix lists

References are `file:line` from the live review threads (Codex/Gemini) on each PR.

### PR1 — caches (PR #16)

Open review threads to resolve (correctness blockers first):

- **CRITICAL — extracted rotated-row OOB / negative-growth crash.** When `seqOffset >=
  maxCacheSize` with `leftPadding > 0`, stripping padding shrinks the buffer below
  `maxCacheSize`; a later `RotatingKVCache.updateInPlace` computes negative `newSize` (crash in
  `zeros`) or writes out of bounds. Keep the full `maxCacheSize` buffer when
  `seqOffset >= maxCacheSize`. `BatchRotatingKVCache.swift:775`.
- **HIGH — negative `_idx` / corruption in `filter` during chunked prefill.** Guard the
  left-shift with `_idx >= Int(minLeftPad)` so partially-written padded prompts don't drive
  `_idx` negative or slice uninitialized data. `BatchKVCache.swift` (`filter`).
- **Row-filter metadata desync.** On cancel/filter between `prepareBatched(...rightPadding:)`
  and `finalize()`, `_rightPadding` (`BatchKVCache.swift:237`) and `_lengths`
  (`BatchRotatingKVCache.swift:646`) keep the old batch shape while rows are reduced →
  wrong-row roll or shape error. Filter both with the same `indices`.
- **`prepareBatched` drops SSM left padding.** The protocol path discards the `leftPadding`
  arg for `ArraysCache`/`MambaCache`, so `createSSMMask` treats pad as real input. Thread it
  through. `BatchedCache.swift:135`.
- **`finalize()` rolls the whole buffer instead of `0..<_idx`** for right-padded cached-prompt
  prefills shorter than the alloc step → loses the prompt tail. Roll only the populated slice.
  `BatchKVCache.swift:526`.
- **`fromSingle` keeps the oldest window, not the newest**, when source `offset > maxSize`
  (over-window) — same case `merge` rejects. Take the suffix window or reject over-window
  sources. `BatchRotatingKVCache.swift:928`. (This is the v1 "merge keeps correct tail" item.)
- **`ChunkedKVCache` leaks through the simple-cache adapter precondition** (it subclasses
  `KVCacheSimple`); use an exact-type check so the factory's chunked rejection actually holds.
  `BatchKVCache.swift:373`.
- **Gemma2 legacy mask bypass.** `Gemma2ModelInner` uses the array overload
  `createAttentionMask(h:cache:[KVCache]?)`, bypasses `BatchKVCache.makeMask`, and passes only
  `offset` to `createCausalMask` → left-padded rows attend to pad KV. Route the legacy helper
  through the batched mask path, **or** denylist Gemma2 from batching. `KVCache.swift:245`.
- **`MambaCache` row extraction returns a plain `ArraysCache`** → `cache as? MambaCache`
  downcasts (e.g. `LFM2.swift:303`) fail and recurrent state is dropped on extract/continue.
  Preserve the concrete subtype or special-case extraction. `KVCache.swift:1317`.
- **Batched caches unregistered for prompt-cache serialization** → `cacheClassName` falls
  through to `KVCache` and `loadPromptCache` rebuilds a `KVCacheSimple` from 4-array batched
  state (fatal/misread). Register `BatchKVCache`/`BatchRotatingKVCache` or explicitly reject
  them from disk round-trips. `BatchKVCache.swift:21`.
- **Keep `keep > 0` rejected** (current code already throws in the factory) until temporal
  valid-mask semantics exist. **Reconcile the divergence:** PR2's `BatchEngineTests` asserts
  `keep > 0` is *accepted*; update that test to expect rejection (or land the valid-mask
  support). `BatchedCache.swift:256`.

### PR2 — batch engine

- **Reconcile the diverged `makeAdoptedBatch` copy** with the PR2 signature
  (`uids` + `processorSources` in, `DecodeBatch` out). PR2 already fixed several adoption bugs
  that PR4 carries a stale copy of — land the canonical version here.
- **Sampler parity** (most already fixed in PR2; verify they survive rebase): top-P→min-P→top-K
  order on unscaled logprobs, temperature at the draw; top-K tie masking via `argPartition`;
  bf16→f32 promotion before logprobs; `fallbackIsGreedy` gating; **penalties applied to raw
  logits before log-softmax** (`DecodeBatch.swift:372`, still **open**); **preserve per-row
  fallback sampler on `extend`/merge** (`DecodeBatch.swift:302`, **open**).
- **Preserve `StopSequenceMatcher` state on adoption** — partial multi-token stop prefixes are
  lost when a migrated row restarts from `makeState()`. `DecodeBatch.swift:160` (**open**).
- **Direct multi-request admission** for fresh `generateBatched` so prefill is batched from the
  start instead of single-then-upgrade (ties to PR4 §5.PR4).
- **Performance (after correctness):** segmented ragged prefill in `PrefillBatch` (don't pay
  longest-prompt cost for short prompts); flatten per-layer cache evals into one `eval` per
  prefill chunk; restore the `.causal` fast path when there's no padding or `B == 1`.

### PR3 — prompt cache

- Keep the protocol-based `PromptCaching` design; it already exposes the read path
  (`fetchNearestCacheResult`) PR4 needs. **Do not land a write-only cache.**
- Confirm nearest-prefix lookup, exact hit, partial hit, and final KV write-back are all
  reachable from PR4's wiring (the read path is the integration contract).
- Defer double-copy elimination unless ownership transfer is trivial; leave a TODO,
  correctness first.

### PR4 — scheduler / API

- **Rebase on PR1/PR2/PR3 tips** (no stale cache/engine copies).
- **Wire prompt-cache fetch** into scheduler/driver/engine: exact hit skips prefill; partial
  hit prefills only the remainder; miss = full prefill. Key adopted entries by **full history,
  not generated-only** (else a suffix lookup reuses another prompt's KV) — already flagged.
- **`wiredMemoryTicket` correctness** (the v1 "latest ticket wins" cluster):
  - store tickets per request/row, not one `activeWiredMemoryTicket`;
  - re-read the ticket inside the drain so a late joiner's policy applies (don't snapshot once
    before the loop — `EngineDriver.swift:421`);
  - honor the ticket on the batched drain path (currently dropped);
  - distinguish an already-running drain from a completed one before restarting
    (`InferenceScheduler.swift:1080`);
  - end the ticket on finish, cancellation, dropped stream, or scheduler shutdown.
- **`maxTokens == nil` → unbounded.** The scheduler already defines
  `defaultMaxTokens { Int.max }` (`InferenceScheduler.swift:1055`) but the engine default is
  `128` and not all paths are consistent. Make **every** batched/upgrade path use `Int.max`
  for `nil` to preserve the single-stream public default (`InferenceScheduler.swift:986`).
  *Note: the bots disagree — Gemini suggests defaulting to 128 for consistency with
  `admitToBatch`; Codex (thread 986) wants unbounded to preserve the documented default. We
  side with Codex. State this in the reply + a source comment so the next round skips it.*
- **`.upgrading` state must survive a third request finishing mid-handshake** — keep the
  scheduler in `.upgrading` until adoption completes, so a thrown/early-finished interleaving
  request can't flip it to `.idle` and clobber the pending upgrade
  (`InferenceScheduler.swift:552`, P1).
- **Failed/unsupported migration must not truncate the first stream** — treat as
  cancellation/internal failure, not `.length`/`finish()` mid-generation
  (`InferenceScheduler.swift:1227`).
- **Preserve the live token at upgrade** — `live.currentToken` is consumed as the adopted
  seed and never delivered, dropping one token per upgraded stream (P1).
- **Keep stop matching for the adopted row** — adopted rows get an empty matcher with no EOS
  fallback, so upgraded requests ignore EOS until `maxTokens` (P1).
- **`runSingle` stop-reason bug** — the `else` branch sets `.cancelled`, so naturally-completed
  single-stream generations report as cancelled; should be `.stop`.
- **Include `unknownTokenId` in batched stopping** to match single-stream.
- **Route fresh SSM/hybrid requests directly into the batch engine** when no single-stream
  migration is needed (`canUpgrade` rejects `MambaCache`/`ArraysCache` migration today, so
  explicit batches get serialized behind the first single request —
  `InferenceScheduler.swift:413`).
- **Per-request `promptCache` on the single path** is ignored unless it matches the container
  cache — honor the request's own cache (`InferenceScheduler.swift:664`).

### PR12 — quantized cache (optional follow-up)

- Stack on PR1; **rebase on `main` after #17 lands and drop the duplicate finfo hunk** (§0.1).
- Flip the PR2 `BatchEngineTests` "rejects quantized" expectation once this lands (quantized
  now routes; chunked still rejected).
- Keep `ChunkedKVCache` rejected with the documented rationale.

---

## 6. Tests

Targeted, failure-mode-driven (not volume). Most must run on the Metal lane (§2.2).

- **PR1:** rotation-wrap parity with mixed prompt lengths; extracted-rotated-row update past
  `maxCacheSize` (the CRITICAL crash); `filter` during partial chunked prefill (negative-idx);
  row cancel between `prepare` and `finalize` (metadata desync); SSM `prepareBatched` left
  padding honored; `finalize` rolls only `0..<_idx`; over-window `fromSingle` keeps newest
  window; incompatible-merge fails loudly; `ChunkedKVCache` rejected via exact-type check;
  Gemma2 mask parity **or** explicit denylist; batched-cache disk round-trip (or explicit
  reject); `keep > 0` rejected.
- **PR2:** batched-vs-single greedy parity for ragged prompts; min-P / penalty / fallback
  sampler parity batched-vs-single; adopted-row stop-matcher state preserved; direct
  `generateBatched` uses one batched path (not single-then-upgrade).
- **PR3/PR4:** prompt-cache exact hit skips prefill; partial hit prefills only the suffix;
  multi-turn `ChatSession` + scheduler doesn't re-prefill full history each turn.
- **PR4:** two concurrent submits both complete with correct token counts; live token
  delivered (not dropped) at upgrade; upgraded row honors EOS and `maxTokens` exactly;
  `maxTokens == nil` generates past 128; naturally-completed single reports `.stop` not
  `.cancelled`; third request mid-handshake can't strand `.upgrading`; unsupported-migration
  doesn't truncate the first stream; late ticket admitted+released during an active drain;
  dropped stream cancels work and releases the ticket.

Suites to run on the Metal lane:

```bash
swift test --filter BatchCacheTests
swift test --filter BatchQuantizedCacheTests      # PR12 only
swift test --filter BatchEngineTests
swift test --filter PromptCacheTests
swift test --filter InferenceSchedulerTests
swift test --filter ModelContainerBatchingTests
```

---

## 7. Working process — the review loop

For **every** PR opened/updated in this stack:

1. **Subscribe to PR activity** on open (so CI + Codex/Gemini events wake the session).
2. On each bot review, **investigate every comment**, then act:
   - **Fix** if confident and in-scope, push the fix, and **reply stating "fixed" + how**.
   - **Ignore** only with justification: **reply stating the decision and why**, *and* add a
     short **source-code comment** explaining the reasoning so the bot skips it next round.
   - Skip silently only for true duplicates/no-ops.
   Always reply with the decision (fixed / ignored + reason) so humans can follow the thread.
3. **Iterate until every thread is fixed or rightfully ignored.** Codex signals approval with a
   **👍 reaction** — watch for it; that's the terminal state for a PR's review.
4. **Manual trigger:** Codex sometimes doesn't fire. If **10 minutes** after a push there's no
   Codex review, comment **`@codex review`** to trigger it manually. (Don't poll with `sleep`;
   schedule a check-in and re-check on wake.)
5. A subscription isn't done until the PR is **merged or closed**.

---

## 8. Assumptions / out of scope

- Continuous batching is scoped to autoregressive LLM/VLM generation in this stack.
- **No outer Flux/BERT/embeddings/speculative interleaving coordinator** in these PRs — §4
  only verifies the seams stay open for it later.
- `ChunkedKVCache` stays rejected.
- PR12 (quantized) is a separate stacked follow-up, gated on #17 landing on `main`.
- Self-hosted macOS+Metal runner on the fork is a prerequisite for the build/numerics gate
  (§2.2).
- Correctness and stack hygiene before sampling vectorization / custom-executor performance.
