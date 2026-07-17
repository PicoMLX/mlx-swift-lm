# Continuous-Batching Audit — 2026-07-13

**Scope:** integration branch `claude/continuous-batching-review-qzilp9` (= `claude/continuous-batching-pr-review-2w9fl3`, tip `26660e0`, PR #23) and the 8-PR chain `cb2-1`…`cb2-8` (PRs #24–#31).
**Method:** 40 agents — 8 verifying every finding of the 2026-07-02 readiness review against HEAD, 10 fresh bug-hunters, 7 assessments (API, interleaving, split ×3, batching3, tests), 15 adversarial verifiers. Every new finding and every "not fixed" claim was independently re-derived by at least one skeptical verifier (two for majors, plus a tiebreaker rule). Code-level review on Linux — no Metal runtime.
**CI ground truth:** integration HEAD is green on the macos-26 Metal runner running the *full* MLXLMTests bundle; all 8 chain branches green at their tips (2026-07-07).

---

## 1. Is the code correct — were the bugs solved?

### Prior-review scoreboard (2026-07-02 review, verified at HEAD)

**The 15 numbered majors:** 10 fixed outright, 1 fixed-but-overshadowed, 1 partial, 3 not fixed.

| Finding | Status at HEAD | Notes |
|---|---|---|
| M-1 penalties on log-probs | **FIXED** (DecodeBatch:389-403 raw logits + per-row logSumExp; pinned by `processorsReceiveRawLogits`) | residual: f32 vs bf16 rounding |
| M-2 adopted-row penalty history | **FIXED** (prompt-inclusive handoff IS:1562; `tokensIncludePrompt` threading verified) | stale doc comment ED:54-58 contradicts it |
| M-3 stop strings lost | **FIXED** (batched path reuses the single path's `StopStringFilter` verbatim; `.stop` honored on all 3 drivers incl. mid-batch write-back) | handler-unit test only; no e2e test |
| M-4 adopted rows ignore EOS | **FIXED** (default EOS matcher BGE:430-434) | matcher *state* still resets on adoption (direct-engine multi-token stops only) |
| M-5 double-driver reentrancy | **FIXED** (driver assigned before awaits) | residual `refreshContext` race → see hot-swap cluster |
| M-6 stale idle-clobber TOCTOU | **FIXED** (admission epoch + `.batched`-only guard + drain ownership) | narrower non-FIFO-actor residue → see hot-swap cluster |
| M-7 cancel during `.upgrading` no-op | **NOT FIXED** structurally, but re-rated **minor**: per-token reap + M-4's EOS matcher bound it to typically 1-2 wasted tokens; worst case one leaked row decoding to natural EOS (needs tool-call/JSON buffering + no EOS) | fix: `.upgrading(SingleRequest)` payload or pending-cancel set |
| M-8 kvScheme gate | **FIXED** (IS:458-461, + LRU and engine guards) | |
| M-9 silent no-cache re-prefill | **PARTIAL** — per-request + batched fetch now work; but `promptCache` defaults nil on every front door and nothing warns | ergonomics, documented opt-in |
| M-10 `GenerationRequest.promptCache` unused | **FIXED** (honored on single, upgrade, batched paths; nil→container normalization) | |
| M-11 topology-blind cache namespace | **FIXED** (type+window+keep salt; quantized excluded by gating; a verifier could not construct a realistic collision) | edges: same-name weight swap; direct API users bypass salt |
| M-12 fetch-path GPU copy+eval under lock | **NOT FIXED** (LRU:228-230 → deepCopy → eval under `OSAllocatedUnfairLock`; insert half was fixed; disk half moot) | the one prior major that still needs real work |
| M-13 Mistral3 scalar llama-4 scaling | **FIXED** (per-row `batchOffsets` scaling, hand-recomputed; single-stream type-gated bit-identical) | no model-level test |
| M-14 Gemma3n sliding-mask displacement | **FIXED but overshadowed** — the clamp arithmetic is correct, but batched Gemma3n crashes *before* reaching it (see NEW-1) | |
| M-15 quantized trim clamp | **NOT FIXED** (BQKV:277-283 still `min(_idx, n)`, confirmed by 3 independent agents; siblings have the clamp; the existing `trimRewinds` test is shaped so it cannot detect it) | latent — class has no scheduler producer |

**33 minors:** 14 fixed, 5 partial, 12 not fixed (nearly all deliberate design trades — actor-held prefill, SSM blanket reject, `Int.max` default with documented parity rationale — or direct-API foot-guns unreachable via the scheduler), 3 moot (ChatSession revert, disk-persistence removal).
**Appendix B (15 open review threads):** 10 fixed, 3 moot, 1 partial (B-3: NemotronLabsDiffusion still scalar-offset, admitted, documented limitation), 1 not fixed (B-14: single-path cache *hits* seed penalty processors from the remainder only — the batched fetch path does it right, so single is now the odd one out; low severity, converges after ~20 tokens).
**§11 items:** honest timing fixed (two `promptTime: 0` edge paths remain: IS:1642, ED:282); fork-name scrub missed one "Layr" at ED:187; Gemma2 numerics change still has **zero pinning tests**; regression-suite gaps unimproved (below).

### batching3 comparison

The rewrite is a **strict improvement**. All 8 bug classes that generated batching3's ~35 fix commits are either structurally impossible now (throwing topology factory vs. silent cache-type fallback; parameter-derived samplers; per-request handlers; **zero `@unchecked Sendable`** vs. batching3's 13) or deliberately gated out (keep>0, SSM, chunked → typed error + serial fallback). Features lost, all deliberate or cheap to restore: ChatSession auto-batching (deliberate — it was batching3's buggiest seam), disk persistence (deliberate), the raw-token batched public entry point (`generateTokensBatched` — machinery survives, trivial to re-expose), `cacheClearInterval` periodic `Memory.clearCache()` (worth restoring for long-running servers), the `skills/…/batching.md` doc (worth porting).

### NEW findings (all adversarially confirmed)

**NEW-1 · CRITICAL — batched Gemma3n kills the process on its first forward pass.**
`Gemma3nText.swift:586` calls `MLXArray.maskFill(for: maskArray.dtype)` with a **bool** mask; mlx-swift 0.31.6's `maskFill` (verified verbatim at the pinned tag) `fatalError`s on non-float dtypes. `BatchRotatingKVCache.makeMask`'s only return is a bool `.array` — every prefill chunk, every decode step — and Gemma3n's layer 0 is a sliding layer, so **no batched Gemma3n request can ever produce a token; the fatalError kills the whole server process**. The trap went live in `996bacd` (which deleted the local non-trapping `maskFill` fallback); the M-14 clamp fix sits *below* the fatal, unreachable. Single-stream prompts >512 tokens hit the same trap — but that half is pre-existing on `main` (also resolves 0.31.6). Two more bugs are stacked behind the fatal, so a fix must address all three or the next one surfaces:
- `Gemma3nText.swift:303/306` casts bool masks to `queries.dtype` → SDPA treats float masks as *additive* (+1/+0) → effectively unmasked attention (major, latent);
- the 8b468da clamp is incomplete for chunk-2+ prefill past the window: it slices an already-correct `cappedOffset+n`-wide mask down to `W` → SDPA shape mismatch (major, latent; also pre-existing on main).
**Recommended action:** either gate Gemma3n out of the batched factory until properly fixed, or fix all three layers (float additive/`where`-based fill keyed on scores dtype; don't slice wider-than-window masks; drop the bool→float cast) — and add a batched regression test with a real sliding layer (today's suite declares `full_attention` everywhere, so `BatchRotatingKVCache` is never exercised through any model).

**NEW-2 · MAJOR — the model hot-swap path is unsound (cluster of 4 verified holes).**
All require `ModelContainer.update(_:)` (public) racing live traffic — no in-repo caller does this, but any server that hot-swaps weights will:
1. `contextEpoch` is checked *before* the upgrade handshake suspension but never re-validated after resume (IS:1347→1352-1554): a swap landing in the ~one-decode-step window splices **old-model KV into a new-model engine** — the exact corruption the epoch's own doc comment says it exists to prevent. Worse, the row's write-back inserts the Franken-KV into the LRU under the unchanged `configuration.name`, **poisoning the prompt cache** for later requests until eviction. (Confirmed 2/2 lenses.)
2. Under sustained traffic a stale driver is never torn down (cleanup only runs when the batch fully idles): all new requests keep decoding on the **old** model unboundedly, with write-backs keyed under the *new* name — same poisoning across the swap.
3. `batchLoopFinished`'s epoch guard assumes bump-after-read; Swift actors are not FIFO mailboxes, so a `hasWork`-served-before-pending-`submit` ordering resurrects a narrow M-6-style stale-idle, and its early-return then skips the `driverStale` teardown (`ensureDriver` never checks `driverStale`).
4. `attachSchedulerIfNeeded` keys on `ObjectIdentifier` of a possibly-deallocated container with no detach API: identity recycling silently serves B's requests with A's retained model; without recycling the scheduler is permanently `schedulerAlreadyAttached`.
**Recommended action:** one hardening pass on `refreshContext`/`driverStale`/epoch: re-check epoch after every handshake resume (the finalize-cleanly fallback at IS:1493-1514 is the right shape), check `driverStale` in `ensureDriver`, move the teardown ahead of the `.batched` guard, add a detach/weak-owner story — or simply document `update()` as unsupported while requests are in flight and fail it loudly.

**NEW-3 · MAJOR — quadratic trie DFS under the LRU lock.**
`LRUPromptCache.search()` (LRU:512-526): the longer-candidate DFS rebuilds the path array per node (`extra + [tok]`), and entry depth isn't stored, so a fetch that diverges from a stored long conversation walks the entire tail at O(N²/2) element copies — ~0.1-0.5 s CPU for one 36k-token tail, multiplied by sibling stored conversations — **while holding the unfair lock, on the driver actor**, stalling every live decode stream. Trigger = new conversation sharing a prefix with stored long ones (multi-turn continuations are O(1)). Fix: store depth/entry-count on nodes or reuse one path buffer.

**NEW-4 · conditional-MAJOR — `canUpgrade` approves upgrades the engine constructor rejects.**
`migrateCaches([]) == []` (non-nil) means a cacheless custom model passes `canUpgrade`, then `BatchGenerationEngine.init`'s new `!probe.isEmpty` guard (c55543b) throws, `ensureDriver` swallows it, and the fallback **ends the running request mid-generation with `.length`** — recurring for every concurrent pair, zero errors surfaced. Same mechanism for unvalidated `InferenceScheduler.Configuration` (e.g. `completionBatchSize: 0`). Regression for cacheless models (they batched before c55543b). Needs out-of-repo inputs → medium practical priority. Fix: make `canUpgrade` mirror engine acceptance + validate `Configuration.init`, or have the ensureDriver-nil fallback rebuild a single `TokenIterator` from the deposited state instead of finalizing.

**Verified minors (12):** double token delivery on the (currently unreachable) migration-failure fallback — a real 27866c1 regression that also skips write-back and misreports `.length`; `startSingleAfterUpgradeFallback`/`startSingleIfIdle` catch paths strand `queuedRequests` (plus a no-throw variant via the `guard let context` early-return); cancels dropped during `drainQueuedRequests`'s local-boxes window; batched cancels never write KV back to the prompt cache (single path does — warm-hit parity gap on a realistic "user hits stop" flow); `finishUpgradedSingle` discards the last token's stop disposition (`.length` instead of `.stop` — the only finalizer violating the codebase's own documented invariant); seeded rows **restart their PRNG key chain on upgrade**, bit-identically replaying the keys that produced tokens 1..N (and the comment at IS:1241-1246 claiming the chains differ is stale — c55543b made them identical); negative temperature = inverted sampling solo vs. greedy batched (out-of-contract param, one-line fix); `BatchQuantizedKVCache.extract` missed the 7e2ae88 pad clamp (second sibling-skew after M-15, latent); `extend()` ignores pending `_rightPadding`/`_lengths` on all three cache classes (public-API foot-gun, engine-unreachable — precondition both sides); longer-hit fetch deep-copies the *full* entry then only logically trims (memory retention for the whole generation; `extend` then pads the whole batch to the oversized width; when prefix ≪ entry the "hit" is a net pessimization); `trimBatched` rolls the full step-rounded buffer instead of the `..<_idx` head slice (perf-only, dormant); hygiene cluster (dangling "TEMPORARY HOME" comment KVCache:~2178, stale "e.g. Gemma2" comment, "Layr" leak ED:187, `tokensIncludePrompt` dead branch + false doc, `generatedTokens += responses.count` undercount vs. pluralization, stale `LiveIteratorState` doc).

### What held up (verified healthy)

`BatchRotatingKVCache` — the gnarliest math — survived a full fresh hand-recomputation (wrap scenario column-by-column, whole matrix): **zero new findings**. The batched **prefix-fetch** feature was hand-traced end-to-end and is sound (alignment, masks, RoPE, budget, keys; adopted-row penalty seeding is actually *better* than the single hit path). Sampler numerics parity (top-p/min-p/top-k order, seed split-order vs. mlx `KeySequence`, Gumbel shift-invariance) verified. Stop-string parity including stops straddling the upgrade boundary. Zero actor hops per token; the "zero-overhead single stream" claim holds (one lock read + `isCancelled` per token). Zero-copy upgrade adoption. The Gemma2 change is a genuine **bugfix** (on `main`, Gemma2 had *no causal masking at all* during multi-token prefill — the bool mask was being *added* as 0/1) — it just still needs its pinning test.

---

## 2. Is performance as good as possible?

No — but the biggest structural win landed. Batched prompt-cache fetch (old #1) is wired and correct; greedy fast path is real end-to-end; waiter admission drains sub-batches. Ranked remaining work (2-8 chats, 7B-30B):

1. **Decode dtype cast after `asyncEval`** (DecodeBatch:426) — still defeats the double-buffer ≈ every greedy token; **one-line fix** (`asType(.int32)` inside the async graph); ~10-20 % decode at 7B.
2. **Admission prefill freezes all decoding** (BGE:263→445-487) — 0.5-6 s stall per cold join, recurs every chat turn once batched; prefix-fetch shrinks it to the remainder on hits but adds its own per-row synchronous freeze (ED:409-422). Needs decode-steps-between-prefill-chunks interleaving.
3. **LRU fetch cost on the hot path**: GPU deep-copy+eval under the lock (M-12) + the quadratic DFS (NEW-3) + full-entry copies on longer hits — all on the scheduler/driver actors. The `PromptCacheSnapshot` seam (cb87e2a) is already in place for moving this out.
4. **Row-finish costs** — ~2L per-layer `.item()` syncs (the first drains the async next step, compounding #1), full-batch gather per layer, 3 copies of the row's KV on the driver between steps; 50-300 ms hiccup per finished row. Host-side `leftPadding`/`batchOffsets` mirrors kill the syncs.
5. **Per-row Swift sampling for temp>0** (the *typical* chat case — default temp is 0.6, so the greedy fast path rarely fires): B independent [1,V] kernel chains, 2 full-vocab argsorts/row/step with top-p; group identical-param rows into one vectorized call — 5-20 % of step time at B=8.
6. Mechanical: `extend` O(k²) growth (pre-allocate/amortize), sync `eval` per prefill chunk (single path uses `asyncEval` — strict regression, free fix), materialized masks even for uniform batches, per-row prefix-hit admissions, lone-survivor batch never downgrades to the single path, no memory-pressure hooks + `LRUPromptCache` default `maxBytes: Int.max`.

Verified already-good: one intended sync barrier per step, zero-copy adoption, in-place 256-step cache growth, one combined eval per prefill chunk, quantized decode stays in `quantizedMM`, per-step wired-limit handling.

---

## 3. Interleaving other model types (embeddings, image gen)

**No such API exists today — and that's fine, because adding it later is non-breaking.** The right seam already exists *internally*: `EngineDriver.drain`'s step-boundary `Task.yield()` (ED:616-641). Nothing can inject work there yet (no `GPUWorkSource`-style protocol). `EmbeddingModel` structurally cannot enter the engine (it's a `BaseLanguageModel` with no KV/prepare/step); interleaving belongs at the driver loop as an aux-work queue (~60-100 LOC, internal: `hasWork`/`step()` protocol or a FIFO of closures, one call site in `drain`, public only via a narrow `perform`-style method).

Speculative decoding: the prep landed (plural `BatchStepResults`, `trimBatched(perRow:)`, `TokenIteratorProtocol` generalization with a graceful decline handshake) but it is *inexpressible* through the scheduler — nothing constructs a speculative iterator on that path (deliberate; `GenerationRequest` has no drafter field). ChatSession speculation bypasses the scheduler and still works, though GPU contention between it and a live batch is arbitrated by nobody — worth documenting.

**The load-bearing pre-upstream move is not adding the seam — it's locking down the accidentally-public engine surface** (next item), so nobody builds interleaving on `engine.next()` and calcifies the wrong API.

---

## 4. Is the API minimal? Where should the LRU cache live?

**~37 new public top-level symbols; roughly 60 % could be internal or `@_spi(Batching)` today with zero loss to the front door.** The request-carrier bloat is fixed (only `GenerationRequest` is public of the 7 carriers; `EngineDriver`/`SchedulerTokenHandler` are internal). Still over-exposed with zero front-door reachability: `BatchGenerationEngine` (incl. `insert/next/adoptActiveBatch/makeAdoptedBatch`), `DecodeBatch`, `PrefillBatch`, `FinishedRowCache`, `BatchStepResult` (public = freezes the 1-token step before speculative needs it pluralized), all four `Batch*Cache` classes + the `BatchedCache` protocol/factory, the `RowSampler` helpers, the `StopSequenceMatcher` trio, and `InferenceScheduler.submit/submitBatch` (dead as public API — internal `attach` means they throw until a container binds). All tests use `@testable`; the repo already has `@_spi(Testing)` precedent. `BatchQuantizedKVCache` is public *dead code* (still no reachable producer) — defer it entirely (it's already out of the chain).

**LRUPromptCache: keep it in the library — your instinct is right, with a sharper rationale.** It's built almost entirely on public KVCache primitives, but it depends on *unstable internal knowledge* (RotatingKVCache `metaState[0] == keep` layout, per-type offset invariants, the ChunkedKVCache-subclasses-KVCacheSimple trap, rules that must mirror `migrateCaches`). Moving it out would freeze `metaState`'s string layout as de-facto public API. Right split (already made): mechanism + one default in-memory policy in the library; storage/eviction elaborations (disk, TTL, cross-process) app-side.

**The hooks are close but need four changes before upstreaming** (each breaking to retrofit later):
1. Shrink `PromptCaching` to fetch + insert — drop `trim` (library never calls it) and the `checkpoint:` parameter (always `false` at all 3 call sites; encodes one conformer's eviction scheme).
2. Symmetrize: `insertCache` takes raw `[KVCache]` while fetch returns an opaque `PromptCacheSnapshot` — make insert take the snapshot too (this is also the MaterializedArray on-ramp).
3. Decide sync-vs-async now. The protocol is sync and called on the scheduler/driver actors; a disk-backed conformer can't work correctly without async (or a documented "return promptly, hydrate off-thread" contract). Retrofitting `async` breaks every conformer.
4. Relocate eligibility/keying policy (`topologySalt`, `canUsePromptCache`, `isCacheCompatible`) off `LRUPromptCache` statics — the scheduler applies them to *every* conformer, so today an app cache cannot widen eligibility (paged/radix over new topologies) without library changes. Also: the salt an app conformer receives is silently rewritten by `topologySalt` — undocumented on the protocol.

---

## 5. Other suggestions

- **Tests are the weakest layer.** No end-to-end upgrade test exists at all (M-2/4/5/6/7 + token-parity live in one untested region), and `InferenceSchedulerTests.swift:9-13` *falsely claims* the macOS CI lane covers it. The regression suite's fixtures declare `full_attention` everywhere, so `BatchRotatingKVCache` is never exercised through any real model — which is exactly how NEW-1 survived. No seeded-sampler parity test (reintroducing the c55543b bug fails nothing), no batched-vs-single penalty parity, no Gemma2/Gemma3n/Mistral3 tests. Top five: (1) fix M-15 + a ragged trim test; (2) one e2e upgrade token-parity test; (3) adopted-row engine tests (EOS + penalty history); (4) a sliding-window-crossing regression config + Mistral3 `llama_4_scaling_beta` ragged parity; (5) penalty/seed parity tests. Delete the false comments.
- **Cheap restores from batching3:** the raw-token batched entry point (`SchedulerTokenHandler.rawToken` machinery already exists), `cacheClearInterval`, and the `skills/…/batching.md` reference doc.
- **Hygiene:** dangling "TEMPORARY HOME" comment (KVCache:~2178), stale "e.g. Gemma2" comment (KVCache:306-310), "Layr" at ED:187, stale `tokensIncludePrompt`/`LiveIteratorState` docs, the two stale-vs-each-other PRNG comments, old "PR1/PR4" numbering in comments.
- Consider a scheduler downgrade path (lone surviving batch row → single iterator) and bounded queues/backpressure before serious server use.

---

## 6. Does the 8-PR split make sense?

**Yes — the re-cut fixes every structural sin of the old diamond stack** (strictly linear, rooted on current main, each branch = one squashed commit, per-branch CI runs exactly the suites that exist at that point, and — critically — the July-7 re-cut folded the scheduler/engine/cache fix commits into the branch bodies, so chain files are blob-identical to the integration tip: reviewers approve what lands). cb2-2 and cb2-3 are model citizens; cb2-8 has real behavioral tests.

**But four things must happen before these go upstream:**

1. **Chain-wide blocker — cb2-1 ships a semantics change whose existing tests fail invisibly.** cb2-1 changes `ArraysCache.prepare` (counts → absolute ends) but the ~9 `KVCacheTests` expectation updates live only on the integration branch (7e2ae88's test hunk). Every cb2 branch is red on the full bundle; fork CI hides it (cb2-1 has *no test step at all*, the others use `-only-testing` filters). Upstream's CI will fail on every branch. Cherry-pick the test hunks into cb2-1 and rebase forward.
2. **The model fixes are in no chain PR.** M-13 (Mistral3, 44 lines → cb2-4 with a parity test) and M-14 (Gemma3n clamp, 15 lines → cb2-3) exist only on the integration branch — and the Gemma3n decision is now bigger than a fold-back: given NEW-1, either gate Gemma3n out of the factory in cb2-2/3 or fix the three-layer mask stack properly.
3. **The Gemma2 change still hides in cb2-1 with zero pinning tests** — the exact thing §13's "PR 0b" rule prohibited, and cb2-1's title also overclaims ("quantized-SDPA mask fix" is a pure declaration move). Extract it or land the pinning test in-PR. Give cb2-1 a test step (the full bundle is model-download-free).
4. **Structural tweaks:** fold cb2-7 into cb2-6 (152 untested lines of pure composition is not a reviewable unit; 7 and 8 are order-independent so this doesn't disturb cb2-8) — or keep it only with tests. cb2-6 at 3,422 lines is ~2× the design estimate; either ship it with a review map or carve out 6a (`SchedulerRequest`+`GenerationRequest`+`SchedulerTokenHandler`+`Evaluate` protocol change, ~390 prod lines — the single-path-parity review surface) from 6b (scheduler+driver+container); do *not* split scheduler from driver. Re-label the commit subjects ("chain 1/6"…"6/8"→"N/8"), replace stale "PR1/PR4" comment references, commit the `recut_chain.py` script the CI comments reference, and fix the false end-to-end-coverage claims in test headers. Open the quantized package later as its own PR (with the M-15 + extract clamps) once it has a producer.

Also fold the new scheduler fixes from this audit (NEW-2 cluster, NEW-4, stranded-queue, stop-disposition) into cb2-6 before opening upstream, and NEW-3 + M-12 into cb2-5.

## 7. MaterializedMLXArray: wait or not?

**What's actually in flight (verified this week):**
- **mlx-swift #418** "MaterializedArray is a Sendable MLXArray" (davidkoski) — `MaterializedArray` = `@unchecked Sendable` *subclass* of `MLXArray`, created via `materialize(x)`/`x.materialized()`; includes `MaterializedModule` and a breaking `Module.update()` `Self`→`Void` change. Open since June 1; feedback incorporated; awaiting angeloskath; author said July 7 he'd "try to get this part in this week"; last activity July 8. Not merged as of today.
- **mlx-swift-lm #335** "Use MaterializedArray for Sendable conformance" (davidkoski, depends on #418) — `ModelContext` becomes `Sendable`, **`ModelContainer` is deprecated** ("still works, but is deprecated"), models become immutable for inference, `TrainableModelContext`/`loadTrainable` for training. 14 commits, **two approvals**, awaiting angeloskath, active July 10. Per its description it does *not* touch KVCache/prompt-cache/generate.
- Context: upstream also has a stalled third-party continuous-batching PR (#263 — you already steered it toward a scheduler-level API in comments) and the open batch-generation feature request (#42), so upstream appetite exists and your stack answers the architecture question raised there.

**Recommendation — split the question in two:**

*For the prompt-cache representation (your original question): do not wait.* `PromptCacheSnapshot` was designed for exactly this — opaque storage, documented adoption plan, no public-surface change when the swap happens. Adopting `MaterializedArray` later is an internal patch to one file (plus removing deep-copy round trips at the actor seams). Since it's an MLXArray subclass, even `[KVCache]` reconstruction stays trivial. The only pre-landing tweak worth making: have `insertCache` take the snapshot type too (§4.2) so both directions of the protocol are representation-proof.

*For the front door (cb2-6/7/8): coordinate before opening upstream.* These PRs bolt the scheduler onto `ModelContainer` (`init(context:scheduler:promptCache:)`, `generate`/`generateBatched`, `loadModelContainer` overloads) — the very type #335 deprecates, authored by the maintainer, with two approvals already. If #335 lands first, your PRs 6-8 arrive extending a deprecated type and will be asked to re-plumb onto Sendable `ModelContext` (which also changes the concurrency story the scheduler's attach/refresh design assumes — a Sendable, immutable-model context actually *simplifies* the hot-swap cluster in NEW-2). And #418's `Module.update()` break forces an mlx-swift floor bump through your chain regardless.

*Practical sequencing:*
1. Open **cb2-1 → cb2-4** (plumbing, both caches, engine) upstream now, after the §6 fold-backs — they touch nothing #335 changes, and they're the layers upstream must digest slowest (the cache math). Waiting gains nothing here.
2. Hold **cb2-5** briefly only to apply the protocol tweaks (§4) — it's also #335-independent — then open it.
3. Hold **cb2-6/7/8** until #418/#335 resolve (both look like weeks, not months — #418 has been "about to land" since July 7), or better: ask davidkoski directly (you're already in dialogue on #263) whether the scheduler should attach to `ModelContext` or its successor, and re-cut the front door accordingly. That single conversation de-risks more than any amount of waiting.
