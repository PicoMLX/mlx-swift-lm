# Qwen3 Omni Thinker

`qwen3_omni_moe` and `qwen3_omni_moe_thinker` load the Thinker for image, video,
audio and text input with text output. The Talker and Code2Wav checkpoint
namespaces are excluded during loading. Speech generation is outside this port.

The vision transformer and MoE language core share the existing Qwen3-VL
implementations. Audio uses the three stride-two convolution frontend,
chunk-local sinusoidal positions, windowed bidirectional encoder attention,
and output projection. Valid mel frames are packed per audio item; encoder
attention cannot cross item boundaries. Feature and placeholder counts must
match exactly.

The processor accepts flat or nested image/feature-extractor metadata. It uses
16 kHz mono Whisper log-mel features by default, including periodic Hann,
centered reflect padding, Slaney-normalized mel filters, log10 dynamic-range
clamping and Whisper normalization. A final partial hop is zero-padded rather
than dropping input samples. Images share Qwen3-VL's sRGB resize/patch pipeline.
Explicit image pixel budgets and video frame sampling controls are honored.

Set `additionalContext["use_audio_in_video"] = true` to extract a video URL or `AVAsset`'s
audio track and interleave its video/audio placeholder tokens by time.
Decoded video frames retain actual sampled timestamps for temporal MRoPE;
floating-point positions preserve non-integral times. Audio presentation-time
gaps and nonzero track starts are retained for synchronization. Ordinary video input
without that option uses video frames only. An unavailable synchronized audio
track is an error. In-memory synchronized inputs can be supplied as prepared
`LMInput` with packed features, interleaved tokens and full-prompt positions in
`ProcessedVideo.positionIds` (`[3, batch, sequence]`).

MRoPE follows the authoritative Transformers Omni time/audio rules, rather than
the pinned mlx-vlm wrapper's generic image/video-only position builder. A single
request's continuation delta lives in an `ArraysCache` beside its first KV
cache, inside `CacheList`. Every forward supplies explicit per-row positions;
`LMOutput.State` is unnecessary. Generic `mergeCaches(_:)` conversion and row
filtering, extension, extraction and media-prefill adoption carry the positional
metadata with the corresponding KV row. Prompt prefill chunks slice both
positions and deepstack residuals. Deepstack residual scatter handles multiple
batch rows.

Qwen3 vision's 72-wide attention heads depend on the resolved MLX attention
padding fix for M5 matrix kernels. The current mlx-swift revision `12f53de5`
uses MLX core `598b041`, which includes upstream MLX #4455: eligible long,
unmasked FP16/BF16 attention with 72/80-wide heads is padded to 96 internally.
Preserve that fix when changing dependencies. This avoids the older kernel
selection issue; numerical parity tests do not establish a latency improvement,
and M5 timing still needs a representative checkpoint benchmark.

Tensor parallelism splits query/key/value attention heads and their output
projection, dense MLP hidden widths, and each routed expert's hidden width.
The router, embeddings, norms, vision and audio encoders remain replicated.
Expert partial results are summed once after routing/weighted reduction.
Float and quantized language weights use the same sharding path. Head counts
and MLP widths must divide the rank count; quantized column shards must also
respect the quantization group size. Unfused adapters and repeated sharding
are rejected.

Regenerate numerical fixtures without model downloads:

```sh
python Tests/MLXLMTests/Resources/generate_qwen3_omni_fixture.py /path/to/pinned/mlx-vlm Tests/MLXLMTests/Resources/qwen3-omni-reference.safetensors
```

The fixture records mlx-vlm commit
`4693cd0eaf022ae7ef61c6fdb55e17e6399879a7`. Unit tests compare audio, vision,
MoE/deepstack, full image/video/audio logits, Whisper features and explicit
time positions; they also exercise chunked media prefill, ragged batching,
decode, reordering, extraction, copied-cache continuation, media adoption and
singleton float/4-bit sharding.

Equation and cache conformance fixtures use float32 CPU execution to isolate
architecture behavior from M5's shape-dependent TF32 GEMM rounding. One pinned
reference correction is explicit in the generator: the Omni vision block MLP
hardcodes exact GELU despite declaring `gelu_pytorch_tanh`; the fixture follows
that declared checkpoint setting and the authoritative Transformers behavior.
The Swift vision core already honors the setting. Published configs may omit
`num_position_embeddings`; its native default is 2304 learned positions.

```sh
DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild test -scheme mlx-swift-lm-Package -destination 'platform=macOS' -skipPackagePluginValidation -only-testing:MLXLMTests/Qwen3OmniTests
Tests/Qwen3OmniDistributedParity/run.sh /path/to/Qwen3OmniDistributedParity Tests/MLXLMTests/Resources/qwen3-omni-reference.safetensors
```

The parity runner launches two actual localhost MLX ring processes and compares
float/4-bit, solo/ragged batch, text/full image-video-audio prefill plus decoding
against unsharded computation. It asserts physically reduced attention/expert
weight shapes and matching greedy tokens. Two ranks on one Mac exercise
collectives and partitioning; they do not validate multi-host transport,
network configuration or performance. Run the same executable on each real
host with the deployment's `MLX_HOSTFILE` before claiming multi-host validation.

The opt-in `Qwen3OmniCheckpointSmoke` executable loads a complete local published
checkpoint and exercises text, image, audio, and a timestamped video with its
audio track. Python's native tokenizer/chat template produces exact prompt IDs;
the executable verifies that the Swift processor supplied those exact messages.
Generated IDs are decoded by the same native tokenizer. Prepared caches are
adopted with `makeBatchedCaches(from:)`, the production admission API, and eight
greedy tokens per row are compared with independent decoding. The native Omni
template defaults to `enable_thinking=True`; forcing `False` inserts additional
think tags and is unsuitable for this Instruct checkpoint's media prompts.

```sh
sh Tests/Qwen3OmniCheckpointSmoke/run.sh /path/to/Qwen3OmniCheckpointSmoke /path/to/Qwen3-Omni-30B-A3B-Instruct-4bit /path/to/python
```

This smoke keeps the checkpoint's 4-bit weights and promotes floating-point
parameters/scales to float32 with `MLX_ENABLE_TF32=0` for strict full-vocabulary
batch/solo comparison. On the downloaded 30B checkpoint it produced useful
picture, sound and video descriptions, matching greedy tokens across four
adopted rows; maximum absolute logit difference was `2.718e-5` and relative
difference `6.58e-7`. Normal half-precision and TF32 execution have different
rounding behavior; these precision settings make the numerical proof reproducible.

`Qwen3OmniSDPABenchmark` is an opt-in bounded Release executable for the M5
attention dependency check. It compares D72 directly with the inherited D80
outer-padding shape and explicit D96, using the same original attention scale,
16 heads, sequence length 1024, three warmups and twenty synchronized forwards
per dtype/shape. Padding is materialized before timing, so the result isolates
SDPA kernel dispatch and excludes projection, encoders and generation. It also
checks padding parity. This is not a benchmark against the older MLX dependency.

Source attribution and licenses are in `ThirdPartyNotices/Qwen3Omni`.
