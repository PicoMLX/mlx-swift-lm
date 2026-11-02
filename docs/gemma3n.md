# Gemma 3n

`MLXVLM` supports Gemma 3n E2B and E4B image and audio inputs with text output.
`gemma3n` selects the MobileNetV5 vision tower, Conformer audio tower, modality
embedding tables and the shared `MLXLLM` Gemma3n language implementation.
Text embeddings retain the language model's square-root scale before modality
tokens are replaced, matching Transformers' `Gemma3nTextScaledWordEmbedding`.

The processor expands image and audio placeholders to 256 and 188 soft tokens,
respectively. Audio is mono, sampled at the checkpoint's sampling rate, and
truncated at 480,000 samples. Array audio inputs must already have that rate.
Feature extraction follows Gemma3n's HTK pre-emphasis and log-mel algorithm.
Video input is not accepted by this Python-compatible processor; sampled frames
can be supplied as separate images.

## Continuous batching

The language decoder uses per-row cache positions and preserves positions when
attention layers share K/V. Only owning layers allocate caches. Image and audio
prefill consumes the feature embeddings and per-layer text inputs, then decoding
needs only the owning caches and ordinary token IDs. Encoders are evaluated once
per prepared input; their feature tensors are not kept as mutable model state.
The accompanying ragged tests cover window rotation, reordering, extraction and
adoption of media-prefilled rows alongside text rows.

## Tensor parallelism

`Gemma3n` and `Gemma3nTextModel` conform to `TensorParallelModel`. Attention
projections are split across ranks, including query-only shared-KV layers.
Nonsparse MLPs are also split. Sparse MLPs remain replicated because their
activation thresholds depend on statistics over the entire intermediate width.
AltUp, Laurel, embeddings, normalization and modality encoders stay replicated.
Attention head counts and each sharded MLP width must divide by the group size.
Quantized checkpoints use the existing quantized sharded linear adapters.
KV-head metadata describes only cache-owning layers and their actual local
projection widths after sharding.

`Tests/Gemma3nDistributedParity/run.sh <Gemma3nDistributedParity executable>`
launches two actual ring-backend ranks on one Mac. It compares float and 4-bit,
dense and sparse, solo and ragged-batched decoding against the unsharded model,
including precomputed media-feature prefill, and checks that attention weight
rows physically shrink. Multi-host transports
must additionally be validated on the intended cluster.

## Reference validation

`Tests/MLXLMTests/Resources/generate_gemma3n_fixture.py` regenerates the tiny
numerical fixtures from `Blaizzy/mlx-vlm` commit
`4693cd0eaf022ae7ef61c6fdb55e17e6399879a7`. They compare MobileNet blocks,
relative audio attention, the full tiny Conformer stack and hard/soft modality
embeddings. Log-mel fixtures come from Transformers' Gemma3n extractor.
No checkpoint downloads are needed for these tests.

The optional `Gemma3nCheckpointSmoke` executable and its `run.sh` accept a local
checkpoint directory and Python with Transformers, tokenizers, NumPy and Pillow.
They use the checkpoint's native chat template and tokenizer, a 64px image
processed at 768px, and a one-second audio clip. Image, audio and text requests
are independently prepared and then adopted into a shared decoding batch.
The published `mlx-community/gemma-3n-E2B-it-4bit` checkpoint passed eight-token
greedy comparisons against independent decoding. Accurate numerical validation
promotes floating parameters and quantization scales to float32 and disables
M5 TF32 (`MLX_ENABLE_TF32=0`); packed 4-bit weights remain quantized. This avoids
shape-dependent half/TF32 rounding between solo and batched Metal kernels.
The measured maximum logit difference was 3.91e-5 (relative 1.35e-6).

The implementation follows the MIT-licensed MLX-VLM port and Google's
Apache-2.0-licensed Transformers Gemma3n preprocessing implementation.
