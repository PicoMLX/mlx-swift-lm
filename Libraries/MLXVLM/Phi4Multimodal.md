# Phi-4 Multimodal

`Phi4Multimodal` accepts text, images and 16 kHz mono audio and generates text.
It ports the `phi4mm` implementation of
[Blaizzy/mlx-vlm at 4693cd0](https://github.com/Blaizzy/mlx-vlm/tree/4693cd0eaf022ae7ef61c6fdb55e17e6399879a7/mlx_vlm/models/phi4mm),
available under the MIT license. The original checkpoint is
`microsoft/Phi-4-multimodal-instruct`.

The port includes a Phi decoder with partial rotary embeddings, the SigLIP2
NaFlex image tower and its projector, and the Cascades Conformer speech encoder
with SpeechLib log-mel features. Video and audio output are outside this model's
input-to-text contract. `UserInput.Audio.array` must contain mono samples already
at 16 kHz; audio URLs are decoded and resampled to that rate.

The raw checkpoint contains separate vision and speech adapters. They remain
immutable and are selected per cached sequence. Each decoder-layer cache is a
`CacheList` containing its attention cache and an `ArraysCache` of two mode flags.
Cold text requests use the checkpoint’s default vision adapter; warm text
continuations retain their own row’s mode. Consequently filtering, reordering,
extending and extracting a batch carry its
adapter selection with it. Decoder calls never mutate shared model weights.
Converted checkpoints with merged adapters load without the separate adapter
modules, preserving their converted behavior.

Tensor parallelism shards language attention heads and MLP width. The media
encoders, embeddings, norms and output head remain replicated. The checkpoint
loader splits fused QKV and gate/up tensors before sharding, including their
quantization auxiliaries and adapter output matrices. For column-parallel
projections the adapter's output matrix is sharded; for row-parallel projections
its input matrix is sharded and reduced. The opposite matrix remains replicated.

The model must be admitted by its application's batching engine only after its
batching policy supports this cache topology and media preparation. The library
implementation alone does not change PicoCore's model admission allowlist.

## Validation

`Phi4MultimodalTests` compares ragged batches and mode isolation, cache row
transforms, media placement, checkpoint splitting, and the single-rank sharding
path. The Python fixture generator uses the pinned reference's network modules
and deterministic tensors, without downloading checkpoint weights. Its seed is
314159: full-rank Gaussian weights use standard deviation 0.035, norm weights
are centered on one, and the waveform includes small Gaussian noise so its
log spectrum does not amplify roundoff in near-zero FFT bins. Full-network
goldens use the unmodified pinned merged-adapter network on the CPU; a separate
Metal golden checks the unfused adapter projection used for per-row routing.
This separates shape-dependent Metal GEMM rounding between merged and unfused
adapters from equation/layout regressions, with the same absolute tolerances.
The CPU audio frontend retains its absolute 2e-4 NumPy check; a separate Metal
FFT/frontend check bounds relative log-feature error below 1e-4.
NaFlex interpolation follows the pinned Gaussian antialias prefilter and its
singleton-output sampling rule, including rectangular grids.
RGB image pixels use a separate Pillow-compatible antialiased bilinear filter,
with fixed-point coefficients, eight-bit rounding after each separable pass, and
Pillow 13's pass-order selection for asymmetric reductions.
`generate_phi4mm_pixels.py` produces independent Pillow RGB goldens for mild
downscaling, upscaling, asymmetric sizes, singleton output, and identity resizing.
The RGB sampler follows [Pillow 13](https://github.com/python-pillow/Pillow/blob/13.0.0/src/libImaging/Resample.c);
its [MIT-CMU notice](../../ThirdPartyNotices/Phi4Multimodal-Pillow-LICENSE.txt) is retained.

Build the `Phi4MultimodalDistributedParity` executable with Xcode, then run:

```sh
sh Tests/Phi4MultimodalDistributedParity/run.sh /path/to/Phi4MultimodalDistributedParity
```

The runner launches two actual local ring ranks, compares float and 4-bit
mixed-modality decoding after both text and full image/audio preparation against
the complete model, and verifies half-width projections plus shard-local KV
head reporting. Local ranks validate partitioning and collectives; separate
hosts must still validate deployment transport and timing.

`Phi4MultimodalCheckpointSmoke` is an optional local-checkpoint executable with
a Python wrapper that uses the checkpoint's native tokenizer and chat template.
It prepares an image, one second of audio and a text prompt independently,
adopts all three caches into one batch, and checks eight greedy tokens per row
against independent decoding. It has been exercised with the original
`microsoft/Phi-4-multimodal-instruct` weights. Its accurate comparison promotes
floating parameters to float32 and disables M5 TF32 (`MLX_ENABLE_TF32=0`) to
separate model/cache correctness from shape-dependent half/TF32 kernel rounding.
Media preprocessing is evaluated before returning across executors, and linears
in both input towers support replacement by quantized checkpoint modules.
