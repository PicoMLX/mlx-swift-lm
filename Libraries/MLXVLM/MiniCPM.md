# MiniCPM media understanding

`MiniCPM` loads `minicpmo` (MiniCPM-o 4.5), `minicpmv4_6` and `minicpmv4_7` checkpoints through `VLMModelFactory`. All three accept text, still images and sampled video frames and generate text. MiniCPM-o also accepts audio. Speech generation and TTS tensors are excluded explicitly.

The models preserve their distinct visual paths: MiniCPM-o uses SigLIP and a learned cross-attention resampler; V 4.6/4.7 use an intermediate ViT merger and spatial projection. V 4.7 applies canvas M-RoPE to thumbnails, detail slices and video frames, including the dense and MoE Qwen3.5 text configurations. V 4.6 uses ordinary one-dimensional language positions. Both 4x and 16x visual compression follow the processor’s `downsample_mode`. The model infers each request’s mode from crop grids and placeholder bounds, including overrides of the checkpoint default; rows with different modes retain independent canvas positions when batched.

The processor preserves native image and slice marker tokens, image IDs, detail row newlines, image slicing and RGB bicubic resampling. The repeated feature positions are represented by private negative IDs in `LMInput`: `-200` image, `-201` video and `-202` audio. They are replaced with encoder features before token embedding lookup. Pixels use `[crop,patches,patch²×channels]` with one spatial `THW` per crop; video crops retain frame order. Canvas positions use `ProcessedImage.positionIds`/`ProcessedVideo.positionIds` with shape `[3,batch,sequence]`. Audio uses mono 16 kHz samples, Whisper's log-mel frontend, encoder and pooled projection; longer recordings are split into 30-second input segments.

Request-specific continuation positions belong to an `ArraysCache` alongside the first full-attention cache in a `CacheList`. There is no mutable model-wide canvas or modality state. Generic cache factories can prefill right-padded prompts and preserve state through filtering, extraction and extension. The Qwen3.5 convolution tail is gathered at each row's actual prefill length before padded positions are discarded. Recurrent state and position metadata do not advertise arbitrary speculative rewind.

Tensor parallelism shards the Qwen decoder's attention and MLP projections, including MoE experts and quantization auxiliaries. Vision, audio, resampling and merger modules are replicated on each rank. Decoder dimensions must divide the group size; invalid sharding is rejected before modifying weights. Models should be loaded using the sharded factory path rather than loading a complete production checkpoint on every rank first.

`MiniCPMTests` exercises ragged chunked prefill/decode, mixed media admission, row filtering/extraction/extension, missing bounds and MoE checkpoint conversion. `MiniCPMReferenceTests` compares tiny deterministic tensors against the pinned Python networks at `4693cd0eaf022ae7ef61c6fdb55e17e6399879a7`, including audio features, Pillow resizing and canvas layout. Regenerate with:

```sh
python Tests/MLXLMTests/Resources/generate_minicpm_reference.py /path/to/mlx-vlm
```

`MiniCPMDistributedParity` and its `run.sh` execute two local MLX ring ranks and compare float/4-bit logits and token sequences against an unsharded model while checking that each rank holds half the query rows. The runner covers Omni, V 4.6, V 4.7 dense and V 4.7 MoE.

This Swift adaptation was written with AI assistance from the MIT-licensed mlx-vlm reference. Canvas positional indexing derives from the Apache-2.0 OpenBMB/Hugging Face implementation. The corresponding notices are included alongside this document.

The opt-in `MiniCPMCheckpointSmoke` runner uses the published MiniCPM-V 4.6 4-bit checkpoint, its actual tokenizer/chat template, image preprocessing and mixed image/text cache adoption. It runs native half precision for greedy-token parity, then preserves the packed 4-bit weights while using float32 compute with `MLX_ENABLE_TF32=0` for strict numerical comparison. On M5 Max the default half/TF32 run produced identical tokens with a maximum logit difference of 0.609375; accurate compute reduced this to 0.00008583 (relative 0.00000209). This records backend rounding separately from the cache and architecture checks.
