# Google Cloud TPU (TorchTPU) Support

`MOSS-Image` supports Google Cloud TPUs via the native PyTorch `torch_tpu` backend (`device="tpu"`), verified against upstream `OpenMOSS-Team/MOSS-Image-Base` modeling semantics and designed for Colab `TPU v6e-1` (32 GiB HBM) and staged / decoupled `TPU v5e` workflows.

## Architecture Summary

1. **Stage 1 — Exact `Qwen3.5-9B` Text Encoding (15.6 GiB BF16)**:
   - Runs unpadded individual prompt forwards on `text_encoder` to preserve Qwen3.5 Gated DeltaNet (GDN) numerical semantics (`pipeline_moss_image.py`).
   - Detaches `[1, L_true, 4096]` hidden states to CPU and stores them in an 8-entry LRU prompt cache, or exports them via `--mode encode-only` for decoupled execution on 16 GiB TPUs.
2. **Stage 2 & 3 — Static-Bucketed `MOSSImageTransformer` (5B DiT) + FP32 `FLUX.2` VAE**:
   - Offloads `text_encoder` before loading `transformer` (~10.2 GiB BF16) and `vae` (~0.3 GiB FP32) onto TPU (`--offload model` / staged), leaving them resident across subsequent seeds and cached prompts.
   - Replaces runtime `float64` timestep sinusoid generation with a precomputed `float64 -> float32` buffer on TPU (zero host sync, zero `float64` MXU fallback).
   - Replaces `mask[index].bool()` data-dependent slicing and `.all()` / `.any()` host-sync checks with static bucket padding (`512, 1024, 2048, 4096`) + `F.scaled_dot_product_attention(..., attn_mask=...)` + cumulative masked RoPE offsets, achieving mathematical parity with unpadded execution while maintaining 100% static tensor shapes for `torch.compile(..., backend="tpu", dynamic=False, fullgraph=True)`.

## Quick Start (Google Colab `TPU v6e-1`)

```bash
bash scripts/setup-tpu.sh
python3 scripts/run_tpu.py --mode probe
python3 scripts/run_tpu.py --mode generate --prompt "A ceramic teapot on linen" --compile
```

### Eager / Defer-and-Fuse / Static Compile Benchmarking

```bash
python3 scripts/run_tpu.py \
  --mode benchmark \
  --bench-modes eager,defer_and_fuse,compile \
  --steps 28 \
  --save-report outputs/tpu_benchmark.json
```

### Decoupled Stage 1 / Stage 2 (for 16 GiB TPU v5e-1)

```bash
python3 scripts/run_tpu.py --mode encode-only --prompt "..." --save-embeds outputs/embeds.pt
python3 scripts/run_tpu.py --mode generate --prompt-embeds outputs/embeds.pt --offload none --compile
```

### Persistent Compilation Cache

Set these before running to reuse XLA executables across restarts:

```bash
export TORCH_TPU_TIER2_COMPILATION_CACHE="${HOME}/.cache/torch_tpu/tier2"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${HOME}/.cache/torch_tpu/tier3"
```

## Notes on Pallas SplashAttention

`tpu_runtime.py` lowers attention through `F.scaled_dot_product_attention(..., attn_mask=...)` for exact mathematical parity across padded prompt buckets. Custom Pallas `splash_attention_kernel` bridges via `torch.tpu.pallas.jax_op` (following `google-pytorch/torch_tpu` FLUX optimization recipes) require fixed block-aligned masks (`128`-token alignment) and live TPU verification and are left as an explicit Phase 2 hardware tuning path.
