<div align="center">
  <p>
    <img src="assets/lockup-light.svg#gh-light-mode-only" alt="MOSS-Image" width="520">
    <img src="assets/lockup-dark.svg#gh-dark-mode-only" alt="MOSS-Image" width="520">
  </p>
  <p>
    <a href="docs/tpu.md"><img alt="Google Cloud TPU · TorchTPU" src="https://img.shields.io/badge/TPU-TorchTPU%20(v6e--1)-4285F4?style=flat-square&amp;logo=googlecloud&amp;logoColor=white"></a>
    &nbsp;
    <a href="notebooks/moss_image_torchtpu_colab.ipynb"><img alt="Colab Notebook" src="https://img.shields.io/badge/Colab-TPU%20Notebook-F9AB00?style=flat-square&amp;logo=googlecolab&amp;logoColor=white"></a>
    &nbsp;
    <a href="https://github.com/azure-swing/MOSS-Image/discussions"><img alt="Fork Discussions" src="https://img.shields.io/badge/GitHub-Discussions-181717?style=flat-square&amp;logo=github"></a>
    &nbsp;
    <a href="https://github.com/OpenMOSS/MOSS-Image"><img alt="Upstream OpenMOSS/MOSS-Image" src="https://img.shields.io/badge/Upstream-OpenMOSS%2FMOSS--Image-0f766e?style=flat-square"></a>
    &nbsp;
    <a href="https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base"><img alt="Hugging Face · MOSS-Image-Base" src="https://img.shields.io/badge/🤗%20Hugging%20Face-MOSS--Image--Base-ffc107?style=flat-square"></a>
    &nbsp;
    <a href="https://arxiv.org/abs/2610.23374"><img alt="arXiv: 2610.23374" src="https://img.shields.io/badge/arXiv-2610.23374-b31b1b?style=flat-square&amp;logo=arxiv&amp;logoColor=white"></a>
  </p>
  <p>
    <strong>English</strong> &nbsp; | &nbsp; <a href="README_ZH.md">简体中文</a>
  </p>
</div>

# MOSS-Image on Google Cloud TPU (TorchTPU)

This fork (`azure-swing/MOSS-Image`, branch `feat/torchtpu-support`) runs **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** natively in PyTorch on **Google Cloud TPU (`TPU v6e-1`, `31.24 GiB` HBM)**. It supports full end-to-end BF16 image generation with FP32 VAE decoding and matches original CUDA BF16 image quality.

> [!IMPORTANT]
> **Memory Recommendation (`< 40 GB` HBM / VRAM)**: When running the **BF16** model on devices with **less than 40 GB of memory** (such as `TPU v6e-1` with `31.24 GiB`), **use `offload`** (`--offload model`, or `--offload auto`, which automatically turns offload on whenever device memory is below `40 GiB`). Keeping both the 9B text encoder and the 5B diffusion model + VAE resident at once exceeds `31.24 GiB`.

## Contents

- [Image Quality vs. Original CUDA Reference](#image-quality-comparison)
- [Image Generation Speed, Operator Efficiency & Offload Performance](#speed-and-efficiency)
- [Quick Start (Colab `TPU v6e-1`)](#quick-start)
- [Technical Details & Architecture](#technical-details)
- [Repository Layout (TPU Additions)](#repository-layout)
- [Next Optimization Roadmap](#next-roadmap)
- [Upstream MOSS-Image & Citation](#upstream-and-citation)

<a id="image-quality-comparison"></a>

## Image Quality vs. Original CUDA Reference

Generated on `TPU v6e-1` (`848×1200`, `55` steps, `guidance_scale = 3.0`, `seed = 136879772936584`) using the exact same prompt and seed as the original CUDA BF16 reference:

| Original CUDA BF16 Reference | TPU `v6e-1` BF16 (`defer_and_fuse` + Pallas) |
| :---: | :---: |
| <img src="assets/comparison/cuda_bf16_ref.png" alt="Original CUDA BF16 Reference" width="380"> | <img src="assets/comparison/tpu_v6e_pallas.png" alt="TPU v6e-1 BF16 Generated Image" width="380"> |

- **TPU (`defer_and_fuse` + Pallas) vs. Original CUDA BF16**: Cosine similarity **`0.998019`**, mean pixel difference **`6.151 / 255`**, **`PSNR = 26.80 dB`** (TPU without Pallas vs. CUDA BF16: `cos = 0.997435`).
- **TPU Pallas vs. TPU Default Text Encoder**: Cosine similarity **`0.999442`**, mean pixel difference **`2.461 / 255`**, **`PSNR = 32.31 dB`**.
- **`drop_tpu` Offload vs. `d2h` Offload**: Exact pixel match (`cosine_similarity = 1.0`, `mean_abs_diff = 0.0`, `PSNR = Infinity`).

<a id="speed-and-efficiency"></a>

## Image Generation Speed, Operator Efficiency & Offload Performance

All measurements below were collected on live Google Colab **`TPU v6e-1`** (`31.24 GiB` HBM, `172.95 GiB` host RAM, `848×1200`, `55` steps, `guidance_scale = 3.0`, `OpenMOSS-Team/MOSS-Image-Base`).

### 1. Image Generation & VAE Speed (All Scenarios)

| Scenario | Total Time | Breakdown (Denoising / VAE / Overhead) | Peak HBM |
| --- | ---: | --- | ---: |
| **Warm Repeat / Same Seed (`defer_and_fuse`)** | **`16.72s`** | `16.43s` denoising (`298.8 ms/step`) + **`~0.29s` VAE decode & postprocess** | `16.33 GiB` |
| **10-Seed Warm Mean (Runs 2–10)** | **`16.73s`** | `16.43s` denoising (`298.7 ms/step`), `246.17s` total for 10 images incl. cold Run 1 (`95.56s`) | `16.25 GiB` |
| **Warm Repeat without Fusion (`eager`)** | `49.08s` | `48.61s` denoising (`883.9 ms/step`) + `~0.47s` VAE decode & postprocess | `24.47 GiB` |
| **Cold Diffusion + VAE (`--prompt-embeds` cached)** | `35.56s` | `16.61s` denoising + first-run XLA compile & VAE (`70.20s` in `eager`) | `14.96 GiB` |
| **Changing Prompts (`drop_tpu` + Async H2D Offload)** | **`37.71s`** | `20.75s` model switch + `16.54s` denoising (`300.8 ms/step`) + `~0.42s` encode & VAE | `20.67 GiB` |
| **Fresh CLI with Disk Prompt Cache Hit** | `41.62s` | `2.20s` model load + `16.60s` denoising + transfer & VAE | `15.24 GiB` |
| **Changing Prompts (`d2h` + Async H2D Offload)** | `84.41s` | `67.08s` model switch + `16.55s` denoising (`300.9 ms/step`) + `~0.78s` encode & VAE | `20.67 GiB` |
| **Cold First Run (Prompt Cache Miss, pre-Plan A)** | `88.42s` | `16.59s` denoising (`301.7 ms/step`) + cold text encode & sync offload (`115.17s` w/o Pallas) | `15.79 GiB` |
| **Strict Eager Cold Baseline (`default`)** | `124.19s` | `52.98s` denoising (`963.2 ms/step`) + cold text encode & sync offload | `24.50 GiB` |

### 2. Theoretical Operator Efficiency (`TPU v6e-1` Peak: `918 TFLOPs` BF16)

| Component / Mode | Achieved Throughput / Latency | Efficiency vs. Peak / Speedup |
| --- | ---: | --- |
| **Diffusion (`defer_and_fuse`, 55 steps)** | **`352.8 TFLOPs`** (`298.8 ms/step`) | **`38.4%` MFU** (Model FLOPs Utilization) |
| **Diffusion (`eager` / `DEFER_NEVER`, 55 steps)** | `119.3 TFLOPs` (`883.9 ms/step`) | `13.0%` MFU (`defer_and_fuse` is **2.96x** faster) |
| **Text Encoder GDN Kernel (`deltanet-tpu` Pallas)** | **`1.082 ms`** recurrence / **`2.030 ms`** full layer | **`4.57x`** faster recurrence / **`3.08x`** faster layer vs. HF `eager` |
| **Text Encoder Forward (`eager+gdn_pallas`)** | **`0.310s`** cold / **`0.161s`** distinct warm / `0.092s` repeat | Down from `1.993s` cold / `1.168s` distinct warm in HF `eager` |

### 3. Offload & Stage-Switching Speeds (`< 40 GB` Memory Mode)

Because BF16 `Qwen3.5-9B` (`15.63 GiB`) and `MOSSImageTransformer` + `VAE` (`9.70 GiB`) plus runtime buffers exceed `31.24 GiB`, `--offload model` swaps stages between CPU RAM and TPU HBM when prompts change. With **Plan A (`drop_tpu` + async `non_blocking` H2D)**, CPU keeps read-only weight snapshots so off-stage TPU weights are dropped directly without copying back to CPU (D2H = `0.0s`):

| Offload Mode | Run 1 Cold Switch | Run 2 Prompt Switch (`diffusion -> text -> diffusion`) | Run 2 Total Generation | Switch Speedup |
| --- | ---: | ---: | ---: | ---: |
| **Pre-async `d2h` baseline** *(4-step)* | `70.00s` | `100.73s` (`50.25s` D2H + `50.48s` H2D) | `103.29s` *(4 steps)* | `1.00x` |
| **`drop_tpu` + sync H2D** *(4-step)* | — | `48.96s` (`0.0s` D2H + `48.03s` H2D) | `50.62s` *(4 steps)* | `2.06x` |
| **`d2h` + async H2D** *(55-step)* | `49.00s` (`29.67s` D2H + `19.33s` H2D) | `67.08s` (`49.44s` D2H + `17.63s` H2D) | `84.41s` | `1.50x` |
| **`drop_tpu` + async H2D** *(55-step)* | **`18.68s`** (`0.0s` D2H + `18.19s` H2D) | **`20.75s`** (`0.0s` D2H + `0.99s` drop/GC + `19.75s` H2D) | **`37.71s`** | **`3.23x`** vs. async `d2h` (**`4.86x`** vs. pre-async `d2h`) |

<a id="quick-start"></a>

## Quick Start (Colab `TPU v6e-1`)

### 1. Environment Setup

```bash
git clone -b feat/torchtpu-support https://github.com/azure-swing/MOSS-Image.git
cd MOSS-Image
bash scripts/setup-tpu.sh

# Optional: install deltanet-tpu without overwriting TorchTPU's pinned libtpu
pip install --no-deps deltanet-tpu

# Recommended persistent XLA compilation caches
export TORCH_TPU_TIER2_COMPILATION_CACHE="${HOME}/.cache/torch_tpu/tier2"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${HOME}/.cache/torch_tpu/tier3"
```

You can also open [`notebooks/moss_image_torchtpu_colab.ipynb`](notebooks/moss_image_torchtpu_colab.ipynb) directly in Google Colab.

### 2. Probe, Generate, and Benchmark

```bash
# Verify TorchTPU device + matmul smoke test
python3 scripts/run_tpu.py --mode probe

# Recommended fast generation (auto-enables offload when HBM < 40 GiB)
python3 scripts/run_tpu.py \
  --mode generate \
  --eager-mode defer_and_fuse \
  --gdn-pallas \
  --prompt-cache-dir outputs/prompt_cache \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --prompt "A ceramic teapot on linen" \
  --output outputs/moss_tpu.png

# A/B stage-switch benchmark (d2h vs drop_tpu)
python3 scripts/run_tpu.py \
  --mode stage-switch-bench \
  --switch-modes d2h,drop_tpu \
  --eager-mode defer_and_fuse \
  --gdn-pallas \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --save-report outputs/stage_switch_55.json
```

See **[docs/tpu.md](docs/tpu.md)** for full CLI flags, `--mode encode-only`, `--mode encode-bench`, and detailed bring-up logs.

---

<a id="technical-details"></a>

## Technical Details & Architecture

```text
Prompt text
   │
   ├─► Disk cache hit (--prompt-cache-dir) / 8-entry CPU LRU hit ──────────┐
   │                                                                       ▼
   └─► Stage 1 [DEFER_NEVER + deltanet-tpu Pallas] (Qwen3.5-9B BF16) ──► CPU embeds [1, L, 4096]
                                                                           │
   Stage 2 [defer_and_fuse] (MOSSImageTransformer BF16, static buckets 512/1024/2048/4096)
     • Precomputed FP32 timestep sinusoid + expanded [1, 1, Q, S] SDPA masks
                                                                           │
                                                                           ▼
   Stage 3 [defer_and_fuse] (AutoencoderKLFlux2 FP32 + NHWC-to-NCHW GroupNorm hook) ──► PNG
```

1. **Staged Three-Phase Runtime (`tpu_runtime.py`)**:
   - **Stage 1 (`Qwen3.5-9B`, `BF16`, `15.63 GiB`)**: Executes unpadded prompt encoding on TPU to preserve causal Gated DeltaNet (GDN) semantics, detaches `[1, L_true, 4096]` hidden states to CPU, and caches them in an 8-entry LRU (`_prompt_cache`) plus optional `--prompt-cache-dir` disk storage.
   - **Stage 2 & 3 (`MOSSImageTransformer` ~5.08B `BF16` / `9.4 GiB` + `AutoencoderKLFlux2` `FP32` / `~0.3 GiB`)**: Under `--offload model` (selected automatically by `--offload auto` when `tpu_budget_gib() < 40`), `text_encoder` and `transformer` swap places across prompt changes, while `vae` stays resident on TPU once loaded across `diffusion <-> text` transitions.
2. **Zero-D2H `drop_tpu` & Async `non_blocking` H2D (`tpu_runtime.py`)**:
   - **`drop_tpu` (`--stage-switch-mode auto` / `drop_tpu`)**: Retains CPU weight snapshots in host RAM (default on hosts with `>= 36 GiB` RAM) and releases off-stage TPU parameters/buffers to zero-byte `meta` tensors via `_release_module_to_meta` (`0.0s` D2H), then restores directly via `_restore_module_from_cpu_snapshot`. On a `172.95 GiB` host, process RSS after Run 2 is `30.91 GiB` with `138.24 GiB` free.
   - **Async `non_blocking=True` H2D**: Batches `.to(tpu, non_blocking=True)` transfers under one `torch.tpu.synchronize()`, raising synthetic 400-tensor (`12.50 GiB`) H2D throughput from `0.53 GiB/s` (`23.78s`) to `1.96 GiB/s` (`6.38s`; pre-pinned reached `2.05 GiB/s` / `6.10s`).
   - **Residual `~5.0 GiB` HBM Plateau**: Both `d2h` (`5.00–5.03 GiB`) and `drop_tpu` (`5.01–5.03 GiB`) retain a constant `~5.0 GiB` floor in `torch.tpu.memory_allocated()` after the first full pass even with `0` live Python TPU tensors (`gc.get_objects()`), reflecting a one-time TorchTPU/PJRT C++ runtime/executable buffer allocation.
3. **Stage-1 Pallas GDN Bridge (`tpu_qwen35.py`) & Split Eager Modes (`--text-eager-mode auto`)**:
   - Routes all 24 `GatedDeltaNet` linear-attention layers (`Hk=16, Hv=32, D=128`, `use_cache=False`) through `torch.tpu.pallas.jax_op` -> `deltanet_tpu.chunk_gdn` with `128`-aligned causal static buckets (`128, 256, 384, 512, 768, 1024`). Zero fallbacks (`96/96` in `encode-bench`, `48/48` in CFG generation).
   - Stage 1 automatically runs in `default` (`DEFER_NEVER`) + Pallas (`0.310s` cold / `0.161s` distinct warm vs. `6.672s` / `1.859s` in `defer_and_fuse+gdn_pallas`), while Stage 2/3 runs in `defer_and_fuse`.
4. **TPU `v6e-1` Hardware Fixes (`tpu_runtime.py`, `scripts/run_tpu.py`)**:
   - **4D Broadcast SDPA Mask `NaN` Fix**: Expands static `text_attn_mask` and `joint_attn_mask` from `[1, 1, 1, S]` to `[1, 1, Q, S]` to avoid a `torch_tpu` Mosaic/cuDNN bug that corrupted `S=30` and emitted `NaN`s at `S >= 512`.
   - **VAE `GroupNorm` Contiguity Pre-Hook**: `ensure_group_norm_contiguous(vae)` converts NHWC convolution outputs to contiguous NCHW before `F.group_norm` calls `.view(...)`.
   - **`--compile` (`e0200`) Isolation**: `transformer.compile(backend="tpu", fullgraph=True)` hits `e0200: RuntimeUnexpectedCoreHalt` (`vmem_address_out_of_range`) at the first `FlowMatchEulerDiscreteScheduler.step()` `.nonzero()` host sync, so `--compile` remains opt-in (`False` by default).

### Stage-1 `encode-bench` Microbenchmarks (`329` pos + `24` neg tokens, pre-async H2D)

| Mode | H2D Transfer | Cold Encode (329 tok) | Neg Encode (24 tok) | Distinct-Prompt Warm | Repeat Warm | Peak HBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `eager` (`DEFER_NEVER`) | `28.653s` | `1.993s` | `0.908s` | `1.168s` | `0.164s` | `15.79 GiB` |
| `defer_and_fuse` | `20.577s` | `11.023s` | `4.659s` | `4.901s` | `0.072s` | `15.73 GiB` |
| **`eager+gdn_pallas`** | `23.860s` | **`0.310s`** | **`0.930s`** | **`0.161s`** | `0.092s` | `15.83 GiB` |
| `defer_and_fuse+gdn_pallas` | `26.311s` | `6.672s` | `1.684s` | `1.859s` | **`0.049s`** | `15.89 GiB` |

*(Full 32-layer embedding parity for `--gdn-pallas` vs. HF: positive `cos = 0.999848`, negative `cos = 0.999758`, zero `NaN`/`Inf`.)*

### CPU Dry-Run & Unit Tests

```bash
ruff check tpu_runtime.py tpu_qwen35.py scripts/run_tpu.py scripts/bench_qwen35_gdn_tpu.py tests/test_tpu_runtime.py tests/test_tpu_qwen35.py
python3 -m unittest discover -s tests -p "test_tpu*.py" -v
```

<a id="repository-layout"></a>

## Repository Layout (TPU Additions)

| File | Role |
| --- | --- |
| [`tpu_runtime.py`](tpu_runtime.py) | Three-stage TPU pipeline (`StagedTPUImagePipeline`), `drop_tpu` + async H2D + resident VAE, split eager contexts, static bucketing, `[1, 1, Q, S]` SDPA mask patch, FP32 sinusoid, `GroupNorm` hook |
| [`tpu_qwen35.py`](tpu_qwen35.py) | `torch.tpu.pallas.jax_op` bridge to `deltanet_tpu.chunk_gdn`, `128`-aligned causal padding, persistent CPU prompt cache |
| [`scripts/run_tpu.py`](scripts/run_tpu.py) | TPU CLI (`probe`, `encode-only`, `encode-bench`, `generate`, `benchmark`, `stage-switch-bench`) |
| [`scripts/bench_qwen35_gdn_tpu.py`](scripts/bench_qwen35_gdn_tpu.py) | Standalone GDN recurrence & `Qwen3_5GatedDeltaNet` layer latency/parity A/B benchmark |
| [`scripts/setup-tpu.sh`](scripts/setup-tpu.sh) | TorchTPU `0.2.0` + PyTorch CPU environment installer for Colab / TPU VMs |
| [`notebooks/moss_image_torchtpu_colab.ipynb`](notebooks/moss_image_torchtpu_colab.ipynb) | End-to-end Google Colab TPU walkthrough |
| [`docs/tpu.md`](docs/tpu.md) | Detailed `TPU v6e-1` bring-up log, error analyses, and Plan A stage-switch benchmark results |
| [`gradio_app.py`](gradio_app.py) / [`inference_runtime.py`](inference_runtime.py) | Device routing & provenance tracking for CUDA, Apple Silicon MPS, and TorchTPU |
| [`tests/test_tpu_runtime.py`](tests/test_tpu_runtime.py) / [`tests/test_tpu_qwen35.py`](tests/test_tpu_qwen35.py) | 32 CPU unit & parity tests for bucketing, patches, Pallas bridge, prompt cache, `drop_tpu`, and split eager CLI |

<a id="next-roadmap"></a>

## Next Optimization Roadmap

Ongoing performance discussions are tracked in **[GitHub Discussions #5](https://github.com/azure-swing/MOSS-Image/discussions/5)** (`Ideas`):
1. Replacing `[1, 1, 4487, 4487]` 4D bool-masked `SDPA` (`~2.38 ms/layer`) with Pallas `SplashAttention` to push Stage-2 DiT MFU beyond `38.4%`.
2. Removing the per-step `.nonzero()` D2H sync in `FlowMatchEulerDiscreteScheduler.step()` and bisecting the `e0200` VMEM fault to enable `transformer.compile(backend="tpu", fullgraph=True)`.
3. Coalescing CFG (`B=2`) across the 41 unskipped DiT blocks and adding smaller `128, 256` default prompt buckets.
4. Quantizing `Qwen3.5-9B` (INT8/W4) so all stages fit inside `31.24 GiB` (`--offload none`), eliminating prompt-switch H2D transfers altogether.

<a id="upstream-and-citation"></a>

## Upstream MOSS-Image & Citation

This work builds on **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)**. For base model cards, CUDA/MPS usage, Gradio, ComfyUI integration, JSON layout generation, and prompt enhancer skills, see upstream documentation:

- [User Guide (EN)](USERGUIDE.md) · [用户指南 (中文)](USERGUIDE_ZH.md)
- [Apple Silicon / MPS](docs/macos.md) · [ComfyUI Integration](comfyui/README.md) · [Prompt Enhancer Skill](skills/moss-image-prompt-enhancer/SKILL.md)
- Models: [MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) · [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) · [Technical Report (arXiv:2610.23374)](https://arxiv.org/abs/2610.23374)

```bibtex
@misc{ai2026mossimagetechnicalreport,
      title={MOSS-Image Technical Report},
      author={MOSI. AI},
      year={2026},
      eprint={2610.23374},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.23374},
}
```

Inference code is licensed under the [Apache License 2.0](LICENSE). Thanks to the OpenMOSS, Diffusers, Qwen, FLUX, TorchTPU, and `deltanet-tpu` teams.
