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

This fork (`azure-swing/MOSS-Image`, branch `feat/torchtpu-support`) adapts **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** (`Qwen3.5-9B` text encoder + 42-layer 5B `MOSSImageTransformer` + FP32 `AutoencoderKLFlux2` VAE) to run natively in PyTorch on **Google Cloud TPU (`torch_tpu`, Colab `TPU v6e-1` / `31.24 GiB` HBM)**.

End-to-end `BF16` / `FP32` image generation, Stage-1 `Qwen3.5-9B` Pallas Gated DeltaNet acceleration (`deltanet-tpu`), split per-stage eager execution, persistent disk prompt caching, and numerical parity vs. CUDA BF16 and Hugging Face references are implemented and verified on live `TPU v6e-1` hardware.

## Contents

- [Summary of Current Work](#summary-of-current-work)
- [Hardware-Verified Benchmarks (`TPU v6e-1`)](#hardware-verified-benchmarks)
- [Quick Start (Colab `TPU v6e-1`)](#quick-start)
- [Runtime Architecture & Hardware Fixes](#runtime-architecture)
- [Repository Layout (TPU Additions)](#repository-layout)
- [Next Optimization Roadmap](#next-roadmap)
- [Upstream MOSS-Image & Citation](#upstream-and-citation)

<a id="summary-of-current-work"></a>

## Summary of Current Work

1. **Staged Three-Phase TPU Runtime (`tpu_runtime.py`)**
   - **Stage 1 (`Qwen3.5-9B`, `BF16`, ~18 GB checkpoint)**: Executes unpadded prompt encoding on TPU to preserve causal Gated DeltaNet (GDN) semantics, caches `[1, L_true, 4096]` hidden states in an 8-entry in-memory LRU cache (`_prompt_cache`), and supports cross-process persistent disk caching via `--prompt-cache-dir`.
   - **Stage 2 & 3 (`MOSSImageTransformer` ~5.08B `BF16` + `AutoencoderKLFlux2` `FP32`)**: Under `--offload model` / `auto` (which switches to `none` only when device HBM budget is `>= 40 GiB`), `text_encoder` is offloaded back to CPU before loading DiT and VAE onto TPU. Repeated seeds or cached prompts keep Stage 2/3 resident on HBM without reloading `Qwen3.5-9B`.
2. **Stage-1 `Qwen3.5-9B` Pallas GDN Bridge (`tpu_qwen35.py`)**
   - Routes all 24 `GatedDeltaNet` linear-attention layers (`Hk=16, Hv=32, D=128`, `use_cache=False`) through `torch.tpu.pallas.jax_op` -> `deltanet_tpu.chunk_gdn` with `128`-aligned causal static buckets (`128, 256, 384, 512, 768, 1024`; larger sequences align up to multiples of `128`).
   - Verified zero fallback calls (`96/96` in `encode-bench`, `48/48` in CFG image generation) and handles single-tensor returns (`output_final_state=False`) with `is_inference=True` in `deltanet-tpu 0.3.0`.
3. **Split Per-Stage Eager Modes (`--text-eager-mode auto`)**
   - Profiling showed Stage 1 (`Qwen3.5-9B`) and Stage 2/3 (`MOSSImageTransformer` + `VAE`) prefer opposite TorchTPU eager modes:
     - **Stage 1** runs fastest cold and across changing prompt lengths in `default` (`DEFER_NEVER`) + `--gdn-pallas` (`0.310s` cold / `0.161s` distinct warm vs. `6.672s` cold / `1.859s` distinct warm in `defer_and_fuse+gdn_pallas`).
     - **Stage 2/3** repeats the same DiT shape across 50–55 denoising steps and runs ~3x faster with ~8 GiB lower peak HBM in `defer_and_fuse` (`16.724s` Run 2 warm total / `16.434s` denoising / `16.331 GiB` vs. `49.082s` / `48.612s` denoising / `24.473 GiB` in `default` eager).
   - Passing `--eager-mode defer_and_fuse --gdn-pallas` with `--text-eager-mode auto` (the CLI default) automatically executes Stage 1 in `default` (`DEFER_NEVER`) and Stage 2/3 in `defer_and_fuse`.
4. **Live `TPU v6e-1` Bugs Diagnosed & Fixed**
   - **4D Broadcast SDPA Mask `NaN` (All-Black Output)**: Expanding static `text_attn_mask` and `joint_attn_mask` from `[1, 1, 1, S]` to `[1, 1, Q, S]` avoids a `torch_tpu` Mosaic/cuDNN lowering bug that corrupted small `S` (`S=30`) and emitted `NaN`s at `S >= 512`.
   - **VAE `GroupNorm` Channels-Last Crash**: Added `ensure_group_norm_contiguous(vae)` pre-hook to convert `torch_tpu` NHWC convolution outputs to contiguous NCHW before `F.group_norm` invokes `.view(...)`.
   - **`--compile` (`e0200`) Isolation**: Documented `e0200: RuntimeUnexpectedCoreHalt` (`vmem_address_out_of_range`) surfaced at the first `FlowMatchEulerDiscreteScheduler.step()` `.nonzero()` host sync under `transformer.compile(backend="tpu", fullgraph=True)`, and kept `--compile` opt-in (`False` by default).

<a id="hardware-verified-benchmarks"></a>

## Hardware-Verified Benchmarks (`TPU v6e-1`)

Measured on live Google Colab `TPU v6e-1` (`31.24 GiB` HBM, `torch 2.14.1+cpu`, `torch_tpu 0.2.0`, `libtpu 0.0.50`, `jax 0.11.2`, `transformers 5.22.1`, `deltanet-tpu 0.3.0`, `OpenMOSS-Team/MOSS-Image-Base`, Tier-3 XLA cache enabled).

### Stage 1: `Qwen3.5-9B` GDN Microbenchmarks & Breakdown

Standalone recurrence & single-layer A/B (`B=1, L=329 -> bucket 384, Hk=16, Hv=32, D=128`, `bench_qwen35_gdn_tpu.py`):

| Benchmark | HF `eager` | HF `defer_and_fuse` | `deltanet-tpu` Pallas | Parity vs. Reference |
| --- | --- | --- | --- | --- |
| GDN recurrence warm median | `4.946 ms` | `4.651 ms` | **`1.082 ms`** (`4.57x` / `4.30x`) | `cos = 0.999992`, `max_abs = 0.000437` |
| Full `Qwen3_5GatedDeltaNet` layer | `6.244 ms` | `3.947 ms` | **`2.030 ms`** (`8/8` Pallas, `0` fallbacks) | `cos = 0.999897`, `max_abs = 0.003906` |

Full 32-layer `Qwen3.5-9B` `--mode encode-bench` (`329` positive + `24` negative tokens):

| Mode | H2D Transfer | Cold Encode (329 tok) | Neg Encode (24 tok) | Distinct-Prompt Warm | Repeat Warm | Peak HBM |
| --- | --- | --- | --- | --- | --- | --- |
| `eager` (`DEFER_NEVER`) | `28.653s` | `1.993s` | `0.908s` | `1.168s` | `0.164s` | `15.79 GiB` |
| `defer_and_fuse` | `20.577s` | `11.023s` | `4.659s` | `4.901s` | `0.072s` | `15.73 GiB` |
| **`eager+gdn_pallas`** | `23.860s` | **`0.310s`** | **`0.930s`** | **`0.161s`** | `0.092s` | `15.83 GiB` |
| `defer_and_fuse+gdn_pallas` | `26.311s` | `6.672s` | `1.684s` | `1.859s` | **`0.049s`** | `15.89 GiB` |

- `encode-bench` isolated **`h2d_transfer_seconds` (`20.58s–28.65s`)** as the dominant remaining cold Stage-1 bottleneck (`model_load_seconds` via mmap was `0.247s`), while actual Pallas forward computation took `0.092s–0.310s`.
- Full 32-layer embedding parity (`--gdn-pallas` vs. HF): positive `cos = 0.999848`, negative `cos = 0.999758`, zero `NaN`/`Inf`.

### Stage 2/3 & End-to-End Image Generation (`848×1200`, 55 Steps, CFG `3.0`)

DiT + VAE benchmark (`--prompt-embeds` cached, `uncond_skip_layers=(22,)`, `329` prompt tokens -> static bucket `512`, `S_joint = 4487`):

| Execution Mode | Cold Run 1 Total | Warm Run 2 Total | Warm Run 2 Denoising (55 steps) | Achieved BF16 Compute (`918 TF` Peak) | Peak HBM |
| --- | --- | --- | --- | --- | --- |
| `eager` (`DEFER_NEVER`) | `70.201s` | `49.082s` | `48.612s` (`883.85 ms/step`) | `119.3 TFLOPs` (**`13.0%` MFU**) | `24.473 GiB` |
| **`defer_and_fuse`** | `35.556s` | **`16.724s`** | **`16.434s` (`298.79 ms/step`)** | **`352.8 TFLOPs` (`38.4%` MFU)** | **`16.331 GiB`** |

Same-seed end-to-end CLI generation (`seed = 136879772936584`, `848×1200`, 55 steps, `defer_and_fuse`, warm Tier-3 DiT cache):

- `defer_and_fuse` + HF Qwen3.5 (`--no-gdn-pallas`): `115.169s` elapsed (`denoising = 24.680s`, `peak = 15.693 GiB`, `load = 27.22s`).
- `defer_and_fuse` + `--gdn-pallas` (`pcache` miss): `88.421s` elapsed (`denoising = 16.593s` / `301.69 ms/step`, `peak = 15.794 GiB`, `48/48` Pallas calls).
- `defer_and_fuse` + `--gdn-pallas --prompt-cache-dir` (**cache hit**): **`41.622s`** elapsed (`load = 2.20s`, `denoising = 16.603s`, `peak = 15.237 GiB`).
- **Same-seed pixel parity (`848×1200`)**: TPU HF vs. TPU Pallas `cos = 0.999442` (`mean_abs = 2.461/255`, `PSNR = 32.31 dB`); vs. CUDA BF16 reference `cos = 0.997435` (TPU HF) / `0.998019` (TPU Pallas).

*(Unit tests covering the latest split `--text-eager-mode auto` transition passed locally on CPU — `ruff check` clean and `28/28` tests in `tests/test_tpu*.py`; live Colab hardware tests above were run immediately prior on `a4b1d85` and earlier `feat/torchtpu-support` commits.)*

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

# Recommended fast split-stage E2E generation:
# Stage 1 automatically runs in `default` + Pallas (--text-eager-mode auto)
# Stage 2/3 DiT+VAE runs in `defer_and_fuse`
python3 scripts/run_tpu.py \
  --mode generate \
  --eager-mode defer_and_fuse \
  --gdn-pallas \
  --prompt-cache-dir outputs/prompt_cache \
  --height 1024 --width 1024 --steps 50 --seed 42 \
  --prompt "A ceramic teapot on linen" \
  --output outputs/moss_tpu.png

# Standalone Stage-1 GDN kernel + layer A/B benchmark (no 18 GB download required)
python3 scripts/bench_qwen35_gdn_tpu.py --include-layer --save-report outputs/gdn_ab.json

# Stage-1 encode breakdown (H2D / cold / neg / distinct warm / repeat warm)
python3 scripts/run_tpu.py \
  --mode encode-bench \
  --bench-modes eager,defer_and_fuse,eager+gdn_pallas,defer_and_fuse+gdn_pallas \
  --runs 3 \
  --save-report outputs/encode_bench.json
```

See **[docs/tpu.md](docs/tpu.md)** for full flags, `--mode encode-only` decoupled workflows, persistent cache details, and full error logs.

<a id="runtime-architecture"></a>

## Runtime Architecture & Hardware Fixes

```text
Prompt text
   │
   ├─► Disk cache hit (--prompt-cache-dir) / LRU hit ─────────────────────┐
   │                                                                      ▼
   └─► Stage 1 [DEFER_NEVER + deltanet-tpu Pallas] (Qwen3.5-9B BF16) ──► CPU embeds [1, L, 4096]
                                                                          │  (Stage 1 offloaded)
   Stage 2 [defer_and_fuse] (MOSSImageTransformer BF16, static buckets 512/1024/2048/4096)
     • Precomputed FP32 timestep sinusoid + expanded [1, 1, Q, S] SDPA masks
                                                                          │
                                                                          ▼
   Stage 3 [defer_and_fuse] (AutoencoderKLFlux2 FP32 + NHWC-to-NCHW GroupNorm hook) ──► PNG
```

- **Why `--offload model` is required on `TPU v6e-1` (`31.24 GiB`)**: `Qwen3.5-9B` (`~18 GB` BF16) plus Stage 2/3 (`MOSSImageTransformer` ~9.5 GiB BF16 weights + FP32 VAE + XLA activations, peaking at `16.33 GiB` in `defer_and_fuse`) exceeds `31.24 GiB` HBM when kept resident together (`--offload auto` selects `"none"` at `>= 40 GiB`). Using `--prompt-cache-dir` skips Stage-1 loading and H2D transfers altogether once prompts are cached, leaving Stage 2/3 resident.
- **CPU Dry-Run & Unit Tests**:
  ```bash
  ruff check tpu_runtime.py tpu_qwen35.py scripts/run_tpu.py scripts/bench_qwen35_gdn_tpu.py tests/test_tpu_runtime.py tests/test_tpu_qwen35.py
  python3 -m unittest discover -s tests -p "test_tpu*.py" -v
  ```

<a id="repository-layout"></a>

## Repository Layout (TPU Additions)

| File | Role |
| --- | --- |
| [`tpu_runtime.py`](tpu_runtime.py) | Three-stage TPU pipeline (`StagedTPUImagePipeline`), split eager contexts, static prompt bucketing, `[1, 1, Q, S]` SDPA mask patch, FP32 timestep sinusoid, `GroupNorm` contiguity hook |
| [`tpu_qwen35.py`](tpu_qwen35.py) | `torch.tpu.pallas.jax_op` bridge to `deltanet_tpu.chunk_gdn`, `128`-aligned causal padding, persistent CPU prompt cache |
| [`scripts/run_tpu.py`](scripts/run_tpu.py) | TPU CLI (`probe`, `encode-only`, `encode-bench`, `generate`, `benchmark`) with `--text-eager-mode auto` |
| [`scripts/bench_qwen35_gdn_tpu.py`](scripts/bench_qwen35_gdn_tpu.py) | Standalone GDN recurrence & `Qwen3_5GatedDeltaNet` layer latency/parity A/B benchmark |
| [`scripts/setup-tpu.sh`](scripts/setup-tpu.sh) | TorchTPU `0.2.0` + PyTorch CPU environment installer for Colab / TPU VMs |
| [`notebooks/moss_image_torchtpu_colab.ipynb`](notebooks/moss_image_torchtpu_colab.ipynb) | End-to-end Google Colab TPU walkthrough |
| [`docs/tpu.md`](docs/tpu.md) | Detailed TPU v6e-1 bring-up log, error analyses, and profiling commands |
| [`gradio_app.py`](gradio_app.py) / [`inference_runtime.py`](inference_runtime.py) | Device routing & provenance tracking for CUDA, Apple Silicon MPS, and TorchTPU |
| [`tests/test_tpu_runtime.py`](tests/test_tpu_runtime.py) / [`tests/test_tpu_qwen35.py`](tests/test_tpu_qwen35.py) | 28 CPU unit & parity tests for bucketing, patches, Pallas bridge, prompt cache, and split eager CLI |

<a id="next-roadmap"></a>

## Next Optimization Roadmap

Ongoing performance discussions and the `TPU v6e-1` MFU analysis are tracked in **[GitHub Discussions #5](https://github.com/azure-swing/MOSS-Image/discussions/5)** (`Ideas`), focusing on:
1. Replacing `[1, 1, 4487, 4487]` 4D bool-masked `SDPA` (`~2.38 ms/layer`) with Pallas `SplashAttention` (block-aligned masks) to raise Stage-2 DiT MFU beyond `38.4%`.
2. Removing the per-step `.nonzero()` D2H sync in `FlowMatchEulerDiscreteScheduler.step()` and bisecting the `e0200` VMEM fault to enable `transformer.compile(backend="tpu", fullgraph=True)`.
3. Coalescing CFG (`B=2`) across the 41 unskipped DiT blocks and adding smaller `128, 256` default prompt buckets.
4. Eliminating the `~24s` Stage-1 H2D bottleneck via coalesced/pinned weight transfers or `Qwen3.5-9B` INT8/W4 quantization so all stages fit in `31.24 GiB` (`--offload none`).

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
