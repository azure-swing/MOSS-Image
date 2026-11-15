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
    <a href="README.md">English</a> &nbsp; | &nbsp; <strong>简体中文</strong>
  </p>
</div>

# MOSS-Image Google Cloud TPU (TorchTPU) 适配与优化

本仓库分支（`azure-swing/MOSS-Image`，默认分支 `feat/torchtpu-support`）基于 **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)**，完成了 **Google Cloud TPU（原生 PyTorch `torch_tpu` / Colab `TPU v6e-1` `31.24 GiB` HBM）** 的全链路适配、正确性修复、Stage 1 Pallas 算子加速与端到端性能基线验证。

目前 `Qwen3.5-9B` 文本编码（含 `deltanet-tpu` Pallas GDN 加速与持久化提示词缓存）、42 层 5B `MOSSImageTransformer` DiT 去噪与 `FP32` `AutoencoderKLFlux2` VAE 解码均已在实机 `TPU v6e-1` 上跑通，并完成了同随机种子下的 GPU BF16 / TPU HF / TPU Pallas 精度与图像一致性对齐。

## 目录

- [当前已完成工作概览](#summary-of-current-work)
- [TPU v6e-1 实测基准与数值对齐](#hardware-verified-benchmarks)
- [快速开始（Colab TPU v6e-1）](#quick-start)
- [分阶段运行时架构](#runtime-architecture)
- [新增与改动文件索引](#repository-layout)
- [后续优化方向](#next-roadmap)
- [上游项目、指南与引用](#upstream-and-citation)

<a id="summary-of-current-work"></a>

## 当前已完成工作概览

1. **三阶段硬件卸载运行时（[tpu_runtime.py](tpu_runtime.py)）**
   - **Stage 1（`Qwen3.5-9B`，BF16，约 18 GB）**：在 TPU 上执行无填充精确前向以保持 Gated DeltaNet（GDN）因果语义，结果转回 CPU 并接入 8 槽位内存 LRU 缓存（`_prompt_cache`），同时支持 `--prompt-cache-dir` 跨进程硬盘缓存（命中后可完全跳过模型加载、权重搬运与编码）。
   - **Stage 2 & 3（`MOSSImageTransformer` 约 5.08B BF16 + `AutoencoderKLFlux2` FP32）**：在 `--offload model` / `auto` 下（仅当 HBM 预算 `>= 40 GiB` 时自动切到 `none`），先将文本编码器卸回 CPU 再把 DiT 与 VAE 迁入 TPU；连续多 seed 出图或命中缓存时 Stage 2/3 常驻 HBM 无需重复搬运。
2. **Stage 1 `Qwen3.5-9B` Pallas GDN 算子桥（[tpu_qwen35.py](tpu_qwen35.py)）**
   - 通过 `torch.tpu.pallas.jax_op` 将 Hugging Face `Qwen3.5` 全 24 层 `GatedDeltaNet` 线性注意力（`Hk=16, Hv=32, D=128`, `use_cache=False`）路由到 `deltanet_tpu.chunk_gdn` Pallas 内核，采用 **128 倍数因果右填充静态分桶**（默认 `128, 256, 384, 512, 768, 1024`，超长按 128 向上对齐）。
   - 修复了 `deltanet-tpu 0.3.0` 在 `output_final_state=False` 时返回单张量的问题并启用 `is_inference=True`；端到端与单层测试均为 **0 fallback**（`encode-bench` `96/96`、CFG 生成 `48/48`）。
3. **分阶段独立 Eager 执行模式（`--text-eager-mode auto`）**
   - 实测显示 Stage 1 与 Stage 2/3 对 TorchTPU 的两种 Eager 模式诉求相反：
     - **Stage 1** 在 `default`（`DEFER_NEVER`）+ `--gdn-pallas` 下冷启动和变长度编码最快（冷态 `0.310s` / 变长热态 `0.161s`，而 `defer_and_fuse+gdn_pallas` 分别需 `6.672s` / `1.859s`）；
     - **Stage 2/3** 连续执行 50–55 步固定尺寸 DiT，在 `defer_and_fuse` 下热态提速约 **3 倍**、峰值 HBM 降约 **8 GiB**（`16.724s` Run 2 热跑总耗时 / `16.434s` 纯去噪 / `16.331 GiB` vs. `49.082s` / `48.612s` 纯去噪 / `24.473 GiB`）。
   - 现在使用 `--eager-mode defer_and_fuse --gdn-pallas` 时，`--text-eager-mode auto` 会自动让 Stage 1 走 `default`（`DEFER_NEVER`）+ Pallas，Stage 2/3 走 `defer_and_fuse`。
4. **TPU v6e-1 实机故障定位与修复**
   - **修复 4D 广播 SDPA 掩码 `NaN` 全黑图**：将 `text_attn_mask` 与 `joint_attn_mask` 由 `[1, 1, 1, S]` 显式扩展至 `[1, 1, Q, S]`，规避 `torch_tpu` Mosaic/cuDNN 在 `S=30` 误差及 `S >= 512` 产出 `NaN` 的问题。
   - **修复 VAE `GroupNorm` channels-last 崩溃**：注入 `ensure_group_norm_contiguous(vae)` 前向钩子，在 `F.group_norm` 调用 `.view(...)` 前将 `torch_tpu` NHWC 卷积输出转成连续 NCHW。
   - **隔离 `--compile` `e0200` 异常**：定位了 `transformer.compile(backend="tpu", fullgraph=True)` 在 `FlowMatchEulerDiscreteScheduler.step()` 首次 `.nonzero()` D2H 同步处暴露的 `vmem_address_out_of_range` core halt，将 CLI 默认 `--compile` 设为 `False`。

<a id="hardware-verified-benchmarks"></a>

## `TPU v6e-1` 实测基准与数值对齐

测试环境：Colab `TPU v6e-1`（`31.24 GiB` HBM、`torch 2.14.1+cpu`、`torch_tpu 0.2.0`、`libtpu 0.0.50`、`jax 0.11.2`、`transformers 5.22.1`、`deltanet-tpu 0.3.0`，模型 `OpenMOSS-Team/MOSS-Image-Base`，开启 Tier-3 XLA 缓存）。

### Stage 1：`Qwen3.5-9B` GDN 算子对比与耗时拆分

单算子与单层 A/B 实测（`B=1, L=329 -> bucket 384, Hk=16, Hv=32, D=128`，[bench_qwen35_gdn_tpu.py](scripts/bench_qwen35_gdn_tpu.py)）：

| 测试项 | HF `eager` | HF `defer_and_fuse` | `deltanet-tpu` Pallas | 数值精度对齐 |
| --- | --- | --- | --- | --- |
| GDN 递推核热态中位数 | `4.946 ms` | `4.651 ms` | **`1.082 ms`**（快 `4.57x` / `4.30x`） | `cos = 0.999992`, `max_abs = 0.000437` |
| 完整 `Qwen3_5GatedDeltaNet` 单层 | `6.244 ms` | `3.947 ms` | **`2.030 ms`**（`8/8` Pallas, 0 fallback） | `cos = 0.999897`, `max_abs = 0.003906` |

32 层完整 `Qwen3.5-9B` `--mode encode-bench`（正向 `329` tokens + 负向 `24` tokens）：

| 模式 | H2D 权重搬运 | 正向冷编码 | 负向编码 | 换长度热编码 | 同长度热编码 | 峰值 HBM |
| --- | --- | --- | --- | --- | --- | --- |
| `eager` (`DEFER_NEVER`) | `28.653s` | `1.993s` | `0.908s` | `1.168s` | `0.164s` | `15.79 GiB` |
| `defer_and_fuse` | `20.577s` | `11.023s` | `4.659s` | `4.901s` | `0.072s` | `15.73 GiB` |
| **`eager+gdn_pallas`** | `23.860s` | **`0.310s`** | **`0.930s`** | **`0.161s`** | `0.092s` | `15.83 GiB` |
| `defer_and_fuse+gdn_pallas` | `26.311s` | `6.672s` | `1.684s` | `1.859s` | **`0.049s`** | `15.89 GiB` |

- `encode-bench` 定位出 **`h2d_transfer_seconds`（`20.58s～28.65s`）** 才是 Stage 1 冷启动的最大瓶颈（`mmap` 模型加载仅 `0.247s`），而 Pallas 实际编码本身仅 `0.092s～0.310s`。
- 完整 32 层输出 embedding 对齐（`--gdn-pallas` vs. HF）：正向余弦 **`0.999848`**、负向 **`0.999758`**、无 `NaN`/`Inf`。

### Stage 2/3 与端到端出图（`848×1200`, 55 步, CFG `3.0`）

DiT + VAE 基准（`--prompt-embeds` 缓存输入，`uncond_skip_layers=(22,)`，`329` tokens → `512` 静态桶，联合长度 `S = 4487`）：

| 执行模式 | 冷跑 Run 1 总耗时 | 热跑 Run 2 总耗时 | 热跑 Run 2 纯去噪（55 步） | 实测有效算力（对比 `918 TF` BF16 峰值） | 峰值 HBM |
| --- | --- | --- | --- | --- | --- |
| `eager` (`DEFER_NEVER`) | `70.201s` | `49.082s` | `48.612s` (`883.85 ms/step`) | `119.3 TFLOPs`（**`13.0%` MFU**） | `24.473 GiB` |
| **`defer_and_fuse`** | `35.556s` | **`16.724s`** | **`16.434s` (`298.79 ms/step`)** | **`352.8 TFLOPs`（`38.4%` MFU）** | **`16.331 GiB`** |

同种子端到端 CLI 出图（`seed = 136879772936584`, `848×1200`, 55 步, `defer_and_fuse`, DiT Tier-3 热缓存）：

- `defer_and_fuse` + HF 原生 Qwen3.5：`115.169s`（去噪 `24.680s`，峰值 `15.693 GiB`，模型加载 `27.22s`）。
- `defer_and_fuse` + `--gdn-pallas`（提示词缓存未命中）：`88.421s`（去噪 `16.593s` / `301.69 ms/step`，峰值 `15.794 GiB`，`48/48` Pallas 命中）。
- `defer_and_fuse` + `--gdn-pallas --prompt-cache-dir`（**缓存命中**）：**`41.622s`**（加载 `2.20s`，去噪 `16.603s`，峰值 `15.237 GiB`）。
- **同种子像素一致性（`848×1200`）**：TPU HF vs. TPU Pallas 像素余弦 **`0.999442`**（平均像素差 `2.461/255`，`PSNR = 32.31 dB`）；对比 CUDA BF16 参考图余弦为 `0.997435`（TPU HF）/ `0.998019`（TPU Pallas）。

*(注：新增分阶段自动模式 `--text-eager-mode auto` 在最后修改后已通过本地 `ruff check` 与 `tests/test_tpu*.py` 全部 28 项单测；上述 `TPU v6e-1` 实机基准数据均采集自本次合并前的各阶段实测提交。)*

<a id="quick-start"></a>

## 快速开始（Colab `TPU v6e-1`）

### 1. 安装环境

```bash
git clone -b feat/torchtpu-support https://github.com/azure-swing/MOSS-Image.git
cd MOSS-Image
bash scripts/setup-tpu.sh

# 可选：安装 deltanet-tpu（务必加 --no-deps，防止覆盖 TorchTPU 锁定的 libtpu）
pip install --no-deps deltanet-tpu

# 建议启用持久化的 XLA 编译缓存
export TORCH_TPU_TIER2_COMPILATION_CACHE="${HOME}/.cache/torch_tpu/tier2"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${HOME}/.cache/torch_tpu/tier3"
```

也可以直接在 Google Colab 中打开 [`notebooks/moss_image_torchtpu_colab.ipynb`](notebooks/moss_image_torchtpu_colab.ipynb) 运行。

### 2. 探针、出图与性能基准

```bash
# 设备探测与基础矩阵乘冒烟测试
python3 scripts/run_tpu.py --mode probe

# 推荐端到端最快分阶段组合：
# Stage 1 自动走 default + Pallas (--text-eager-mode auto)，Stage 2/3 走 defer_and_fuse
python3 scripts/run_tpu.py \
  --mode generate \
  --eager-mode defer_and_fuse \
  --gdn-pallas \
  --prompt-cache-dir outputs/prompt_cache \
  --height 1024 --width 1024 --steps 50 --seed 42 \
  --prompt "A ceramic teapot on linen" \
  --output outputs/moss_tpu.png

# 独立 Stage-1 GDN 算子与单层 A/B 基准（无需下载 18 GB 权重）
python3 scripts/bench_qwen35_gdn_tpu.py --include-layer --save-report outputs/gdn_ab.json

# Stage-1 编码各阶段耗时拆解（H2D / 冷编译 / 负向 / 变长热态 / 同长热态）
python3 scripts/run_tpu.py \
  --mode encode-bench \
  --bench-modes eager,defer_and_fuse,eager+gdn_pallas,defer_and_fuse+gdn_pallas \
  --runs 3 \
  --save-report outputs/encode_bench.json
```

完整 CLI 参数、`--mode encode-only` 编码与出图解耦、Tier-3 缓存与排障日志见 **[docs/tpu.md](docs/tpu.md)**。

<a id="runtime-architecture"></a>

## 分阶段运行时架构

```text
Prompt 文本
   │
   ├─► 命中硬盘缓存 (--prompt-cache-dir) / 内存 LRU ─────────────────────┐
   │                                                                      ▼
   └─► Stage 1 [DEFER_NEVER + deltanet-tpu Pallas] (Qwen3.5-9B BF16) ──► CPU embeds [1, L, 4096]
                                                                          │  (卸出 Stage 1)
   Stage 2 [defer_and_fuse] (MOSSImageTransformer BF16, 静态桶 512/1024/2048/4096)
     • 预计算 FP32 时间步正弦缓冲 + [1, 1, Q, S] 静态展开注意力掩码
                                                                          │
                                                                          ▼
   Stage 3 [defer_and_fuse] (AutoencoderKLFlux2 FP32 + GroupNorm 连续化钩子) ──► 输出 PNG
```

- **为什么 `TPU v6e-1`（`31.24 GiB`）需要 `--offload model`**：`Qwen3.5-9B`（约 `18 GB` BF16）加上 Stage 2/3（`MOSSImageTransformer` 约 9.5 GiB BF16 + FP32 VAE + XLA 激活峰值 `16.33 GiB`）会超过 `31.24 GiB` HBM（因此 `offload="auto"` 仅在显存 `>= 40 GiB` 时选择常驻 `"none"`）。配合 `--prompt-cache-dir`，命中缓存后完全跳过 Stage 1 加载与 H2D 搬运，Stage 2/3 即可一直常驻 TPU 连续出图。
- **本地代码规范与单元测试**：
  ```bash
  ruff check tpu_runtime.py tpu_qwen35.py scripts/run_tpu.py scripts/bench_qwen35_gdn_tpu.py tests/test_tpu_runtime.py tests/test_tpu_qwen35.py
  python3 -m unittest discover -s tests -p "test_tpu*.py" -v
  ```

<a id="repository-layout"></a>

## 新增与改动文件索引

| 文件 | 用途 |
| --- | --- |
| [`tpu_runtime.py`](tpu_runtime.py) | 三阶段 TPU 流水线（`StagedTPUImagePipeline`）、分阶段 Eager 上下文、静态 prompt 分桶、`[1, 1, Q, S]` SDPA 掩码扩展、FP32 timestep 缓冲与 `GroupNorm` 钩子 |
| [`tpu_qwen35.py`](tpu_qwen35.py) | `torch.tpu.pallas.jax_op` ↔ `deltanet_tpu.chunk_gdn` 桥接、128 对齐因果右填充与硬盘缓存 |
| [`scripts/run_tpu.py`](scripts/run_tpu.py) | TPU CLI 入口（`probe` / `encode-only` / `encode-bench` / `generate` / `benchmark`）与 `--text-eager-mode auto` |
| [`scripts/bench_qwen35_gdn_tpu.py`](scripts/bench_qwen35_gdn_tpu.py) | 独立 Qwen3.5 GDN 递推核与完整 GDN 层精度/耗时基准脚本 |
| [`scripts/setup-tpu.sh`](scripts/setup-tpu.sh) | Colab / TPU VM 一键安装脚本（TorchTPU `0.2.0` + PyTorch CPU） |
| [`notebooks/moss_image_torchtpu_colab.ipynb`](notebooks/moss_image_torchtpu_colab.ipynb) | Google Colab TPU 端到端交互式 Notebook |
| [`docs/tpu.md`](docs/tpu.md) | TPU v6e-1 实机调试日志、错误根因定位与性能对比说明 |
| [`gradio_app.py`](gradio_app.py) / [`inference_runtime.py`](inference_runtime.py) | CUDA / Apple Silicon MPS / TorchTPU 设备路由与产物元数据记录 |
| [`tests/test_tpu_runtime.py`](tests/test_tpu_runtime.py) / [`tests/test_tpu_qwen35.py`](tests/test_tpu_qwen35.py) | 覆盖分桶、补丁、Pallas 桥、硬盘缓存与分阶段 Eager CLI 的 28 项单元测试 |

<a id="next-roadmap"></a>

## 后续优化方向

进一步的 `TPU v6e-1` 算力利用率（MFU）拆解与后续优化计划已发布至 **[GitHub Discussions #5](https://github.com/azure-swing/MOSS-Image/discussions/5)**（`Ideas` 板块），核心包括：
1. Stage 2 用 Pallas `SplashAttention`（块级稀疏掩码）替换 `[1, 1, 4487, 4487]` 4D bool mask `SDPA`（目前单层占 `2.38 ms`），突破 `38.4%` MFU；
2. 消除 `FlowMatchEulerDiscreteScheduler.step()` 中每步触发 1 次的 `.nonzero()` D2H 同步并二分修通 `transformer.compile(backend="tpu", fullgraph=True)`；
3. 将未跳层的 CFG 合并为 `batch=2` 批执行并把 `--buckets` 下探出 `128, 256` 档位；
4. 用权重连续打包 / pinned H2D 或 `Qwen3.5-9B` INT8 / W4 量化尝试在 `31.24 GiB` 上实现 Stage 1+2+3 全常驻，彻底消掉 `~24s` H2D 搬运。

<a id="upstream-and-citation"></a>

## 上游项目、指南与引用

本仓库基于 **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** 构建。基础模型权重、CUDA / MPS 用法、Gradio 界面、ComfyUI 原生/插件工作流、JSON Layout 排版控制与提示词增强 Skill 详见原项目文档：

- [中文用户指南](USERGUIDE_ZH.md) · [English User Guide](USERGUIDE.md)
- [Apple Silicon / MPS 说明](docs/macos_zh.md) · [ComfyUI 集成文档](comfyui/README_ZH.md) · [提示词增强 Skill](skills/moss-image-prompt-enhancer/SKILL.md)
- 模型下载：[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) · [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) · [技术报告 (arXiv:2610.23374)](https://arxiv.org/abs/2610.23374)

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

推理代码采用 [Apache License 2.0](LICENSE)。感谢 OpenMOSS、Diffusers、Qwen、FLUX、TorchTPU 与 `deltanet-tpu` 团队。
