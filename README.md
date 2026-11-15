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
    <strong>简体中文</strong> &nbsp; | &nbsp; <a href="README_EN.md">English</a> &nbsp; | &nbsp; <a href="README_JA.md">日本語</a>
  </p>
</div>

# MOSS-Image Google Cloud TPU 适配版 (TorchTPU)

本仓库（`azure-swing/MOSS-Image`，`feat/torchtpu-support` 分支）将 **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** 适配到了 **Google Cloud TPU（`TPU v6e-1`，`31.24 GiB` HBM）** 原生 PyTorch 运行，完整跑通 BF16 端到端生图与 FP32 VAE 解码，生成画质与原版 CUDA BF16 保持高度一致。

> [!IMPORTANT]
> **显存使用建议（`< 40 GB`）**：如果设备显存 / HBM **小于 `40 GB`**（例如 `TPU v6e-1` 为 `31.24 GiB`），运行 **BF16** 精度模型时**建议使用 `offload`**（`--offload model` 或默认 `--offload auto`，显存低于 `40 GiB` 时会自动开启）。9B 文本编码器、5B 扩散模型与 VAE 同时常驻会超出 `31.24 GiB` 容量。

## 目录

- [生成效果对比（TPU vs. 原版 CUDA）](#image-quality-comparison)
- [图像生成速度、理论算子效率与 Offload 耗时](#speed-and-efficiency)
- [快速开始（Colab `TPU v6e-1`）](#quick-start)
- [技术细节与底层实现](#technical-details)
- [新增/改动文件说明](#repository-layout)
- [后续优化路线图](#next-roadmap)
- [上游项目与引用](#upstream-and-citation)

<a id="image-quality-comparison"></a>

## 生成效果对比（TPU vs. 原版 CUDA）

同提示词、同随机种子（`848×1200`，`55` 步，CFG `3.0`，`seed = 136879772936584`）在 `TPU v6e-1` 与原版 CUDA BF16 下的直接出图对比：

| 原版 CUDA BF16 参考图 | TPU `v6e-1` BF16 生成图（`defer_and_fuse` + Pallas） |
| :---: | :---: |
| <img src="assets/comparison/cuda_bf16_ref.png" alt="原版 CUDA BF16 参考图" width="380"> | <img src="assets/comparison/tpu_v6e_pallas.png" alt="TPU v6e-1 BF16 生成图" width="380"> |

- **TPU（`defer_and_fuse` + Pallas）vs. 原版 CUDA BF16**：像素余弦相似度 **`0.998019`**，平均像素差 **`6.151 / 255`**，**`PSNR = 26.80 dB`**（未开 Pallas 时相对 CUDA BF16 为 `cos = 0.997435`）。
- **TPU Pallas vs. TPU 默认文本编码器**：像素余弦相似度 **`0.999442`**，平均像素差 **`2.461 / 255`**，**`PSNR = 32.31 dB`**。
- **`drop_tpu` 卸载 vs. `d2h` 卸载**：输出像素完全一致（`cosine_similarity = 1.0`，`mean_abs_diff = 0.0`，`PSNR = Infinity`）。

<a id="speed-and-efficiency"></a>

## 图像生成速度、理论算子效率与 Offload 耗时

以下数据均实测于 Google Colab **`TPU v6e-1`**（`31.24 GiB` HBM，`172.95 GiB` 主机内存，`848×1200`，`55` 步，CFG `3.0`，模型 `OpenMOSS-Team/MOSS-Image-Base`）。

### 1. 各场景图像生成与 VAE 解码速度

| 使用场景 | 总耗时 | 耗时拆分（去噪 / VAE / 其余开销） | 峰值显存 (HBM) |
| --- | ---: | --- | ---: |
| **同提示词热态续生（`defer_and_fuse`）** | **`16.72s`** | `16.43s` 去噪（`298.8 ms/step`）+ **`~0.29s` VAE 解码与后处理** | `16.33 GiB` |
| **10 连跑热态均值（第 2–10 张）** | **`16.73s`** | `16.43s` 去噪（`298.7 ms/step`），10 张含首张冷跑（`95.56s`）共 `246.17s` | `16.25 GiB` |
| **无融合热态基准（`eager`）** | `49.08s` | `48.61s` 去噪（`883.9 ms/step`）+ `~0.47s` VAE 解码与后处理 | `24.47 GiB` |
| **冷态仅扩散 + VAE（`--prompt-embeds` 输入）** | `35.56s` | `16.61s` 去噪 + 首次图编译与 VAE 解码（`eager` 下为 `70.20s`） | `14.96 GiB` |
| **换提示词生图（`drop_tpu` + 异步 H2D Offload）** | **`37.71s`** | `20.75s` 阶段切换 + `16.54s` 去噪（`300.8 ms/step`）+ `~0.42s` 编码与 VAE | `20.67 GiB` |
| **新进程命中磁盘提示词缓存** | `41.62s` | `2.20s` 加载 + `16.60s` 去噪 + 权重搬运与 VAE | `15.24 GiB` |
| **换提示词生图（`d2h` + 异步 H2D Offload）** | `84.41s` | `67.08s` 阶段切换 + `16.55s` 去噪（`300.9 ms/step`）+ `~0.78s` 编码与 VAE | `20.67 GiB` |
| **首次冷跑 CLI（缓存未命中，Plan A 优化前）** | `88.42s` | `16.59s` 去噪（`301.7 ms/step`）+ 冷文本编码与同步卸载（无 Pallas 为 `115.17s`） | `15.79 GiB` |
| **严格 Eager 冷跑基线（`default`）** | `124.19s` | `52.98s` 去噪（`963.2 ms/step`）+ 冷文本编码与同步卸载 | `24.50 GiB` |

### 2. 理论算子效率（对比 `TPU v6e-1` BF16 峰值 `918 TFLOPs`）

| 模块 / 执行模式 | 实测有效算力 / 耗时 | 算力利用率 (MFU) / 提速倍率 |
| --- | ---: | --- |
| **扩散去噪（`defer_and_fuse`，55 步）** | **`352.8 TFLOPs`**（`298.8 ms/step`） | **`38.4%` MFU** |
| **扩散去噪（`eager` / `DEFER_NEVER`，55 步）** | `119.3 TFLOPs`（`883.9 ms/step`） | `13.0%` MFU（`defer_and_fuse` 快 **2.96x**） |
| **文本编码 GDN 核（`deltanet-tpu` Pallas）** | **`1.082 ms`** 递推核 / **`2.030 ms`** 完整层 | 递推核快 **`4.57x`** / 整层快 **`3.08x`**（对比 HF `eager`） |
| **文本编码前向（`eager+gdn_pallas`）** | **`0.310s`** 冷态 / **`0.161s`** 换词热态 / `0.092s` 重复 | HF `eager` 原版为 `1.993s` 冷态 / `1.168s` 换词热态 |

### 3. `< 40 GB` 显存下的 Offload 切换速度对比

由于 BF16 精度的 `Qwen3.5-9B`（`15.63 GiB`）与扩散模型 + VAE（`9.70 GiB`）加上运行时缓存会超过 `31.24 GiB`，开启 `--offload model` 后，更换提示词时需要在 CPU 内存和 TPU HBM 之间切换模块。通过 **Plan A（`drop_tpu` 零回拷卸载 + 异步 `non_blocking` H2D）**，主机内存常驻只读权重快照，离场模块直接释放而不做 D2H 拷贝（D2H = `0.0s`）：

| Offload 模式 | Run 1 首次切换 | Run 2 换词切换（`diffusion -> text -> diffusion`） | Run 2 端到端总耗时 | 切换提速 |
| --- | ---: | ---: | ---: | ---: |
| **同步 H2D + `d2h` 基线** *(4 步)* | `70.00s` | `100.73s`（`50.25s` D2H + `50.48s` H2D） | `103.29s` *(4 步)* | `1.00x` |
| **仅开启 `drop_tpu`** *(4 步)* | — | `48.96s`（`0.0s` D2H + `48.03s` H2D） | `50.62s` *(4 步)* | `2.06x` |
| **`d2h` + 异步 H2D** *(55 步)* | `49.00s`（`29.67s` D2H + `19.33s` H2D） | `67.08s`（`49.44s` D2H + `17.63s` H2D） | `84.41s` | `1.50x` |
| **`drop_tpu` + 异步 H2D** *(55 步)* | **`18.68s`**（`0.0s` D2H + `18.19s` H2D） | **`20.75s`**（`0.0s` D2H + `0.99s` 释放/GC + `19.75s` H2D） | **`37.71s`** | **较异步 `d2h` 快 `3.23x`（较同步基线快 `4.86x`）** |

<a id="quick-start"></a>

## 快速开始（Colab `TPU v6e-1`）

### 1. 安装环境

```bash
git clone -b feat/torchtpu-support https://github.com/azure-swing/MOSS-Image.git
cd MOSS-Image
bash scripts/setup-tpu.sh

# 可选：安装 deltanet-tpu（务必加 --no-deps，避免覆盖 TorchTPU 锁定的 libtpu）
pip install --no-deps deltanet-tpu

# 建议开启持久化 XLA 编译缓存
export TORCH_TPU_TIER2_COMPILATION_CACHE="${HOME}/.cache/torch_tpu/tier2"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${HOME}/.cache/torch_tpu/tier3"
```

也可以直接在 Google Colab 中打开 [`notebooks/moss_image_torchtpu_colab.ipynb`](notebooks/moss_image_torchtpu_colab.ipynb) 一键运行。

### 2. 硬件检测、出图与性能压测

```bash
# 检测 TorchTPU 设备与基础矩阵算子
python3 scripts/run_tpu.py --mode probe

# 推荐快速出图命令（显存 < 40 GiB 时自动开启 offload）
python3 scripts/run_tpu.py \
  --mode generate \
  --eager-mode defer_and_fuse \
  --gdn-pallas \
  --prompt-cache-dir outputs/prompt_cache \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --prompt "A ceramic teapot on linen" \
  --output outputs/moss_tpu.png

# 对比 d2h 与 drop_tpu 的交互式换词切换耗时
python3 scripts/run_tpu.py \
  --mode stage-switch-bench \
  --switch-modes d2h,drop_tpu \
  --eager-mode defer_and_fuse \
  --gdn-pallas \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --save-report outputs/stage_switch_55.json
```

完整 CLI 参数、`--mode encode-only` 离线编码、`--mode encode-bench` 及排障日志请参阅 **[docs/tpu.md](docs/tpu.md)**。

---

<a id="technical-details"></a>

## 技术细节与底层实现

```text
Prompt 文本
   │
   ├─► 命中磁盘缓存 (--prompt-cache-dir) / 内存 8 条 LRU ─────────────────┐
   │                                                                      ▼
   └─► Stage 1 [DEFER_NEVER + deltanet-tpu Pallas] (Qwen3.5-9B BF16) ──► CPU 隐状态 [1, L, 4096]
                                                                          │
   Stage 2 [defer_and_fuse] (MOSSImageTransformer BF16, 静态桶 512/1024/2048/4096)
     • 预计算 FP32 timestep 正余弦编码 + 显式扩展 [1, 1, Q, S] SDPA 掩码
                                                                          │
                                                                          ▼
   Stage 3 [defer_and_fuse] (AutoencoderKLFlux2 FP32 + NHWC→NCHW GroupNorm 钩子) ──► PNG
```

1. **三阶段分时调度（`tpu_runtime.py`）**：
   - **Stage 1（`Qwen3.5-9B`, `BF16`, `15.63 GiB`）**：按真实长度非填充前向以保持因果 Gated DeltaNet (GDN) 语义一致，输出 `[1, L_true, 4096]` 隐状态卸载到 CPU 内存（8 条 LRU `_prompt_cache` + 可选 `--prompt-cache-dir`）。
   - **Stage 2 & 3（`MOSSImageTransformer` ~5.08B `BF16` / `9.4 GiB` + `AutoencoderKLFlux2` `FP32` / `~0.3 GiB`）**：`--offload model`（`< 40 GiB` 时由 `--offload auto` 自动选择）在换提示词时交替调度 `text_encoder` 与 `transformer`；`vae` 在首次进入扩散阶段后保持常驻 TPU（仅显式切回 `cpu` 时释放）。
2. **`drop_tpu` 零回拷与异步 `non_blocking` H2D（`tpu_runtime.py`）**：
   - **`drop_tpu`（`--stage-switch-mode auto` / `drop_tpu`）**：主机内存大于等于 `36 GiB` 时默认启用。离场模块通过 `_release_module_to_meta` 直接替换为零字节 `meta` 张量（`0.0s` D2H），再经 `_restore_module_from_cpu_snapshot` 恢复。在 `172.95 GiB` 主机上，Run 2 结束时进程 RSS 为 `30.91 GiB`，仍余 `138.24 GiB` 可用内存。
   - **异步 `non_blocking=True` H2D**：将参数与 buffer 的 `.to(tpu, non_blocking=True)` 合并到单次 `torch.tpu.synchronize()` 下流水线传输，400 张量（`12.50 GiB`）合成传输吞吐由 `0.53 GiB/s`（`23.78s`）提升至 `1.96 GiB/s`（`6.38s`；预固定内存可达 `2.05 GiB/s` / `6.10s`）。
   - **`~5.0 GiB` HBM 常驻平台定位**：首次完整推理后，`d2h`（`5.00–5.03 GiB`）与 `drop_tpu`（`5.01–5.03 GiB`）均在 `torch.tpu.memory_allocated()` 维持约 `5.0 GiB` 底盘，此时 Python `gc.get_objects()` 中存活 TPU 张量为 `0`，确认为 TorchTPU/PJRT C++ 运行时与编译缓存的一次性常驻占用。
3. **Stage 1 Pallas GDN 桥接（`tpu_qwen35.py`）与分阶段 Eager 模式（`--text-eager-mode auto`）**：
   - 将全部 24 层 `GatedDeltaNet`（`Hk=16, Hv=32, D=128`, `use_cache=False`）通过 `torch.tpu.pallas.jax_op` 接入 `deltanet_tpu.chunk_gdn` 与 `128` 对齐静态桶（`128, 256, 384, 512, 768, 1024`），实测 0 回退（`encode-bench` `96/96`、CFG 生图 `48/48`）。
   - Stage 1 自动选用 `default`（`DEFER_NEVER`）+ Pallas，Stage 2/3 自动选用 `defer_and_fuse`。
4. **`TPU v6e-1` 硬件问题定位与修复（`tpu_runtime.py`、`scripts/run_tpu.py`）**：
   - **修复 4D 广播 SDPA 掩码 `NaN` 全黑图**：将 `text_attn_mask` 与 `joint_attn_mask` 从 `[1, 1, 1, S]` 扩展至 `[1, 1, Q, S]`，规避 `torch_tpu` Mosaic/cuDNN 在 `S=30` 误差及 `S >= 512` 输出 `NaN` 的问题。
   - **修复 VAE `GroupNorm` 内存布局崩溃**：通过 `ensure_group_norm_contiguous(vae)` 前向钩子将 NHWC 卷积输出转成连续 NCHW 后再执行 `F.group_norm`。
   - **隔离 `--compile` `e0200` 核崩溃**：`transformer.compile(backend="tpu", fullgraph=True)` 会在调度器首次 `.nonzero()` D2H 同步处触发 `vmem_address_out_of_range` 硬件异常，默认保持 `--compile=False`。

### Stage 1 `encode-bench` 微基准拆分（`329` 正向 + `24` 负向 tokens，异步 H2D 优化前）

| 模式 | H2D 权重搬运 | 正向冷编码 | 负向编码 | 换长度热编码 | 同长度热编码 | 峰值 HBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `eager` (`DEFER_NEVER`) | `28.653s` | `1.993s` | `0.908s` | `1.168s` | `0.164s` | `15.79 GiB` |
| `defer_and_fuse` | `20.577s` | `11.023s` | `4.659s` | `4.901s` | `0.072s` | `15.73 GiB` |
| **`eager+gdn_pallas`** | `23.860s` | **`0.310s`** | **`0.930s`** | **`0.161s`** | `0.092s` | `15.83 GiB` |
| `defer_and_fuse+gdn_pallas` | `26.311s` | `6.672s` | `1.684s` | `1.859s` | **`0.049s`** | `15.89 GiB` |

*（32 层 `--gdn-pallas` 输出 embedding 精度对齐：正向 `cos = 0.999848`，负向 `cos = 0.999758`，无 `NaN`/`Inf`。）*

### 本地代码检查与单元测试

```bash
ruff check tpu_runtime.py tpu_qwen35.py scripts/run_tpu.py scripts/bench_qwen35_gdn_tpu.py tests/test_tpu_runtime.py tests/test_tpu_qwen35.py
python3 -m unittest discover -s tests -p "test_tpu*.py" -v
```

<a id="repository-layout"></a>

## 新增/改动文件说明

| 文件 | 职责 |
| --- | --- |
| [`tpu_runtime.py`](tpu_runtime.py) | 三阶段 TPU 流水线（`StagedTPUImagePipeline`）、`drop_tpu` + 异步 H2D + 常驻 VAE、分阶段 eager 上下文、静态桶、`[1, 1, Q, S]` SDPA 补丁、FP32 正余弦与 `GroupNorm` 钩子 |
| [`tpu_qwen35.py`](tpu_qwen35.py) | `torch.tpu.pallas.jax_op` 桥接 `deltanet_tpu.chunk_gdn`、`128` 对齐因果填充与持久化 CPU 提示词缓存 |
| [`scripts/run_tpu.py`](scripts/run_tpu.py) | TPU CLI（`probe`、`encode-only`、`encode-bench`、`generate`、`benchmark`、`stage-switch-bench`） |
| [`scripts/bench_qwen35_gdn_tpu.py`](scripts/bench_qwen35_gdn_tpu.py) | 独立 GDN 递推核与 `Qwen3_5GatedDeltaNet` 单层 A/B 测速与对齐脚本 |
| [`scripts/setup-tpu.sh`](scripts/setup-tpu.sh) | Colab / TPU VM 一键安装脚本（TorchTPU `0.2.0` + PyTorch CPU） |
| [`notebooks/moss_image_torchtpu_colab.ipynb`](notebooks/moss_image_torchtpu_colab.ipynb) | Google Colab 端到端 TPU 示例 Notebook |
| [`docs/tpu.md`](docs/tpu.md) | `TPU v6e-1` 详细调试日志、报错分析与 Plan A 阶段切换实测结果 |
| [`gradio_app.py`](gradio_app.py) / [`inference_runtime.py`](inference_runtime.py) | CUDA、Apple Silicon MPS 与 TorchTPU 多后端路由与设备信息记录 |
| [`tests/test_tpu_runtime.py`](tests/test_tpu_runtime.py) / [`tests/test_tpu_qwen35.py`](tests/test_tpu_qwen35.py) | 32 项本地 CPU 单元测试与数值对齐校验 |

<a id="next-roadmap"></a>

## 后续优化路线图

性能优化讨论记录在 **[GitHub Discussions #5](https://github.com/azure-swing/MOSS-Image/discussions/5)**：
1. 使用 Pallas `SplashAttention` 替代 `[1, 1, 4487, 4487]` 4D 掩码 `SDPA`（`~2.38 ms/layer`），将 DiT MFU 推高到 `38.4%` 以上。
2. 消除 `FlowMatchEulerDiscreteScheduler.step()` 每步的 `.nonzero()` D2H 同步并定位 `e0200` VMEM 异常，解锁 `transformer.compile(backend="tpu", fullgraph=True)`。
3. 将未跳层的 CFG 合并为 `batch=2` 批执行并新增 `128, 256` 提示词小桶。
4. 探索 `Qwen3.5-9B` INT8/W4 量化，使全部阶段可同时常驻 `31.24 GiB`（`--offload none`），彻底消除换词 H2D 耗时。

<a id="upstream-and-citation"></a>

## 上游项目与引用

本项目基于 **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** 构建。基础模型介绍、CUDA/MPS 用法、Gradio、ComfyUI 集成及结构化提示词指南请参见上游文档：

- [用户指南 (中文)](USERGUIDE_ZH.md) · [User Guide (EN)](USERGUIDE.md)
- [Apple Silicon / MPS](docs/macos.md) · [ComfyUI 集成](comfyui/README.md) · [提示词增强 Skill](skills/moss-image-prompt-enhancer/SKILL.md)
- 模型权重：[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) · [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) · [技术报告 (arXiv:2610.23374)](https://arxiv.org/abs/2610.23374)

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

推理代码采用 [Apache License 2.0](LICENSE) 协议开源。感谢 OpenMOSS、Diffusers、Qwen、FLUX、TorchTPU 与 `deltanet-tpu` 团队。
