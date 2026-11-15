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
    <a href="README.md">简体中文</a> &nbsp; | &nbsp; <a href="README_EN.md">English</a> &nbsp; | &nbsp; <strong>日本語</strong>
  </p>
</div>

# MOSS-Image Google Cloud TPU 対応版 (TorchTPU)

本リポジトリ（`azure-swing/MOSS-Image`、`feat/torchtpu-support` ブランチ）は、**[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** を **Google Cloud TPU（`TPU v6e-1`、`31.24 GiB` HBM）** 上のネイティブ PyTorch で動作させたフォークです。BF16 でのエンドツーエンド画像生成と FP32 VAE デコードに対応し、元の CUDA BF16 参照画像と極めて高い一致性を保ちます。

> [!IMPORTANT]
> **メモリに関する推奨事項（`< 40 GB` HBM / VRAM）**：デバイスメモリが **`40 GB` 未満**（例：`TPU v6e-1` の `31.24 GiB`）の環境で **BF16** 精度モデルを実行する場合は、**`offload` の使用を推奨します**（`--offload model`、または `40 GiB` 未満で自動的に有効になる `--offload auto`）。9B テキストエンコーダと 5B 拡散モデル + VAE を同時に常駐させると `31.24 GiB` を超えます。

## 目次

- [生成画像の品質比較（TPU vs. CUDA 原版）](#image-quality-comparison)
- [画像生成速度・理論演算効率・Offload 性能](#speed-and-efficiency)
- [クイックスタート（Colab `TPU v6e-1`）](#quick-start)
- [技術詳細・アーキテクチャ](#technical-details)
- [リポジトリ構成（TPU 追加・変更点）](#repository-layout)
- [今後の最適化ロードマップ](#next-roadmap)
- [上流プロジェクト・引用](#upstream-and-citation)

<a id="image-quality-comparison"></a>

## 生成画像の品質比較（TPU vs. CUDA 原版）

同一プロンプト・同一シード（`848×1200`、`55` ステップ、`guidance_scale = 3.0`、`seed = 136879772936584`）における元の CUDA BF16 参照画像と `TPU v6e-1` 生成画像の比較です：

| 元の CUDA BF16 参照画像 | TPU `v6e-1` BF16 生成画像（`defer_and_fuse` + Pallas） |
| :---: | :---: |
| <img src="assets/comparison/cuda_bf16_ref.png" alt="元の CUDA BF16 参照画像" width="380"> | <img src="assets/comparison/tpu_v6e_pallas.png" alt="TPU v6e-1 BF16 生成画像" width="380"> |

- **TPU（`defer_and_fuse` + Pallas）vs. CUDA BF16 参照画像**：コサイン類似度 **`0.998019`**、平均画素誤差 **`6.151 / 255`**、**`PSNR = 26.80 dB`**（Pallas 未使用の TPU vs. CUDA BF16 は `cos = 0.997435`）。
- **TPU Pallas vs. TPU デフォルトエンコーダ**：コサイン類似度 **`0.999442`**、平均画素誤差 **`2.461 / 255`**、**`PSNR = 32.31 dB`**。
- **`drop_tpu` Offload vs. `d2h` Offload**：完全一致（`cosine_similarity = 1.0`、`mean_abs_diff = 0.0`、`PSNR = Infinity`）。

<a id="speed-and-efficiency"></a>

## 画像生成速度・理論演算効率・Offload 性能

以下の数値は Google Colab **`TPU v6e-1`**（`31.24 GiB` HBM、`172.95 GiB` ホスト RAM、`848×1200`、`55` ステップ、CFG `3.0`、`OpenMOSS-Team/MOSS-Image-Base`）での実測値です。

### 1. 各シナリオの画像生成・VAE 速度

| シナリオ | 合計時間 | 内訳（デノイズ / VAE / その他） | ピーク HBM |
| --- | ---: | --- | ---: |
| **同一プロンプト連続生成（ウォーム / `defer_and_fuse`）** | **`16.72s`** | `16.43s` デノイズ（`298.8 ms/step`）+ **`~0.29s` VAE デコード・後処理** | `16.33 GiB` |
| **10 シード連続生成の平均（2〜10 枚目）** | **`16.73s`** | `16.43s` デノイズ（`298.7 ms/step`）、1 枚目のコールド実行（`95.56s`）込みで全 10 枚 `246.17s` | `16.25 GiB` |
| **融合なしウォーム基準（`eager`）** | `49.08s` | `48.61s` デノイズ（`883.9 ms/step`）+ `~0.47s` VAE デコード・後処理 | `24.47 GiB` |
| **コールド拡散 + VAE（`--prompt-embeds` 入力）** | `35.56s` | `16.61s` デノイズ + 初回 XLA コンパイル・VAE（`eager` では `70.20s`） | `14.96 GiB` |
| **プロンプト変更（`drop_tpu` + 非同期 H2D Offload）** | **`37.71s`** | `20.75s` ステージ切替 + `16.54s` デノイズ（`300.8 ms/step`）+ `~0.42s` エンコード・VAE | `20.67 GiB` |
| **新規プロセス・ディスクキャッシュヒット時** | `41.62s` | `2.20s` ロード + `16.60s` デノイズ + 転送・VAE | `15.24 GiB` |
| **プロンプト変更（`d2h` + 非同期 H2D Offload）** | `84.41s` | `67.08s` ステージ切替 + `16.55s` デノイズ（`300.9 ms/step`）+ `~0.78s` エンコード・VAE | `20.67 GiB` |
| **初回コールド CLI（キャッシュミス、Plan A 以前）** | `88.42s` | `16.59s` デノイズ（`301.7 ms/step`）+ コールドテキストエンコード・同期 Offload（Pallas なし `115.17s`） | `15.79 GiB` |
| **Strict Eager コールド基準（`default`）** | `124.19s` | `52.98s` デノイズ（`963.2 ms/step`）+ コールドテキストエンコード・同期 Offload | `24.50 GiB` |

### 2. 理論演算効率（`TPU v6e-1` BF16 ピーク性能 `918 TFLOPs` 対比）

| コンポーネント / モード | 実効スループット / レイテンシ | ピーク効率 (MFU) / 高速化率 |
| --- | ---: | --- |
| **拡散モデル（`defer_and_fuse`、55 ステップ）** | **`352.8 TFLOPs`**（`298.8 ms/step`） | **`38.4%` MFU** |
| **拡散モデル（`eager` / `DEFER_NEVER`、55 ステップ）** | `119.3 TFLOPs`（`883.9 ms/step`） | `13.0%` MFU（`defer_and_fuse` が **2.96x** 高速） |
| **テキストエンコーダ GDN カーネル（`deltanet-tpu` Pallas）** | **`1.082 ms`** 再帰カーネル / **`2.030 ms`** 単一層 | HF `eager` 比で再帰 **`4.57x`** / 単一層 **`3.08x`** 高速 |
| **テキストエンコーダ推論（`eager+gdn_pallas`）** | **`0.310s`** コールド / **`0.161s`** 長さ変更ウォーム / `0.092s` 同一反復 | HF `eager` は `1.993s` コールド / `1.168s` 長さ変更ウォーム |

### 3. `< 40 GB` 環境における Offload 切替速度比較

BF16 の `Qwen3.5-9B`（`15.63 GiB`）と拡散モデル + VAE（`9.70 GiB`）およびランタイムバッファは合計 `31.24 GiB` を超えるため、`--offload model` ではプロンプト変更時に CPU RAM と TPU HBM 間でステージを切り替えます。**Plan A（`drop_tpu` + 非同期 `non_blocking` H2D）** ではホスト RAM に読み取り専用の重みスナップショットを保持し、D2H コピーを行わず直接 TPU 上の重みを破棄します（D2H = `0.0s`）：

| Offload モード | Run 1 コールド切替 | Run 2 プロンプト切替（`diffusion -> text -> diffusion`） | Run 2 生成合計 | 切替高速化 |
| --- | ---: | ---: | ---: | ---: |
| **同期 H2D + `d2h` 基準** *(4 ステップ)* | `70.00s` | `100.73s`（`50.25s` D2H + `50.48s` H2D） | `103.29s` *(4 ステップ)* | `1.00x` |
| **`drop_tpu` + 同期 H2D** *(4 ステップ)* | — | `48.96s`（`0.0s` D2H + `48.03s` H2D） | `50.62s` *(4 ステップ)* | `2.06x` |
| **`d2h` + 非同期 H2D** *(55 ステップ)* | `49.00s`（`29.67s` D2H + `19.33s` H2D） | `67.08s`（`49.44s` D2H + `17.63s` H2D） | `84.41s` | `1.50x` |
| **`drop_tpu` + 非同期 H2D** *(55 ステップ)* | **`18.68s`**（`0.0s` D2H + `18.19s` H2D） | **`20.75s`**（`0.0s` D2H + `0.99s` 解放/GC + `19.75s` H2D） | **`37.71s`** | **非同期 `d2h` 比 `3.23x`（同期基準比 `4.86x`）** |

<a id="quick-start"></a>

## クイックスタート（Colab `TPU v6e-1`）

### 1. 環境構築

```bash
git clone -b feat/torchtpu-support https://github.com/azure-swing/MOSS-Image.git
cd MOSS-Image
bash scripts/setup-tpu.sh

# 任意：TorchTPU 固定の libtpu を上書きしないよう --no-deps で導入
pip install --no-deps deltanet-tpu

# XLA コンパイルキャッシュの永続化（推奨）
export TORCH_TPU_TIER2_COMPILATION_CACHE="${HOME}/.cache/torch_tpu/tier2"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${HOME}/.cache/torch_tpu/tier3"
```

Google Colab で [`notebooks/moss_image_torchtpu_colab.ipynb`](notebooks/moss_image_torchtpu_colab.ipynb) を直接開いて実行することもできます。

### 2. 動作確認・画像生成・ベンチマーク

```bash
# TorchTPU デバイスおよび基本演算の疎通確認
python3 scripts/run_tpu.py --mode probe

# 推奨の高速生成コマンド（HBM < 40 GiB の環境では自動的に offload が有効）
python3 scripts/run_tpu.py \
  --mode generate \
  --eager-mode defer_and_fuse \
  --gdn-pallas \
  --prompt-cache-dir outputs/prompt_cache \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --prompt "A ceramic teapot on linen" \
  --output outputs/moss_tpu.png

# d2h と drop_tpu のステージ切替ベンチマーク
python3 scripts/run_tpu.py \
  --mode stage-switch-bench \
  --switch-modes d2h,drop_tpu \
  --eager-mode defer_and_fuse \
  --gdn-pallas \
  --height 848 --width 1200 --steps 55 --seed 136879772936584 \
  --save-report outputs/stage_switch_55.json
```

全 CLI オプション、`--mode encode-only`、`--mode encode-bench`、詳細な障害解析ログは **[docs/tpu.md](docs/tpu.md)** を参照してください。

---

<a id="technical-details"></a>

## 技術詳細・アーキテクチャ

```text
Prompt テキスト
   │
   ├─► ディスクキャッシュ (--prompt-cache-dir) / CPU 8件 LRU ヒット ───────┐
   │                                                                       ▼
   └─► Stage 1 [DEFER_NEVER + deltanet-tpu Pallas] (Qwen3.5-9B BF16) ──► CPU 隠れ状態 [1, L, 4096]
                                                                           │
   Stage 2 [defer_and_fuse] (MOSSImageTransformer BF16, 静的バケット 512/1024/2048/4096)
     • 事前計算 FP32 timestep 正弦波 + 拡張済み [1, 1, Q, S] SDPA マスク
                                                                           │
                                                                           ▼
   Stage 3 [defer_and_fuse] (AutoencoderKLFlux2 FP32 + NHWC→NCHW GroupNorm フック) ──► PNG
```

1. **3 ステージ分割ランタイム（`tpu_runtime.py`）**：
   - **Stage 1（`Qwen3.5-9B`, `BF16`, `15.63 GiB`）**：因果 Gated DeltaNet (GDN) の数値整合性を保つためパディングなしでエンコードし、`[1, L_true, 4096]` を CPU 側 8 件 LRU キャッシュ（`_prompt_cache`）および `--prompt-cache-dir` に保存します。
   - **Stage 2 & 3（`MOSSImageTransformer` 約 5.08B `BF16` / `9.4 GiB` + `AutoencoderKLFlux2` `FP32` / `~0.3 GiB`）**：`--offload model`（`tpu_budget_gib() < 40` で `--offload auto` が自動選択）によりプロンプト変更時のみ `text_encoder` と `transformer` を入れ替え、`vae` は初回ロード後に TPU へ常駐させます。
2. **`drop_tpu` ゼロ D2H 破棄と非同期 `non_blocking` H2D（`tpu_runtime.py`）**：
   - **`drop_tpu`（`--stage-switch-mode auto` / `drop_tpu`）**：ホスト RAM `>= 36 GiB` 環境でデフォルト有効。`_release_module_to_meta` でオフステージのパラメータを 0 バイト `meta` テンソルへ解放し（D2H = `0.0s`）、`_restore_module_from_cpu_snapshot` で CPU スナップショットから直接復元します。`172.95 GiB` ホスト環境での Run 2 後プロセス RSS は `30.91 GiB`（空き `138.24 GiB`）です。
   - **非同期 `non_blocking=True` H2D**：`.to(tpu, non_blocking=True)` を単一の `torch.tpu.synchronize()` でまとめ、400 テンソル（`12.50 GiB`）合成転送速度を `0.53 GiB/s`（`23.78s`）から `1.96 GiB/s`（`6.38s`、事前 pin で `2.05 GiB/s` / `6.10s`）へ向上しました。
   - **約 `5.0 GiB` の残留 HBM プラットフォーム**：初回完全実行後は `d2h`（`5.00–5.03 GiB`）・`drop_tpu`（`5.01–5.03 GiB`）ともに `torch.tpu.memory_allocated()` に約 `5.0 GiB` が残りますが、Python 側の生存 TPU テンソルは `0` 個（`gc.get_objects()`）であり、TorchTPU/PJRT C++ ランタイムおよびコンパイルキャッシュによる一度限りの固定領域です。
3. **Stage 1 Pallas GDN ブリッジ（`tpu_qwen35.py`）とステージ別 Eager 切替（`--text-eager-mode auto`）**：
   - 全 24 層の `GatedDeltaNet`（`Hk=16, Hv=32, D=128`, `use_cache=False`）を `torch.tpu.pallas.jax_op` 経由で `deltanet_tpu.chunk_gdn`（`128` 境界バケット `128, 256, 384, 512, 768, 1024`）にルーティングし、フォールバック 0 回（`encode-bench` `96/96`、CFG 生成 `48/48`）で動作します。
   - Stage 1 は `default`（`DEFER_NEVER`）+ Pallas、Stage 2/3 は `defer_and_fuse` を自動選択します。
4. **`TPU v6e-1` ハードウェア不具合修正（`tpu_runtime.py`, `scripts/run_tpu.py`）**：
   - **4D ブロードキャスト SDPA マスクの `NaN` 回避**：`text_attn_mask` / `joint_attn_mask` を `[1, 1, 1, S]` から `[1, 1, Q, S]` へ明示拡張し、`S >= 512` での `NaN`（全黒画像）を解消しました。
   - **VAE `GroupNorm` NHWC→NCHW 連続化フック**：`ensure_group_norm_contiguous(vae)` により `.view(...)` 前の非連続テンソル例外を防ぎます。
   - **`--compile` (`e0200`) の分離**：`transformer.compile(backend="tpu", fullgraph=True)` はスケジューラの `.nonzero()` D2H 同期箇所で `vmem_address_out_of_range` ハードウェア例外を起こすため、デフォルトでは `False` としています。

### Stage 1 `encode-bench` 詳細計測（`329` 正トークン + `24` 負トークン、非同期 H2D 最適化前）

| モード | H2D 転送 | コールドエンコード | 負プロンプト | 長さ変更ウォーム | 同一長さ反復 | ピーク HBM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `eager` (`DEFER_NEVER`) | `28.653s` | `1.993s` | `0.908s` | `1.168s` | `0.164s` | `15.79 GiB` |
| `defer_and_fuse` | `20.577s` | `11.023s` | `4.659s` | `4.901s` | `0.072s` | `15.73 GiB` |
| **`eager+gdn_pallas`** | `23.860s` | **`0.310s`** | **`0.930s`** | **`0.161s`** | `0.092s` | `15.83 GiB` |
| `defer_and_fuse+gdn_pallas` | `26.311s` | `6.672s` | `1.684s` | `1.859s` | **`0.049s`** | `15.89 GiB` |

*（32 層 `--gdn-pallas` 出力埋め込みの一致度：正 `cos = 0.999848`、負 `cos = 0.999758`、`NaN`/`Inf` ゼロ。）*

### ローカルコードチェックと単体テスト

```bash
ruff check tpu_runtime.py tpu_qwen35.py scripts/run_tpu.py scripts/bench_qwen35_gdn_tpu.py tests/test_tpu_runtime.py tests/test_tpu_qwen35.py
python3 -m unittest discover -s tests -p "test_tpu*.py" -v
```

<a id="repository-layout"></a>

## リポジトリ構成（TPU 追加・変更点）

| ファイル | 役割 |
| --- | --- |
| [`tpu_runtime.py`](tpu_runtime.py) | 3 ステージ TPU パイプライン（`StagedTPUImagePipeline`）、`drop_tpu` + 非同期 H2D + 常駐 VAE、ステージ別 eager、静的バケット、`[1, 1, Q, S]` SDPA パッチ、FP32 正弦波、`GroupNorm` フック |
| [`tpu_qwen35.py`](tpu_qwen35.py) | `torch.tpu.pallas.jax_op` による `deltanet_tpu.chunk_gdn` ブリッジ、`128` アライン因果パディング、永続 CPU プロンプトキャッシュ |
| [`scripts/run_tpu.py`](scripts/run_tpu.py) | TPU CLI（`probe`, `encode-only`, `encode-bench`, `generate`, `benchmark`, `stage-switch-bench`） |
| [`scripts/bench_qwen35_gdn_tpu.py`](scripts/bench_qwen35_gdn_tpu.py) | 単体 GDN 再帰カーネル & `Qwen3_5GatedDeltaNet` 層 A/B 計測スクリプト |
| [`scripts/setup-tpu.sh`](scripts/setup-tpu.sh) | Colab / TPU VM 用 TorchTPU `0.2.0` + PyTorch CPU セットアップスクリプト |
| [`notebooks/moss_image_torchtpu_colab.ipynb`](notebooks/moss_image_torchtpu_colab.ipynb) | Google Colab エンドツーエンド TPU ノートブック |
| [`docs/tpu.md`](docs/tpu.md) | `TPU v6e-1` 詳細立ち上げログ、エラー解析、Plan A ステージ切替ベンチマーク結果 |
| [`gradio_app.py`](gradio_app.py) / [`inference_runtime.py`](inference_runtime.py) | CUDA、Apple Silicon MPS、TorchTPU のデバイスルーティングとメタデータ記録 |
| [`tests/test_tpu_runtime.py`](tests/test_tpu_runtime.py) / [`tests/test_tpu_qwen35.py`](tests/test_tpu_qwen35.py) | 32 件の CPU 単体・数値一致テスト |

<a id="next-roadmap"></a>

## 今後の最適化ロードマップ

性能改善の議論は **[GitHub Discussions #5](https://github.com/azure-swing/MOSS-Image/discussions/5)** で継続中です：
1. `[1, 1, 4487, 4487]` 4D マスク付き `SDPA`（`~2.38 ms/layer`）を Pallas `SplashAttention` に置き換え、DiT の MFU を `38.4%` 以上へ向上。
2. `FlowMatchEulerDiscreteScheduler.step()` 内の毎ステップ `.nonzero()` D2H 同期を除去し、`e0200` VMEM エラーを解消して `transformer.compile(backend="tpu", fullgraph=True)` を有効化。
3. スキップされない 41 層の CFG を `batch=2` に統合し、`128, 256` の小型プロンプトバケットを追加。
4. `Qwen3.5-9B` の INT8/W4 量子化により全ステージを `31.24 GiB`（`--offload none`）に収め、プロンプト切替時の H2D 転送をゼロ化。

<a id="upstream-and-citation"></a>

## 上流プロジェクト・引用

本作業は **[OpenMOSS/MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)** に基づいています。ベースモデル、CUDA/MPS 利用方法、Gradio、ComfyUI 連携、構造化プロンプト拡張については上流ドキュメントをご覧ください：

- [ユーザーガイド (EN)](USERGUIDE.md) · [用户指南 (中文)](USERGUIDE_ZH.md)
- [Apple Silicon / MPS](docs/macos.md) · [ComfyUI 連携](comfyui/README.md) · [Prompt Enhancer Skill](skills/moss-image-prompt-enhancer/SKILL.md)
- モデル：[MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base) · [MOSS-Image-SFT](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT) · [技術レポート (arXiv:2610.23374)](https://arxiv.org/abs/2610.23374)

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

推論コードは [Apache License 2.0](LICENSE) の下で提供されています。OpenMOSS、Diffusers、Qwen、FLUX、TorchTPU、および `deltanet-tpu` チームに感謝します。
