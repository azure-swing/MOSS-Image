# Google Cloud TPU (TorchTPU) Support

The native PyTorch `torch_tpu` backend (`device="tpu"`) targets Colab `TPU v6e-1` (32 GiB HBM). The transformer patch has CPU parity tests against the original modeling code, and both strict eager (`default`) and `defer_and_fuse` execution have been validated end-to-end on live Colab `TPU v6e-1` hardware.

## Architecture Summary

1. **Stage 1 — Exact `Qwen3.5-9B` Text Encoding (15.6 GiB BF16)**:
   - Runs unpadded individual prompt forwards on `text_encoder` to preserve Qwen3.5 Gated DeltaNet (GDN) numerical semantics (`pipeline_moss_image.py`).
   - Detaches `[1, L_true, 4096]` hidden states to CPU and stores them in an 8-entry LRU prompt cache, or exports them via `--mode encode-only`.
2. **Stage 2 & 3 — Static-Bucketed `MOSSImageTransformer` (5B DiT) + FP32 `FLUX.2` VAE**:
   - Offloads `text_encoder` before loading `transformer` (~10.2 GiB BF16) and `vae` (~0.3 GiB FP32) onto TPU (`--offload model` / staged), leaving them resident across subsequent seeds and cached prompts.
   - Uses a precomputed float32 frequency buffer for the DiT timestep sinusoid.
   - Uses static bucket padding (`512, 1024, 2048, 4096`), masked SDPA, and cumulative RoPE offsets to keep the DiT's input shapes fixed within a bucket.

## Quick Start (Google Colab `TPU v6e-1`)

```bash
bash scripts/setup-tpu.sh
python3 scripts/run_tpu.py --mode probe
python3 scripts/run_tpu.py --mode generate --prompt "A ceramic teapot on linen"
```

> Runs strict eager (`default` / `DEFER_NEVER`, `--no-compile`) by default, which is verified end-to-end on `TPU v6e-1`. For ~3x faster warm generation (`16.72s` vs `49.08s` at `848x1200`, 55 steps) and ~8 GiB lower peak HBM (`16.3 GiB` vs `24.5 GiB`), pass `--eager-mode defer_and_fuse`. Pass `--compile` only for experimental DiT compilation debugging (`e0200: RuntimeUnexpectedCoreHalt`).

### Eager / Defer-and-Fuse Benchmarking

```bash
python3 scripts/run_tpu.py \
  --mode benchmark \
  --bench-modes eager,defer_and_fuse \
  --steps 28 \
  --save-report outputs/tpu_benchmark.json
```

Each mode reports its first and subsequent generations separately. The first generation can include prompt encoding, model transfers, and compilation. Cache hits can reduce compilation time on any run. The report includes the actual denoising step count and mean denoising time per step. Use `--prompt-embeds` to compare runs without prompt encoding or stage transfers.

### Decoupled text encoding and generation

Encode on a TPU with enough HBM for the BF16 text encoder and its activations, or pass `--allow-cpu` on a host with sufficient RAM. Copy the resulting file to the generation host. The generation command loads the DiT and VAE and can be evaluated separately on a 16 GiB TPU v5e-1. Use the same model and guidance settings for both commands.

```bash
python3 scripts/run_tpu.py --mode encode-only --prompt "..." --save-embeds outputs/embeds.pt
python3 scripts/run_tpu.py --mode generate --prompt-embeds outputs/embeds.pt --offload none
```

### Persistent Compilation Cache

Set these before running to reuse XLA executables across restarts:

```bash
export TORCH_TPU_TIER2_COMPILATION_CACHE="${HOME}/.cache/torch_tpu/tier2"
export TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT="${HOME}/.cache/torch_tpu/tier3"
```

## Notes on Pallas SplashAttention

`tpu_runtime.py` lowers attention through `F.scaled_dot_product_attention(..., attn_mask=...)` for exact mathematical parity across padded prompt buckets. Custom Pallas `splash_attention_kernel` bridges via `torch.tpu.pallas.jax_op` (following `google-pytorch/torch_tpu` FLUX optimization recipes) require fixed block-aligned masks (`128`-token alignment) and live TPU verification and are left as an explicit Phase 2 hardware tuning path.

## Live Colab `TPU v6e-1` Validation & Known Errors (`2026-11-13`)

Tested commit `f41afbd` (`feat/torchtpu-support`) on a live Google Colab `V6E1` runtime (`TFRT TPU v6 lite`, PJRT C API `0.116`, `libtpu 0.0.50`, `Python 3.13.18`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/content/tpu_cache/tier3`).

### Verified Working Stages

- `bash scripts/setup-tpu.sh`: exited `0`.
- `python3 scripts/run_tpu.py --mode probe`: exited `0`.
- Checkpoint loading (`759/759` weights), Stage 1 `Qwen3.5-9B` prompt encoding (`DEFER_NEVER` + Tier-3 cache), Stage 2 `MOSSImageTransformer` transfer, and (with `--no-compile`) the multi-step DiT denoising loop + `FlowMatchEulerDiscreteScheduler.step()` completed on hardware.

### Error 1 — `--compile` Triggers Asynchronous TPU Core Halt (`e0200` / `vmem_address_out_of_range`)

**Commands**

```bash
python3 scripts/run_tpu.py --mode generate --offload model --compile --steps 28 --seed 42 --output outputs/colab_tpu_sample.png
python3 scripts/run_tpu.py --mode generate --offload model --steps 2 --seed 42 --output outputs/eager_test.png  # tested on f41afbd when --compile defaulted to True
```

**Observed Error**

Surfaced at step 0 during the first host D2H sync inside `diffusers/schedulers/scheduling_flow_match_euler_discrete.py:414` (`(schedule_timesteps == timestep).nonzero()`):

```text
E0000 ... pjrt_transfer.cc:391] Async D2H ToLiteral transfer failed: INTERNAL: E0200: RuntimeUnexpectedCoreHalt: Program or fatal error occurred; computation may be invalid: (launch_id=0)
...
RuntimeError: failed to copy buffer from device into host buffer -- the XLA compiler failed with: e0200: RuntimeUnexpectedCoreHalt: Program or fatal error occurred; computation may be invalid: (launch_id=0)Assertion args: INTERNAL: Fatal error occurred during TPU execution. ...
Detailed error: Node 1 halted unexpectedly at tag:pc SparseCoreSequencer:0:0x5 (from SparseCoreSequencer:0:0x5): no debugging message found for this tag:pc. ; HLO module: tt_jit_nonzero
  0x0x0_SC1 SparseCoreTileExecuteCoreSequencer:0..15 halted at tag:pc SparseCoreTileExecuteCoreSequencer:0:0 ...
Summary of reported fatal error messages by core:
core location: 0x0x0_SC1, error type: UserFatal, message: *** [Physical location: tpu15:pe4:2] A UserFatal error has occurred, all pending programs will fail, and results may be corrupted.
...
Anomalies (4 total):
[user] chip_tc_tcs_tc_main_fatal_errors_tc_vector_program_errors0.node: 0x20000000000: vmem_address_out_of_range_vld0
[user] chip_tc_tcs_tc_main_fatal_errors_tc_vector_program_errors0.node: 0x40000000000: vmem_address_out_of_range_vld1
```

**Isolation Results**

- Running `torch.tensor([False, True, False], device="tpu").nonzero()` in isolation exited `0` (`tensor([[1]], device='tpu:0')`).
- Running the scheduler shape `t = torch.linspace(1000.0, 1.0, 29, device="tpu"); (t[:-1] == t[0]).nonzero()` under `torch.tpu.EagerMode.DEFER_NEVER` with `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT` set also exited `0` (`tensor([[0]], device='tpu:0')`).
- Passing `--no-compile` bypassed the core halt and finished the DiT denoising loop and `scheduler.step()` cleanly.
- **Scope proven by logs**: Enabling `--compile` (`transformer.compile(backend="tpu", dynamic=False, fullgraph=True)`) causes `e0200: RuntimeUnexpectedCoreHalt` (`vmem_address_out_of_range_vld0`/`vld1`) surfaced at the first synchronous host D2H copy (`scheduler.step()` -> `.nonzero()`), whereas `--no-compile` completes Qwen3.5 encoding and the DiT + scheduler denoising loop cleanly. The exact compiled operation or lowering triggering the VMEM out-of-range fault has not yet been isolated with intermediate device synchronizations.

### Error 2 — Eager `--no-compile` Fails in VAE Decode (`torch.group_norm` Channels-Last View)

**Command**

```bash
python3 scripts/run_tpu.py --mode generate --offload model --no-compile --steps 2 --seed 42 --output outputs/nocompile_test.png
```

**Observed Error**

After `Qwen3.5-9B` encoding, 2 DiT denoising steps, and `FlowMatchEulerDiscreteScheduler` all completed, Stage 3 (`AutoencoderKLFlux2` decode in `vae.py:300` -> `unet_2d_blocks.py:746` -> `resnet.py:326` -> `F.group_norm`) failed:

```text
RuntimeError: view(): cannot create a view of shape [1, 32, 196608] from the input tensor of shape [1, 384, 128, 128] and strides [384, 1, 49152, 384]; consider creating a new tensor using reshape() instead of taking a view
```

- **Root cause**: `torch_tpu` convolution/decoder blocks output a non-contiguous channels-last (`NHWC`, strides `[384, 1, 49152, 384]`) tensor, and eager `torch.group_norm` attempts a `.view(...)` reshape without calling `.contiguous()`.

### Error 3 — Broadcast 4D SDPA Masks (`[1, 1, 1, S]`) Produce `NaN`s at `S >= 512` (All-Black Image)

Tested commit `70a8c7e` on `TPU v6e-1` with `--mode generate --offload model --height 1200 --width 848 --steps 55`:

- **Observed Symptom**: All three stages (`Qwen3.5-9B`, 55-step DiT denoising, `AutoencoderKLFlux2` decode with the `GroupNorm` hook) exited `0` (`peak_hbm_gib: 24.153`), but saved a solid-black `(1200, 848, 3)` PNG (`min=0, max=0`) alongside `RuntimeWarning: invalid value encountered in cast` in `diffusers/image_processor.py`.
- **Bisection & Root Cause**:
  - `Qwen3.5-9B` embeddings were clean (`nan=False`, `absmax=50.0`), while DiT output latents already contained `NaN`.
  - Forward hooks isolated the first `NaN` immediately inside attention output projections: `text_embedder.blocks.0.proj` when bucket-padded (`S=512`), and `layers.0.proj.o` when unpadded (`S=3975+30=4005`, after `text_embedder` passed at `S=30`).
  - Standalone TPU `F.scaled_dot_product_attention` tests showed `torch_tpu`'s lowered Mosaic/cuDNN kernel silently miscomputes 4D boolean masks that broadcast the query axis (`[1, 1, 1, S]`, `maxdiff=2.1367` at `S=30`) and emits `NaN` for `S >= 512` (`S=512`, `S=4005`). Expanding the mask to `[1, 1, Q, S]` matches CPU `float32` SDPA to BF16 precision even with poisoned masked keys/values (`valid_maxdiff=0.0029` at `S=4487`, `2.38 ms`).

### Colab CLI Operational Note (`google-colab-cli 0.8.0`)

- `colab new --tpu v6e1` may hit `503 Service Unavailable`, proxy idle disconnects (`RemoteDisconnected`), or `ReadTimeout` (`COLAB_CLI_REQUEST_TIMEOUT=70`) if backend allocation finishes after the HTTP wait window. Always run `colab sessions` afterward to inspect or adopt `[?]` background-allocated runtimes and `colab stop -s <name>` when finished so compute units are not leaked.

### Applied Fixes & End-to-End Hardware Verification (`TPU v6e-1`)

1. **Default `--compile` to `False` (`scripts/run_tpu.py`)**: `scripts/run_tpu.py` defaults `--compile` to `False` and `--bench-modes` to `"eager,defer_and_fuse"`, keeping `--compile` opt-in while the compiled DiT `vmem_address_out_of_range` lowering issue is isolated further.
2. **Enforce NCHW-contiguous `GroupNorm` inputs (`tpu_runtime.py`)**: `ensure_group_norm_contiguous(vae)` installs an idempotent forward pre-hook on all `nn.GroupNorm` submodules in `StagedTPUImagePipeline` so channels-last convolution outputs (`[384, 1, 49152, 384]`) are converted via `.contiguous()` before `torch.group_norm` calls `.view(...)`.
3. **Expand 4D static SDPA mask query dimension (`tpu_runtime.py`)**: `MOSSImageTransformer.forward` expands `text_attn_mask` and `joint_attn_mask` from `[1, 1, 1, S]` to `[1, 1, Q, S]`, backed by a guard in `_attention`, eliminating `NaN`s in both padded text projection and joint DiT blocks.

Verified on live Colab `TPU v6e-1` (`TFRT TPU v6 lite`, `PJRT 0.116`, `libtpu 0.0.50`, `OpenMOSS-Team/MOSS-Image-Base`, `848x1200`, `55 steps`, `guidance_scale=3.0`, `uncond_skip_layers=(22,)`, `329` prompt tokens -> bucket `512`, `TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT=/content/tpu_cache/tier3`):

- **Stage 1 standalone `Qwen3.5-9B` encode (`--mode encode-only`)**: `39.52s`, `15.801 GiB` peak HBM.
- **Full cold end-to-end strict eager run (`prompt` + `negative_prompt` -> PNG)**: `124.19s` total (`52.978s` denoising / `963.23 ms/step`, `24.496 GiB` peak HBM, verified valid RGB image).
- **`--mode benchmark` (`848x1200`, 55 steps, cached `--prompt-embeds`)**:
  - `eager` (`DEFER_NEVER`): Run 1 `70.201s` (`48.883s` denoising, `24.472 GiB`); Run 2 warm `49.082s` (`48.612s` denoising / `883.85 ms/step`, `24.473 GiB`).
  - `defer_and_fuse`: Run 1 `35.556s` (`16.605s` denoising, `14.963 GiB`); Run 2 warm `16.724s` (`16.434s` denoising / `298.79 ms/step`, `16.331 GiB`).
- **10-seed end-to-end batch (`StagedTPUImagePipeline`, `defer_and_fuse`, `--offload model`)**: `246.17s` total across all 10 distinct random seeds (`load_seconds=3.326s`, Run 1 cold `95.563s` including Stage 1 Qwen encoding + compile, Runs 2–10 warm mean `16.734s` total / `16.429s` denoising / `298.71 ms/step`, `16.248 GiB` peak HBM; 10/10 verified valid RGB images).

