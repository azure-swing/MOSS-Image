# Google Cloud TPU (TorchTPU) Support

The native PyTorch `torch_tpu` backend (`device="tpu"`) targets Colab `TPU v6e-1` (32 GiB HBM). The transformer patch has CPU parity tests against the original modeling code. TPU compilation, image quality, and HBM use require hardware validation.

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

Each mode reports its first and subsequent generations separately. The first generation can include prompt encoding, model transfers, and compilation. Cache hits can reduce compilation time on any run. The report includes the actual denoising step count and mean denoising time per step. Use `--prompt-embeds` to compare runs without prompt encoding or stage transfers.

### Decoupled text encoding and generation

Encode on a TPU with enough HBM for the BF16 text encoder and its activations, or pass `--allow-cpu` on a host with sufficient RAM. Copy the resulting file to the generation host. The generation command loads the DiT and VAE and can be evaluated separately on a 16 GiB TPU v5e-1. Use the same model and guidance settings for both commands.

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
python3 scripts/run_tpu.py --mode generate --offload model --steps 2 --seed 42 --output outputs/eager_test.png  # --compile defaults to True
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
- **Root cause**: The `UserFatal` core halt (`vmem_address_out_of_range_vld0`/`vld1`) is triggered asynchronously inside compiled DiT execution (`transformer.compile(backend="tpu", dynamic=False, fullgraph=True)`), and is only reported when `scheduler.step()` triggers the next synchronous D2H copy.

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

### Colab CLI Operational Note (`google-colab-cli 0.8.0`)

- `colab new --tpu v6e1` may hit `503 Service Unavailable`, proxy idle disconnects (`RemoteDisconnected`), or `ReadTimeout` (`COLAB_CLI_REQUEST_TIMEOUT=70`) if backend allocation finishes after the HTTP wait window. Always run `colab sessions` afterward to inspect or adopt `[?]` background-allocated runtimes and `colab stop -s <name>` when finished so compute units are not leaked.

### Recommended Follow-up Fixes (Pending Live TPU Re-verification)

1. Synchronize/isolate `torch.compile` on `MOSSImageTransformer` (or test eager/`defer_and_fuse` vs op-level `fullgraph=False` to pinpoint which DiT op lowers with out-of-range VMEM loads on `v6e`).
2. Guard `AutoencoderKLFlux2` / `GroupNorm` inputs with `.contiguous()` (or run the ~0.3 GiB FP32 VAE decode on CPU) so channels-last convolution outputs do not break `torch.group_norm` `.view()`.

