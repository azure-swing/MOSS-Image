#!/usr/bin/env python3
"""Run or benchmark MOSS-Image on Google Cloud TPU via TorchTPU (torch_tpu)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import tpu_runtime  # noqa: E402
from inference_runtime import DEFAULT_MODEL, model_source  # noqa: E402


def _parse_buckets(raw: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not values or any(v <= 0 for v in values):
        raise ValueError(f"Invalid bucket list: {raw!r}")
    return values


def _require_tpu_or_override(allow_cpu: bool) -> torch.device:
    if tpu_runtime.is_available():
        return tpu_runtime.tpu_device()
    if allow_cpu:
        return torch.device("cpu")
    raise RuntimeError(
        "Google Cloud TPU / torch_tpu accelerator was not detected. "
        "Run on a Colab TPU runtime (with torch_tpu installed) or pass --allow-cpu for dry checks."
    )


def run_probe(args) -> dict:
    info = {
        "torch_version": torch.__version__,
        "accelerator_tpu_available": tpu_runtime.is_available(),
        "device_name": tpu_runtime.device_name(),
        "tpu_budget_gib": round(tpu_runtime.tpu_budget_gib(), 2),
        "pallas_available": hasattr(getattr(torch, "tpu", None), "pallas"),
        "eager_modes": (
            list(tpu_runtime.VALID_EAGER_MODES)
            if hasattr(getattr(torch, "tpu", None), "set_eager_mode")
            else []
        ),
    }
    if tpu_runtime.is_available() or args.allow_cpu:
        dev = _require_tpu_or_override(args.allow_cpu)
        tpu_runtime.configure_precision(args.matmul_precision)
        tpu_runtime.configure_eager_mode(args.eager_mode)
        t0 = time.perf_counter()
        a = torch.randn((512, 512), device=dev, dtype=torch.bfloat16)
        b = torch.randn((512, 512), device=dev, dtype=torch.bfloat16)
        c = a @ b
        _ = float(c[0, 0].float().item())
        tpu_runtime.synchronize()
        info["matmul_smoke_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        info["peak_hbm_gib"] = tpu_runtime.max_memory_allocated_gib()
    print(json.dumps(info, indent=2))
    return info


def run_encode_only(args) -> dict:
    dev = _require_tpu_or_override(args.allow_cpu)
    root = model_source(args.model, local_files_only=args.local_files_only)
    module, _ = tpu_runtime._import_pipeline_module(root)

    root_path = Path(root)
    tokenizer_path = root_path / "tokenizer"
    if not tokenizer_path.is_dir():
        tokenizer_path = root_path / "text_encoder"

    common = {"local_files_only": args.local_files_only}
    t0 = time.perf_counter()
    tokenizer = module.AutoTokenizer.from_pretrained(tokenizer_path, **common)
    text_encoder = module.AutoModel.from_pretrained(
        root_path / "text_encoder", dtype=torch.bfloat16, **common
    ).eval()
    pipe = module.MOSSImagePipeline(
        transformer=None,
        scheduler=module.MOSSImageScheduler(),
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        vae=None,
    )
    text_encoder.to(dev)
    tpu_runtime.configure_eager_mode(args.eager_mode)

    with tpu_runtime._override_execution_device(pipe, dev):
        pos_embeds, pos_mask = pipe.encode_prompt(
            args.prompt, max_sequence_length=args.max_sequence_length
        )
        neg_embeds, neg_mask = (None, None)
        if args.guidance_scale != 1.0:
            neg_embeds, neg_mask = pipe.encode_prompt(
                args.negative_prompt, max_sequence_length=args.max_sequence_length
            )

    tpu_runtime.synchronize()
    elapsed = time.perf_counter() - t0

    out_path = Path(args.save_embeds)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "prompt_embeds": pos_embeds.detach().cpu(),
        "prompt_attention_mask": pos_mask.detach().cpu().to(torch.bool),
        "negative_prompt_embeds": neg_embeds.detach().cpu() if neg_embeds is not None else None,
        "negative_prompt_attention_mask": (
            neg_mask.detach().cpu().to(torch.bool) if neg_mask is not None else None
        ),
    }
    torch.save(payload, out_path)
    text_encoder.to("cpu")
    tpu_runtime.empty_cache()

    summary = {
        "saved_embeds": str(out_path.resolve()),
        "prompt_tokens": int(payload["prompt_attention_mask"].sum().item()),
        "encode_seconds": round(elapsed, 2),
        "peak_hbm_gib": tpu_runtime.max_memory_allocated_gib(),
    }
    print(json.dumps(summary, indent=2))
    return summary


def _run_once(pipe, args, embeds_payload: dict | None, generator_seed: int) -> dict:
    gen = torch.Generator("cpu").manual_seed(generator_seed)
    tpu_runtime.reset_peak_memory_stats()
    tpu_runtime.synchronize()
    t0 = time.perf_counter()

    call_kwargs = {
        "height": args.height,
        "width": args.width,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "generator": gen,
        "max_sequence_length": args.max_sequence_length,
        "output_type": "pil",
    }
    if embeds_payload is not None:
        call_kwargs["prompt_embeds"] = embeds_payload["prompt_embeds"]
        call_kwargs["prompt_attention_mask"] = embeds_payload["prompt_attention_mask"]
        if args.guidance_scale != 1.0:
            call_kwargs["negative_prompt_embeds"] = embeds_payload["negative_prompt_embeds"]
            call_kwargs["negative_prompt_attention_mask"] = embeds_payload[
                "negative_prompt_attention_mask"
            ]
        seq_len = int(embeds_payload["prompt_embeds"].shape[1])
    else:
        call_kwargs["prompt"] = args.prompt
        call_kwargs["negative_prompt"] = args.negative_prompt
        seq_len = None

    result = pipe(**call_kwargs)
    tpu_runtime.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gib = tpu_runtime.max_memory_allocated_gib()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(out_path)
    return {
        "elapsed_seconds": round(elapsed, 3),
        "ms_per_step": round(elapsed * 1000.0 / max(args.steps, 1), 2),
        "peak_hbm_gib": round(peak_gib, 3) if peak_gib is not None else None,
        "prompt_seq_len_input": seq_len,
        "output_image": str(out_path.resolve()),
    }


def run_generate_or_benchmark(args) -> dict:
    dev = _require_tpu_or_override(args.allow_cpu)
    buckets = _parse_buckets(args.buckets)
    embeds_payload = (
        torch.load(args.prompt_embeds, map_location="cpu", weights_only=True)
        if args.prompt_embeds
        else None
    )
    include_text_encoder = embeds_payload is None

    if args.mode == "generate":
        t0 = time.perf_counter()
        pipe = tpu_runtime.load_pipeline(
            args.model,
            offload=args.offload,
            compile_transformer=args.compile,
            local_files_only=args.local_files_only,
            prompt_buckets=buckets,
            eager_mode=args.eager_mode,
            matmul_precision=args.matmul_precision,
            include_text_encoder=include_text_encoder,
            device=dev,
        )
        load_seconds = round(time.perf_counter() - t0, 2)
        runs = []
        for idx in range(args.runs):
            stats = _run_once(pipe, args, embeds_payload, args.seed + idx)
            stats["run_index"] = idx + 1
            runs.append(stats)
        report = {
            "mode": "generate",
            "model": args.model,
            "device": tpu_runtime.device_name() or str(dev),
            "offload": args.offload,
            "eager_mode": args.eager_mode,
            "compile": args.compile,
            "buckets": list(buckets),
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "load_seconds": load_seconds,
            "runs": runs,
        }
    else:
        bench_configs = []
        for item in [m.strip() for m in args.bench_modes.split(",") if m.strip()]:
            if item == "eager":
                bench_configs.append(("eager", "default", False))
            elif item == "defer_and_fuse":
                bench_configs.append(("defer_and_fuse", "defer_and_fuse", False))
            elif item == "compile":
                bench_configs.append(("compile", "default", True))
            else:
                raise ValueError(f"Unknown benchmark mode: {item!r}")

        results = {}
        for name, eager_mode, compile_flag in bench_configs:
            pipe = tpu_runtime.load_pipeline(
                args.model,
                offload=args.offload,
                compile_transformer=compile_flag,
                local_files_only=args.local_files_only,
                prompt_buckets=buckets,
                eager_mode=eager_mode,
                matmul_precision=args.matmul_precision,
                include_text_encoder=include_text_encoder,
                device=dev,
            )
            mode_runs = []
            for idx in range(max(args.runs, 2)):
                stats = _run_once(pipe, args, embeds_payload, args.seed)
                stats["run_index"] = idx + 1
                mode_runs.append(stats)
            results[name] = mode_runs
            del pipe
            tpu_runtime.empty_cache()

        report = {
            "mode": "benchmark",
            "model": args.model,
            "device": tpu_runtime.device_name() or str(dev),
            "height": args.height,
            "width": args.width,
            "steps": args.steps,
            "buckets": list(buckets),
            "results": results,
        }

    if args.save_report:
        report_path = Path(args.save_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MOSS-Image on Google Cloud TPU (TorchTPU).")
    parser.add_argument(
        "--mode",
        choices=("probe", "encode-only", "generate", "benchmark"),
        default="generate",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default="A studio macro photo of a ceramic teapot on a linen table.")
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--prompt-embeds", default=None, help="Path to precomputed .pt prompt embeddings.")
    parser.add_argument("--save-embeds", default="outputs/prompt_embeds.pt")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--max-sequence-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offload", choices=tpu_runtime.VALID_OFFLOAD_MODES, default="auto")
    parser.add_argument("--eager-mode", choices=tpu_runtime.VALID_EAGER_MODES, default="default")
    parser.add_argument("--matmul-precision", choices=tpu_runtime.VALID_MATMUL_PRECISIONS, default="high")
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--buckets", default="512,1024,2048,4096")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--bench-modes", default="eager,defer_and_fuse,compile")
    parser.add_argument("--output", default="outputs/moss_tpu.png")
    parser.add_argument("--save-report", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true", help="Fallback to CPU for dry runs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.mode == "probe":
        run_probe(args)
    elif args.mode == "encode-only":
        run_encode_only(args)
    else:
        run_generate_or_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
