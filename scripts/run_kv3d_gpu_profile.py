#!/usr/bin/env python3
"""Run Qwen3 offline 3D KV profiling on GPU."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoConfig

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d import ProfilingSample, filter_profiling_plan, generate_profiling_plan, load_hf_dataset_split
from src.kv3d.datasets import row_to_sample_for_dataset
from src.kv3d.executor import load_model_bundle, run_model_profiling_plan
from src.kv3d.io import write_profile_artifacts
from src.kv3d.runner import run_offline_3d_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--samples", type=Path, default=None)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--config-name", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--span-size", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--max-heads", type=int, default=0)
    parser.add_argument("--num-chunks", type=int, default=0)
    parser.add_argument("--num-spans", type=int, default=0)
    parser.add_argument("--include-addition", action="store_true")
    parser.add_argument("--include-chunk-level", dest="include_chunk_level", action="store_true", default=True)
    parser.add_argument("--no-include-chunk-level", dest="include_chunk_level", action="store_false")
    parser.add_argument("--chunk-targets", type=Path, default=None)
    parser.add_argument("--span-targets", type=Path, default=None)
    parser.add_argument("--methods", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_samples(args: argparse.Namespace) -> list[ProfilingSample]:
    spec = json.loads(args.spec.read_text()) if args.spec else {}
    if args.dataset_name:
        rows = load_hf_dataset_split(args.dataset_name, args.split, config_name=args.config_name or None)
        samples = [row_to_sample_for_dataset(args.dataset_name, dict(row), spec=spec) for row in rows]
    elif args.samples:
        payload = json.loads(args.samples.read_text())
        samples = [ProfilingSample(**item) for item in payload]
    else:
        raise SystemExit("provide either --samples or --dataset-name")
    if args.sample_offset < 0:
        raise SystemExit("--sample-offset must be non-negative")
    return samples[args.sample_offset : args.sample_offset + args.max_samples]


def _method_filter(methods: str) -> set[str]:
    return {method.strip() for method in methods.split(",") if method.strip()}


def _load_chunk_targets(path: Path | None) -> dict[str, set[tuple[int, int]]] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    targets: dict[str, set[tuple[int, int]]] = {}
    for sample_id, items in payload.items():
        targets[str(sample_id)] = {tuple(map(int, item)) for item in items}
    return targets


def main() -> None:
    args = parse_args()
    samples = _load_samples(args)
    if not samples:
        raise SystemExit("no samples loaded")

    if args.max_layers <= 0 or args.max_heads <= 0:
        model_config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
        default_layers = int(model_config.num_hidden_layers)
        default_heads = int(getattr(model_config, "num_key_value_heads", model_config.num_attention_heads))
    else:
        default_layers = args.max_layers
        default_heads = args.max_heads
    num_layers = args.max_layers or default_layers
    num_heads = args.max_heads or default_heads
    span_size = int(getattr(args, "span_size", 0) or 0)
    block_size = span_size or args.chunk_size
    if block_size <= 0:
        raise SystemExit("--span-size/--chunk-size must be positive")
    explicit_num_spans = int(getattr(args, "num_spans", 0) or 0)
    explicit_num_chunks = int(getattr(args, "num_chunks", 0) or 0)
    num_chunks = explicit_num_spans or explicit_num_chunks or max(1, (args.max_context_tokens + block_size - 1) // block_size)
    target_path = getattr(args, "span_targets", None) or args.chunk_targets
    sample_ids = [sample.sample_id for sample in samples]
    plan = generate_profiling_plan(
        sample_ids=sample_ids,
        num_layers=num_layers,
        num_heads=num_heads,
        num_chunks=num_chunks,
        include_addition=args.include_addition,
        include_chunk_level=args.include_chunk_level,
        chunk_targets=_load_chunk_targets(target_path),
    )
    plan = filter_profiling_plan(plan, methods=_method_filter(args.methods))
    bundle = load_model_bundle(args.model_name, device_map=args.device_map, dtype=args.dtype)
    records = run_model_profiling_plan(
        bundle=bundle,
        samples=samples,
        plan=plan,
        chunk_size=block_size,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens=args.max_new_tokens,
    )
    result = run_offline_3d_profile(
        samples=samples,
        num_layers=num_layers,
        num_heads=num_heads,
        num_chunks=num_chunks,
        include_addition=args.include_addition,
        include_chunk_level=args.include_chunk_level,
        records=records,
        model_name=args.model_name,
    )

    result["plan"] = plan
    attention_enabled = any(record.get("attention_divergence") is not None for record in result["records"])
    write_profile_artifacts(
        result=result,
        output_dir=args.output_dir,
        summary={
            "record_count": len(result["records"]),
            "plan_size": len(plan),
            "model_name": args.model_name,
            "chunk_size": block_size,
            "span_size": block_size if span_size else None,
            "num_spans": num_chunks if span_size else None,
            "max_context_tokens": args.max_context_tokens,
            "max_new_tokens": args.max_new_tokens,
            "include_addition": args.include_addition,
            "include_chunk_level": args.include_chunk_level,
            "attention_divergence_enabled": attention_enabled,
            "sample_offset": args.sample_offset,
            "max_samples": args.max_samples,
            "methods": args.methods,
            "profiling_mode": "two_stage_layer_head_then_span" if span_size else "chunk",
        },
    )


if __name__ == "__main__":
    main()
