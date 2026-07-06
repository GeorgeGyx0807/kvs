#!/usr/bin/env python3
"""Build a JSON run specification for offline 3D KV profiling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d.run_spec import build_kv3d_run_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, required=True)
    parser.add_argument("--max-context-tokens", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--include-addition", action="store_true")
    parser.add_argument("--include-chunk-level", dest="include_chunk_level", action="store_true", default=True)
    parser.add_argument("--no-include-chunk-level", dest="include_chunk_level", action="store_false")
    parser.add_argument("--shard-size", type=int, default=None)
    parser.add_argument("--methods", default="")
    parser.add_argument("--min-samples-gate", type=int, required=True)
    parser.add_argument("--min-heterogeneity-range", type=float, default=None)
    parser.add_argument("--no-require-stability", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = build_kv3d_run_spec(
        profile_name=args.profile_name,
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        split=args.split,
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_chunks=args.num_chunks,
        chunk_size=args.chunk_size,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens=args.max_new_tokens,
        include_addition=args.include_addition,
        include_chunk_level=args.include_chunk_level,
        min_samples_gate=args.min_samples_gate,
        min_heterogeneity_range=args.min_heterogeneity_range,
        require_stability=not args.no_require_stability,
        shard_size=args.shard_size,
        methods=args.methods,
        spec_path=str(args.output),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(spec, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
