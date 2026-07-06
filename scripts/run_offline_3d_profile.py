#!/usr/bin/env python3
"""CLI for the offline 3D KV profiling orchestrator."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d import (
    ProfilingSample,
    annotate_record_token_spans,
    load_hf_dataset_split,
    run_offline_3d_profile,
    row_to_sample_for_dataset,
)
from src.kv3d.io import read_records, write_profile_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--include-addition", action="store_true")
    parser.add_argument("--include-chunk-level", dest="include_chunk_level", action="store_true", default=True)
    parser.add_argument("--no-include-chunk-level", dest="include_chunk_level", action="store_false")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", default="")
    parser.add_argument("--config-name", default="")
    parser.add_argument("--split", default="")
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--records", type=Path, default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec.read_text()) if args.spec else {}
    if args.dataset_name and args.split:
        dataset = load_hf_dataset_split(
            args.dataset_name,
            args.split,
            config_name=args.config_name or None,
        )
        samples = [row_to_sample_for_dataset(args.dataset_name, dict(row), spec=spec) for row in dataset]
    elif args.dataset_name or args.split:
        raise SystemExit("--dataset-name and --split must be provided together")
    else:
        sample_payload = json.loads(args.samples.read_text())
        samples = [ProfilingSample(**item) for item in sample_payload]
    records = read_records(args.records) if args.records else []
    if records and args.chunk_size is not None:
        records = annotate_record_token_spans(
            records,
            chunk_size=args.chunk_size,
            max_context_tokens=args.max_context_tokens,
        )
    result = run_offline_3d_profile(
        samples=samples,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_chunks=args.num_chunks,
        include_addition=args.include_addition,
        include_chunk_level=args.include_chunk_level,
        records=records,
    )
    attention_enabled = any(record.get("attention_divergence") is not None for record in result["records"])
    write_profile_artifacts(
        result=result,
        output_dir=args.output_dir,
        summary={
            "record_count": len(result["records"]),
            "plan_size": len(result["plan"]),
            "chunk_size": args.chunk_size,
            "max_context_tokens": args.max_context_tokens,
            "include_addition": args.include_addition,
            "include_chunk_level": args.include_chunk_level,
            "attention_divergence_enabled": attention_enabled,
        },
    )


if __name__ == "__main__":
    main()
