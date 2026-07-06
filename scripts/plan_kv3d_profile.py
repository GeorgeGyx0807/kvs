#!/usr/bin/env python3
"""CLI for generating 3D KV profiling plans."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d import generate_profiling_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-ids", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--include-addition", action="store_true")
    parser.add_argument("--include-chunk-level", dest="include_chunk_level", action="store_true", default=True)
    parser.add_argument("--no-include-chunk-level", dest="include_chunk_level", action="store_false")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = json.loads(args.sample_ids.read_text())
    plan = generate_profiling_plan(
        sample_ids=sample_ids,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_chunks=args.num_chunks,
        include_addition=args.include_addition,
        include_chunk_level=args.include_chunk_level,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
