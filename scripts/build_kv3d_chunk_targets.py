#!/usr/bin/env python3
"""Build stage-2 chunk targets from a stage-1 profiling run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d.io import read_records
from src.kv3d.stage_selection import build_chunk_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--top-k-per-sample", type=int, default=4)
    parser.add_argument("--middle-k-per-sample", type=int, default=2)
    parser.add_argument("--low-k-per-sample", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_records(args.run_dir / "records.jsonl")
    chunk_targets = build_chunk_targets(
        records,
        top_k_per_sample=args.top_k_per_sample,
        middle_k_per_sample=args.middle_k_per_sample,
        low_k_per_sample=args.low_k_per_sample,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(chunk_targets, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
