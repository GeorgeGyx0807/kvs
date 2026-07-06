#!/usr/bin/env python3
"""CLI for aggregating 3D KV profiling records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d.analysis import aggregate_profiling_records
from src.kv3d.io import read_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    text = args.input.read_text().strip()
    if not text:
        payload = []
    elif text.startswith("["):
        payload = json.loads(text)
    else:
        payload = read_records(args.input)
    tables = aggregate_profiling_records(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(tables, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
