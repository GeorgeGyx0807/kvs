#!/usr/bin/env python3
"""Validate an offline 3D KV profiling run directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d.run_validation import validate_kv3d_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-samples", type=int, default=1)
    parser.add_argument("--require-stability", action="store_true")
    parser.add_argument("--min-heterogeneity-range", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_kv3d_run(
        args.run_dir,
        min_samples=args.min_samples,
        require_stability=args.require_stability,
        min_heterogeneity_range=args.min_heterogeneity_range,
    )
    payload = result.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)
    if not result.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
