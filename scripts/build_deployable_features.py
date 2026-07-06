#!/usr/bin/env python3
"""CLI for filtering deployable features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.features import build_deployable_feature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = json.loads(args.input.read_text())
    filtered = build_deployable_feature(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(filtered, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
