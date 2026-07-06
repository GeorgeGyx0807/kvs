#!/usr/bin/env python3
"""CLI for selector evaluation on tiny JSON inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.selectors import select_by_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--kv-bytes", type=Path, required=True)
    parser.add_argument("--budget-bytes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = json.loads(args.scores.read_text())
    kv_bytes = json.loads(args.kv_bytes.read_text())
    chosen = select_by_score(scores=scores, kv_bytes=kv_bytes, budget_bytes=args.budget_bytes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"selected": chosen}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
