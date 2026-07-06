#!/usr/bin/env python3
"""CLI for Pareto frontier computation from JSON points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.frontier import FrontierPoint, pareto_frontier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    points = [FrontierPoint(**item) for item in payload]
    frontier = pareto_frontier(points)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([point.__dict__ for point in frontier], indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
