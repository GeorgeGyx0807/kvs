#!/usr/bin/env python3
"""CLI for writing the first-round summary placeholder."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "# 第一轮阶段总结报告\n\n"
        "当前仍处于阶段 0：问题校准，尚未接入真实 LongBench/RULER 跑数。\n"
    )


if __name__ == "__main__":
    main()
