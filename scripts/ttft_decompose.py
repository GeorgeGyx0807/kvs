#!/usr/bin/env python3
"""CLI for TTFT decomposition on synthetic or real inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ttft import decompose_ttft


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-bytes", type=float, required=True)
    parser.add_argument("--bandwidth", type=float, required=True)
    parser.add_argument("--packaging-time", type=float, required=True)
    parser.add_argument("--receiving-time", type=float, required=True)
    parser.add_argument("--first-token-compute-time", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    breakdown = decompose_ttft(
        kv_bytes=args.kv_bytes,
        bandwidth_bytes_per_sec=args.bandwidth,
        packaging_time=args.packaging_time,
        receiving_time=args.receiving_time,
        first_token_compute_time=args.first_token_compute_time,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(breakdown.__dict__, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
