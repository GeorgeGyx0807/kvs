#!/usr/bin/env python3
"""CLI for writing a minimal oracle label record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.oracle import OracleLabel, score_oracle_label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--decode-attention-mass", type=float, required=True)
    parser.add_argument("--kv-bytes", type=float, required=True)
    parser.add_argument("--ttft-penalty", type=float, required=True)
    parser.add_argument("--lambda-bytes", type=float, default=1.0)
    parser.add_argument("--lambda-ttft", type=float, default=1.0)
    parser.add_argument("--oracle-rank", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    utility = score_oracle_label(
        decode_attention_mass=args.decode_attention_mass,
        kv_bytes=args.kv_bytes,
        ttft_penalty=args.ttft_penalty,
        lambda_bytes=args.lambda_bytes,
        lambda_ttft=args.lambda_ttft,
    )
    label = OracleLabel(
        sample_id=args.sample_id,
        block_id=args.block_id,
        decode_attention_mass=args.decode_attention_mass,
        oracle_utility=utility,
        oracle_rank=args.oracle_rank,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(label.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
