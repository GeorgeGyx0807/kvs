#!/usr/bin/env python3
"""CLI for writing a minimal block feature record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.features import BlockFeature


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--position-start", type=int, required=True)
    parser.add_argument("--position-end", type=int, required=True)
    parser.add_argument("--kv-bytes", type=int, required=True)
    parser.add_argument("--prefill-attention-mass", type=float, required=True)
    parser.add_argument("--prefill-entropy", type=float, required=True)
    parser.add_argument("--hidden-norm", type=float, required=True)
    parser.add_argument("--similarity", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = BlockFeature(
        sample_id=args.sample_id,
        block_id=args.block_id,
        layer=args.layer,
        head=args.head,
        position_start=args.position_start,
        position_end=args.position_end,
        kv_bytes=args.kv_bytes,
        prefill_attention_mass=args.prefill_attention_mass,
        prefill_entropy=args.prefill_entropy,
        hidden_norm=args.hidden_norm,
        similarity=args.similarity,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
