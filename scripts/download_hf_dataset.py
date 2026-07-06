#!/usr/bin/env python3
"""Download a Hugging Face dataset split and export profiling samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d import export_samples_json, load_hf_dataset_split, row_to_sample_for_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--config-name", default=None)
    parser.add_argument("--split", required=True)
    parser.add_argument("--spec", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = json.loads(args.spec.read_text()) if args.spec else {}
    dataset = load_hf_dataset_split(args.dataset_name, args.split, config_name=args.config_name)
    samples = [row_to_sample_for_dataset(args.dataset_name, dict(row), spec=spec) for row in dataset]
    export_samples_json(samples, args.output)


if __name__ == "__main__":
    main()
