#!/usr/bin/env python3
"""CLI for checking deployable feature leakage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.validation import validate_deployable_fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    record = json.loads(args.input.read_text())
    validate_deployable_fields(list(record.keys()))


if __name__ == "__main__":
    main()
