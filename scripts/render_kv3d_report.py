#!/usr/bin/env python3
"""CLI for rendering a 3D KV profiling markdown report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d import render_profiling_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--main-dataset", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = json.loads(args.plan.read_text())
    records_text = args.records.read_text().strip()
    records = [json.loads(line) for line in records_text.splitlines() if line] if records_text else []
    tables = json.loads(args.tables.read_text())
    report = render_profiling_report(
        plan=plan,
        records=records,
        tables=tables,
        experiment_name=args.experiment_name,
        model_name=args.model_name,
        main_dataset=args.main_dataset,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)


if __name__ == "__main__":
    main()
