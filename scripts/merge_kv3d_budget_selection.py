#!/usr/bin/env python3
"""Merge budgeted KV selection shards and regenerate derived artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d.budget_selection import BudgetEvaluationResult
from src.kv3d.budget_selection import aggregate_budget_results
from src.kv3d.budget_selection import quality_curve_rows
from src.kv3d.budget_selection import render_budget_figures
from src.kv3d.budget_selection import render_budget_report
from src.kv3d.budget_selection import write_csv_rows
from src.kv3d.budget_selection import write_jsonl_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_results(path: Path) -> list[BudgetEvaluationResult]:
    if not path.exists():
        raise FileNotFoundError(f"missing shard result file: {path}")
    allowed_fields = {field.name for field in fields(BudgetEvaluationResult)}
    results: list[BudgetEvaluationResult] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            filtered = {key: value for key, value in row.items() if key in allowed_fields}
            try:
                results.append(BudgetEvaluationResult(**filtered))
            except TypeError as exc:
                raise ValueError(f"invalid row in {path}:{line_number}: {exc}") from exc
    return results


def main() -> None:
    args = parse_args()
    results: list[BudgetEvaluationResult] = []
    for shard_dir in args.shard_dirs:
        results.extend(_load_results(shard_dir / "budget_selection_results.jsonl"))
    if not results:
        raise SystemExit("no budget selection rows found in shard results")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate_budget_results(results)
    curve_rows = quality_curve_rows(summary_rows)
    write_jsonl_results(results, args.output_dir / "budget_selection_results.jsonl")
    write_csv_rows(summary_rows, args.output_dir / "budget_selection_summary.csv")
    write_csv_rows(curve_rows, args.output_dir / "budget_quality_curves.csv")
    render_budget_figures(summary_rows, args.output_dir / "figures")
    render_budget_report(summary_rows, args.output_dir / "report_budget_selection.md")
    (args.output_dir / "merge_config.json").write_text(
        json.dumps(
            {
                "shard_dirs": [str(path) for path in args.shard_dirs],
                "output_dir": str(args.output_dir),
                "row_count": len(results),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
