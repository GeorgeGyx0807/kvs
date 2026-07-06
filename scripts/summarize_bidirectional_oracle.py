#!/usr/bin/env python3
"""Summarize multi-task bidirectional oracle outputs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any


TABLES = {
    "oracle_thresholds.csv": "oracle_thresholds_all.csv",
    "baseline_summary.csv": "baseline_summary_all.csv",
    "oracle_quality_curves.csv": "oracle_quality_curves_all.csv",
    "oracle_labels.csv": "oracle_labels_all.csv",
    "block_features_labels.csv": "block_features_labels_all.csv",
    "selection_frequency_layer.csv": "selection_frequency_layer_all.csv",
    "selection_frequency_head.csv": "selection_frequency_head_all.csv",
    "selection_frequency_span.csv": "selection_frequency_span_all.csv",
    "selection_frequency_block.csv": "selection_frequency_block_all.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _mean(values: list[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    if not kept:
        return None
    return round(fmean(kept), 6)


def task_dirs(input_dir: Path) -> list[Path]:
    summary_dir = input_dir / "summary"
    dirs: list[Path] = []
    for path in sorted(input_dir.rglob("baseline_summary.csv")):
        if summary_dir in path.parents:
            continue
        dirs.append(path.parent)
    return dirs


def merge_tables(input_dir: Path, output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {output_name: [] for output_name in TABLES.values()}
    for task_dir in task_dirs(input_dir):
        task_name = task_dir.name
        for input_name, output_name in TABLES.items():
            rows = _read_csv(task_dir / input_name)
            for row in rows:
                row = dict(row)
                row.setdefault("task_name", task_name)
                if input_name == "baseline_summary.csv":
                    row["task_name"] = task_name
                merged[output_name].append(row)
    for output_name, rows in merged.items():
        _write_csv(rows, output_dir / output_name)
    return merged


def task_dependency_rows(baseline_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in baseline_rows:
        by_task[str(row["task_name"])][str(row["method"])] = row
    rows: list[dict[str, Any]] = []
    for task_name, methods in sorted(by_task.items()):
        full = methods.get("full_kv", {})
        base = methods.get("b_only", {})
        full_f1 = _float(full.get("mean_f1"))
        base_f1 = _float(base.get("mean_f1"))
        full_contains = _float(full.get("mean_contains"))
        base_contains = _float(base.get("mean_contains"))
        rows.append(
            {
                "task_name": task_name,
                "full_f1": full_f1,
                "b_only_f1": base_f1,
                "f1_gap_full_minus_b_only": None if full_f1 is None or base_f1 is None else round(full_f1 - base_f1, 6),
                "full_contains": full_contains,
                "b_only_contains": base_contains,
                "contains_gap_full_minus_b_only": None
                if full_contains is None or base_contains is None
                else round(full_contains - base_contains, 6),
            }
        )
    return rows


def threshold_summary_rows(threshold_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[float | None]] = defaultdict(list)
    for row in threshold_rows:
        buckets[(str(row["task_name"]), str(row["threshold"]))].append(_float(row.get("min_kv_ratio")))
    rows: list[dict[str, Any]] = []
    for (task_name, threshold), values in sorted(buckets.items()):
        rows.append(
            {
                "task_name": task_name,
                "threshold": threshold,
                "mean_min_kv_ratio": _mean(values),
                "sample_count": len(values),
                "reached_count": sum(value is not None for value in values),
            }
        )
    return rows


def render_report(
    *,
    dependency_rows: list[dict[str, Any]],
    threshold_rows: list[dict[str, Any]],
    label_rows: list[dict[str, Any]],
    frequency_layer_rows: list[dict[str, Any]],
) -> str:
    sorted_dependency = sorted(
        dependency_rows,
        key=lambda row: 0.0 if row["f1_gap_full_minus_b_only"] is None else float(row["f1_gap_full_minus_b_only"]),
        reverse=True,
    )
    lines = [
        "# LongBench Bidirectional Greedy Oracle Summary",
        "",
        "## Do small KV subsets approach full KV?",
        "",
    ]
    for row in threshold_rows:
        if str(row["threshold"]) in {"0.8", "0.80", "0.85", "0.9", "0.90", "0.95"}:
            lines.append(
                f"- {row['task_name']} threshold {row['threshold']}: mean min KV ratio={row['mean_min_kv_ratio']} "
                f"({row['reached_count']}/{row['sample_count']} reached)."
            )
    lines.extend(["", "## Which tasks depend more on KV?", ""])
    for row in sorted_dependency:
        lines.append(
            f"- {row['task_name']}: full-b_only F1 gap={row['f1_gap_full_minus_b_only']}, "
            f"contains gap={row['contains_gap_full_minus_b_only']}."
        )
    lines.extend(["", "## Do oracle-selected layer/head/span patterns differ by task?", ""])
    by_task_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frequency_layer_rows:
        by_task_layer[str(row["task_name"])].append(row)
    for task_name, rows in sorted(by_task_layer.items()):
        top = sorted(rows, key=lambda row: float(row.get("selection_rate") or 0.0), reverse=True)[:5]
        top_text = ", ".join(f"L{row.get('layer')}={float(row.get('selection_rate') or 0.0):.3g}" for row in top)
        lines.append(f"- {task_name}: top layers {top_text or 'NA'}.")
    lines.extend(["", "## Are these labels suitable for selector training?", ""])
    selected_count = sum(1 for row in label_rows if str(row.get("selected")) in {"1", "1.0", "True", "true"})
    lines.append(f"- Label rows: {len(label_rows)} total, {selected_count} selected positives.")
    lines.append("- Positive labels come from per-sample oracle trajectories, while block_features_labels_all.csv keeps span coordinates and labels for selector training.")
    lines.append("- Check full_kv quality and threshold_summary.csv before treating a task's labels as reliable training supervision.")
    lines.extend(["", "## Output Tables", ""])
    for output_name in sorted(TABLES.values()):
        lines.append(f"- {output_name}")
    lines.extend(["- task_dependency.csv", "- threshold_summary.csv", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    merged = merge_tables(args.input_dir, args.output_dir)
    dependency = task_dependency_rows(merged["baseline_summary_all.csv"])
    threshold_summary = threshold_summary_rows(merged["oracle_thresholds_all.csv"])
    _write_csv(dependency, args.output_dir / "task_dependency.csv")
    _write_csv(threshold_summary, args.output_dir / "threshold_summary.csv")
    report = render_report(
        dependency_rows=dependency,
        threshold_rows=threshold_summary,
        label_rows=merged["oracle_labels_all.csv"],
        frequency_layer_rows=merged["selection_frequency_layer_all.csv"],
    )
    (args.output_dir / "report.md").write_text(report)


if __name__ == "__main__":
    main()
