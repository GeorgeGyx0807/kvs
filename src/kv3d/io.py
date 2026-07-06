"""Persistence helpers for offline 3D KV profiling."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from typing import Any

from .figures import render_profiling_figures
from .records import KV3DProfilingRecord, ProfilingManifest
from .report import build_key_findings


def write_manifest(manifest: ProfilingManifest, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


def write_records(records: Iterable[KV3DProfilingRecord | dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict() if hasattr(record, "to_dict") else record, sort_keys=True) for record in records]
    output.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_records(input_path: Path) -> list[dict[str, object]]:
    text = input_path.read_text().strip()
    if not text:
        return []
    return [json.loads(line) for line in text.splitlines() if line]


def write_summary(summary: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True))


def write_csv_table(rows: Iterable[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        output.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_profile_artifacts(
    *,
    result: dict[str, Any],
    output_dir: Path,
    summary: dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(manifest=ProfilingManifest(**result["manifest"]), output=output_dir / "manifest.json")
    (output_dir / "samples.json").write_text(json.dumps(result["samples"], indent=2, sort_keys=True))
    (output_dir / "plan.json").write_text(json.dumps(result["plan"], indent=2, sort_keys=True))
    write_records(result["records"], output_dir / "records.jsonl")
    write_records(result["records"], output_dir / "raw_results.jsonl")
    write_summary(summary, output_dir / "summary.json")
    (output_dir / "tables.json").write_text(json.dumps(result["tables"], indent=2, sort_keys=True))
    (output_dir / "key_findings.json").write_text(
        json.dumps(build_key_findings(result["tables"]), indent=2, sort_keys=True)
    )
    write_csv_table(result["tables"]["layer_by_method"], output_dir / "layer_importance.csv")
    write_csv_table(result["tables"]["head_by_method"], output_dir / "head_importance.csv")
    write_csv_table(result["tables"]["chunk_by_method"], output_dir / "chunk_importance.csv")
    write_csv_table(result["tables"].get("span_by_method", []), output_dir / "span_importance.csv")
    write_csv_table(result["tables"]["block_utility"], output_dir / "block_utility.csv")
    write_csv_table(result["tables"]["attention_divergence"], output_dir / "attention_divergence.csv")
    write_csv_table(
        result["tables"]["attention_divergence_by_method"],
        output_dir / "attention_divergence_by_method.csv",
    )
    write_csv_table(result["tables"]["attention_correlation_matrix"], output_dir / "attention_correlation_matrix.csv")
    write_csv_table(result["tables"]["stability"], output_dir / "stability.csv")
    write_csv_table(result["tables"]["stability_by_task"], output_dir / "stability_by_task.csv")
    write_csv_table(result["tables"]["heterogeneity"], output_dir / "heterogeneity.csv")
    write_csv_table(result["tables"]["addition_removal_alignment"], output_dir / "addition_removal_alignment.csv")
    write_csv_table(result["tables"]["budget_curves"], output_dir / "budget_curves.csv")
    render_profiling_figures(result["tables"], output_dir / "figures")
    (output_dir / "report.md").write_text(result["report"])
