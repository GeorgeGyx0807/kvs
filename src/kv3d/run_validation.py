"""Validation helpers for completed offline 3D KV profiling run directories."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .report import build_key_findings


REQUIRED_FILES = [
    "manifest.json",
    "samples.json",
    "plan.json",
    "records.jsonl",
    "raw_results.jsonl",
    "summary.json",
    "tables.json",
    "key_findings.json",
    "layer_importance.csv",
    "head_importance.csv",
    "chunk_importance.csv",
    "block_utility.csv",
    "stability.csv",
    "stability_by_task.csv",
    "heterogeneity.csv",
    "addition_removal_alignment.csv",
    "budget_curves.csv",
    "report.md",
    "figures/fig_layer_importance.png",
    "figures/fig_layer_head_heatmap.png",
    "figures/fig_chunk_position_heatmap.png",
    "figures/fig_budget_quality_curve.png",
    "figures/fig_budget_nll_curve.png",
    "figures/fig_latency_bytes_curve.png",
    "figures/fig_stability.png",
]

REQUIRED_TABLES = [
    "layer_by_method",
    "head_by_method",
    "chunk_by_method",
    "block_utility",
    "attention_divergence",
    "attention_divergence_by_method",
    "attention_correlation_matrix",
    "stability",
    "stability_by_task",
    "heterogeneity",
    "addition_removal_alignment",
    "budget_curves",
]

CSV_TABLE_FILES = {
    "layer_by_method": "layer_importance.csv",
    "head_by_method": "head_importance.csv",
    "chunk_by_method": "chunk_importance.csv",
    "span_by_method": "span_importance.csv",
    "block_utility": "block_utility.csv",
    "attention_divergence": "attention_divergence.csv",
    "attention_divergence_by_method": "attention_divergence_by_method.csv",
    "attention_correlation_matrix": "attention_correlation_matrix.csv",
    "stability": "stability.csv",
    "stability_by_task": "stability_by_task.csv",
    "heterogeneity": "heterogeneity.csv",
    "addition_removal_alignment": "addition_removal_alignment.csv",
    "budget_curves": "budget_curves.csv",
}

REQUIRED_KEY_FINDING_SECTIONS = [
    "top_layer",
    "top_head",
    "top_chunk",
    "top_block",
    "stability",
    "task_stability",
    "heterogeneity",
    "addition_removal_alignment",
]

KEY_FINDING_ID_FIELDS = {
    "top_layer": ["method", "layer"],
    "top_head": ["method", "head"],
    "top_chunk": ["method", "chunk"],
    "top_span": ["method", "span_id", "span_start", "span_end"],
    "top_block": ["sample_id", "method", "layer", "head", "chunk", "token_start", "token_end"],
    "stability": ["method", "top_k"],
    "task_stability": ["dataset", "task_name", "method", "top_k"],
    "heterogeneity": ["method", "axis", "range_mean_utility"],
    "addition_removal_alignment": ["dataset", "task_name", "method_pair", "block_count"],
    "attention_divergence": ["method", "mean_attention_js_divergence"],
}


@dataclass(frozen=True)
class KV3DRunValidationResult:
    ok: bool
    run_dir: str
    record_count: int
    sample_count: int
    plan_size: int
    methods: dict[str, int]
    table_rows: dict[str, int]
    gates: dict[str, Any]
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path, issues: list[str]) -> Any:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        issues.append(f"missing required file: {path.name}")
    except json.JSONDecodeError as exc:
        issues.append(f"invalid JSON in {path.name}: {exc}")
    return None


def _read_jsonl(path: Path, issues: list[str]) -> list[dict[str, Any]]:
    try:
        text = path.read_text().strip()
    except FileNotFoundError:
        issues.append(f"missing required file: {path.name}")
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            issues.append(f"invalid JSONL in {path.name}:{line_no}: {exc}")
    return rows


def _csv_row_count(path: Path, issues: list[str]) -> int | None:
    try:
        with path.open(newline="") as handle:
            return sum(1 for _ in csv.DictReader(handle))
    except FileNotFoundError:
        issues.append(f"missing required file: {path.name}")
    except csv.Error as exc:
        issues.append(f"invalid CSV in {path.name}: {exc}")
    return None


def _normalize_csv_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _compare_csv_to_table(path: Path, rows: list[dict[str, Any]], issues: list[str]) -> None:
    try:
        with path.open(newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
    except FileNotFoundError:
        issues.append(f"missing required file: {path.name}")
        return
    except csv.Error as exc:
        issues.append(f"invalid CSV in {path.name}: {exc}")
        return
    if len(csv_rows) != len(rows):
        issues.append(f"CSV row count {len(csv_rows)} != tables.json rows {len(rows)}: {path.name}")
        return
    if len(csv_rows) != len(rows):
        return
    for index, (csv_row, table_row) in enumerate(zip(csv_rows, rows), start=1):
        csv_keys = set(csv_row)
        table_keys = set(table_row)
        if csv_keys != table_keys:
            issues.append(f"CSV row mismatch: {path.name} row {index}")
            return
        for key in csv_keys:
            if _normalize_csv_value(csv_row.get(key)) != _normalize_csv_value(table_row.get(key)):
                issues.append(f"CSV row mismatch: {path.name} row {index}")
                return


def _max_heterogeneity_range(rows: Any) -> float:
    if not isinstance(rows, list):
        return 0.0
    values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("range_mean_utility")
        if value is None:
            continue
        values.append(float(value))
    return max(values) if values else 0.0


def _finding_disagrees(expected: Any, actual: Any, fields: list[str]) -> bool:
    if not isinstance(expected, dict) or not expected:
        return False
    if not isinstance(actual, dict) or not actual:
        return True
    for field in fields:
        if field not in expected:
            continue
        if actual.get(field) != expected.get(field):
            return True
    return False


def _profile_identity(item: dict[str, Any]) -> tuple[str, str, Any, Any, Any]:
    key = item.get("key")
    if isinstance(key, dict):
        layer = key.get("layer")
        head = key.get("head")
        chunk = key.get("chunk")
    else:
        layer = item.get("layer")
        head = item.get("head")
        chunk = item.get("chunk")
    method = str(item.get("method"))
    if method in {"full_kv", "b_only"}:
        layer, head, chunk = None, None, None
    elif method == "remove_layer":
        head, chunk = None, None
    elif method == "remove_layer_head":
        chunk = None
    return (str(item.get("sample_id")), method, layer, head, chunk)


def _format_profile_identity(identity: tuple[str, str, Any, Any, Any]) -> str:
    sample_id, method, layer, head, chunk = identity
    return f"sample={sample_id} method={method} layer={layer} head={head} chunk={chunk}"


def _profile_identity_counts(items: list[dict[str, Any]]) -> dict[tuple[str, str, Any, Any, Any], int]:
    counts: dict[tuple[str, str, Any, Any, Any], int] = {}
    for item in items:
        identity = _profile_identity(item)
        counts[identity] = counts.get(identity, 0) + 1
    return counts


def _has_profile_key(item: dict[str, Any]) -> bool:
    key = item.get("key")
    if isinstance(key, dict):
        return True
    return item.get("layer") is not None


def _has_task_metadata(row: dict[str, Any]) -> bool:
    return row.get("dataset") not in (None, "") and row.get("task_name") not in (None, "")


def _check_metric_snapshot(payload: Any, *, label: str, method: str, issues: list[str]) -> None:
    metric = payload if isinstance(payload, dict) else {}
    if not any(metric.get(name) is not None for name in ("accuracy", "f1", "contains")):
        issues.append(f"{label} has no quality metric: {method}")
    for metric_name in ("nll", "ttft_ms", "prefill_ms", "decode_ms"):
        if metric.get(metric_name) is None:
            issues.append(f"{label} missing required metric {metric_name}: {method}")


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _check_no_nan_inf(payload: Any, *, label: str, issues: list[str]) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            _check_no_nan_inf(value, label=f"{label}.{key}", issues=issues)
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            _check_no_nan_inf(value, label=f"{label}[{index}]", issues=issues)
    elif not _is_finite_number(payload):
        issues.append(f"non-finite numeric value: {label}")


def _check_attention_divergence(payload: Any, *, method: str, issues: list[str]) -> None:
    if payload is None:
        return
    if not isinstance(payload, dict):
        issues.append(f"attention_divergence must be object: {method}")
        return
    for field in ("js_divergence", "kl_divergence"):
        value = payload.get(field)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            issues.append(f"attention divergence is not numeric: {method}.{field}")
        elif not math.isfinite(float(value)):
            issues.append(f"attention divergence is not finite: {method}.{field}")
        elif float(value) < 0:
            issues.append(f"attention divergence is negative: {method}.{field}")


def _check_full_kv_sanity(records: list[dict[str, Any]], issues: list[str]) -> None:
    for record in records:
        if record.get("method") != "full_kv":
            continue
        sample_id = str(record.get("sample_id"))
        if record.get("selected_kv_bytes") in (None, 0):
            issues.append(f"full_kv selected_kv_bytes must be positive: {sample_id}")
        metric = record.get("metric") if isinstance(record.get("metric"), dict) else {}
        if metric.get("nll") is None:
            issues.append(f"full_kv missing nll: {sample_id}")
        if not any(metric.get(name) is not None for name in ("accuracy", "f1", "contains")):
            issues.append(f"full_kv missing quality sanity metric: {sample_id}")


def validate_kv3d_run(
    run_dir: Path,
    *,
    min_samples: int = 1,
    require_stability: bool = False,
    min_heterogeneity_range: float | None = None,
) -> KV3DRunValidationResult:
    issues: list[str] = []
    for relative in REQUIRED_FILES:
        path = run_dir / relative
        if not path.exists():
            issues.append(f"missing required file: {relative}")
        elif path.is_file() and path.stat().st_size == 0 and relative.endswith((".json", ".jsonl", ".md", ".png")):
            issues.append(f"empty required file: {relative}")

    manifest = _read_json(run_dir / "manifest.json", issues) or {}
    summary = _read_json(run_dir / "summary.json", issues) or {}
    include_chunk_level = bool(summary.get("include_chunk_level", True))
    include_addition = bool(summary.get("include_addition", False))
    attention_enabled = bool(summary.get("attention_divergence_enabled", False))
    span_mode = bool(summary.get("span_size")) or summary.get("profiling_mode") == "two_stage_layer_head_then_span"

    if span_mode:
        for relative in [
            "span_importance.csv",
            "figures/fig_layer_head_span_heatmap.png",
            "figures/fig_span_position_curve.png",
            "figures/fig_top_span_stability.png",
        ]:
            path = run_dir / relative
            if not path.exists():
                issues.append(f"missing required file: {relative}")
            elif path.is_file() and path.stat().st_size == 0 and relative.endswith((".json", ".jsonl", ".md", ".png")):
                issues.append(f"empty required file: {relative}")

    if attention_enabled:
        for relative in [
            "attention_divergence.csv",
            "attention_divergence_by_method.csv",
            "figures/fig_attention_divergence.png",
        ]:
            path = run_dir / relative
            if not path.exists():
                issues.append(f"missing required file: {relative}")
            elif path.is_file() and path.stat().st_size == 0 and relative.endswith((".json", ".jsonl", ".md", ".png", ".csv")):
                issues.append(f"empty required file: {relative}")

    samples = _read_json(run_dir / "samples.json", issues) or []
    plan = _read_json(run_dir / "plan.json", issues) or []
    tables = _read_json(run_dir / "tables.json", issues) or {}
    key_findings = _read_json(run_dir / "key_findings.json", issues) or {}
    records = _read_jsonl(run_dir / "records.jsonl", issues)
    raw_results = _read_jsonl(run_dir / "raw_results.jsonl", issues)
    _check_no_nan_inf(manifest, label="manifest", issues=issues)
    _check_no_nan_inf(summary, label="summary", issues=issues)
    _check_no_nan_inf(samples, label="samples", issues=issues)
    _check_no_nan_inf(plan, label="plan", issues=issues)
    _check_no_nan_inf(tables, label="tables", issues=issues)
    _check_no_nan_inf(key_findings, label="key_findings", issues=issues)
    _check_no_nan_inf(records, label="records", issues=issues)
    _check_no_nan_inf(raw_results, label="raw_results", issues=issues)

    if manifest.get("baseline") != "full-KV":
        issues.append("manifest baseline must be full-KV")
    if manifest.get("agent_pair") != "same_checkpoint":
        issues.append("manifest agent_pair must be same_checkpoint")

    methods: dict[str, int] = {}
    for record in records:
        method = str(record.get("method", ""))
        methods[method] = methods.get(method, 0) + 1
        if method in {"remove_layer", "remove_layer_head", "remove_layer_head_chunk", "add_layer_head_chunk"}:
            key = record.get("key")
            if key is None:
                issues.append(f"block method has no key: {method}")
            elif method in {"remove_layer_head_chunk", "add_layer_head_chunk"} and isinstance(key, dict):
                if key.get("token_start") is None or key.get("token_end") is None:
                    issues.append(f"chunk-level record has no token_start/token_end: {method}")
                if span_mode and any(key.get(field) is None for field in ("span_id", "span_start", "span_end", "span_size")):
                    issues.append(f"span-level record has no span_id/span_start/span_end/span_size: {method}")
        metric = record.get("metric") or {}
        if not any(metric.get(name) is not None for name in ("accuracy", "f1", "contains", "nll", "ttft_ms")):
            issues.append(f"record has no quality/timing metric: {method}")
        _check_metric_snapshot(metric, label="record", method=method, issues=issues)
        if record.get("selected_kv_bytes") is None:
            issues.append(f"record has no selected_kv_bytes: {method}")
        if method.startswith("remove_"):
            delta_vs_full = record.get("delta_vs_full")
            if not isinstance(delta_vs_full, dict):
                issues.append(f"removal record missing delta_vs_full: {method}")
            else:
                _check_metric_snapshot(delta_vs_full, label="delta_vs_full", method=method, issues=issues)
        if method.startswith("add_"):
            delta_vs_base = record.get("delta_vs_base")
            if not isinstance(delta_vs_base, dict):
                issues.append(f"addition record missing delta_vs_base: {method}")
            else:
                _check_metric_snapshot(delta_vs_base, label="delta_vs_base", method=method, issues=issues)
        _check_attention_divergence(record.get("attention_divergence"), method=method, issues=issues)

    planned_methods: set[str] = set()
    if isinstance(plan, list):
        planned_methods = {str(item.get("method")) for item in plan if isinstance(item, dict)}
    required_methods = sorted(planned_methods) if planned_methods else ["full_kv", "b_only"]
    for baseline_method in ("full_kv", "b_only"):
        if baseline_method not in required_methods:
            required_methods.append(baseline_method)
    for method in required_methods:
        if methods.get(method, 0) == 0:
            issues.append(f"missing required method: {method}")

    if isinstance(samples, list):
        methods_by_sample: dict[str, set[str]] = {}
        for record in records:
            sample_id = str(record.get("sample_id"))
            methods_by_sample.setdefault(sample_id, set()).add(str(record.get("method", "")))
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("sample_id") is None:
                continue
            sample_id = str(sample["sample_id"])
            for baseline_method in ("full_kv", "b_only"):
                if baseline_method not in methods_by_sample.get(sample_id, set()):
                    issues.append(f"sample missing baseline method {baseline_method}: {sample_id}")
    _check_full_kv_sanity(records, issues)

    record_count = len(records)
    if summary.get("record_count") is not None and int(summary["record_count"]) != record_count:
        issues.append(f"summary record_count {summary['record_count']} != records.jsonl count {record_count}")
    if summary.get("plan_size") is not None and int(summary["plan_size"]) != len(plan):
        issues.append(f"summary plan_size {summary['plan_size']} != plan.json count {len(plan)}")
    raw_result_count = len(raw_results)
    if raw_result_count != record_count:
        issues.append(f"raw_results.jsonl count {raw_result_count} != records.jsonl count {record_count}")
    if isinstance(plan, list):
        plan_identities = {_profile_identity(item) for item in plan if isinstance(item, dict)}
        record_identity_counts = _profile_identity_counts(records)
        raw_result_identity_counts = _profile_identity_counts(raw_results)
        record_identities = set(record_identity_counts)
        raw_result_identities = set(raw_result_identity_counts)
        for identity in sorted(plan_identities, key=lambda item: tuple(str(part) for part in item)):
            if identity not in record_identities:
                issues.append(f"missing record for planned profile: {_format_profile_identity(identity)}")
        for identity in sorted(record_identities, key=lambda item: tuple(str(part) for part in item)):
            if record_identity_counts[identity] > 1:
                issues.append(f"duplicate record identity: {_format_profile_identity(identity)}")
            if identity not in plan_identities:
                issues.append(f"unexpected record not present in plan: {_format_profile_identity(identity)}")
            if identity not in raw_result_identities:
                issues.append(f"raw_results.jsonl missing record identity: {_format_profile_identity(identity)}")
        for identity in sorted(raw_result_identities, key=lambda item: tuple(str(part) for part in item)):
            if raw_result_identity_counts[identity] > 1:
                issues.append(f"raw_results.jsonl duplicate record identity: {_format_profile_identity(identity)}")
            if identity not in record_identities:
                issues.append(f"raw_results.jsonl unexpected record identity: {_format_profile_identity(identity)}")

    sample_ids = {str(sample.get("sample_id")) for sample in samples if sample.get("sample_id") is not None}
    for record in records:
        sample_id = str(record.get("sample_id"))
        if sample_ids and sample_id not in sample_ids:
            issues.append(f"record sample_id not present in samples.json: {sample_id}")

    table_rows: dict[str, int] = {}
    required_tables = list(REQUIRED_TABLES)
    if not attention_enabled:
        required_tables = [name for name in required_tables if not name.startswith("attention_divergence")]
    if span_mode:
        required_tables.append("span_by_method")
    for table_name in required_tables:
        rows = tables.get(table_name)
        if rows is None:
            issues.append(f"missing required table: {table_name}")
            table_rows[table_name] = 0
        elif not isinstance(rows, list):
            issues.append(f"table is not a list: {table_name}")
            table_rows[table_name] = 0
        else:
            table_rows[table_name] = len(rows)
            if table_name in {"layer_by_method", "head_by_method", "chunk_by_method", "block_utility", "budget_curves"} and not rows:
                issues.append(f"required table has no rows: {table_name}")
            if table_name == "attention_divergence":
                for index, row in enumerate(rows, start=1):
                    if not isinstance(row, dict):
                        continue
                    for field in ("attention_js_divergence", "attention_kl_divergence"):
                        value = row.get(field)
                        if value is not None and float(value) < 0:
                            issues.append(f"attention_divergence table has negative {field}: row {index}")
                            break
            csv_name = CSV_TABLE_FILES.get(table_name)
            if csv_name is not None:
                csv_rows = _csv_row_count(run_dir / csv_name, issues)
                if csv_rows is not None and csv_rows != len(rows):
                    issues.append(f"CSV row count {csv_rows} != tables.json {table_name} rows {len(rows)}: {csv_name}")
                elif csv_rows is not None:
                    _compare_csv_to_table(run_dir / csv_name, rows, issues)

    block_utility_rows = tables.get("block_utility") if isinstance(tables, dict) else None
    if isinstance(block_utility_rows, list):
        keyed_record_identities = {_profile_identity(record) for record in records if _has_profile_key(record)}
        block_utility_identities = {_profile_identity(row) for row in block_utility_rows if isinstance(row, dict)}
        for identity in sorted(keyed_record_identities, key=lambda item: tuple(str(part) for part in item)):
            if identity not in block_utility_identities:
                issues.append(f"block_utility missing record identity: {_format_profile_identity(identity)}")
        for identity in sorted(block_utility_identities, key=lambda item: tuple(str(part) for part in item)):
            if identity not in keyed_record_identities:
                issues.append(f"block_utility unexpected identity: {_format_profile_identity(identity)}")
        for row in block_utility_rows:
            if isinstance(row, dict) and not _has_task_metadata(row):
                issues.append(f"block_utility row missing dataset/task_name: {_format_profile_identity(_profile_identity(row))}")
                break

    for table_name in ("stability_by_task", "addition_removal_alignment"):
        rows = tables.get(table_name) if isinstance(tables, dict) else None
        if isinstance(rows, list):
            for index, row in enumerate(rows, start=1):
                if isinstance(row, dict) and not _has_task_metadata(row):
                    issues.append(f"{table_name} row missing dataset/task_name: row {index}")
                    break
    if attention_enabled:
        rows = tables.get("attention_divergence") if isinstance(tables, dict) else None
        if isinstance(rows, list):
            for index, row in enumerate(rows, start=1):
                if isinstance(row, dict) and not _has_task_metadata(row):
                    issues.append(f"attention_divergence row missing dataset/task_name: row {index}")
                    break

    if not isinstance(key_findings, dict):
        issues.append("key_findings.json must contain an object")
    else:
        required_key_finding_sections = ["top_layer", "top_head"]
        if include_chunk_level:
            required_key_finding_sections.extend(["top_chunk", "top_block", "stability", "task_stability", "heterogeneity"])
        if span_mode:
            required_key_finding_sections.append("top_span")
        if include_chunk_level and include_addition:
            required_key_finding_sections.append("addition_removal_alignment")
        if attention_enabled:
            required_key_finding_sections.append("attention_divergence")
        for section in required_key_finding_sections:
            value = key_findings.get(section)
            if not isinstance(value, dict) or not value:
                issues.append(f"key_findings missing required section: {section}")
        if isinstance(tables, dict):
            list_tables = {name: rows for name, rows in tables.items() if isinstance(rows, list)}
            expected_findings = build_key_findings(list_tables)
            finding_fields = dict(KEY_FINDING_ID_FIELDS)
            if not include_chunk_level:
                for field in ("top_chunk", "top_span", "top_block", "stability", "task_stability", "heterogeneity", "addition_removal_alignment"):
                    finding_fields.pop(field, None)
            elif not span_mode:
                finding_fields.pop("top_span", None)
            elif not include_addition:
                finding_fields.pop("addition_removal_alignment", None)
            if not attention_enabled:
                finding_fields.pop("attention_divergence", None)
            for section, fields in finding_fields.items():
                if _finding_disagrees(expected_findings.get(section), key_findings.get(section), fields):
                    issues.append(f"key_findings section disagrees with tables.json: {section}")

    sample_count = len(samples) if isinstance(samples, list) else 0
    if sample_count < min_samples:
        issues.append(f"sample_count {sample_count} < required {min_samples}")
    if require_stability and table_rows.get("stability", 0) == 0:
        issues.append("stability evidence is required but stability table has no rows")
    if min_heterogeneity_range is not None:
        max_range = _max_heterogeneity_range(tables.get("heterogeneity"))
        if max_range < min_heterogeneity_range:
            issues.append(f"max heterogeneity range {round(max_range, 6)} < required {min_heterogeneity_range}")

    return KV3DRunValidationResult(
        ok=not issues,
        run_dir=str(run_dir),
        record_count=record_count,
        sample_count=sample_count,
        plan_size=len(plan) if isinstance(plan, list) else 0,
        methods=methods,
        table_rows=table_rows,
        gates={
            "min_samples": min_samples,
            "require_stability": require_stability,
            "min_heterogeneity_range": min_heterogeneity_range,
            "include_chunk_level": include_chunk_level,
            "include_addition": include_addition,
            "attention_divergence_enabled": attention_enabled,
        },
        issues=issues,
    )
