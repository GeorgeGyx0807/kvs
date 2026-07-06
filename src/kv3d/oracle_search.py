"""Bidirectional greedy oracle helpers for LongBench KV block search."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from math import ceil
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Callable, Iterable, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


@dataclass(frozen=True, order=True)
class OracleBlock:
    sample_id: str
    layer: int
    kv_head: int
    span_id: int
    span_start: int
    span_end: int

    @property
    def block_id(self) -> tuple[int, int, int]:
        return (self.layer, self.kv_head, self.span_id)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OracleEval:
    sample_id: str
    method: str
    direction: str
    stage: str
    step_index: int
    selected_blocks: tuple[OracleBlock, ...]
    prediction: str
    gold_answer: str
    answers: tuple[str, ...]
    f1: float | None
    contains: float | None
    exact: float | None
    nll: float | None
    selected_kv_bytes: int
    full_kv_bytes: int
    kv_ratio: float
    ttft_ms: float | None
    prefill_ms: float | None
    decode_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["selected_blocks"] = [block.to_dict() for block in self.selected_blocks]
        return payload


@dataclass(frozen=True)
class OracleStep:
    sample_id: str
    task_name: str
    direction: str
    stage: str
    step_index: int
    action: str
    changed_block: OracleBlock | None
    result: OracleEval

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["changed_block"] = None if self.changed_block is None else self.changed_block.to_dict()
        payload["result"] = self.result.to_dict()
        return payload


def task_family_for_longbench(task_name: str) -> str:
    lowered = task_name.lower()
    if "retrieval" in lowered:
        return "retrieval"
    return "qa"


def exact_match(prediction: str, answers: Sequence[str]) -> float:
    from .metrics import normalize_text

    normalized_prediction = normalize_text(prediction)
    return 1.0 if any(normalize_text(str(answer)) == normalized_prediction for answer in answers) else 0.0


def _none_low(value: float | None) -> float:
    return float("-inf") if value is None else float(value)


def _none_high(value: float | None) -> float:
    return float("inf") if value is None else float(value)


def quality_key(result: OracleEval, *, task_family: str) -> tuple[float, ...]:
    if task_family == "retrieval":
        primary = (_none_low(result.exact), _none_low(result.contains), _none_low(result.f1))
    else:
        primary = (_none_low(result.f1), _none_low(result.contains))
    return (
        *primary,
        -_none_high(result.nll),
        -float(result.kv_ratio),
        -float(len(result.selected_blocks)),
    )


def best_quality_eval(results: Iterable[OracleEval], *, task_family: str) -> OracleEval:
    items = list(results)
    if not items:
        raise ValueError("at least one candidate result is required")
    return max(items, key=lambda item: quality_key(item, task_family=task_family))


def prune_bottom_fraction(
    scores: dict[Any, float],
    *,
    discard_fraction: float,
) -> list[Any]:
    if not 0.0 <= discard_fraction < 1.0:
        raise ValueError("discard_fraction must be in [0, 1)")
    if not scores:
        return []
    discard_count = min(len(scores) - 1, int(len(scores) * discard_fraction))
    ordered = sorted(scores, key=lambda key: (float(scores[key]), key))
    return sorted(ordered[discard_count:])


def build_block_universe(
    *,
    sample_id: str,
    layers: Sequence[int],
    layer_heads: Sequence[tuple[int, int]],
    span_size: int,
    max_context_tokens: int,
) -> list[OracleBlock]:
    if span_size <= 0:
        raise ValueError("span_size must be positive")
    if max_context_tokens <= 0:
        raise ValueError("max_context_tokens must be positive")
    kept_layers = set(layers)
    span_count = ceil(max_context_tokens / span_size)
    blocks: list[OracleBlock] = []
    for layer, kv_head in sorted(set(layer_heads)):
        if layer not in kept_layers:
            continue
        for span_id in range(span_count):
            start = span_id * span_size
            end = min(start + span_size, max_context_tokens)
            if end <= start:
                continue
            blocks.append(
                OracleBlock(
                    sample_id=sample_id,
                    layer=layer,
                    kv_head=kv_head,
                    span_id=span_id,
                    span_start=start,
                    span_end=end,
                )
            )
    return blocks


def oracle_blocks_to_kv3d_keys(blocks: Iterable[OracleBlock]):
    from .blocks import KV3DKey

    return [
        KV3DKey(sample_id=block.sample_id, layer=block.layer, head=block.kv_head, chunk=block.span_id)
        for block in blocks
    ]


def best_forward_addition(
    *,
    current_blocks: Sequence[OracleBlock],
    candidate_blocks: Sequence[OracleBlock],
    evaluate: Callable[[list[OracleBlock]], OracleEval],
    task_family: str,
) -> tuple[OracleBlock, OracleEval]:
    current_ids = {block.block_id for block in current_blocks}
    candidates = [block for block in candidate_blocks if block.block_id not in current_ids]
    if not candidates:
        raise ValueError("no forward candidates remain")
    evaluated: list[tuple[OracleBlock, OracleEval]] = []
    for block in candidates:
        selected = sorted([*current_blocks, block], key=lambda item: item.block_id)
        evaluated.append((block, evaluate(selected)))
    return max(evaluated, key=lambda item: quality_key(item[1], task_family=task_family))


def best_backward_removal(
    *,
    current_blocks: Sequence[OracleBlock],
    evaluate: Callable[[list[OracleBlock]], OracleEval],
    task_family: str,
) -> tuple[OracleBlock, OracleEval]:
    if not current_blocks:
        raise ValueError("no backward candidates remain")
    evaluated: list[tuple[OracleBlock, OracleEval]] = []
    for block in current_blocks:
        selected = [item for item in current_blocks if item.block_id != block.block_id]
        evaluated.append((block, evaluate(sorted(selected, key=lambda item: item.block_id))))
    return max(evaluated, key=lambda item: quality_key(item[1], task_family=task_family))


def quality_scalar(result: OracleEval, *, task_family: str) -> float:
    if task_family == "retrieval":
        if result.exact is not None:
            return float(result.exact)
        if result.contains is not None:
            return float(result.contains)
        return float(result.f1 or 0.0)
    if result.f1 is not None:
        return float(result.f1)
    return float(result.contains or 0.0)


def quality_ratio(result: OracleEval, full_eval: OracleEval, *, task_family: str) -> float:
    full_quality = quality_scalar(full_eval, task_family=task_family)
    current_quality = quality_scalar(result, task_family=task_family)
    if full_quality <= 0.0:
        return 1.0 if current_quality >= full_quality else 0.0
    return current_quality / full_quality


def compute_threshold_rows(
    *,
    task_name: str,
    evals: Iterable[OracleEval],
    full_eval: OracleEval,
    thresholds: Sequence[float] = (0.80, 0.85, 0.90, 0.95),
    task_family: str | None = None,
) -> list[dict[str, Any]]:
    family = task_family or task_family_for_longbench(task_name)
    rows: list[dict[str, Any]] = []
    candidates = sorted(evals, key=lambda item: (item.kv_ratio, -quality_scalar(item, task_family=family)))
    for threshold in thresholds:
        hit = next(
            (
                item
                for item in candidates
                if quality_ratio(item, full_eval, task_family=family) + 1e-12 >= float(threshold)
            ),
            None,
        )
        rows.append(
            {
                "task_name": task_name,
                "threshold": float(threshold),
                "min_kv_ratio": None if hit is None else hit.kv_ratio,
                "method": None if hit is None else hit.method,
            }
        )
    return rows


def label_rows_from_steps(
    *,
    sample_id: str,
    task_name: str,
    universe: Sequence[OracleBlock],
    steps: Sequence[OracleStep],
) -> list[dict[str, Any]]:
    selected: dict[tuple[int, int, int], OracleStep] = {}
    for step in sorted(steps, key=lambda item: item.step_index):
        for block in step.result.selected_blocks:
            selected.setdefault(block.block_id, step)

    rows: list[dict[str, Any]] = []
    for block in sorted(universe, key=lambda item: item.block_id):
        first_step = selected.get(block.block_id)
        rows.append(
            {
                "sample_id": sample_id,
                "task_name": task_name,
                "layer": block.layer,
                "kv_head": block.kv_head,
                "span_id": block.span_id,
                "span_start": block.span_start,
                "span_end": block.span_end,
                "selected": 1 if first_step is not None else 0,
                "first_selected_step": None if first_step is None else first_step.step_index,
                "selected_direction": None if first_step is None else first_step.direction,
            }
        )
    return rows


def label_source_steps(
    *,
    steps: Sequence[OracleStep],
    full_eval_by_sample: dict[str, OracleEval],
    task_family: str,
    threshold: float = 0.95,
) -> list[OracleStep]:
    chosen: list[OracleStep] = []
    for sample_id in sorted({step.sample_id for step in steps}):
        full_eval = full_eval_by_sample.get(sample_id)
        sample_steps = [
            step
            for step in steps
            if step.sample_id == sample_id
            and step.stage == "span"
            and step.result.method in {"forward_greedy", "backward_greedy"}
        ]
        if full_eval is None or not sample_steps:
            continue
        threshold_hits = [
            step
            for step in sample_steps
            if quality_ratio(step.result, full_eval, task_family=task_family) + 1e-12 >= threshold
        ]
        if threshold_hits:
            chosen.append(
                min(
                    threshold_hits,
                    key=lambda step: (
                        step.result.kv_ratio,
                        -quality_scalar(step.result, task_family=task_family),
                        step.step_index,
                    ),
                )
            )
        else:
            chosen.append(max(sample_steps, key=lambda step: quality_key(step.result, task_family=task_family)))
    return chosen


def frequency_rows(label_rows: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows = list(label_rows)

    def grouped(keys: tuple[str, ...]) -> list[dict[str, Any]]:
        buckets: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in rows:
            group_key = tuple(row[key] for key in keys)
            buckets.setdefault(group_key, []).append(row)
        output: list[dict[str, Any]] = []
        for group_key, bucket in sorted(buckets.items()):
            selected_values = [float(row["selected"]) for row in bucket]
            output.append(
                {
                    **dict(zip(keys, group_key)),
                    "block_count": len(bucket),
                    "selected_count": int(sum(selected_values)),
                    "selection_rate": fmean(selected_values) if selected_values else 0.0,
                }
            )
        return output

    return {
        "layer": grouped(("task_name", "layer")),
        "head": grouped(("task_name", "layer", "kv_head")),
        "span": grouped(("task_name", "span_id")),
        "block": grouped(("task_name", "layer", "kv_head", "span_id")),
    }


def _write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(rows)
    path.write_text("\n".join(json.dumps(item, sort_keys=True) for item in items) + ("\n" if items else ""))


def _write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    items = list(rows)
    if not items:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in items:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None]
    if not kept:
        return None
    return round(fmean(kept), 6)


def aggregate_oracle_evals(evals: Iterable[OracleEval]) -> list[dict[str, Any]]:
    buckets: dict[str, list[OracleEval]] = {}
    for item in evals:
        buckets.setdefault(item.method, []).append(item)
    rows: list[dict[str, Any]] = []
    for method, bucket in sorted(buckets.items()):
        rows.append(
            {
                "method": method,
                "sample_count": len(bucket),
                "mean_f1": _mean(item.f1 for item in bucket),
                "mean_contains": _mean(item.contains for item in bucket),
                "mean_exact": _mean(item.exact for item in bucket),
                "mean_nll": _mean(item.nll for item in bucket),
                "mean_kv_ratio": _mean(item.kv_ratio for item in bucket),
                "mean_selected_kv_bytes": _mean(float(item.selected_kv_bytes) for item in bucket),
                "mean_ttft_ms": _mean(item.ttft_ms for item in bucket),
                "mean_selected_block_count": _mean(float(len(item.selected_blocks)) for item in bucket),
            }
        )
    return rows


def oracle_curve_rows(evals: Iterable[OracleEval], *, task_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evals:
        rows.append(
            {
                "sample_id": item.sample_id,
                "task_name": task_name,
                "method": item.method,
                "direction": item.direction,
                "stage": item.stage,
                "step_index": item.step_index,
                "f1": item.f1,
                "contains": item.contains,
                "exact": item.exact,
                "nll": item.nll,
                "selected_kv_bytes": item.selected_kv_bytes,
                "full_kv_bytes": item.full_kv_bytes,
                "kv_ratio": item.kv_ratio,
                "ttft_ms": item.ttft_ms,
                "selected_block_count": len(item.selected_blocks),
            }
        )
    return rows


def _trajectory_rows(steps: Sequence[OracleStep]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in steps:
        payload = step.to_dict()
        payload["selected_block_count"] = len(step.result.selected_blocks)
        payload["selected_blocks"] = [block.to_dict() for block in step.result.selected_blocks]
        rows.append(payload)
    return rows


def _feature_label_rows(label_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in label_rows:
        span_start = int(row["span_start"])
        span_end = int(row["span_end"])
        rows.append(
            {
                **row,
                "span_size": span_end - span_start,
                "span_midpoint": (span_start + span_end) / 2.0,
            }
        )
    return rows


def _render_heatmap(rows: list[dict[str, Any]], *, x_key: str, y_key: str, value_key: str, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_bytes(b"")
        return
    xs = sorted({row[x_key] for row in rows})
    ys = sorted({row[y_key] for row in rows})
    x_index = {value: idx for idx, value in enumerate(xs)}
    y_index = {value: idx for idx, value in enumerate(ys)}
    matrix = [[0.0 for _ in xs] for _ in ys]
    for row in rows:
        matrix[y_index[row[y_key]]][x_index[row[x_key]]] = float(row.get(value_key) or 0.0)
    plt.figure(figsize=(max(6, len(xs) * 0.4), max(3, len(ys) * 0.3)))
    plt.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(label=value_key)
    plt.xticks(range(len(xs)), xs, rotation=90)
    plt.yticks(range(len(ys)), ys)
    plt.xlabel(x_key)
    plt.ylabel(y_key)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def render_oracle_report(
    *,
    task_name: str,
    threshold_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    frequency_tables: dict[str, list[dict[str, Any]]],
    config: dict[str, Any],
) -> str:
    full_row = next((row for row in baseline_rows if row["method"] == "full_kv"), None)
    base_row = next((row for row in baseline_rows if row["method"] == "b_only"), None)
    threshold_text = ", ".join(
        f"{row['threshold']}: {row['min_kv_ratio'] if row['min_kv_ratio'] is not None else 'not reached'}"
        for row in threshold_rows
    )
    top_layers = sorted(frequency_tables.get("layer", []), key=lambda row: float(row.get("selection_rate") or 0.0), reverse=True)[
        :5
    ]
    top_layer_text = ", ".join(f"L{row.get('layer')}={row.get('selection_rate'):.3g}" for row in top_layers) or "NA"
    lines = [
        f"# Bidirectional Greedy Oracle Report: {task_name}",
        "",
        "## Setup",
        "",
        f"- Max context tokens: {config.get('max_context_tokens', 'NA')}",
        f"- Span size: {config.get('span_size', 'NA')}",
        f"- Model: {config.get('model_name', 'NA')}",
        "- Quality rule: task metric first; NLL is only a tie-break.",
        "",
        "## Do small KV subsets approach full KV?",
        "",
        f"- Threshold minimum KV ratios: {threshold_text or 'NA'}",
        f"- Full KV mean F1: {None if full_row is None else full_row.get('mean_f1')}",
        f"- b_only mean F1: {None if base_row is None else base_row.get('mean_f1')}",
        "",
        "## Which tasks depend more on KV?",
        "",
        "- Compare the full_kv to b_only gap and the threshold rows across task reports; larger gaps and higher threshold ratios indicate stronger KV dependence.",
        "",
        "## Do oracle-selected layer/head/span patterns differ by task?",
        "",
        f"- Top selected layers for this task: {top_layer_text}",
        "- Use selection_frequency_layer.csv, selection_frequency_head.csv, selection_frequency_span.csv, and heatmaps for cross-task comparison.",
        "",
        "## Are these labels suitable for selector training?",
        "",
        "- Labels are generated from real selected-KV reruns and keep per-block selection, first selected step, direction, span coordinates, quality, KV ratio, and timing artifacts.",
        "- They are suitable as oracle upper-bound labels if sanity checks show full_kv is valid and greedy trajectories beat random/uniform baselines at comparable KV ratios.",
        "",
    ]
    return "\n".join(lines)


def write_oracle_artifacts(
    *,
    output_dir: Path,
    task_name: str,
    baselines: Sequence[OracleEval],
    steps: Sequence[OracleStep],
    universe: Sequence[OracleBlock],
    full_eval_by_sample: dict[str, OracleEval],
    config: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    step_evals = [step.result for step in steps]
    all_evals = [*baselines, *step_evals]
    _write_jsonl((item.to_dict() for item in baselines), output_dir / "baselines.jsonl")
    _write_jsonl((item.to_dict() for item in step_evals), output_dir / "oracle_step_results.jsonl")
    _write_jsonl(_trajectory_rows(steps), output_dir / "oracle_trajectories.jsonl")
    baseline_rows = aggregate_oracle_evals(baselines)
    _write_csv(baseline_rows, output_dir / "baseline_summary.csv")
    curve_rows = oracle_curve_rows(all_evals, task_name=task_name)
    _write_csv(curve_rows, output_dir / "oracle_quality_curves.csv")

    threshold_rows: list[dict[str, Any]] = []
    for sample_id, full_eval in full_eval_by_sample.items():
        sample_evals = [item for item in step_evals if item.sample_id == sample_id]
        threshold_rows.extend(
            compute_threshold_rows(
                task_name=task_name,
                evals=sample_evals,
                full_eval=full_eval,
                thresholds=(0.80, 0.85, 0.90, 0.95),
                task_family=task_family_for_longbench(task_name),
            )
        )
    _write_csv(threshold_rows, output_dir / "oracle_thresholds.csv")

    label_rows: list[dict[str, Any]] = []
    label_steps = label_source_steps(
        steps=steps,
        full_eval_by_sample=full_eval_by_sample,
        task_family=task_family_for_longbench(task_name),
        threshold=0.95,
    )
    for sample_id in sorted({block.sample_id for block in universe}):
        sample_universe = [block for block in universe if block.sample_id == sample_id]
        sample_steps = [step for step in label_steps if step.sample_id == sample_id]
        label_rows.extend(label_rows_from_steps(sample_id=sample_id, task_name=task_name, universe=sample_universe, steps=sample_steps))
    _write_csv(label_rows, output_dir / "oracle_labels.csv")
    _write_csv(_feature_label_rows(label_rows), output_dir / "block_features_labels.csv")

    frequencies = frequency_rows(label_rows)
    _write_csv(frequencies["layer"], output_dir / "selection_frequency_layer.csv")
    _write_csv(frequencies["head"], output_dir / "selection_frequency_head.csv")
    _write_csv(frequencies["span"], output_dir / "selection_frequency_span.csv")
    _write_csv(frequencies["block"], output_dir / "selection_frequency_block.csv")
    _render_heatmap(
        frequencies["layer"],
        x_key="layer",
        y_key="task_name",
        value_key="selection_rate",
        output_path=output_dir / "figures" / "layer_selection_heatmap.png",
        title="Layer selection rate",
    )
    _render_heatmap(
        frequencies["head"],
        x_key="kv_head",
        y_key="layer",
        value_key="selection_rate",
        output_path=output_dir / "figures" / "head_selection_heatmap.png",
        title="Layer-head selection rate",
    )
    _render_heatmap(
        frequencies["span"],
        x_key="span_id",
        y_key="task_name",
        value_key="selection_rate",
        output_path=output_dir / "figures" / "span_selection_heatmap.png",
        title="Span selection rate",
    )
    report = render_oracle_report(
        task_name=task_name,
        threshold_rows=threshold_rows,
        baseline_rows=baseline_rows,
        frequency_tables=frequencies,
        config=config,
    )
    (output_dir / "report.md").write_text(report)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str))
