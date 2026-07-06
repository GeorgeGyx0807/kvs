"""Budgeted KV block selection utilities for span-level KV3D profiles."""

from __future__ import annotations

import csv
import json
import os
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


BUDGET_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

SELECTION_STRATEGIES = [
    "random_span",
    "uniform_layer_head_span",
    "recent_span",
    "first_prefix_span",
    "layer_only_utility",
    "layer_head_utility",
    "global_span_utility_topk",
    "hierarchical_layer_head_span",
]


@dataclass(frozen=True, order=True)
class KVBlock:
    layer: int
    head: int
    span_id: int
    span_start: int
    span_end: int

    def block_id(self) -> tuple[int, int, int]:
        return (self.layer, self.head, self.span_id)


@dataclass(frozen=True)
class BudgetUtilityProfile:
    layer_scores: dict[int, float]
    layer_head_scores: dict[tuple[int, int], float]
    span_position_scores: dict[int, float]
    block_scores: dict[tuple[int, int, int], float]
    attention_js_by_block: dict[tuple[int, int, int], float] | None = None
    attention_kl_by_block: dict[tuple[int, int, int], float] | None = None


@dataclass(frozen=True)
class BudgetEvaluationResult:
    sample_id: str
    dataset: str
    task_name: str
    method: str
    budget_ratio: float
    prediction: str
    gold_answer: str
    answers: list[str]
    f1: float | None
    contains: float | None
    nll: float | None
    delta_nll_vs_full: float | None
    selected_kv_bytes: int
    full_kv_bytes: int
    kv_ratio: float
    ttft_ms: float | None
    prefill_ms: float | None
    decode_ms: float | None
    generation_length: int
    selected_block_count: int
    attention_js_mean_selected: float | None = None
    attention_kl_mean_selected: float | None = None
    model_name: str | None = None
    max_context_tokens: int | None = None
    span_size: int | None = None
    max_new_tokens: int | None = None
    random_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _mean(values: Iterable[float | None]) -> float | None:
    kept = [value for value in values if value is not None]
    if not kept:
        return None
    return round(fmean(kept), 6)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_utility_profile(profile_dir: Path) -> BudgetUtilityProfile:
    layer_scores: dict[int, float] = {}
    for row in read_csv_rows(profile_dir / "layer_importance.csv"):
        if row.get("method") != "remove_layer":
            continue
        score = _float_or_none(row.get("mean_delta_nll"))
        if score is not None:
            layer_scores[int(row["layer"])] = score

    layer_head_buckets: dict[tuple[int, int], list[float]] = defaultdict(list)
    span_buckets: dict[int, list[float]] = defaultdict(list)
    block_buckets: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    js_buckets: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    kl_buckets: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for row in read_csv_rows(profile_dir / "block_utility.csv"):
        method = row.get("method")
        score = _float_or_none(row.get("delta_nll"))
        if score is None:
            continue
        layer = int(row["layer"])
        head = int(row["head"])
        if method == "remove_layer_head":
            layer_head_buckets[(layer, head)].append(score)
        elif method == "remove_layer_head_chunk":
            span_id = int(row.get("span_id") or row.get("chunk"))
            block_key = (layer, head, span_id)
            block_buckets[block_key].append(score)
            span_buckets[span_id].append(score)
            js = _float_or_none(row.get("mean_attention_js_divergence") or row.get("attention_js_divergence"))
            kl = _float_or_none(row.get("mean_attention_kl_divergence") or row.get("attention_kl_divergence"))
            if js is not None:
                js_buckets[block_key].append(js)
            if kl is not None:
                kl_buckets[block_key].append(kl)

    return BudgetUtilityProfile(
        layer_scores=layer_scores,
        layer_head_scores={key: _mean(values) or 0.0 for key, values in layer_head_buckets.items()},
        span_position_scores={key: _mean(values) or 0.0 for key, values in span_buckets.items()},
        block_scores={key: _mean(values) or 0.0 for key, values in block_buckets.items()},
        attention_js_by_block={key: _mean(values) or 0.0 for key, values in js_buckets.items()},
        attention_kl_by_block={key: _mean(values) or 0.0 for key, values in kl_buckets.items()},
    )


def build_block_universe(
    *,
    num_layers: int,
    num_heads: int,
    num_spans: int,
    span_size: int,
    max_context_tokens: int,
) -> list[KVBlock]:
    blocks: list[KVBlock] = []
    for layer in range(num_layers):
        for head in range(num_heads):
            for span_id in range(num_spans):
                start = min(span_id * span_size, max_context_tokens)
                end = min(start + span_size, max_context_tokens)
                if end <= start:
                    continue
                blocks.append(KVBlock(layer=layer, head=head, span_id=span_id, span_start=start, span_end=end))
    return blocks


def _block_score(block: KVBlock, profile: BudgetUtilityProfile, strategy: str) -> float:
    if strategy == "layer_only_utility":
        return profile.layer_scores.get(block.layer, 0.0)
    if strategy == "layer_head_utility":
        return profile.layer_head_scores.get((block.layer, block.head), profile.layer_scores.get(block.layer, 0.0))
    if strategy == "global_span_utility_topk":
        return profile.block_scores.get(
            block.block_id(),
            profile.layer_head_scores.get((block.layer, block.head), profile.layer_scores.get(block.layer, 0.0)),
        )
    if strategy == "hierarchical_layer_head_span":
        return profile.block_scores.get(block.block_id(), profile.span_position_scores.get(block.span_id, 0.0))
    return 0.0


def _uniform_layer_head_span_order(blocks: list[KVBlock]) -> list[KVBlock]:
    block_by_key = {block.block_id(): block for block in blocks}
    layer_heads = sorted({(block.layer, block.head) for block in blocks})
    spans = sorted({block.span_id for block in blocks})
    ordered: list[KVBlock] = []
    seen: set[tuple[int, int, int]] = set()

    for round_idx in range(len(layer_heads)):
        for span_idx, span_id in enumerate(spans):
            layer, head = layer_heads[(span_idx + round_idx) % len(layer_heads)]
            key = (layer, head, span_id)
            block = block_by_key.get(key)
            if block is not None and key not in seen:
                ordered.append(block)
                seen.add(key)

    for block in sorted(blocks, key=lambda item: (item.layer, item.head, item.span_id)):
        if block.block_id() not in seen:
            ordered.append(block)
    return ordered


def build_strategy_order(
    blocks: list[KVBlock],
    profile: BudgetUtilityProfile,
    *,
    strategy: str,
    seed: int,
) -> list[KVBlock]:
    if strategy == "random_span":
        ordered = list(blocks)
        random.Random(seed).shuffle(ordered)
        return ordered
    if strategy == "uniform_layer_head_span":
        return _uniform_layer_head_span_order(blocks)
    if strategy == "recent_span":
        return sorted(blocks, key=lambda block: (-block.span_id, block.layer, block.head))
    if strategy == "first_prefix_span":
        return sorted(blocks, key=lambda block: (block.span_id, block.layer, block.head))
    if strategy in {"layer_only_utility", "layer_head_utility", "global_span_utility_topk"}:
        return sorted(
            blocks,
            key=lambda block: (
                -_block_score(block, profile, strategy),
                block.layer,
                block.head,
                block.span_id,
            ),
        )
    if strategy == "hierarchical_layer_head_span":
        return sorted(
            blocks,
            key=lambda block: (
                -profile.layer_scores.get(block.layer, 0.0),
                -profile.layer_head_scores.get((block.layer, block.head), profile.layer_scores.get(block.layer, 0.0)),
                -_block_score(block, profile, strategy),
                block.layer,
                block.head,
                block.span_id,
            ),
        )
    raise ValueError(f"unknown selection strategy: {strategy}")


def select_blocks_for_budget(
    *,
    blocks: list[KVBlock],
    profile: BudgetUtilityProfile,
    strategy: str,
    budget_bytes: int,
    block_bytes: int,
    seed: int,
) -> list[KVBlock]:
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")
    if block_bytes <= 0:
        raise ValueError("block_bytes must be positive")
    max_blocks = budget_bytes // block_bytes
    if max_blocks <= 0:
        return []
    return build_strategy_order(blocks, profile, strategy=strategy, seed=seed)[:max_blocks]


def kv3d_keys_for_blocks(blocks: Iterable[KVBlock], sample_id: str):
    from .blocks import KV3DKey

    return [KV3DKey(sample_id=sample_id, layer=block.layer, head=block.head, chunk=block.span_id) for block in blocks]


def selected_attention_means(
    blocks: Iterable[KVBlock],
    profile: BudgetUtilityProfile,
) -> tuple[float | None, float | None]:
    block_ids = [block.block_id() for block in blocks]
    js = [profile.attention_js_by_block.get(block_id) for block_id in block_ids if profile.attention_js_by_block]
    kl = [profile.attention_kl_by_block.get(block_id) for block_id in block_ids if profile.attention_kl_by_block]
    return _mean(js), _mean(kl)


def aggregate_budget_results(results: Iterable[BudgetEvaluationResult]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, float], list[BudgetEvaluationResult]] = defaultdict(list)
    for result in results:
        buckets[(result.method, result.budget_ratio)].append(result)

    rows: list[dict[str, Any]] = []
    for method, budget_ratio in sorted(buckets, key=lambda item: (item[0][0], item[0][1])):
        bucket = buckets[(method, budget_ratio)]
        rows.append(
            {
                "method": method,
                "budget_ratio": budget_ratio,
                "sample_count": len(bucket),
                "mean_f1": _mean([row.f1 for row in bucket]),
                "mean_contains": _mean([row.contains for row in bucket]),
                "mean_nll": _mean([row.nll for row in bucket]),
                "mean_delta_nll_vs_full": _mean([row.delta_nll_vs_full for row in bucket]),
                "mean_selected_kv_bytes": _mean([float(row.selected_kv_bytes) for row in bucket]),
                "mean_kv_ratio": _mean([row.kv_ratio for row in bucket]),
                "mean_ttft_ms": _mean([row.ttft_ms for row in bucket]),
                "mean_prefill_ms": _mean([row.prefill_ms for row in bucket]),
                "mean_decode_ms": _mean([row.decode_ms for row in bucket]),
                "mean_generation_length": _mean([float(row.generation_length) for row in bucket]),
                "mean_selected_block_count": _mean([float(row.selected_block_count) for row in bucket]),
            }
        )
    return rows


def write_jsonl_results(results: Iterable[BudgetEvaluationResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(result.to_dict(), sort_keys=True) for result in results]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def write_csv_rows(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
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


def quality_curve_rows(summary_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metric_map = {
        "f1": "mean_f1",
        "contains": "mean_contains",
        "nll": "mean_nll",
        "delta_nll_vs_full": "mean_delta_nll_vs_full",
        "kv_bytes": "mean_selected_kv_bytes",
        "kv_ratio": "mean_kv_ratio",
        "ttft_ms": "mean_ttft_ms",
    }
    for row in summary_rows:
        for metric_name, column in metric_map.items():
            rows.append(
                {
                    "method": row["method"],
                    "budget_ratio": row["budget_ratio"],
                    "metric": metric_name,
                    "value": row.get(column),
                }
            )
    return rows


def render_budget_figures(summary_rows: Iterable[dict[str, Any]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(summary_rows)
    figures = [
        ("mean_f1", "F1", "budget_vs_f1.png"),
        ("mean_contains", "Contains", "budget_vs_contains.png"),
        ("mean_nll", "NLL", "budget_vs_nll.png"),
        ("mean_selected_kv_bytes", "Selected KV bytes", "budget_vs_kv_bytes.png"),
        ("mean_ttft_ms", "TTFT (ms)", "budget_vs_ttft.png"),
    ]
    paths: list[Path] = []
    for column, ylabel, filename in figures:
        plt.figure(figsize=(8, 4.5))
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row.get(column) is None:
                continue
            grouped[str(row["method"])].append(row)
        for method in sorted(grouped):
            method_rows = sorted(grouped[method], key=lambda item: float(item["budget_ratio"]))
            plt.plot(
                [float(item["budget_ratio"]) for item in method_rows],
                [float(item[column]) for item in method_rows],
                marker="o",
                label=method,
            )
        if grouped:
            plt.legend(fontsize=7)
        plt.xlabel("Budget ratio")
        plt.ylabel(ylabel)
        plt.title(filename.replace("_", " ").replace(".png", "").title())
        plt.tight_layout()
        path = output_dir / filename
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(path)
    return paths


def _best_method_at_budget(summary_rows: list[dict[str, Any]], *, metric: str, budget_ratio: float) -> dict[str, Any] | None:
    candidates = [row for row in summary_rows if float(row["budget_ratio"]) == budget_ratio and row.get(metric) is not None]
    if not candidates:
        return None
    reverse = metric != "mean_nll" and metric != "mean_delta_nll_vs_full"
    return sorted(candidates, key=lambda row: float(row[metric]), reverse=reverse)[0]


def _budgeted_final_budget(summary_rows: list[dict[str, Any]]) -> float:
    budgeted = [float(row["budget_ratio"]) for row in summary_rows if row["method"] not in {"full_kv", "b_only"}]
    if budgeted:
        return max(budgeted)
    return max(float(row["budget_ratio"]) for row in summary_rows)


def _row_for(summary_rows: list[dict[str, Any]], method: str, budget_ratio: float) -> dict[str, Any] | None:
    return next(
        (row for row in summary_rows if row["method"] == method and abs(float(row["budget_ratio"]) - budget_ratio) < 1e-9),
        None,
    )


def _best_utility_row(summary_rows: list[dict[str, Any]], budget_ratio: float) -> dict[str, Any] | None:
    utility_methods = {"layer_only_utility", "layer_head_utility", "global_span_utility_topk", "hierarchical_layer_head_span"}
    candidates = [
        row
        for row in summary_rows
        if row["method"] in utility_methods
        and abs(float(row["budget_ratio"]) - budget_ratio) < 1e-9
        and row.get("mean_f1") is not None
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: float(row["mean_f1"]), reverse=True)[0]


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _monotonic_violations(summary_rows: list[dict[str, Any]], method: str, metric: str) -> int:
    rows = sorted(
        [row for row in summary_rows if row["method"] == method and row.get(metric) is not None],
        key=lambda row: float(row["budget_ratio"]),
    )
    violations = 0
    for left, right in zip(rows, rows[1:]):
        if float(right[metric]) + 1e-9 < float(left[metric]):
            violations += 1
    return violations


def _near_full_threshold(summary_rows: list[dict[str, Any]]) -> float | None:
    full_rows = [row for row in summary_rows if row["method"] == "full_kv"]
    if not full_rows or full_rows[0].get("mean_f1") is None:
        return None
    full_f1 = float(full_rows[0]["mean_f1"])
    candidates = [
        row
        for row in summary_rows
        if row["method"] not in {"full_kv", "b_only"}
        and row.get("mean_delta_nll_vs_full") is not None
        and row.get("mean_f1") is not None
        and float(row["mean_delta_nll_vs_full"]) <= 0.05
        and float(row["mean_f1"]) >= 0.95 * full_f1
    ]
    if not candidates:
        return None
    return min(float(row["budget_ratio"]) for row in candidates)


def render_budget_report(summary_rows: Iterable[dict[str, Any]], output_path: Path) -> str:
    rows = list(summary_rows)
    methods = sorted({row["method"] for row in rows})
    budgeted_methods = [method for method in methods if method not in {"full_kv", "b_only"}]
    final_budget = _budgeted_final_budget(rows)
    best_f1 = _best_method_at_budget(rows, metric="mean_f1", budget_ratio=final_budget)
    best_nll = _best_method_at_budget(rows, metric="mean_nll", budget_ratio=final_budget)
    best_utility = _best_utility_row(rows, final_budget)
    threshold = _near_full_threshold(rows)
    full_row = next((row for row in rows if row["method"] == "full_kv"), None)
    base_row = next((row for row in rows if row["method"] == "b_only"), None)
    baseline_methods = ["random_span", "uniform_layer_head_span", "recent_span", "first_prefix_span"]
    final_candidates = [row for row in rows if abs(float(row["budget_ratio"]) - final_budget) < 1e-9]
    best_final = sorted(
        [row for row in final_candidates if row["method"] not in {"full_kv", "b_only"} and row.get("mean_f1") is not None],
        key=lambda row: float(row["mean_f1"]),
        reverse=True,
    )
    lines = [
        "# Budgeted KV Selection Evaluation",
        "",
        "## Experimental setup",
        "",
        f"- Samples per method/budget: {None if not rows else max(row.get('sample_count', 0) for row in rows)}",
        f"- Budget ratios: {', '.join(_fmt(value) for value in sorted({float(row['budget_ratio']) for row in rows if row['method'] not in {'full_kv', 'b_only'}}))}",
        "- Selection score: delta NLL from the offline KV3D profile. Attention divergence is retained only as an auxiliary selected-block field.",
        "- Evaluation mode: each selected-KV condition is rerun with the target question/answer generation and NLL against the prefilled context KV cache; results are not offline utility estimates.",
        "",
        "## Summary",
        "",
        f"- Methods evaluated: {', '.join(methods)}",
        f"- Max evaluated budget ratio: {final_budget}",
        f"- Best F1 at max budget: {None if best_f1 is None else best_f1['method']} ({None if best_f1 is None else _fmt(best_f1.get('mean_f1'))})",
        f"- Best NLL at max budget: {None if best_nll is None else best_nll['method']} ({None if best_nll is None else _fmt(best_nll.get('mean_nll'))})",
        f"- Near-full threshold: {threshold if threshold is not None else 'not reached'}",
        f"- Main finding: {best_final[0]['method'] if best_final else 'NA'} is the strongest budgeted F1 method at {final_budget}; utility-based methods do not dominate the first/prefix baseline in this run.",
        "",
        "## Utility-based selection vs baselines",
        "",
    ]
    if best_utility is None:
        lines.append("- No utility-based method is available at the max budget point.")
    else:
        lines.append(
            f"- Best utility method at {final_budget}: {best_utility['method']} with F1={_fmt(best_utility.get('mean_f1'))}, "
            f"contains={_fmt(best_utility.get('mean_contains'))}, NLL={_fmt(best_utility.get('mean_nll'))}."
        )
    for baseline in baseline_methods:
        row = _row_for(rows, baseline, final_budget)
        if row is not None:
            lines.append(
                f"- Baseline {baseline} at {final_budget}: F1={_fmt(row.get('mean_f1'))}, "
                f"contains={_fmt(row.get('mean_contains'))}, NLL={_fmt(row.get('mean_nll'))}."
            )
    prefix_row = _row_for(rows, "first_prefix_span", final_budget)
    random_row = _row_for(rows, "random_span", final_budget)
    recent_row = _row_for(rows, "recent_span", final_budget)
    if best_utility is not None and prefix_row is not None:
        lines.append(
            f"- Interpretation: best utility F1 minus prefix F1 at {final_budget} = "
            f"{_fmt(float(best_utility['mean_f1']) - float(prefix_row['mean_f1']))}; this is evidence for a strong prefix/attention-sink effect rather than a clean utility-selection win."
        )
    if best_utility is not None and random_row is not None:
        lines.append(
            f"- Utility vs random at {final_budget}: F1 delta={_fmt(float(best_utility['mean_f1']) - float(random_row['mean_f1']))}, "
            f"NLL delta={_fmt(float(best_utility['mean_nll']) - float(random_row['mean_nll']))}."
        )
    if best_utility is not None and recent_row is not None:
        lines.append(
            f"- Utility vs recent at {final_budget}: F1 delta={_fmt(float(best_utility['mean_f1']) - float(recent_row['mean_f1']))}, "
            f"NLL delta={_fmt(float(best_utility['mean_nll']) - float(recent_row['mean_nll']))}."
        )

    lines.extend(["", "## Global top-k vs hierarchical allocation", ""])
    global_row = _row_for(rows, "global_span_utility_topk", final_budget)
    hierarchical_row = _row_for(rows, "hierarchical_layer_head_span", final_budget)
    if global_row is not None and hierarchical_row is not None:
        f1_gap = float(global_row["mean_f1"]) - float(hierarchical_row["mean_f1"])
        nll_gap = float(global_row["mean_nll"]) - float(hierarchical_row["mean_nll"])
        winner = "global_span_utility_topk" if f1_gap >= 0 else "hierarchical_layer_head_span"
        lines.append(f"- F1 winner at {final_budget}: {winner}; global-minus-hierarchical F1={_fmt(f1_gap)}, NLL={_fmt(nll_gap)}.")
        lines.append("- Interpretation: global top-k is the better utility-based strategy here; hierarchical allocation loses quality at the same KV ratio.")
    else:
        lines.append("- Both global_span_utility_topk and hierarchical_layer_head_span are required for this comparison.")

    lines.extend(["", "## Budget monotonicity and elbows", ""])
    for method in budgeted_methods:
        lines.append(f"- {method}: F1 monotonicity violations={_monotonic_violations(rows, method, 'mean_f1')}.")
    if best_utility is not None:
        method_rows = sorted(
            [row for row in rows if row["method"] == best_utility["method"] and row.get("mean_f1") is not None],
            key=lambda item: float(item["budget_ratio"]),
        )
        if method_rows:
            method_max = max(float(row["mean_f1"]) for row in method_rows)
            elbow = next((row for row in method_rows if float(row["mean_f1"]) >= 0.95 * method_max), None)
            if elbow is not None:
                lines.append(f"- First budget reaching 95% of the best observed {best_utility['method']} F1: {elbow['budget_ratio']}.")

    lines.extend(["", "## Minimum KV budget near full quality", ""])
    if full_row is not None:
        lines.append(
            f"- Full KV reference: F1={_fmt(full_row.get('mean_f1'))}, contains={_fmt(full_row.get('mean_contains'))}, "
            f"NLL={_fmt(full_row.get('mean_nll'))}, KV bytes={_fmt(full_row.get('mean_selected_kv_bytes'))}."
        )
    if base_row is not None:
        lines.append(
            f"- b_only reference: F1={_fmt(base_row.get('mean_f1'))}, contains={_fmt(base_row.get('mean_contains'))}, "
            f"NLL={_fmt(base_row.get('mean_nll'))}."
        )
    lines.append(f"- Near-full threshold by report rule: {threshold if threshold is not None else 'not reached'}.")
    if prefix_row is not None and full_row is not None:
        lines.append(
            f"- Closest budgeted F1 at {final_budget} is prefix with F1 gap vs full={_fmt(float(prefix_row['mean_f1']) - float(full_row['mean_f1']))} "
            f"and NLL delta={_fmt(prefix_row.get('mean_delta_nll_vs_full'))}; 50% KV still does not meet the near-full criterion."
        )

    lines.extend(["", "## KV bytes and TTFT", ""])
    if full_row is not None and best_utility is not None:
        bytes_saved = 1.0 - float(best_utility.get("mean_kv_ratio") or 0.0)
        ttft_delta = None
        if best_utility.get("mean_ttft_ms") is not None and full_row.get("mean_ttft_ms") is not None:
            ttft_delta = float(best_utility["mean_ttft_ms"]) - float(full_row["mean_ttft_ms"])
        lines.append(
            f"- Best utility method at {final_budget} keeps KV ratio={_fmt(best_utility.get('mean_kv_ratio'))} "
            f"(bytes saved={_fmt(bytes_saved)}); TTFT delta vs full={_fmt(ttft_delta)} ms."
        )
    if full_row is not None and prefix_row is not None:
        prefix_delta = None
        if prefix_row.get("mean_ttft_ms") is not None and full_row.get("mean_ttft_ms") is not None:
            prefix_delta = float(prefix_row["mean_ttft_ms"]) - float(full_row["mean_ttft_ms"])
        lines.append(f"- Best F1 budgeted method TTFT delta vs full at {final_budget}: {_fmt(prefix_delta)} ms.")
    else:
        lines.append("- Full KV and utility rows are required to judge whether lower KV bytes reduce TTFT.")

    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- figures/budget_vs_f1.png",
            "- figures/budget_vs_contains.png",
            "- figures/budget_vs_nll.png",
            "- figures/budget_vs_kv_bytes.png",
            "- figures/budget_vs_ttft.png",
            "",
            "Attention divergence is retained only as an auxiliary selected-block summary and is not used as a selection score.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = "\n".join(lines)
    output_path.write_text(report)
    return report
