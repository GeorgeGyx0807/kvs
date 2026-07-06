"""Aggregation helpers for offline 3D KV profiling."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from statistics import fmean, pstdev
from typing import Any, Iterable

from .blocks import KV3DKey
from .records import KV3DAttentionDivergenceSnapshot, KV3DMetricSnapshot, KV3DProfilingRecord


def _mean_or_none(values: list[float | None]) -> float | None:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    return round(fmean(filtered), 6)


def _pearson_or_none(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) < 2 or len(left) != len(right):
        return None
    left_mean = fmean(left)
    right_mean = fmean(right)
    numerator = sum((left_value - left_mean) * (right_value - right_mean) for left_value, right_value in zip(left, right))
    left_var = sum((left_value - left_mean) ** 2 for left_value in left)
    right_var = sum((right_value - right_mean) ** 2 for right_value in right)
    denominator = (left_var * right_var) ** 0.5
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _rank_values(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        mean_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = mean_rank
        i = j
    return ranks


def _spearman_or_none(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(right) < 2 or len(left) != len(right):
        return None
    return _pearson_or_none(_rank_values(left), _rank_values(right))


def _metric_from_payload(payload: dict[str, Any] | None) -> KV3DMetricSnapshot | None:
    if payload is None:
        return None
    return KV3DMetricSnapshot(
        accuracy=payload.get("accuracy"),
        f1=payload.get("f1"),
        contains=payload.get("contains"),
        nll=payload.get("nll"),
        ttft_ms=payload.get("ttft_ms"),
        prefill_ms=payload.get("prefill_ms"),
        decode_ms=payload.get("decode_ms"),
    )


def _attention_from_payload(payload: dict[str, Any] | None) -> KV3DAttentionDivergenceSnapshot | None:
    if payload is None:
        return None
    return KV3DAttentionDivergenceSnapshot(
        js_divergence=payload.get("js_divergence"),
        kl_divergence=payload.get("kl_divergence"),
    )


def _coerce_record(record: KV3DProfilingRecord | dict[str, Any]) -> KV3DProfilingRecord:
    if isinstance(record, KV3DProfilingRecord):
        return record
    key_payload = record.get("key")
    key = None
    if key_payload is not None:
        key = KV3DKey(
            sample_id=str(key_payload["sample_id"]),
            layer=int(key_payload["layer"]),
            head=int(key_payload["head"]),
            chunk=int(key_payload["chunk"]),
            token_start=None if key_payload.get("token_start") is None else int(key_payload["token_start"]),
            token_end=None if key_payload.get("token_end") is None else int(key_payload["token_end"]),
        )
    metric = _metric_from_payload(record.get("metric"))
    if metric is None:
        raise ValueError("profiling record is missing metric")
    return KV3DProfilingRecord(
        sample_id=str(record["sample_id"]),
        method=str(record["method"]),
        key=key,
        selected_kv_bytes=int(record["selected_kv_bytes"]),
        metric=metric,
        delta_vs_full=_metric_from_payload(record.get("delta_vs_full")),
        delta_vs_base=_metric_from_payload(record.get("delta_vs_base")),
        attention_divergence=_attention_from_payload(record.get("attention_divergence")),
    )



def _summary_columns(bucket: list[KV3DProfilingRecord]) -> dict[str, Any]:
    return {
        "record_count": len(bucket),
        "sample_count": len({record.sample_id for record in bucket}),
        "mean_selected_kv_bytes": round(fmean([record.selected_kv_bytes for record in bucket]), 6),
        "mean_metric_accuracy": _mean_or_none([record.metric.accuracy for record in bucket]),
        "mean_metric_f1": _mean_or_none([record.metric.f1 for record in bucket]),
        "mean_metric_contains": _mean_or_none([record.metric.contains for record in bucket]),
        "mean_metric_nll": _mean_or_none([record.metric.nll for record in bucket]),
        "mean_metric_ttft_ms": _mean_or_none([record.metric.ttft_ms for record in bucket]),
        "mean_metric_prefill_ms": _mean_or_none([record.metric.prefill_ms for record in bucket]),
        "mean_metric_decode_ms": _mean_or_none([record.metric.decode_ms for record in bucket]),
        "mean_delta_accuracy": _mean_or_none(
            [None if record.delta_vs_full is None else record.delta_vs_full.accuracy for record in bucket]
        ),
        "mean_delta_f1": _mean_or_none(
            [None if record.delta_vs_full is None else record.delta_vs_full.f1 for record in bucket]
        ),
        "mean_delta_contains": _mean_or_none(
            [None if record.delta_vs_full is None else record.delta_vs_full.contains for record in bucket]
        ),
        "mean_delta_nll": _mean_or_none(
            [None if record.delta_vs_full is None else record.delta_vs_full.nll for record in bucket]
        ),
        "mean_delta_ttft_ms": _mean_or_none(
            [None if record.delta_vs_full is None else record.delta_vs_full.ttft_ms for record in bucket]
        ),
        "mean_delta_prefill_ms": _mean_or_none(
            [None if record.delta_vs_full is None else record.delta_vs_full.prefill_ms for record in bucket]
        ),
        "mean_delta_decode_ms": _mean_or_none(
            [None if record.delta_vs_full is None else record.delta_vs_full.decode_ms for record in bucket]
        ),
        "mean_delta_base_accuracy": _mean_or_none(
            [None if record.delta_vs_base is None else record.delta_vs_base.accuracy for record in bucket]
        ),
        "mean_delta_base_f1": _mean_or_none(
            [None if record.delta_vs_base is None else record.delta_vs_base.f1 for record in bucket]
        ),
        "mean_delta_base_contains": _mean_or_none(
            [None if record.delta_vs_base is None else record.delta_vs_base.contains for record in bucket]
        ),
        "mean_delta_base_nll": _mean_or_none(
            [None if record.delta_vs_base is None else record.delta_vs_base.nll for record in bucket]
        ),
        "mean_delta_base_ttft_ms": _mean_or_none(
            [None if record.delta_vs_base is None else record.delta_vs_base.ttft_ms for record in bucket]
        ),
        "mean_delta_base_prefill_ms": _mean_or_none(
            [None if record.delta_vs_base is None else record.delta_vs_base.prefill_ms for record in bucket]
        ),
        "mean_delta_base_decode_ms": _mean_or_none(
            [None if record.delta_vs_base is None else record.delta_vs_base.decode_ms for record in bucket]
        ),
        "mean_attention_js_divergence": _mean_or_none(
            [None if record.attention_divergence is None else record.attention_divergence.js_divergence for record in bucket]
        ),
        "mean_attention_kl_divergence": _mean_or_none(
            [None if record.attention_divergence is None else record.attention_divergence.kl_divergence for record in bucket]
        ),
    }


def _group_records(records: Iterable[KV3DProfilingRecord], axis: str) -> list[dict[str, Any]]:
    grouped: dict[int, list[KV3DProfilingRecord]] = defaultdict(list)
    for record in records:
        if record.key is None:
            continue
        grouped[getattr(record.key, axis)].append(record)

    table: list[dict[str, Any]] = []
    for axis_value in sorted(grouped):
        table.append({axis: axis_value, **_summary_columns(grouped[axis_value])})
    return table


def _group_records_by_method(records: Iterable[KV3DProfilingRecord], axis: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[KV3DProfilingRecord]] = defaultdict(list)
    for record in records:
        if record.key is None:
            continue
        grouped[(record.method, getattr(record.key, axis))].append(record)

    table: list[dict[str, Any]] = []
    for method, axis_value in sorted(grouped):
        table.append({"method": method, axis: axis_value, **_summary_columns(grouped[(method, axis_value)])})
    return table


def _span_columns(record: KV3DProfilingRecord) -> dict[str, Any]:
    if record.key is None or record.key.token_start is None or record.key.token_end is None:
        return {}
    return {
        "kv_head": record.key.head,
        "span_id": record.key.chunk,
        "span_start": record.key.token_start,
        "span_end": record.key.token_end,
        "span_size": record.key.token_end - record.key.token_start,
    }


def _span_group_records_by_method(records: Iterable[KV3DProfilingRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, int, int, int], list[KV3DProfilingRecord]] = defaultdict(list)
    for record in records:
        if record.method not in {"remove_layer_head_chunk", "add_layer_head_chunk"}:
            continue
        if record.key is None or record.key.token_start is None or record.key.token_end is None:
            continue
        span_size = record.key.token_end - record.key.token_start
        grouped[(record.method, record.key.chunk, record.key.token_start, record.key.token_end, span_size)].append(record)

    table: list[dict[str, Any]] = []
    for method, span_id, span_start, span_end, span_size in sorted(grouped):
        table.append(
            {
                "method": method,
                "span_id": span_id,
                "span_start": span_start,
                "span_end": span_end,
                "span_size": span_size,
                **_summary_columns(grouped[(method, span_id, span_start, span_end, span_size)]),
            }
        )
    return table


def _budget_ratio(record: KV3DProfilingRecord, full_bytes_by_sample: dict[str, int]) -> float | None:
    full_bytes = full_bytes_by_sample.get(record.sample_id)
    if full_bytes is None or full_bytes <= 0:
        return None
    return record.selected_kv_bytes / full_bytes


def _budget_curve_rows(records: list[KV3DProfilingRecord]) -> list[dict[str, Any]]:
    full_bytes_by_sample = {
        record.sample_id: record.selected_kv_bytes
        for record in records
        if record.method == "full_kv" and record.selected_kv_bytes > 0
    }
    grouped: dict[str, list[KV3DProfilingRecord]] = defaultdict(list)
    for record in records:
        grouped[record.method].append(record)

    table: list[dict[str, Any]] = []
    for method in sorted(grouped):
        bucket = grouped[method]
        ratios = [_budget_ratio(record, full_bytes_by_sample) for record in bucket]
        table.append({"method": method, "mean_budget_ratio": _mean_or_none(ratios), **_summary_columns(bucket)})
    return table


def _metric_value(metric: KV3DMetricSnapshot | None, name: str) -> float | None:
    if metric is None:
        return None
    return getattr(metric, name)


def _first_value(values: Iterable[float | None]) -> float | None:
    for value in values:
        if value is not None:
            return value
    return None


def _utility_score(record: KV3DProfilingRecord) -> float | None:
    if record.method.startswith("add_"):
        return _first_value(
            [
                _metric_value(record.delta_vs_base, "f1"),
                _metric_value(record.delta_vs_base, "accuracy"),
                _metric_value(record.delta_vs_base, "contains"),
                None if _metric_value(record.delta_vs_base, "nll") is None else -float(_metric_value(record.delta_vs_base, "nll")),
            ]
        )
    return _first_value(
        [
            None if _metric_value(record.delta_vs_full, "f1") is None else -float(_metric_value(record.delta_vs_full, "f1")),
            None
            if _metric_value(record.delta_vs_full, "accuracy") is None
            else -float(_metric_value(record.delta_vs_full, "accuracy")),
            None
            if _metric_value(record.delta_vs_full, "contains") is None
            else -float(_metric_value(record.delta_vs_full, "contains")),
            _metric_value(record.delta_vs_full, "nll"),
        ]
    )


def _block_utility_rows(
    records: list[KV3DProfilingRecord],
    sample_metadata_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.key is None:
            continue
        sample_metadata = (sample_metadata_by_id or {}).get(record.sample_id, {})
        span_columns = _span_columns(record)
        row = {
            "sample_id": record.sample_id,
            "dataset": sample_metadata.get("dataset"),
            "task_name": sample_metadata.get("task_name"),
            "method": record.method,
            "layer": record.key.layer,
            "head": record.key.head,
            "chunk": record.key.chunk,
            "token_start": record.key.token_start,
            "token_end": record.key.token_end,
            "selected_kv_bytes": record.selected_kv_bytes,
            "metric_accuracy": record.metric.accuracy,
            "metric_f1": record.metric.f1,
            "metric_contains": record.metric.contains,
            "metric_nll": record.metric.nll,
            "metric_ttft_ms": record.metric.ttft_ms,
            "metric_prefill_ms": record.metric.prefill_ms,
            "metric_decode_ms": record.metric.decode_ms,
            "delta_accuracy": _metric_value(record.delta_vs_full, "accuracy"),
            "delta_f1": _metric_value(record.delta_vs_full, "f1"),
            "delta_contains": _metric_value(record.delta_vs_full, "contains"),
            "delta_nll": _metric_value(record.delta_vs_full, "nll"),
            "delta_ttft_ms": _metric_value(record.delta_vs_full, "ttft_ms"),
            "delta_prefill_ms": _metric_value(record.delta_vs_full, "prefill_ms"),
            "delta_decode_ms": _metric_value(record.delta_vs_full, "decode_ms"),
            "delta_base_accuracy": _metric_value(record.delta_vs_base, "accuracy"),
            "delta_base_f1": _metric_value(record.delta_vs_base, "f1"),
            "delta_base_contains": _metric_value(record.delta_vs_base, "contains"),
            "delta_base_nll": _metric_value(record.delta_vs_base, "nll"),
            "delta_base_ttft_ms": _metric_value(record.delta_vs_base, "ttft_ms"),
            "delta_base_prefill_ms": _metric_value(record.delta_vs_base, "prefill_ms"),
            "delta_base_decode_ms": _metric_value(record.delta_vs_base, "decode_ms"),
            "utility_score": _utility_score(record),
        }
        if span_columns:
            row.update(
                {
                    **span_columns,
                    "kv_bytes": record.selected_kv_bytes,
                    "timing_ttft_ms": record.metric.ttft_ms,
                    "timing_prefill_ms": record.metric.prefill_ms,
                    "timing_decode_ms": record.metric.decode_ms,
                    "contains_change": _metric_value(record.delta_vs_full, "contains"),
                }
            )
        rows.append(row)
    return rows


def _top_k_values(sample_blocks: dict[str, list[tuple[float, tuple[int, int, int]]]]) -> list[int]:
    if not sample_blocks:
        return []
    smallest_sample = min(len(blocks) for blocks in sample_blocks.values())
    return [top_k for top_k in (1, 2, 5, 10) if top_k <= smallest_sample]


def _stability_rows_from_grouped(
    grouped: dict[Any, dict[str, list[tuple[float, tuple[int, int, int]]]]],
    *,
    row_prefix_by_group: dict[Any, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_key in sorted(grouped):
        sample_blocks = {
            sample_id: sorted(blocks, key=lambda item: (-item[0], item[1]))
            for sample_id, blocks in grouped[group_key].items()
        }
        if len(sample_blocks) < 2:
            continue
        sample_ids = sorted(sample_blocks)
        for top_k in _top_k_values(sample_blocks):
            overlaps: list[float] = []
            for left_id, right_id in combinations(sample_ids, 2):
                left = {block_id for _, block_id in sample_blocks[left_id][:top_k]}
                right = {block_id for _, block_id in sample_blocks[right_id][:top_k]}
                union = left | right
                if not union:
                    continue
                overlaps.append(len(left & right) / len(union))
            if overlaps:
                rows.append(
                    {
                        **(row_prefix_by_group or {}).get(group_key, {"method": group_key}),
                        "sample_count": len(sample_ids),
                        "top_k": top_k,
                        "pair_count": len(overlaps),
                        "mean_topk_jaccard": _mean_or_none(overlaps),
                    }
                )
    return rows


def _stability_rows(records: list[KV3DProfilingRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[tuple[float, tuple[int, int, int]]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.key is None:
            continue
        score = _utility_score(record)
        if score is None:
            continue
        block_id = (record.key.layer, record.key.head, record.key.chunk)
        grouped[record.method][record.sample_id].append((score, block_id))
    return _stability_rows_from_grouped(grouped)


def _stability_by_task_rows(
    records: list[KV3DProfilingRecord],
    sample_metadata_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, list[tuple[float, tuple[int, int, int]]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    row_prefix_by_group: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        if record.key is None:
            continue
        metadata = (sample_metadata_by_id or {}).get(record.sample_id, {})
        dataset = metadata.get("dataset")
        task_name = metadata.get("task_name")
        if dataset is None or task_name is None:
            continue
        score = _utility_score(record)
        if score is None:
            continue
        group_key = (str(dataset), str(task_name), record.method)
        row_prefix_by_group[group_key] = {"dataset": str(dataset), "task_name": str(task_name), "method": record.method}
        block_id = (record.key.layer, record.key.head, record.key.chunk)
        grouped[group_key][record.sample_id].append((score, block_id))
    return _stability_rows_from_grouped(grouped, row_prefix_by_group=row_prefix_by_group)


def _heterogeneity_rows(records: list[KV3DProfilingRecord]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for record in records:
        if record.key is None:
            continue
        score = _utility_score(record)
        if score is None:
            continue
        for axis in ("layer", "head", "chunk"):
            grouped[(record.method, axis, getattr(record.key, axis))].append(score)

    by_method_axis: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    for (method, axis, axis_value), scores in grouped.items():
        by_method_axis[(method, axis)].append((axis_value, round(fmean(scores), 6)))

    rows: list[dict[str, Any]] = []
    for method, axis in sorted(by_method_axis):
        values = sorted(by_method_axis[(method, axis)], key=lambda item: item[0])
        if len(values) < 2:
            continue
        utilities = [utility for _, utility in values]
        min_value = min(utilities)
        max_value = max(utilities)
        mean_value = fmean(utilities)
        std_value = pstdev(utilities)
        top_axis_value, top_utility = max(values, key=lambda item: item[1])
        bottom_axis_value, bottom_utility = min(values, key=lambda item: item[1])
        rows.append(
            {
                "method": method,
                "axis": axis,
                "axis_value_count": len(values),
                "min_mean_utility": round(min_value, 6),
                "max_mean_utility": round(max_value, 6),
                "range_mean_utility": round(max_value - min_value, 6),
                "std_mean_utility": round(std_value, 6),
                "cv_mean_utility": None if mean_value == 0 else round(std_value / abs(mean_value), 6),
                "top_axis_value": top_axis_value,
                "top_mean_utility": top_utility,
                "bottom_axis_value": bottom_axis_value,
                "bottom_mean_utility": bottom_utility,
                "top_bottom_gap": round(top_utility - bottom_utility, 6),
            }
        )
    return rows


def _addition_removal_alignment_rows(
    records: list[KV3DProfilingRecord],
    sample_metadata_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    remove_method = "remove_layer_head_chunk"
    add_method = "add_layer_head_chunk"
    grouped: dict[tuple[str | None, str | None], dict[tuple[str, int, int, int], dict[str, float]]] = defaultdict(dict)
    samples_by_group: dict[tuple[str | None, str | None], set[str]] = defaultdict(set)
    for record in records:
        if record.key is None or record.method not in {remove_method, add_method}:
            continue
        score = _utility_score(record)
        if score is None:
            continue
        metadata = (sample_metadata_by_id or {}).get(record.sample_id, {})
        group_key = (metadata.get("dataset"), metadata.get("task_name"))
        block_key = (record.sample_id, record.key.layer, record.key.head, record.key.chunk)
        grouped[group_key].setdefault(block_key, {})[record.method] = score
        samples_by_group[group_key].add(record.sample_id)

    rows: list[dict[str, Any]] = []
    for dataset, task_name in sorted(grouped, key=lambda item: (str(item[0]), str(item[1]))):
        paired = [
            (block_key, scores[remove_method], scores[add_method])
            for block_key, scores in grouped[(dataset, task_name)].items()
            if remove_method in scores and add_method in scores
        ]
        if len(paired) < 2:
            continue
        remove_scores = [remove_score for _, remove_score, _ in paired]
        add_scores = [add_score for _, _, add_score in paired]
        remove_ranked = [block_key for block_key, _, _ in sorted(paired, key=lambda item: (-item[1], item[0]))]
        add_ranked = [block_key for block_key, _, _ in sorted(paired, key=lambda item: (-item[2], item[0]))]
        row = {
            "dataset": dataset,
            "task_name": task_name,
            "method_pair": f"{remove_method}:{add_method}",
            "sample_count": len(samples_by_group[(dataset, task_name)]),
            "block_count": len(paired),
            "pearson_utility": _pearson_or_none(remove_scores, add_scores),
        }
        for top_k in (1, 2, 5, 10):
            if top_k > len(paired):
                continue
            left = set(remove_ranked[:top_k])
            right = set(add_ranked[:top_k])
            union = left | right
            row[f"top_{top_k}_jaccard"] = None if not union else round(len(left & right) / len(union), 6)
        rows.append(row)
    return rows


def _attention_divergence_rows(
    records: list[KV3DProfilingRecord],
    sample_metadata_by_id: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.key is None or record.attention_divergence is None:
            continue
        sample_metadata = (sample_metadata_by_id or {}).get(record.sample_id, {})
        rows.append(
            {
                "sample_id": record.sample_id,
                "dataset": sample_metadata.get("dataset"),
                "task_name": sample_metadata.get("task_name"),
                "method": record.method,
                "layer": record.key.layer,
                "head": record.key.head,
                "chunk": record.key.chunk,
                **_span_columns(record),
                "attention_js_divergence": record.attention_divergence.js_divergence,
                "attention_kl_divergence": record.attention_divergence.kl_divergence,
                "delta_nll": _metric_value(record.delta_vs_full, "nll"),
                "delta_f1": _metric_value(record.delta_vs_full, "f1"),
                "utility_score": _utility_score(record),
            }
        )
    return rows


def _attention_divergence_summary_rows(records: list[KV3DProfilingRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[KV3DProfilingRecord]] = defaultdict(list)
    for record in records:
        if record.key is None or record.attention_divergence is None:
            continue
        grouped[record.method].append(record)

    rows: list[dict[str, Any]] = []
    for method in sorted(grouped):
        bucket = grouped[method]
        js_values = [record.attention_divergence.js_divergence for record in bucket if record.attention_divergence.js_divergence is not None]
        kl_values = [record.attention_divergence.kl_divergence for record in bucket if record.attention_divergence.kl_divergence is not None]
        delta_nll = [_metric_value(record.delta_vs_full, "nll") for record in bucket if _metric_value(record.delta_vs_full, "nll") is not None]
        utility_scores = [_utility_score(record) for record in bucket if _utility_score(record) is not None]
        js_for_nll = [
            (record.attention_divergence.js_divergence, _metric_value(record.delta_vs_full, "nll"))
            for record in bucket
            if record.attention_divergence.js_divergence is not None and _metric_value(record.delta_vs_full, "nll") is not None
        ]
        js_for_utility = [
            (record.attention_divergence.js_divergence, _utility_score(record))
            for record in bucket
            if record.attention_divergence.js_divergence is not None and _utility_score(record) is not None
        ]
        if not js_values and not kl_values:
            continue
        rows.append(
            {
                "method": method,
                "record_count": len(bucket),
                "mean_attention_js_divergence": _mean_or_none(js_values),
                "mean_attention_kl_divergence": _mean_or_none(kl_values),
                "mean_delta_nll": _mean_or_none(delta_nll),
                "mean_utility_score": _mean_or_none(utility_scores),
                "pearson_js_delta_nll": _pearson_or_none([x for x, _ in js_for_nll], [y for _, y in js_for_nll]) if js_for_nll else None,
                "spearman_js_delta_nll": _spearman_or_none([x for x, _ in js_for_nll], [y for _, y in js_for_nll]) if js_for_nll else None,
                "pearson_js_utility": _pearson_or_none([x for x, _ in js_for_utility], [y for _, y in js_for_utility]) if js_for_utility else None,
                "spearman_js_utility": _spearman_or_none([x for x, _ in js_for_utility], [y for _, y in js_for_utility]) if js_for_utility else None,
            }
        )
    return rows


def _attention_correlation_matrix_rows(records: list[KV3DProfilingRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    divergence_metrics = {
        "attention_js_divergence": lambda record: record.attention_divergence.js_divergence,
        "attention_kl_divergence": lambda record: record.attention_divergence.kl_divergence,
    }
    outcome_metrics = {
        "delta_nll": lambda record: _metric_value(record.delta_vs_full, "nll"),
        "delta_f1": lambda record: _metric_value(record.delta_vs_full, "f1"),
    }
    grouped: dict[str, list[KV3DProfilingRecord]] = defaultdict(list)
    for record in records:
        if record.key is None or record.attention_divergence is None:
            continue
        grouped[record.method].append(record)

    for method in sorted(grouped):
        bucket = grouped[method]
        for divergence_name, divergence_getter in divergence_metrics.items():
            for outcome_name, outcome_getter in outcome_metrics.items():
                paired = [
                    (divergence_getter(record), outcome_getter(record))
                    for record in bucket
                    if divergence_getter(record) is not None and outcome_getter(record) is not None
                ]
                if len(paired) < 2:
                    continue
                left = [float(value) for value, _ in paired]
                right = [float(value) for _, value in paired]
                rows.append(
                    {
                        "method": method,
                        "divergence_metric": divergence_name,
                        "outcome_metric": outcome_name,
                        "record_count": len(paired),
                        "pearson": _pearson_or_none(left, right),
                        "spearman": _spearman_or_none(left, right),
                    }
                )
    return rows


def aggregate_profiling_records(
    records: Iterable[KV3DProfilingRecord | dict[str, Any]],
    sample_metadata_by_id: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    records = [_coerce_record(record) for record in records]
    return {
        "layer": _group_records(records, "layer"),
        "head": _group_records(records, "head"),
        "chunk": _group_records(records, "chunk"),
        "layer_by_method": _group_records_by_method(records, "layer"),
        "head_by_method": _group_records_by_method(records, "head"),
        "chunk_by_method": _group_records_by_method(records, "chunk"),
        "span_by_method": _span_group_records_by_method(records),
        "block_utility": _block_utility_rows(records, sample_metadata_by_id=sample_metadata_by_id),
        "attention_divergence": _attention_divergence_rows(records, sample_metadata_by_id=sample_metadata_by_id),
        "attention_divergence_by_method": _attention_divergence_summary_rows(records),
        "attention_correlation_matrix": _attention_correlation_matrix_rows(records),
        "stability": _stability_rows(records),
        "stability_by_task": _stability_by_task_rows(records, sample_metadata_by_id=sample_metadata_by_id),
        "heterogeneity": _heterogeneity_rows(records),
        "addition_removal_alignment": _addition_removal_alignment_rows(
            records,
            sample_metadata_by_id=sample_metadata_by_id,
        ),
        "budget_curves": _budget_curve_rows(records),
    }
