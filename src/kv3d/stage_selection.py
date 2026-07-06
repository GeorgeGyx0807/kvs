"""Helpers for selecting stage-2 chunk profiling targets."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .analysis import _coerce_record
from .analysis import _utility_score
from .records import KV3DProfilingRecord


def build_chunk_targets(
    records: list[KV3DProfilingRecord] | list[dict[str, Any]],
    *,
    top_k_per_sample: int = 4,
    middle_k_per_sample: int = 2,
    low_k_per_sample: int = 2,
) -> dict[str, list[list[int]]]:
    if top_k_per_sample <= 0:
        raise ValueError("top_k_per_sample must be positive")
    if middle_k_per_sample < 0 or low_k_per_sample < 0:
        raise ValueError("middle_k_per_sample and low_k_per_sample must be non-negative")

    grouped: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
    for record in (_coerce_record(record) for record in records):
        if record.key is None or record.method != "remove_layer_head":
            continue
        score = _utility_score(record)
        if score is None:
            continue
        grouped[record.sample_id].append((score, record.key.layer, record.key.head))

    chunk_targets: dict[str, list[list[int]]] = {}
    for sample_id, items in grouped.items():
        ranked = sorted(items, key=lambda item: (-item[0], item[1], item[2]))
        selected: list[tuple[float, int, int]] = []
        selected.extend(ranked[:top_k_per_sample])
        if middle_k_per_sample and len(ranked) > top_k_per_sample:
            middle_index = len(ranked) // 2
            middle_start = max(top_k_per_sample, middle_index - (middle_k_per_sample // 2))
            middle_end = min(len(ranked), middle_start + middle_k_per_sample)
            selected.extend(ranked[middle_start:middle_end])
        if low_k_per_sample:
            selected.extend(ranked[-low_k_per_sample:])

        seen: set[tuple[int, int]] = set()
        targets: list[list[int]] = []
        for _, layer, head in selected:
            if (layer, head) in seen:
                continue
            seen.add((layer, head))
            targets.append([layer, head])
        chunk_targets[sample_id] = targets
    return chunk_targets
