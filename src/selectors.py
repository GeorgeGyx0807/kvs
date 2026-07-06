"""Selector and budgeting utilities."""

from __future__ import annotations

from typing import Iterable


def select_by_score(
    scores: dict[str, float],
    kv_bytes: dict[str, int],
    budget_bytes: int,
    use_value_density: bool = True,
) -> list[str]:
    if budget_bytes < 0:
        raise ValueError("budget_bytes must be non-negative")

    def key(item: tuple[str, float]) -> tuple[float, float]:
        block_id, score = item
        size = max(kv_bytes.get(block_id, 1), 1)
        density = score / size if use_value_density else score
        return (density, score)

    ordered = sorted(scores.items(), key=key, reverse=True)
    chosen: list[str] = []
    total = 0
    for block_id, _ in ordered:
        size = kv_bytes.get(block_id, 0)
        if total + size > budget_bytes:
            continue
        chosen.append(block_id)
        total += size
    return chosen


def top_k(block_ids: Iterable[str], k: int) -> list[str]:
    if k < 0:
        raise ValueError("k must be non-negative")
    return list(block_ids)[:k]
