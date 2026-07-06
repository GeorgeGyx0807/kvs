"""Pareto frontier utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FrontierPoint:
    selector: str
    budget: float
    accuracy: float
    kv_bytes: float
    ttft: float


def pareto_frontier(points: list[FrontierPoint]) -> list[FrontierPoint]:
    frontier: list[FrontierPoint] = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            better_or_equal = (
                other.accuracy >= point.accuracy
                and other.kv_bytes <= point.kv_bytes
                and other.ttft <= point.ttft
            )
            strictly_better = (
                other.accuracy > point.accuracy
                or other.kv_bytes < point.kv_bytes
                or other.ttft < point.ttft
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    return frontier
