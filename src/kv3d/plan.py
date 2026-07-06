"""Experiment planning helpers for offline 3D KV profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ProfilingSpec:
    sample_id: str
    method: str
    layer: int | None = None
    head: int | None = None
    chunk: int | None = None
    budget_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_profiling_plan(
    sample_ids: Iterable[str],
    num_layers: int,
    num_heads: int,
    num_chunks: int,
    include_addition: bool = False,
    include_chunk_level: bool = True,
    chunk_targets: dict[str, set[tuple[int, int]]] | None = None,
) -> list[dict[str, Any]]:
    if num_layers <= 0 or num_heads <= 0 or num_chunks <= 0:
        raise ValueError("num_layers, num_heads, and num_chunks must be positive")

    plan: list[ProfilingSpec] = []
    for sample_id in sample_ids:
        plan.append(ProfilingSpec(sample_id=sample_id, method="full_kv"))
        plan.append(ProfilingSpec(sample_id=sample_id, method="b_only"))

        for layer in range(num_layers):
            plan.append(ProfilingSpec(sample_id=sample_id, method="remove_layer", layer=layer))
            for head in range(num_heads):
                plan.append(
                    ProfilingSpec(
                        sample_id=sample_id,
                        method="remove_layer_head",
                        layer=layer,
                        head=head,
                    )
                )
                if not include_chunk_level:
                    continue
                target_heads = chunk_targets.get(sample_id, set()) if chunk_targets is not None else None
                if target_heads is not None and (layer, head) not in target_heads:
                    continue
                for chunk in range(num_chunks):
                    plan.append(
                        ProfilingSpec(
                            sample_id=sample_id,
                            method="remove_layer_head_chunk",
                            layer=layer,
                            head=head,
                            chunk=chunk,
                        )
                    )
                    if include_addition:
                        plan.append(
                            ProfilingSpec(
                                sample_id=sample_id,
                                method="add_layer_head_chunk",
                                layer=layer,
                                head=head,
                                chunk=chunk,
                            )
                        )

    return [spec.to_dict() for spec in plan]


def filter_profiling_plan(
    plan: Iterable[dict[str, Any]],
    methods: set[str],
    *,
    keep_baselines: bool = True,
) -> list[dict[str, Any]]:
    if not methods:
        return list(plan)
    baseline_methods = {"full_kv", "b_only"} if keep_baselines else set()
    allowed = methods | baseline_methods
    return [item for item in plan if str(item["method"]) in allowed]
