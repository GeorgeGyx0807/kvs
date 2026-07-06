"""Profiling records for offline 3D KV utility analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .blocks import KV3DKey


@dataclass(frozen=True)
class KV3DMetricSnapshot:
    accuracy: float | None = None
    f1: float | None = None
    contains: float | None = None
    nll: float | None = None
    ttft_ms: float | None = None
    prefill_ms: float | None = None
    decode_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KV3DAttentionDivergenceSnapshot:
    js_divergence: float | None = None
    kl_divergence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KV3DProfilingRecord:
    sample_id: str
    method: str
    key: KV3DKey | None
    selected_kv_bytes: int
    metric: KV3DMetricSnapshot
    delta_vs_full: KV3DMetricSnapshot | None = None
    delta_vs_base: KV3DMetricSnapshot | None = None
    attention_divergence: KV3DAttentionDivergenceSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"] = None if self.key is None else self.key.to_dict()
        payload["metric"] = self.metric.to_dict()
        payload["delta_vs_full"] = None if self.delta_vs_full is None else self.delta_vs_full.to_dict()
        payload["delta_vs_base"] = None if self.delta_vs_base is None else self.delta_vs_base.to_dict()
        payload["attention_divergence"] = (
            None if self.attention_divergence is None else self.attention_divergence.to_dict()
        )
        return payload


@dataclass(frozen=True)
class ProfilingManifest:
    experiment_name: str
    model_name: str
    agent_pair: str
    main_dataset: str
    auxiliary_dataset: str
    baseline: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
