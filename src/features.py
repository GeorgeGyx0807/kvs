"""Block/page feature extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class BlockFeature:
    sample_id: str
    block_id: str
    layer: int
    head: int
    position_start: int
    position_end: int
    kv_bytes: int
    prefill_attention_mass: float
    prefill_entropy: float
    hidden_norm: float
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEPLOYABLE_FIELDS = {
    "sample_id",
    "block_id",
    "layer",
    "head",
    "position_start",
    "position_end",
    "kv_bytes",
    "prefill_attention_mass",
    "prefill_entropy",
    "hidden_norm",
    "similarity",
}


def build_deployable_feature(record: dict[str, Any]) -> dict[str, Any]:
    forbidden = {"decode_attention", "decode_attention_mass", "oracle", "future", "full_decode"}
    bad = forbidden.intersection(record)
    if bad:
        raise ValueError(f"deployable feature contains forbidden fields: {sorted(bad)}")
    return {k: record[k] for k in DEPLOYABLE_FIELDS if k in record}
