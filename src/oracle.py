"""Oracle label utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class OracleLabel:
    sample_id: str
    block_id: str
    decode_attention_mass: float
    oracle_utility: float
    oracle_rank: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_oracle_label(
    decode_attention_mass: float,
    kv_bytes: float,
    ttft_penalty: float,
    lambda_bytes: float = 1.0,
    lambda_ttft: float = 1.0,
) -> float:
    return decode_attention_mass - lambda_bytes * kv_bytes - lambda_ttft * ttft_penalty
