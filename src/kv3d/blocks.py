"""3D KV block definitions and chunk helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class KV3DKey:
    sample_id: str
    layer: int
    head: int
    chunk: int
    token_start: int | None = None
    token_end: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.token_start is not None and self.token_end is not None:
            payload["kv_head"] = self.head
            payload["span_id"] = self.chunk
            payload["span_start"] = self.token_start
            payload["span_end"] = self.token_end
            payload["span_size"] = self.token_end - self.token_start
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class KV3DBlock:
    key: KV3DKey
    token_start: int
    token_end: int
    kv_bytes: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["key"] = self.key.to_dict()
        return payload


def chunk_index_for_position(position: int, chunk_size: int) -> int:
    if position < 0:
        raise ValueError("position must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    return position // chunk_size


def chunk_bounds(chunk_index: int, chunk_size: int) -> tuple[int, int]:
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    start = chunk_index * chunk_size
    return start, start + chunk_size
