"""TTFT decomposition utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TTFTBreakdown:
    communication_time: float
    packaging_time: float
    receiving_time: float
    first_token_compute_time: float

    @property
    def total_ttft(self) -> float:
        return (
            self.communication_time
            + self.packaging_time
            + self.receiving_time
            + self.first_token_compute_time
        )


def estimate_communication_time(kv_bytes: float, bandwidth_bytes_per_sec: float) -> float:
    if bandwidth_bytes_per_sec <= 0:
        raise ValueError("bandwidth_bytes_per_sec must be positive")
    if kv_bytes < 0:
        raise ValueError("kv_bytes must be non-negative")
    return kv_bytes / bandwidth_bytes_per_sec


def decompose_ttft(
    kv_bytes: float,
    bandwidth_bytes_per_sec: float,
    packaging_time: float,
    receiving_time: float,
    first_token_compute_time: float,
) -> TTFTBreakdown:
    communication_time = estimate_communication_time(kv_bytes, bandwidth_bytes_per_sec)
    return TTFTBreakdown(
        communication_time=communication_time,
        packaging_time=packaging_time,
        receiving_time=receiving_time,
        first_token_compute_time=first_token_compute_time,
    )
