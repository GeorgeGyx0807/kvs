"""Attention profile helpers for offline 3D KV profiling."""

from __future__ import annotations

from dataclasses import dataclass
import math
from statistics import fmean
from typing import Any

import torch

from .records import KV3DAttentionDivergenceSnapshot


@dataclass(frozen=True)
class AttentionChunkProfile:
    chunk_masses: dict[tuple[int, int], list[float]]


def _group_size(num_attention_heads: int, num_key_value_heads: int) -> int:
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError("attention dimensions must be positive")
    return max(1, num_attention_heads // num_key_value_heads)


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        if not values:
            return []
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def _chunk_distribution(vector: torch.Tensor, *, chunk_size: int, context_length: int) -> list[float]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if context_length <= 0:
        return []
    values: list[float] = []
    for start in range(0, context_length, chunk_size):
        end = min(start + chunk_size, context_length)
        values.append(float(vector[start:end].sum().item()))
    return _normalize(values)


def attention_chunk_profile(
    attentions: Any,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    chunk_size: int,
    context_length: int,
) -> AttentionChunkProfile | None:
    if attentions is None:
        return None
    group_size = _group_size(num_attention_heads, num_key_value_heads)
    chunk_masses: dict[tuple[int, int], list[float]] = {}
    for layer_idx, layer_attn in enumerate(attentions):
        if layer_attn is None:
            continue
        # Expected shape: [batch, heads, query_len, key_len]
        layer_tensor = layer_attn.detach().float()
        key_length = min(int(layer_tensor.shape[-1]), context_length)
        if key_length <= 0:
            continue
        for head_idx in range(num_key_value_heads):
            start = head_idx * group_size
            end = min(num_attention_heads, start + group_size)
            head_slice = layer_tensor[:, start:end, :, :key_length]
            if head_slice.numel() == 0:
                continue
            vector = head_slice.mean(dim=(0, 1, 2))
            chunk_masses[(layer_idx, head_idx)] = _chunk_distribution(
                vector,
                chunk_size=chunk_size,
                context_length=key_length,
            )
    if not chunk_masses:
        return None
    return AttentionChunkProfile(chunk_masses=chunk_masses)


def _safe_kl(left: list[float], right: list[float], eps: float = 1e-12) -> float | None:
    if len(left) != len(right) or not left:
        return None
    left = _normalize([max(value, eps) for value in left])
    right = _normalize([max(value, eps) for value in right])
    total = 0.0
    for left_value, right_value in zip(left, right):
        total += left_value * math.log(left_value / right_value)
    return float(total)


def _safe_js(left: list[float], right: list[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    midpoint = _normalize([(left_value + right_value) / 2.0 for left_value, right_value in zip(left, right)])
    left_kl = _safe_kl(left, midpoint)
    right_kl = _safe_kl(right, midpoint)
    if left_kl is None or right_kl is None:
        return None
    return 0.5 * (left_kl + right_kl)


def compare_attention_profiles(
    full_profile: AttentionChunkProfile | None,
    current_profile: AttentionChunkProfile | None,
    *,
    layer: int | None,
    head: int | None,
) -> KV3DAttentionDivergenceSnapshot | None:
    if full_profile is None or current_profile is None:
        return None
    if layer is None:
        return None
    target_heads: list[int]
    if head is None:
        target_heads = sorted({scope_head for scope_layer, scope_head in full_profile.chunk_masses if scope_layer == layer})
    else:
        target_heads = [int(head)]
    js_values: list[float] = []
    kl_values: list[float] = []
    for scope_head in target_heads:
        full = full_profile.chunk_masses.get((layer, scope_head))
        current = current_profile.chunk_masses.get((layer, scope_head))
        if full is None or current is None:
            continue
        js_value = _safe_js(full, current)
        kl_value = _safe_kl(full, current)
        if js_value is not None:
            js_values.append(js_value)
        if kl_value is not None:
            kl_values.append(kl_value)
    if not js_values and not kl_values:
        return None
    return KV3DAttentionDivergenceSnapshot(
        js_divergence=None if not js_values else round(fmean(js_values), 6),
        kl_divergence=None if not kl_values else round(fmean(kl_values), 6),
    )
