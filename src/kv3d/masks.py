"""Attention mask helpers for 3D KV profiling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from .blocks import KV3DKey, chunk_bounds


@dataclass(frozen=True)
class MaskSpec:
    layer: int
    mask: torch.Tensor


class SequentialLayerMaskDict(dict):
    def __init__(self, masks: list[torch.Tensor], layer_type: str = "full_attention"):
        super().__init__()
        self.masks = masks
        self.layer_type = layer_type
        self._cursor = 0

    def reset(self) -> None:
        self._cursor = 0

    def __getitem__(self, key: str) -> torch.Tensor:
        if key != self.layer_type:
            return super().__getitem__(key)
        if self._cursor >= len(self.masks):
            raise IndexError("mask sequence exhausted")
        mask = self.masks[self._cursor]
        self._cursor += 1
        return mask


def _head_group_size(num_attention_heads: int, num_key_value_heads: int) -> int:
    if num_attention_heads <= 0 or num_key_value_heads <= 0:
        raise ValueError("head counts must be positive")
    if num_attention_heads % num_key_value_heads != 0:
        raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
    return num_attention_heads // num_key_value_heads


def _causal_mask(seq_len: int, num_attention_heads: int, device: torch.device | str | None, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.full((1, num_attention_heads, seq_len, seq_len), float("-inf"), device=device, dtype=dtype)
    tri = torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))
    mask[:, :, tri] = 0
    return mask


def _positions_for_layer(
    layer: int,
    num_key_value_heads: int,
    seq_len: int,
    chunk_size: int,
    blocks: Iterable[KV3DKey],
) -> dict[int, set[int]]:
    by_head: dict[int, set[int]] = {head: set() for head in range(num_key_value_heads)}
    for block in blocks:
        if block.layer != layer:
            continue
        if block.head not in by_head:
            continue
        start, end = chunk_bounds(block.chunk, chunk_size)
        by_head[block.head].update(range(start, min(end, seq_len)))
    return by_head


def _positions_by_layer(
    num_layers: int,
    num_key_value_heads: int,
    seq_len: int,
    chunk_size: int,
    blocks: Iterable[KV3DKey],
) -> dict[int, dict[int, set[int]]]:
    by_layer = {
        layer: {head: set() for head in range(num_key_value_heads)}
        for layer in range(num_layers)
    }
    for block in blocks:
        if block.layer not in by_layer or block.head not in by_layer[block.layer]:
            continue
        start, end = chunk_bounds(block.chunk, chunk_size)
        by_layer[block.layer][block.head].update(range(start, min(end, seq_len)))
    return by_layer


def build_layer_masks(
    *,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    seq_len: int,
    chunk_size: int,
    kept_blocks: Iterable[KV3DKey] | None = None,
    removed_blocks: Iterable[KV3DKey] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> list[MaskSpec]:
    if kept_blocks is not None and removed_blocks is not None:
        raise ValueError("provide only one of kept_blocks or removed_blocks")
    head_group = _head_group_size(num_attention_heads, num_key_value_heads)
    kept_by_layer = (
        _positions_by_layer(num_layers, num_key_value_heads, seq_len, chunk_size, kept_blocks)
        if kept_blocks is not None
        else None
    )
    removed_by_layer = (
        _positions_by_layer(num_layers, num_key_value_heads, seq_len, chunk_size, removed_blocks)
        if removed_blocks is not None
        else None
    )
    mask_specs: list[MaskSpec] = []
    for layer in range(num_layers):
        mask = _causal_mask(seq_len, num_attention_heads, device=device, dtype=dtype)
        if kept_by_layer is not None:
            allowed = kept_by_layer[layer]
            for head in range(num_key_value_heads):
                start_head = head * head_group
                end_head = start_head + head_group
                allowed_positions = allowed[head]
                for pos in range(seq_len):
                    if pos not in allowed_positions:
                        mask[:, start_head:end_head, :, pos] = float("-inf")
        elif removed_by_layer is not None:
            removed = removed_by_layer[layer]
            for head in range(num_key_value_heads):
                start_head = head * head_group
                end_head = start_head + head_group
                removed_positions = removed[head]
                for pos in removed_positions:
                    mask[:, start_head:end_head, pos:, pos] = float("-inf")
        mask_specs.append(MaskSpec(layer=layer, mask=mask))
    return mask_specs


def _apply_context_block_mask(
    *,
    mask: torch.Tensor,
    layer: int,
    num_key_value_heads: int,
    head_group: int,
    context_length: int,
    chunk_size: int,
    query_start: int,
    kept_blocks: Iterable[KV3DKey] | None,
    removed_blocks: Iterable[KV3DKey] | None,
    kept_positions: dict[int, set[int]] | None = None,
    removed_positions_by_head: dict[int, set[int]] | None = None,
) -> None:
    if kept_blocks is not None or kept_positions is not None:
        allowed = kept_positions or _positions_for_layer(layer, num_key_value_heads, context_length, chunk_size, kept_blocks or [])
        for head in range(num_key_value_heads):
            start_head = head * head_group
            end_head = start_head + head_group
            allowed_positions = allowed[head]
            disallowed = torch.ones(context_length, device=mask.device, dtype=torch.bool)
            if allowed_positions:
                allowed_index = torch.tensor(sorted(allowed_positions), device=mask.device, dtype=torch.long)
                disallowed[allowed_index] = False
            mask[:, start_head:end_head, query_start:, :context_length][:, :, :, disallowed] = float("-inf")
    elif removed_blocks is not None or removed_positions_by_head is not None:
        removed = removed_positions_by_head or _positions_for_layer(
            layer, num_key_value_heads, context_length, chunk_size, removed_blocks or []
        )
        for head in range(num_key_value_heads):
            start_head = head * head_group
            end_head = start_head + head_group
            removed_positions = removed[head]
            if removed_positions:
                removed_index = torch.tensor(sorted(removed_positions), device=mask.device, dtype=torch.long)
                mask[:, start_head:end_head, query_start:, removed_index] = float("-inf")


def build_prefill_layer_masks(
    *,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    context_length: int,
    total_length: int,
    chunk_size: int,
    kept_blocks: Iterable[KV3DKey] | None = None,
    removed_blocks: Iterable[KV3DKey] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    if kept_blocks is not None and removed_blocks is not None:
        raise ValueError("provide only one of kept_blocks or removed_blocks")
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if total_length < context_length:
        raise ValueError("total_length must be >= context_length")
    head_group = _head_group_size(num_attention_heads, num_key_value_heads)
    kept_by_layer = (
        _positions_by_layer(num_layers, num_key_value_heads, context_length, chunk_size, kept_blocks)
        if kept_blocks is not None
        else None
    )
    removed_by_layer = (
        _positions_by_layer(num_layers, num_key_value_heads, context_length, chunk_size, removed_blocks)
        if removed_blocks is not None
        else None
    )
    masks: list[torch.Tensor] = []
    for layer in range(num_layers):
        mask = _causal_mask(total_length, num_attention_heads, device=device, dtype=dtype)
        _apply_context_block_mask(
            mask=mask,
            layer=layer,
            num_key_value_heads=num_key_value_heads,
            head_group=head_group,
            context_length=context_length,
            chunk_size=chunk_size,
            query_start=context_length,
            kept_blocks=kept_blocks,
            removed_blocks=removed_blocks,
            kept_positions=None if kept_by_layer is None else kept_by_layer[layer],
            removed_positions_by_head=None if removed_by_layer is None else removed_by_layer[layer],
        )
        masks.append(mask)
    return masks


def build_decode_layer_masks(
    *,
    num_layers: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    context_length: int,
    total_kv_length: int,
    chunk_size: int,
    query_length: int = 1,
    kept_blocks: Iterable[KV3DKey] | None = None,
    removed_blocks: Iterable[KV3DKey] | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> list[torch.Tensor]:
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    if total_kv_length < context_length:
        raise ValueError("total_kv_length must be >= context_length")
    if query_length <= 0:
        raise ValueError("query_length must be positive")
    if total_kv_length < query_length:
        raise ValueError("total_kv_length must be >= query_length")
    if kept_blocks is not None and removed_blocks is not None:
        raise ValueError("provide only one of kept_blocks or removed_blocks")
    head_group = _head_group_size(num_attention_heads, num_key_value_heads)
    kept_by_layer = (
        _positions_by_layer(num_layers, num_key_value_heads, context_length, chunk_size, kept_blocks)
        if kept_blocks is not None
        else None
    )
    removed_by_layer = (
        _positions_by_layer(num_layers, num_key_value_heads, context_length, chunk_size, removed_blocks)
        if removed_blocks is not None
        else None
    )
    decode_masks: list[torch.Tensor] = []
    for layer in range(num_layers):
        mask = torch.full(
            (1, num_attention_heads, query_length, total_kv_length),
            float("-inf"),
            device=device,
            dtype=dtype,
        )
        for q in range(query_length):
            allowed_end = total_kv_length - query_length + q
            mask[:, :, q, : allowed_end + 1] = 0

        _apply_context_block_mask(
            mask=mask,
            layer=layer,
            num_key_value_heads=num_key_value_heads,
            head_group=head_group,
            context_length=context_length,
            chunk_size=chunk_size,
            query_start=0,
            kept_blocks=kept_blocks,
            removed_blocks=removed_blocks,
            kept_positions=None if kept_by_layer is None else kept_by_layer[layer],
            removed_positions_by_head=None if removed_by_layer is None else removed_by_layer[layer],
        )
        decode_masks.append(mask)
    return decode_masks


def selection_kv_bytes(
    *,
    selected_blocks: Iterable[KV3DKey],
    seq_len: int,
    head_dim: int,
    dtype_bytes: int = 2,
    chunk_size: int,
) -> int:
    total = 0
    for block in selected_blocks:
        start, end = chunk_bounds(block.chunk, chunk_size)
        if start >= seq_len:
            continue
        block_tokens = min(end, seq_len) - start
        if block_tokens <= 0:
            continue
        total += block_tokens * head_dim * 2 * dtype_bytes
    return total
