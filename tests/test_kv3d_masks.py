from src.kv3d.blocks import KV3DKey
from src.kv3d.masks import (
    SequentialLayerMaskDict,
    build_decode_layer_masks,
    build_layer_masks,
    build_prefill_layer_masks,
    selection_kv_bytes,
)


def test_build_layer_masks_creates_one_mask_per_layer():
    specs = build_layer_masks(
        num_layers=3,
        num_attention_heads=32,
        num_key_value_heads=8,
        seq_len=16,
        chunk_size=4,
        kept_blocks=[KV3DKey(sample_id="s", layer=0, head=0, chunk=0)],
    )

    assert len(specs) == 3
    assert specs[0].mask.shape == (1, 32, 16, 16)


def test_selection_kv_bytes_counts_selected_tokens():
    selected = [KV3DKey(sample_id="s", layer=0, head=0, chunk=0)]
    assert selection_kv_bytes(selected_blocks=selected, seq_len=16, head_dim=128, chunk_size=4) == 2048


def test_build_decode_layer_masks_blocks_removed_context_tokens():
    masks = build_decode_layer_masks(
        num_layers=2,
        num_attention_heads=32,
        num_key_value_heads=8,
        context_length=8,
        total_kv_length=9,
        chunk_size=4,
        removed_blocks=[KV3DKey(sample_id="s", layer=0, head=0, chunk=0)],
    )

    assert len(masks) == 2
    assert masks[0].shape == (1, 32, 1, 9)
    assert masks[0][0, 0, 0, 0].item() == float("-inf")
    assert masks[0][0, 0, 0, 4].item() == 0.0
    assert masks[0][0, 0, 0, 8].item() == 0.0


def test_build_prefill_layer_masks_preserves_suffix_tokens():
    masks = build_prefill_layer_masks(
        num_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        context_length=8,
        total_length=10,
        chunk_size=4,
        removed_blocks=[KV3DKey(sample_id="s", layer=0, head=0, chunk=0)],
    )

    assert masks[0].shape == (1, 32, 10, 10)
    assert masks[0][0, 0, 2, 0].item() == 0.0
    assert masks[0][0, 0, 9, 0].item() == float("-inf")
    assert masks[0][0, 0, 9, 4].item() == 0.0
    assert masks[0][0, 0, 9, 8].item() == 0.0


def test_sequential_layer_mask_dict_returns_masks_in_order():
    first = build_decode_layer_masks(
        num_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        context_length=4,
        total_kv_length=5,
        chunk_size=4,
    )[0]
    second = build_decode_layer_masks(
        num_layers=1,
        num_attention_heads=32,
        num_key_value_heads=8,
        context_length=4,
        total_kv_length=6,
        chunk_size=4,
    )[0]
    mapping = SequentialLayerMaskDict([first, second])

    assert mapping["full_attention"] is first
    assert mapping["full_attention"] is second
