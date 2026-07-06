from src.kv3d.blocks import KV3DKey
from src.kv3d.executor import _addition_blocks, _clone_past_key_values, _record_key_for_spec


def test_clone_past_key_values_detaches_dynamic_cache_tensors():
    import torch
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache()
    cache.update(
        torch.ones((1, 2, 3, 4)),
        torch.full((1, 2, 3, 4), 2.0),
        layer_idx=0,
    )

    cloned = _clone_past_key_values(cache)
    cloned.layers[0].keys.add_(5.0)
    cloned.layers[0].values.add_(7.0)

    assert torch.all(cache.layers[0].keys == 1.0)
    assert torch.all(cache.layers[0].values == 2.0)


def test_addition_blocks_keeps_single_layer_head_chunk():
    blocks = _addition_blocks(sample_id="s", method="add_layer_head_chunk", layer=1, head=2, chunk=3)

    assert blocks == [KV3DKey(sample_id="s", layer=1, head=2, chunk=3)]


def test_record_key_for_layer_uses_axis_placeholder_zeroes():
    key = _record_key_for_spec(sample_id="s", layer=1, head=None, chunk=None, chunk_size=128, context_length=300)

    assert key == KV3DKey(sample_id="s", layer=1, head=0, chunk=0, token_start=0, token_end=128)
