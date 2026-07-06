import pytest

from src.kv3d import KV3DBlock, KV3DKey, chunk_bounds, chunk_index_for_position


def test_chunk_index_for_position_uses_floor_division():
    assert chunk_index_for_position(0, 16) == 0
    assert chunk_index_for_position(31, 16) == 1


def test_chunk_helpers_reject_bad_inputs():
    with pytest.raises(ValueError):
        chunk_index_for_position(-1, 16)
    with pytest.raises(ValueError):
        chunk_bounds(0, 0)


def test_block_serialization_keeps_3d_identity():
    block = KV3DBlock(
        key=KV3DKey(sample_id="s1", layer=2, head=3, chunk=4),
        token_start=64,
        token_end=80,
        kv_bytes=1024,
    )
    payload = block.to_dict()
    assert payload["key"]["layer"] == 2
    assert payload["token_end"] == 80

