import pytest

from src.features import build_deployable_feature


def test_build_deployable_feature_rejects_decode_fields():
    with pytest.raises(ValueError):
        build_deployable_feature(
            {
                "sample_id": "s1",
                "block_id": "b1",
                "decode_attention_mass": 0.2,
            }
        )

