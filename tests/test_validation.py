import pytest

from src.validation import validate_deployable_fields


def test_validate_deployable_fields_rejects_forbidden_names():
    with pytest.raises(ValueError):
        validate_deployable_fields(["sample_id", "decode_attention_mass"])

