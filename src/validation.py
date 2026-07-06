"""Leakage and consistency checks."""

from __future__ import annotations


FORBIDDEN_DEPLOYABLE_FIELDS = {
    "decode_attention",
    "decode_attention_mass",
    "oracle",
    "future",
    "full_decode",
}


def validate_deployable_fields(fields: list[str]) -> None:
    bad = sorted(FORBIDDEN_DEPLOYABLE_FIELDS.intersection(fields))
    if bad:
        raise ValueError(f"deployable feature fields contain forbidden entries: {bad}")
