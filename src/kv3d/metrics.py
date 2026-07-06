"""Metric helpers for offline 3D KV profiling."""

from __future__ import annotations

import re
from collections import Counter


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", lowered)
    return " ".join(lowered.split())


def contains_answer(prediction: str, answers: tuple[str, ...] | list[str]) -> float:
    normalized_prediction = normalize_text(prediction)
    for answer in answers:
        if normalize_text(str(answer)) in normalized_prediction:
            return 1.0
    return 0.0


def token_f1(prediction: str, answers: tuple[str, ...] | list[str]) -> float:
    pred_tokens = normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    pred_counts = Counter(pred_tokens)
    for answer in answers:
        answer_tokens = normalize_text(str(answer)).split()
        if not answer_tokens:
            continue
        common = pred_counts & Counter(answer_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(answer_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
