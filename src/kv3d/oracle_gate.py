"""Full-KV quality gates for oracle sample selection."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence


@dataclass(frozen=True)
class GateThresholds:
    qa_f1: float = 0.30
    qmsum_rouge_l: float = 0.20


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    metric_name: str
    metric_value: float
    threshold: float
    reason: str


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for token in left:
        current = [0]
        for index, other in enumerate(right, start=1):
            if token == other:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, answers: Sequence[str]) -> float:
    pred_tokens = _tokens(prediction)
    if not pred_tokens:
        return 0.0
    best = 0.0
    for answer in answers:
        answer_tokens = _tokens(str(answer))
        if not answer_tokens:
            continue
        lcs = _lcs_length(pred_tokens, answer_tokens)
        if lcs <= 0:
            continue
        precision = lcs / len(pred_tokens)
        recall = lcs / len(answer_tokens)
        if precision + recall <= 0.0:
            continue
        best = max(best, 2.0 * precision * recall / (precision + recall))
    return best


def full_kv_gate_decision(
    *,
    task_name: str,
    f1: float | None,
    contains: float | None,
    exact: float | None,
    prediction: str,
    answers: Sequence[str],
    thresholds: GateThresholds | None = None,
) -> GateDecision:
    thresholds = thresholds or GateThresholds()
    task = task_name.lower()
    f1_value = float(f1 or 0.0)
    contains_value = float(contains or 0.0)
    exact_value = float(exact or 0.0)
    if "retrieval" in task:
        accepted = exact_value >= 1.0 or contains_value >= 1.0
        metric_value = max(exact_value, contains_value)
        return GateDecision(
            accepted=accepted,
            metric_name="exact_or_contains",
            metric_value=metric_value,
            threshold=1.0,
            reason="retrieval requires full_kv exact or contains hit",
        )
    if task == "qmsum":
        rouge = rouge_l_f1(prediction, answers)
        return GateDecision(
            accepted=rouge >= thresholds.qmsum_rouge_l,
            metric_name="rouge_l_f1",
            metric_value=rouge,
            threshold=thresholds.qmsum_rouge_l,
            reason="qmsum requires full_kv ROUGE-L F1 above threshold",
        )
    accepted = contains_value >= 1.0 or f1_value >= thresholds.qa_f1
    return GateDecision(
        accepted=accepted,
        metric_name="contains_or_f1",
        metric_value=max(contains_value, f1_value),
        threshold=thresholds.qa_f1,
        reason="qa requires full_kv contains hit or F1 above threshold",
    )
