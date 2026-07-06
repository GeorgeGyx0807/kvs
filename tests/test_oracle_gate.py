from src.kv3d.oracle_gate import GateThresholds
from src.kv3d.oracle_gate import full_kv_gate_decision
from src.kv3d.oracle_gate import rouge_l_f1


def test_rouge_l_f1_uses_longest_common_subsequence():
    score = rouge_l_f1("the quick brown animal", ["quick brown fox"])

    assert round(score, 6) == 0.571429


def test_full_kv_gate_accepts_qa_by_contains_or_f1():
    thresholds = GateThresholds(qa_f1=0.30, qmsum_rouge_l=0.20)

    accepted_by_contains = full_kv_gate_decision(
        task_name="hotpotqa",
        f1=0.0,
        contains=1.0,
        exact=0.0,
        prediction="wrong surface form",
        answers=["answer"],
        thresholds=thresholds,
    )
    accepted_by_f1 = full_kv_gate_decision(
        task_name="qasper",
        f1=0.31,
        contains=0.0,
        exact=0.0,
        prediction="partial answer",
        answers=["answer"],
        thresholds=thresholds,
    )
    rejected = full_kv_gate_decision(
        task_name="narrativeqa",
        f1=0.29,
        contains=0.0,
        exact=0.0,
        prediction="weak overlap",
        answers=["answer"],
        thresholds=thresholds,
    )

    assert accepted_by_contains.accepted is True
    assert accepted_by_f1.accepted is True
    assert rejected.accepted is False


def test_full_kv_gate_requires_retrieval_exact_or_contains():
    thresholds = GateThresholds(qa_f1=0.30, qmsum_rouge_l=0.20)

    accepted = full_kv_gate_decision(
        task_name="passage_retrieval_en",
        f1=1.0,
        contains=1.0,
        exact=0.0,
        prediction="passage 3",
        answers=["passage 3"],
        thresholds=thresholds,
    )
    rejected = full_kv_gate_decision(
        task_name="passage_retrieval_en",
        f1=1.0,
        contains=0.0,
        exact=0.0,
        prediction="overlap but not a hit",
        answers=["passage 3"],
        thresholds=thresholds,
    )

    assert accepted.accepted is True
    assert rejected.accepted is False


def test_full_kv_gate_uses_rouge_l_for_qmsum():
    thresholds = GateThresholds(qa_f1=0.30, qmsum_rouge_l=0.50)

    accepted = full_kv_gate_decision(
        task_name="qmsum",
        f1=0.0,
        contains=0.0,
        exact=0.0,
        prediction="the team discussed microphone recognition results",
        answers=["the team discussed microphone recognition results and downsampling"],
        thresholds=thresholds,
    )
    rejected = full_kv_gate_decision(
        task_name="qmsum",
        f1=1.0,
        contains=0.0,
        exact=0.0,
        prediction="the remote control cost was discussed",
        answers=["microphone recognition results and downsampling were discussed"],
        thresholds=thresholds,
    )

    assert accepted.accepted is True
    assert accepted.metric_name == "rouge_l_f1"
    assert rejected.accepted is False
