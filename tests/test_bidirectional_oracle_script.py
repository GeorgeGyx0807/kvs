import json

import scripts.run_bidirectional_oracle as oracle_script
from src.kv3d.oracle_gate import GateDecision
from src.kv3d.oracle_search import OracleBlock
from src.kv3d.oracle_search import OracleEval
from src.kv3d.oracle_search import OracleStep


def _eval(method: str, selected_blocks=()):
    return OracleEval(
        sample_id="s1",
        method=method,
        direction="baseline",
        stage="baseline",
        step_index=0,
        selected_blocks=tuple(selected_blocks),
        prediction="answer",
        gold_answer="answer",
        answers=("answer",),
        f1=1.0,
        contains=1.0,
        exact=1.0,
        nll=1.0,
        selected_kv_bytes=100,
        full_kv_bytes=100,
        kv_ratio=1.0,
        ttft_ms=1.0,
        prefill_ms=1.0,
        decode_ms=0.0,
    )


def test_main_runs_each_task_and_writes_task_artifacts(monkeypatch, tmp_path):
    calls = []
    block = OracleBlock("s1", 0, 0, 0, 0, 16)
    step = OracleStep(
        sample_id="s1",
        task_name="qasper",
        direction="forward",
        stage="span",
        step_index=1,
        action="add",
        changed_block=block,
        result=_eval("forward_greedy", selected_blocks=(block,)),
    )

    class Args:
        model_name = "mock-model"
        device_map = "cpu"
        dtype = "float32"
        dataset_name = "THUDM/LongBench"
        tasks = "qasper,qmsum"
        split = "test"
        sample_offset = 0
        max_samples = 20
        max_context_tokens = 2048
        span_size = 16
        max_new_tokens = 64
        qmsum_max_new_tokens = 256
        full_kv_gate = False
        gate_qa_f1 = 0.30
        gate_qmsum_rouge_l = 0.20
        gate_candidate_multiplier = 5
        max_candidate_samples = 0
        num_layers = 36
        num_heads = 8
        discard_fraction = 0.30
        prefix_spans = 1
        max_forward_steps = 1
        max_backward_steps = 1
        max_span_candidates = 4
        seed = 123
        sanity = False
        output_dir = tmp_path

    monkeypatch.setattr(oracle_script, "parse_args", lambda: Args)
    monkeypatch.setattr(oracle_script, "load_model_bundle", lambda *args, **kwargs: object())
    monkeypatch.setattr(oracle_script, "_load_task_samples", lambda *args, **kwargs: ["sample"])

    def fake_run_task_oracle(**kwargs):
        calls.append((kwargs["task_name"], kwargs["samples"], kwargs["args"].max_context_tokens, kwargs["args"].max_new_tokens))
        return {
            "baselines": [_eval("full_kv", selected_blocks=(block,)), _eval("b_only")],
            "steps": [step],
            "universe": [block],
            "full_eval_by_sample": {"s1": _eval("full_kv", selected_blocks=(block,))},
        }

    monkeypatch.setattr(oracle_script, "run_task_oracle", fake_run_task_oracle)

    oracle_script.main()

    assert calls == [("qasper", ["sample"], 2048, 64), ("qmsum", ["sample"], 2048, 256)]
    assert (tmp_path / "qasper" / "oracle_step_results.jsonl").exists()
    assert (tmp_path / "qmsum" / "report.md").exists()


def test_parse_args_defaults_use_64_tokens_for_non_qmsum(monkeypatch):
    monkeypatch.setattr("sys.argv", ["run_bidirectional_oracle.py"])
    args = oracle_script.parse_args()

    assert args.max_new_tokens == 64
    assert args.qmsum_max_new_tokens == 256


def test_gate_row_includes_full_prediction_for_diagnostics():
    row = oracle_script._gate_row_for_full_eval(
        task_name="qasper",
        full_eval=_eval("full_kv"),
        decision=GateDecision(
            accepted=False,
            metric_name="contains_or_f1",
            metric_value=0.25,
            threshold=0.30,
            reason="contains_or_f1<0.3",
        ),
        max_new_tokens=64,
    )

    assert row["full_prediction"] == "answer"
    assert json.loads(row["answers_json"]) == ["answer"]


def test_clean_generated_prediction_removes_followup_question():
    assert oracle_script._clean_generated_prediction("answer\n\nQuestion: next\nAnswer: no") == "answer"


def test_oracle_suffix_for_retrieval_requests_paragraph_label_only():
    suffix = oracle_script._oracle_suffix_for_task("passage_retrieval_en", "which paragraph?")

    assert "which paragraph?" in suffix
    assert "only the paragraph label" in suffix
    assert suffix.endswith("Answer:")


def test_sanity_mode_reduces_search_scope(monkeypatch, tmp_path):
    class Args:
        model_name = "mock-model"
        device_map = "cpu"
        dtype = "float32"
        dataset_name = "THUDM/LongBench"
        tasks = "narrativeqa,qasper"
        split = "test"
        sample_offset = 0
        max_samples = 20
        max_context_tokens = 2048
        span_size = 16
        max_new_tokens = 64
        qmsum_max_new_tokens = 256
        full_kv_gate = False
        gate_qa_f1 = 0.30
        gate_qmsum_rouge_l = 0.20
        gate_candidate_multiplier = 5
        max_candidate_samples = 0
        num_layers = 36
        num_heads = 8
        discard_fraction = 0.30
        prefix_spans = 1
        max_forward_steps = 0
        max_backward_steps = 0
        max_span_candidates = 0
        seed = 123
        sanity = True
        output_dir = tmp_path

    monkeypatch.setattr(oracle_script, "parse_args", lambda: Args)
    oracle_script.apply_sanity_defaults(Args)

    assert Args.tasks == "narrativeqa"
    assert Args.max_samples == 1
    assert Args.max_context_tokens == 128
    assert Args.max_new_tokens == 64
    assert Args.num_layers == 2
    assert Args.num_heads == 2
    assert Args.max_forward_steps == 2
    assert Args.max_backward_steps == 2
    assert Args.max_span_candidates == 4
