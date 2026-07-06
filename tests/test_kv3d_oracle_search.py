from src.kv3d.oracle_search import OracleBlock
from src.kv3d.oracle_search import OracleEval
from src.kv3d.oracle_search import OracleStep
from src.kv3d.oracle_search import best_forward_addition
from src.kv3d.oracle_search import best_quality_eval
from src.kv3d.oracle_search import best_backward_removal
from src.kv3d.oracle_search import build_block_universe
from src.kv3d.oracle_search import compute_threshold_rows
from src.kv3d.oracle_search import label_rows_from_steps
from src.kv3d.oracle_search import oracle_eval_to_dict
from src.kv3d.oracle_search import prune_bottom_fraction
from src.kv3d.oracle_search import quality_ratio
from src.kv3d.oracle_search import render_oracle_report
from src.kv3d.oracle_search import write_oracle_artifacts


def _eval(
    *,
    sample_id: str = "s1",
    method: str = "candidate",
    selected_blocks: tuple[OracleBlock, ...] = (),
    f1: float = 0.0,
    contains: float = 0.0,
    exact: float | None = None,
    nll: float | None = None,
    kv_ratio: float = 0.0,
) -> OracleEval:
    return OracleEval(
        sample_id=sample_id,
        method=method,
        direction="forward",
        stage="span",
        step_index=0,
        selected_blocks=selected_blocks,
        prediction="",
        gold_answer="answer",
        answers=("answer",),
        f1=f1,
        contains=contains,
        exact=exact,
        nll=nll,
        selected_kv_bytes=int(kv_ratio * 1000),
        full_kv_bytes=1000,
        kv_ratio=kv_ratio,
        ttft_ms=10.0,
        prefill_ms=8.0,
        decode_ms=2.0,
    )


def test_best_quality_eval_uses_task_metric_before_nll():
    high_quality_bad_nll = _eval(f1=0.8, contains=1.0, nll=5.0, kv_ratio=0.4)
    low_quality_good_nll = _eval(f1=0.7, contains=1.0, nll=1.0, kv_ratio=0.2)

    assert best_quality_eval([low_quality_good_nll, high_quality_bad_nll], task_family="qa") == high_quality_bad_nll


def test_best_quality_eval_uses_nll_then_smaller_kv_ratio_only_as_tiebreaks():
    larger = _eval(f1=0.8, contains=1.0, nll=2.0, kv_ratio=0.4)
    smaller = _eval(f1=0.8, contains=1.0, nll=2.0, kv_ratio=0.2)
    better_nll = _eval(f1=0.8, contains=1.0, nll=1.5, kv_ratio=0.5)

    assert best_quality_eval([larger, smaller], task_family="qa") == smaller
    assert best_quality_eval([larger, better_nll, smaller], task_family="qa") == better_nll


def test_retrieval_quality_prefers_exact_then_contains_before_f1():
    exact_hit = _eval(f1=0.1, contains=0.0, exact=1.0, nll=9.0)
    contains_hit = _eval(f1=1.0, contains=1.0, exact=0.0, nll=1.0)

    assert best_quality_eval([contains_hit, exact_hit], task_family="retrieval") == exact_hit


def test_prune_bottom_fraction_discards_weakest_30_percent_but_keeps_at_least_one():
    scores = {idx: float(idx) for idx in range(10)}

    assert prune_bottom_fraction(scores, discard_fraction=0.30) == [3, 4, 5, 6, 7, 8, 9]
    assert prune_bottom_fraction({0: 0.0}, discard_fraction=0.30) == [0]


def test_build_block_universe_uses_layer_head_span_ids_and_context_bounds():
    blocks = build_block_universe(
        sample_id="s1",
        layers=[1],
        layer_heads=[(1, 2)],
        span_size=16,
        max_context_tokens=40,
    )

    assert [block.block_id for block in blocks] == [(1, 2, 0), (1, 2, 1), (1, 2, 2)]
    assert blocks[-1].span_start == 32
    assert blocks[-1].span_end == 40


def test_best_forward_addition_selects_candidate_with_best_result():
    first = OracleBlock("s1", 0, 0, 0, 0, 16)
    second = OracleBlock("s1", 0, 0, 1, 16, 32)
    evaluated: dict[tuple[tuple[int, int, int], ...], OracleEval] = {
        (first.block_id,): _eval(selected_blocks=(first,), f1=0.2, contains=0.0, nll=1.0, kv_ratio=0.1),
        (second.block_id,): _eval(selected_blocks=(second,), f1=0.5, contains=1.0, nll=3.0, kv_ratio=0.1),
    }

    candidate, result = best_forward_addition(
        current_blocks=[],
        candidate_blocks=[first, second],
        evaluate=lambda blocks: evaluated[tuple(block.block_id for block in blocks)],
        task_family="qa",
    )

    assert candidate == second
    assert result.f1 == 0.5


def test_best_backward_removal_selects_deletion_with_smallest_quality_loss():
    first = OracleBlock("s1", 0, 0, 0, 0, 16)
    second = OracleBlock("s1", 0, 0, 1, 16, 32)
    evaluated: dict[tuple[tuple[int, int, int], ...], OracleEval] = {
        (second.block_id,): _eval(selected_blocks=(second,), f1=0.6, contains=1.0, nll=2.0, kv_ratio=0.1),
        (first.block_id,): _eval(selected_blocks=(first,), f1=0.1, contains=0.0, nll=1.0, kv_ratio=0.1),
    }

    removed, result = best_backward_removal(
        current_blocks=[first, second],
        evaluate=lambda blocks: evaluated[tuple(block.block_id for block in blocks)],
        task_family="qa",
    )

    assert removed == first
    assert result.selected_blocks == (second,)


def test_quality_ratio_and_threshold_rows_report_minimum_kv_ratio():
    full = _eval(method="full_kv", f1=0.8, contains=1.0, kv_ratio=1.0)
    weak = _eval(f1=0.6, contains=1.0, kv_ratio=0.2)
    strong = _eval(f1=0.76, contains=1.0, kv_ratio=0.4)
    stronger = _eval(f1=0.8, contains=1.0, kv_ratio=0.6)

    assert quality_ratio(strong, full, task_family="qa") == 0.95
    rows = compute_threshold_rows(
        task_name="narrativeqa",
        evals=[weak, stronger, strong],
        full_eval=full,
        thresholds=[0.80, 0.95, 1.0],
        task_family="qa",
    )

    assert rows == [
        {"task_name": "narrativeqa", "threshold": 0.8, "min_kv_ratio": 0.4, "method": "candidate"},
        {"task_name": "narrativeqa", "threshold": 0.95, "min_kv_ratio": 0.4, "method": "candidate"},
        {"task_name": "narrativeqa", "threshold": 1.0, "min_kv_ratio": 0.6, "method": "candidate"},
    ]


def test_label_rows_from_steps_marks_selected_blocks_and_features():
    first = OracleBlock("s1", 1, 2, 3, 48, 64)
    second = OracleBlock("s1", 1, 2, 4, 64, 80)
    step = OracleStep(
        sample_id="s1",
        task_name="qasper",
        direction="forward",
        stage="span",
        step_index=1,
        action="add",
        changed_block=first,
        result=_eval(selected_blocks=(first,), f1=1.0, contains=1.0, kv_ratio=0.5),
    )

    rows = label_rows_from_steps(sample_id="s1", task_name="qasper", universe=[first, second], steps=[step])

    assert rows == [
        {
            "sample_id": "s1",
            "task_name": "qasper",
            "layer": 1,
            "kv_head": 2,
            "span_id": 3,
            "span_start": 48,
            "span_end": 64,
            "selected": 1,
            "first_selected_step": 1,
            "selected_direction": "forward",
        },
        {
            "sample_id": "s1",
            "task_name": "qasper",
            "layer": 1,
            "kv_head": 2,
            "span_id": 4,
            "span_start": 64,
            "span_end": 80,
            "selected": 0,
            "first_selected_step": None,
            "selected_direction": None,
        },
    ]


def test_write_oracle_artifacts_emits_required_tables_and_report(tmp_path):
    block = OracleBlock("s1", 1, 2, 3, 48, 64)
    full = _eval(method="full_kv", selected_blocks=(block,), f1=1.0, contains=1.0, kv_ratio=1.0)
    step = OracleStep(
        sample_id="s1",
        task_name="qasper",
        direction="forward",
        stage="span",
        step_index=1,
        action="add",
        changed_block=block,
        result=_eval(selected_blocks=(block,), f1=0.9, contains=1.0, kv_ratio=0.1),
    )

    write_oracle_artifacts(
        output_dir=tmp_path,
        task_name="qasper",
        baselines=[full],
        steps=[step],
        universe=[block],
        full_eval_by_sample={"s1": full},
        config={"model_name": "mock", "span_size": 16},
    )

    assert (tmp_path / "oracle_step_results.jsonl").exists()
    assert (tmp_path / "oracle_trajectories.jsonl").exists()
    assert (tmp_path / "oracle_thresholds.csv").read_text().startswith("task_name,threshold,min_kv_ratio,method")
    assert (tmp_path / "oracle_labels.csv").read_text().startswith(
        "sample_id,task_name,layer,kv_head,span_id,span_start,span_end,selected"
    )
    assert (tmp_path / "block_features_labels.csv").exists()
    assert (tmp_path / "selection_frequency_layer.csv").exists()
    assert (tmp_path / "figures" / "layer_selection_heatmap.png").exists()
    assert "Do small KV subsets approach full KV?" in (tmp_path / "report.md").read_text()


def test_render_oracle_report_mentions_task_dependence_and_selector_suitability():
    report = render_oracle_report(
        task_name="qasper",
        threshold_rows=[{"task_name": "qasper", "threshold": 0.95, "min_kv_ratio": 0.1, "method": "forward_greedy"}],
        baseline_rows=[{"method": "full_kv", "mean_f1": 1.0}, {"method": "b_only", "mean_f1": 0.0}],
        frequency_tables={"layer": [{"task_name": "qasper", "layer": 1, "selection_rate": 1.0}]},
        config={"max_context_tokens": 2048},
    )

    assert "Which tasks depend more on KV?" in report
    assert "Do oracle-selected layer/head/span patterns differ by task?" in report
    assert "Are these labels suitable for selector training?" in report


def test_oracle_eval_to_dict_compacts_large_non_span_selected_block_lists():
    blocks = tuple(OracleBlock("s1", 0, 0, idx, idx * 16, (idx + 1) * 16) for idx in range(3))
    base = _eval(method="coarse_remove_layer", selected_blocks=blocks, kv_ratio=0.9)
    coarse = OracleEval(**{**base.__dict__, "stage": "layer"})
    span = OracleEval(
        **{
            **coarse.__dict__,
            "method": "forward_greedy",
            "stage": "span",
            "selected_blocks": blocks,
        }
    )

    coarse_payload = oracle_eval_to_dict(coarse, max_inline_selected_blocks=2)
    span_payload = oracle_eval_to_dict(span, max_inline_selected_blocks=2)

    assert coarse_payload["selected_blocks"] == []
    assert coarse_payload["selected_blocks_omitted"] is True
    assert coarse_payload["selected_block_count"] == 3
    assert len(span_payload["selected_blocks"]) == 3
    assert span_payload["selected_blocks_omitted"] is False
