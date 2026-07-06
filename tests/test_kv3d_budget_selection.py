from src.kv3d.budget_selection import BudgetEvaluationResult
from src.kv3d.budget_selection import BudgetUtilityProfile
from src.kv3d.budget_selection import KVBlock
from src.kv3d.budget_selection import aggregate_budget_results
from src.kv3d.budget_selection import build_strategy_order
from src.kv3d.budget_selection import render_budget_report
from src.kv3d.budget_selection import select_blocks_for_budget


def _blocks():
    return [
        KVBlock(layer=layer, head=head, span_id=span, span_start=span * 16, span_end=(span + 1) * 16)
        for layer in range(2)
        for head in range(2)
        for span in range(4)
    ]


def _profile():
    return BudgetUtilityProfile(
        layer_scores={0: 0.1, 1: 0.8},
        layer_head_scores={(0, 0): 0.2, (0, 1): 0.3, (1, 0): 0.7, (1, 1): 0.9},
        span_position_scores={0: 0.4, 1: 0.1, 2: 0.2, 3: 0.3},
        block_scores={(1, 1, 2): 2.0, (0, 0, 0): 1.5},
        attention_js_by_block={(1, 1, 2): 0.99},
    )


def test_prefix_and_recent_orders_select_expected_span_edges():
    blocks = _blocks()

    prefix = select_blocks_for_budget(
        blocks=blocks,
        profile=_profile(),
        strategy="first_prefix_span",
        budget_bytes=4,
        block_bytes=1,
        seed=7,
    )
    recent = select_blocks_for_budget(
        blocks=blocks,
        profile=_profile(),
        strategy="recent_span",
        budget_bytes=4,
        block_bytes=1,
        seed=7,
    )

    assert {block.span_id for block in prefix} == {0}
    assert {block.span_id for block in recent} == {3}


def test_uniform_layer_head_span_spreads_budget_across_spans_and_heads():
    selected = select_blocks_for_budget(
        blocks=_blocks(),
        profile=_profile(),
        strategy="uniform_layer_head_span",
        budget_bytes=4,
        block_bytes=1,
        seed=7,
    )

    assert {block.span_id for block in selected} == {0, 1, 2, 3}
    assert {(block.layer, block.head) for block in selected} == {(0, 0), (0, 1), (1, 0), (1, 1)}


def test_global_span_utility_uses_observed_delta_nll_before_fallback():
    selected = select_blocks_for_budget(
        blocks=_blocks(),
        profile=_profile(),
        strategy="global_span_utility_topk",
        budget_bytes=3,
        block_bytes=1,
        seed=7,
    )

    assert [(block.layer, block.head, block.span_id) for block in selected] == [
        (1, 1, 2),
        (0, 0, 0),
        (1, 1, 0),
    ]


def test_layer_only_and_layer_head_utility_ignore_attention_scores():
    blocks = _blocks()

    layer_only = select_blocks_for_budget(
        blocks=blocks,
        profile=_profile(),
        strategy="layer_only_utility",
        budget_bytes=2,
        block_bytes=1,
        seed=7,
    )
    layer_head = select_blocks_for_budget(
        blocks=blocks,
        profile=_profile(),
        strategy="layer_head_utility",
        budget_bytes=2,
        block_bytes=1,
        seed=7,
    )

    assert all(block.layer == 1 for block in layer_only)
    assert all((block.layer, block.head) == (1, 1) for block in layer_head)


def test_random_strategy_is_seeded_and_budget_limited():
    left = select_blocks_for_budget(
        blocks=_blocks(),
        profile=_profile(),
        strategy="random_span",
        budget_bytes=5,
        block_bytes=1,
        seed=123,
    )
    right = select_blocks_for_budget(
        blocks=_blocks(),
        profile=_profile(),
        strategy="random_span",
        budget_bytes=5,
        block_bytes=1,
        seed=123,
    )

    assert left == right
    assert len(left) == 5


def test_hierarchical_order_prioritizes_layer_then_head_then_span():
    ordered = build_strategy_order(_blocks(), _profile(), strategy="hierarchical_layer_head_span", seed=7)

    assert [(block.layer, block.head, block.span_id) for block in ordered[:4]] == [
        (1, 1, 2),
        (1, 1, 0),
        (1, 1, 3),
        (1, 1, 1),
    ]


def test_aggregate_budget_results_computes_summary_rows():
    rows = [
        BudgetEvaluationResult(
            sample_id="s1",
            dataset="LongBench",
            task_name="narrativeqa",
            method="global_span_utility_topk",
            budget_ratio=0.1,
            prediction="a",
            gold_answer="a",
            answers=["a"],
            f1=1.0,
            contains=1.0,
            nll=2.0,
            delta_nll_vs_full=0.5,
            selected_kv_bytes=100,
            full_kv_bytes=1000,
            kv_ratio=0.1,
            ttft_ms=10.0,
            prefill_ms=7.0,
            decode_ms=3.0,
            generation_length=2,
            selected_block_count=4,
        ),
        BudgetEvaluationResult(
            sample_id="s2",
            dataset="LongBench",
            task_name="narrativeqa",
            method="global_span_utility_topk",
            budget_ratio=0.1,
            prediction="b",
            gold_answer="c",
            answers=["c"],
            f1=0.0,
            contains=0.0,
            nll=4.0,
            delta_nll_vs_full=1.0,
            selected_kv_bytes=100,
            full_kv_bytes=1000,
            kv_ratio=0.1,
            ttft_ms=12.0,
            prefill_ms=8.0,
            decode_ms=4.0,
            generation_length=3,
            selected_block_count=4,
        ),
    ]

    summary = aggregate_budget_results(rows)

    assert summary == [
        {
            "method": "global_span_utility_topk",
            "budget_ratio": 0.1,
            "sample_count": 2,
            "mean_f1": 0.5,
            "mean_contains": 0.5,
            "mean_nll": 3.0,
            "mean_delta_nll_vs_full": 0.75,
            "mean_selected_kv_bytes": 100.0,
            "mean_kv_ratio": 0.1,
            "mean_ttft_ms": 11.0,
            "mean_prefill_ms": 7.5,
            "mean_decode_ms": 3.5,
            "mean_generation_length": 2.5,
            "mean_selected_block_count": 4.0,
        }
    ]


def test_budget_report_answers_required_experiment_questions(tmp_path):
    rows = [
        {"method": "full_kv", "budget_ratio": 1.0, "sample_count": 2, "mean_f1": 0.8, "mean_contains": 0.5, "mean_nll": 1.0, "mean_delta_nll_vs_full": 0.0, "mean_selected_kv_bytes": 1000.0, "mean_kv_ratio": 1.0, "mean_ttft_ms": 100.0, "mean_prefill_ms": 80.0, "mean_decode_ms": 20.0, "mean_generation_length": 2.0, "mean_selected_block_count": 10.0},
        {"method": "random_span", "budget_ratio": 0.5, "sample_count": 2, "mean_f1": 0.4, "mean_contains": 0.2, "mean_nll": 2.0, "mean_delta_nll_vs_full": 1.0, "mean_selected_kv_bytes": 500.0, "mean_kv_ratio": 0.5, "mean_ttft_ms": 70.0, "mean_prefill_ms": 55.0, "mean_decode_ms": 15.0, "mean_generation_length": 2.0, "mean_selected_block_count": 5.0},
        {"method": "global_span_utility_topk", "budget_ratio": 0.5, "sample_count": 2, "mean_f1": 0.7, "mean_contains": 0.5, "mean_nll": 1.1, "mean_delta_nll_vs_full": 0.1, "mean_selected_kv_bytes": 500.0, "mean_kv_ratio": 0.5, "mean_ttft_ms": 68.0, "mean_prefill_ms": 53.0, "mean_decode_ms": 15.0, "mean_generation_length": 2.0, "mean_selected_block_count": 5.0},
        {"method": "hierarchical_layer_head_span", "budget_ratio": 0.5, "sample_count": 2, "mean_f1": 0.6, "mean_contains": 0.4, "mean_nll": 1.3, "mean_delta_nll_vs_full": 0.3, "mean_selected_kv_bytes": 500.0, "mean_kv_ratio": 0.5, "mean_ttft_ms": 69.0, "mean_prefill_ms": 54.0, "mean_decode_ms": 15.0, "mean_generation_length": 2.0, "mean_selected_block_count": 5.0},
    ]

    report = render_budget_report(rows, tmp_path / "report.md")

    assert "Utility-based selection vs baselines" in report
    assert "Global top-k vs hierarchical allocation" in report
    assert "Budget monotonicity and elbows" in report
    assert "Minimum KV budget near full quality" in report
    assert "KV bytes and TTFT" in report
