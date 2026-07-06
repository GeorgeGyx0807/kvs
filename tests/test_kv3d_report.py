from src.kv3d import KV3DKey, KV3DMetricSnapshot, KV3DProfilingRecord
from src.kv3d.analysis import aggregate_profiling_records
from src.kv3d.plan import generate_profiling_plan
from src.kv3d.report import build_key_findings, render_profiling_report


def test_render_profiling_report_includes_plan_and_axis_sections():
    plan = generate_profiling_plan(sample_ids=["s1"], num_layers=1, num_heads=1, num_chunks=1)
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
            selected_kv_bytes=128,
            metric=KV3DMetricSnapshot(accuracy=0.5, nll=1.0, ttft_ms=10.0),
            delta_vs_full=KV3DMetricSnapshot(accuracy=-0.2, nll=0.3, ttft_ms=1.5),
        )
    ]
    tables = aggregate_profiling_records(records)

    report = render_profiling_report(
        plan=plan,
        records=records,
        tables=tables,
        experiment_name="offline_3d_kv_utility_profiling",
        model_name="Qwen3-8B",
        main_dataset="LongBench",
    )

    assert "# Offline 3D KV Utility Profiling Report" in report
    assert "Plan size: 5" in report
    assert "## Layer Profile" in report
    assert "## Head Profile" in report
    assert "## Chunk Profile" in report
    assert "## Span Profile by Method" in report
    assert "## Key Findings" in report
    assert "Top layer candidate" in report
    assert "Heterogeneity snapshot" in report
    assert "## Figures" in report
    assert "## Top-k Stability by Task" in report
    assert "## Addition vs Removal Alignment" in report


def test_build_key_findings_exports_machine_readable_summary():
    tables = {
        "layer_by_method": [{"method": "remove_layer", "layer": 1, "mean_delta_nll": 0.4}],
        "head_by_method": [{"method": "remove_layer_head", "head": 2, "mean_delta_f1": -0.3}],
        "chunk_by_method": [{"method": "remove_layer_head_chunk", "chunk": 3, "mean_delta_nll": 0.2}],
        "span_by_method": [
            {"method": "remove_layer_head_chunk", "span_id": 3, "span_start": 48, "span_end": 64, "mean_delta_nll": 0.25}
        ],
        "block_utility": [
            {
                "method": "remove_layer_head_chunk",
                "sample_id": "s1",
                "layer": 1,
                "head": 2,
                "chunk": 3,
                "utility_score": 0.5,
            }
        ],
        "stability": [{"method": "remove_layer_head_chunk", "top_k": 1, "mean_topk_jaccard": 0.25}],
        "stability_by_task": [
            {
                "dataset": "LongBench",
                "task_name": "narrativeqa",
                "method": "remove_layer_head_chunk",
                "top_k": 1,
                "mean_topk_jaccard": 0.5,
            }
        ],
        "heterogeneity": [
            {
                "method": "remove_layer_head_chunk",
                "axis": "chunk",
                "range_mean_utility": 0.2,
            }
        ],
        "addition_removal_alignment": [
            {
                "dataset": "LongBench",
                "task_name": "narrativeqa",
                "method_pair": "remove_layer_head_chunk:add_layer_head_chunk",
                "pearson_utility": -0.4,
            }
        ],
    }

    findings = build_key_findings(tables)

    assert findings["top_layer"]["layer"] == 1
    assert findings["top_head"]["head"] == 2
    assert findings["top_chunk"]["chunk"] == 3
    assert findings["top_span"]["span_id"] == 3
    assert findings["top_block"]["sample_id"] == "s1"
    assert findings["stability"]["mean_topk_jaccard"] == 0.25
    assert findings["task_stability"]["task_name"] == "narrativeqa"
    assert findings["heterogeneity"]["range_mean_utility"] == 0.2
    assert findings["addition_removal_alignment"]["pearson_utility"] == -0.4
