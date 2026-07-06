from src.kv3d import KV3DKey, KV3DMetricSnapshot, KV3DProfilingRecord
from src.kv3d.analysis import aggregate_profiling_records


def test_aggregate_profiling_records_groups_by_3d_axes():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_block",
            key=KV3DKey(sample_id="s1", layer=0, head=1, chunk=2),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(
                accuracy=0.8,
                f1=0.7,
                contains=1.0,
                nll=1.5,
                ttft_ms=20.0,
                prefill_ms=10.0,
                decode_ms=5.0,
            ),
            delta_vs_full=KV3DMetricSnapshot(
                accuracy=-0.1,
                f1=-0.2,
                contains=0.0,
                nll=0.2,
                ttft_ms=1.0,
                prefill_ms=0.5,
                decode_ms=0.25,
            ),
            delta_vs_base=KV3DMetricSnapshot(
                accuracy=0.3,
                f1=0.2,
                contains=1.0,
                nll=-0.4,
                ttft_ms=3.0,
                prefill_ms=1.0,
                decode_ms=2.0,
            ),
        ),
        KV3DProfilingRecord(
            sample_id="s2",
            method="remove_block",
            key=KV3DKey(sample_id="s2", layer=0, head=1, chunk=2),
            selected_kv_bytes=120,
            metric=KV3DMetricSnapshot(
                accuracy=0.6,
                f1=0.5,
                contains=0.0,
                nll=2.5,
                ttft_ms=30.0,
                prefill_ms=20.0,
                decode_ms=7.0,
            ),
            delta_vs_full=KV3DMetricSnapshot(
                accuracy=-0.2,
                f1=-0.3,
                contains=-1.0,
                nll=0.3,
                ttft_ms=2.0,
                prefill_ms=1.5,
                decode_ms=0.75,
            ),
            delta_vs_base=KV3DMetricSnapshot(
                accuracy=0.1,
                f1=0.4,
                contains=0.0,
                nll=-0.2,
                ttft_ms=5.0,
                prefill_ms=2.0,
                decode_ms=3.0,
            ),
        ),
    ]

    tables = aggregate_profiling_records(records)

    assert tables["layer"][0]["layer"] == 0
    assert tables["layer"][0]["record_count"] == 2
    assert tables["head"][0]["head"] == 1
    assert tables["chunk"][0]["chunk"] == 2
    assert tables["layer"][0]["mean_metric_f1"] == 0.6
    assert tables["layer"][0]["mean_metric_contains"] == 0.5
    assert tables["layer"][0]["mean_metric_prefill_ms"] == 15.0
    assert tables["layer"][0]["mean_metric_decode_ms"] == 6.0
    assert tables["layer"][0]["mean_delta_accuracy"] == -0.15
    assert tables["layer"][0]["mean_delta_f1"] == -0.25
    assert tables["layer"][0]["mean_delta_contains"] == -0.5
    assert tables["layer"][0]["mean_delta_prefill_ms"] == 1.0
    assert tables["layer"][0]["mean_delta_decode_ms"] == 0.5
    assert tables["layer"][0]["mean_delta_base_accuracy"] == 0.2
    assert tables["layer"][0]["mean_delta_base_f1"] == 0.3
    assert tables["layer"][0]["mean_delta_base_nll"] == -0.3
    assert tables["layer"][0]["mean_delta_base_ttft_ms"] == 4.0
    assert tables["layer_by_method"][0]["method"] == "remove_block"
    assert tables["layer_by_method"][0]["layer"] == 0
    assert tables["head_by_method"][0]["method"] == "remove_block"
    assert tables["chunk_by_method"][0]["method"] == "remove_block"


def test_aggregate_profiling_records_builds_observed_budget_curve_rows():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="full_kv",
            key=None,
            selected_kv_bytes=1000,
            metric=KV3DMetricSnapshot(accuracy=1.0, f1=1.0, nll=1.0, ttft_ms=100.0),
        ),
        KV3DProfilingRecord(
            sample_id="s1",
            method="b_only",
            key=None,
            selected_kv_bytes=0,
            metric=KV3DMetricSnapshot(accuracy=0.0, f1=0.0, nll=3.0, ttft_ms=10.0),
            delta_vs_full=KV3DMetricSnapshot(accuracy=-1.0, f1=-1.0, nll=2.0, ttft_ms=-90.0),
        ),
        KV3DProfilingRecord(
            sample_id="s1",
            method="add_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
            selected_kv_bytes=250,
            metric=KV3DMetricSnapshot(accuracy=0.5, f1=0.25, nll=2.0, ttft_ms=30.0),
            delta_vs_full=KV3DMetricSnapshot(accuracy=-0.5, f1=-0.75, nll=1.0, ttft_ms=-70.0),
            delta_vs_base=KV3DMetricSnapshot(accuracy=0.5, f1=0.25, nll=-1.0, ttft_ms=20.0),
        ),
    ]

    tables = aggregate_profiling_records(records)

    budget_rows = {row["method"]: row for row in tables["budget_curves"]}
    assert budget_rows["full_kv"]["mean_budget_ratio"] == 1.0
    assert budget_rows["b_only"]["mean_budget_ratio"] == 0.0
    assert budget_rows["add_layer_head_chunk"]["mean_budget_ratio"] == 0.25
    assert budget_rows["add_layer_head_chunk"]["mean_delta_base_nll"] == -1.0


def test_aggregate_profiling_records_exports_block_utility_rows():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=1, chunk=2),
            selected_kv_bytes=512,
            metric=KV3DMetricSnapshot(accuracy=0.5, f1=0.25, nll=2.0, ttft_ms=20.0),
            delta_vs_full=KV3DMetricSnapshot(accuracy=-0.2, f1=-0.3, nll=0.4, ttft_ms=-5.0),
        )
    ]

    tables = aggregate_profiling_records(records)

    assert tables["block_utility"] == [
        {
            "sample_id": "s1",
            "dataset": None,
            "task_name": None,
            "method": "remove_layer_head_chunk",
            "layer": 0,
            "head": 1,
            "chunk": 2,
            "token_start": None,
            "token_end": None,
            "selected_kv_bytes": 512,
            "metric_accuracy": 0.5,
            "metric_f1": 0.25,
            "metric_contains": None,
            "metric_nll": 2.0,
            "metric_ttft_ms": 20.0,
            "metric_prefill_ms": None,
            "metric_decode_ms": None,
            "delta_accuracy": -0.2,
            "delta_f1": -0.3,
            "delta_contains": None,
            "delta_nll": 0.4,
            "delta_ttft_ms": -5.0,
            "delta_prefill_ms": None,
            "delta_decode_ms": None,
            "delta_base_accuracy": None,
            "delta_base_f1": None,
            "delta_base_contains": None,
            "delta_base_nll": None,
            "delta_base_ttft_ms": None,
            "delta_base_prefill_ms": None,
            "delta_base_decode_ms": None,
            "utility_score": 0.3,
        }
    ]


def test_aggregate_profiling_records_exports_span_budget_fields():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=2, head=3, chunk=5, token_start=80, token_end=96),
            selected_kv_bytes=4096,
            metric=KV3DMetricSnapshot(contains=0.0, f1=0.1, nll=2.0, ttft_ms=20.0, prefill_ms=8.0, decode_ms=12.0),
            delta_vs_full=KV3DMetricSnapshot(contains=-1.0, f1=-0.2, nll=0.5, ttft_ms=-3.0, prefill_ms=-1.0, decode_ms=-2.0),
        )
    ]

    tables = aggregate_profiling_records(records)

    row = tables["block_utility"][0]
    assert row["layer"] == 2
    assert row["kv_head"] == 3
    assert row["span_id"] == 5
    assert row["span_start"] == 80
    assert row["span_end"] == 96
    assert row["span_size"] == 16
    assert row["contains_change"] == -1.0
    assert row["kv_bytes"] == 4096
    assert row["timing_ttft_ms"] == 20.0
    assert row["timing_prefill_ms"] == 8.0
    assert row["timing_decode_ms"] == 12.0


def test_aggregate_profiling_records_reports_top_block_stability():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.9),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.4),
        ),
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=1, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.8),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.2),
        ),
        KV3DProfilingRecord(
            sample_id="s2",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s2", layer=0, head=0, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.7),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.3),
        ),
        KV3DProfilingRecord(
            sample_id="s2",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s2", layer=1, head=0, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.6),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.1),
        ),
    ]

    tables = aggregate_profiling_records(records)

    assert tables["stability"] == [
        {
            "method": "remove_layer_head_chunk",
            "sample_count": 2,
            "top_k": 1,
            "pair_count": 1,
            "mean_topk_jaccard": 1.0,
        },
        {
            "method": "remove_layer_head_chunk",
            "sample_count": 2,
            "top_k": 2,
            "pair_count": 1,
            "mean_topk_jaccard": 0.333333,
        },
    ]


def test_aggregate_profiling_records_reports_task_scoped_stability():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.9),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.4),
        ),
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=1, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.8),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.2),
        ),
        KV3DProfilingRecord(
            sample_id="s2",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s2", layer=0, head=0, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.7),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.3),
        ),
        KV3DProfilingRecord(
            sample_id="s2",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s2", layer=1, head=0, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.6),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.1),
        ),
        KV3DProfilingRecord(
            sample_id="s3",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s3", layer=1, head=1, chunk=1),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.5),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.5),
        ),
    ]
    sample_metadata_by_id = {
        "s1": {"dataset": "LongBench", "task_name": "narrativeqa"},
        "s2": {"dataset": "LongBench", "task_name": "narrativeqa"},
        "s3": {"dataset": "LongBench", "task_name": "qasper"},
    }

    tables = aggregate_profiling_records(records, sample_metadata_by_id=sample_metadata_by_id)

    assert tables["stability_by_task"] == [
        {
            "dataset": "LongBench",
            "task_name": "narrativeqa",
            "method": "remove_layer_head_chunk",
            "sample_count": 2,
            "top_k": 1,
            "pair_count": 1,
            "mean_topk_jaccard": 1.0,
        },
        {
            "dataset": "LongBench",
            "task_name": "narrativeqa",
            "method": "remove_layer_head_chunk",
            "sample_count": 2,
            "top_k": 2,
            "pair_count": 1,
            "mean_topk_jaccard": 0.333333,
        },
    ]


def test_aggregate_profiling_records_reports_addition_removal_alignment():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.7),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.4),
        ),
        KV3DProfilingRecord(
            sample_id="s1",
            method="add_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
            selected_kv_bytes=10,
            metric=KV3DMetricSnapshot(f1=0.4),
            delta_vs_base=KV3DMetricSnapshot(f1=0.3),
        ),
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=1, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.8),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.2),
        ),
        KV3DProfilingRecord(
            sample_id="s1",
            method="add_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=1, chunk=0),
            selected_kv_bytes=10,
            metric=KV3DMetricSnapshot(f1=0.2),
            delta_vs_base=KV3DMetricSnapshot(f1=0.1),
        ),
    ]
    sample_metadata_by_id = {"s1": {"dataset": "LongBench", "task_name": "narrativeqa"}}

    tables = aggregate_profiling_records(records, sample_metadata_by_id=sample_metadata_by_id)

    assert tables["addition_removal_alignment"] == [
        {
            "dataset": "LongBench",
            "task_name": "narrativeqa",
            "method_pair": "remove_layer_head_chunk:add_layer_head_chunk",
            "sample_count": 1,
            "block_count": 2,
            "pearson_utility": 1.0,
            "top_1_jaccard": 1.0,
            "top_2_jaccard": 1.0,
        }
    ]


def test_aggregate_profiling_records_reports_axis_heterogeneity():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.8),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.1),
        ),
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s1", layer=1, head=0, chunk=1),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.6),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.3),
        ),
        KV3DProfilingRecord(
            sample_id="s2",
            method="remove_layer_head_chunk",
            key=KV3DKey(sample_id="s2", layer=0, head=1, chunk=0),
            selected_kv_bytes=100,
            metric=KV3DMetricSnapshot(f1=0.7),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.2),
        ),
    ]

    tables = aggregate_profiling_records(records)

    heterogeneity = {(row["method"], row["axis"]): row for row in tables["heterogeneity"]}
    layer_row = heterogeneity[("remove_layer_head_chunk", "layer")]
    assert layer_row["axis_value_count"] == 2
    assert layer_row["min_mean_utility"] == 0.15
    assert layer_row["max_mean_utility"] == 0.3
    assert layer_row["range_mean_utility"] == 0.15
    assert layer_row["std_mean_utility"] == 0.075
    assert layer_row["cv_mean_utility"] == 0.333333
    assert layer_row["top_axis_value"] == 1
    assert layer_row["bottom_axis_value"] == 0
