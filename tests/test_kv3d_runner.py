from src.kv3d import KV3DKey, KV3DMetricSnapshot, KV3DProfilingRecord
from src.kv3d.runner import ProfilingSample, run_offline_3d_profile


def test_run_offline_3d_profile_produces_manifest_records_and_tables():
    samples = [
        ProfilingSample(
            sample_id="s1",
            dataset="LongBench",
            task_name="qa",
            prompt="p1",
            gold_answer="a1",
        )
    ]

    result = run_offline_3d_profile(
        samples=samples,
        num_layers=1,
        num_heads=1,
        num_chunks=1,
        include_addition=True,
    )

    assert result["manifest"]["main_dataset"] == "LongBench"
    assert result["manifest"]["baseline"] == "full-KV"
    assert len(result["plan"]) == 6
    assert result["records"] == []
    assert result["tables"]["layer"] == []


def test_run_offline_3d_profile_accepts_external_records():
    samples = [
        ProfilingSample(
            sample_id="s1",
            dataset="LongBench",
            task_name="narrativeqa",
            prompt="p1",
            gold_answer="a1",
        )
    ]
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer",
            key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
            selected_kv_bytes=10,
            metric=KV3DMetricSnapshot(accuracy=0.5, nll=1.0, ttft_ms=2.0),
            delta_vs_full=KV3DMetricSnapshot(accuracy=-0.1, nll=0.2, ttft_ms=0.3),
        )
    ]

    result = run_offline_3d_profile(
        samples=samples,
        num_layers=1,
        num_heads=1,
        num_chunks=1,
        records=records,
    )

    assert len(result["records"]) == 1
    assert result["tables"]["layer"][0]["record_count"] == 1
    assert result["tables"]["block_utility"][0]["dataset"] == "LongBench"
    assert result["tables"]["block_utility"][0]["task_name"] == "narrativeqa"


def test_run_offline_3d_profile_preserves_record_token_span_from_dict():
    samples = [
        ProfilingSample(
            sample_id="s1",
            dataset="LongBench",
            task_name="qa",
            prompt="p1",
            gold_answer="a1",
        )
    ]
    records = [
        {
            "sample_id": "s1",
            "method": "remove_layer_head_chunk",
            "key": {
                "sample_id": "s1",
                "layer": 0,
                "head": 1,
                "chunk": 2,
                "token_start": 256,
                "token_end": 384,
            },
            "selected_kv_bytes": 10,
            "metric": {"accuracy": 0.5, "nll": 1.0, "ttft_ms": 2.0},
            "delta_vs_full": {"accuracy": -0.1, "nll": 0.2, "ttft_ms": 0.3},
        }
    ]

    result = run_offline_3d_profile(
        samples=samples,
        num_layers=1,
        num_heads=2,
        num_chunks=3,
        records=records,
    )

    assert result["records"][0]["key"]["token_start"] == 256
    assert result["records"][0]["key"]["token_end"] == 384
    assert result["tables"]["block_utility"][0]["token_start"] == 256
    assert result["tables"]["block_utility"][0]["token_end"] == 384
