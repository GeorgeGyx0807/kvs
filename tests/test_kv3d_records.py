from src.kv3d import KV3DKey, KV3DMetricSnapshot, KV3DProfilingRecord


def test_profiling_record_serializes_nested_metric_fields():
    record = KV3DProfilingRecord(
        sample_id="s1",
        method="add_layer_head_chunk",
        key=KV3DKey(sample_id="s1", layer=1, head=2, chunk=3, token_start=384, token_end=512),
        selected_kv_bytes=2048,
        metric=KV3DMetricSnapshot(accuracy=0.9, nll=1.2, ttft_ms=34.5),
        delta_vs_base=KV3DMetricSnapshot(accuracy=0.4, nll=-0.8, ttft_ms=10.0),
    )
    payload = record.to_dict()
    assert payload["key"]["chunk"] == 3
    assert payload["key"]["token_start"] == 384
    assert payload["key"]["token_end"] == 512
    assert payload["metric"]["nll"] == 1.2
    assert payload["delta_vs_base"]["accuracy"] == 0.4
