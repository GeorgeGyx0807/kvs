from src.kv3d import KV3DKey, KV3DMetricSnapshot, KV3DProfilingRecord
from src.kv3d.stage_selection import build_chunk_targets


def test_build_chunk_targets_prefers_high_utility_layer_heads():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head",
            key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
            selected_kv_bytes=1,
            metric=KV3DMetricSnapshot(f1=0.1),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.9),
        ),
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head",
            key=KV3DKey(sample_id="s1", layer=1, head=1, chunk=0),
            selected_kv_bytes=1,
            metric=KV3DMetricSnapshot(f1=0.9),
            delta_vs_full=KV3DMetricSnapshot(f1=-0.1),
        ),
    ]

    targets = build_chunk_targets(records, top_k_per_sample=1, middle_k_per_sample=0, low_k_per_sample=0)

    assert targets == {"s1": [[0, 0]]}


def test_build_chunk_targets_includes_top_middle_and_low_bands():
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_layer_head",
            key=KV3DKey(sample_id="s1", layer=index, head=0, chunk=0),
            selected_kv_bytes=1,
            metric=KV3DMetricSnapshot(f1=1.0 - index * 0.1),
            delta_vs_full=KV3DMetricSnapshot(f1=-(1.0 - index * 0.1)),
        )
        for index in range(6)
    ]

    targets = build_chunk_targets(records, top_k_per_sample=1, middle_k_per_sample=1, low_k_per_sample=1)

    assert targets == {"s1": [[0, 0], [3, 0], [5, 0]]}
