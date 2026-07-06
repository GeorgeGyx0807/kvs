import json

from src.kv3d import KV3DKey, KV3DMetricSnapshot, KV3DProfilingRecord, ProfilingManifest
from src.kv3d.io import write_manifest, write_records, write_summary


def test_kv3d_io_writes_manifest_records_and_summary(tmp_path):
    manifest = ProfilingManifest(
        experiment_name="offline_3d_kv_utility_profiling",
        model_name="Qwen3-8B",
        agent_pair="same_checkpoint",
        main_dataset="LongBench",
        auxiliary_dataset="RULER",
        baseline="full-KV",
    )
    records = [
        KV3DProfilingRecord(
            sample_id="s1",
            method="remove_block",
            key=KV3DKey(sample_id="s1", layer=1, head=2, chunk=3),
            selected_kv_bytes=128,
            metric=KV3DMetricSnapshot(accuracy=0.5),
        )
    ]

    manifest_path = tmp_path / "manifest.json"
    records_path = tmp_path / "records.jsonl"
    summary_path = tmp_path / "summary.json"

    write_manifest(manifest, manifest_path)
    write_records(records, records_path)
    write_summary({"record_count": 1}, summary_path)

    assert json.loads(manifest_path.read_text())["model_name"] == "Qwen3-8B"
    record_payload = json.loads(records_path.read_text().strip())
    assert record_payload["key"]["layer"] == 1
    assert json.loads(summary_path.read_text())["record_count"] == 1
