import json
import subprocess
import sys
from pathlib import Path

from src.kv3d import KV3DKey, KV3DMetricSnapshot, KV3DProfilingRecord
from src.kv3d.io import write_records


def test_analyze_kv3d_profile_cli_reads_records_jsonl(tmp_path):
    records_path = tmp_path / "records.jsonl"
    output_path = tmp_path / "tables.json"
    write_records(
        [
            KV3DProfilingRecord(
                sample_id="s1",
                method="full_kv",
                key=None,
                selected_kv_bytes=100,
                metric=KV3DMetricSnapshot(accuracy=1.0, nll=1.0),
            ),
            KV3DProfilingRecord(
                sample_id="s1",
                method="remove_layer_head_chunk",
                key=KV3DKey(sample_id="s1", layer=0, head=1, chunk=2),
                selected_kv_bytes=50,
                metric=KV3DMetricSnapshot(accuracy=0.5, nll=1.5),
                delta_vs_full=KV3DMetricSnapshot(accuracy=-0.5, nll=0.5),
            ),
        ],
        records_path,
    )

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts/analyze_kv3d_profile.py")),
            "--input",
            str(records_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    tables = json.loads(output_path.read_text())
    assert tables["layer_by_method"][0]["method"] == "remove_layer_head_chunk"
    assert tables["budget_curves"][0]["method"] == "full_kv"
