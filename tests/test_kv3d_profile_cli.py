import json
import subprocess
import sys
from pathlib import Path

from src.kv3d import KV3DKey, KV3DMetricSnapshot, KV3DProfilingRecord
from src.kv3d.io import write_records


def test_run_offline_3d_profile_cli_roundtrip_with_records(tmp_path):
    samples_path = tmp_path / "samples.json"
    samples_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "s1",
                    "dataset": "LongBench",
                    "task_name": "qa",
                    "prompt": "p1",
                    "gold_answer": "a1",
                }
            ]
        )
    )
    records_path = tmp_path / "records.jsonl"
    write_records(
        [
            KV3DProfilingRecord(
                sample_id="s1",
                method="remove_layer",
                key=KV3DKey(sample_id="s1", layer=0, head=0, chunk=0),
                selected_kv_bytes=10,
                metric=KV3DMetricSnapshot(accuracy=0.5),
            )
        ],
        records_path,
    )

    output_dir = tmp_path / "out"
    cmd = [
        sys.executable,
        str(Path("scripts/run_offline_3d_profile.py")),
        "--samples",
        str(samples_path),
        "--records",
        str(records_path),
        "--num-layers",
        "1",
        "--num-heads",
        "1",
        "--num-chunks",
        "1",
        "--output-dir",
        str(output_dir),
        "--config-name",
        "longbench_qa",
        "--chunk-size",
        "128",
        "--max-context-tokens",
        "300",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["record_count"] == 1
    output_record = json.loads((output_dir / "records.jsonl").read_text().strip())
    assert output_record["key"]["token_start"] == 0
    assert output_record["key"]["token_end"] == 128
    tables = json.loads((output_dir / "tables.json").read_text())
    assert tables["block_utility"][0]["token_start"] == 0
    assert tables["block_utility"][0]["token_end"] == 128
    assert (output_dir / "raw_results.jsonl").exists()
    assert (output_dir / "layer_importance.csv").exists()
    assert (output_dir / "head_importance.csv").exists()
    assert (output_dir / "chunk_importance.csv").exists()
    assert (output_dir / "block_utility.csv").exists()
    assert (output_dir / "attention_divergence.csv").exists()
    assert (output_dir / "attention_divergence_by_method.csv").exists()
    assert (output_dir / "attention_correlation_matrix.csv").exists()
    assert (output_dir / "stability.csv").exists()
    assert (output_dir / "stability_by_task.csv").exists()
    assert (output_dir / "heterogeneity.csv").exists()
    assert (output_dir / "addition_removal_alignment.csv").exists()
    assert (output_dir / "budget_curves.csv").exists()
    assert (output_dir / "figures" / "fig_layer_importance.png").exists()
    assert (output_dir / "figures" / "fig_layer_head_heatmap.png").exists()
    assert (output_dir / "figures" / "fig_attention_divergence.png").exists()
    assert (output_dir / "tables.json").exists()
    assert (output_dir / "key_findings.json").exists()
    assert (output_dir / "report.md").exists()
