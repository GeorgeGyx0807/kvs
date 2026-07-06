import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_shard(path: Path, sample_id: str, layer: int):
    path.mkdir()
    _write_json(
        path / "samples.json",
        [
            {
                "sample_id": sample_id,
                "dataset": "LongBench",
                "task_name": "qa",
                "prompt": f"prompt-{sample_id}",
                "gold_answer": "answer",
            }
        ],
    )
    _write_json(
        path / "manifest.json",
        {
            "experiment_name": "offline_3d_kv_utility_profiling",
            "model_name": "Qwen3-8B",
            "agent_pair": "same_checkpoint",
            "main_dataset": "LongBench",
            "auxiliary_dataset": "RULER",
            "baseline": "full-KV",
        },
    )
    _write_jsonl(
        path / "records.jsonl",
        [
            {
                "sample_id": sample_id,
                "method": "full_kv",
                "key": None,
                "selected_kv_bytes": 100,
                "metric": {"f1": 1.0, "nll": 1.0},
            },
            {
                "sample_id": sample_id,
                "method": "b_only",
                "key": None,
                "selected_kv_bytes": 0,
                "metric": {"f1": 0.0, "nll": 2.0},
                "delta_vs_full": {"f1": -1.0, "nll": 1.0},
            },
            {
                "sample_id": sample_id,
                "method": "remove_layer",
                "key": {"sample_id": sample_id, "layer": layer, "head": 0, "chunk": 0},
                "selected_kv_bytes": 90,
                "metric": {"f1": 0.6, "nll": 1.4},
                "delta_vs_full": {"f1": -0.4, "nll": 0.4},
            },
            {
                "sample_id": sample_id,
                "method": "remove_layer_head",
                "key": {"sample_id": sample_id, "layer": layer, "head": 0, "chunk": 0},
                "selected_kv_bytes": 95,
                "metric": {"f1": 0.7, "nll": 1.3},
                "delta_vs_full": {"f1": -0.3, "nll": 0.3},
            },
            {
                "sample_id": sample_id,
                "method": "remove_layer_head_chunk",
                "key": {"sample_id": sample_id, "layer": layer, "head": 0, "chunk": 0},
                "selected_kv_bytes": 99,
                "metric": {"f1": 0.8, "nll": 1.2},
                "delta_vs_full": {"f1": -0.2, "nll": 0.2},
            },
            {
                "sample_id": sample_id,
                "method": "add_layer_head_chunk",
                "key": {"sample_id": sample_id, "layer": layer, "head": 0, "chunk": 0},
                "selected_kv_bytes": 1,
                "metric": {"f1": 0.2, "nll": 1.8},
                "delta_vs_full": {"f1": -0.8, "nll": 0.8},
                "delta_vs_base": {"f1": 0.2, "nll": -0.2},
            },
        ],
    )


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_merge_kv3d_shards_cli_writes_combined_artifacts(tmp_path):
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    output_dir = tmp_path / "merged"
    _write_shard(shard_a, "s1", 0)
    _write_shard(shard_b, "s2", 1)

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts/merge_kv3d_shards.py")),
            "--shard-dirs",
            str(shard_a),
            str(shard_b),
            "--num-layers",
            "2",
            "--num-heads",
            "1",
            "--num-chunks",
            "1",
            "--include-addition",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text())
    assert summary["shard_count"] == 2
    assert summary["sample_count"] == 2
    assert summary["record_count"] == 12
    assert summary["plan_size"] == 20
    assert (output_dir / "raw_results.jsonl").exists()
    assert (output_dir / "block_utility.csv").exists()
    assert (output_dir / "figures" / "fig_stability.png").exists()


def test_merge_kv3d_shards_deduplicates_identical_records(tmp_path):
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    output_dir = tmp_path / "merged"
    _write_shard(shard_a, "s1", 0)
    _write_shard(shard_b, "s1", 0)

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts/merge_kv3d_shards.py")),
            "--shard-dirs",
            str(shard_a),
            str(shard_b),
            "--num-layers",
            "1",
            "--num-heads",
            "1",
            "--num-chunks",
            "1",
            "--include-addition",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    assert len(_load_jsonl(output_dir / "records.jsonl")) == 6


def test_merge_kv3d_shards_rejects_conflicting_duplicate_records(tmp_path):
    shard_a = tmp_path / "shard_a"
    shard_b = tmp_path / "shard_b"
    output_dir = tmp_path / "merged"
    _write_shard(shard_a, "s1", 0)
    _write_shard(shard_b, "s1", 0)
    records = _load_jsonl(shard_b / "records.jsonl")
    records[0]["metric"]["f1"] = 0.5
    _write_jsonl(shard_b / "records.jsonl", records)

    completed = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/merge_kv3d_shards.py")),
            "--shard-dirs",
            str(shard_a),
            str(shard_b),
            "--num-layers",
            "1",
            "--num-heads",
            "1",
            "--num-chunks",
            "1",
            "--include-addition",
            "--output-dir",
            str(output_dir),
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "conflicting duplicate record" in completed.stderr
