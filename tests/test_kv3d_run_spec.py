import json
import subprocess
import sys
from pathlib import Path

from src.kv3d.run_spec import build_kv3d_run_spec


def test_build_kv3d_run_spec_computes_plan_size_and_commands():
    spec = build_kv3d_run_spec(
        profile_name="pilot",
        dataset_name="THUDM/LongBench",
        config_name="narrativeqa",
        split="test",
        model_name="Qwen/Qwen3-8B",
        output_dir="outputs/kv3d_pilot",
        max_samples=20,
        num_layers=2,
        num_heads=2,
        num_chunks=4,
        chunk_size=128,
        max_context_tokens=512,
        max_new_tokens=64,
        include_addition=True,
        min_samples_gate=20,
        min_heterogeneity_range=0.001,
        shard_size=5,
    )

    assert spec["plan_size"] == 800
    assert spec["per_sample_plan_size"] == 40
    assert spec["validation_gate"]["min_samples"] == 20
    assert "--include-addition" in spec["commands"]["run_gpu_profile"]
    assert len(spec["commands"]["run_shards"]) == 4
    assert "--sample-offset 5 --max-samples 5" in spec["commands"]["run_shards"][1]
    assert len(spec["commands"]["validate_shards"]) == 4
    assert "--run-dir outputs/kv3d_pilot_shard_001" in spec["commands"]["validate_shards"][1]
    assert "--output outputs/kv3d_pilot_shard_001/validation.json" in spec["commands"]["validate_shards"][1]
    assert "--min-samples 5" in spec["commands"]["validate_shards"][1]
    assert "scripts/merge_kv3d_shards.py" in spec["commands"]["merge_shards"]
    assert "outputs/kv3d_pilot_shard_003" in spec["commands"]["merge_shards"]
    assert "scripts/kv3d_shard_status.py" in spec["commands"]["shard_status"]
    assert "--min-samples 20" in spec["commands"]["validate"]
    assert "--min-heterogeneity-range 0.001" in spec["commands"]["validate"]


def test_build_kv3d_run_spec_cli_writes_json(tmp_path):
    output_path = tmp_path / "spec.json"

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts/build_kv3d_run_spec.py")),
            "--profile-name",
            "pilot",
            "--dataset-name",
            "THUDM/LongBench",
            "--config-name",
            "narrativeqa",
            "--split",
            "test",
            "--model-name",
            "Qwen/Qwen3-8B",
            "--output-dir",
            "outputs/kv3d_pilot",
            "--max-samples",
            "20",
            "--num-layers",
            "2",
            "--num-heads",
            "2",
            "--num-chunks",
            "4",
            "--chunk-size",
            "128",
            "--max-context-tokens",
            "512",
            "--max-new-tokens",
            "64",
            "--include-addition",
            "--shard-size",
            "5",
            "--min-samples-gate",
            "20",
            "--min-heterogeneity-range",
            "0.001",
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    payload = json.loads(output_path.read_text())
    assert payload["plan_size"] == 800
    assert payload["profile_name"] == "pilot"
    assert payload["dimensions"]["max_new_tokens"] == 64
    assert len(payload["commands"]["run_shards"]) == 4
    assert len(payload["commands"]["validate_shards"]) == 4
    assert "merge_shards" in payload["commands"]
    assert "shard_status" in payload["commands"]
