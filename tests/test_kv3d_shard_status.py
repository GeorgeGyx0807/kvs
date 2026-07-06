import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_kv3d_shard_status_reports_complete_missing_and_failed(tmp_path):
    spec_path = tmp_path / "spec.json"
    shard_ok = tmp_path / "run_shard_000"
    shard_missing = tmp_path / "run_shard_001"
    shard_failed = tmp_path / "run_shard_002"
    output_path = tmp_path / "status.json"
    _write_json(
        spec_path,
        {
            "commands": {
                "run_shards": [
                    f"python3 scripts/run_kv3d_gpu_profile.py --output-dir {shard_ok}",
                    f"python3 scripts/run_kv3d_gpu_profile.py --output-dir {shard_missing}",
                    f"python3 scripts/run_kv3d_gpu_profile.py --output-dir {shard_failed}",
                ]
            },
            "sharding": {"shard_count": 3},
        },
    )
    shard_ok.mkdir()
    _write_json(shard_ok / "validation.json", {"ok": True, "record_count": 10, "issues": []})
    shard_failed.mkdir()
    _write_json(shard_failed / "validation.json", {"ok": False, "record_count": 4, "issues": ["missing required method: add_layer_head_chunk"]})

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts/kv3d_shard_status.py")),
            "--spec",
            str(spec_path),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    payload = json.loads(output_path.read_text())
    assert payload["summary"] == {
        "total": 3,
        "complete": 1,
        "missing": 1,
        "failed": 1,
        "pending_validation": 0,
    }
    statuses = {item["shard_dir"]: item for item in payload["shards"]}
    assert statuses[str(shard_ok)]["status"] == "complete"
    assert statuses[str(shard_missing)]["status"] == "missing"
    assert statuses[str(shard_failed)]["status"] == "failed"
    assert statuses[str(shard_failed)]["issues"] == ["missing required method: add_layer_head_chunk"]
