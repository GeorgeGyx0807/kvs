#!/usr/bin/env python3
"""Inspect completion status for KV3D pilot shard runs."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _output_dir_from_command(command: str) -> str:
    parts = shlex.split(command)
    for index, part in enumerate(parts):
        if part == "--output-dir" and index + 1 < len(parts):
            return parts[index + 1]
    raise ValueError(f"shard command has no --output-dir: {command}")


def _read_validation(shard_dir: Path) -> dict[str, Any] | None:
    validation_path = shard_dir / "validation.json"
    if not validation_path.exists():
        return None
    return json.loads(validation_path.read_text())


def _status_for_shard(shard_dir: Path) -> dict[str, Any]:
    if not shard_dir.exists():
        return {
            "shard_dir": str(shard_dir),
            "status": "missing",
            "record_count": 0,
            "issues": [f"missing shard directory: {shard_dir}"],
        }
    validation = _read_validation(shard_dir)
    if validation is None:
        return {
            "shard_dir": str(shard_dir),
            "status": "pending_validation",
            "record_count": 0,
            "issues": [f"missing validation.json: {shard_dir}"],
        }
    ok = bool(validation.get("ok"))
    return {
        "shard_dir": str(shard_dir),
        "status": "complete" if ok else "failed",
        "record_count": int(validation.get("record_count", 0)),
        "issues": validation.get("issues", []),
    }


def build_shard_status(spec_path: Path) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text())
    commands = spec.get("commands", {}).get("run_shards", [])
    shards = [_status_for_shard(Path(_output_dir_from_command(command))) for command in commands]
    summary = {
        "total": len(shards),
        "complete": sum(1 for shard in shards if shard["status"] == "complete"),
        "missing": sum(1 for shard in shards if shard["status"] == "missing"),
        "failed": sum(1 for shard in shards if shard["status"] == "failed"),
        "pending_validation": sum(1 for shard in shards if shard["status"] == "pending_validation"),
    }
    return {"spec": str(spec_path), "summary": summary, "shards": shards}


def main() -> None:
    args = parse_args()
    payload = build_shard_status(args.spec)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
