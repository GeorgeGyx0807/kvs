#!/usr/bin/env python3
"""Merge offline 3D KV profiling shard directories into one analyzed run."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d.io import read_records, write_profile_artifacts
from src.kv3d.runner import ProfilingSample, run_offline_3d_profile
from src.kv3d.report import render_profiling_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--num-chunks", type=int, required=True)
    parser.add_argument("--span-size", type=int, default=0)
    parser.add_argument("--num-spans", type=int, default=0)
    parser.add_argument("--include-addition", action="store_true")
    parser.add_argument("--include-chunk-level", dest="include_chunk_level", action="store_true", default=True)
    parser.add_argument("--no-include-chunk-level", dest="include_chunk_level", action="store_false")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _load_samples(path: Path) -> list[ProfilingSample]:
    payload = json.loads((path / "samples.json").read_text())
    return [ProfilingSample(**item) for item in payload]


def _model_name(shard_dirs: list[Path], override: str) -> str:
    if override:
        return override
    for shard_dir in shard_dirs:
        manifest_path = shard_dir / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("model_name"):
                return str(manifest["model_name"])
    return "Qwen3-8B"


def _record_identity(record: dict[str, object]) -> tuple[object, ...]:
    key = record.get("key")
    if isinstance(key, dict):
        block = (key.get("layer"), key.get("head"), key.get("chunk"))
    else:
        block = (None, None, None)
    return (record.get("sample_id"), record.get("method"), *block)


def _profile_identity(item: dict[str, object]) -> tuple[object, ...]:
    key = item.get("key")
    if isinstance(key, dict):
        layer = key.get("layer")
        head = key.get("head")
        chunk = key.get("chunk")
    else:
        layer = item.get("layer")
        head = item.get("head")
        chunk = item.get("chunk")
    method = item.get("method")
    if method in {"full_kv", "b_only"}:
        layer, head, chunk = None, None, None
    elif method == "remove_layer":
        head, chunk = None, None
    elif method == "remove_layer_head":
        chunk = None
    return (item.get("sample_id"), method, layer, head, chunk)


def _canonical(record: dict[str, object]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def _deduplicate_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_identity: dict[tuple[object, ...], dict[str, object]] = {}
    canonical_by_identity: dict[tuple[object, ...], str] = {}
    for record in records:
        identity = _record_identity(record)
        canonical = _canonical(record)
        if identity in canonical_by_identity:
            if canonical_by_identity[identity] != canonical:
                raise ValueError(f"conflicting duplicate record: {identity}")
            continue
        by_identity[identity] = record
        canonical_by_identity[identity] = canonical
    return [by_identity[identity] for identity in sorted(by_identity, key=lambda item: tuple(str(part) for part in item))]


def _deduplicate_plan(plan: list[dict[str, object]]) -> list[dict[str, object]]:
    by_identity: dict[tuple[object, ...], dict[str, object]] = {}
    canonical_by_identity: dict[tuple[object, ...], str] = {}
    for item in plan:
        identity = _profile_identity(item)
        canonical = _canonical(item)
        if identity in canonical_by_identity:
            if canonical_by_identity[identity] != canonical:
                raise ValueError(f"conflicting duplicate plan item: {identity}")
            continue
        by_identity[identity] = item
        canonical_by_identity[identity] = canonical
    return [by_identity[identity] for identity in sorted(by_identity, key=lambda item: tuple(str(part) for part in item))]


def main() -> None:
    args = parse_args()
    samples_by_id: dict[str, ProfilingSample] = {}
    records: list[dict[str, object]] = []
    plan: list[dict[str, object]] = []
    for shard_dir in args.shard_dirs:
        for sample in _load_samples(shard_dir):
            samples_by_id.setdefault(sample.sample_id, sample)
        records.extend(read_records(shard_dir / "records.jsonl"))
        plan_path = shard_dir / "plan.json"
        if plan_path.exists():
            plan.extend(json.loads(plan_path.read_text()))
    try:
        records = _deduplicate_records(records)
        if plan:
            plan = _deduplicate_plan(plan)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    samples = [samples_by_id[sample_id] for sample_id in sorted(samples_by_id)]
    result = run_offline_3d_profile(
        samples=samples,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_chunks=args.num_spans or args.num_chunks,
        include_addition=args.include_addition,
        include_chunk_level=args.include_chunk_level,
        records=records,
        model_name=_model_name(args.shard_dirs, args.model_name),
    )
    if plan:
        result["plan"] = plan
        result["report"] = render_profiling_report(
            plan=plan,
            records=result["records"],
            tables=result["tables"],
            experiment_name=result["manifest"]["experiment_name"],
            model_name=result["manifest"]["model_name"],
            main_dataset=result["manifest"]["main_dataset"],
        )
    attention_enabled = any(record.get("attention_divergence") is not None for record in result["records"])
    write_profile_artifacts(
        result=result,
        output_dir=args.output_dir,
        summary={
            "record_count": len(result["records"]),
            "plan_size": len(result["plan"]),
            "sample_count": len(result["samples"]),
            "shard_count": len(args.shard_dirs),
            "source_shards": [str(path) for path in args.shard_dirs],
            "include_addition": args.include_addition,
            "include_chunk_level": args.include_chunk_level,
            "attention_divergence_enabled": attention_enabled,
            "span_size": args.span_size or None,
            "num_spans": (args.num_spans or args.num_chunks) if args.span_size else None,
            "profiling_mode": "two_stage_layer_head_then_span" if args.span_size else "chunk",
        },
    )


if __name__ == "__main__":
    main()
