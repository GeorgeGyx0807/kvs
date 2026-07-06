import json
import subprocess
import sys
import csv
from pathlib import Path

from src.kv3d.run_validation import validate_kv3d_run


def _write_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _write_jsonl(path: Path, rows):
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _write_csv(path: Path, rows):
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metric(f1: float, *, nll: float = 1.0):
    return {
        "accuracy": 0.0,
        "contains": 0.0,
        "decode_ms": 2.0,
        "f1": f1,
        "nll": nll,
        "prefill_ms": 3.0,
        "ttft_ms": 5.0,
    }


def _delta(f1: float, *, nll: float = 0.1):
    return {
        "accuracy": 0.0,
        "contains": 0.0,
        "decode_ms": 0.2,
        "f1": f1,
        "nll": nll,
        "prefill_ms": 0.3,
        "ttft_ms": 0.5,
    }


def _write_minimal_valid_run(path: Path):
    path.mkdir()
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
    _write_json(path / "summary.json", {"record_count": 6, "plan_size": 6})
    _write_json(path / "samples.json", [{"sample_id": "s1", "dataset": "LongBench", "task_name": "qa", "prompt": "p"}])
    _write_json(
        path / "plan.json",
        [
            {"sample_id": "s1", "method": "full_kv"},
            {"sample_id": "s1", "method": "b_only"},
            {"sample_id": "s1", "method": "remove_layer", "layer": 0},
            {"sample_id": "s1", "method": "remove_layer_head", "layer": 0, "head": 0},
            {"sample_id": "s1", "method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0},
            {"sample_id": "s1", "method": "add_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0},
        ],
    )
    records = [
        {"sample_id": "s1", "method": "full_kv", "key": None, "selected_kv_bytes": 100, "metric": _metric(1.0)},
        {"sample_id": "s1", "method": "b_only", "key": None, "selected_kv_bytes": 0, "metric": _metric(0.0)},
        {
            "sample_id": "s1",
            "method": "remove_layer",
            "key": {"sample_id": "s1", "layer": 0, "head": 0, "chunk": 0, "token_start": 0, "token_end": 128},
            "selected_kv_bytes": 90,
            "metric": _metric(0.5),
            "delta_vs_full": _delta(-0.5),
        },
        {
            "sample_id": "s1",
            "method": "remove_layer_head",
            "key": {"sample_id": "s1", "layer": 0, "head": 0, "chunk": 0, "token_start": 0, "token_end": 128},
            "selected_kv_bytes": 95,
            "metric": _metric(0.6),
            "delta_vs_full": _delta(-0.4),
        },
        {
            "sample_id": "s1",
            "method": "remove_layer_head_chunk",
            "key": {"sample_id": "s1", "layer": 0, "head": 0, "chunk": 0, "token_start": 0, "token_end": 128},
            "selected_kv_bytes": 99,
            "metric": _metric(0.7),
            "delta_vs_full": _delta(-0.3),
        },
        {
            "sample_id": "s1",
            "method": "add_layer_head_chunk",
            "key": {"sample_id": "s1", "layer": 0, "head": 0, "chunk": 0, "token_start": 0, "token_end": 128},
            "selected_kv_bytes": 1,
            "metric": _metric(0.2),
            "delta_vs_base": _delta(0.2),
        },
    ]
    _write_jsonl(path / "records.jsonl", records)
    _write_jsonl(path / "raw_results.jsonl", records)
    tables = {
        "layer_by_method": [{"method": "remove_layer", "layer": 0}],
        "head_by_method": [{"method": "remove_layer_head", "head": 0}],
        "chunk_by_method": [{"method": "remove_layer_head_chunk", "chunk": 0}],
        "block_utility": [
                {"sample_id": "s1", "dataset": "LongBench", "task_name": "qa", "method": "remove_layer", "layer": 0, "head": 0, "chunk": 0},
                {"sample_id": "s1", "dataset": "LongBench", "task_name": "qa", "method": "remove_layer_head", "layer": 0, "head": 0, "chunk": 0},
                {"sample_id": "s1", "dataset": "LongBench", "task_name": "qa", "method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0},
                {"sample_id": "s1", "dataset": "LongBench", "task_name": "qa", "method": "add_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0},
        ],
        "stability": [{"method": "remove_layer_head_chunk", "top_k": 1, "mean_topk_jaccard": 1.0}],
        "stability_by_task": [
            {
                "dataset": "LongBench",
                "task_name": "qa",
                "method": "remove_layer_head_chunk",
                "top_k": 1,
                "mean_topk_jaccard": 1.0,
            }
        ],
        "heterogeneity": [
            {
                "method": "remove_layer_head_chunk",
                "axis": "chunk",
                "range_mean_utility": 0.1,
                "axis_value_count": 2,
            }
        ],
        "addition_removal_alignment": [
            {
                "dataset": "LongBench",
                "task_name": "qa",
                "method_pair": "remove_layer_head_chunk:add_layer_head_chunk",
                "block_count": 2,
            }
        ],
        "attention_correlation_matrix": [],
        "budget_curves": [{"method": "full_kv"}],
    }
    _write_json(path / "tables.json", tables)
    _write_json(
        path / "key_findings.json",
        {
            "top_layer": {"method": "remove_layer", "layer": 0},
            "top_head": {"method": "remove_layer_head", "head": 0},
            "top_chunk": {"method": "remove_layer_head_chunk", "chunk": 0},
            "top_block": {"sample_id": "s1", "method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0},
            "stability": {"method": "remove_layer_head_chunk", "top_k": 1},
            "task_stability": {"dataset": "LongBench", "task_name": "qa", "method": "remove_layer_head_chunk", "top_k": 1},
            "heterogeneity": {"method": "remove_layer_head_chunk", "axis": "chunk", "range_mean_utility": 0.1},
            "addition_removal_alignment": {
                "dataset": "LongBench",
                "task_name": "qa",
                "method_pair": "remove_layer_head_chunk:add_layer_head_chunk",
                "block_count": 2,
            },
        },
    )
    _write_csv(path / "layer_importance.csv", tables["layer_by_method"])
    _write_csv(path / "head_importance.csv", tables["head_by_method"])
    _write_csv(path / "chunk_importance.csv", tables["chunk_by_method"])
    _write_csv(path / "block_utility.csv", tables["block_utility"])
    _write_csv(path / "stability.csv", tables["stability"])
    _write_csv(path / "stability_by_task.csv", tables["stability_by_task"])
    _write_csv(path / "heterogeneity.csv", tables["heterogeneity"])
    _write_csv(path / "addition_removal_alignment.csv", tables["addition_removal_alignment"])
    _write_csv(path / "attention_correlation_matrix.csv", tables["attention_correlation_matrix"])
    _write_csv(path / "budget_curves.csv", tables["budget_curves"])
    (path / "report.md").write_text("# report\n")
    figures = path / "figures"
    figures.mkdir()
    for name in [
        "fig_layer_importance.png",
        "fig_layer_head_heatmap.png",
        "fig_chunk_position_heatmap.png",
        "fig_budget_quality_curve.png",
        "fig_budget_nll_curve.png",
        "fig_latency_bytes_curve.png",
        "fig_stability.png",
    ]:
        (figures / name).write_bytes(b"png")


def test_validate_kv3d_run_accepts_complete_profile_directory(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)

    result = validate_kv3d_run(run_dir)

    assert result.ok is True
    assert result.record_count == 6
    assert result.methods["add_layer_head_chunk"] == 1
    assert result.issues == []


def test_validate_kv3d_run_rejects_missing_addition_and_figures(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["include_addition"] = True
    _write_json(run_dir / "summary.json", summary)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    records = [record for record in records if record["method"] != "add_layer_head_chunk"]
    _write_jsonl(run_dir / "records.jsonl", records)
    (run_dir / "figures" / "fig_stability.png").unlink()

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "missing required method: add_layer_head_chunk" in result.issues
    assert "missing required file: figures/fig_stability.png" in result.issues


def test_validate_kv3d_run_rejects_sample_missing_base_baseline(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    records = [record for record in records if record["method"] != "b_only"]
    _write_jsonl(run_dir / "records.jsonl", records)
    _write_json(run_dir / "summary.json", {"record_count": len(records), "plan_size": 6})

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "sample missing baseline method b_only: s1" in result.issues


def test_validate_kv3d_run_rejects_chunk_records_without_token_span(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    for record in records:
        if record["method"] == "remove_layer_head_chunk":
            record["key"].pop("token_start")
            record["key"].pop("token_end")
    _write_jsonl(run_dir / "records.jsonl", records)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "chunk-level record has no token_start/token_end: remove_layer_head_chunk" in result.issues


def test_validate_kv3d_run_requires_span_artifacts_and_fields_for_span_mode(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary.update({"span_size": 16, "num_spans": 1, "profiling_mode": "two_stage_layer_head_then_span"})
    _write_json(run_dir / "summary.json", summary)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    for record in records:
        key = record.get("key")
        if record["method"] in {"remove_layer_head_chunk", "add_layer_head_chunk"} and isinstance(key, dict):
            key.update({"kv_head": key["head"], "span_id": key["chunk"], "span_start": 0, "span_end": 128, "span_size": 128})
    _write_jsonl(run_dir / "records.jsonl", records)
    _write_jsonl(run_dir / "raw_results.jsonl", records)
    tables = json.loads((run_dir / "tables.json").read_text())
    tables["span_by_method"] = [
        {"method": "remove_layer_head_chunk", "span_id": 0, "span_start": 0, "span_end": 128, "span_size": 128}
    ]
    _write_json(run_dir / "tables.json", tables)
    _write_csv(run_dir / "span_importance.csv", tables["span_by_method"])
    key_findings = json.loads((run_dir / "key_findings.json").read_text())
    key_findings["top_span"] = {"method": "remove_layer_head_chunk", "span_id": 0, "span_start": 0, "span_end": 128}
    _write_json(run_dir / "key_findings.json", key_findings)
    for name in ["fig_layer_head_span_heatmap.png", "fig_span_position_curve.png", "fig_top_span_stability.png"]:
        (run_dir / "figures" / name).write_bytes(b"png")

    result = validate_kv3d_run(run_dir)

    assert result.ok is True
    assert result.table_rows["span_by_method"] == 1


def test_validate_kv3d_run_rejects_span_record_without_span_fields(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary.update({"span_size": 16, "num_spans": 1, "profiling_mode": "two_stage_layer_head_then_span"})
    _write_json(run_dir / "summary.json", summary)
    tables = json.loads((run_dir / "tables.json").read_text())
    tables["span_by_method"] = [
        {"method": "remove_layer_head_chunk", "span_id": 0, "span_start": 0, "span_end": 128, "span_size": 128}
    ]
    _write_json(run_dir / "tables.json", tables)
    _write_csv(run_dir / "span_importance.csv", tables["span_by_method"])
    key_findings = json.loads((run_dir / "key_findings.json").read_text())
    key_findings["top_span"] = {"method": "remove_layer_head_chunk", "span_id": 0, "span_start": 0, "span_end": 128}
    _write_json(run_dir / "key_findings.json", key_findings)
    for name in ["fig_layer_head_span_heatmap.png", "fig_span_position_curve.png", "fig_top_span_stability.png"]:
        (run_dir / "figures" / name).write_bytes(b"png")

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "span-level record has no span_id/span_start/span_end/span_size: remove_layer_head_chunk" in result.issues


def test_validate_kv3d_run_rejects_record_without_kv_bytes(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    records[0].pop("selected_kv_bytes")
    _write_jsonl(run_dir / "records.jsonl", records)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "record has no selected_kv_bytes: full_kv" in result.issues


def test_validate_kv3d_run_rejects_record_without_timing_metric(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    records[0]["metric"].pop("prefill_ms")
    _write_jsonl(run_dir / "records.jsonl", records)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "record missing required metric prefill_ms: full_kv" in result.issues


def test_validate_kv3d_run_rejects_missing_attention_artifacts_when_enabled(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    _write_json(
        run_dir / "summary.json",
        {
            "record_count": 6,
            "plan_size": 6,
            "attention_divergence_enabled": True,
        },
    )

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "missing required file: attention_divergence.csv" in result.issues


def test_validate_kv3d_run_rejects_non_finite_numbers(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    records[0]["metric"]["nll"] = float("inf")
    _write_jsonl(run_dir / "records.jsonl", records)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "non-finite numeric value: records[0].metric.nll" in result.issues


def test_validate_kv3d_run_rejects_negative_attention_divergence(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    records[2]["attention_divergence"] = {"js_divergence": -0.1, "kl_divergence": 0.2}
    _write_jsonl(run_dir / "records.jsonl", records)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "attention divergence is negative: remove_layer.js_divergence" in result.issues


def test_validate_kv3d_run_rejects_removal_record_without_delta_vs_full(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    for record in records:
        if record["method"] == "remove_layer_head_chunk":
            record.pop("delta_vs_full")
    _write_jsonl(run_dir / "records.jsonl", records)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "removal record missing delta_vs_full: remove_layer_head_chunk" in result.issues


def test_validate_kv3d_run_rejects_addition_record_without_delta_vs_base(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    for record in records:
        if record["method"] == "add_layer_head_chunk":
            record.pop("delta_vs_base")
    _write_jsonl(run_dir / "records.jsonl", records)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "addition record missing delta_vs_base: add_layer_head_chunk" in result.issues


def test_validate_kv3d_run_rejects_missing_record_for_planned_block(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    records = [record for record in records if record["method"] != "remove_layer_head_chunk"]
    _write_jsonl(run_dir / "records.jsonl", records)
    _write_json(run_dir / "summary.json", {"record_count": len(records), "plan_size": 6})

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "missing record for planned profile: sample=s1 method=remove_layer_head_chunk layer=0 head=0 chunk=0" in result.issues


def test_validate_kv3d_run_rejects_unplanned_record(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    extra = dict(records[-1])
    extra["key"] = {"sample_id": "s1", "layer": 0, "head": 0, "chunk": 99, "token_start": 12672, "token_end": 12800}
    records.append(extra)
    _write_jsonl(run_dir / "records.jsonl", records)
    _write_json(run_dir / "summary.json", {"record_count": len(records), "plan_size": 6})

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "unexpected record not present in plan: sample=s1 method=add_layer_head_chunk layer=0 head=0 chunk=99" in result.issues


def test_validate_kv3d_run_rejects_duplicate_record_identity(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    records = [json.loads(line) for line in (run_dir / "records.jsonl").read_text().splitlines()]
    records.append(dict(records[-1]))
    _write_jsonl(run_dir / "records.jsonl", records)
    _write_json(run_dir / "summary.json", {"record_count": len(records), "plan_size": 6})

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "duplicate record identity: sample=s1 method=add_layer_head_chunk layer=0 head=0 chunk=0" in result.issues


def test_validate_kv3d_run_rejects_raw_results_that_do_not_match_records(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    raw_results = [json.loads(line) for line in (run_dir / "raw_results.jsonl").read_text().splitlines()]
    raw_results = [record for record in raw_results if record["method"] != "add_layer_head_chunk"]
    _write_jsonl(run_dir / "raw_results.jsonl", raw_results)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "raw_results.jsonl count 5 != records.jsonl count 6" in result.issues
    assert "raw_results.jsonl missing record identity: sample=s1 method=add_layer_head_chunk layer=0 head=0 chunk=0" in result.issues


def test_validate_kv3d_run_rejects_block_utility_table_missing_record_identity(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    tables = json.loads((run_dir / "tables.json").read_text())
    tables["block_utility"] = []
    _write_json(run_dir / "tables.json", tables)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "block_utility missing record identity: sample=s1 method=remove_layer layer=0 head=None chunk=None" in result.issues


def test_validate_kv3d_run_rejects_block_utility_without_task_metadata(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    tables = json.loads((run_dir / "tables.json").read_text())
    tables["block_utility"][0]["task_name"] = None
    _write_json(run_dir / "tables.json", tables)
    _write_csv(run_dir / "block_utility.csv", tables["block_utility"])

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "block_utility row missing dataset/task_name: sample=s1 method=remove_layer layer=0 head=None chunk=None" in result.issues


def test_validate_kv3d_run_rejects_csv_row_count_that_disagrees_with_tables(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    (run_dir / "block_utility.csv").write_text("header\n")

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "CSV row count 0 != tables.json block_utility rows 4: block_utility.csv" in result.issues


def test_validate_kv3d_run_rejects_csv_row_content_that_disagrees_with_tables(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    rows = [row for row in csv.DictReader((run_dir / "block_utility.csv").open())]
    rows[0]["method"] = "remove_block"
    _write_csv(run_dir / "block_utility.csv", rows)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "CSV row mismatch: block_utility.csv row 1" in result.issues


def test_validate_kv3d_run_rejects_incomplete_key_findings(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    key_findings = json.loads((run_dir / "key_findings.json").read_text())
    key_findings.pop("heterogeneity")
    _write_json(run_dir / "key_findings.json", key_findings)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "key_findings missing required section: heterogeneity" in result.issues


def test_validate_kv3d_run_rejects_key_findings_that_disagree_with_tables(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    key_findings = json.loads((run_dir / "key_findings.json").read_text())
    key_findings["top_layer"]["layer"] = 999
    _write_json(run_dir / "key_findings.json", key_findings)

    result = validate_kv3d_run(run_dir)

    assert result.ok is False
    assert "key_findings section disagrees with tables.json: top_layer" in result.issues


def test_validate_kv3d_run_cli_writes_json_summary(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    output_path = tmp_path / "validation.json"

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts/validate_kv3d_run.py")),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    payload = json.loads(output_path.read_text())
    assert payload["ok"] is True
    assert payload["record_count"] == 6


def test_validate_kv3d_run_evidence_gate_rejects_too_few_samples(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)

    result = validate_kv3d_run(run_dir, min_samples=2)

    assert result.ok is False
    assert "sample_count 1 < required 2" in result.issues


def test_validate_kv3d_run_evidence_gate_requires_stability_rows(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    tables = json.loads((run_dir / "tables.json").read_text())
    tables["stability"] = []
    _write_json(run_dir / "tables.json", tables)

    result = validate_kv3d_run(run_dir, require_stability=True)

    assert result.ok is False
    assert "stability evidence is required but stability table has no rows" in result.issues


def test_validate_kv3d_run_evidence_gate_requires_min_heterogeneity(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    tables = json.loads((run_dir / "tables.json").read_text())
    tables["heterogeneity"][0]["range_mean_utility"] = 0.0
    _write_json(run_dir / "tables.json", tables)

    result = validate_kv3d_run(run_dir, min_heterogeneity_range=0.05)

    assert result.ok is False
    assert "max heterogeneity range 0.0 < required 0.05" in result.issues


def test_validate_kv3d_run_cli_applies_evidence_gate(tmp_path):
    run_dir = tmp_path / "run"
    _write_minimal_valid_run(run_dir)
    output_path = tmp_path / "validation.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(Path("scripts/validate_kv3d_run.py")),
            "--run-dir",
            str(run_dir),
            "--output",
            str(output_path),
            "--min-samples",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
    )

    assert completed.returncode == 1
    payload = json.loads(output_path.read_text())
    assert payload["ok"] is False
    assert "sample_count 1 < required 2" in payload["issues"]
