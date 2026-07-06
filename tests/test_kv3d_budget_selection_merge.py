import json
import subprocess
import sys
from pathlib import Path

from src.kv3d.budget_selection import BudgetEvaluationResult, write_jsonl_results


def _result(sample_id: str, method: str, budget: float, f1: float) -> BudgetEvaluationResult:
    return BudgetEvaluationResult(
        sample_id=sample_id,
        dataset="LongBench",
        task_name="narrativeqa",
        method=method,
        budget_ratio=budget,
        prediction="p",
        gold_answer="g",
        answers=["g"],
        f1=f1,
        contains=0.0,
        nll=2.0,
        delta_nll_vs_full=0.5,
        selected_kv_bytes=10,
        full_kv_bytes=100,
        kv_ratio=0.1,
        ttft_ms=1.0,
        prefill_ms=0.7,
        decode_ms=0.3,
        generation_length=1,
        selected_block_count=1,
    )


def test_merge_budget_selection_shards_writes_combined_outputs(tmp_path):
    shard_a = tmp_path / "a"
    shard_b = tmp_path / "b"
    out = tmp_path / "out"
    write_jsonl_results([_result("s1", "full_kv", 1.0, 1.0)], shard_a / "budget_selection_results.jsonl")
    write_jsonl_results([_result("s2", "full_kv", 1.0, 0.5)], shard_b / "budget_selection_results.jsonl")

    subprocess.run(
        [
            sys.executable,
            str(Path("scripts/merge_kv3d_budget_selection.py")),
            "--shard-dirs",
            str(shard_a),
            str(shard_b),
            "--output-dir",
            str(out),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    rows = [json.loads(line) for line in (out / "budget_selection_results.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert (out / "budget_selection_summary.csv").exists()
    assert (out / "budget_quality_curves.csv").exists()
    assert (out / "figures" / "budget_vs_f1.png").exists()
    assert "Budgeted KV Selection Evaluation" in (out / "report_budget_selection.md").read_text()
