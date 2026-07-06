import csv
import subprocess
import sys
from pathlib import Path


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_summarize_bidirectional_oracle_merges_task_outputs(tmp_path):
    root = tmp_path / "oracle"
    for task, full_f1, base_f1, threshold in [
        ("qasper", 0.8, 0.2, 0.25),
        ("qmsum", 0.6, 0.55, 0.5),
    ]:
        task_dir = root / task
        _write_csv(
            task_dir / "baseline_summary.csv",
            [
                {"method": "full_kv", "sample_count": 1, "mean_f1": full_f1, "mean_contains": 1.0, "mean_nll": 1.0, "mean_kv_ratio": 1.0},
                {"method": "b_only", "sample_count": 1, "mean_f1": base_f1, "mean_contains": 0.0, "mean_nll": 2.0, "mean_kv_ratio": 0.0},
            ],
        )
        _write_csv(
            task_dir / "oracle_thresholds.csv",
            [{"task_name": task, "threshold": 0.95, "min_kv_ratio": threshold, "method": "forward_greedy"}],
        )
        _write_csv(
            task_dir / "oracle_quality_curves.csv",
            [
                {"sample_id": "s1", "task_name": task, "method": "forward_greedy", "direction": "forward", "stage": "span", "step_index": 1, "f1": full_f1, "contains": 1.0, "exact": 0.0, "nll": 1.1, "kv_ratio": threshold}
            ],
        )
        _write_csv(
            task_dir / "oracle_labels.csv",
            [{"sample_id": "s1", "task_name": task, "layer": 1, "kv_head": 2, "span_id": 3, "span_start": 48, "span_end": 64, "selected": 1}],
        )
        _write_csv(
            task_dir / "block_features_labels.csv",
            [{"sample_id": "s1", "task_name": task, "layer": 1, "kv_head": 2, "span_id": 3, "span_start": 48, "span_end": 64, "selected": 1, "span_size": 16}],
        )
        _write_csv(
            task_dir / "selection_frequency_layer.csv",
            [{"task_name": task, "layer": 1, "block_count": 1, "selected_count": 1, "selection_rate": 1.0}],
        )
        _write_csv(
            task_dir / "selection_frequency_head.csv",
            [{"task_name": task, "layer": 1, "kv_head": 2, "block_count": 1, "selected_count": 1, "selection_rate": 1.0}],
        )
        _write_csv(
            task_dir / "selection_frequency_span.csv",
            [{"task_name": task, "span_id": 3, "block_count": 1, "selected_count": 1, "selection_rate": 1.0}],
        )

    subprocess.run(
        [
            sys.executable,
            "scripts/summarize_bidirectional_oracle.py",
            "--input-dir",
            str(root),
            "--output-dir",
            str(root / "summary"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    report = (root / "summary" / "report.md").read_text()
    assert (root / "summary" / "oracle_thresholds_all.csv").exists()
    assert (root / "summary" / "oracle_labels_all.csv").exists()
    assert (root / "summary" / "block_features_labels_all.csv").exists()
    assert "Do small KV subsets approach full KV?" in report
    assert "Which tasks depend more on KV?" in report
    assert "qasper" in report
    assert "qmsum" in report
