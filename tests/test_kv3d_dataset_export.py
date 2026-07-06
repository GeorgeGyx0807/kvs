import json

from src.kv3d.datasets import export_samples_json, longbench_row_to_sample


def test_export_samples_json_writes_serialized_samples(tmp_path):
    rows = [
        {
            "id": "lb-1",
            "dataset": "LongBench",
            "task_name": "qa",
            "input": "What is X?",
            "context": "X is Y.",
            "answers": ["Y"],
        }
    ]
    samples = [longbench_row_to_sample(row) for row in rows]
    output = tmp_path / "samples.json"

    export_samples_json(samples, output)

    payload = json.loads(output.read_text())
    assert payload[0]["sample_id"] == "lb-1"
    assert payload[0]["context"] == "X is Y."

