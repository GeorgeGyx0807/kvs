import json

from src.kv3d.datasets import export_samples_json, row_to_sample


def test_sample_export_roundtrip_with_spec(tmp_path):
    row = {
        "sid": "x1",
        "source": "RULER",
        "kind": "stress",
        "query": "Q?",
        "ctx": "C.",
        "label": "A",
        "alts": ["A"],
    }
    spec = {
        "sample_id": "sid",
        "dataset": "source",
        "task_name": "kind",
        "prompt": "query",
        "context": "ctx",
        "gold_answer": "label",
        "answers": "alts",
    }
    sample = row_to_sample(row, spec=spec)
    output = tmp_path / "samples.json"
    export_samples_json([sample], output)

    payload = json.loads(output.read_text())
    assert payload[0]["sample_id"] == "x1"
    assert payload[0]["dataset"] == "RULER"

