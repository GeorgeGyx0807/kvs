import json
from pathlib import Path
from zipfile import ZipFile

from src.kv3d.datasets import load_hf_dataset_split, longbench_row_to_sample, row_to_sample


def test_longbench_row_to_sample_maps_context_and_answers():
    row = {
        "_id": "lb-1",
        "dataset": "narrativeqa",
        "input": "What is X?",
        "context": "X is Y.",
        "answers": ["Y"],
        "language": "en",
        "length": 42,
    }

    sample = longbench_row_to_sample(row)

    assert sample.sample_id == "lb-1"
    assert sample.dataset == "LongBench"
    assert sample.task_name == "narrativeqa"
    assert sample.prompt == "What is X?"
    assert sample.context == "X is Y."
    assert sample.answers == ("Y",)
    assert sample.gold_answer == "Y"


def test_row_to_sample_supports_configured_field_names():
    row = {
        "sid": "r1",
        "source": "RULER",
        "kind": "stress",
        "query": "What?",
        "ctx": "Because.",
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

    assert sample.sample_id == "r1"
    assert sample.dataset == "RULER"
    assert sample.task_name == "stress"
    assert sample.prompt == "What?"
    assert sample.context == "Because."
    assert sample.gold_answer == "A"
    assert sample.answers == ("A",)


def test_load_hf_dataset_split_supports_config_name(monkeypatch):
    import src.kv3d.datasets as datasets_mod

    calls = {}

    def fake_load_dataset(dataset_name: str, config_name: str | None = None, split: str | None = None):
        calls["dataset_name"] = dataset_name
        calls["config_name"] = config_name
        calls["split"] = split
        return [{"id": "x"}]

    monkeypatch.setattr(datasets_mod, "load_dataset", fake_load_dataset, raising=False)

    result = datasets_mod.load_hf_dataset_split("OtherDataset", "train", config_name="v1")

    assert result == [{"id": "x"}]
    assert calls == {"dataset_name": "OtherDataset", "config_name": "v1", "split": "train"}


def test_load_hf_dataset_split_requires_datasets_library(monkeypatch):
    import src.kv3d.datasets as datasets_mod

    def fake_loader(dataset_name: str, split: str, config_name: str | None = None):
        raise RuntimeError("failed to load Hugging Face dataset 'x' split 'y'")

    monkeypatch.setattr(datasets_mod, "load_dataset", fake_loader, raising=False)

    try:
        datasets_mod.load_hf_dataset_split("x", "y")
    except RuntimeError as exc:
        assert "failed to load Hugging Face dataset" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_load_hf_dataset_split_can_read_longbench_zip(monkeypatch, tmp_path):
    import src.kv3d.datasets as datasets_mod

    zip_path = tmp_path / "data.zip"
    with ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "data/narrativeqa.jsonl",
            json.dumps({"_id": "lb-1", "dataset": "narrativeqa", "input": "Q?", "context": "C.", "answers": ["A"]})
            + "\n",
        )

    monkeypatch.setattr(datasets_mod, "hf_hub_download", lambda **kwargs: str(zip_path), raising=False)

    rows = datasets_mod.load_hf_dataset_split("THUDM/LongBench", "test", config_name="narrativeqa")

    assert rows[0]["_id"] == "lb-1"
    assert rows[0]["dataset"] == "narrativeqa"
