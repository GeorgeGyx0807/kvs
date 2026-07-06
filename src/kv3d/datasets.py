"""Dataset adapters for offline 3D KV profiling."""

from __future__ import annotations

import json
import os
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile
from dataclasses import replace
from typing import Any

from .runner import ProfilingSample

try:  # pragma: no cover - import guard
    from datasets import load_dataset
except Exception:  # pragma: no cover - dependency guard
    load_dataset = None  # type: ignore[assignment]

try:  # pragma: no cover - import guard
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - dependency guard
    hf_hub_download = None  # type: ignore[assignment]


DEFAULT_SAMPLE_SPEC: dict[str, str] = {
    "sample_id": "id",
    "dataset": "dataset",
    "task_name": "task_name",
    "prompt": "input",
    "context": "context",
    "gold_answer": "answer",
    "answers": "answers",
    "language": "language",
    "length": "length",
}


def _pick_field(row: dict[str, Any], candidates: list[str], default: Any = "") -> Any:
    for candidate in candidates:
        if candidate in row and row[candidate] is not None:
            return row[candidate]
    return default


def row_to_sample(row: dict[str, Any], spec: dict[str, str] | None = None) -> ProfilingSample:
    spec = {**DEFAULT_SAMPLE_SPEC, **(spec or {})}
    sample_id = str(_pick_field(row, [spec["sample_id"], "sample_id", "id", "_id"], default=""))
    dataset = str(_pick_field(row, [spec["dataset"], "dataset"], default="LongBench"))
    task_name = str(_pick_field(row, [spec["task_name"], "task_name", "task"], default="unknown"))
    prompt = str(_pick_field(row, [spec["prompt"], "prompt", "input", "question"], default=""))
    context = str(_pick_field(row, [spec["context"], "context", "passage", "doc"], default=""))
    gold_answer = str(_pick_field(row, [spec["gold_answer"], "gold_answer", "answer"], default=""))
    answers_raw = _pick_field(row, [spec["answers"], "answers", "gold_answers"], default=[])
    if isinstance(answers_raw, str):
        answers = (answers_raw,)
    else:
        answers = tuple(str(item) for item in answers_raw)
    if not gold_answer and answers:
        gold_answer = answers[0]
    language = _pick_field(row, [spec["language"], "language"], default=None)
    length = _pick_field(row, [spec["length"], "length"], default=None)
    consumed_keys = {
        spec["sample_id"],
        spec["dataset"],
        spec["task_name"],
        spec["prompt"],
        spec["context"],
        spec["gold_answer"],
        spec["answers"],
        spec["language"],
        spec["length"],
        "sample_id",
        "id",
        "dataset",
        "task_name",
        "task",
        "prompt",
        "input",
        "question",
        "context",
        "passage",
        "doc",
        "gold_answer",
        "answer",
        "answers",
        "gold_answers",
        "language",
        "length",
    }
    metadata = {key: value for key, value in row.items() if key not in consumed_keys}
    return ProfilingSample(
        sample_id=sample_id,
        dataset=dataset,
        task_name=task_name,
        prompt=prompt,
        context=context,
        gold_answer=gold_answer,
        answers=answers,
        language=None if language is None else str(language),
        length=None if length is None else int(length),
        metadata=metadata,
    )


def longbench_row_to_sample(row: dict[str, Any]) -> ProfilingSample:
    sample = row_to_sample(row, spec={"dataset": "LongBench"})
    task_name = str(_pick_field(row, ["dataset", "task_name", "task"], default=sample.task_name))
    sample_id = str(_pick_field(row, ["_id", "id", "sample_id"], default=sample.sample_id))
    return replace(sample, dataset="LongBench", task_name=task_name, sample_id=sample_id)


def row_to_sample_for_dataset(
    dataset_name: str,
    row: dict[str, Any],
    spec: dict[str, str] | None = None,
) -> ProfilingSample:
    if "longbench" in dataset_name.lower():
        return longbench_row_to_sample(row)
    return row_to_sample(row, spec=spec)


def _dataset_from_longbench_zip(config_name: str, cache_dir: Path | None = None) -> list[dict[str, Any]]:
    if hf_hub_download is None:
        raise RuntimeError("huggingface_hub is required to download LongBench")
    zip_path = hf_hub_download(
        repo_id="THUDM/LongBench",
        repo_type="dataset",
        filename="data.zip",
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    target_name = f"data/{config_name}.jsonl"
    with ZipFile(zip_path) as zf:
        with zf.open(target_name) as stream:
            rows = []
            for raw_line in TextIOWrapper(stream, encoding="utf-8"):
                line = raw_line.strip()
                if line:
                    rows.append(json.loads(line))
            return rows


def load_hf_dataset_split(
    dataset_name: str,
    split: str,
    config_name: str | None = None,
    cache_dir: Path | None = None,
):
    if "longbench" in dataset_name.lower():
        longbench_config = config_name or split
        if not longbench_config:
            raise RuntimeError("LongBench requires a config_name or split")
        return _dataset_from_longbench_zip(longbench_config, cache_dir=cache_dir)
    if load_dataset is None:
        raise RuntimeError("datasets library is required to load Hugging Face datasets")
    try:
        if config_name:
            return load_dataset(dataset_name, config_name, split=split)
        return load_dataset(dataset_name, split=split)
    except Exception as exc:  # pragma: no cover - network / hub guard
        config_suffix = f" config {config_name!r}" if config_name else ""
        raise RuntimeError(f"failed to load Hugging Face dataset {dataset_name!r}{config_suffix} split {split!r}") from exc


def export_samples_json(samples: list[ProfilingSample], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = [sample.to_dict() for sample in samples]
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))
