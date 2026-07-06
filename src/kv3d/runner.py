"""Offline 3D KV profiling orchestration helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from .analysis import aggregate_profiling_records
from .blocks import chunk_bounds
from .blocks import KV3DKey
from .plan import generate_profiling_plan
from .records import KV3DAttentionDivergenceSnapshot, KV3DMetricSnapshot, KV3DProfilingRecord, ProfilingManifest
from .report import render_profiling_report


@dataclass(frozen=True)
class ProfilingSample:
    sample_id: str
    dataset: str
    task_name: str
    prompt: str
    context: str = ""
    gold_answer: str = ""
    answers: tuple[str, ...] = ()
    language: str | None = None
    length: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def annotate_record_token_spans(
    records: Iterable[dict[str, Any]],
    *,
    chunk_size: int,
    max_context_tokens: int | None = None,
) -> list[dict[str, Any]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    annotated: list[dict[str, Any]] = []
    for record in records:
        payload = dict(record)
        key_payload = payload.get("key")
        if isinstance(key_payload, dict) and (
            key_payload.get("token_start") is None or key_payload.get("token_end") is None
        ):
            key = dict(key_payload)
            token_start, token_end = chunk_bounds(int(key["chunk"]), chunk_size)
            if max_context_tokens is not None:
                token_start = min(token_start, max_context_tokens)
                token_end = min(token_end, max_context_tokens)
            key["token_start"] = token_start
            key["token_end"] = token_end
            payload["key"] = key
        annotated.append(payload)
    return annotated


def _record_from_dict(payload: dict[str, Any]) -> KV3DProfilingRecord:
    key_payload = payload.get("key")
    key = None
    if key_payload is not None:
        key = KV3DKey(
            sample_id=str(key_payload["sample_id"]),
            layer=int(key_payload["layer"]),
            head=int(key_payload["head"]),
            chunk=int(key_payload["chunk"]),
            token_start=None if key_payload.get("token_start") is None else int(key_payload["token_start"]),
            token_end=None if key_payload.get("token_end") is None else int(key_payload["token_end"]),
        )
    metric_payload = payload["metric"]
    delta_payload = payload.get("delta_vs_full")
    delta_base_payload = payload.get("delta_vs_base")
    attention_payload = payload.get("attention_divergence")
    return KV3DProfilingRecord(
        sample_id=str(payload["sample_id"]),
        method=str(payload["method"]),
        key=key,
        selected_kv_bytes=int(payload["selected_kv_bytes"]),
        metric=KV3DMetricSnapshot(
            accuracy=metric_payload.get("accuracy"),
            f1=metric_payload.get("f1"),
            contains=metric_payload.get("contains"),
            nll=metric_payload.get("nll"),
            ttft_ms=metric_payload.get("ttft_ms"),
            prefill_ms=metric_payload.get("prefill_ms"),
            decode_ms=metric_payload.get("decode_ms"),
        ),
        delta_vs_full=None
        if delta_payload is None
        else KV3DMetricSnapshot(
            accuracy=delta_payload.get("accuracy"),
            f1=delta_payload.get("f1"),
            contains=delta_payload.get("contains"),
            nll=delta_payload.get("nll"),
            ttft_ms=delta_payload.get("ttft_ms"),
            prefill_ms=delta_payload.get("prefill_ms"),
            decode_ms=delta_payload.get("decode_ms"),
        ),
        delta_vs_base=None
        if delta_base_payload is None
        else KV3DMetricSnapshot(
            accuracy=delta_base_payload.get("accuracy"),
            f1=delta_base_payload.get("f1"),
            contains=delta_base_payload.get("contains"),
            nll=delta_base_payload.get("nll"),
            ttft_ms=delta_base_payload.get("ttft_ms"),
            prefill_ms=delta_base_payload.get("prefill_ms"),
            decode_ms=delta_base_payload.get("decode_ms"),
        ),
        attention_divergence=None
        if attention_payload is None
        else KV3DAttentionDivergenceSnapshot(
            js_divergence=attention_payload.get("js_divergence"),
            kl_divergence=attention_payload.get("kl_divergence"),
        ),
    )


def run_offline_3d_profile(
    samples: Iterable[ProfilingSample],
    num_layers: int,
    num_heads: int,
    num_chunks: int,
    include_addition: bool = False,
    include_chunk_level: bool = True,
    records: Iterable[KV3DProfilingRecord] | Iterable[dict[str, Any]] | None = None,
    experiment_name: str = "offline_3d_kv_utility_profiling",
    model_name: str = "Qwen3-8B",
    agent_pair: str = "same_checkpoint",
    auxiliary_dataset: str = "RULER",
) -> dict[str, Any]:
    samples = list(samples)
    sample_ids = [sample.sample_id for sample in samples]
    main_dataset = samples[0].dataset if samples else "LongBench"
    manifest = ProfilingManifest(
        experiment_name=experiment_name,
        model_name=model_name,
        agent_pair=agent_pair,
        main_dataset=main_dataset,
        auxiliary_dataset=auxiliary_dataset,
        baseline="full-KV",
    )
    plan = generate_profiling_plan(
        sample_ids=sample_ids,
        num_layers=num_layers,
        num_heads=num_heads,
        num_chunks=num_chunks,
        include_addition=include_addition,
        include_chunk_level=include_chunk_level,
    )
    parsed_records: list[KV3DProfilingRecord] = []
    for record in records or []:
        if isinstance(record, KV3DProfilingRecord):
            parsed_records.append(record)
        else:
            parsed_records.append(_record_from_dict(record))
    sample_metadata_by_id = {
        sample.sample_id: {"dataset": sample.dataset, "task_name": sample.task_name}
        for sample in samples
    }
    tables = aggregate_profiling_records(parsed_records, sample_metadata_by_id=sample_metadata_by_id)
    report = render_profiling_report(
        plan=plan,
        records=parsed_records,
        tables=tables,
        experiment_name=experiment_name,
        model_name=model_name,
        main_dataset=main_dataset,
    )
    return {
        "manifest": manifest.to_dict(),
        "samples": [sample.to_dict() for sample in samples],
        "plan": plan,
        "records": [record.to_dict() for record in parsed_records],
        "tables": tables,
        "report": report,
        "stage": "stage2" if include_chunk_level else "stage1",
    }
