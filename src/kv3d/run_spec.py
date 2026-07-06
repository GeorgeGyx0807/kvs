"""Run specification helpers for offline 3D KV profiling experiments."""

from __future__ import annotations

from math import ceil
from typing import Any

from transformers import AutoConfig


def _resolve_model_dimensions(model_name: str, num_layers: int, num_heads: int) -> tuple[int, int]:
    if num_layers > 0 and num_heads > 0:
        return num_layers, num_heads
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    resolved_layers = num_layers if num_layers > 0 else int(config.num_hidden_layers)
    resolved_heads = num_heads if num_heads > 0 else int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    return resolved_layers, resolved_heads


def _per_sample_plan_size(
    num_layers: int,
    num_heads: int,
    num_chunks: int,
    include_addition: bool,
    include_chunk_level: bool,
) -> int:
    if num_layers <= 0 or num_heads <= 0 or num_chunks <= 0:
        raise ValueError("num_layers, num_heads, and num_chunks must be positive")
    addition_factor = 2 if include_addition else 1
    chunk_term = num_layers * num_heads * num_chunks * addition_factor if include_chunk_level else 0
    return 2 + num_layers + (num_layers * num_heads) + chunk_term


def _validation_command(
    *,
    run_dir: str,
    min_samples: int,
    require_stability: bool,
    min_heterogeneity_range: float | None,
) -> str:
    validate_parts = [
        "python3 scripts/validate_kv3d_run.py",
        f"--run-dir {run_dir}",
        f"--output {run_dir}/validation.json",
        f"--min-samples {min_samples}",
    ]
    if require_stability:
        validate_parts.append("--require-stability")
    if min_heterogeneity_range is not None:
        validate_parts.append(f"--min-heterogeneity-range {min_heterogeneity_range}")
    return " ".join(validate_parts)


def build_kv3d_run_spec(
    *,
    profile_name: str,
    dataset_name: str,
    config_name: str,
    split: str,
    model_name: str,
    output_dir: str,
    max_samples: int,
    num_layers: int,
    num_heads: int,
    num_chunks: int,
    chunk_size: int,
    max_context_tokens: int,
    max_new_tokens: int,
    include_addition: bool,
    include_chunk_level: bool = True,
    min_samples_gate: int,
    min_heterogeneity_range: float | None,
    require_stability: bool = True,
    shard_size: int | None = None,
    methods: str = "",
    spec_path: str | None = None,
) -> dict[str, Any]:
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if chunk_size <= 0 or max_context_tokens <= 0 or max_new_tokens <= 0:
        raise ValueError("token and chunk parameters must be positive")
    if shard_size is not None and shard_size <= 0:
        raise ValueError("shard_size must be positive when provided")

    num_layers, num_heads = _resolve_model_dimensions(model_name, num_layers, num_heads)
    per_sample = _per_sample_plan_size(num_layers, num_heads, num_chunks, include_addition, include_chunk_level)
    include_addition_flag = " --include-addition" if include_addition else ""
    include_chunk_level_flag = " --include-chunk-level" if include_chunk_level else " --no-include-chunk-level"
    methods_flag = f" --methods {methods}" if methods else ""
    shard_count = ceil(max_samples / (shard_size or max_samples))
    shard_dirs = [f"{output_dir}_shard_{index:03d}" for index in range(shard_count)]
    shard_sample_counts = [min(shard_size or max_samples, max_samples - offset) for offset in range(0, max_samples, shard_size or max_samples)]
    shard_status_spec = spec_path or "<run-spec.json>"
    merge_command = (
        "python3 scripts/merge_kv3d_shards.py "
        f"--shard-dirs {' '.join(shard_dirs)} "
        f"--num-layers {num_layers} "
        f"--num-heads {num_heads} "
        f"--num-chunks {num_chunks}"
        f"{include_addition_flag} "
        f"{include_chunk_level_flag} "
        f"--model-name {model_name} "
        f"--output-dir {output_dir}"
    )

    return {
        "profile_name": profile_name,
        "model_name": model_name,
        "dataset": {
            "dataset_name": dataset_name,
            "config_name": config_name,
            "split": split,
            "max_samples": max_samples,
        },
        "dimensions": {
            "num_layers": num_layers,
            "num_heads": num_heads,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "max_context_tokens": max_context_tokens,
            "max_new_tokens": max_new_tokens,
            "include_addition": include_addition,
            "include_chunk_level": include_chunk_level,
        },
        "per_sample_plan_size": per_sample,
        "plan_size": per_sample * max_samples,
        "validation_gate": {
            "min_samples": min_samples_gate,
            "require_stability": require_stability,
            "min_heterogeneity_range": min_heterogeneity_range,
        },
        "commands": {
            "run_gpu_profile": (
                "python3 scripts/run_kv3d_gpu_profile.py "
                f"--model-name {model_name} "
                f"--dataset-name {dataset_name} "
                f"--config-name {config_name} "
                f"--split {split} "
                f"--max-samples {max_samples} "
                f"--chunk-size {chunk_size} "
                f"--max-context-tokens {max_context_tokens} "
                f"--max-new-tokens {max_new_tokens} "
                f"--max-layers {num_layers} "
                f"--max-heads {num_heads} "
                f"--num-chunks {num_chunks}"
                f"{include_addition_flag} "
                f"{include_chunk_level_flag} "
                f"{methods_flag}"
                f"--output-dir {output_dir}"
            ),
            "run_shards": [
                (
                    "python3 scripts/run_kv3d_gpu_profile.py "
                    f"--model-name {model_name} "
                    f"--dataset-name {dataset_name} "
                    f"--config-name {config_name} "
                    f"--split {split} "
                    f"--sample-offset {offset} "
                    f"--max-samples {min(shard_size or max_samples, max_samples - offset)} "
                    f"--chunk-size {chunk_size} "
                    f"--max-context-tokens {max_context_tokens} "
                    f"--max-new-tokens {max_new_tokens} "
                    f"--max-layers {num_layers} "
                    f"--max-heads {num_heads} "
                    f"--num-chunks {num_chunks}"
                    f"{include_addition_flag} "
                    f"{include_chunk_level_flag} "
                    f"{methods_flag}"
                    f"--output-dir {output_dir}_shard_{index:03d}"
                )
                for index, offset in enumerate(range(0, max_samples, shard_size or max_samples))
            ],
            "validate_shards": [
                _validation_command(
                    run_dir=shard_dir,
                    min_samples=sample_count,
                    require_stability=require_stability,
                    min_heterogeneity_range=min_heterogeneity_range,
                )
                for shard_dir, sample_count in zip(shard_dirs, shard_sample_counts)
            ],
            "merge_shards": merge_command,
            "shard_status": (
                "python3 scripts/kv3d_shard_status.py "
                f"--spec {shard_status_spec} "
                f"--output {output_dir}/shard_status.json"
            ),
            "validate": _validation_command(
                run_dir=output_dir,
                min_samples=min_samples_gate,
                require_stability=require_stability,
                min_heterogeneity_range=min_heterogeneity_range,
            ),
        },
        "sharding": {
            "shard_size": shard_size,
            "shard_count": shard_count,
            "methods": methods,
        },
    }
