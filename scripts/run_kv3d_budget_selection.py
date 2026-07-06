#!/usr/bin/env python3
"""Run real budgeted KV selection evaluation from a span-level KV3D profile."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.kv3d.budget_selection import BUDGET_RATIOS
from src.kv3d.budget_selection import SELECTION_STRATEGIES
from src.kv3d.budget_selection import BudgetEvaluationResult
from src.kv3d.budget_selection import aggregate_budget_results
from src.kv3d.budget_selection import build_block_universe
from src.kv3d.budget_selection import kv3d_keys_for_blocks
from src.kv3d.budget_selection import load_utility_profile
from src.kv3d.budget_selection import quality_curve_rows
from src.kv3d.budget_selection import render_budget_figures
from src.kv3d.budget_selection import render_budget_report
from src.kv3d.budget_selection import select_blocks_for_budget
from src.kv3d.budget_selection import selected_attention_means
from src.kv3d.budget_selection import write_csv_rows
from src.kv3d.budget_selection import write_jsonl_results
from src.kv3d.datasets import load_hf_dataset_split, row_to_sample_for_dataset
from src.kv3d.executor import _answer_nll
from src.kv3d.executor import _answer_nll_from_context_cache
from src.kv3d.executor import _device_of
from src.kv3d.executor import _encode
from src.kv3d.executor import _generate_greedy
from src.kv3d.executor import _generate_greedy_from_context_cache
from src.kv3d.executor import _metric_for_prediction
from src.kv3d.executor import _prefill_context_cache
from src.kv3d.executor import format_question_prompt
from src.kv3d.executor import load_model_bundle
from src.kv3d.masks import selection_kv_bytes
from src.kv3d.runner import ProfilingSample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dataset-name", default="THUDM/LongBench")
    parser.add_argument("--config-name", default="narrativeqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-context-tokens", type=int, default=512)
    parser.add_argument("--span-size", type=int, default=16)
    parser.add_argument("--num-spans", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=36)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--strategies", default=",".join(SELECTION_STRATEGIES))
    parser.add_argument("--budget-ratios", default=",".join(str(value) for value in BUDGET_RATIOS))
    return parser.parse_args()


def _load_samples(args: argparse.Namespace) -> list[ProfilingSample]:
    rows = load_hf_dataset_split(args.dataset_name, args.split, config_name=args.config_name or None)
    samples = [row_to_sample_for_dataset(args.dataset_name, dict(row), spec={}) for row in rows]
    return samples[args.sample_offset : args.sample_offset + args.max_samples]


def _parse_csv_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _generation_length(bundle, text: str) -> int:
    if not text:
        return 0
    encoded = bundle.tokenizer(text, return_tensors="pt", add_special_tokens=False)
    return int(encoded["input_ids"].shape[-1])


def _result_for_metric(
    *,
    sample: ProfilingSample,
    method: str,
    budget_ratio: float,
    prediction: str,
    metric,
    full_nll: float | None,
    selected_kv_bytes: int,
    full_kv_bytes: int,
    generation_length: int,
    selected_block_count: int,
    attention_js_mean_selected: float | None,
    attention_kl_mean_selected: float | None,
    args: argparse.Namespace,
) -> BudgetEvaluationResult:
    delta_nll = None if metric.nll is None or full_nll is None else metric.nll - full_nll
    return BudgetEvaluationResult(
        sample_id=sample.sample_id,
        dataset=sample.dataset,
        task_name=sample.task_name,
        method=method,
        budget_ratio=budget_ratio,
        prediction=prediction,
        gold_answer=sample.gold_answer,
        answers=list(sample.answers),
        f1=metric.f1,
        contains=metric.contains,
        nll=metric.nll,
        delta_nll_vs_full=delta_nll,
        selected_kv_bytes=selected_kv_bytes,
        full_kv_bytes=full_kv_bytes,
        kv_ratio=0.0 if full_kv_bytes <= 0 else selected_kv_bytes / full_kv_bytes,
        ttft_ms=metric.ttft_ms,
        prefill_ms=metric.prefill_ms,
        decode_ms=metric.decode_ms,
        generation_length=generation_length,
        selected_block_count=selected_block_count,
        attention_js_mean_selected=attention_js_mean_selected,
        attention_kl_mean_selected=attention_kl_mean_selected,
        model_name=args.model_name,
        max_context_tokens=args.max_context_tokens,
        span_size=args.span_size,
        max_new_tokens=args.max_new_tokens,
        random_seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    strategies = _parse_csv_strings(args.strategies)
    budget_ratios = _parse_csv_floats(args.budget_ratios)
    samples = _load_samples(args)
    profile = load_utility_profile(args.profile_dir)
    blocks = build_block_universe(
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_spans=args.num_spans,
        span_size=args.span_size,
        max_context_tokens=args.max_context_tokens,
    )
    bundle = load_model_bundle(args.model_name, device_map=args.device_map, dtype=args.dtype)
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = _device_of(model)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))

    results: list[BudgetEvaluationResult] = []
    for sample_index, sample in enumerate(samples, start=1):
        print(f"[budget-selection] sample {sample_index}/{len(samples)} {sample.sample_id}", flush=True)
        context_ids = _encode(tokenizer, sample.context, device, max_tokens=args.max_context_tokens)
        suffix_ids = _encode(tokenizer, f"\n\nQuestion: {sample.prompt}\nAnswer:", device)
        context_length = int(context_ids.shape[-1])
        context_cache, _context_prefill_ms = _prefill_context_cache(model=model, input_ids=context_ids)
        active_blocks = [block for block in blocks if block.span_start < context_length]
        full_keys = kv3d_keys_for_blocks(active_blocks, sample.sample_id)
        full_kv_bytes = selection_kv_bytes(
            selected_blocks=full_keys,
            seq_len=context_length,
            head_dim=head_dim,
            chunk_size=args.span_size,
        )

        full_generation = _generate_greedy_from_context_cache(
            model=model,
            tokenizer=tokenizer,
            context_cache=context_cache,
            suffix_ids=suffix_ids,
            context_length=context_length,
            max_new_tokens=args.max_new_tokens,
            chunk_size=args.span_size,
        )
        full_nll = _answer_nll_from_context_cache(
            model=model,
            tokenizer=tokenizer,
            context_cache=context_cache,
            suffix_ids=suffix_ids,
            answer=sample.gold_answer,
            context_length=context_length,
            chunk_size=args.span_size,
        )
        full_metric = _metric_for_prediction(
            prediction=full_generation.text,
            answers=sample.answers,
            nll=full_nll,
            ttft_ms=full_generation.ttft_ms,
            prefill_ms=full_generation.prefill_ms,
            decode_ms=full_generation.decode_ms,
        )
        results.append(
            _result_for_metric(
                sample=sample,
                method="full_kv",
                budget_ratio=1.0,
                prediction=full_generation.text,
                metric=full_metric,
                full_nll=full_nll,
                selected_kv_bytes=full_kv_bytes,
                full_kv_bytes=full_kv_bytes,
                generation_length=_generation_length(bundle, full_generation.text),
                selected_block_count=len(active_blocks),
                attention_js_mean_selected=None,
                attention_kl_mean_selected=None,
                args=args,
            )
        )

        question_ids = _encode(tokenizer, format_question_prompt(sample), device)
        base_generation = _generate_greedy(
            model=model,
            tokenizer=tokenizer,
            input_ids=question_ids,
            max_new_tokens=args.max_new_tokens,
        )
        base_nll = _answer_nll(model=model, tokenizer=tokenizer, prompt_ids=question_ids, answer=sample.gold_answer)
        base_metric = _metric_for_prediction(
            prediction=base_generation.text,
            answers=sample.answers,
            nll=base_nll,
            ttft_ms=base_generation.ttft_ms,
            prefill_ms=base_generation.prefill_ms,
            decode_ms=base_generation.decode_ms,
        )
        results.append(
            _result_for_metric(
                sample=sample,
                method="b_only",
                budget_ratio=0.0,
                prediction=base_generation.text,
                metric=base_metric,
                full_nll=full_nll,
                selected_kv_bytes=0,
                full_kv_bytes=full_kv_bytes,
                generation_length=_generation_length(bundle, base_generation.text),
                selected_block_count=0,
                attention_js_mean_selected=None,
                attention_kl_mean_selected=None,
                args=args,
            )
        )

        block_bytes = max(1, full_kv_bytes // max(len(active_blocks), 1))
        for strategy in strategies:
            print(f"[budget-selection] sample {sample_index}/{len(samples)} strategy={strategy}", flush=True)
            for budget_ratio in budget_ratios:
                selected_blocks = select_blocks_for_budget(
                    blocks=active_blocks,
                    profile=profile,
                    strategy=strategy,
                    budget_bytes=int(full_kv_bytes * budget_ratio),
                    block_bytes=block_bytes,
                    seed=args.seed,
                )
                selected_keys = kv3d_keys_for_blocks(selected_blocks, sample.sample_id)
                selected_kv_bytes = selection_kv_bytes(
                    selected_blocks=selected_keys,
                    seq_len=context_length,
                    head_dim=head_dim,
                    chunk_size=args.span_size,
                )
                generation = _generate_greedy_from_context_cache(
                    model=model,
                    tokenizer=tokenizer,
                    context_cache=context_cache,
                    suffix_ids=suffix_ids,
                    context_length=context_length,
                    max_new_tokens=args.max_new_tokens,
                    chunk_size=args.span_size,
                    kept_blocks=selected_keys,
                )
                nll = _answer_nll_from_context_cache(
                    model=model,
                    tokenizer=tokenizer,
                    context_cache=context_cache,
                    suffix_ids=suffix_ids,
                    answer=sample.gold_answer,
                    context_length=context_length,
                    chunk_size=args.span_size,
                    kept_blocks=selected_keys,
                )
                metric = _metric_for_prediction(
                    prediction=generation.text,
                    answers=sample.answers,
                    nll=nll,
                    ttft_ms=generation.ttft_ms,
                    prefill_ms=generation.prefill_ms,
                    decode_ms=generation.decode_ms,
                )
                attention_js, attention_kl = selected_attention_means(selected_blocks, profile)
                results.append(
                    _result_for_metric(
                        sample=sample,
                        method=strategy,
                        budget_ratio=budget_ratio,
                        prediction=generation.text,
                        metric=metric,
                        full_nll=full_nll,
                        selected_kv_bytes=selected_kv_bytes,
                        full_kv_bytes=full_kv_bytes,
                        generation_length=_generation_length(bundle, generation.text),
                        selected_block_count=len(selected_blocks),
                        attention_js_mean_selected=attention_js,
                        attention_kl_mean_selected=attention_kl,
                        args=args,
                    )
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = aggregate_budget_results(results)
    curve_rows = quality_curve_rows(summary_rows)
    write_jsonl_results(results, args.output_dir / "budget_selection_results.jsonl")
    write_csv_rows(summary_rows, args.output_dir / "budget_selection_summary.csv")
    write_csv_rows(curve_rows, args.output_dir / "budget_quality_curves.csv")
    render_budget_figures(summary_rows, args.output_dir / "figures")
    render_budget_report(summary_rows, args.output_dir / "report_budget_selection.md")
    (args.output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
