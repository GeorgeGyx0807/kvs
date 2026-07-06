#!/usr/bin/env python3
"""Run LongBench bidirectional greedy oracle selected-KV search."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
from src.kv3d.oracle_search import OracleBlock
from src.kv3d.oracle_search import OracleEval
from src.kv3d.oracle_search import OracleStep
from src.kv3d.oracle_search import best_backward_removal
from src.kv3d.oracle_search import best_forward_addition
from src.kv3d.oracle_search import build_block_universe
from src.kv3d.oracle_search import exact_match
from src.kv3d.oracle_search import oracle_blocks_to_kv3d_keys
from src.kv3d.oracle_search import prune_bottom_fraction
from src.kv3d.oracle_search import task_family_for_longbench
from src.kv3d.oracle_search import write_oracle_artifacts
from src.kv3d.runner import ProfilingSample


DEFAULT_TASKS = "narrativeqa,qasper,multifieldqa_en,hotpotqa,passage_retrieval_en,qmsum"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3-8B")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--dataset-name", default="THUDM/LongBench")
    parser.add_argument("--tasks", default=DEFAULT_TASKS)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--max-context-tokens", type=int, default=2048)
    parser.add_argument("--span-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=0)
    parser.add_argument("--num-heads", type=int, default=0)
    parser.add_argument("--discard-fraction", type=float, default=0.30)
    parser.add_argument("--prefix-spans", type=int, default=1)
    parser.add_argument("--max-forward-steps", type=int, default=0)
    parser.add_argument("--max-backward-steps", type=int, default=0)
    parser.add_argument("--max-span-candidates", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/longbench_bidirectional_oracle"))
    return parser.parse_args()


def apply_sanity_defaults(args: argparse.Namespace) -> None:
    if not args.sanity:
        return
    args.tasks = args.tasks.split(",")[0].strip()
    args.max_samples = min(args.max_samples, 1)
    args.max_context_tokens = min(args.max_context_tokens, 128)
    args.max_new_tokens = min(args.max_new_tokens, 4)
    args.num_layers = 2 if args.num_layers <= 0 else min(args.num_layers, 2)
    args.num_heads = 2 if args.num_heads <= 0 else min(args.num_heads, 2)
    args.max_forward_steps = 2 if args.max_forward_steps <= 0 else min(args.max_forward_steps, 2)
    args.max_backward_steps = 2 if args.max_backward_steps <= 0 else min(args.max_backward_steps, 2)
    args.max_span_candidates = 4 if args.max_span_candidates <= 0 else min(args.max_span_candidates, 4)


def _parse_tasks(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _load_task_samples(args: argparse.Namespace, task_name: str) -> list[ProfilingSample]:
    rows = load_hf_dataset_split(args.dataset_name, args.split, config_name=task_name)
    samples = [row_to_sample_for_dataset(args.dataset_name, dict(row), spec={}) for row in rows]
    return samples[args.sample_offset : args.sample_offset + args.max_samples]


def _kv_bytes_for_blocks(
    *,
    blocks: Sequence[OracleBlock],
    sample_id: str,
    context_length: int,
    head_dim: int,
    span_size: int,
) -> int:
    keys = oracle_blocks_to_kv3d_keys(
        [
            OracleBlock(
                sample_id=sample_id,
                layer=block.layer,
                kv_head=block.kv_head,
                span_id=block.span_id,
                span_start=block.span_start,
                span_end=block.span_end,
            )
            for block in blocks
        ]
    )
    return selection_kv_bytes(selected_blocks=keys, seq_len=context_length, head_dim=head_dim, chunk_size=span_size)


def _make_eval(
    *,
    sample: ProfilingSample,
    method: str,
    direction: str,
    stage: str,
    step_index: int,
    selected_blocks: Sequence[OracleBlock],
    prediction: str,
    nll: float | None,
    selected_kv_bytes: int,
    full_kv_bytes: int,
    ttft_ms: float | None,
    prefill_ms: float | None,
    decode_ms: float | None,
) -> OracleEval:
    metric = _metric_for_prediction(
        prediction=prediction,
        answers=sample.answers,
        nll=nll,
        ttft_ms=ttft_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
    )
    return OracleEval(
        sample_id=sample.sample_id,
        method=method,
        direction=direction,
        stage=stage,
        step_index=step_index,
        selected_blocks=tuple(sorted(selected_blocks, key=lambda block: block.block_id)),
        prediction=prediction,
        gold_answer=sample.gold_answer,
        answers=sample.answers,
        f1=metric.f1,
        contains=metric.contains,
        exact=exact_match(prediction, sample.answers),
        nll=metric.nll,
        selected_kv_bytes=selected_kv_bytes,
        full_kv_bytes=full_kv_bytes,
        kv_ratio=0.0 if full_kv_bytes <= 0 else selected_kv_bytes / full_kv_bytes,
        ttft_ms=metric.ttft_ms,
        prefill_ms=metric.prefill_ms,
        decode_ms=metric.decode_ms,
    )


def _coarse_scores(
    *,
    baseline_quality: OracleEval,
    candidates: Iterable[tuple[object, OracleEval]],
    task_family: str,
) -> dict[object, float]:
    from src.kv3d.oracle_search import quality_scalar

    full_quality = quality_scalar(baseline_quality, task_family=task_family)
    scores = {}
    for key, result in candidates:
        scores[key] = full_quality - quality_scalar(result, task_family=task_family)
    return scores


def _limit_blocks(blocks: list[OracleBlock], max_span_candidates: int) -> list[OracleBlock]:
    if max_span_candidates <= 0 or len(blocks) <= max_span_candidates:
        return blocks
    prefix = sorted(blocks, key=lambda block: (block.span_id, block.layer, block.kv_head))[: max_span_candidates // 2]
    suffix = sorted(blocks, key=lambda block: (-block.span_id, block.layer, block.kv_head))[: max_span_candidates - len(prefix)]
    merged = {block.block_id: block for block in [*prefix, *suffix]}
    return sorted(merged.values(), key=lambda block: block.block_id)


def _baseline_blocks(
    *,
    universe: Sequence[OracleBlock],
    strategy: str,
    full_kv_bytes: int,
    target_ratio: float,
    seed: int,
) -> list[OracleBlock]:
    block_count = max(1, int(len(universe) * target_ratio))
    if strategy == "prefix":
        return sorted(universe, key=lambda block: (block.span_id, block.layer, block.kv_head))[:block_count]
    if strategy == "uniform":
        return sorted(universe, key=lambda block: (block.span_id, block.layer, block.kv_head))[:: max(1, len(universe) // block_count)][
            :block_count
        ]
    if strategy == "random":
        rng = random.Random(seed)
        blocks = list(universe)
        rng.shuffle(blocks)
        return sorted(blocks[:block_count], key=lambda block: block.block_id)
    raise ValueError(f"unknown baseline strategy: {strategy}")


def run_task_oracle(*, bundle, task_name: str, samples: Sequence[ProfilingSample], args: argparse.Namespace) -> dict:
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = _device_of(model)
    num_layers = args.num_layers or int(model.config.num_hidden_layers)
    num_heads = args.num_heads or int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
    task_family = task_family_for_longbench(task_name)

    baselines: list[OracleEval] = []
    steps: list[OracleStep] = []
    universe_all: list[OracleBlock] = []
    full_eval_by_sample: dict[str, OracleEval] = {}

    for sample_index, sample in enumerate(samples, start=1):
        print(f"[oracle] task={task_name} sample={sample_index}/{len(samples)} id={sample.sample_id}", flush=True)
        context_ids = _encode(tokenizer, sample.context, device, max_tokens=args.max_context_tokens)
        suffix_ids = _encode(tokenizer, f"\n\nQuestion: {sample.prompt}\nAnswer:", device)
        context_length = int(context_ids.shape[-1])
        context_cache, _ = _prefill_context_cache(model=model, input_ids=context_ids)
        layer_heads_all = [(layer, head) for layer in range(num_layers) for head in range(num_heads)]
        full_universe = build_block_universe(
            sample_id=sample.sample_id,
            layers=list(range(num_layers)),
            layer_heads=layer_heads_all,
            span_size=args.span_size,
            max_context_tokens=context_length,
        )
        full_keys = oracle_blocks_to_kv3d_keys(full_universe)
        full_kv_bytes = selection_kv_bytes(
            selected_blocks=full_keys,
            seq_len=context_length,
            head_dim=head_dim,
            chunk_size=args.span_size,
        )

        def eval_selected(
            selected: Sequence[OracleBlock],
            *,
            method: str,
            direction: str,
            stage: str,
            step_index: int,
        ) -> OracleEval:
            selected_keys = oracle_blocks_to_kv3d_keys(selected)
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
            selected_kv_bytes = _kv_bytes_for_blocks(
                blocks=selected,
                sample_id=sample.sample_id,
                context_length=context_length,
                head_dim=head_dim,
                span_size=args.span_size,
            )
            return _make_eval(
                sample=sample,
                method=method,
                direction=direction,
                stage=stage,
                step_index=step_index,
                selected_blocks=selected,
                prediction=generation.text,
                nll=nll,
                selected_kv_bytes=selected_kv_bytes,
                full_kv_bytes=full_kv_bytes,
                ttft_ms=generation.ttft_ms,
                prefill_ms=generation.prefill_ms,
                decode_ms=generation.decode_ms,
            )

        full_eval = eval_selected(full_universe, method="full_kv", direction="baseline", stage="baseline", step_index=0)
        baselines.append(full_eval)
        full_eval_by_sample[sample.sample_id] = full_eval

        question_ids = _encode(tokenizer, format_question_prompt(sample), device)
        b_generation = _generate_greedy(model=model, tokenizer=tokenizer, input_ids=question_ids, max_new_tokens=args.max_new_tokens)
        b_nll = _answer_nll(model=model, tokenizer=tokenizer, prompt_ids=question_ids, answer=sample.gold_answer)
        baselines.append(
            _make_eval(
                sample=sample,
                method="b_only",
                direction="baseline",
                stage="baseline",
                step_index=0,
                selected_blocks=[],
                prediction=b_generation.text,
                nll=b_nll,
                selected_kv_bytes=0,
                full_kv_bytes=full_kv_bytes,
                ttft_ms=b_generation.ttft_ms,
                prefill_ms=b_generation.prefill_ms,
                decode_ms=b_generation.decode_ms,
            )
        )

        layer_removal_results = []
        for layer in range(num_layers):
            selected = [block for block in full_universe if block.layer != layer]
            result = eval_selected(selected, method="coarse_remove_layer", direction="backward", stage="layer", step_index=layer)
            layer_removal_results.append((layer, result))
            steps.append(
                OracleStep(
                    sample_id=sample.sample_id,
                    task_name=task_name,
                    direction="backward",
                    stage="layer",
                    step_index=layer,
                    action="coarse_remove_layer",
                    changed_block=None,
                    result=result,
                )
            )
        kept_layers = prune_bottom_fraction(
            _coarse_scores(baseline_quality=full_eval, candidates=layer_removal_results, task_family=task_family),
            discard_fraction=args.discard_fraction,
        )

        layer_head_removal_results = []
        for layer_head_index, (layer, head) in enumerate([(layer, head) for layer in kept_layers for head in range(num_heads)]):
            selected = [block for block in full_universe if not (block.layer == layer and block.kv_head == head)]
            result = eval_selected(
                selected,
                method="coarse_remove_layer_head",
                direction="backward",
                stage="layer_head",
                step_index=layer_head_index,
            )
            layer_head_removal_results.append(
                ((layer, head), result)
            )
            steps.append(
                OracleStep(
                    sample_id=sample.sample_id,
                    task_name=task_name,
                    direction="backward",
                    stage="layer_head",
                    step_index=layer_head_index,
                    action="coarse_remove_layer_head",
                    changed_block=None,
                    result=result,
                )
            )
        kept_layer_heads = prune_bottom_fraction(
            _coarse_scores(baseline_quality=full_eval, candidates=layer_head_removal_results, task_family=task_family),
            discard_fraction=args.discard_fraction,
        )

        universe = build_block_universe(
            sample_id=sample.sample_id,
            layers=kept_layers,
            layer_heads=kept_layer_heads,
            span_size=args.span_size,
            max_context_tokens=context_length,
        )
        universe = _limit_blocks(universe, args.max_span_candidates)
        universe_all.extend(universe)

        for baseline_strategy in ("prefix", "random", "uniform"):
            selected = _baseline_blocks(
                universe=universe,
                strategy=baseline_strategy,
                full_kv_bytes=full_kv_bytes,
                target_ratio=0.25,
                seed=args.seed,
            )
            baselines.append(
                eval_selected(
                    selected,
                    method=baseline_strategy,
                    direction="baseline",
                    stage="span",
                    step_index=0,
                )
            )

        current_forward = [
            block
            for block in universe
            if args.prefix_spans > 0 and block.span_id < args.prefix_spans
        ]
        if current_forward:
            result = eval_selected(current_forward, method="forward_greedy", direction="forward", stage="span", step_index=0)
            steps.append(
                OracleStep(
                    sample_id=sample.sample_id,
                    task_name=task_name,
                    direction="forward",
                    stage="span",
                    step_index=0,
                    action="seed",
                    changed_block=None,
                    result=result,
                )
            )
        max_forward_steps = args.max_forward_steps or len(universe)
        for step_index in range(1, max_forward_steps + 1):
            if len(current_forward) >= len(universe):
                break
            changed, result = best_forward_addition(
                current_blocks=current_forward,
                candidate_blocks=universe,
                evaluate=lambda blocks, idx=step_index: eval_selected(
                    blocks,
                    method="forward_greedy",
                    direction="forward",
                    stage="span",
                    step_index=idx,
                ),
                task_family=task_family,
            )
            current_forward = list(result.selected_blocks)
            steps.append(
                OracleStep(
                    sample_id=sample.sample_id,
                    task_name=task_name,
                    direction="forward",
                    stage="span",
                    step_index=step_index,
                    action="add",
                    changed_block=changed,
                    result=result,
                )
            )

        current_backward = list(universe)
        max_backward_steps = args.max_backward_steps or len(universe)
        for step_index in range(1, max_backward_steps + 1):
            if not current_backward:
                break
            changed, result = best_backward_removal(
                current_blocks=current_backward,
                evaluate=lambda blocks, idx=step_index: eval_selected(
                    blocks,
                    method="backward_greedy",
                    direction="backward",
                    stage="span",
                    step_index=idx,
                ),
                task_family=task_family,
            )
            current_backward = list(result.selected_blocks)
            steps.append(
                OracleStep(
                    sample_id=sample.sample_id,
                    task_name=task_name,
                    direction="backward",
                    stage="span",
                    step_index=step_index,
                    action="remove",
                    changed_block=changed,
                    result=result,
                )
            )

    return {
        "baselines": baselines,
        "steps": steps,
        "universe": universe_all,
        "full_eval_by_sample": full_eval_by_sample,
    }


def main() -> None:
    args = parse_args()
    apply_sanity_defaults(args)
    tasks = _parse_tasks(args.tasks)
    if not tasks:
        raise SystemExit("no LongBench tasks requested")
    bundle = load_model_bundle(args.model_name, device_map=args.device_map, dtype=args.dtype)
    for task_name in tasks:
        samples = _load_task_samples(args, task_name)
        if not samples:
            raise SystemExit(f"no samples loaded for task {task_name}")
        result = run_task_oracle(bundle=bundle, task_name=task_name, samples=samples, args=args)
        task_output = args.output_dir / task_name
        write_oracle_artifacts(
            output_dir=task_output,
            task_name=task_name,
            baselines=result["baselines"],
            steps=result["steps"],
            universe=result["universe"],
            full_eval_by_sample=result["full_eval_by_sample"],
            config={**vars(args), "task_name": task_name, "sanity": bool(args.sanity)},
        )


if __name__ == "__main__":
    main()
