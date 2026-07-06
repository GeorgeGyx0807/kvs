"""GPU executor for offline 3D KV profiling."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .attention import AttentionChunkProfile
from .attention import attention_chunk_profile
from .attention import compare_attention_profiles
from .blocks import KV3DKey
from .blocks import chunk_bounds
from .masks import SequentialLayerMaskDict, build_decode_layer_masks, build_prefill_layer_masks, selection_kv_bytes
from .metrics import contains_answer, token_f1
from .records import KV3DAttentionDivergenceSnapshot, KV3DMetricSnapshot, KV3DProfilingRecord
from .runner import ProfilingSample


@dataclass(frozen=True)
class ModelBundle:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


@dataclass(frozen=True)
class GreedyGenerationResult:
    text: str
    prefill_ms: float
    ttft_ms: float
    decode_ms: float
    attention_profile: AttentionChunkProfile | None = None


def load_model_bundle(
    model_name: str,
    *,
    device_map: str = "auto",
    dtype: str = "bfloat16",
) -> ModelBundle:
    torch_dtype = getattr(torch, dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    return ModelBundle(model=model, tokenizer=tokenizer)


def _device_of(model) -> torch.device:
    return next(model.parameters()).device


def format_context_prompt(sample: ProfilingSample) -> str:
    return f"{sample.context}\n\nQuestion: {sample.prompt}\nAnswer:"


def format_question_prompt(sample: ProfilingSample) -> str:
    return f"Question: {sample.prompt}\nAnswer:"


def _encode(tokenizer, text: str, device: torch.device, max_tokens: int | None = None) -> torch.Tensor:
    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=max_tokens is not None,
        max_length=max_tokens,
    )
    return encoded["input_ids"].to(device)


def _context_and_question_ids(
    tokenizer,
    sample: ProfilingSample,
    device: torch.device,
    max_context_tokens: int,
) -> tuple[torch.Tensor, int]:
    context_ids = _encode(tokenizer, sample.context, device, max_tokens=max_context_tokens)
    suffix_ids = _encode(tokenizer, f"\n\nQuestion: {sample.prompt}\nAnswer:", device)
    return torch.cat([context_ids, suffix_ids], dim=-1), int(context_ids.shape[-1])


def _synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _clone_past_key_values(past_key_values):
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "layers"):
        from transformers.cache_utils import DynamicCache

        cache_data = []
        for layer in past_key_values.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                continue
            cache_data.append((keys.clone(), values.clone()))
        return DynamicCache(cache_data)
    return tuple((keys.clone(), values.clone()) for keys, values in past_key_values)


def _prefill_context_cache(*, model, input_ids: torch.Tensor):
    device = input_ids.device
    _synchronize_if_needed(device)
    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            use_cache=True,
            logits_to_keep=1,
        )
    _synchronize_if_needed(device)
    return outputs.past_key_values, (time.perf_counter() - start) * 1000


def _generate_greedy_from_context_cache(
    *,
    model,
    tokenizer,
    context_cache,
    suffix_ids: torch.Tensor,
    context_length: int,
    max_new_tokens: int,
    chunk_size: int | None = None,
    kept_blocks: Iterable[KV3DKey] | None = None,
    removed_blocks: Iterable[KV3DKey] | None = None,
) -> GreedyGenerationResult:
    device = suffix_ids.device
    dtype = getattr(model, "dtype", torch.float32)
    num_layers = int(model.config.num_hidden_layers)
    num_attention_heads = int(model.config.num_attention_heads)
    num_key_value_heads = int(model.config.num_key_value_heads)
    generated_tokens: list[torch.Tensor] = []
    past_key_values = _clone_past_key_values(context_cache)
    next_input = suffix_ids
    prefill_ms = 0.0
    decode_ms = 0.0

    for step in range(max_new_tokens):
        past_length = int(past_key_values.get_seq_length()) if past_key_values is not None else context_length + step
        total_kv_length = past_length + int(next_input.shape[-1])
        if chunk_size is not None and (kept_blocks is not None or removed_blocks is not None):
            masks = build_decode_layer_masks(
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                context_length=context_length,
                total_kv_length=total_kv_length,
                query_length=int(next_input.shape[-1]),
                chunk_size=chunk_size,
                kept_blocks=kept_blocks,
                removed_blocks=removed_blocks,
                device=device,
                dtype=dtype,
            )
            attention_mask = SequentialLayerMaskDict(masks)
        else:
            attention_mask = None

        _synchronize_if_needed(device)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs = model(
                input_ids=next_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
        _synchronize_if_needed(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if step == 0:
            prefill_ms += elapsed_ms
        else:
            decode_ms += elapsed_ms
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        next_input = next_token
        if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
            break

    if generated_tokens:
        generated = torch.cat(generated_tokens, dim=-1)[0]
    else:
        generated = torch.empty((0,), dtype=suffix_ids.dtype, device=device)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return GreedyGenerationResult(text=text, prefill_ms=prefill_ms, ttft_ms=prefill_ms + decode_ms, decode_ms=decode_ms)


def _generate_greedy(
    *,
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int | None = None,
    chunk_size: int | None = None,
    kept_blocks: Iterable[KV3DKey] | None = None,
    removed_blocks: Iterable[KV3DKey] | None = None,
) -> GreedyGenerationResult:
    device = input_ids.device
    dtype = getattr(model, "dtype", torch.float32)
    prompt_len = int(input_ids.shape[-1])
    num_layers = int(model.config.num_hidden_layers)
    num_attention_heads = int(model.config.num_attention_heads)
    num_key_value_heads = int(model.config.num_key_value_heads)
    generated_tokens: list[torch.Tensor] = []
    past_key_values = None
    next_input = input_ids
    prefill_ms = 0.0
    decode_ms = 0.0
    attention_profile: AttentionChunkProfile | None = None

    for step in range(max_new_tokens):
        if step == 0:
            if context_length is not None and chunk_size is not None and (kept_blocks is not None or removed_blocks is not None):
                masks = build_prefill_layer_masks(
                    num_layers=num_layers,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    context_length=context_length,
                    total_length=prompt_len,
                    chunk_size=chunk_size,
                    kept_blocks=kept_blocks,
                    removed_blocks=removed_blocks,
                    device=device,
                    dtype=dtype,
                )
                attention_mask = SequentialLayerMaskDict(masks)
            else:
                attention_mask = None
        else:
            past_length = int(past_key_values.get_seq_length()) if past_key_values is not None else prompt_len + step - 1
            total_kv_length = past_length + int(next_input.shape[-1])
            if context_length is not None and chunk_size is not None and (kept_blocks is not None or removed_blocks is not None):
                masks = build_decode_layer_masks(
                    num_layers=num_layers,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    context_length=context_length,
                    total_kv_length=total_kv_length,
                    query_length=int(next_input.shape[-1]),
                    chunk_size=chunk_size,
                    kept_blocks=kept_blocks,
                    removed_blocks=removed_blocks,
                    device=device,
                    dtype=dtype,
                )
                attention_mask = SequentialLayerMaskDict(masks)
            else:
                attention_mask = None

        _synchronize_if_needed(device)
        start = time.perf_counter()
        capture_attentions = step == 0 and context_length is not None and chunk_size is not None
        with torch.inference_mode():
            outputs = model(
                input_ids=next_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
                output_attentions=capture_attentions,
            )
        _synchronize_if_needed(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if step == 0:
            prefill_ms += elapsed_ms
            if capture_attentions and outputs.attentions is not None:
                attention_profile = attention_chunk_profile(
                    outputs.attentions,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    chunk_size=chunk_size,
                    context_length=context_length,
                )
        else:
            decode_ms += elapsed_ms
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        next_input = next_token
        if tokenizer.eos_token_id is not None and int(next_token.item()) == int(tokenizer.eos_token_id):
            break

    if generated_tokens:
        generated = torch.cat(generated_tokens, dim=-1)[0]
    else:
        generated = torch.empty((0,), dtype=input_ids.dtype, device=device)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return GreedyGenerationResult(
        text=text,
        prefill_ms=prefill_ms,
        ttft_ms=prefill_ms + decode_ms,
        decode_ms=decode_ms,
        attention_profile=attention_profile,
    )


def _metric_for_prediction(
    *,
    prediction: str,
    answers: tuple[str, ...],
    nll: float | None,
    ttft_ms: float | None,
    prefill_ms: float | None,
    decode_ms: float | None,
) -> KV3DMetricSnapshot:
    contains = contains_answer(prediction, answers)
    f1 = token_f1(prediction, answers)
    return KV3DMetricSnapshot(
        accuracy=contains,
        f1=f1,
        contains=contains,
        nll=nll,
        ttft_ms=ttft_ms,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
    )


def _answer_nll(
    *,
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    answer: str,
    context_length: int | None = None,
    chunk_size: int | None = None,
    kept_blocks: Iterable[KV3DKey] | None = None,
    removed_blocks: Iterable[KV3DKey] | None = None,
) -> float | None:
    if not answer:
        return None
    device = prompt_ids.device
    answer_ids = _encode(tokenizer, " " + answer, device)
    if answer_ids.numel() == 0:
        return None
    input_ids = torch.cat([prompt_ids, answer_ids], dim=-1)
    labels = torch.full_like(input_ids, -100)
    labels[:, prompt_ids.shape[-1] :] = input_ids[:, prompt_ids.shape[-1] :]
    attention_mask = None
    if context_length is not None and chunk_size is not None and (kept_blocks is not None or removed_blocks is not None):
        masks = build_prefill_layer_masks(
            num_layers=int(model.config.num_hidden_layers),
            num_attention_heads=int(model.config.num_attention_heads),
            num_key_value_heads=int(model.config.num_key_value_heads),
            context_length=context_length,
            total_length=int(input_ids.shape[-1]),
            chunk_size=chunk_size,
            kept_blocks=kept_blocks,
            removed_blocks=removed_blocks,
            device=device,
            dtype=getattr(model, "dtype", torch.float32),
        )
        attention_mask = SequentialLayerMaskDict(masks)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
    if outputs.loss is None:
        return None
    return float(outputs.loss.detach().float().item())


def _answer_nll_from_context_cache(
    *,
    model,
    tokenizer,
    context_cache,
    suffix_ids: torch.Tensor,
    answer: str,
    context_length: int,
    chunk_size: int | None = None,
    kept_blocks: Iterable[KV3DKey] | None = None,
    removed_blocks: Iterable[KV3DKey] | None = None,
) -> float | None:
    if not answer:
        return None
    device = suffix_ids.device
    answer_ids = _encode(tokenizer, " " + answer, device)
    if answer_ids.numel() == 0:
        return None
    input_ids = torch.cat([suffix_ids, answer_ids], dim=-1)
    labels = torch.full_like(input_ids, -100)
    labels[:, suffix_ids.shape[-1] :] = input_ids[:, suffix_ids.shape[-1] :]
    past_key_values = _clone_past_key_values(context_cache)
    past_length = int(past_key_values.get_seq_length()) if past_key_values is not None else context_length
    total_kv_length = past_length + int(input_ids.shape[-1])
    attention_mask = None
    if chunk_size is not None and (kept_blocks is not None or removed_blocks is not None):
        masks = build_decode_layer_masks(
            num_layers=int(model.config.num_hidden_layers),
            num_attention_heads=int(model.config.num_attention_heads),
            num_key_value_heads=int(model.config.num_key_value_heads),
            context_length=context_length,
            total_kv_length=total_kv_length,
            query_length=int(input_ids.shape[-1]),
            chunk_size=chunk_size,
            kept_blocks=kept_blocks,
            removed_blocks=removed_blocks,
            device=device,
            dtype=getattr(model, "dtype", torch.float32),
        )
        attention_mask = SequentialLayerMaskDict(masks)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            use_cache=False,
        )
    if outputs.loss is None:
        return None
    return float(outputs.loss.detach().float().item())


def _full_blocks(sample_id: str, num_layers: int, num_key_value_heads: int, num_chunks: int) -> list[KV3DKey]:
    return [
        KV3DKey(sample_id=sample_id, layer=layer, head=head, chunk=chunk)
        for layer in range(num_layers)
        for head in range(num_key_value_heads)
        for chunk in range(num_chunks)
    ]


def _removed_blocks(
    *,
    sample_id: str,
    method: str,
    layer: int | None,
    head: int | None,
    chunk: int | None,
    num_key_value_heads: int,
    num_chunks: int,
) -> list[KV3DKey]:
    if method == "remove_layer":
        return [
            KV3DKey(sample_id=sample_id, layer=int(layer), head=head_idx, chunk=chunk_idx)
            for head_idx in range(num_key_value_heads)
            for chunk_idx in range(num_chunks)
        ]
    if method == "remove_layer_head":
        return [
            KV3DKey(sample_id=sample_id, layer=int(layer), head=int(head), chunk=chunk_idx)
            for chunk_idx in range(num_chunks)
        ]
    if method == "remove_layer_head_chunk":
        return [KV3DKey(sample_id=sample_id, layer=int(layer), head=int(head), chunk=int(chunk))]
    return []


def _addition_blocks(
    *,
    sample_id: str,
    method: str,
    layer: int | None,
    head: int | None,
    chunk: int | None,
) -> list[KV3DKey]:
    if method == "add_layer_head_chunk":
        return [KV3DKey(sample_id=sample_id, layer=int(layer), head=int(head), chunk=int(chunk))]
    return []


def _record_key_for_spec(
    *,
    sample_id: str,
    layer: int | None,
    head: int | None,
    chunk: int | None,
    chunk_size: int,
    context_length: int,
) -> KV3DKey | None:
    if layer is None:
        return None
    chunk_index = int(chunk) if chunk is not None else 0
    token_start, token_end = chunk_bounds(chunk_index, chunk_size)
    return KV3DKey(
        sample_id=sample_id,
        layer=int(layer),
        head=int(head) if head is not None else 0,
        chunk=chunk_index,
        token_start=min(token_start, context_length),
        token_end=min(token_end, context_length),
    )


def _delta(metric: KV3DMetricSnapshot, baseline: KV3DMetricSnapshot) -> KV3DMetricSnapshot:
    def diff(value: float | None, base: float | None) -> float | None:
        if value is None or base is None:
            return None
        return value - base

    return KV3DMetricSnapshot(
        accuracy=diff(metric.accuracy, baseline.accuracy),
        f1=diff(metric.f1, baseline.f1),
        contains=diff(metric.contains, baseline.contains),
        nll=diff(metric.nll, baseline.nll),
        ttft_ms=diff(metric.ttft_ms, baseline.ttft_ms),
        prefill_ms=diff(metric.prefill_ms, baseline.prefill_ms),
        decode_ms=diff(metric.decode_ms, baseline.decode_ms),
    )


def _attention_divergence_from_profiles(
    *,
    full_profile: AttentionChunkProfile | None,
    current_profile: AttentionChunkProfile | None,
    method: str,
    layer: int | None,
    head: int | None,
) -> KV3DAttentionDivergenceSnapshot | None:
    if not method.startswith(("remove_", "add_")):
        return None
    return compare_attention_profiles(full_profile, current_profile, layer=layer, head=head)


def run_model_profiling_plan(
    *,
    bundle: ModelBundle,
    samples: Iterable[ProfilingSample],
    plan: Iterable[dict],
    chunk_size: int,
    max_context_tokens: int,
    max_new_tokens: int,
) -> list[KV3DProfilingRecord]:
    model = bundle.model
    tokenizer = bundle.tokenizer
    device = _device_of(model)
    num_layers = int(model.config.num_hidden_layers)
    num_attention_heads = int(model.config.num_attention_heads)
    num_key_value_heads = int(model.config.num_key_value_heads)
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads))
    records: list[KV3DProfilingRecord] = []
    plan_by_sample: dict[str, list[dict]] = {}
    for item in plan:
        plan_by_sample.setdefault(str(item["sample_id"]), []).append(item)

    for sample in samples:
        context_input_ids, context_length = _context_and_question_ids(
            tokenizer,
            sample,
            device,
            max_context_tokens=max_context_tokens,
        )
        seq_len = int(context_input_ids.shape[-1])
        num_chunks = (context_length + chunk_size - 1) // chunk_size
        all_blocks = _full_blocks(sample.sample_id, num_layers, num_key_value_heads, num_chunks)

        full_generation = _generate_greedy(
            model=model,
            tokenizer=tokenizer,
            input_ids=context_input_ids,
            max_new_tokens=max_new_tokens,
            context_length=context_length,
            chunk_size=chunk_size,
        )
        full_nll = _answer_nll(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=context_input_ids,
            answer=sample.gold_answer,
        )
        full_metric = _metric_for_prediction(
            prediction=full_generation.text,
            answers=sample.answers,
            nll=full_nll,
            ttft_ms=full_generation.ttft_ms,
            prefill_ms=full_generation.prefill_ms,
            decode_ms=full_generation.decode_ms,
        )
        full_bytes = selection_kv_bytes(selected_blocks=all_blocks, seq_len=context_length, head_dim=head_dim, chunk_size=chunk_size)
        records.append(
            KV3DProfilingRecord(
                sample_id=sample.sample_id,
                method="full_kv",
                key=None,
                selected_kv_bytes=full_bytes,
                metric=full_metric,
            )
        )

        question_input_ids = _encode(tokenizer, format_question_prompt(sample), device)
        b_generation = _generate_greedy(
            model=model,
            tokenizer=tokenizer,
            input_ids=question_input_ids,
            max_new_tokens=max_new_tokens,
        )
        b_nll = _answer_nll(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=question_input_ids,
            answer=sample.gold_answer,
        )
        b_metric = _metric_for_prediction(
            prediction=b_generation.text,
            answers=sample.answers,
            nll=b_nll,
            ttft_ms=b_generation.ttft_ms,
            prefill_ms=b_generation.prefill_ms,
            decode_ms=b_generation.decode_ms,
        )
        records.append(
            KV3DProfilingRecord(
                sample_id=sample.sample_id,
                method="b_only",
                key=None,
                selected_kv_bytes=0,
                metric=b_metric,
                delta_vs_full=_delta(b_metric, full_metric),
            )
        )

        for spec in plan_by_sample.get(sample.sample_id, []):
            method = str(spec["method"])
            if method in {"full_kv", "b_only"}:
                continue
            if not (method.startswith("remove_") or method.startswith("add_")):
                continue
            layer = spec.get("layer")
            head = spec.get("head")
            chunk = spec.get("chunk")
            removed: list[KV3DKey] | None = None
            kept: list[KV3DKey] | None = None
            delta_vs_base = None
            if method.startswith("remove_"):
                removed = _removed_blocks(
                    sample_id=sample.sample_id,
                    method=method,
                    layer=layer,
                    head=head,
                    chunk=chunk,
                    num_key_value_heads=num_key_value_heads,
                    num_chunks=num_chunks,
                )
                removed_lookup = {(block.layer, block.head, block.chunk) for block in removed}
                kept = [
                    block for block in all_blocks
                    if (block.layer, block.head, block.chunk) not in removed_lookup
                ]
            else:
                kept = _addition_blocks(
                    sample_id=sample.sample_id,
                    method=method,
                    layer=layer,
                    head=head,
                    chunk=chunk,
                )
            generation = _generate_greedy(
                model=model,
                tokenizer=tokenizer,
                input_ids=context_input_ids,
                max_new_tokens=max_new_tokens,
                context_length=context_length,
                chunk_size=chunk_size,
                kept_blocks=kept if method.startswith("add_") else None,
                removed_blocks=removed,
            )
            nll = _answer_nll(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=context_input_ids,
                answer=sample.gold_answer,
                context_length=context_length,
                chunk_size=chunk_size,
                kept_blocks=kept if method.startswith("add_") else None,
                removed_blocks=removed,
            )
            metric = _metric_for_prediction(
                prediction=generation.text,
                answers=sample.answers,
                nll=nll,
                ttft_ms=generation.ttft_ms,
                prefill_ms=generation.prefill_ms,
                decode_ms=generation.decode_ms,
            )
            if method.startswith("add_"):
                delta_vs_base = _delta(metric, b_metric)
            key = _record_key_for_spec(
                sample_id=sample.sample_id,
                layer=layer,
                head=head,
                chunk=chunk,
                chunk_size=chunk_size,
                context_length=context_length,
            )
            attention_divergence = _attention_divergence_from_profiles(
                full_profile=full_generation.attention_profile,
                current_profile=generation.attention_profile,
                method=method,
                layer=int(layer) if layer is not None else None,
                head=int(head) if head is not None else None,
            )
            records.append(
                KV3DProfilingRecord(
                    sample_id=sample.sample_id,
                    method=method,
                    key=key,
                    selected_kv_bytes=selection_kv_bytes(
                        selected_blocks=kept,
                        seq_len=context_length,
                        head_dim=head_dim,
                        chunk_size=chunk_size,
                    ),
                    metric=metric,
                    delta_vs_full=_delta(metric, full_metric),
                    delta_vs_base=delta_vs_base,
                    attention_divergence=attention_divergence,
                )
            )
    return records
