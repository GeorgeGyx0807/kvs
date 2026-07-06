"""Markdown report rendering for offline 3D KV profiling."""

from __future__ import annotations

from typing import Any


def _render_table(title: str, rows: list[dict[str, Any]]) -> str:
    lines = [f"## {title}", ""]
    if not rows:
        lines.append("_No rows._")
        lines.append("")
        return "\n".join(lines)

    headers = list(rows[0].keys())
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def _score_from_row(row: dict[str, Any]) -> float | None:
    delta_nll = row.get("mean_delta_nll")
    if delta_nll is not None:
        return float(delta_nll)
    delta_f1 = row.get("mean_delta_f1")
    if delta_f1 is not None:
        return -float(delta_f1)
    utility = row.get("utility_score")
    if utility is not None:
        return float(utility)
    return None


def _top_row(rows: list[dict[str, Any]], *, method: str | None = None) -> dict[str, Any] | None:
    candidates = [row for row in rows if method is None or row.get("method") == method]
    scored = [(score, row) for row in candidates if (score := _score_from_row(row)) is not None]
    if not scored:
        return candidates[0] if candidates else None
    return max(scored, key=lambda item: item[0])[1]


def build_key_findings(tables: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    return {
        "top_layer": _top_row(tables.get("layer_by_method", []), method="remove_layer") or _top_row(tables.get("layer", [])),
        "top_head": _top_row(tables.get("head_by_method", []), method="remove_layer_head") or _top_row(tables.get("head", [])),
        "top_chunk": _top_row(tables.get("chunk_by_method", []), method="remove_layer_head_chunk")
        or _top_row(tables.get("chunk", [])),
        "top_span": _top_row(tables.get("span_by_method", []), method="remove_layer_head_chunk"),
        "top_block": _top_row(tables.get("block_utility", []), method="remove_layer_head_chunk"),
        "stability": _top_row(tables.get("stability", []), method="remove_layer_head_chunk"),
        "task_stability": _top_row(tables.get("stability_by_task", []), method="remove_layer_head_chunk"),
        "heterogeneity": _top_row(tables.get("heterogeneity", []), method="remove_layer_head_chunk"),
        "addition_removal_alignment": _top_row(tables.get("addition_removal_alignment", [])),
        "attention_divergence": _top_row(tables.get("attention_divergence_by_method", [])),
        "attention_correlation": _top_row(tables.get("attention_correlation_matrix", [])),
    }


def _render_key_findings(tables: dict[str, list[dict[str, Any]]]) -> str:
    lines = ["## Key Findings", ""]
    findings = build_key_findings(tables)
    layer = findings["top_layer"]
    head = findings["top_head"]
    chunk = findings["top_chunk"]
    span = findings["top_span"]
    block = findings["top_block"]
    stability = findings["stability"]
    heterogeneity = findings["heterogeneity"]
    attention = findings["attention_divergence"]

    if layer is not None:
        lines.append(
            f"- Top layer candidate: layer {layer.get('layer')} "
            f"(mean_delta_nll={layer.get('mean_delta_nll')}, mean_delta_f1={layer.get('mean_delta_f1')})."
        )
    if head is not None:
        lines.append(
            f"- Top head candidate: head {head.get('head')} "
            f"(mean_delta_nll={head.get('mean_delta_nll')}, mean_delta_f1={head.get('mean_delta_f1')})."
        )
    if chunk is not None:
        lines.append(
            f"- Top chunk candidate: chunk {chunk.get('chunk')} "
            f"(mean_delta_nll={chunk.get('mean_delta_nll')}, mean_delta_f1={chunk.get('mean_delta_f1')})."
        )
    if span is not None:
        lines.append(
            f"- Top span candidate: span {span.get('span_id')} "
            f"({span.get('span_start')}-{span.get('span_end')}, "
            f"mean_delta_nll={span.get('mean_delta_nll')}, mean_delta_f1={span.get('mean_delta_f1')})."
        )
    if block is not None:
        lines.append(
            f"- Top layer-head-chunk block: layer {block.get('layer')}, head {block.get('head')}, "
            f"chunk {block.get('chunk')} (utility_score={block.get('utility_score')})."
        )
    if stability is not None:
        lines.append(
            f"- Stability snapshot: top-{stability.get('top_k')} mean Jaccard is "
            f"{stability.get('mean_topk_jaccard')} for {stability.get('method')}."
        )
    if heterogeneity is not None:
        lines.append(
            f"- Heterogeneity snapshot: {heterogeneity.get('axis')} range is "
            f"{heterogeneity.get('range_mean_utility')} across "
            f"{heterogeneity.get('axis_value_count')} values "
            f"(top={heterogeneity.get('top_axis_value')}, bottom={heterogeneity.get('bottom_axis_value')})."
        )
    else:
        lines.append("- Heterogeneity snapshot: insufficient axis diversity for a range estimate.")
    if attention is not None:
        lines.append(
            f"- Attention divergence snapshot: {attention.get('method')} "
            f"mean JS={attention.get('mean_attention_js_divergence')} "
            f"mean KL={attention.get('mean_attention_kl_divergence')}."
        )
    if len(lines) == 2:
        lines.append("- No block-level profiling rows are available yet.")
    lines.append("")
    return "\n".join(lines)


def _render_figures() -> str:
    figure_names = [
        "figures/fig_layer_importance.png",
        "figures/fig_layer_head_heatmap.png",
        "figures/fig_chunk_position_heatmap.png",
        "figures/fig_layer_head_span_heatmap.png",
        "figures/fig_span_position_curve.png",
        "figures/fig_top_span_stability.png",
        "figures/fig_budget_quality_curve.png",
        "figures/fig_budget_nll_curve.png",
        "figures/fig_latency_bytes_curve.png",
        "figures/fig_stability.png",
        "figures/fig_attention_divergence.png",
    ]
    lines = ["## Figures", ""]
    lines.extend(f"- {name}" for name in figure_names)
    lines.append("")
    return "\n".join(lines)


def render_profiling_report(
    plan: list[dict[str, Any]],
    records: list[Any],
    tables: dict[str, list[dict[str, Any]]],
    experiment_name: str,
    model_name: str,
    main_dataset: str,
) -> str:
    lines = [
        "# Offline 3D KV Utility Profiling Report",
        "",
        f"- Experiment: {experiment_name}",
        f"- Model: {model_name}",
        f"- Main dataset: {main_dataset}",
        f"- Plan size: {len(plan)}",
        f"- Record count: {len(records)}",
        "",
        "## Scope",
        "",
        "This phase profiles KV utility across layer, head, and chunk dimensions.",
        "It prioritizes removal/addition profiling over budgeted selection.",
        "",
        _render_key_findings(tables),
        _render_figures(),
        _render_table("Layer Profile", tables.get("layer", [])),
        _render_table("Layer Profile by Method", tables.get("layer_by_method", [])),
        _render_table("Head Profile", tables.get("head", [])),
        _render_table("Head Profile by Method", tables.get("head_by_method", [])),
        _render_table("Chunk Profile", tables.get("chunk", [])),
        _render_table("Chunk Profile by Method", tables.get("chunk_by_method", [])),
        _render_table("Span Profile by Method", tables.get("span_by_method", [])),
        _render_table("Block Utility", tables.get("block_utility", [])),
        _render_table("Attention Divergence", tables.get("attention_divergence", [])),
        _render_table("Attention Divergence by Method", tables.get("attention_divergence_by_method", [])),
        _render_table("Attention Correlation Matrix", tables.get("attention_correlation_matrix", [])),
        _render_table("Top-k Stability", tables.get("stability", [])),
        _render_table("Top-k Stability by Task", tables.get("stability_by_task", [])),
        _render_table("Heterogeneity Summary", tables.get("heterogeneity", [])),
        _render_table("Addition vs Removal Alignment", tables.get("addition_removal_alignment", [])),
        _render_table("Observed Budget Curves", tables.get("budget_curves", [])),
    ]
    return "\n".join(lines).rstrip() + "\n"
