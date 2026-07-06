"""Figure rendering for offline 3D KV profiling outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def _numeric(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _save_current(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160)
    plt.close()
    return output


def _plot_layer_importance(tables: dict[str, list[dict[str, Any]]], output: Path) -> Path:
    rows = [row for row in tables.get("layer_by_method", []) if row.get("method") in {"remove_layer", "remove_layer_head_chunk"}]
    if not rows:
        rows = tables.get("layer_by_method", [])
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    if not df.empty:
        df["importance"] = [
            _numeric(row.get("mean_delta_nll"))
            if _numeric(row.get("mean_delta_nll")) is not None
            else -(_numeric(row.get("mean_delta_f1")) or 0.0)
            for row in rows
        ]
        for method, group in df.groupby("method", sort=True):
            group = group.sort_values("layer")
            plt.plot(group["layer"], group["importance"], marker="o", label=str(method))
        plt.legend()
    plt.xlabel("Layer")
    plt.ylabel("Importance")
    plt.title("Layer Importance")
    return _save_current(output)


def _pivot_block_utility(
    tables: dict[str, list[dict[str, Any]]],
    *,
    method: str,
    index: str,
    columns: str,
) -> pd.DataFrame:
    rows = [row for row in tables.get("block_utility", []) if row.get("method") == method and _numeric(row.get("utility_score")) is not None]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(values="utility_score", index=index, columns=columns, aggfunc="mean", fill_value=0.0)


def _plot_heatmap(data: pd.DataFrame, output: Path, *, title: str, xlabel: str, ylabel: str) -> Path:
    plt.figure(figsize=(7, 4.5))
    if not data.empty:
        plt.imshow(data.values, aspect="auto", cmap="viridis")
        plt.colorbar(label="Utility")
        plt.xticks(range(len(data.columns)), data.columns)
        plt.yticks(range(len(data.index)), data.index)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return _save_current(output)


def _plot_layer_head_span_heatmap(tables: dict[str, list[dict[str, Any]]], output: Path) -> Path:
    rows = [
        row
        for row in tables.get("block_utility", [])
        if row.get("method") == "remove_layer_head_chunk" and _numeric(row.get("utility_score")) is not None
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        span_column = "span_id" if "span_id" in df.columns else "chunk"
        df["layer_head"] = [f"L{int(row.layer)}-H{int(row.head)}" for row in df.itertuples()]
        data = df.pivot_table(values="utility_score", index="layer_head", columns=span_column, aggfunc="mean", fill_value=0.0)
    else:
        data = pd.DataFrame()
    return _plot_heatmap(
        data,
        output,
        title="Layer-Head-Span Utility",
        xlabel="Span",
        ylabel="Layer-Head",
    )


def _plot_span_position_curve(tables: dict[str, list[dict[str, Any]]], output: Path) -> Path:
    rows = [row for row in tables.get("span_by_method", []) if row.get("method") == "remove_layer_head_chunk"]
    if not rows:
        rows = [
            row
            for row in tables.get("chunk_by_method", [])
            if row.get("method") == "remove_layer_head_chunk"
        ]
    df = pd.DataFrame(rows)
    plt.figure(figsize=(7, 4))
    if not df.empty:
        span_column = "span_id" if "span_id" in df.columns else "chunk"
        metric = "mean_delta_nll" if "mean_delta_nll" in df.columns else "mean_delta_f1"
        if metric in df.columns:
            df = df.dropna(subset=[span_column, metric]).sort_values(span_column)
            plt.plot(df[span_column], df[metric], marker="o")
    plt.xlabel("Span")
    plt.ylabel("Mean delta NLL")
    plt.title("Span Position Utility")
    return _save_current(output)


def _plot_top_span_stability(tables: dict[str, list[dict[str, Any]]], output: Path) -> Path:
    df = pd.DataFrame(
        [row for row in tables.get("stability", []) if row.get("method") == "remove_layer_head_chunk"]
    )
    plt.figure(figsize=(7, 4))
    if not df.empty:
        df = df.sort_values("top_k")
        plt.plot(df["top_k"], df["mean_topk_jaccard"], marker="o")
    plt.xlabel("Top-k spans")
    plt.ylabel("Mean Jaccard")
    plt.title("Top Span Stability")
    return _save_current(output)


def _plot_budget_metric(
    tables: dict[str, list[dict[str, Any]]],
    output: Path,
    *,
    metric: str,
    ylabel: str,
    title: str,
) -> Path:
    df = pd.DataFrame(tables.get("budget_curves", []))
    plt.figure(figsize=(7, 4))
    if not df.empty and "mean_budget_ratio" in df and metric in df:
        df = df.dropna(subset=["mean_budget_ratio", metric]).sort_values("mean_budget_ratio")
        plt.scatter(df["mean_budget_ratio"], df[metric])
        for _, row in df.iterrows():
            plt.annotate(str(row.get("method", "")), (row["mean_budget_ratio"], row[metric]), fontsize=7)
    plt.xlabel("KV budget ratio")
    plt.ylabel(ylabel)
    plt.title(title)
    return _save_current(output)


def _plot_latency(tables: dict[str, list[dict[str, Any]]], output: Path) -> Path:
    df = pd.DataFrame(tables.get("budget_curves", []))
    plt.figure(figsize=(7, 4))
    if not df.empty and "mean_selected_kv_bytes" in df:
        for metric, label in [("mean_metric_ttft_ms", "TTFT"), ("mean_metric_decode_ms", "Decode")]:
            if metric in df:
                metric_df = df.dropna(subset=["mean_selected_kv_bytes", metric]).sort_values("mean_selected_kv_bytes")
                plt.plot(metric_df["mean_selected_kv_bytes"], metric_df[metric], marker="o", label=label)
        if plt.gca().has_data():
            plt.legend()
    plt.xlabel("Selected KV bytes")
    plt.ylabel("Milliseconds")
    plt.title("KV Bytes vs Latency")
    return _save_current(output)


def _plot_stability(tables: dict[str, list[dict[str, Any]]], output: Path) -> Path:
    df = pd.DataFrame(tables.get("stability", []))
    plt.figure(figsize=(7, 4))
    if not df.empty:
        for method, group in df.groupby("method", sort=True):
            group = group.sort_values("top_k")
            plt.plot(group["top_k"], group["mean_topk_jaccard"], marker="o", label=str(method))
        plt.legend()
    plt.xlabel("Top-k blocks")
    plt.ylabel("Mean Jaccard")
    plt.title("Top-k Stability")
    return _save_current(output)


def _plot_attention_divergence(tables: dict[str, list[dict[str, Any]]], output: Path) -> Path:
    df = pd.DataFrame(tables.get("attention_divergence", []))
    plt.figure(figsize=(7, 4))
    if not df.empty and "attention_js_divergence" in df and "delta_nll" in df:
        df = df.dropna(subset=["attention_js_divergence", "delta_nll"])
        if not df.empty:
            plt.scatter(df["attention_js_divergence"], df["delta_nll"], s=18, alpha=0.8)
            for _, row in df.iterrows():
                plt.annotate(str(row.get("method", "")), (row["attention_js_divergence"], row["delta_nll"]), fontsize=6)
    plt.xlabel("Attention JS divergence")
    plt.ylabel("Delta NLL")
    plt.title("Attention Divergence vs Utility")
    return _save_current(output)


def render_profiling_figures(tables: dict[str, list[dict[str, Any]]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        _plot_layer_importance(tables, output_dir / "fig_layer_importance.png"),
        _plot_heatmap(
            _pivot_block_utility(tables, method="remove_layer_head_chunk", index="layer", columns="head"),
            output_dir / "fig_layer_head_heatmap.png",
            title="Layer-Head Utility",
            xlabel="Head",
            ylabel="Layer",
        ),
        _plot_heatmap(
            _pivot_block_utility(tables, method="remove_layer_head_chunk", index="layer", columns="chunk"),
            output_dir / "fig_chunk_position_heatmap.png",
            title="Chunk Position Utility",
            xlabel="Chunk",
            ylabel="Layer",
        ),
        _plot_layer_head_span_heatmap(tables, output_dir / "fig_layer_head_span_heatmap.png"),
        _plot_span_position_curve(tables, output_dir / "fig_span_position_curve.png"),
        _plot_top_span_stability(tables, output_dir / "fig_top_span_stability.png"),
        _plot_budget_metric(
            tables,
            output_dir / "fig_budget_quality_curve.png",
            metric="mean_metric_f1",
            ylabel="F1",
            title="Budget vs Quality",
        ),
        _plot_budget_metric(
            tables,
            output_dir / "fig_budget_nll_curve.png",
            metric="mean_metric_nll",
            ylabel="NLL",
            title="Budget vs NLL",
        ),
        _plot_latency(tables, output_dir / "fig_latency_bytes_curve.png"),
        _plot_stability(tables, output_dir / "fig_stability.png"),
        _plot_attention_divergence(tables, output_dir / "fig_attention_divergence.png"),
    ]
    return paths
