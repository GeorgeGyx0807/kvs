from src.kv3d.figures import render_profiling_figures


def test_render_profiling_figures_writes_core_pngs(tmp_path):
    tables = {
        "layer_by_method": [
            {"method": "remove_layer", "layer": 0, "mean_delta_nll": 0.1, "mean_delta_f1": -0.2},
            {"method": "remove_layer", "layer": 1, "mean_delta_nll": 0.4, "mean_delta_f1": -0.1},
        ],
        "block_utility": [
            {"method": "remove_layer_head_chunk", "layer": 0, "head": 0, "chunk": 0, "span_id": 0, "utility_score": 0.4},
            {"method": "remove_layer_head_chunk", "layer": 0, "head": 1, "chunk": 1, "span_id": 1, "utility_score": 0.1},
            {"method": "remove_layer_head_chunk", "layer": 1, "head": 0, "chunk": 0, "span_id": 0, "utility_score": 0.2},
            {"method": "add_layer_head_chunk", "layer": 1, "head": 1, "chunk": 1, "utility_score": 0.3},
        ],
        "span_by_method": [
            {"method": "remove_layer_head_chunk", "span_id": 0, "span_start": 0, "span_end": 16, "mean_delta_nll": 0.2},
            {"method": "remove_layer_head_chunk", "span_id": 1, "span_start": 16, "span_end": 32, "mean_delta_nll": 0.1},
        ],
        "budget_curves": [
            {
                "method": "b_only",
                "mean_budget_ratio": 0.0,
                "mean_metric_f1": 0.1,
                "mean_metric_nll": 2.0,
                "mean_metric_ttft_ms": 10.0,
                "mean_metric_decode_ms": 5.0,
                "mean_selected_kv_bytes": 0,
            },
            {
                "method": "full_kv",
                "mean_budget_ratio": 1.0,
                "mean_metric_f1": 0.8,
                "mean_metric_nll": 1.0,
                "mean_metric_ttft_ms": 100.0,
                "mean_metric_decode_ms": 20.0,
                "mean_selected_kv_bytes": 1000,
            },
        ],
        "stability": [
            {"method": "remove_layer_head_chunk", "top_k": 1, "mean_topk_jaccard": 0.5},
            {"method": "remove_layer_head_chunk", "top_k": 2, "mean_topk_jaccard": 0.75},
        ],
    }

    paths = render_profiling_figures(tables, tmp_path)

    expected_names = {
        "fig_layer_importance.png",
        "fig_layer_head_heatmap.png",
        "fig_chunk_position_heatmap.png",
        "fig_layer_head_span_heatmap.png",
        "fig_span_position_curve.png",
        "fig_top_span_stability.png",
        "fig_budget_quality_curve.png",
        "fig_budget_nll_curve.png",
        "fig_latency_bytes_curve.png",
        "fig_stability.png",
        "fig_attention_divergence.png",
    }
    assert {path.name for path in paths} == expected_names
    assert all(path.stat().st_size > 0 for path in paths)
