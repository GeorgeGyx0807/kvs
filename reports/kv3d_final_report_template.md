# Offline 3D KV Utility Profiling Final Report

> This report is for the completed 20-sample GPU pilot and should only be filled after the pilot run has been validated.

## Experiment Summary

- Model:
- Dataset:
- Task:
- Samples:
- Layers:
- KV heads:
- Chunks:
- Chunk size:
- Include addition profiling:

## Main Findings

### Layer heterogeneity

### Head heterogeneity

### Chunk heterogeneity

### Removal vs addition alignment

### Cross-sample and cross-task stability

## Evidence Tables

- `layer_importance.csv`
- `head_importance.csv`
- `chunk_importance.csv`
- `block_utility.csv`
- `stability.csv`
- `stability_by_task.csv`
- `heterogeneity.csv`
- `addition_removal_alignment.csv`
- `budget_curves.csv`

## Figures

- `figures/fig_layer_importance.png`
- `figures/fig_layer_head_heatmap.png`
- `figures/fig_chunk_position_heatmap.png`
- `figures/fig_budget_quality_curve.png`
- `figures/fig_budget_nll_curve.png`
- `figures/fig_latency_bytes_curve.png`
- `figures/fig_stability.png`

## Validation

- Validation command:
- Validation result:
- Issues:

## Notes

- This report should summarize the offline profiling evidence only.
- It should not claim a finalized budget selector or online scheduler.
