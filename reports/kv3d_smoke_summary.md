# Offline 3D KV Profiling Smoke Summary

## Status

This repository now has a runnable offline 3D KV profiling smoke path for `Qwen/Qwen3-8B` on LongBench samples.

This is not yet a formal profiling result over all layers, heads, chunks, samples, and tasks. It is a verified smoke run proving that the data loader, model executor, layer-head-chunk masking, metric recording, aggregation, and report generation can execute end to end.

## Data

- Source: Hugging Face `THUDM/LongBench`
- Config: `narrativeqa`
- Split: `test`
- Local file: `data/longbench_narrativeqa_samples.json`
- Loaded samples: 200

The loader reads official LongBench `data.zip` directly because the installed `datasets` version rejects the legacy script dataset path.

## Model

- Model: `Qwen/Qwen3-8B`
- Agents A/B: same checkpoint and tokenizer
- Qwen3 config observed locally:
  - layers: 36
  - attention heads: 32
  - KV heads: 8
  - head dim: 128

## Verified Smoke Command

```bash
python3 scripts/run_kv3d_gpu_profile.py \
  --samples data/longbench_narrativeqa_samples.json \
  --model-name Qwen/Qwen3-8B \
  --max-samples 1 \
  --max-context-tokens 256 \
  --max-new-tokens 2 \
  --chunk-size 128 \
  --max-layers 1 \
  --max-heads 1 \
  --output-dir outputs/kv3d_smoke_nll
```

## Outputs

- `outputs/kv3d_smoke_nll/records.jsonl`
- `outputs/kv3d_smoke_nll/tables.json`
- `outputs/kv3d_smoke_nll/report.md`
- `outputs/kv3d_smoke_nll/summary.json`
- `outputs/kv3d_smoke_nll/plan.json`
- `outputs/kv3d_smoke_nll/samples.json`

The smoke produced 6 records:

- `full_kv`
- `b_only`
- `remove_layer`
- `remove_layer_head`
- two `remove_layer_head_chunk` records

All recorded NLL values in the latest smoke are finite.

## Current Limitation

The formal objective still requires larger offline profiling over enough layers, KV heads, chunks, samples, and tasks to support a claim about heterogeneity and stability. The current smoke only validates the execution path.
