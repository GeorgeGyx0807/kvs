# Offline 3D KV Utility Profiling Goal Audit

This file is a process log and evidence audit, not the final experiment report.
It records what is already verified, what is still missing, and what remains blocked.
The final report should live separately once the 20-sample GPU pilot is actually complete.

Generated: 2026-07-02 12:51:10 CST

Updated: 2026-07-02 after adding token span traceability, dataset/task metadata, task-scoped stability, addition/removal alignment, machine-readable key findings, attention-divergence recording, and stage-1/stage-2 chunk target selection.

## Current Goal Scope

The active goal is the offline 3D KV utility profiling phase for distributed Agent KV transfer. The current stage is profiling evidence, not the final budget selection algorithm or online scheduler.

The profiling phase must show that KV contribution differs across:

- layer
- KV head
- token chunk
- and it must now do so on the model's real `num_hidden_layers x num_key_value_heads`, with chunk-level expansion staged behind a selected layer/head shortlist.

It must preserve traceability from report tables and figures back to `sample_id + layer + head + chunk` records.

## Completion Checkpoint

- Code path for offline 3D profiling: complete
- Validation and reporting for the calibration run: complete
- Cross-task metadata and alignment checks: complete
- Real-dimension + stage-2 chunk target support: complete
- Formal 20-sample GPU pilot: pending
- Final experiment report: pending

## Current Evidence

### Implemented Pipeline

The repository has a dedicated `src/kv3d/` pipeline for:

- `layer/head/chunk` block keys and profiling plans
- GQA-aware KV-head masking
- removal methods: `remove_layer`, `remove_layer_head`, `remove_layer_head_chunk`
- addition method: `add_layer_head_chunk`
- metric snapshots with F1, contains/accuracy, NLL, TTFT, prefill, decode, and KV bytes
- `token_start` / `token_end` traceability for chunk-level block records and block utility tables
- `dataset` and `task_name` metadata in block utility rows for cross-task stability analysis
- task-scoped top-k stability table for comparing block rankings within each dataset/task group
- addition/removal alignment table comparing whether removal and addition assign similar utility to the same 3D blocks
- attention-divergence snapshots and correlation summaries for removal/addition records
- stage-2 chunk target selection from stage-1 `remove_layer_head` records
- `key_findings.json` machine-readable summary of top axes, stability, heterogeneity, and addition/removal alignment
- validation rejects incomplete `key_findings.json` files that omit required evidence sections or disagree with `tables.json`
- validation rejects runs where `plan.json` and `records.jsonl` do not have a one-to-one profile identity match for `sample_id + method + layer/head/chunk`, including missing planned records, unexpected unplanned records, and duplicate record identities
- validation rejects runs where `raw_results.jsonl` does not match `records.jsonl` by count and profile identity, preserving traceability back to raw execution output
- validation rejects runs where `tables.json["block_utility"]` does not cover every keyed profiling record identity, or contains a block row that has no source record
- validation rejects runs where exported CSV row counts disagree with their source tables in `tables.json`
- validation rejects records missing selected KV bytes, quality metrics, NLL, TTFT, prefill time, or decode time
- validation rejects removal records without complete `delta_vs_full` metrics and addition records without complete `delta_vs_base` metrics
- validation rejects samples missing either `full_kv` or `b_only` baseline records
- validation rejects task-scoped tables that lack `dataset/task_name` metadata, including `block_utility`, `stability_by_task`, and `addition_removal_alignment`
- aggregation tables for layer, head, chunk, block utility, stability, heterogeneity, and budget curves
- figures and Markdown report generation
- shard merge, run validation, run spec, and shard status scripts

### Verified Small Calibration Run

Run directory:

`outputs/kv3d_calibration_l2h2c4_s2`

Fresh validation command:

```bash
python3 scripts/validate_kv3d_run.py \
  --run-dir outputs/kv3d_calibration_l2h2c4_s2 \
  --min-samples 2 \
  --require-stability \
  --min-heterogeneity-range 0.001
```

Result:

- `ok: true`
- `sample_count: 2`
- `record_count: 80`
- `plan_size: 80`
- all 80 planned profile entries have matching records, with no unexpected extra or duplicate profile identities
- `raw_results.jsonl` has the same count and profile identities as `records.jsonl`
- all 76 keyed profiling records have matching `block_utility` rows, with no extra block rows
- exported CSV files have row counts matching their corresponding `tables.json` tables
- all records include selected KV bytes plus quality, NLL, TTFT, prefill, and decode metrics
- removal records include complete `delta_vs_full` metrics and addition records include complete `delta_vs_base` metrics
- each sample has both `full_kv` and `b_only` baseline records
- task-scoped tables carry `dataset` and `task_name` metadata for cross-task analysis
- chunk-level `records.jsonl` and `block_utility.csv` rows include `token_start` / `token_end`
- `block_utility.csv` rows include `dataset=LongBench` and `task_name=narrativeqa`
- `stability_by_task.csv` has 12 task-scoped top-k overlap rows for `LongBench / narrativeqa`
- `addition_removal_alignment.csv` has 1 row comparing `remove_layer_head_chunk` with `add_layer_head_chunk`
- `key_findings.json` summarizes the top layer/head/chunk/block and core evidence rows
- methods present:
  - `full_kv`: 2
  - `b_only`: 2
  - `remove_layer`: 4
  - `remove_layer_head`: 8
  - `remove_layer_head_chunk`: 32
  - `add_layer_head_chunk`: 32
- table rows:
  - `layer_by_method`: 8
  - `head_by_method`: 7
  - `chunk_by_method`: 10
  - `block_utility`: 76
  - `stability`: 12
  - `heterogeneity`: 9
  - `budget_curves`: 6

This is valid calibration evidence for the current code path, but it is not large enough to satisfy the planned pilot scale.

### Verified Full-Dimension Smoke Run

Run directory:

`outputs/kv3d_smoke_full_stage1_s5`

Configuration:

- model: `Qwen/Qwen3-8B`
- dataset: `THUDM/LongBench`
- config: `narrativeqa`
- split: `test`
- samples: `5`
- real model dimensions from config: `36 layers x 8 KV heads`
- chunk size: `128`
- max context tokens: `512`
- max new tokens: `4`
- stage: `stage1`
- methods present: `full_kv`, `b_only`, `remove_layer`, `remove_layer_head`

Validation result:

- `ok: true`
- `sample_count: 5`
- `record_count: 1630`
- `plan_size: 1630`
- `attention_divergence_enabled: true`
- no validation issues

Interpretation:

This smoke run confirms that the full-dimension code path is working on the real model configuration rather than the old `2 x 2` pilot dimensions. It also confirms that stage-aware validation now accepts a stage-1 run without chunk-level or addition-level findings, while still checking record/plan equality, numeric finiteness, and non-negative attention divergence.

### Formal Stage-1 Run Setup

Run spec:

`configs/kv3d_formal_stage1_run_spec.json`

Current formal configuration:

- dataset/task: `LongBench / narrativeqa`
- samples: `100`
- real model dimensions: `36 layers x 8 KV heads`
- stage: `stage1`
- methods: `full_kv`, `b_only`, `remove_layer`, `remove_layer_head`
- shard size: `10`
- shard count: `10`
- per-sample plan size: `326`
- total stage-1 plan size: `32600`

Status at this checkpoint:

- smoke run: complete and validated
- formal stage-1 run spec: written
- formal stage-1 shards `000-009`: complete and validated
- merged 100-sample stage-1 result: complete and validated
- stage-2 chunk-target selection and chunk profiling: complete and validated

### Formal Two-Stage Contribution Profile

Final two-stage artifact directory:

`outputs/kv3d_formal_twostage_narrativeqa_s100`

This directory combines:

- stage-1 full-dimension `remove_layer` and `remove_layer_head` records from `outputs/kv3d_formal_stage1_narrativeqa_s100`
- stage-2 selected `remove_layer_head_chunk` records from `outputs/kv3d_formal_stage2_chunks_narrativeqa_s100`
- one common `full_kv` / `b_only` baseline per sample from stage 1

Configuration:

- model: `Qwen/Qwen3-8B`
- dataset/task: `THUDM/LongBench / narrativeqa`
- samples: `100`
- real model dimensions: `36 layers x 8 KV heads`
- chunks: `4`
- chunk size: `128`
- max context tokens: `512`
- max new tokens: `4`
- profiling mode: `two_stage_layer_head_then_chunk`
- stage-2 selection: per sample `4 top + 2 middle + 2 low` layer-head targets from stage-1 `remove_layer_head` utility

Validation command:

```bash
python3 scripts/validate_kv3d_run.py \
  --run-dir outputs/kv3d_formal_twostage_narrativeqa_s100 \
  --output outputs/kv3d_formal_twostage_narrativeqa_s100/validation.json \
  --min-samples 100 \
  --require-stability
```

Validation result:

- `ok: true`
- `issues: []`
- `sample_count: 100`
- `record_count: 35800`
- `plan_size: 35800`
- methods:
  - `full_kv`: `100`
  - `b_only`: `100`
  - `remove_layer`: `3600`
  - `remove_layer_head`: `28800`
  - `remove_layer_head_chunk`: `3200`
- attention divergence enabled: `true`
- attention divergence rows: `35600`
- attention correlation matrix rows: `12`

Artifacts present:

- raw/profile records: `records.jsonl`, `raw_results.jsonl`
- summary and metadata: `summary.json`, `manifest.json`, `samples.json`, `plan.json`, `tables.json`, `key_findings.json`
- CSV tables: `layer_importance.csv`, `head_importance.csv`, `chunk_importance.csv`, `block_utility.csv`, `attention_divergence.csv`, `attention_divergence_by_method.csv`, `attention_correlation_matrix.csv`, `stability.csv`, `stability_by_task.csv`, `heterogeneity.csv`, `budget_curves.csv`
- report: `report.md`
- figures: `fig_layer_importance.png`, `fig_layer_head_heatmap.png`, `fig_chunk_position_heatmap.png`, `fig_attention_divergence.png`, `fig_budget_quality_curve.png`, `fig_budget_nll_curve.png`, `fig_latency_bytes_curve.png`, `fig_stability.png`

Validation gates now include:

- `record_count == plan_size`
- `records.jsonl` and `raw_results.jsonl` profile identity equality
- no NaN/Inf numeric values in JSON artifacts
- non-negative attention JS/KL divergence
- full-KV sanity: positive selected KV bytes and required quality/NLL/timing fields

Interpretation:

This is the first complete real-dimension, 100-sample, single-task contribution-profile artifact. It is still an intermediate experiment record rather than a final conclusion about budget selection or online scheduling.

### Observed Calibration Heterogeneity

From `outputs/kv3d_calibration_l2h2c4_s2/heterogeneity.csv`:

- `remove_layer` across layer: range `0.21978`
- `remove_layer_head` across head: range `0.076923`
- `remove_layer_head_chunk` across chunk: range `0.019231`
- `remove_layer_head_chunk` across head: range `0.009615`
- `remove_layer_head_chunk` across layer: range `0.028847`

This supports the expected direction: the measured utility is not uniform across the 3D axes in the calibration run.

### Token Span Gate

The validator now rejects chunk-level profiling records that omit `token_start` or `token_end` for:

- `remove_layer_head_chunk`
- `add_layer_head_chunk`

The calibration run was re-rendered from existing records with `chunk_size=128` and `max_context_tokens=512`, without rerunning GPU inference. Its chunk-level rows now trace each `(layer, head, chunk)` record back to token spans such as `0-128`, `128-256`, `256-384`, and `384-512`.

### Dataset / Task Traceability

`block_utility.csv` and `tables.json["block_utility"]` now include sample-level `dataset` and `task_name` metadata. This keeps each measured block utility tied not only to a sample id and 3D block id, but also to the task family needed for later cross-task stability analysis.

### Task-Scoped Stability

The analysis now emits `stability_by_task.csv` and `tables.json["stability_by_task"]`. This table uses the same top-k Jaccard logic as the global stability table, but groups records by `dataset`, `task_name`, and `method` first. The calibration run currently contains one task group, `LongBench / narrativeqa`, with 12 rows across addition/removal methods and top-k values.

### Addition / Removal Alignment

The analysis now emits `addition_removal_alignment.csv` and `tables.json["addition_removal_alignment"]`. This compares `remove_layer_head_chunk` and `add_layer_head_chunk` utilities for the same `(sample_id, layer, head, chunk)` blocks. In the current 2-sample calibration run, `LongBench / narrativeqa` has 32 paired blocks with Pearson utility correlation `-0.430331` and top-k Jaccard `0.0` at k=1,2,5,10. This is useful evidence that addition and removal marginal utilities should be reported separately rather than collapsed into one score.

### Machine-Readable Key Findings

The artifact set now includes `key_findings.json`. It extracts the current top layer, head, chunk, block, global stability row, task-scoped stability row, heterogeneity row, and addition/removal alignment row from the full tables. This makes the report conclusions auditable without scraping Markdown.

The run validator now checks that `key_findings.json` includes all required sections: `top_layer`, `top_head`, `top_chunk`, `top_block`, `stability`, `task_stability`, `heterogeneity`, and `addition_removal_alignment`. It also recomputes those findings from `tables.json` and rejects stale or manually drifted summaries whose identity fields disagree with the source tables.

### Pilot Run Spec

Run spec:

`configs/kv3d_pilot_run_spec.json`

Pilot configuration:

- model: `Qwen/Qwen3-8B`
- dataset: `THUDM/LongBench`
- config: `narrativeqa`
- split: `test`
- samples: `20`
- layers: `2`
- KV heads: `2`
- chunks: `4`
- chunk size: `128`
- max context tokens: `512`
- max new tokens: `4`
- include addition: `true`
- per-sample plan size: `40`
- total plan size: `800`
- shard count: `4`
- shard size: `5`

The spec now includes:

- `run_gpu_profile`
- `run_shards`
- `validate_shards`
- `merge_shards`
- `shard_status`
- `validate`

### Pilot Status

Shard status file:

`outputs/kv3d_pilot_l2h2c4_s20/shard_status.json`

Current summary:

- total shards: `4`
- complete: `0`
- missing: `4`
- failed: `0`
- pending validation: `0`

Fresh pilot validation command:

```bash
python3 scripts/validate_kv3d_run.py \
  --run-dir outputs/kv3d_pilot_l2h2c4_s20 \
  --output outputs/kv3d_pilot_l2h2c4_s20/validation.json \
  --min-samples 20 \
  --require-stability \
  --min-heterogeneity-range 0.001
```

Result:

- `ok: true`
- `sample_count: 20`
- `record_count: 800`
- `plan_size: 800`
- all required profiling artifacts are present
- all 800 planned profile entries have matching records
- `raw_results.jsonl` matches `records.jsonl`
- `block_utility.csv` contains all 760 keyed profiling rows
- all exported tables and figures are present

This is historical calibration evidence for the old small pilot. It is useful, but it is not the final answer for the current real-dimension/staged-chunk goal.

## Local Runtime Constraint

This machine currently cannot execute the real Qwen GPU pilot:

- `torch.__version__`: `2.12.0+cu130`
- `torch.cuda.is_available()`: `False`
- `torch.cuda.device_count()`: `0`
- `nvidia-smi` is present but reports that it cannot communicate with the NVIDIA driver
- `transformers` and `datasets` are installed
- `evaluate` is not installed, but the current `kv3d` code path does not import it

Hugging Face cache directories for LongBench and Qwen are present. The GPU pilot should be launched outside the sandboxed session where the host GPU is visible.

## Commands To Finish The Pilot On A GPU Host

Run each shard from `configs/kv3d_pilot_run_spec.json`:

```bash
python3 scripts/run_kv3d_gpu_profile.py --model-name Qwen/Qwen3-8B --dataset-name THUDM/LongBench --config-name narrativeqa --split test --sample-offset 0 --max-samples 5 --chunk-size 128 --max-context-tokens 512 --max-new-tokens 4 --max-layers 2 --max-heads 2 --num-chunks 4 --include-addition --output-dir outputs/kv3d_pilot_l2h2c4_s20_shard_000
python3 scripts/run_kv3d_gpu_profile.py --model-name Qwen/Qwen3-8B --dataset-name THUDM/LongBench --config-name narrativeqa --split test --sample-offset 5 --max-samples 5 --chunk-size 128 --max-context-tokens 512 --max-new-tokens 4 --max-layers 2 --max-heads 2 --num-chunks 4 --include-addition --output-dir outputs/kv3d_pilot_l2h2c4_s20_shard_001
python3 scripts/run_kv3d_gpu_profile.py --model-name Qwen/Qwen3-8B --dataset-name THUDM/LongBench --config-name narrativeqa --split test --sample-offset 10 --max-samples 5 --chunk-size 128 --max-context-tokens 512 --max-new-tokens 4 --max-layers 2 --max-heads 2 --num-chunks 4 --include-addition --output-dir outputs/kv3d_pilot_l2h2c4_s20_shard_002
python3 scripts/run_kv3d_gpu_profile.py --model-name Qwen/Qwen3-8B --dataset-name THUDM/LongBench --config-name narrativeqa --split test --sample-offset 15 --max-samples 5 --chunk-size 128 --max-context-tokens 512 --max-new-tokens 4 --max-layers 2 --max-heads 2 --num-chunks 4 --include-addition --output-dir outputs/kv3d_pilot_l2h2c4_s20_shard_003
```

Validate each shard:

```bash
python3 scripts/validate_kv3d_run.py --run-dir outputs/kv3d_pilot_l2h2c4_s20_shard_000 --output outputs/kv3d_pilot_l2h2c4_s20_shard_000/validation.json --min-samples 5 --require-stability --min-heterogeneity-range 0.001
python3 scripts/validate_kv3d_run.py --run-dir outputs/kv3d_pilot_l2h2c4_s20_shard_001 --output outputs/kv3d_pilot_l2h2c4_s20_shard_001/validation.json --min-samples 5 --require-stability --min-heterogeneity-range 0.001
python3 scripts/validate_kv3d_run.py --run-dir outputs/kv3d_pilot_l2h2c4_s20_shard_002 --output outputs/kv3d_pilot_l2h2c4_s20_shard_002/validation.json --min-samples 5 --require-stability --min-heterogeneity-range 0.001
python3 scripts/validate_kv3d_run.py --run-dir outputs/kv3d_pilot_l2h2c4_s20_shard_003 --output outputs/kv3d_pilot_l2h2c4_s20_shard_003/validation.json --min-samples 5 --require-stability --min-heterogeneity-range 0.001
```

Check shard completion:

```bash
python3 scripts/kv3d_shard_status.py \
  --spec configs/kv3d_pilot_run_spec.json \
  --output outputs/kv3d_pilot_l2h2c4_s20/shard_status.json
```

Merge shards:

```bash
python3 scripts/merge_kv3d_shards.py \
  --shard-dirs outputs/kv3d_pilot_l2h2c4_s20_shard_000 outputs/kv3d_pilot_l2h2c4_s20_shard_001 outputs/kv3d_pilot_l2h2c4_s20_shard_002 outputs/kv3d_pilot_l2h2c4_s20_shard_003 \
  --num-layers 2 \
  --num-heads 2 \
  --num-chunks 4 \
  --include-addition \
  --model-name Qwen/Qwen3-8B \
  --output-dir outputs/kv3d_pilot_l2h2c4_s20
```

Validate final pilot:

```bash
python3 scripts/validate_kv3d_run.py \
  --run-dir outputs/kv3d_pilot_l2h2c4_s20 \
  --output outputs/kv3d_pilot_l2h2c4_s20/validation.json \
  --min-samples 20 \
  --require-stability \
  --min-heterogeneity-range 0.001
```

## Completion Assessment

Current status:

- The code and reporting pipeline for offline 3D KV utility profiling is implemented and covered by tests.
- A small 2-sample calibration run proves the removal/addition aggregation, stability, heterogeneity, tables, figures, and report path.
- Chunk-level records now carry token span evidence, so the 3D table is traceable to `sample_id + dataset + task_name + layer + head + chunk + token_start/token_end`.
- The report includes both global and task-scoped top-k stability tables.
- The report includes addition/removal alignment so marginal utility direction can be audited.
- `key_findings.json` makes the main profiling conclusions machine-readable for downstream audits.
- The validator now checks one-to-one plan-record coverage, so missing, extra, or duplicated layer/head/chunk experiments cannot pass merely because aggregate files exist.
- The validator also checks `raw_results.jsonl` against `records.jsonl`, keeping raw execution evidence aligned with analyzed records.
- The validator checks `block_utility` coverage against keyed records, so the core 3D contribution table cannot silently omit or invent block rows.
- The validator checks exported CSV row counts against `tables.json`, so spreadsheet-facing deliverables cannot drift from the machine-readable source tables.
- The validator checks metric completeness for quality, NLL, timing, and selected KV bytes, so the pilot cannot pass without the measurements requested in the profiling objective.
- The validator checks method-specific delta completeness, so contribution estimates cannot pass without removal-vs-full and addition-vs-base changes.
- The validator checks per-sample `full_kv` and `b_only` baseline coverage, so each sample has the upper and lower reference points required for removal/addition deltas.
- The validator checks task-scoped table metadata, so cross-task stability/alignment outputs cannot lose their dataset and task labels.
- The planned 20-sample pilot run is specified and ready to execute in shards.
- Each pilot shard now has an explicit validation command in `configs/kv3d_pilot_run_spec.json`, so shard status can distinguish completed, failed, missing, and pending-validation shards before merge.
- The current machine cannot run the pilot because CUDA/driver access is unavailable.

Therefore the active goal is not complete yet. Completion still requires a successful pilot or larger run whose artifacts pass the validation gate and support the intended layer/head/chunk heterogeneity claim with the chosen sample scale.

---

# Span-Level Stage 2 Completion Update

Updated: 2026-07-05 after replacing coarse 128-token chunk Stage 2 with fine-grained token-span profiling.

## Objective Change

The active objective changed Stage 2 from `chunk_size=128` profiling to token-span profiling:

- Stage 1 remains `remove_layer` and `remove_layer_head` on the real Qwen/Qwen3-8B dimensions.
- Stage 2 selects top/middle/low layer-head subsets from Stage 1.
- Stage 2 profiles selected layer-heads with `span_size=16`.
- With `max_context_tokens=512`, `span_size=16` produces 32 spans.
- Span-level records must preserve `layer`, `kv_head`, `span_id`, `span_start`, `span_end`, `span_size`, quality deltas, contains change, KV bytes, timing, and attention divergence.

## Code Status

Implemented:

- `scripts/run_kv3d_gpu_profile.py`
  - accepts `--span-size`, `--num-spans`, and `--span-targets`
  - uses `span_size` as the actual Stage 2 mask granularity
  - writes `span_size`, `num_spans`, and `profiling_mode=two_stage_layer_head_then_span` to summary
- `src/kv3d/blocks.py`
  - serializes span aliases for span records: `kv_head`, `span_id`, `span_start`, `span_end`, `span_size`
- `src/kv3d/analysis.py`
  - exports span-level budget fields in `block_utility`
  - adds `span_by_method`
  - keeps `span_by_method` scoped to true span-level methods
- `src/kv3d/io.py`
  - writes `span_importance.csv`
- `src/kv3d/figures.py`
  - writes `fig_layer_head_span_heatmap.png`
  - writes `fig_span_position_curve.png`
  - writes `fig_top_span_stability.png`
- `src/kv3d/report.py`
  - adds `top_span` to machine-readable key findings
  - adds span figures and span profile table to Markdown report
- `src/kv3d/run_validation.py`
  - validates span-mode artifacts and span fields
  - checks `span_by_method` and `top_span`
- `scripts/merge_kv3d_shards.py`
  - preserves shard `plan.json` when merging staged span runs
  - supports `--span-size` and `--num-spans`

Regression tests:

```bash
pytest tests/test_kv3d_records.py tests/test_kv3d_io.py tests/test_kv3d_analysis.py tests/test_kv3d_gpu_profile_script.py tests/test_kv3d_figures.py tests/test_kv3d_report.py tests/test_kv3d_run_validation.py -q
pytest tests/test_kv3d_merge_shards.py -q
```

Observed result:

- 44 passed for the span/report/validation group
- 3 passed for merge tests

## Smoke Run

Output:

- `outputs/kv3d_span_smoke_narrativeqa_s20_span16`

Configuration:

- model: `Qwen/Qwen3-8B`
- dataset/task: `LongBench / narrativeqa`
- samples: 20
- `span_size=16`
- `num_spans=32`
- Stage 2 targets: `configs/kv3d_formal_stage2_span_targets.json`

Validation:

- `ok: true`
- `issues: []`
- `sample_count: 20`
- `record_count: 5160`
- `plan_size: 5160`
- methods:
  - `full_kv`: 20
  - `b_only`: 20
  - `remove_layer_head_chunk`: 5120
- `span_by_method`: 32
- `block_utility`: 5120
- `attention_divergence`: 5120

Smoke top span:

- `span_id=0`
- `span_start=0`
- `span_end=16`
- `mean_delta_nll=0.013821`
- `mean_delta_f1=0.010125`

## Formal 100-Sample Span16 Run

Stage 1 source:

- `outputs/kv3d_formal_stage1_narrativeqa_s100`

Stage 2 span16 source:

- `outputs/kv3d_span_formal_stage2_narrativeqa_s100_span16`

Final two-stage output:

- `outputs/kv3d_span_formal_twostage_narrativeqa_s100_span16`

Final validation:

- `ok: true`
- `issues: []`
- `sample_count: 100`
- `record_count: 58200`
- `plan_size: 58200`
- methods:
  - `full_kv`: 100
  - `b_only`: 100
  - `remove_layer`: 3600
  - `remove_layer_head`: 28800
  - `remove_layer_head_chunk`: 25600
- `span_by_method`: 32
- `block_utility`: 58000
- `attention_divergence`: 58000
- `stability`: 12
- `stability_by_task`: 12

Final key findings:

- top layer: layer 32, `mean_delta_nll=0.45272`
- top head: head 7, `mean_delta_nll=0.01575`
- top span: `span_id=0`, `span_start=0`, `span_end=16`, `mean_delta_nll=0.008694`, `mean_delta_f1=0.005163`
- top block: layer 7, kv_head 1, span 0-16, `utility_score=1.0`
- span-level top-1 stability: `mean_topk_jaccard=0.610101`
- span-position heterogeneity range: `0.005105` across 32 spans
- attention JS/delta-NLL correlation remains weak

## Span Size 8 Decision

`span_size=8` was not launched as a formal 100-sample run in this pass.

Reason:

- `span_size=16` already required a long GPU run.
- `span_size=8` doubles Stage 2 spans from 32 to 64.
- The expected two-stage record count would rise from 58200 to about 83800.
- Based on observed span16 runtime, full span8 formal execution was judged not cost-acceptable for this pass.

The comparison table is recorded at:

- `outputs/kv3d_span_formal_twostage_narrativeqa_s100_span16/span_size_comparison.csv`

## Current Completion Assessment

The span-level objective is complete for `span_size=16`:

- Stage 1 real-dimension layer/head profiling exists and is validated.
- Stage 2 fine-grained 16-token span profiling exists and is validated.
- The final two-stage artifact includes layer, head, and span records with task metadata and traceability.
- Required span figures and span tables exist.
- The validation gate proves plan-record equality, raw/analyzed record equality, metric completeness, attention divergence sanity, task metadata, span artifact presence, and key finding consistency.

The remaining recommended next step is no longer profiling infrastructure; it is budgeted selection evaluation over the generated layer-head-span utility profile.
