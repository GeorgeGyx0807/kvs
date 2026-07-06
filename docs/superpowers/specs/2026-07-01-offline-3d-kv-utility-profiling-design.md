# Offline 3D KV Utility Profiling Design

## 1. Goal

Build an offline profiling pipeline for distributed KV transfer that measures how much each KV block contributes to target-agent reasoning quality across three dimensions:

- `layer`
- `head`
- `token/chunk`

This phase does not train a selector and does not solve the final budgeted optimization problem yet. It only establishes evidence that KV utility is heterogeneous in the 3D block space.

## 2. Fixed Assumptions

- Model: `Qwen3-8B`
- Agent A and Agent B: same checkpoint, same architecture, same tokenizer
- Main dataset: `LongBench`
- Auxiliary stress test: `RULER`
- Experiment type: offline profiling, not online scheduling
- Main comparison baseline: full-KV transfer

## 3. Questions the Profiling Must Answer

1. Which layers matter most?
2. Which heads matter most inside a layer?
3. Which token/chunk positions matter most inside a head?
4. Is the importance distribution stable across samples and tasks?
5. Do removal and addition experiments tell the same story?

## 4. Core Experimental Protocol

For each sample:

1. Run `full-KV` transfer as the upper bound.
2. Run `b_only` or empty-context baseline.
3. Split the full KV cache into `block(layer, head, chunk)` units.
4. Measure block importance by:
   - removal from full KV
   - addition from empty/base KV
5. Record:
   - `accuracy`
   - `F1`
   - `contains`
   - `NLL`
   - `TTFT`
   - `prefill_time`
   - `decode_time`
   - `KV bytes`

## 5. Block Definition

A block is identified by:

- `sample_id`
- `layer`
- `head`
- `chunk`
- `token_start`
- `token_end`

Each block contains both `K` and `V`.

If the model uses GQA/MQA, the profiling must respect the actual KV-head axis, not assume it matches the attention-head axis.

## 6. Profiling Modes

### 6.1 Removal Profiling

Start from full KV and remove one block or one block group:

- one layer
- one `(layer, head)`
- one `(layer, head, chunk)`

Record the quality drop and timing change.

### 6.2 Addition Profiling

Start from empty KV or a minimal base set, then add blocks incrementally.

This is used to expose marginal gains and interaction effects that removal alone may hide.

## 7. Outputs

Required artifacts:

- `layer_importance.csv`
- `head_importance.csv`
- `chunk_importance.csv`
- `budget_curves.csv`
- `raw_results.jsonl`
- `report.md`

Required figures:

- layer importance curve
- layer-head heatmap
- chunk position heatmap
- budget-quality curve
- budget-NLL curve
- KV bytes vs TTFT/decode curve

## 8. Repository Mapping

The current repository already contains a first-round TTFT/KV selection skeleton. This design reuses that structure but changes the research question.

Existing reusable pieces:

- `src/ttft.py`
- `src/frontier.py`
- `src/oracle.py`
- `src/selectors.py`
- `src/features.py`

New work should live in `src/kv3d/` and should not overwrite the first-round protocol.

## 9. Minimal Acceptance Criteria

The phase is ready when all of the following are true:

1. The design is consistent with `Qwen3-8B`, same-model A/B agents, and `LongBench` as the main dataset.
2. The repository has 3D block data structures for `layer/head/chunk`.
3. The repository can represent removal and addition profiling records.
4. The protocol explicitly separates profiling from final budget optimization.
5. The experiment outputs can be traced back to `sample_id + block_id`.

## 10. Non-Goals

- No final selector training yet
- No online bandwidth scheduler
- No KV quantization work yet
- No claim that a block's utility is independent of other blocks

