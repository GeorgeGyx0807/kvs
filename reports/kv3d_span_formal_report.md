# 离线三维 KV 贡献画像：16-token Span 正式实验报告

## 1. 本轮完成内容

本轮将 Stage 2 从粗粒度 `chunk_size=128` 改为 fine-grained token-span profiling。Stage 1 保持真实模型维度的 `remove_layer` 与 `remove_layer_head` profiling；Stage 2 从 Stage 1 的 top/middle/low layer-head 子集中选择候选，并在候选 layer-head 上使用 `span_size=16` 做 removal profiling。

最终 two-stage 输出目录：

- `outputs/kv3d_span_formal_twostage_narrativeqa_s100_span16`

核心设置：

- 模型：`Qwen/Qwen3-8B`
- 任务：`LongBench / narrativeqa`
- 样本数：100
- Stage 1：36 layers x 8 KV heads
- Stage 2：每样本 8 个 layer-head targets
- `max_context_tokens=512`
- `span_size=16`
- `num_spans=32`
- `max_new_tokens=4`

## 2. 生成结果

最终 two-stage 记录数：

| method | records |
| --- | ---: |
| `full_kv` | 100 |
| `b_only` | 100 |
| `remove_layer` | 3600 |
| `remove_layer_head` | 28800 |
| `remove_layer_head_chunk` | 25600 |
| total | 58200 |

最终 validation：

- `ok: true`
- `issues: []`
- `sample_count: 100`
- `record_count: 58200`
- `plan_size: 58200`
- `span_by_method: 32`
- `block_utility: 58000`
- `attention_divergence: 58000`

新增/关键 artifact：

- `span_importance.csv`
- `span_size_comparison.csv`
- `figures/fig_layer_head_span_heatmap.png`
- `figures/fig_span_position_curve.png`
- `figures/fig_top_span_stability.png`
- `key_findings.json`
- `validation.json`

## 3. 主要发现

### 3.1 Span-level 信号跑通且可验证

`span_size=16` 在 512-token context 下产生 32 个 spans。最终 `span_by_method` 正好有 32 行，说明 Stage 2 已从旧的 4 个 128-token chunks 转为 32 个 16-token spans。

每条 span-level block 记录保存了：

- `layer`
- `head` / `kv_head`
- `chunk` / `span_id`
- `token_start` / `token_end`
- `span_start` / `span_end`
- `span_size`
- `delta_nll`
- `delta_f1`
- `contains_change`
- `selected_kv_bytes` / `kv_bytes`
- timing fields
- attention divergence

### 3.2 当前 top span 位于开头位置

最终 key findings 中，top span 为：

- `span_id=0`
- `span_start=0`
- `span_end=16`
- `mean_delta_nll=0.008694`
- `mean_delta_f1=0.005163`

top layer/head/span block 为：

- `layer=7`
- `kv_head=1`
- `span_id=0`
- `span_start=0`
- `span_end=16`
- `utility_score=1.0`

这说明在当前 NarrativeQA 100 样本、512-token 截断、候选 layer-head 子集下，开头 span 对若干样本具有明显贡献。但这仍是单任务结论，需要跨任务确认。

### 3.3 稳定性仍然可用

Stage 2 span-level top-1 stability：

- `mean_topk_jaccard=0.610101`
- `sample_count=100`
- `pair_count=4950`

这比 20-sample smoke 的 `0.494737` 更稳定，说明 100 样本正式实验比 smoke 更可靠。

### 3.4 注意力 divergence 仍不是强解释变量

Stage 2 span-level attention divergence summary：

- `mean_attention_js_divergence=0.000007`
- `mean_attention_kl_divergence=0.000028`
- JS 与 `delta_nll` Pearson 约 `0.072497`
- JS 与 `delta_nll` Spearman 约 `-0.008529`

结论与 chunk 版一致：attention divergence 能作为辅助 trace，但目前不能单独替代质量/NLL 指标。

## 4. Span Size 对比

`span_size=16` 正式完成。`span_size=8` 会把 512-token context 划为 64 spans，Stage 2 span records 从 25600 增至约 51200，最终 two-stage 记录量约 83800。基于本轮 span16 的 GPU 耗时，span8 正式运行预计接近 2 倍成本，因此本轮暂不启动正式 span8。

对比表已写入：

- `outputs/kv3d_span_formal_twostage_narrativeqa_s100_span16/span_size_comparison.csv`

## 5. 当前结论

本轮目标的核心部分已经完成：代码和实验都已从 128-token chunk profiling 升级到 16-token token-span profiling，并在真实模型维度、100 样本 NarrativeQA 上完成验证。

当前结果支持后续 token/span-level KV budget allocation 的实验基础：已有 layer -> head -> span 的可追溯贡献画像、稳定性表、span position curve 和 layer-head-span heatmap。下一步应做真正的 budgeted selection：按 span utility 选择 KV blocks，并与 full KV、B-only、随机 span、均匀 span、最近 span 策略比较。

