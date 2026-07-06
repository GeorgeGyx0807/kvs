# 离线三维 KV 贡献画像实验报告

## 1. 实验目的

本实验目标是验证“离线三维 KV 贡献画像”的基本可行性：在不改模型权重的前提下，通过离线扰动 KV cache，估计不同层、KV head 和上下文 chunk 对回答质量、NLL、延迟与注意力分布的影响，为后续 KV 预算选择提供依据。

本轮实验不直接宣称已经得到最终预算策略，而是完成了一个可验证的画像管线：先在完整层/头维度上做 removal profiling，再对筛选出的层-头组合做 chunk 级细化。

## 2. 实验设置

- 模型：`Qwen/Qwen3-8B`
- 任务：`LongBench / narrativeqa`
- 样本数：100
- 模型维度：36 层，8 个 KV heads
- 上下文截断：`max_context_tokens=512`
- 输出长度：`max_new_tokens=4`
- chunk 划分：每 128 个 token 一个 chunk，共 4 个 chunk，分别覆盖约 `[0,128)`, `[128,256)`, `[256,384)`, `[384,512)`
- 主输出目录：`outputs/kv3d_formal_twostage_narrativeqa_s100`

两阶段设计如下：

1. Stage 1：对 36 层和 36 x 8 个层-头组合做 removal profiling，并保留 `full_kv`、`b_only` 基线。
2. Stage 2：根据 Stage 1 的层-头贡献结果，为每个样本选择 8 个代表性层-头组合，再做 4 个 chunk 的细粒度 removal profiling。

本轮没有运行 addition profiling，因此 `addition_removal_alignment` 为空；当前结论主要来自 removal 方向。

## 3. 已完成内容

本轮已经完成完整 GPU 画像实验，而不是只写了代码。

已生成并验证的数据包括：

- `full_kv`: 100 条
- `b_only`: 100 条
- `remove_layer`: 3600 条
- `remove_layer_head`: 28800 条
- `remove_layer_head_chunk`: 3200 条
- 总记录数：35800
- 计划记录数：35800
- 样本数：100

验证结果：

- `ok: true`
- `issues: []`
- `record_count == plan_size`
- `records.jsonl` 与 `raw_results.jsonl` 一致
- 未发现 NaN/Inf
- 注意力 JS/KL divergence 非负
- `full_kv` 基线具备质量、NLL、KV bytes 和 timing 字段

同时，代码层面已经补齐了实验可追溯性检查：`block_utility`、`stability_by_task`、`addition_removal_alignment` 等任务级结果必须带有 `dataset/task_name`，否则 validation 会失败。这保证跨任务扩展时不会丢失任务上下文。

## 4. 主要发现

### 4.1 层维度存在明显差异

以 remove layer 后的 NLL 上升作为主要信号，部分层被移除后损失明显更大，说明层维度上的 KV 贡献不是均匀的。

NLL 影响较大的层包括：

| layer | mean_delta_nll | mean_delta_f1 |
| --- | ---: | ---: |
| 32 | 0.452720 | -0.002959 |
| 34 | 0.439800 | 0.001992 |
| 0 | 0.418317 | -0.014934 |
| 35 | 0.335982 | 0.011676 |
| 7 | 0.328838 | -0.008946 |
| 33 | 0.284188 | -0.007529 |

F1 下降较明显的层包括 layer 11、26、0、7、27、33。这说明 NLL 与 F1 的敏感层并不完全一致：NLL 更像连续的模型置信度信号，F1 受短答案、生成波动和 exact overlap 影响更大。

### 4.2 KV head 维度差异存在，但整体弱于层维度

在 `remove_layer_head` 结果中，KV head 的差异比层差异小，但仍能看到排序。

按 NLL 影响排序：

| head | mean_delta_nll | mean_delta_f1 |
| --- | ---: | ---: |
| 7 | 0.015750 | 0.004343 |
| 3 | 0.010496 | 0.004245 |
| 2 | 0.009735 | 0.003979 |
| 4 | 0.007411 | 0.005736 |
| 6 | 0.004766 | 0.004404 |

这支持“head 维度有贡献差异”的判断，但当前 100 样本下 head 间差距较小，后续需要更大样本和更多任务确认稳定性。

### 4.3 层-头组合有更强的局部信号

按 `remove_layer_head` 的平均 NLL 影响，若聚合到具体层-头组合，较强组合包括：

| layer | head | mean_delta_nll | mean_delta_f1 |
| ---: | ---: | ---: | ---: |
| 32 | 7 | 0.286422 | 0.011302 |
| 9 | 7 | 0.274500 | -0.025435 |
| 7 | 1 | 0.234067 | -0.013611 |
| 34 | 5 | 0.203700 | -0.006556 |
| 13 | 3 | 0.178114 | -0.014427 |
| 33 | 6 | 0.176349 | 0.012731 |

这说明只看“第几个 head”是不够的；同一个 head 在不同层的作用可能不同。三维画像的核心价值也在这里：层、头、位置 chunk 的交互比单轴排名更有信息量。

### 4.4 chunk 维度有位置差异，但当前只是候选层-头上的局部证据

Stage 2 只对筛选出的层-头组合做 chunk 细化，因此 chunk 结论不是全模型全层头的穷举结论。

在当前 4 个 chunk 中：

| chunk | token range | mean_delta_nll | mean_delta_f1 | records |
| ---: | --- | ---: | ---: | ---: |
| 0 | `[0,128)` | 0.008871 | 0.005051 | 800 |
| 1 | `[128,256)` | -0.005035 | 0.002681 | 800 |
| 2 | `[256,384)` | -0.000802 | 0.002915 | 800 |
| 3 | `[384,512)` | 0.001822 | -0.000778 | 800 |

按 NLL 看，chunk 0 和 chunk 3 对当前候选组合更敏感；按 F1 看，chunk 3 出现轻微负向影响。由于 NarrativeQA 的证据可能分布在长上下文不同位置，当前 512 token 截断和 4 token 生成会压缩 chunk 信号，后续应扩大上下文长度和输出长度。

### 4.5 稳定性达到阶段性可用水平

跨样本 top-k Jaccard 稳定性如下：

| method | top_k=1 | top_k=2 | top_k=5 | top_k=10 |
| --- | ---: | ---: | ---: | ---: |
| remove_layer | 0.608889 | 0.606128 | 0.528236 | 0.592780 |
| remove_layer_head | 0.564646 | 0.551246 | 0.603140 | 0.643328 |
| remove_layer_head_chunk | 0.593535 | 0.599461 | 0.637003 | 0.606281 |

这些数值说明画像不是完全随机噪声，尤其 top-k 聚合后有一定稳定性。但它还不是强到可以直接作为最终部署策略的程度，更适合作为候选选择和分析依据。

### 4.6 注意力扰动与质量变化相关性较弱

注意力 divergence 已经成功记录：

| method | mean JS | mean KL | JS 与 delta_nll Spearman |
| --- | ---: | ---: | ---: |
| remove_layer | 0.000019 | 0.000076 | -0.032414 |
| remove_layer_head | 0.000019 | 0.000076 | -0.017193 |
| remove_layer_head_chunk | 0.000038 | 0.000148 | 0.079218 |

目前注意力分布变化与 NLL/F1 的相关性很弱。这是一个重要发现：注意力扰动可以作为解释性辅助信号，但当前不能单独替代质量指标来判断 KV 贡献。

### 4.7 `b_only` 与 `full_kv` 基线符合预期

`full_kv` 平均 F1 为 0.144538，平均 NLL 为 4.055955；`b_only` 平均 F1 为 0.129357，平均 NLL 为 4.727505。移除 KV 后 NLL 明显变差，F1 也下降，说明实验管线确实捕捉到了 KV 信息对模型输出的贡献。

## 5. 当前是否支持核心想法

结论：当前实验是阶段性成功，支持“离线三维 KV 贡献画像有信号、可验证、可扩展”的想法，但还不能充分证明最终预算选择策略一定有效。

已经支持的部分：

- 三维画像管线能在真实模型维度上跑通。
- 层、层-头、chunk 维度都能产生可排序的贡献信号。
- 画像结果通过 validation，记录完整且可追溯。
- 稳定性不是随机水平，具备进一步扩展价值。
- 注意力 divergence 可记录，但不能替代质量指标。

还没有充分支持的部分：

- 还没有跨任务验证，因此不能说该规律泛化到 LongBench 其他任务。
- 还没有 addition profiling，因此不能验证“移除重要块”和“逐步添加重要块”的对齐程度。
- 还没有真正按画像排序执行预算选择曲线，因此不能证明同等 KV budget 下优于随机、均匀或启发式策略。
- 当前输出长度只有 4 token，F1 信号较弱，答案质量评价可能不充分。
- 当前上下文只取 512 token，尚未体现长上下文 KV 压缩的完整场景。

## 6. 局限性

1. 单任务限制：目前正式实验只覆盖 `narrativeqa`，结果可能受该任务的数据分布影响。
2. 小样本限制：100 样本足以验证 pipeline 和观察初步规律，但不足以支撑强统计结论。
3. 生成长度限制：`max_new_tokens=4` 有利于控制 GPU 成本，但对 QA 任务的 F1/contains 评价偏短。
4. chunk 只做候选细化：Stage 2 不是 36 x 8 x 4 的完整穷举，而是对每样本选出的 8 个层-头组合做局部 chunk profiling。
5. addition 缺失：当前无法评估 addition-removal alignment。
6. 还缺少策略验证：已有画像，不等于已有可部署预算选择算法。

## 7. 下一步建议

优先级最高的是把“画像”推进到“预算选择验证”：

1. 加入 budgeted selection 实验：按画像分数选择 KV block，与随机选择、均匀按层选择、只保留最近 chunk、full KV、b_only 对比。
2. 补 addition profiling：验证高贡献块在 removal 和 addition 两个方向是否一致，补齐 `addition_removal_alignment`。
3. 扩大任务范围：至少加入 LongBench 中 3 到 5 个不同类型任务，例如 narrativeqa、qasper、multifieldqa、hotpotqa、2wikimqa。
4. 提高生成长度：将 `max_new_tokens` 从 4 提高到 32 或 64，使 F1/contains 更有解释力。
5. 拉长上下文：从 512 token 扩展到 2k、4k 或更长，观察 chunk 位置效应是否增强。
6. 做统计检验：对预算策略收益做 bootstrap confidence interval 或 paired test，避免只看均值。
7. 固化报告图表：将 layer importance、layer-head heatmap、chunk heatmap、budget-quality curve 作为正式论文图或阶段汇报图。

## 8. 总结

本轮实验已经完成了一个真实维度、100 样本、GPU 跑通并通过验证的离线三维 KV 贡献画像实验。结果表明 KV 贡献在层、层-头和 chunk 维度上并非均匀分布，且这种差异具有一定跨样本稳定性。

当前最重要的结论不是“已经证明最终压缩算法有效”，而是“已经证明画像信号存在，并且管线足够可靠，可以进入预算策略验证阶段”。下一阶段应围绕预算选择曲线和跨任务复现实验展开，这一步完成后，才适合把主张从“画像可行”推进到“画像能指导有效 KV 预算分配”。

