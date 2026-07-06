# TTFT-aware KV 子集选择：第一轮论文实验协议与阶段总结

## 1. 修订后的总目标

验证一个论文命题：在长上下文任务里，KV 子集选择可以在通信预算约束下形成稳定的性能-开销权衡，而且这种权衡不是随机造成的。

这轮不追求最强结果，只验证命题是否成立、证据链是否闭合、结论是否可复现。

## 2. 第一轮要验证什么

### 2.1 TTFT 是可拆的

TTFT 不是黑箱，至少能分出：
- 通信
- 打包
- 接收
- 首 token 推理

### 2.2 block 级选择是有信息的

block/page 级特征应当能够和 oracle 标签建立稳定关系，而不是纯噪声。

### 2.3 full-KV decode 可以提供教师信号

这个教师信号只用于离线分析、标签构造和上界分析，不进入部署特征。

### 2.4 部署侧只看 prefill 特征也能做选择

最终 selector 必须只依赖 prefill 可得信息，不能依赖 decode attention。

### 2.5 不同选择策略在同一预算下可比较

至少比较：
- random
- prefill attention top-k
- heuristic scorer
- table route
- neural predictor
- oracle upper bound

### 2.6 能得到第一版 Pareto 前沿

输出 accuracy-KV size-TTFT 的前沿曲线，作为第一轮主证据。

## 3. 数据集与模型角色

### 3.1 主集

`LongBench`

用途：
- 作为主实验集
- 展示长上下文任务上的主要结论
- 产出第一版前沿图和 baseline 表

### 3.2 辅集

`RULER`

用途：
- 做更可控的长上下文压力测试
- 检查规律是否稳定，不只是在自然分布任务上成立

### 3.3 模型

`Qwen3-8B`

说明：
- source / target 尽量使用同一 checkpoint
- 这样避免额外变量干扰第一轮结论

## 4. 第一轮的证据链

这一轮只接受下面几类证据：

1. TTFT 分解图
2. block/page feature table
3. oracle label table
4. deployable feature table
5. selector 对比表
6. greedy / frontier 结果

## 5. 第一轮验收标准

必须同时满足：

1. 有一份固定小校准集，能稳定跑完整流程。
2. 每个样本都能按 `sample_id + block_id` 对齐特征表和标签表。
3. `deployable feature table` 中不包含 decode attention 字段。
4. 能输出至少一张 TTFT 分解图、一张前沿图、一张 baseline 对比表。
5. 同一预算下，oracle 明显优于 random。
6. 至少一个非随机方法能稳定跑出结果。
7. 固定 seed 与配置后结果可复现。

## 6. 第一轮的边界

这轮不承诺：
- SOTA
- 全任务覆盖
- 复杂系统部署
- 所有 selector 都优于 heuristic

这轮只证明：
- 命题成立
- 证据链闭合
- 结论可复现

## 7. 当前进度总结

### 已完成

- 研究目标从“最小可验证闭环”修订为“论文命题验证”
- 第一轮的验证重点已明确
- 主集与辅集已确定为 `LongBench` / `RULER`
- 论文式证据链和验收标准已定义

### 还未完成

- TTFT 分解脚本
- block/page 元数据提取
- full-KV decode oracle 标签导出
- deployable feature table
- selector 与 frontier 计算
- baseline 实验跑数

## 8. 下一步做什么

下一步进入第一轮实验协议细化，具体包括：

1. 定义 LongBench 的任务子集和上下文切片方式
2. 定义 RULER 的辅助验证任务
3. 定义 TTFT 分解口径
4. 定义 block/page 特征字段
5. 定义 oracle 标签构造规则
6. 定义 selector、预算和评估指标
7. 把这些协议固化为可执行脚本和配置

## 9. 整体实验进行到哪一步

当前整体实验处于：

**阶段 0：问题校准**

含义是：
- 题目已经收敛
- 论文命题已经明确
- 数据集角色已经定好
- 但尚未进入正式跑数与结果生成

## 10. 阶段总结报告

### 本次做了什么

- 将第一轮目标从“最小可验证闭环”修订为“论文命题验证”
- 明确了第一轮的六个核心验证点
- 按文档要求收敛了数据集角色：
  - `LongBench` 作为主集
  - `RULER` 作为辅集
  - `Qwen3-8B` 作为统一模型
- 将验收标准写成可检查的证据链
- 明确了当前不应承诺的范围，避免把论文做成系统堆叠

### 下一步做什么

下一步不是直接宣称结果，而是进入第一轮实验协议细化与实现准备：

1. 选定 LongBench 的具体任务子集
2. 规定 TTFT 分解口径
3. 规定 block/page 特征字段
4. 规定 oracle 标签的构造方式
5. 规定 selector、预算、基线和评估指标
6. 把这些协议固化为脚本和配置

### 当前进行到整体实验的哪一步

当前仍处于 **阶段 0：问题校准**。

这意味着：
- 题目、命题、边界已经明确
- 但还没有进入正式的 TTFT 分解、标签生成和 frontier 跑数
- 现阶段的成果是“实验协议已定”，不是“实验结果已出”
