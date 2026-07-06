# 第一轮阶段总结报告

## 当前状态

当前仍处于 **阶段 0：问题校准**，并完成了**方法层骨架与单元契约验证**。

## 做了什么

- 明确了第一轮论文命题
- 固定了主集 `LongBench`、辅集 `RULER`、模型 `Qwen3-8B`
- 写出了第一轮实验协议
- 记录了第一轮数据表 schema
- 写入了第一轮配置骨架
- 建立了实验目录结构
- 实现了 TTFT、特征、oracle、selector、frontier、validation 的最小纯函数
- 补了单元测试，并通过了 7 项 pytest 验证

## 得到了什么证据

- 得到了协议层证据和方法层单元测试证据
- 7 项 pytest 已通过，说明核心契约没有明显逻辑错误
- 目前还没有真实 LongBench/RULER 跑数证据
- 目前没有 TTFT 分解图、frontier 图或 baseline 表
- 目前没有 full-KV decode 标签或 selector 真实跑数结果

## 哪些验收已通过

- 实验目标已明确
- 数据集角色已明确
- 证据链结构已明确
- 数据表 schema 已明确
- 方法层核心纯函数契约已通过单元测试

## 哪些还没通过

- TTFT 分解还未接入真实数据或 LongBench/RULER 跑数
- block/page 特征提取还未接入真实数据
- oracle 标签导出还未基于 full-KV decode 跑出真实结果
- deployable feature table 还未基于真实样本构建
- selector 还未在真实预算与真实任务上运行
- frontier 还未基于真实实验生成
- 复现性还未在真实数据流程上验证

## 下一步做什么

1. 把纯函数封装进脚本入口
2. 接入 LongBench 小样本校准集
3. 接入 RULER 辅助样本
4. 跑真实 TTFT 分解和 feature/oracle 导出
5. 生成 selector 对比、frontier 和图表
6. 再做一次固定 seed 的复现检查
