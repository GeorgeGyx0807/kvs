# 第一轮脚本说明

当前脚本目录包含第一轮最小骨架，并已接通 LongBench 小样本 smoke；正式 LongBench/RULER 批处理仍需后续扩大跑数。

预期脚本：
- `ttft_decompose.py`
- `extract_block_features.py`
- `export_oracle_labels.py`
- `build_deployable_features.py`
- `check_leakage.py`
- `run_selectors.py`
- `compute_frontier.py`
- `summarize_first_round.py`

这些脚本的共同原则：
- 只接受可复现配置
- 只输出可检查的表和图
- 不把 oracle 信息写进部署特征

## 目前的约束

- 这些脚本现在只支持最小 JSON / 参数输入
- `download_hf_dataset.py` 已能从 Hugging Face `THUDM/LongBench` 的官方 `data.zip` 导出 LongBench 样本
- `run_kv3d_gpu_profile.py` 已能在沙箱外 GPU 上运行 Qwen3-8B 的小样本 3D KV profiling smoke
- 当前仍未承担正式全量 LongBench / RULER 批处理
- 目前不输出论文图表，只输出最小可验证结果

## 目标模式下的正确理解

这套脚本现在只是让实验协议有了可执行入口。
真正的实验证据仍然要来自后续接入真实样本、真实 decode、真实 selector 结果之后的输出。
