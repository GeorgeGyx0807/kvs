# 第一轮实验数据表 Schema

## 1. block_feature_table

用途：保存部署时可见的 block/page 特征。

必须字段：
- `sample_id`
- `block_id`
- `layer`
- `head`
- `position_start`
- `position_end`
- `kv_bytes`
- `prefill_attention_mass`
- `prefill_entropy`
- `hidden_norm`
- `similarity`

约束：
- 只能包含 prefill 可见信息
- 不能包含 decode attention
- 不能包含 oracle label

## 2. oracle_label_table

用途：保存 full-KV decode 导出的离线教师信号。

必须字段：
- `sample_id`
- `block_id`
- `decode_attention_mass`
- `oracle_utility`
- `oracle_rank`

约束：
- 只用于离线分析、标签构造和上界比较
- 不能进入 deployable feature table

## 3. deployable_feature_table

用途：保存最终 selector 的输入特征。

必须字段：
- `sample_id`
- `block_id`
- `layer`
- `head`
- `position_start`
- `position_end`
- `kv_bytes`
- `prefill_attention_mass`
- `prefill_entropy`
- `hidden_norm`
- `similarity`

显式禁止字段：
- `decode_attention`
- `decode_attention_mass`
- `oracle`
- `future`
- `full_decode`

## 4. 对齐键

三张表必须都能通过以下键对齐：
- `sample_id`
- `block_id`

