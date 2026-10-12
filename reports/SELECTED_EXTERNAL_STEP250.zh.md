# Step 250 封存模型的外部诊断

> 2026-10-11 泄露审查更新：V5 已在 Step 556 保存断点并暂停；同数据 frozen/LoRA 队列已停止。CUB/DTD 同图/同场景与 DocVQA 实体交叉尚未整改完成，历史分数仅作探索性结果。以 [数据泄露复核](DATA_LEAKAGE_REVIEW.zh.md) 为准；不得直接恢复旧主结果流程。

这是按首轮完整验证封存的中途 checkpoint，不是 2000 步训练最终模型；结果只作诊断，不用于挑选 checkpoint 或架构。完整 LoRA＋head 使用与评测一致的加载器。

官方 MMEB ScienceQA（1000 查询）：完整输入 14.40%；text-only 7.10%。相同候选向量，成对差值与区间见 JSON。

| Urban1k 999 子集 | 文搜图 R@1 % | 图搜文 R@1 % |
|---|---:|---:|
| general_3cad7dfc34c7_cpu | 22.82 | 25.53 |
| omnijev_raw_cpu | 2.30 | 4.00 |

| 证据属性 | 模型 | 原图及干扰图均优于证据改变图 % | 相对 raw 差值 pp [95% CI] |
|---|---|---:|---:|
| color | qwen3_vl | 90.62 | 65.62 [51.56, 78.12] |
| color | clip_b32 | 68.75 | 43.75 [29.69, 57.81] |
| color | siglip2_b16 | 85.94 | 60.94 [45.31, 75.00] |
| color | flair_global | 84.38 | 59.38 [45.31, 73.44] |
| color | flair_native | 84.38 | 59.38 [45.31, 71.88] |
| color | omnijev_raw_cpu | 25.00 | 0.00 [0.00, 0.00] |
| color | general_3cad7dfc34c7_cpu | 70.31 | 45.31 [31.25, 59.38] |
| number | qwen3_vl | 98.44 | 71.88 [60.94, 82.81] |
| number | clip_b32 | 23.44 | -3.12 [-18.75, 12.50] |
| number | siglip2_b16 | 100.00 | 73.44 [62.50, 84.38] |
| number | flair_global | 50.00 | 23.44 [6.25, 40.62] |
| number | flair_native | 46.88 | 20.31 [3.12, 37.50] |
| number | omnijev_raw_cpu | 26.56 | 0.00 [0.00, 0.00] |
| number | general_3cad7dfc34c7_cpu | 40.62 | 14.06 [0.00, 28.12] |
| position | qwen3_vl | 26.56 | 3.12 [-7.81, 14.06] |
| position | clip_b32 | 21.88 | -1.56 [-12.50, 9.38] |
| position | siglip2_b16 | 56.25 | 32.81 [12.50, 51.56] |
| position | flair_global | 28.12 | 4.69 [-10.94, 20.31] |
| position | flair_native | 28.12 | 4.69 [-9.38, 18.75] |
| position | omnijev_raw_cpu | 23.44 | 0.00 [0.00, 0.00] |
| position | general_3cad7dfc34c7_cpu | 28.12 | 4.69 [-10.94, 20.31] |
| shape | qwen3_vl | 70.31 | 40.62 [25.00, 56.25] |
| shape | clip_b32 | 48.44 | 18.75 [4.69, 32.81] |
| shape | siglip2_b16 | 75.00 | 45.31 [29.69, 59.38] |
| shape | flair_global | 62.50 | 32.81 [15.62, 48.44] |
| shape | flair_native | 70.31 | 40.62 [24.96, 56.25] |
| shape | omnijev_raw_cpu | 29.69 | 0.00 [0.00, 0.00] |
| shape | general_3cad7dfc34c7_cpu | 68.75 | 39.06 [25.00, 53.12] |
| macro_all_attributes | qwen3_vl | 71.48 | 45.31 [37.50, 52.73] |
| macro_all_attributes | clip_b32 | 40.62 | 14.45 [7.03, 22.27] |
| macro_all_attributes | siglip2_b16 | 79.30 | 53.12 [45.70, 60.55] |
| macro_all_attributes | flair_global | 56.25 | 30.08 [21.09, 39.06] |
| macro_all_attributes | flair_native | 57.42 | 31.25 [22.66, 39.84] |
| macro_all_attributes | omnijev_raw_cpu | 26.17 | 0.00 [0.00, 0.00] |
| macro_all_attributes | general_3cad7dfc34c7_cpu | 51.95 | 25.78 [17.97, 34.38] |

证据区间按完整场景做 10000 次成对重采样，四属性同时保留，未做多重比较校正。位置项同时交换两个物体；64/64 等边界比例的经验 bootstrap 可退化，不意味着总体零不确定性。

无关干扰不变在这里指给定查询的检索偏好保持，不能要求通用向量完全相等：对当前查询无关的物体属性，可能是另一个查询的关键证据。固定向量在所有查询间复用，测试时没有重新按查询编码候选。

单一合成渲染风格不等于真实场景机制已证实；Urban1k 强基线接近饱和。来源留出、污染检查和这些诊断均不能单独支持 SOTA 或发表级结论。
