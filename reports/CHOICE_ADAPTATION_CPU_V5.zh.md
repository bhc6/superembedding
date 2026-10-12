# 冻结 Step 250 的 choice 专精实验

May demonstrate frozen task-prompt adaptation. These training sources and validation examples are already known; cannot establish unseen-task generalization or publication-level novelty. No query-specific or answer-dependent choice construction.

所有方向在评测前封存；全部报告，没有按结果选择最佳提示。保持四方向、共享投影和输入文字，只有 choice 文本改变。CPU FP32 与原 GPU 验证分数分开；组内比较精度一致。

| 模型 | 来源 | 方向 | 索引 | R@1 % | text-only % | VSR balanced accuracy % |
|---|---|---|---|---:|---:|---:|
| step250 | chartqa | original | both_sides | 16.92 | 7.69 | — |
| step250 | chartqa | original | query_only_original_index | 16.92 | 7.69 | — |
| step250 | chartqa | paraphrase | both_sides | 16.92 | 7.69 | — |
| step250 | chartqa | paraphrase | query_only_original_index | 12.31 | 9.23 | — |
| step250 | chartqa | specialized | both_sides | 7.69 | 9.23 | — |
| step250 | chartqa | specialized | query_only_original_index | 7.69 | 4.62 | — |
| step250 | vsr | original | both_sides | 55.45 | 57.27 | 59.42 |
| step250 | vsr | original | query_only_original_index | 55.45 | 57.27 | 59.42 |
| step250 | vsr | paraphrase | both_sides | 50.00 | 47.27 | 57.28 |
| step250 | vsr | paraphrase | query_only_original_index | 48.18 | 56.36 | 54.45 |
| step250 | vsr | specialized | both_sides | 56.36 | 60.91 | 57.88 |
| step250 | vsr | specialized | query_only_original_index | 55.45 | 60.91 | 53.96 |
| step0_reconstructed | chartqa | original | both_sides | 1.54 | 1.54 | — |
| step0_reconstructed | chartqa | original | query_only_original_index | 1.54 | 1.54 | — |
| step0_reconstructed | chartqa | paraphrase | both_sides | 3.08 | 3.08 | — |
| step0_reconstructed | chartqa | paraphrase | query_only_original_index | 3.08 | 3.08 | — |
| step0_reconstructed | chartqa | specialized | both_sides | 1.54 | 1.54 | — |
| step0_reconstructed | chartqa | specialized | query_only_original_index | 3.08 | 6.15 | — |
| step0_reconstructed | vsr | original | both_sides | 61.82 | 61.82 | 50.00 |
| step0_reconstructed | vsr | original | query_only_original_index | 61.82 | 61.82 | 50.00 |
| step0_reconstructed | vsr | paraphrase | both_sides | 61.82 | 61.82 | 50.00 |
| step0_reconstructed | vsr | paraphrase | query_only_original_index | 61.82 | 61.82 | 50.00 |
| step0_reconstructed | vsr | specialized | both_sides | 57.27 | 61.82 | 49.96 |
| step0_reconstructed | vsr | specialized | query_only_original_index | 61.82 | 61.82 | 50.00 |
| raw_pretrained | chartqa | original | both_sides | 3.08 | 3.08 | — |
| raw_pretrained | chartqa | original | query_only_original_index | 3.08 | 3.08 | — |
| raw_pretrained | chartqa | paraphrase | both_sides | 3.08 | 3.08 | — |
| raw_pretrained | chartqa | paraphrase | query_only_original_index | 3.08 | 3.08 | — |
| raw_pretrained | chartqa | specialized | both_sides | 0.00 | 1.54 | — |
| raw_pretrained | chartqa | specialized | query_only_original_index | 3.08 | 1.54 | — |
| raw_pretrained | vsr | original | both_sides | 61.82 | 61.82 | 50.00 |
| raw_pretrained | vsr | original | query_only_original_index | 61.82 | 61.82 | 50.00 |
| raw_pretrained | vsr | paraphrase | both_sides | 61.82 | 61.82 | 50.00 |
| raw_pretrained | vsr | paraphrase | query_only_original_index | 61.82 | 61.82 | 50.00 |
| raw_pretrained | vsr | specialized | both_sides | 60.91 | 61.82 | 49.26 |
| raw_pretrained | vsr | specialized | query_only_original_index | 61.82 | 61.82 | 50.00 |

both_sides 对查询和候选都用任务固定方向，须重建该任务索引；query_only_original_index 只改查询方向并保留原候选向量。没有按每个查询生成候选向量。

JSON 包含问题类型、VSR 混淆矩阵/预测类别分布、相对原方向的成对区间、视觉贡献变化及训练前后提示收益差。区间为探索性，未校正多重检验。小样本和已见训练来源不能支持未见任务泛化。
