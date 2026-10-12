# 同协议基线与问答捷径检查

完整验证：760 张图像，每图一条按 ID 固定选取的监督。每来源使用该来源全部验证目标作候选，保留重复答案并沿用训练正例规则；指标为 8 来源等权 R@1。问答控制只覆盖 ChartQA、CLEVR、DocVQA、VSR 四来源。

| 模型 / 执行路径 | 8 来源平均 % | chart2text | chartqa | clevr | docvqa | localized_narratives | screen2words | textcaps | vsr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| general_3cad7dfc34c7_cuda | 41.56 | 46.25 | 16.92 | 38.46 | 18.75 | 46.36 | 42.00 | 68.29 | 55.45 |
| qwen2b_cpu | 67.85 | 66.25 | 61.54 | 73.50 | 52.08 | 81.82 | 51.00 | 86.59 | 70.00 |
| qwen_api_cpu | 83.77 | 88.75 | 89.23 | 79.49 | 93.75 | 90.00 | 59.00 | 89.02 | 80.91 |
| raw_cached | 12.61 | 5.00 | 3.08 | 9.40 | 0.00 | 6.36 | 3.00 | 12.20 | 61.82 |
| raw_cuda | 12.49 | 1.25 | 4.62 | 9.40 | 0.00 | 5.45 | 4.00 | 13.41 | 61.82 |
| step0_reconstructed_cpu | 12.80 | 6.25 | 1.54 | 8.55 | 2.08 | 8.18 | 3.00 | 10.98 | 61.82 |
| step0_reconstructed_cuda | 12.17 | 5.00 | 1.54 | 8.55 | 1.04 | 5.45 | 3.00 | 10.98 | 61.82 |
| 训练日志 Step 250 / CUDA BF16 | 41.56 | 46.25 | 16.92 | 38.46 | 18.75 | 46.36 | 42.00 | 68.29 | 55.45 |
| 训练日志 Step 500 / CUDA BF16 | 47.39 | 51.25 | 21.54 | 44.44 | 27.08 | 52.73 | 42.00 | 68.29 | 71.82 |

raw_cached / step0_reconstructed_cpu 使用冻结 CPU FP32 特征；不能直接把它们与训练日志的差当作精度受控的训练增益。CUDA 对照完成后单列。Step 0 原始文件已被覆盖，现在是按原种子、原构造器重建；RNG 与历史断点一致和参数差检查通过，但没有原始 Step 0 哈希，不能声称找回了原件。

Qwen API 是供应商可变别名，不等同于本地 2B。qwen2b 使用官方固定权重、默认指令及末 token 池化，图像限制 65536 pixels；这不是官方默认高分辨率设置。模型间原生 tokenizer/图像处理不同，同协议指相同数据、查询、候选、正例与指标。

## 多数类与 text-only

| 模型 / 控制 | 四问答来源平均 % | ChartQA % | CLEVR % | DocVQA % | VSR % |
|---|---:|---:|---:|---:|---:|
| train_source_majority | 21.87 | 1.54 | 23.08 | 1.04 | 61.82 |
| train_type_majority | 25.29 | 1.54 | 36.75 | 1.04 | 61.82 |
| general_3cad7dfc34c7_cuda / qa_full | 32.40 | 16.92 | 38.46 | 18.75 | 55.45 |
| general_3cad7dfc34c7_cuda / qa_text_only | 24.96 | 9.23 | 19.66 | 14.58 | 56.36 |
| qwen2b_cpu / qa_full | 64.28 | 61.54 | 73.50 | 52.08 | 70.00 |
| qwen2b_cpu / qa_text_only | 33.92 | 15.38 | 39.32 | 21.88 | 59.09 |
| qwen_api_cpu / qa_full | 85.84 | 89.23 | 79.49 | 93.75 | 80.91 |
| qwen_api_cpu / qa_text_only | 32.71 | 12.31 | 28.21 | 23.96 | 66.36 |
| raw_cuda / qa_full | 18.96 | 4.62 | 9.40 | 0.00 | 61.82 |
| raw_cuda / qa_text_only | 18.59 | 3.08 | 4.27 | 5.21 | 61.82 |
| step0_reconstructed_cuda / qa_full | 18.24 | 1.54 | 8.55 | 1.04 | 61.82 |
| step0_reconstructed_cuda / qa_text_only | 17.74 | 1.54 | 3.42 | 4.17 | 61.82 |

多数答案只从训练记录统计，每训练图固定一条监督；同频按答案字典序，未见问题类型回退到来源多数类。多数答案不在验证候选中则计错，不按验证标签重新选择。按问题类型的结果使用仅看问题文字的固定规则，不冒充官方类型。text-only 只删除查询图像，使用原封不动的候选向量。

## 问题类型明细

| 模型 | 来源 | 问题类型 | n | 完整输入 % | text-only % |
|---|---|---|---:|---:|---:|
| general_3cad7dfc34c7_cuda | chartqa | color | 1 | 100.00 | 0.00 |
| general_3cad7dfc34c7_cuda | chartqa | comparison | 4 | 0.00 | 0.00 |
| general_3cad7dfc34c7_cuda | chartqa | count | 14 | 28.57 | 7.14 |
| general_3cad7dfc34c7_cuda | chartqa | location | 1 | 0.00 | 0.00 |
| general_3cad7dfc34c7_cuda | chartqa | other | 40 | 12.50 | 10.00 |
| general_3cad7dfc34c7_cuda | chartqa | time | 4 | 0.00 | 0.00 |
| general_3cad7dfc34c7_cuda | chartqa | yes_no | 1 | 100.00 | 100.00 |
| general_3cad7dfc34c7_cuda | clevr | color | 25 | 24.00 | 20.00 |
| general_3cad7dfc34c7_cuda | clevr | comparison | 2 | 50.00 | 0.00 |
| general_3cad7dfc34c7_cuda | clevr | count | 32 | 31.25 | 6.25 |
| general_3cad7dfc34c7_cuda | clevr | material | 14 | 42.86 | 21.43 |
| general_3cad7dfc34c7_cuda | clevr | other | 12 | 41.67 | 16.67 |
| general_3cad7dfc34c7_cuda | clevr | shape | 18 | 44.44 | 44.44 |
| general_3cad7dfc34c7_cuda | clevr | yes_no | 14 | 64.29 | 21.43 |
| general_3cad7dfc34c7_cuda | docvqa | count | 5 | 0.00 | 0.00 |
| general_3cad7dfc34c7_cuda | docvqa | location | 2 | 50.00 | 50.00 |
| general_3cad7dfc34c7_cuda | docvqa | other | 75 | 21.33 | 17.33 |
| general_3cad7dfc34c7_cuda | docvqa | time | 14 | 7.14 | 0.00 |
| general_3cad7dfc34c7_cuda | vsr | spatial_statement_verification | 110 | 55.45 | 56.36 |
| qwen2b_cpu | chartqa | color | 1 | 100.00 | 0.00 |
| qwen2b_cpu | chartqa | comparison | 4 | 50.00 | 25.00 |
| qwen2b_cpu | chartqa | count | 14 | 64.29 | 14.29 |
| qwen2b_cpu | chartqa | location | 1 | 100.00 | 0.00 |
| qwen2b_cpu | chartqa | other | 40 | 60.00 | 12.50 |
| qwen2b_cpu | chartqa | time | 4 | 50.00 | 25.00 |
| qwen2b_cpu | chartqa | yes_no | 1 | 100.00 | 100.00 |
| qwen2b_cpu | clevr | color | 25 | 80.00 | 36.00 |
| qwen2b_cpu | clevr | comparison | 2 | 50.00 | 50.00 |
| qwen2b_cpu | clevr | count | 32 | 65.62 | 28.12 |
| qwen2b_cpu | clevr | material | 14 | 57.14 | 50.00 |
| qwen2b_cpu | clevr | other | 12 | 75.00 | 33.33 |
| qwen2b_cpu | clevr | shape | 18 | 100.00 | 61.11 |
| qwen2b_cpu | clevr | yes_no | 14 | 64.29 | 35.71 |
| qwen2b_cpu | docvqa | count | 5 | 20.00 | 0.00 |
| qwen2b_cpu | docvqa | location | 2 | 50.00 | 50.00 |
| qwen2b_cpu | docvqa | other | 75 | 53.33 | 22.67 |
| qwen2b_cpu | docvqa | time | 14 | 57.14 | 21.43 |
| qwen2b_cpu | vsr | spatial_statement_verification | 110 | 70.00 | 59.09 |
| qwen_api_cpu | chartqa | color | 1 | 100.00 | 0.00 |
| qwen_api_cpu | chartqa | comparison | 4 | 75.00 | 25.00 |
| qwen_api_cpu | chartqa | count | 14 | 92.86 | 14.29 |
| qwen_api_cpu | chartqa | location | 1 | 100.00 | 0.00 |
| qwen_api_cpu | chartqa | other | 40 | 90.00 | 10.00 |
| qwen_api_cpu | chartqa | time | 4 | 75.00 | 25.00 |
| qwen_api_cpu | chartqa | yes_no | 1 | 100.00 | 0.00 |
| qwen_api_cpu | clevr | color | 25 | 92.00 | 20.00 |
| qwen_api_cpu | clevr | comparison | 2 | 50.00 | 50.00 |
| qwen_api_cpu | clevr | count | 32 | 56.25 | 31.25 |
| qwen_api_cpu | clevr | material | 14 | 71.43 | 35.71 |
| qwen_api_cpu | clevr | other | 12 | 83.33 | 25.00 |
| qwen_api_cpu | clevr | shape | 18 | 100.00 | 38.89 |
| qwen_api_cpu | clevr | yes_no | 14 | 92.86 | 14.29 |
| qwen_api_cpu | docvqa | count | 5 | 80.00 | 0.00 |
| qwen_api_cpu | docvqa | location | 2 | 100.00 | 0.00 |
| qwen_api_cpu | docvqa | other | 75 | 96.00 | 28.00 |
| qwen_api_cpu | docvqa | time | 14 | 85.71 | 14.29 |
| qwen_api_cpu | vsr | spatial_statement_verification | 110 | 80.91 | 66.36 |
| raw_cuda | chartqa | color | 1 | 0.00 | 0.00 |
| raw_cuda | chartqa | comparison | 4 | 0.00 | 0.00 |
| raw_cuda | chartqa | count | 14 | 14.29 | 7.14 |
| raw_cuda | chartqa | location | 1 | 0.00 | 0.00 |
| raw_cuda | chartqa | other | 40 | 2.50 | 0.00 |
| raw_cuda | chartqa | time | 4 | 0.00 | 25.00 |
| raw_cuda | chartqa | yes_no | 1 | 0.00 | 0.00 |
| raw_cuda | clevr | color | 25 | 4.00 | 4.00 |
| raw_cuda | clevr | comparison | 2 | 0.00 | 0.00 |
| raw_cuda | clevr | count | 32 | 25.00 | 12.50 |
| raw_cuda | clevr | material | 14 | 14.29 | 0.00 |
| raw_cuda | clevr | other | 12 | 0.00 | 0.00 |
| raw_cuda | clevr | shape | 18 | 0.00 | 0.00 |
| raw_cuda | clevr | yes_no | 14 | 0.00 | 0.00 |
| raw_cuda | docvqa | count | 5 | 0.00 | 0.00 |
| raw_cuda | docvqa | location | 2 | 0.00 | 0.00 |
| raw_cuda | docvqa | other | 75 | 0.00 | 5.33 |
| raw_cuda | docvqa | time | 14 | 0.00 | 7.14 |
| raw_cuda | vsr | spatial_statement_verification | 110 | 61.82 | 61.82 |
| step0_reconstructed_cuda | chartqa | color | 1 | 0.00 | 0.00 |
| step0_reconstructed_cuda | chartqa | comparison | 4 | 0.00 | 0.00 |
| step0_reconstructed_cuda | chartqa | count | 14 | 0.00 | 7.14 |
| step0_reconstructed_cuda | chartqa | location | 1 | 0.00 | 0.00 |
| step0_reconstructed_cuda | chartqa | other | 40 | 2.50 | 0.00 |
| step0_reconstructed_cuda | chartqa | time | 4 | 0.00 | 0.00 |
| step0_reconstructed_cuda | chartqa | yes_no | 1 | 0.00 | 0.00 |
| step0_reconstructed_cuda | clevr | color | 25 | 4.00 | 4.00 |
| step0_reconstructed_cuda | clevr | comparison | 2 | 0.00 | 0.00 |
| step0_reconstructed_cuda | clevr | count | 32 | 21.88 | 9.38 |
| step0_reconstructed_cuda | clevr | material | 14 | 14.29 | 0.00 |
| step0_reconstructed_cuda | clevr | other | 12 | 0.00 | 0.00 |
| step0_reconstructed_cuda | clevr | shape | 18 | 0.00 | 0.00 |
| step0_reconstructed_cuda | clevr | yes_no | 14 | 0.00 | 0.00 |
| step0_reconstructed_cuda | docvqa | count | 5 | 0.00 | 20.00 |
| step0_reconstructed_cuda | docvqa | location | 2 | 0.00 | 0.00 |
| step0_reconstructed_cuda | docvqa | other | 75 | 1.33 | 2.67 |
| step0_reconstructed_cuda | docvqa | time | 14 | 0.00 | 7.14 |
| step0_reconstructed_cuda | vsr | spatial_statement_verification | 110 | 61.82 | 61.82 |

小样本类型仅作描述性分析。每来源完整输入减 text-only 的成对 bootstrap 区间在对应 JSON 中；按查询重采样，不把共享答案视为独立任务，不做多重检验后的显著性宣称。

## 外部测试与机制评测的锁定流程

当前 Step 250 是首轮完整验证选出的中途 checkpoint，已封存完整 LoRA＋head 并验证重载一致性。最终模型仍由预定 2000 步训练的验证分数选择。最终器先重跑同协议完整输入/text-only，再评测官方 MMEB ScienceQA、Urban1k 共同子集和成对证据诊断；不依外部结果换 checkpoint。

外部数据已做 RGB/dHash 和感知哈希近重复筛查，不能保证穷尽预训练污染。已有合成证据集只有一种渲染风格，不能替代自然图像、未见组合的机制证据；还不能宣称达到发表水平。

## 执行状态

### priority_gpu

```json
{
  "state": "completed",
  "training_resume_supervisor_pid": 8688,
  "resumed_from_step": 278,
  "selected_step": 250,
  "selection": "First full validation checkpoint; interim, not final 2000-step best",
  "checkpoint_sha256": "a51170d12c2886838599b61663566f8626b3bf643f2aecccccf9e6f57430ce38",
  "export_sha256": "3cad7dfc34c749bc8b0b25782c3026cd98f49ef5adcbc0f1fbf7cca2a52af016",
  "test_used_for_selection": false,
  "pid": 33400
}
```

### qwen2b

```json
{
  "state": "completed",
  "result": "artifacts\\results\\v5\\validation_controls\\qwen2b_cpu.json",
  "unique_objects": 1688
}
```

### training

```json
{
  "state": "paused",
  "step": 556,
  "last_checkpoint_step": 556
}
```

