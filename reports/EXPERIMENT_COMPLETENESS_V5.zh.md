# 当前实验完整性审计

审计时间：2026-10-11T15:35:20.557224+00:00。完整项目完成：否。

数据泄露审查未通过：CUB/DTD 已确认跨划分同图/同场景；V5 DocVQA 存在实体和固定答案交叉。下列数值回放通过不代表实验无泄露。详见 DATA_LEAKAGE_REVIEW.zh.md。

主训练已审计到 Step 556，验证最佳为 Step 500，8 来源平均 R@1 为 47.39%。实时步数以训练状态文件为准。

| 检查 | 状态 |
|---|---|
| training_history_checkpoints_and_implementation | 通过 |
| selected_step250_export_all_tensors | 通过 |
| queued_frozen_lora_control_implementation | 通过 |
| validation_raw_cuda | 通过 |
| validation_step0_reconstructed_cuda | 通过 |
| validation_general_3cad7dfc34c7_cuda | 通过 |
| validation_qwen2b_cpu | 通过 |
| validation_qwen_api_cpu | 通过 |
| train_only_majority_controls | 通过 |
| choice_all_72_cells_and_cache_replay | 通过 |
| frozen_architecture_grid | 通过 |
| scienceqa_general_3cad7dfc34c7_cpu | 通过 |
| scienceqa_general_3cad7dfc34c7_cpu_text_only | 通过 |
| transfer_general_3cad7dfc34c7_cpu | 通过 |
| transfer_omnijev_raw_cpu | 通过 |
| scienceqa_omnijev_raw_cpu | 通过 |
| scienceqa_omnijev_raw_cpu_text_only | 通过 |

本次从实际缓存向量重算检索预测和分数，验证分组覆盖与断点；不把运行状态 completed 当作充分证据。

## 尚未完成

- LEAKAGE BLOCK: CUB/DTD confirmed duplicate photos/scenes; V5 DocVQA entity overlap; all historical results exploratory until versioned remediation and fresh runs
- 2000-step general training and final validation-selected export/evaluation
- Matched frozen-versus-LoRA run waits for parent training AND final evaluation
- VLM2Vec/GME/WeMM/Ovis inference comparisons (download is not evaluation)
- Unseen natural/compositional evidence and broader data/method study
- True single-branch/direction-count/decision-pretraining controls and general LoRA seed replication
- Final data rights/model card/release audit; no Git publication until complete

## 已修复与证据边界

历史 transfer 缓存键缺少输入字节与实现哈希，当前回放一致不代表历史文件从未变化。已建立事后输入清单；最终训练评测入口新增评测前/后输入及代码哈希校验，不改动其他队列已封存的评测脚本。

ScienceQA 的 CPU 同精度 raw 与 text-only 缺项由独立后台队列补跑，状态在 external_control_completion/status.json；仅当结果存在并通过回放才算完成。

冻结/LoRA 测试加入默认 pytest；同时修复其导入时全局更改 CUDA_VISIBLE_DEVICES 的副作用。合并后 70 项测试通过，包括 CPU/CUDA 精确断点回放。该修复只涉及测试，不修改现有训练实现。

初次数值审计发现 BLAS 使用 2 与 4 个 CPU 线程时，raw/Step 0 的重复答案可能选到不同的候选记录 ID，但正确性和全部分数一致。按原执行器线程数重算后，所有预测记录逐条一致；保留线程设置作为复现条件。

## 结果不能被解释为全面成功

Step 250 专精 choice：ChartQA 双方重编码从 16.92% 降至 7.69%；VSR 总分从 55.45% 升至 56.36%，但 balanced accuracy 从 59.42% 降至 57.88%，text-only 反而更高。这不支持当前提示方案改善视觉泛化。

Step 250 外部 ScienceQA 为 14.4%，text-only 7.1%；合成证据宏平均从 raw 的 26.17% 升至 51.95%，但空间项差值区间跨零。更强基线和自然证据、最终模型结果仍需完整报告。

不修改已封存训练或评测实现，不按外部结果选模型；不提交 Git。
