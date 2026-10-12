# 新增论文基线：受限配置直接比较

2026-10-12 更新：WeMM-Embedding-2B 与 GME-2B 的预定 CPU 基线任务全部完成：VSR110、760 条历史验证及 QA text-only、ScienceQA1000 全模态/text-only。每模型缓存 3864 个不同输入的向量。

| 模型/输入 | 正确题数 | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| WeMM-2B 全模态 | 79/110 | 71.82% | 68.56% |
| WeMM-2B text-only | 67/110 | 60.91% | 51.54% |
| GME-2B 全模态 | 67/110 | 60.91% | 58.82% |
| GME-2B text-only | 62/110 | 56.36% | 50.60% |

| 模型 | 8 来源 macro R@1 | 四 QA 来源全模态 | 四 QA 来源 text-only | ScienceQA1000 全模态 | ScienceQA1000 text-only |
|---|---:|---:|---:|---:|---:|
| WeMM-2B | 75.83% | 74.22% | 21.79% | 77.40% | 28.10% |
| GME-2B | 45.57% | 27.35% | 18.70% | 27.30% | 12.40% |

项目模型 CUDA BF16 训练日志：Step500 八来源 macro R@1 为 47.39%，Step750 为 53.06%。Step750 ChartQA 为 30.77%，VSR 为 82/110（74.55%）。Qwen3-VL-Embedding-2B CPU 历史同协议 macro R@1 为 67.85%；WeMM 为 75.83%。模型继续改善，但当前广泛任务指标仍落后于两个更强 embedding 基线。不同设备精度须分列；单项 VSR 不能代替整体结论。

已从全部缓存向量重算 110 条全模态和 text-only 预测、候选 ID、分来源指标及成对图像增益，和原结果完全一致。此核验是缓存数值回放，不是重新推理全部输入。记录见 `artifacts/results/v5/paper_baseline_replay_audit.json`。

同一历史 110 题上，Step500 embedding CPU FP32 为 80/110，Qwen3-VL-Embedding-2B 为 77/110，原生 OmniJev noul 为 69/110，API 为 89/110；各臂详情以 `VSR_NOUL_DIRECT_COMPARISON.zh.md` 为准。Step500 的旧 GPU BF16 是 79/110，不能与 CPU 版混记。仅一题差距不支持优于 WeMM 的显著性或 SOTA 结论。API 是决策接口，各模型原生预处理不同，不是受控的参数量/算力消融。

当前 WeMM/GME 协议为 CPU FP32、2 线程、最多 65536 图像像素及 1800 token，使用固定官方权重、原生最后有效 token 池化及 L2 归一化。WeMM 使用审读的原生 embedding 方法及 tokenizer 追加的 `<embedding>`。GME 将公开 forward 的普通三轴位置编号与池化迁移到 Transformers 5；权重完整加载、无 missing/unexpected keys，但未与独立 Transformers 4.51 环境逐向量对照，因此应标注兼容实现和限制。

这些是受限配置下的直接推理比较，不是原论文全 benchmark/headline 复现。历史 VSR110 是项目内部按图像划分的开发诊断集，不是官方独立 test；上游预训练污染未知。不能利用这些分数选择最终正式测试的 checkpoint/prompt。

上述完整队列已经执行完毕。逐向量缓存绑定协议、图像字节和官方权重，支持断点续跑。失败 v1/v2 记录保留。

VLM2Vec-V2 已检查实际 420 个 DoRA 张量：它们位于 28 层语言 backbone 的 q/k/v/o/down projection，尚需完成 Transformers 5 模块路径迁移与前向等价核验。Ovis 官方代码仓库当前固定版本仅含说明文件，所下载权重实际仍含 thinker/talker/token2wav；不能仅按模型卡描述就把未经核验的加载结果当作官方 embedding。两者尚未完成本地评测。
