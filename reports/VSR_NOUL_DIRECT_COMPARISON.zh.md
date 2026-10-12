# VSR 110 题：原生 noul、Qwen 与 Step 500

所请求对照已完成并重放缓存核验。Yes 68 / No 42；CPU FP32（API 精度由服务方管理）。这是事后内部验证诊断，不是无泄露主结果。

2026-10-12 独立复核：全部 110 题的标签已对照原始来源记录；输入问题未注入 ground truth 字段，110 张图片的字节哈希互异。重新逐题计数得到下表相同的正确题数，并已重放本地/API 缓存向量的候选排序与正确性。

另按封存顺序预先固定首、中、末三题（vsr:1057、vsr:1846、vsr:989），重新加载官方 OmniJev 与实际 Step500 权重做新推理：两组原生 noul 的 6 个四位小数概率均与缓存完全一致；Step500 图文及 text-only 的 6 个新向量与缓存最大绝对差均为 0，预测一致。这是三题的新推理抽查，不是 110 题全部重新推理；Qwen API 未重新请求。证据：`artifacts/results/v5/vsr_fresh_verification.json`。

精度必须分别报告：历史 GPU BF16 训练验证中 Step500 为 79/110（71.82%）；本表 CPU FP32 评测为 80/110（72.73%）。不同计算设置下正确题数相差一题，不能把本表数字当成历史训练验证原值。

| 模型/接口 | 正确题数 | Accuracy | Balanced accuracy | Text-only accuracy |
|---|---:|---:|---:|---:|
| original/noul | 69/110 | 62.73% | 64.85% | 38.18% |
| step500/embedding | 80/110 | 72.73% | 72.93% | 48.18% |
| qwen2b | 77/110 | 70.00% | 74.82% | 59.09% |
| qwen_api | 89/110 | 80.91% | 77.28% | 66.36% |
| step500/noul | 74/110 | 67.27% | 68.52% | 41.82% |
| original/choice | 72/110 | 65.45% | 57.04% | 60.91% |
| step500/choice | 77/110 | 70.00% | 67.54% | 60.00% |

Always-Yes：68/110，Accuracy 61.82%，Balanced accuracy 50%。Qwen2B 指本地 embedding 模型，不是生成式问答；Qwen API 独立列示，不能冒称某个公开参数规模。

noul 使用官方原 decision head、LM features、温度/偏置与四位小数概率，固定 P(yes)≥0.5，不在此数据上调阈值。两条原生 API 等价性检查已保存；本次逐题核验概率后处理。Step500 embedding 使用已训练 LoRA 和同一步 projection，逐题重放全部 110 个候选的检索。

| 配对比较 | Accuracy 差值（百分点） | 95% 分层配对 bootstrap 区间 |
|---|---:|---|
| step500/embedding/full − original/noul/full | 10.00 | [0.00, 20.91] |
| step500/embedding/full − qwen2b/full | 2.73 | [-7.27, 12.73] |
| step500/embedding/full − qwen_api/full | -8.18 | [-18.18, 2.73] |
| step500/noul/full − original/noul/full | 4.55 | [-1.82, 11.82] |

区间仅描述这组查询的重采样不确定性，不含训练种子、方法选择或预训练污染，不作多重比较校正。

21/110 个完整问题（含指令、规范化后）在历史训练中出现过；JSON 另给 seen/unseen 分组，仅作诊断。换措辞不是新实体。VSR 旧验证也曾参与整体 checkpoint 选择，本次固定 Step500 不会使其变成未见测试。

划分来源补充：V5 的八个训练来源均采自 `HuggingFaceM4/the_cauldron` 的 `train`，本地再按原始图像 RGB 哈希分桶为约 90% train / 10% val；这里的 VSR 110 题不是官方 VSR validation/test。代码见 `scripts/prepare_general_data.py`。当前 VSR 为 913 个训练图像记录、110 个内部验证图像记录，跨 train/val 的原始 RGB 哈希与保存图像文件哈希交集均为 0；问句并未作为分组键，因此不同图片可以配同一句话。

21 个重复问句对应 25 个匹配训练问答条目，不能直接算成 21 个泄露样本。具体例子：验证 `vsr:1373` 和训练 `vsr:174` 都问 “The person is touching the pizza.” 是否符合图片，但图片不同，验证答案 Yes、训练答案 No；验证 `vsr:144` 与训练 `vsr:160` 同问 “The pizza is touching the person.”，也分别为 No、Yes。问句重复属于文本捷径/语言先验风险；是否样本泄露还要看图像、近重复场景和实体。上述零精确哈希交集不排除近重复，也不解决既有验证选择造成的开发暴露。

原始流程另有冻结 projection 三个学习率对照，因父训练队列停止尚未完成；本报告不把缺失组算作已完成，也不为它们等待或修改正在续训的 GPU 任务。

机器可读结果与逐题预测：`artifacts/results/v5/vsr_noul_direct_comparison.json`。
