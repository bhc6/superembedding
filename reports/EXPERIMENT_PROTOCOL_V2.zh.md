# 本地属性记忆实验协议 v2（测试前冻结）

日期：2026-10-11。旧 fashion600 全部作为开发数据；新样本排除这些 ID、像素重复与 dHash 距离 <=4 的相似图。公开 Fashion Product Images Small 固定 revision，两个 shard，2400 图；联合属性分层约 60/20/20。同属性值及组合在训练中出现，不宣称组合外泛化。dHash 不能保证语义级商品完全去重。

固定属性：主色、商品类型、目录受众。OmniJev 0.8B v1.1，65536 像素预算，冻结 backbone；每属性 256d。训练 seed 17/42/91，AdamW，batch 64，500 更新，lr 网格 1e-4/1e-3，每50步验证。所有候选保留；每架构用三 seed 平均验证目标选择 lr，测试仅最终汇总时打开。

架构：raw、冻结乘法 gate、可训练 gate、可训练 gate + raw 残差。以上用单属性 triplet（margin .2）；附加 raw + 联合属性 triplet 的无 output projection、768→768、768→256 三种。同一额外损失、相同采样序列和预算比较这三种；单属性 loss + 全属性交集 loss 等权。前四种验证按属性宏 mAP，后三种按 (属性宏 mAP + 三属性交集 mAP)/2。三属性联合 triplet 从同组合正例采样；负例优先同两属性，以加强区别性。测试报告单属性、所有两属性交集（选择对应 slot 拼接）、三属性交集、R@1、mAP、有效 query 覆盖；output projection 只用于完整三属性向量，保留 slots 支持子属性查询。

外部直接对照：FLAIR 官方 merged30m checkpoint，在相同图、相同固定方向提示上编码，报告冻结特征及相同监督预算 raw projection；这是固定方向图到图检索适配，非 FLAIR 原论文文本到图指标重现。Omni-Attribute 无可用实现/权重不能声称实测。另测 OmniJev 禁用 LoRA 的 Qwen 基础网络分支特征 + 同 raw head（仍保留 OmniJev 特殊 token 嵌入，明确为消融，非原生 Qwen embedding）。不同预训练数据规模不能据此宣称方法全面优越。

恢复协议：原子 last.pt 包括参数、optimizer、step、Python/torch/CUDA RNG、数据/config 指纹；保留 best_head.pt 和每250步快照。进度只在完成 optimizer step 后保存；resume 参数变化应拒绝。训练脚本不读测试标签指标，最后独立 evaluation。LoRA 真图训练也用同样状态恢复，单独报告显存和适用范围。

结果无论正负都保留。随机种子标准差是训练变异，不代表跨数据集置信区间；此小样本研究不能证明新颖性、决策证据迁移或通用多模态记忆能力。
