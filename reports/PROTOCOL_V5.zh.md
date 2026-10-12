# V5：通用证据记忆训练与独立复现（进行中）

本阶段不宣称达到 SOTA、证明新颖性或已达到发表水平。V3/V4 的属性实验已经完成；V5 扩展为图像、文本、图文混合的统一检索训练。2026-10-11 冻结第一阶段配置。未完成全部工作前不新增 GitHub 提交或发布。

## 必须检验的研究假设

从 OmniJev 决策预训练继承的证据表示，经过固定方向分支和共享投影，能否产生可提前编码、跨查询复用的记忆向量，并在证据变化、无关干扰、未见领域上优于同维度全局向量？接投影层、多向量或指令检索本身都不是新颖性证明。

WeMM 已使用 Qwen3.5 backbone 和专用 embedding token；FLAIR 已实现文本条件注意力池化；VLM2Vec、Qwen3-VL、Ovis 已系统训练任务指令检索。比较必须覆盖这些工作，不能只对比旧 CLIP 或在 Fashion/CUB/DTD 上拟合属性头。

## 已冻结的初始训练配置

- 初始化：官方 OmniJev-0.8B-v1.1 adapter + Qwen3.5-0.8B；embedding projection 随机初始化。禁止使用历史 Fashion/CUB/DTD 训练头。
- 结构：4 个固定方向（语义实体、外观属性、空间关系、文字数字），每个方向 raw hidden → 共享 LayerNorm/Linear(256) → L2；拼接并 L2 得到 1024 维。候选编码不依赖查询。此次不加 output projection。
- 选择理由：沿用 V3/V4 支持的 raw 起点，不宣称这些旧测试已经证明新领域或新方向数量最优。方向的解释性也必须独立检验。
- 可训练部分：已有 LoRA 与 projection；冻结稠密 backbone。保持原 OmniJev 分支隔离路径，不把普通串行选项当成独立 heads。
- 数据：the_cauldron pinned train-only revision；8 来源各计划 1024 图像，自然图像描述、TextCaps、Screen2Words、Chart2Text、DocVQA、ChartQA、CLEVR、VSR。图像限长边 640；训练 max_pixels=65536，文本最多 256 tokens。这是存储/显存折中，可能损害 OCR/文档能力。
- 来源级留出：AI2D、ScienceQA 各计划 96 张，仅作新来源诊断，不参加训练、验证选模。它们仍是原始 train split，不能叫官方测试。
- 按图像哈希确定 train/val；跨分割完全相同 RGB 或完全相同 dHash 排除；再与完整官方 MMEB ScienceQA、Urban1k 图像做排除。尚未完成穷尽近重复和语义去污染，继承模型预训练污染未知。
- 训练任务：描述→图像、图像+问题→答案。每步同源 16 对，均衡轮换来源；同图、同描述或相同规范化答案为多正例，避免明显假负例。没有启用未经验证的在线难负例。
- 方法：multi-positive InfoNCE，temperature=0.05；逐对象梯度缓存/回放；AdamW，lr=1e-5，weight_decay=.01，梯度裁剪 1；warmup 100 步后 cosine 到 0.1 倍。
- 初始预算 2000 步。前 500 步限制为各来源采样序列前 256 个位置，此后使用全来源清单。每 250 步用全部来源内验证图像选 best，各图按固定种子选一条监督，指标为来源内检索 R@1 的宏平均。V5.1 共 760 张验证图像。
- 禁用 LoRA dropout、启用确定性 CUDA；缓存回放逐对象检查向量一致性。代码、数据、模型、配置哈希写入 checkpoint，禁止配置变更后伪装无缝续跑。

该阶段只有图像/文本/图文，不包含音频、视频或真正的长期交互记忆训练。约 8 千图像和小批次并不等同于顶级论文的千万级数据、1024 全局 batch 或多机预算。若验证支持继续，再单独冻结扩大数据、蒸馏/难例等第二阶段协议，保留第一阶段权重。

## V5.1：首次验证前的显式修复

第 60 步暂停原运行 `general_stage1`，在任何验证选模发生前迁移到 `general_stage1_v51`，协议 `v5-general-2-full-validation`。原断点和原实现快照保留。训练记录 7,388 条及顺序完全相同；迁移前后的模型、optimizer、全部 RNG、步号状态摘要一致，来源队列保留。验证从每域最多 16 条监测改为所有有效验证图像。迁移证据见 `artifacts/results/v5/validation_migration.json`。

近重复审计：63 位 DCT pHash 距离≤6 且 64 位 dHash 距离≤6，共 87 对候选、41 条不同验证记录。人工确认 ChartQA:16929 与 Chart2Text:5810 是同一张 Boston Red Sox 图表，编码/缩放不同；另有同应用相似界面。部分图表只是模板相同，不能认定重复。本次保守地隔离全部 41 条验证候选，可能改变验证分布，需在报告中保留这一限制；没有修改训练集。新分割为 train=7388、val=760、OOD=163，manifest SHA256 `470b7a9b2ca8017fdaad15c404f54b9179a0d556434807d252e89102e9a92f44`。

对完整外部 ScienceQA/Urban1k 的 2,000 张图没有筛出上述阈值的近重复候选；这不是穷尽去污染证明。原始审计和隔离清单均保留。

## 恢复和保存

每 5 步原子写 `last.pt`，包含全部可训练权重、AdamW、Python/torch/CUDA RNG、来源采样队列；验证最佳保存 `best.pt` 的完整恢复状态。每 500 步保存不可变可训练权重里程碑。不可变里程碑是推理权重，恢复训练使用 last/best；不把只有权重的文件称为完整断点。空间不足停止，保留所有已产生的里程碑、历史 V3/V4 训练和发布资产。

后台 supervisor 最多自动重启 2 次。数据、指纹、非有限梯度或磁盘错误需修复后重启，不能无穷重试。`training.lock` 阻止重复启动同一目录。创建 `STOP_AFTER_CHECKPOINT` 可在完成当前优化器步并保存后暂停。

真实 CUDA 检查：文本、图像、混合输入都反向传播；300 个 LoRA 参数张量有非零梯度，峰值约 3659 MiB；确定性修复后真实模型从同一断点重放优化器更新，参数/优化器/RNG 状态哈希一致。CPU 单测另比较完整批次与梯度回放，并验证重复目标正例掩码。详见 `artifacts/results/v5/general_resume_audit.json`。

## 基准协议与结论限制

1. MMEB ScienceQA：完整官方 1000 查询、每题原始 736 候选，Hit@1；固定 TIGER-Lab/MMEB-eval revision。保持原始正例文本与候选列表，确定性文本哈希处理分数相同候选，避免索引 0 偏置。单个任务不是 MMEB 全榜。
2. Urban1k：官方 FLAIR 1000 条注释、完整候选集合、原生检索函数。FLAIR 原生文本条件池化与全局表示分别报告；第一样本对齐官方评分实现。
3. Urban1k API 对比：官方注释 index 548 / 592.jpg 描述为空，API 不接受空文本。因此所有模型统一另报 999 非空子集。不可与官方完整 1000 条表混合。
4. CLIP/SigLIP 在 ScienceQA 的图像+文本均值融合是明确标注的启发式控制，并非其论文提出的问答模型。DINOv2 没有原生文本空间，不伪造跨模态分数。
5. Qwen API alias 不等于公开 2B/8B checkpoint。缓存原始向量、usage、request_id 和时间；单独标注 API 结果。
6. 榜单报告分数与本机实测严格分表。冻结官方 leaderboard revision，按 190 任务报告覆盖数和缺失计零均值；不拿只覆盖部分任务的可用均值冒充总分。
7. 旧测试以及本阶段已看过的公开测试，只作报告与诊断；不得根据其结果重选新架构、学习率或 checkpoint。后续确认性结论仍需未见任务、预注册消融、多种子和置信区间。

证据诊断 v1 的无训练对照已扩展到 Qwen API、CLIP-B/32、SigLIP2-B/16、FLAIR 原生条件池化及全局向量。FLAIR 原生评分与官方函数核对；其查询相关池化是重排对照，不冒充固定 ANN 向量。统计按 64 个完整场景做 10,000 次成对 bootstrap，保留每个场景内四种干预和不同模型配对；区间未作多重比较校正，满分时经验 bootstrap 退化，另说明 Wilson 区间。结果见 `EVIDENCE_V5.zh.md`。

ScienceQA 的完整输入、纯文本、纯图像和错配图片控制使用同一冻结候选库、同一 1000 查询，错配为固定循环位移。同期 8 查询 API 漂移探针通过。该消融检验正确模态对分数的贡献，不排除预训练污染，也不证明因果推理能力。

## 仍未完成的发表所需证据

- VLM2Vec-V2、GME、WeMM、Ovis 官方 checkpoint 在相同任务上的本机推理复现；完整大模型当前受存储限制，不能把读代码/引用分数写成复现完成。
- 同参数量独立属性头 vs 共享头；单全局向量 vs 固定多分支；不同 K、同总维度、output projection；OmniJev 决策预训练 vs 原始 backbone；LoRA vs 冻结；计算/存储预算一致。
- 足够规模的独立验证与跨域测试；自然细粒度和多条件组合检索。已生成原创成对证据诊断 v1（64 场景、256 干预、384 图像，CC0），但只有一种渲染/提示风格，需要新增未见组合、独立渲染和自然任务复核，避免合成集自证。
- 候选一次编码后的跨查询复用、查询数增加时成本曲线、近邻索引表现、关键证据保留与无关扰动鲁棒性。不能只用一般对比损失就声称已经迁移决策敏感性。
- 训练数据逐来源许可证核对、可发布重建清单、非敏感元数据、模型卡；不得把 the_cauldron 的集合许可误当成全部原图的再分发授权。

## 主要原始资料

- VLM2Vec-V2: https://arxiv.org/abs/2507.04590
- Qwen3-VL Embedding: https://arxiv.org/abs/2601.04720
- MMEB-V3: https://arxiv.org/abs/2604.23321
- WeMM: https://arxiv.org/abs/2608.24053 ; https://github.com/Tencent/WeMM-Embedding
- Ovis: https://arxiv.org/abs/2609.25165 ; https://github.com/ATH-MaaS/Ovis-Omni-Embedding
- FLAIR: https://arxiv.org/abs/2412.03561 ; https://github.com/ExplainableML/flair
- 官方模型榜: https://huggingface.co/spaces/TIGER-Lab/MMEB-Leaderboard

检出的代码：VLM2Vec `d00b5dcdeaa099f9ddb221ea081128d609c22acb`；WeMM `e8175bbff1071b580b3a6df8ec2c106e8383a361`；Ovis `cf8859fccd6fba6a5eaa67499f713a6a334b4630`。Qwen 复现必须注意末层归一化后池化、system 指令、候选端不加任务指令以及官方分辨率/空白字符；VLM2Vec 仓库的复现审计记录过 7–9 点落差。
