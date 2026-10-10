# RTX 5070 Laptop 本地训练实测

日期：2026-10-11（Asia/Shanghai）。这份报告是本机真实运行结果，与之前论文中的作者报告分开。

此为早期600图开发记录，后续正式结果见 [Fashion2400主结论](FORMAL_CONCLUSIONS.zh.md)。下文“当前”均指开发阶段。旧权重下载见 GitHub Release 的 development_models_fashion600.zip，解压后保留 runs/fashion600 目录结构。

## 结论

这台 **RTX 5070 Laptop 8GB** 已跑通 OmniJev-0.8B 的属性特征提取、三种 embedding head 训练、留出集检索评估，以及现有 **LoRA + projection 的真实反向传播、更新、保存与重载**。

用户提出的 `P(opt(u) × tanh(que(zq)))` 已实现。不过本轮小样本结果不支持它优于 `P(u)`：冻结门控的测试 macro mAP 是 **45.13%**，连门控一起训练是 **55.60%**，直接投影 `u` 是 **74.48%**。因此保留所提结构作为可训练研究方案，目前更好的检索 checkpoint 是直接投影对照。

## 环境与模型

| 项目 | 实际配置 |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop GPU，8150.56 MiB |
| 驱动 | 591.74，驱动显示支持 CUDA 13.1 |
| PyTorch | 2.10.0+cu128，CUDA runtime 12.8，包含 sm_120 |
| Transformers / PEFT | 5.5.0 / 0.19.1 |
| Backbone | Qwen/Qwen3.5-0.8B，revision `2fc06364715b967f1860aea9cf38778875588b17` |
| 决策 adapter | tinnel123/OmniJev-0.8B v1.1，revision `3c9aa908f3c9d3f3ccf16bb281ee705d9e98252f` |
| 精度 | Backbone BF16；projection 与归一化 FP32 |
| 图像预算 | `max_pixels=65536`，按图像比例缩放 |
| 选项方向 | 颜色、商品类型、商品目录目标人群，共 3 个 |
| 输出 | `slots[3,256]`，拼接 `embedding[768]` |
| Windows attention | 标准 PyTorch fallback；`MSO_FLA=0` |

CUDA BF16 矩阵乘法及 backward 已通过：[cuda_check.json](../artifacts/results/cuda_check.json)。完整环境版本见 [requirements-local.lock.txt](../requirements-local.lock.txt)。官方 adapter 下载文件已与其 release manifest 的 SHA-256 校验一致。

## 数据与指标

从公开 Fashion Product Images Small 的固定 shard 取 600 张真实商品图，396 train / 132 val / 72 test。按联合属性分层，商品 ID 与完全相同的解码图像均去重。训练和测试共享属性值与组合；未排除近似商品或其他角度，不是官方 benchmark 划分。

所有三种训练 head 都使用相同图片、属性方向、triplet 采样种子、batch 64、300 次更新及 256 维单 slot 输出。每个 triplet 只监督被指定的属性。验证集 macro mAP 选择 checkpoint，之后对测试集评估一次。三种配置的测试结果是在开发过程中依次查看的，因此该小测试集今后应视为开发数据，正式结论需新的 holdout 和多随机种子。

这里的 R@K 是**按单属性 slot 检索同属性标签图片**，每张查询在同一测试图集检索，排除自身；mAP 在其余 71 张候选上计算。三个属性均有 72 个有效查询，macro 是三个属性指标的平均。它不是全局拼接向量的综合语义检索，也不是“满足全部属性”的指标。

## 实际检索结果

| 模型 | 可训练参数 | 选中 step | 验证 macro mAP | 测试 R@1 | 测试 R@5 | 测试 macro mAP |
|---|---:|---:|---:|---:|---:|---:|
| 原 OmniJev `normalize(u)` | 0 | — | — | 62.50% | 90.28% | 49.52% |
| 冻结 opt/que，训练门控后 projection | 264,448 | 275 | 43.62% | 45.37% | 80.56% | 45.13% |
| 训练 opt/que + 门控后 projection | 4,462,848 | 300 | 57.40% | 57.41% | 85.65% | 55.60% |
| 直接 `P(u)`，训练 projection | 264,448 | 300 | 72.89% | 78.70% | 94.44% | 74.48% |

原始 `u` 每 slot 是 1024 维，其他三行每 slot 是 256 维，因此原始行不是等维度对照。冻结门控与直接投影使用相同学习率 1e-3；训练门控的学习率为 1e-4，并且参数量不同，不能把其结果当作只改变一个因素的严格架构消融。

每属性测试 mAP：

| 方法 | 颜色 | 商品类型 | 目录人群 |
|---|---:|---:|---:|
| 原始 `u` | 25.71% | 62.25% | 60.61% |
| 冻结门控 + projection | 23.60% | 58.01% | 53.79% |
| 门控 + projection 一起训练 | 29.24% | 53.64% | 83.91% |
| 直接 `P(u)` | 56.32% | 75.68% | 91.45% |

这些结果说明原决策门控不能默认作为现成的属性度量空间。它在当前设置下可能压缩了检索需要的差异，也可能需要不同训练预算或目标；当前实验不能区分这些原因。门控方案验证曲线仍在上升，300 步不是其充分收敛的证据。没有比较原始 Qwen 初始化，因此也不能据此证明 OmniJev 决策训练带来了增益。

原始记录：[gated](../artifacts/development/gated_metrics.json)、[gated_tuned](../artifacts/development/gated_tuned_metrics.json)、[raw](../artifacts/development/raw_metrics.json)。

## 显存、速度与真实 LoRA 训练

| 阶段 | 本机测量 |
|---|---|
| 特征提取峰值 allocated / reserved | 1871.61 / 1888 MiB，约 1.83 / 1.84 GiB |
| 全量提取 | 先 smoke 3 张，再提取剩余 597 张；后者含加载 225.72 秒，平均 0.353 秒/图 |
| 缓存后 head 训练 | 每种 300 更新约 1.5–2.3 秒（训练/评估循环；不含进程启动、加载、特征提取） |
| 冻结门控 / 直接投影 / 解冻门控 head 训练 allocated 峰值 | 47.81 / 29.16 / 105.87 MiB；不含已卸载 backbone |
| LoRA + projection，3 次真实更新 | 20,711,680 个可训练参数，训练及保存阶段约 13.74 秒 |
| LoRA 峰值 allocated / reserved | **6614.95 / 6626 MiB，约 6.46 / 6.47 GiB** |

显存是 PyTorch allocator 测量，CUDA context、显示占用等会额外占用。训练期间 `nvidia-smi` 一次观测约 6800 MiB，总显存 8151 MiB。这个实测仅覆盖 **0.8B、当前图像预算、3 个属性方向、每步一个 triplet、FP32 AdamW 状态、BF16 backbone**，不能外推到 2B/4B、更高分辨率或更多方向。

LoRA smoke 从已训练的冻结门控 projection 开始，仅启用已有 LoRA 参数及 projection，dense backbone 冻结。学习率 1e-5、triplet margin 0.5、累计 1 个 triplet/更新。三步 loss 为 0.4986 / 0.4745 / 0.5626，来自不同 triplet，不将其作为收敛曲线。探测的 LoRA 参数有非零梯度，最大权重变化 **3.005e-5**，证明实际更新而非仅执行前向。

**这三步验证训练可行性，不是完成了高质量 LoRA 微调。** LoRA 版本尚未跑留出集检索评估，不能套用上面的 head-only 效果。记录见 [training.json](../artifacts/development/lora_smoke_training.json)。

## 可用产物与验证

- 用户提出的门控结构，训练过的 head：[gated/best_head.pt](../runs/fashion600/gated/best_head.pt)。
- 解冻门控训练后的 head：[gated_tuned/best_head.pt](../runs/fashion600/gated_tuned/best_head.pt)。
- 当前开发集效果最好的直接投影 head：[raw/best_head.pt](../runs/fashion600/raw/best_head.pt)。
- LoRA smoke adapter：[adapter](../runs/fashion600/lora_smoke/adapter)，配套 [embedding_head.pt](../runs/fashion600/lora_smoke/embedding_head.pt)。
- 训练后 adapter + head 重新加载，输出与保存前参考 **最大差值 0**；原 adapter + raw head 重载与缓存路径最大差值约 **1.06e-7**。均输出 `[3,256]` 与单位范数的 `[768]`。
- [reload_validation.json](../artifacts/development/reload_validation.json) 保存上述比较。
- 单元测试 15 项通过，包括门控复制、梯度、slot 独立性、固定方向校验、真实 branch 函数接线、LoRA 梯度隔离、检索排除自身及权重往返。toy 测试与真实 GPU smoke 分开记录。

执行步骤与更长训练命令见 [本地训练说明](../reports/LOCAL_TRAINING.zh.md)。继续扩大 LoRA 实验时，应重提特征并重新评估；原 decision 输出和校准不继承为有效结果。
