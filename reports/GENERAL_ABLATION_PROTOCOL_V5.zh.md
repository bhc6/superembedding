# 通用冻结特征架构消融：预先固定协议

2026-10-11，在任何本组训练/验证分数产生前固定。与正在运行的 LoRA 长训分开，原预训练 OmniJev 为共同冻结特征提取器，不加载长训中途 checkpoint。本组不覆盖 LoRA/冻结或真正单分支/多分支消融。

数据使用去污染后的 `data_v51` 清单（SHA256 `470b7a9b2ca8017fdaad15c404f54b9179a0d556434807d252e89102e9a92f44`）。每个训练来源按 SHA256(`general-ablation-v1:` + id) 排序取前 128 张，共 1024 张；全部 760 张验证及 163 张来源留出。每图一条固定监督，选择规则与长训验证一致，`random.Random(record id)`。来源留出只在全部学习率选择封存后评分；它们是原始 train split 的 AI2D/ScienceQA，不叫官方 test。

CPU float32 逐对象提取四方向 raw hidden，每侧 4×1024；图像预算 65536 像素、文本 256 tokens。只保留 hidden 分片，预计不足 100 MB；不复制图像或 backbone。CUDA 在子进程中禁用，最多两个 CPU 线程。分片指纹、模型和数据身份检查支持断点继续。

| 架构 | 输出维度 | 用途 |
|---|---:|---|
| shared_concat | 1024 | 共享 LN/Linear(256)，分方向 L2 后拼接 L2 |
| output_equal | 1024 | 相同 slots 后加 1024→1024 无偏置投影，单位阵初始化 |
| output_small | 256 | 相同 slots 后加 1024→256 投影 |
| independent_equal | 1024 | 每方向独立 LN/Linear(256)，从相同共享初始化复制 |
| independent_matched | 256 | 每方向独立 LN/Linear(64)，接近共享头参数预算，维度也改变 |
| mean_pool | 1024 | 四个 raw hidden 取均值后 LN/Linear(1024) |

参数量逐项报告，不能把加投影的对照称为完全参数匹配。mean_pool 仍计算四个分支，不能冒充真正单分支 backbone。方向数 K 固定为 4，本组不测 K 最优值。等维投影初始映射与直接拼接一致，训练数据和采样顺序按种子配对。

代码实测参数数：shared_concat 264448、output_equal 1313024、output_small 526592、independent_equal 1057792、independent_matched 270592、mean_pool 1051648。独立近预算版本比共享头多约 2.3% 参数，不能标成严格相等。投影初始等价、独立头初始等价及分支均值置换不变性已单测；训练器 optimizer/RNG/采样与验证历史的中断重放也已独立通过。

6 架构 × 3 seeds（17/42/91）× 2 学习率（1e-4/1e-3），共 36 次。每次 1000 updates，同源 batch=32，来源循环均衡随机打乱，multi-positive InfoNCE τ=.05，AdamW wd=.01、clip=1。学习率恒定，不沿用 LoRA 的小学习率。训练来源内同图/同描述/同答案采用共同多正例规则。

每 100 步及 step 0 完整验证，指标为八来源内 R@1 宏平均。每架构按三个种子平均最佳验证分数选一个学习率，平分取较低学习率；checkpoint 平分保留较早步。保留 step 0 入选，不能写成训练收益。所有 last/best 含 optimizer/RNG/来源队列，每次 100 步保存，不删除差方案。

全部 36 次结束后封存选择，统一评测 18 个选中模型及未训练 raw concat，报告留出来源 R@1/MRR、参数、维度、正例数量、种子标准差和验证重放误差。本组不能证明官方 ScienceQA/完整 MMEB 的架构排序，也不能按留出分数重选长训架构。

入口：`scripts/run_general_ablation.py`；每阶段日志和状态记录在独立目录。任务完成不会自动提交 Git。
