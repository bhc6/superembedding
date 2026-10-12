# 通用冻结特征的架构消融

Frozen pretrained OmniJev, 128 train images/source, one fixed supervision/image. OOD from upstream train sources, not official test.

36 次训练；每架构按三个种子的平均验证分数选择一个学习率，封存后评测来源留出。标准差是训练种子标准差。

| 架构 | 参数 | 输出维度 | 验证 R@1 % | 来源留出 R@1 % ± SD | 选中步 |
|---|---:|---:|---:|---:|---|
| shared_concat | 264448 | 1024 | 26.95 | 30.42 ± 2.42 | [1000, 700, 700] |
| output_equal | 1313024 | 1024 | 27.24 | 30.12 ± 3.53 | [900, 900, 700] |
| output_small | 526592 | 256 | 24.48 | 28.30 ± 1.16 | [900, 600, 700] |
| independent_equal | 1057792 | 1024 | 28.09 | 30.24 ± 0.83 | [800, 1000, 1000] |
| independent_matched | 270592 | 256 | 26.54 | 28.23 ± 1.05 | [900, 900, 1000] |
| mean_pool | 1051648 | 1024 | 27.58 | 25.99 ± 1.46 | [600, 700, 700] |

未训练 raw concat（4096 维）来源留出 R@1：25.34%。

各来源候选及正例数量见 JSON；AI2D/ScienceQA 来源留出不是官方测试。step 0 入选不是训练收益。

等维 output projection 以单位阵初始化，初始相似度等价，但增加参数；独立头同维版约四倍参数，另列近参数预算版。均值池化仍计算四个方向，不等于单分支 backbone。

本实验冻结预训练 OmniJev，不检验长训 LoRA 后的架构排序；数据规模和固定单监督限制须保留，不据此宣称通用最优。
