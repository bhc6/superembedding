# 执行记录

测试前写入 EXPERIMENT_PROTOCOL_V2.zh.md；不根据测试表现新增超参数搜索。

- 新样本 2400 图来自两个固定版本 shard；排除开发600图、解码重复和 dHash<=4，得到 40 个联合属性组合。最小新样本两两 dHash 距离实测为5。
- FLAIR 官方 checkpoint 使用旧 NumPy scalar 元数据：增加限定 safe_globals 兼容，保持 weights_only 加载；未修改官方网络权重。
- OmniJev 官方 head.pt 使用 mlu:0 设备标记：显式 map_location=cpu 后加载，解决 NVIDIA 环境兼容问题。
- 54组正式缓存训练：OmniJev7架构×3seed×2lr=42，FLAIR raw 6，禁用LoRA基础网络 raw 6。FLAIR正式训练早于主矩阵完成，使用完全相同训练器。
- CPU/CUDA 状态恢复单测通过；真实缓存训练连续4步 vs 2步中断恢复至4步的所有参数最大差0。状态包括 optimizer、RNG、已完成step；训练数据和特征内容纳入指纹。
- 为减少等待，OmniJev和禁用LoRA特征提取有部分同时运行；记录的提取耗时受到GPU并发影响，不用于公平推理速度排名。
- 所有OmniJev head和FLAIR head训练完成后，根据验证集选择学习率再打开这两组测试指标。此后只继续原先写定的基础网络对照和100步LoRA随访，不改架构、学习率或预算。LoRA仅根据raw head验证集选择seed42的起点。
- LoRA随访固定 seed42、100updates、每25步全验证集、lr1e-5、margin.2；第10步主动保存退出后重启。它不是与54组缓存训练等预算的主对照。
