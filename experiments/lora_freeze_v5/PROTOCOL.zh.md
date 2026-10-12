# V5 冻结与 LoRA 联合训练的匹配对照

这是独立补充实验，不修改正在运行的训练实现、数据、配置或暂停标记。

主要问题：在同一数据、同一方向和同一优化日程下，开放原有 LoRA 更新，相比只训练 projection 是否改善验证和迁移表现？已有 36 次冻结 head 消融的数据规模、batch、学习率不同，不能回答这个问题。

## 固定条件

- 父运行：`general_stage1_v51`，2000 步、seed 20261011、batch 16。
- 初始模型：已经审计的 CUDA 重建 Step 0，原 OmniJev LoRA + 随机共享 LN/Linear256；不从 Step 250 开始冻结。
- 数据：`data_v51` 的完整同一清单，8 来源、7388 条训练记录、760 条验证记录。
- 每步严格复现父运行的来源、16 个图像记录、各自被采样的监督文字、重采样逻辑；前 500 步相同课程限制，之后相同完整采样池。
- 四个方向、文本 256 tokens、max_pixels 65536、CUDA BF16 backbone、FP32 head/归一化/相似度均相同。
- Multi-positive InfoNCE，温度 .05；AdamW wd .01；warmup 100 + cosine 到 .1 倍；clip 1；禁用 dropout。
- 只冻结原 LoRA，保留其权重，训练共享 projection；缓存原始分支 hidden，逐对象回放 head 梯度，避免重复执行冻结 backbone。缓存不跨不同权重/精度复用。
- 主对照 head 学习率 1e-5，与父运行一致；另预先固定 1e-4、1e-3 两组作为冻结方案的学习率敏感性对照。后两组不能称为仅改变冻结开关的单变量实验。
- 同一采样日程依次更新三个 head，不增加任何一组的数据量或更新步数。

## 保存与审计

`--prepare` 在独立目录封存协议与 2000 步采样表；逐项对照主训练 pair 构造，验证保存的 Step 60 Python sampler RNG 和来源队列完全一致，并核对已记录步的来源。

同精度 Step 0 全验证必须复现既有 CUDA 重建基线，否则停止。每 5 步保存三个 head、optimizer、Torch/CUDA RNG、下一采样位置和验证历史；每 500 步保留 head 权重快照。原始冻结 LoRA 与训练 head 一起导出，可恢复父框架所需的完整参数集合。

Step 0 是重建，不是找回的原始文件。缺少原始 Step 0 哈希，因此初始化一致性有明确的证据边界。

## 比较和解释

每 250 步完整验证，比较同一步号的冻结/LoRA 曲线；主 LR 单独报告。次要调参结果按验证选学习率/步号，平分取较低 LR/较早步。Step 0 也保留，不把未更新起点称为训练收益。

最终报告八来源宏平均及分来源 R@1。独立 pipeline 在训练结束后自动核验导出，并按相同 CUDA 协议评测冻结主 LR、冻结验证调参模型和父运行的最终验证最佳模型：完整内部验证/QA text-only、完整 MMEB ScienceQA 及 text-only、Urban1k common 999 和证据诊断。外部结果不参与选择，不调用覆盖全局报告的脚本。

结果保存在模型哈希命名的既有评测目录及本实验 `final_comparison.json`。内部验证和 ScienceQA 输出配对 bootstrap 差值区间；这是固定模型下的 query 重采样，不覆盖训练种子方差、选择偏差或共享图像/答案相关性。只凭内部验证不能宣布 LoRA 泛化更好。冻结允许选择 Step 0，父运行只在正步数验证点选择，报告保留这一差异。

本次为单个配对种子，与已运行主模型对齐；不是多种子确认性结论。不匹配 FLOPs/墙钟时间，另记录运行成本。原 decision head 能力保持需要独立评测，本对照不替代它。

## 执行

在项目根目录运行：

```powershell
.venv/Scripts/python.exe -B experiments/lora_freeze_v5/runner.py --prepare
.venv/Scripts/python.exe -B experiments/lora_freeze_v5/test_runner.py
.venv/Scripts/python.exe -B experiments/lora_freeze_v5/pipeline.py --seal
.venv/Scripts/python.exe -B -u experiments/lora_freeze_v5/pipeline.py
```

输出：`D:/DEMO/superembedding-storage/superembedding/v5/lora_freeze_matched_v1`。

排队进程只读父状态，只有主 supervisor 完成训练及最终评测（`stage_complete`），且显存至少 4500 MiB 连续三次检查通过才启动 GPU 训练。不会暂停、恢复或抢占现有任务。最多等待 96 小时，超时写状态退出；相同命令可续等。自己的 `STOP` 标记只影响本实验。任何失败保留特征和 last，修复后显式重启，不进行无界重试。

冻结/LoRA 同预算比较仍属于事后补充分析；不能冒充在主模型训练前预注册的实验。
