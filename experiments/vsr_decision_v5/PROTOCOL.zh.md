# VSR：原 decision head 与 embedding 的完整对照

固定使用 V5 的 110 条 VSR 内部验证样本（68 Yes、42 No）；这不是官方 VSR test。采样问句、候选、图像哈希与现有验证一致。不用本实验结果调整 prompt、阈值、学习率或选择 checkpoint。

四组主对照：

1. 原 OmniJev backbone/LoRA + 原 decision head，以官方 choice 类型选择 Yes./No.。
2. 原 OmniJev backbone/LoRA 冻结 + 训练到 Step 500 的 projection。沿用已经排队的严格匹配训练；主 LR=1e-5，额外完整报告 1e-4、1e-3。
3. 固定 Step 500 更新后的 LoRA + 同一步 embedding head。
4. 相同 Step 500 LoRA + 未修改的原 decision head。

两组 decision 模型另测官方 noul 接口，以校准后的 P(yes)>=0.5 判为 Yes；作为补充，不按成绩从 choice/noul 选优。保留官方 LM features、温度与原 head 权重。核对加载原 head 的每个 tensor；加载 Step 500 仅替换 LoRA 和 embedding head。开始前用前两条样本核验完整图像的自定义调用与官方 ask 完全一致，并验证交换 choice 顺序不改变判断。

全部模型评测 full 和 text-only。text-only 真正删除图像 token/像素，不用空白图代替。每组输出逐题预测、accuracy、balanced accuracy、两类召回率、混淆矩阵、预测 Yes 比例、full−text-only 差值、模型配对差值和按真实标签分层的 bootstrap 区间。choice 按官方返回的二选一计算，另保存 abstain 概率，不删去低置信度样本。embedding 保持原验证的 110 条重复答案候选和排序，标签相同即为正确。

决策使用原问题文字和官方模板；embedding 使用现有训练前缀和四方向模板。原生方法的整体差异不是只换 head 的单变量实验。原 decision 对比更新 LoRA 后的同一 decision head，则固定了接口和 prompt，检验决策能力保持。

全部采用 CPU FP32、单线程、Windows below-normal priority，CUDA_VISIBLE_DEVICES=-1。CPU 结果可能不同于既有 CUDA BF16 分数。基线优先复用同协议的本地 Qwen3-VL-Embedding-2B 逐题预测，不能混用 Qwen API。另列 Always-Yes/No 与重建随机 head；随机 head 明确不等于训练过的冻结 projection。

只新增本目录和独立输出 `D:/DEMO/superembedding-storage/superembedding/v5/vsr_decision_step500_v1`。每条题/模态保存结果，原模型 hidden 缓存后可直接用于冻结 projection 的 CPU 评测。任务中断后同命令恢复；自己的 STOP 文件只停止本实验。原模型与 Step 500 完成后，pipeline 释放模型，最多等 96 小时，等匹配训练的三份 Step 500 快照出现后自动完成冻结组。无需等待其 2000 步训练和全部外部评测结束。

运行方式：

```powershell
.venv/Scripts/python.exe -B experiments/vsr_decision_v5/test_run.py
.venv/Scripts/python.exe -B experiments/vsr_decision_v5/run.py --prepare
.venv/Scripts/python.exe -B -u experiments/vsr_decision_v5/run.py --arm original --limit 1
.venv/Scripts/python.exe -B -u experiments/vsr_decision_v5/run.py --pipeline
```

汇总见独立目录 `SUMMARY.zh.md` / `summary.json`，执行进度见 `pipeline_status.json` / `worker_status.json`。`--limit` 只做冒烟测试，不会输出完整任务分数。`--report` 仅汇总已完整完成的组，不能把部分样本冒充总体成绩。

本实验是单种子、事后内部诊断。区间未校正多重比较，不覆盖训练种子、验证选择偏差或预训练污染不确定性。不能单凭这 110 条证明通用空间推理优越性。
