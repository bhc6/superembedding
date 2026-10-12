# Qwen3.5-0.8B 原生问答对照

使用本地固定 revision 的 Qwen3.5-0.8B 原始权重，不加载 OmniJev LoRA、decision head 或特殊 option token。统一 VSR110 内部验证样本、原问句、图像上限 65536 pixels，CPU FP32、单线程、低优先级。样本、图像、代码、权重和模板均封存哈希。不是官方 VSR test，预训练污染未知。

主指标是原生生成：官方 chat template，enable_thinking=False，greedy、1 beam、最多 16 新 token。仅完整生成且全文为 Yes/No（忽略大小写和首尾空白，允许末尾句号或感叹号）解析为有效答案。其余输出或达到长度上限未结束均算错，不删题、不在本集调提示。

补充指标是第一个输出位置的原始 logit(Yes)-logit(No)，确认两者均为单 token，固定零阈值；用于排序 AUC/分布，与无约束生成答题成绩分别报告，不称为校准后的概率。所有题评测 full 和 text-only，后者真正移除图像。保存原始生成文字、token、分数和逐题耗时，逐输入断点续跑。

只修改本独立实验目录和输出目录 `D:/DEMO/superembedding-storage/superembedding/v5/vsr_qwen35_native_v1`，不修改或抢占主训练。结果见 `SUMMARY.zh.md`、`result.json`，进度见 `status.json`。自己的 STOP 文件可在下个输入前停止。

```powershell
.venv/Scripts/python.exe -B experiments/vsr_qwen_native/test_run.py
.venv/Scripts/python.exe -B -u experiments/vsr_qwen_native/run.py --max-new 2
.venv/Scripts/python.exe -B -u experiments/vsr_qwen_native/run.py
```

两项计分测试及首题 full/text-only 冒烟通过之后启动全量。已有预测直接复用；不将冒烟准确率称为整体结果。
