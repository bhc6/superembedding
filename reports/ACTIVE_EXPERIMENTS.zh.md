# 并行实验运行状态

快照时间：2026-10-12T14:42:07.102178+00:00。重新运行 scripts/report_active_experiments.py 可刷新。

| 任务 | 实际状态 |
|---|---|
| OmniJev LoRA + embedding 长训 | running；Step 1043/2000；最近 checkpoint 1040 |
| CPU 论文基线队列 | completed；当前 None；WeMM→GME 串行 |
| WeMM-2B / GME-2B | 已缓存 3864 / 3864 个向量；缓存数不等于完成题数 |
| 官方 VSR 2195 题筛查 | screened_review_required；已处理 2195；隔离候选 385 |
| 冻结 backbone 三学习率对照 | waiting_for_continuation；长训完成且显存连续三次足够后接续 |

CPU 基线先跑同协议 VSR110，再跑 760 条历史验证及问答 text-only，最后跑 ScienceQA1000 全模态/text-only。采用 FP32 和 65536 像素预算，属于受限配置直接比较，不是论文 headline 的完整复现。每个向量独立缓存，绑定输入字节、官方权重和实现哈希。

长训完整 optimizer/RNG checkpoint 每 5 步保存。冻结对照也保留 optimizer/RNG 和固定采样序列，最终保存固定 Step2000 的完整 backbone adapter + head。旧封存脚本、旧 STOP 和失败 v1/v2 基线记录保留。

新 VSR 以 41999 张历史图像清单进行 RGB 精确与双感知哈希筛查。隔离候选不等于确认同图，未隔离也不等于已完成场景/裁剪审查；不自动启动正式评测。官方原始 HTTP 图像链接可访问，HTTPS 证书域名不匹配，未关闭 TLS 校验。

本轮检查：22 项相关 pytest 通过；WeMM/GME 各三条真实图像推理通过；冻结对照采样预检通过。当前任务全部使用 D 盘。

## 尚未完成

- VLM2Vec-V2: official 420 DoRA tensors inspected; Transformers 5 module mapping and forward equivalence still need verification.
- Ovis: pinned repository has README only; downloaded checkpoint includes thinker/talker/token2wav. Native embedding preprocessing and memory-compatible loading remain to be verified.
- Fresh official VSR: complete acquisition, input sealing and scene/crop review before inference. Filtered subset is not an official full-test score.
- Fixed-terminal full LoRA+head roundtrip audit and cleared independent evaluation; validation best remains diagnostic.
- Natural/compositional paired evidence, actual branch-count/decision-pretraining controls, training seeds, broader data and method experiments.
- CUB/DTD/Fashion separate family leakage remediation, final data rights/model card/release audit.

项目尚未全部完成，未提交或推送 Git。
