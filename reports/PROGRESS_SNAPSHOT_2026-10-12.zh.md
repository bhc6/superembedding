# 2026-10-12 中途进展提交

本次按用户最新要求先同步一次 GitHub，更新此前“全部完成后一次提交”的安排。提交范围为项目代码、实验协议、结果与审计记录；不是最终模型 Release，不表示泄漏审查通过或论文实验全部完成。checkpoint、optimizer/RNG 状态、缓存、上游模型、原始图片与 `.env` 继续保留在本地忽略目录及 D 盘存储。

## 已完成与新结果

- WeMM-Embedding-2B 与 GME-2B：VSR110、八来源验证760、问答 text-only、ScienceQA1000 全模态/text-only 全部完成。每模型 3864 个不同输入向量；全部八个结果文件的预测和指标已从缓存逐条重算一致。该核验不是对全部样本重新做模型推理。
- 八来源 macro R@1：Step500 47.39%，Step750 53.06%，Step1000 53.77%（VSR84/110，ChartQA36.92%）；WeMM 75.83%，GME 45.57%，Qwen3-VL-Embedding-2B 67.85%。当前模型仍落后于较强的通用 embedding 基线。
- 36 组冻结特征架构训练已完成；shared concat 来源留出 30.42±2.42%，等维 output projection 30.12±3.53%，尚无必要增加 output projection 的证据。
- 官方 VSR 2195 题全部完成初步图像筛查：RGB 精确重叠 0 条、近重复候选隔离 385 条，1810 条仍需场景/裁剪复核。过滤后的子集不能当作官方全 test 成绩。
- 本次提交前 CPU 测试：75 passed。检查待提交文件未发现本地配置中的密钥或常见 token 格式；无新增权重、缓存或原图文件。

## 正在运行及待完成

GPU 主训练由 Step556 的完整恢复状态延续到预定 Step2000，每 5 步保存 optimizer/RNG checkpoint。实际步号见 `ACTIVE_EXPERIMENTS.zh.md` 的时间戳快照。三学习率冻结 backbone 对照已通过历史采样序列预检，等待长训结束且 GPU 资源满足条件后接续。

VLM2Vec/Ovis 适配与推理、最终完整 LoRA+head 导出重载核验、经审查的独立测试、自然/组合成对证据、分支数量与决策预训练消融、多种子长训、数据泄漏整改和数据权利/模型卡仍未完成。正式候选固定 Step2000；旧验证 best 仅为诊断，不具备正式选模资格。

## 文件与复现边界

封存的训练/推理实现保持原样；`prepare_official_vsr_holdout_v2.py` 显式保留原有 CRLF 字节，防止 Git 换行转换破坏已记录的实现 SHA256。数据重建清单只含来源、行号和哈希；本机路径出现在诊断及运行溯源信息中，不代表上传了相应本地文件。

旧报告保留对应时间点的状态，部分仍提到 Step556 暂停或旧 validation-best 流程；当前状态以 README 顶部、`STEP556_CONTINUATION.zh.md` 和本次快照为准。历史 CUB/DTD 及 DocVQA 的问题仍公开保留，现有数值不支持“零泄漏”、SOTA 或机制已被证明的结论。
