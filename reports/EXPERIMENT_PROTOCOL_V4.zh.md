# 主流 embedding 对照实验 v4

本轮在查看任何新模型测试指标前固定以下协议。v3 的测试结果已经公开，因此本轮不是新的未见测试集；不使用它调参，也不重新选择 v3 模型。

## 模型与输入

| 后端 | 模型 | 输出宽度 | 固定属性条件 |
|---|---|---:|---|
| clip_b32 | OpenAI CLIP ViT-B/32 | 512 | 无 |
| siglip2_b16 | Google SigLIP2 B/16 224 | 768 | 无 |
| dinov2_small | Meta DINOv2 Small | 384 | 无；视觉自监督对照 |
| qwen3_vl | qwen3-vl-embedding API | 2560 | 无 |
| qwen3_vl_instruct | 同上 | 每个属性 2560 | 每个方向独立 instruct 请求 |
| tongyi_plus | tongyi-embedding-vision-plus-2026-03-06 | 1152 | 无 |

本地模型使用官方预处理、float32 推理和固定仓库 revision，见 `superembedding/baselines.py`。API 使用原始图像，Qwen 使用完整 2560 维；Tongyi 使用原生 1152 维、res_level=1。Qwen API 是可变服务别名，不能声称固定开源权重可复现；记录请求日期、参数、request ID、实际 usage 和缓存向量指纹。在线调用成本不等于本地模型编码成本。

属性指令固定为 `Represent the image for retrieval by {direction}. Focus on this attribute and ignore other attributes.`。方向直接来自 v3 元数据，不含标签值，不依赖最终检索查询。预检只使用两张训练图像，确认 instruct 确实改变输出。所有图像按相同固定方向编码一次，可跨查询复用。

## 公平比较

1. 全部 28,263 张图像，沿用 v3 Fashion / CUB / DTD 的 train、val、test 和 25/50/100% 嵌套训练子集。
2. 冻结原始向量独立报告。通用图像 embedding 在每个属性任务使用同一向量，不能宣称具有属性选择能力；指令版本和 OmniJev / FLAIR 使用不同属性 slot。
3. 监督适配使用 v3 相同共享 LN+Linear(256) raw head、triplet loss、1000 步、batch64、lr={1e-4,1e-3}、seed={17,42,91}，每250步验证，以验证属性宏平均 mAP 选 checkpoint 和学习率。每个模型每个数据集每个数据量6次训练，总计324次。通用向量重复成多个 slot 后，经共享 head 仍相同，这个对照衡量通用单向量适配，不等价于多个独立属性 head。
4. 原始维度不同，原生 embedding 主表同时报告维度。监督适配单 slot 均256维；不得把重复slot的 K×256 存储写成新增信息容量。
5. 不改 v3 结果，已有训练对照使用其原有验证选择。指标沿用 observed-set overlap、完整 mAP、R@1、R@5，排除自身、缺失标注和没有正例的查询。DTD 仍只用 split1，不是官方10划分分类榜。
6. API 按8张一批、最多4并发；每批原子缓存、退避重试、有限失败即停。图像以解码 RGB 哈希检查；缓存校验顺序、图像哈希、形状、非有限值与来源指纹。训练保留 best_head 与完整 last/optimizer/RNG 状态。

## 范围

此轮覆盖图文对比学习、sigmoid图文学习、视觉自监督、专用多模态服务、指令条件 embedding，再与已有决策模型/FLAIR比较。不是所有 embedding 产品的穷举。纯文本 embedding 不直接接收图像，必须另设 caption 管线才能有效比较，不能用标签文本代替图像制造结果。Omni-Attribute 缺少已确认可复现权重时只保留文献对照。

官方 API 资料（本轮获取，2026-10-11）：

- https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference
- https://help.aliyun.com/zh/model-studio/embedding

数据许可、研究用途和不重新发布原始图片的约束继续沿用 v3。`.env` 只保存在本地，禁止进入 Git、发布资产和日志。
