# 主流 embedding 的结构差别

这些模型都输出向量，但向量形成的位置、是否可接受属性条件、训练目标不同。

| 方法 | 图像路径 | 输出来源 | 本轮属性条件 |
|---|---|---|---|
| CLIP ViT-B/32 | 224px，32px patch，12层 ViT，hidden768、12 attention heads | pooled视觉状态→512维视觉投影；原训练对齐独立文本塔 | 无；同一全局向量复用 |
| SigLIP2 B/16 | 224px，16px patch，12层 ViT，hidden768、12 heads | 官方 attention pooling，768维；图文 sigmoid 目标及 SigLIP2 训练改进 | 无；API `get_image_features` 只编码图像 |
| DINOv2 Small | 短边256后中心裁剪224px，14px patch，12层 ViT，hidden384、6 heads | 最终 CLS token，384维；视觉自监督，没有文本塔 | 无；检查视觉表征本身能贡献多少 |
| FLAIR | 既有官方 ViT-B/16，图像局部 token 与文本方向 | direction text 作 query，对视觉 token 做 attention pooling，512维 | 是，既有固定方向文本 |
| OmniJev raw | Qwen3.5-0.8B + 已训练 OmniJev adapter；保留原选项分支 attention | 各选项 hidden（1024维）→共享 LN+Linear256 | 是，每个选项为一个固定属性方向 |
| Qwen3-VL embedding API | 提供者托管的专用多模态 embedding 模型 | 原生2560维 | 通用版不传 instruct；属性版每个方向独立传 instruct |
| Tongyi Vision Plus 2026-03-06 | 官方说明基于 Qwen3，托管服务，res_level=1 | 原生1152维 | 本轮不用属性指令 |

本地尺寸来自固定模型配置及官方 image processor；DINOv2 config 中的预训练 image_size 不能直接当作本轮输入尺寸，实际以保存的 processor resize/crop 为准。完整配置记录在 `sources/baseline_architectures_v4.json`。

按固定配置构建模型后计数：CLIP checkpoint 总参数151,277,313，视觉塔87,456,000（另有视觉输出投影）；SigLIP2 总参数375,187,970，视觉塔92,884,224；DINOv2 总参数22,056,576。CLIP/SigLIP2 的通用图像提取不调用文本塔，所以总参数量不等于本轮图像编码实际参与的参数量。

对于 API，文档没有充分披露可核实的层数、参数量、实际 pooling、精确权重 revision，因此不能由2560维反推一个具体开源 Qwen checkpoint，也不能称它和本地0.8B OmniJev参数量相同。Qwen3-VL embedding 服务也不等同于把 Qwen chat API 的生成文字当 embedding。

## 原始向量与监督适配

通用模型首先产生 `g(x) ∈ R^D`；不同属性任务都用同一向量。为接入相同训练器，复制为 `u_k=g(x)`，随后使用与 OmniJev raw 相同的共享投影：

`e_k = normalize(Linear(LayerNorm(u_k)))`。

因此通用模型的 `e_1=e_2=...=e_K`。复制与拼接不会创造属性分离能力；它只是通用单向量的同预算监督适配对照。不同属性独立的 projection heads 是另一种设计，需要额外参数和新的实验，不能把本轮结果解释成该设计的上限。

OmniJev、FLAIR 和 Qwen instruct 在投影之前已按固定方向形成不同 `u_k`。本轮可以检验这些方向条件是否改善检索，但不能仅凭这个对照证明 attention、decision sensitivity 或某个单独预训练目标的因果收益。

## output projection 的关系

本轮新基线使用相同 raw head，集中回答不同 backbone 的表现。全局向量由归一化 slot 拼接再归一化。v3 已有的 concat、等宽投影、压缩投影实验保留；不要把 backbone 对比和输出投影选择混在一起。是否需要投影，仍取决于存储预算和验证集表现；任意单属性检索仍须保留独立 slots。

## 来源

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [固定 CLIP checkpoint](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268)
- [固定 SigLIP2 checkpoint](https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2)
- [固定 DINOv2 checkpoint](https://huggingface.co/facebook/dinov2-small/tree/ed25f3a31f01632728cabb09d1542f84ab7b0056)
- [DashScope 多模态 embedding API](https://help.aliyun.com/zh/model-studio/multimodal-embedding-api-reference)
- OmniJev 与 FLAIR 的源代码、版本和对应位置见 `ARCHITECTURE_COMPARISON.zh.md`。
