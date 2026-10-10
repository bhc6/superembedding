# 本地训练：OmniJev-0.8B 门控属性 embedding

目标设备：RTX 5070 Laptop，8GB 显存，Windows。项目根目录 `D:\DEMO\superembedding`。

## 已实现的结构

```text
OmniJev-0.8B + 固定编码指令 + K 个属性方向
  → 独立选项分支得到 u_i[1024]、z_q[1024]
  → 原 checkpoint 的 opt(u_i) × tanh(que(z_q))
  → LayerNorm(1024) → Linear(1024,256)
  → 每个 slot L2 normalize
  → 拼接 + 整体 L2 normalize
```

主实现：[heads.py](../superembedding/heads.py)。输入可为 `[K,D]` 或 `[B,K,D]`。`mode="gated"` 使用原决策门控；`mode="raw"` 实现相同输出维度的 `P(u_i)` 对照。新增 projection 默认有 264,448 个可训练参数。

集成：[omnijev.py](../superembedding/omnijev.py)。调用真实 `mso.branch.branch_questions`，设 `lm_head=None`，不引入候选组内归一化 LM 特征。`forward` 保留新 head 的梯度；`extract_features` 冻结 backbone；`encode` 是无梯度推理。

**缓存训练范围是 projection，以及可选的原 opt/que 门控层副本。** 此路径冻结 backbone/LoRA。另有 `train_lora.py` 从真实图片联合训练现有 LoRA 和 projection，dense backbone 保持冻结。首轮先用真实模型提取缓存，建立检索对照，再用少量 LoRA 步骤实测显存与梯度。

## 环境

系统原有 PyTorch 是 CPU 版，保留原环境。新环境为项目内 `.venv`，采用官方 CUDA 12.8 wheel；驱动显示的 CUDA 13.1 是驱动支持上限，不要求 PyTorch wheel 版本相同。

```powershell
Set-Location D:\DEMO\superembedding
uv venv --python C:/Python313/python.exe .venv
uv pip install --python .venv/Scripts/python.exe torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/Scripts/python.exe -r requirements-local.txt
.venv/Scripts/python.exe scripts/check_cuda.py
.venv/Scripts/python.exe scripts/download_models.py
```

Windows 本轮设置 `MSO_FLA=0`，使用标准 PyTorch fallback，避免依赖可选 Triton/FLA。脚本已设默认值。

Transformers 固定为 **5.5.0**：实际发现 5.3.0 的 `Qwen3_5DynamicCache` 没有 OmniJev 需要的 `.layers` 接口。5.5.0 使用统一缓存接口，已通过真实模型前向。不要将依赖改回 5.3.0。

## 数据与训练命令

公开数据：[ashraq/fashion-product-images-small](https://huggingface.co/datasets/ashraq/fashion-product-images-small)。固定 revision `3859c76db2f6f3d3b9a3863345e3ccdbff75879d`，只读取第一个 parquet shard。选择六类商品、八种颜色及 Men/Women 两个商品目录标签，采样 600 张。

实际划分：396 train / 132 val / 72 test。按颜色、商品类型、目录人群联合分层；排除相同商品 ID 和像素完全相同的图片。不保证排除不同角度或近似商品；属于小规模开发实验，不是官方 benchmark。

三个方向为颜色、商品类型、商品目录目标人群；后者是商品标签。标签不会作为输入文本提供给模型，编码只接收图片与固定方向指令。

```powershell
.venv/Scripts/python.exe scripts/prepare_fashion.py --count 600

# 先跑 3 张确认实际 CUDA 前向和显存
.venv/Scripts/python.exe scripts/extract_features.py --limit 3

# 可恢复：已缓存的图片不重复推理
.venv/Scripts/python.exe scripts/extract_features.py

# 主方案：训练门控后的 projection，原 opt/que 冻结
.venv/Scripts/python.exe scripts/train_cached.py --mode gated --output runs/fashion600/gated

# 同输出维度、同训练数据和步骤的直接投影对照
.venv/Scripts/python.exe scripts/train_cached.py --mode raw --output runs/fashion600/raw

# 后续可独立训练 gate + projection；不要覆盖已有实验
.venv/Scripts/python.exe scripts/train_cached.py --mode gated --train-gate --lr 0.0001 --output runs/fashion600/gated_tuned

# 真实图片 LoRA + projection 反向与显存测试；每步一个 triplet
.venv/Scripts/python.exe scripts/train_lora.py --steps 3 --output runs/fashion600/lora_smoke
```

训练目标是被标注属性上的 cosine triplet margin loss：正例共享该属性值，负例属性值不同。其他属性可以相同或不同。采样仅使用训练集；每 25 步按验证集 macro mAP 选 checkpoint（包含初始模型）；选好后评估一次测试集。

这验证属性检索，不验证“决策初始化优于原始 Qwen”的因果主张。后者还需要原始 Qwen、相同监督与容量的初始化对照。

LoRA 脚本的 `--accumulation` 是每次更新累计的 triplet 数，默认 1；`--steps` 是 optimizer 更新次数。三步实验只验证真实 backward、参数变化和显存，不验证检索效果。它会另存新 adapter 和 embedding head。LoRA 改变后原来的缓存失效，需要重新提取；继承的决策 head 与原概率校准也不再视为已验证。

## 文件输出与使用

- `data/fashion600/records.jsonl`：图片、标签与固定划分。
- `data/fashion600/metadata.json`：数据 revision、方向和协议。
- `runs/fashion600/features/*.pt`：真实 OmniJev 的 `u`、`zq`。
- `runs/fashion600/features/extraction.json`：真实设备、提取耗时和峰值显存。
- `runs/fashion600/{gated,raw}/best_head.pt`：projection、门控权重、固定指令与方向顺序；不包含 backbone。
- `runs/fashion600/{gated,raw}/metrics.json`：训练配置、验证历史、测试集 R@1/R@5/mAP。

权重可直接用于新图片：

```powershell
.venv/Scripts/python.exe scripts/encode_image.py --image D:/path/to/product.jpg --head runs/fashion600/gated/best_head.pt --output runs/product.pt
```

输出 `slots[3,256]` 和 `embedding[768]`。针对某个属性检索时比较相应 slot；全局拼接比较相当于等权组合三个属性。颜色实验的 slot 指标不等同于全局拼接向量的指标。

只修改属性顺序、固定指令或模型权重也可能改变向量语义；不能把不同配置的缓存混入同一索引。head checkpoint 加载会核对指令与顺序；同一路径下替换 backbone 权重仍需人工建立新实验目录。

## 验证边界

单元测试使用真实 head 类和真实 branch 函数，配合可缓存的 toy backbone 检查接线，不替代真实模型验证。首轮真实运行的设备记录与效果以 `runs/` 中实际输出为准；未生成对应文件前，不视为已经完成该阶段。
