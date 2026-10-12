# V5 进展与实测结果（未完成全部研究）

> 2026-10-11 泄露审查更新：V5 已在 Step 556 保存断点并暂停；同数据 frozen/LoRA 队列已停止。CUB/DTD 同图/同场景与 DocVQA 实体交叉尚未整改完成，历史分数仅作探索性结果。以 [数据泄露复核](DATA_LEAKAGE_REVIEW.zh.md) 为准；不得直接恢复旧主结果流程。

生成于 2026-10-11T14:24:46.420066+00:00。

所有分数均为百分数。后台运行不等于训练完成，启动验证也不等于检索能力提高。本阶段未提交/发布 GitHub。

## 完整官方 MMEB ScienceQA 子任务

1000 查询，每个查询原始 736 候选，Hit@1。官方 TIGER-Lab/MMEB-eval 固定 revision；这不是完整 MMEB 排名。

| 本机实测 | Hit@1 | 95% Wilson 区间 |
|---|---:|---:|
| qwen3_vl | 73.10 | 70.27–75.76 |
| clip_b32 | 6.20 | 4.87–7.87 |
| siglip2_b16 | 3.90 | 2.87–5.29 |
| omnijev_raw | 0.20 | 0.05–0.73 |

CLIP/SigLIP 为原生图文向量等权融合的启发式基线；OmniJev raw 为固定方向原始 hidden 归一化拼接，没有训练 embedding projection。Qwen API 原生支持混合图文；其 alias 不能冒充公开 2B 或 8B checkpoint。

OmniJev raw 的低分说明直接拿决策 hidden 做通用跨模态余弦检索没有自动得到对齐空间；不能因此否定训练后迁移假设，也不能隐去这个负结果。

## ScienceQA 模态干预

同一组 1000 查询、同一冻结候选向量库；不重训。错配图片为固定循环位移，8 条同期完整输入 API 漂移探针通过。

| 输入 | Hit@1 | 完整输入优势（百分点） | 成对 bootstrap 95% 区间 |
|---|---:|---:|---:|
| full | 73.1 | 0.0 | 0.0–0.0 |
| text_only | 32.6 | 40.5 | 37.3–43.8 |
| image_only | 29.3 | 43.8 | 40.4–47.0 |
| shifted_image | 33.3 | 39.8 | 36.5–43.0 |

该差异支持正确图像对分数有贡献，不证明模型推理能力或无预训练污染。区间按查询重采样，未校正重复答案/图像间依赖。

## FLAIR 论文官方权重推理复现

完整 Urban1k 1000 图像/1000 描述，官方注释、tokenizer、图像 transform 和评分函数。float32 推理，官方入口默认 AMP；不是从头重训 FLAIR。

| FLAIR | 文搜图 R@1 | 图搜文 R@1 |
|---|---:|---:|
| native_text_conditioned | 87.70 | 83.50 |
| global | 86.10 | 82.70 |

首样本 native 分数与官方 `compute_similarity_scores_attn_pool` 最大绝对差 1.72e-05。原生 FLAIR 需要每个查询对图像 token 做条件池化，其在线成本与固定缓存向量不同。

## API 与本地模型共同协议：Urban1k 999 非空子集

官方注释的 index 548 / 592.jpg 是空描述，API 拒绝空文本。以下对所有模型同时排除此图像及描述，不能与上面的 1000 条协议混写。

| 模型 / 模式 | 文搜图 R@1 | 图搜文 R@1 |
|---|---:|---:|
| flair / native_text_conditioned | 87.79 | 83.88 |
| flair / global | 86.19 | 82.78 |
| clip_b32 / native_global | 46.85 | 61.06 |
| siglip2_b16 / native_global | 33.73 | 38.04 |
| qwen3_vl / native | 99.30 | 98.80 |

强 API 基线接近饱和，因此 Urban1k 只保留为兼容性诊断，不承担本项目主创新结论。预训练污染未知。

## 可发布的成对证据诊断 v1

64 个原始合成场景、256 个属性干预案例、384 张原创 PNG；CC0 图像/标注，生成器随仓库开源。仅作诊断，不进入训练。每图只编码一次，跨查询复用。

| Qwen API | 原图优于证据改变图 | 原图与干扰图都优于证据改变图 |
|---|---:|---:|
| color | 95.31 | 90.62 |
| number | 100.00 | 98.44 |
| position | 40.62 | 26.56 |
| shape | 79.69 | 70.31 |

位置干预同时交换 A/B 的左右位置，应与颜色、形状、数字分开分析；当前只用一种绘图风格和一个查询模板。每四条共享一个场景，不能视为独立样本。低空间分数是诊断线索，不是新颖性或真实应用优势证明。

同协议 CLIP / SigLIP 及场景成对 bootstrap 结果见 `EVIDENCE_V5.zh.md`。若报告存在，该表包括不训练的本地对照，不能只按 Urban1k 分数认定 Qwen 在所有证据任务上最强。

## 新一代模型官方报告（非本机复现）

| 上游提交 | 190任务缺失计零均值 | 数值覆盖任务 |
|---|---:|---:|
| WeMM-Embedding-9B | 59.33 | 179/190 |
| Ovis-Omni-Embedding-3B | 58.45 | 190/190 |
| Wianmu-3B-v0 | 58.16 | 190/190 |
| WeMM-Embedding-4B | 57.97 | 179/190 |
| Ovis-Omni-Embedding-v0.5-3B | 57.42 | 189/190 |
| WeMM-Embedding-2B | 55.77 | 179/190 |

固定榜单 revision `578a5b28e13abc7d6e9fbce2ea5be8bee75ceed6`。这是上游提交的审计汇总，不是本机 SOTA 复现；舍入顺序与官方显示可能差 0.01。WeMM 已用 Qwen3.5，因此这一 backbone 选择本身不新。

## 后台训练快照

活动运行：`D:\DEMO\superembedding-storage\superembedding\v5\general_stage1_v51`。

数据：{'train': 7388, 'val': 760, 'ood': 163}；22014 条图文/问答监督（包括验证与留出）。8 个训练来源。排除 30 条与 ScienceQA/Urban1k 完全 RGB 或完全 dHash 匹配记录，清单 SHA256 `470b7a9b2ca8017fdaad15c404f54b9179a0d556434807d252e89102e9a92f44`。

第 60 步、首次验证前显式迁移：隔离 41 条验证近重复候选；训练记录/顺序、权重/优化器/RNG 和采样队列保留。原运行、断点及代码快照保留。验证改为全部 760 条（每图固定一条监督），不再使用每来源 16 条小监测集。

近重复筛查采用 pHash≤6 且 dHash≤6。人工确认 ChartQA/Chart2Text 至少一对内容相同，另有界面近重复；部分只是相同图表模板，保守隔离可能改变验证分布。87 对候选对应 41 个验证记录；外部完整 ScienceQA/Urban1k 未筛出候选，但该筛查不证明穷尽去污染。

```json
{
  "step": 544,
  "source": "localized_narratives",
  "loss": 0.35770511627197266,
  "lr": 8.840830521805525e-06,
  "gradient_norm": 18.483354568481445,
  "lora_probe_grad_norm": 0.13246619701385498,
  "lora_probe_delta_from_launch": 0.0006098529556766152,
  "gradient_replay_max_error": 0.0,
  "seconds": 64.48159250000026,
  "peak_allocated_MiB": 5257.9150390625,
  "pid": 5484,
  "state": "running",
  "last_checkpoint_step": 540,
  "completed_this_launch": 266,
  "elapsed_seconds": 18370.6001426
}
```

## 通用冻结特征架构对照

6 结构 × 3 seeds × 2 学习率，共 36 次；每来源 128 张训练图，完整验证。CPU 原始 OmniJev 特征，和 LoRA 长训独立。协议见 `GENERAL_ABLATION_PROTOCOL_V5.zh.md`。

```json
{
  "pipeline": {
    "state": "completed",
    "stages": 3,
    "publication": "not submitted"
  },
  "features": {
    "state": "completed",
    "pairs": 1947,
    "fingerprint": "4cc925e8b11347416cdf6d731ab51e079b469e27cbb0e3fee0f42f796a911053"
  }
}
```

结果见 `GENERAL_ABLATION_V5.zh.md`；参数量、维度和计算路径不同处单独列明，不将本组称为完整通用 LoRA 架构最优性证明。


### 已完成的冻结迁移评测：general_3cad7dfc34c7_cpu

```json
{
  "backend": "general_3cad7dfc34c7_cpu",
  "checkpoint": "D:\\DEMO\\superembedding-storage\\superembedding\\v5\\validation_selected_step250\\general_model.pt",
  "urban1k_common_999": {
    "image_to_text_R@1": 0.25525525212287903,
    "image_to_text_R@5": 0.46746745705604553,
    "image_to_text_R@10": 0.5775775909423828,
    "text_to_image_R@1": 0.22822822630405426,
    "text_to_image_R@5": 0.4464464485645294,
    "text_to_image_R@10": 0.5605605840682983
  },
  "evidence_diagnostic": {
    "color": {
      "anchor_preference": 0.859375,
      "robust_both_preference": 0.703125
    },
    "number": {
      "anchor_preference": 0.703125,
      "robust_both_preference": 0.40625
    },
    "position": {
      "anchor_preference": 0.59375,
      "robust_both_preference": 0.28125
    },
    "shape": {
      "anchor_preference": 0.84375,
      "robust_both_preference": 0.6875
    }
  },
  "scope": "Frozen transfer only; never used for checkpoint selection",
  "trained_projection": true,
  "max_pixels": 65536,
  "max_text_tokens": 256,
  "device": "cpu",
  "dtype": "float32"
}
```


### 已完成的冻结迁移评测：omnijev_raw_cpu

```json
{
  "backend": "omnijev_raw_cpu",
  "checkpoint": null,
  "urban1k_common_999": {
    "image_to_text_R@1": 0.0400400385260582,
    "image_to_text_R@5": 0.11011011153459549,
    "image_to_text_R@10": 0.1511511504650116,
    "text_to_image_R@1": 0.023023022338747978,
    "text_to_image_R@5": 0.07807807624340057,
    "text_to_image_R@10": 0.11011011153459549
  },
  "evidence_diagnostic": {
    "color": {
      "anchor_preference": 0.46875,
      "robust_both_preference": 0.25
    },
    "number": {
      "anchor_preference": 0.578125,
      "robust_both_preference": 0.265625
    },
    "position": {
      "anchor_preference": 0.34375,
      "robust_both_preference": 0.234375
    },
    "shape": {
      "anchor_preference": 0.53125,
      "robust_both_preference": 0.296875
    }
  },
  "scope": "Frozen transfer only; never used for checkpoint selection",
  "trained_projection": false,
  "max_pixels": 65536,
  "max_text_tokens": 256,
  "device": "cpu",
  "dtype": "float32"
}
```


冻结模型 `general_3cad7dfc34c7_cpu` ScienceQA Hit@1：14.40%。包含中途诊断和 text-only 控制时应分别阅读，不能自动视作最终模型结果；checkpoint 由验证而非本测试选择。


冻结模型 `general_3cad7dfc34c7_cpu_text_only` ScienceQA Hit@1：7.10%。包含中途诊断和 text-only 控制时应分别阅读，不能自动视作最终模型结果；checkpoint 由验证而非本测试选择。

配置和操作说明见 `PROTOCOL_V5.zh.md`、`RUNNING_V5.zh.md`。原子 last/best 保存优化器/RNG，固定里程碑保存权重。已有 V3/V4 checkpoint 全部保留。

同协议 Step 0/raw/主流基线及问答捷径结果见 `VALIDATION_CONTROLS_V5.zh.md`；原始 Step 0 已覆盖，明确区分重建权重。GPU 优先评测时长训会在完整断点暂停，并由有限任务驱动器恢复。

## 未完成项

训练是否完成及是否已有最终冻结评测，以上方状态和对应结果文件为准；没有结果文件就不能报告新模型主结果。磁盘扩容及迁移后，官方模型下载队列已启动；VLM2Vec/GME/WeMM/Ovis 尚未完成本机推理复现。下载不等于评测完成。没有完整 MMEB-V3、音视频能力结论、多种子确认性实验或新架构的完整参数/计算匹配消融。下一阶段须用未见任务检验空间/证据敏感性，并审计天然负例和近重复、各来源许可，才能讨论发表。

## D 盘存储

实际数据、断点和模型缓存在 `D:/DEMO/superembedding-storage`；已迁移 23.42 GB，C 盘旧路径为目录链接。详见 `STORAGE_D.zh.md`。

官方模型下载状态（不代表推理复现完成）：
```json
{
  "state": "completed",
  "models": [
    {
      "repo": "Qwen/Qwen3-VL-Embedding-2B",
      "revision": "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda",
      "path": "D:\\DEMO\\superembedding-storage\\official_models\\Qwen--Qwen3-VL-Embedding-2B",
      "scope": "Official files downloaded and hashed only; inference reproduction pending; remote code not executed."
    },
    {
      "repo": "tencent/WeMM-Embedding-2B",
      "revision": "bbd6cd4bf52cfc6716f752a2df80b2706720bd95",
      "path": "D:\\DEMO\\superembedding-storage\\official_models\\tencent--WeMM-Embedding-2B",
      "scope": "Official files downloaded and hashed only; inference reproduction pending; remote code not executed."
    },
    {
      "repo": "VLM2Vec/VLM2Vec-V2.0",
      "revision": "e39ff079b8275ef876d3656da8c0bddbff3c4dde",
      "path": "D:\\DEMO\\superembedding-storage\\official_models\\VLM2Vec--VLM2Vec-V2.0",
      "scope": "Official files downloaded and hashed only; inference reproduction pending; remote code not executed."
    },
    {
      "repo": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
      "revision": "9cfa6413f704a7c1cf5064d240748e10c876b286",
      "path": "D:\\DEMO\\superembedding-storage\\official_models\\Alibaba-NLP--gme-Qwen2-VL-2B-Instruct",
      "scope": "Official files downloaded and hashed only; inference reproduction pending; remote code not executed."
    },
    {
      "repo": "ATH-MaaS/Ovis-Omni-Embedding-3B",
      "revision": "9726d1ad56420a07ba745aafaf5d7030963cda4e",
      "path": "D:\\DEMO\\superembedding-storage\\official_models\\ATH-MaaS--Ovis-Omni-Embedding-3B",
      "scope": "Official files downloaded and hashed only; inference reproduction pending; remote code not executed."
    },
    {
      "repo": "Qwen/Qwen2-VL-2B-Instruct",
      "revision": "895c3a49bc3fa70a340399125c650a463535e71c",
      "path": "D:\\DEMO\\superembedding-storage\\official_models\\Qwen--Qwen2-VL-2B-Instruct",
      "scope": "Official files downloaded and hashed only; inference reproduction pending; remote code not executed."
    }
  ]
}
```
