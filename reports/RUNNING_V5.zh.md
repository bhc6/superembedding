# V5 后台训练操作

> 2026-10-12 更正：Step 556 可作为已知历史数据暴露下的续训起点；新隔离评测不必一律从零训练。原运行保留暂停，独立目录恢复至固定 Step 2000，旧验证 best 不作正式选模。见 [续训协议](STEP556_CONTINUATION.zh.md)。泄露整改和发布仍未放行。

> 2026-10-11 泄露审查更新：V5 已在 Step 556 保存断点并暂停；同数据 frozen/LoRA 队列已停止。CUB/DTD 同图/同场景与 DocVQA 实体交叉尚未整改完成，历史分数仅作探索性结果。以 [数据泄露复核](DATA_LEAKAGE_REVIEW.zh.md) 为准；不得直接恢复旧主结果流程。

工作目录：`D:\DEMO\superembedding`。数据、当前训练和断点位于 `D:\DEMO\superembedding-storage\superembedding\v5`。41 个测试通过，真实 OmniJev 文本/图像梯度回放、checkpoint optimizer/RNG 重放通过；不代表训练质量通过。

## 查看

```powershell
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/general_stage1_v51/status.json
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/general_stage1_v51/supervisor.json
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/general_stage1_v51/worker_attempt_0.log -Tail 10
```

`status.json` 的 `step` 是完成的优化器更新数；`last_checkpoint_step` 才是可恢复到的步数。`supervisor.json`/日志中的 PID 因 Windows Python 启动器可能和实际 Python worker 不同，进程核对要同时看命令行。首次已有真实更新后才可以称为训练运行中。

## 暂停与继续

```powershell
New-Item D:/DEMO/superembedding-storage/superembedding/v5/general_stage1_v51/STOP_AFTER_CHECKPOINT -ItemType File
```

等待 `status.json` 变为 `paused`；会在当前更新完成后保存。恢复前移除此显式暂停标记，再运行：

```powershell
.venv/Scripts/python.exe scripts/supervise_general.py --data D:/DEMO/superembedding-storage/superembedding/v5/data_v51 --output D:/DEMO/superembedding-storage/superembedding/v5/general_stage1_v51 --steps 2000
```

需要隐藏后台窗口时使用 PowerShell `Start-Process -WindowStyle Hidden`，见已有 supervisor 配置中的完整命令。不要重复启动；目录有 FileLock。机器休眠/关机时不会继续算，重开后按相同命令从 last 恢复。配置、实现、数据指纹变更会拒绝旧断点；新实验使用独立目录。

## 复现入口

```powershell
.venv/Scripts/python.exe scripts/prepare_general_data.py --output D:/DEMO/superembedding-storage/superembedding/v5/data --per-source 1024
.venv/Scripts/python.exe scripts/prepare_mmeb_scienceqa.py
.venv/Scripts/python.exe scripts/audit_general_data.py --data D:/DEMO/superembedding-storage/superembedding/v5/data
.venv/Scripts/python.exe scripts/evaluate_mmeb_scienceqa.py --backend qwen3_vl
.venv/Scripts/python.exe scripts/evaluate_mmeb_scienceqa.py --backend clip_b32
.venv/Scripts/python.exe scripts/evaluate_mmeb_scienceqa.py --backend siglip2_b16
.venv/Scripts/python.exe scripts/evaluate_mmeb_scienceqa.py --backend omnijev_raw
.venv/Scripts/python.exe scripts/evaluate_urban1k.py --backend flair
.venv/Scripts/python.exe scripts/evaluate_urban1k_api.py
.venv/Scripts/python.exe scripts/make_evidence_diagnostic.py
.venv/Scripts/python.exe scripts/evaluate_evidence_api.py
.venv/Scripts/python.exe scripts/report_general_progress.py
```

Urban1k 使用 `BeichenZhang/Urban1k` revision `953fcf1d1a3bde031d9424a91a0154e594b5a586` 的 `Urban1k.zip`，images 放 `.../v5/urban1k/images`；注释复制本仓库固定 FLAIR checkout 的 `datasets/Urban1k/annotations/annotations.json`。审计程序要求 1000 张 Urban1k 和 1000 张 ScienceQA 完整图像后才允许封存训练清单。

不要在训练运行时重新构建训练 manifest。所有 API shard 保存 request_id、usage、时间和向量， `.env` 不进入 Git；恢复会复用已有分片。API alias 没有不可变版本，不把 API 分数标成开源 2B/8B 权重成绩。

## 保留策略

- `last.pt`：最近 5 步边界的完整恢复状态。
- `best.pt`：仅由验证选中的完整恢复状态。
- `step_000500_weights.pt` 等：固定里程碑全部训练参数，保留不覆盖；不是完整 optimizer 断点。
- `history.jsonl`：所有已记录优化器更新，包括可能在故障恢复时重放的步号；分析以恢复链为准。
- `STOP_AFTER_CHECKPOINT`：用户显式暂停标记；恢复前移除。

监督任务预算 2000 步，最多重启 2 次；磁盘不足或确定性数据/数值错误会保留已有文件后停止。训练完成后需要独立评测、复现补齐和报告审核，再按用户要求一次性提交 GitHub。

已排队的完成动作：`finish_general_stage.py` 自动读取验证最佳 `best.pt`，运行完整 ScienceQA、Urban1k 999 共同子集和成对证据诊断，并补测原始 OmniJev 表示，再更新进展报告。不会按测试结果自动换 checkpoint，也不会自动发布未完成研究。完成状态写 `final_evaluation.json`；若评测失败，修复后可单独运行该脚本，不必重训。

导出修复：finalizer 从 `best.pt` 生成完整 LoRA+projection 的 `export/general_model.pt`，真实重载检查通过后，正式评测直接使用这个模型包。训练器留下的最后一步 head-only 文件保留为 `last_step_head_only_DO_NOT_DEPLOY.pt`，不可用旧 `encode_image.py` 部署。通用推理入口是 `encode_general.py`；详情见 `EXPORT_CONSISTENCY_V5.zh.md`。

2026-10-11 已实际在第 6 步受控保存并退出，重启日志确认 `resumed_step: 6`，随后完成第 7 步更新。`background_checkpoint_audit.json` 验证了 304 个训练张量、304 个 optimizer state、CUDA RNG 和实现哈希。初期每步约 67–68 秒，2000 步加验证预计约 38–42 小时；只是按早期吞吐估计，不是完成时间保证。


V5.1 已在第 60 步从原运行显式迁移，并确认恢复后的第 61 步完成。活动指针在 `runs/v5/active_general_run.json`。原 `general_stage1` 是被替代的保留运行，暂停标记应保留，不要恢复它。新运行使用 `data_v51` 清单；该目录引用原 `data` 中的图像，因此两个目录都要保留。

首次验证前保守隔离 41 条近重复候选，完整验证集为 760 图（每图固定一条监督）。恢复状态迁移哈希一致，训练 7388 条记录及其顺序未变。详见 `PROTOCOL_V5.zh.md`。验证扩大后，原 38–42 小时估计不再包含准确的验证耗时，需要首次完整验证后更新。

新增复现入口（CPU 对照不占训练显存）：

```powershell
.venv/Scripts/python.exe scripts/evaluate_evidence_local.py --backend clip_b32
.venv/Scripts/python.exe scripts/evaluate_evidence_local.py --backend siglip2_b16
.venv/Scripts/python.exe scripts/evaluate_evidence_flair.py
.venv/Scripts/python.exe scripts/report_evidence_comparison.py
.venv/Scripts/python.exe scripts/scienceqa_modality_controls.py
```

迁移后第 65 步完整断点已通过读取审计：304 个训练张量、304 个 AdamW 状态，optimizer 步号全部为 65；代码快照、来源队列继承链与完整 760 条验证清单通过。审计文件：`artifacts/results/v5/migration_resume_audit.json`。

最终冻结评测在加载 checkpoint 时检查全部 LoRA 和 projection 参数是否齐全，以及方向顺序、基础权重来源和编码实现哈希，防止 `strict=False` 静默保留随机投影。完成后自动合并 Qwen、CLIP、SigLIP、FLAIR 两模式、原始 OmniJev 和训练模型的场景成对统计；这仍不等于完成完整 SOTA 或多种子架构实验。

## 独立 CPU 架构对照

`general_ablation` 是独立的 CPU 后台流水线，依次提取原始 OmniJev 的 1947 对特征、训练 36 个 head、封存验证选择并评测 18 个选中模型。图像引用 `data_v51`，不复制上游图片。原始特征分片、所有 last/best 均保留。步骤和种子见 `GENERAL_ABLATION_PROTOCOL_V5.zh.md`。

```powershell
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/general_ablation/pipeline_status.json
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/general_ablation_features/status.json
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/general_ablation/stage_0.log -Tail 5
# 仅在流水线退出后按原命令恢复，不要重复启动
.venv/Scripts/python.exe scripts/run_general_ablation.py --data D:/DEMO/superembedding-storage/superembedding/v5/data_v51 --root D:/DEMO/superembedding-storage/superembedding/v5/general_ablation
```

提取阶段创建 `general_ablation_features/STOP_AFTER_SHARD` 可在当前分片保存后暂停；恢复前移除。训练阶段 last/best 每 100 步保存，失败后相同入口恢复。真实训练器在固定合成 1024 维特征上验证过 3→6 步中断恢复，与连续 6 步的模型/optimizer/RNG/采样队列/验证历史摘要完全一致；这不算检索效果验证。证据在 `general_ablation_resume_audit.json`。


2026-10-11 存储已迁移到 D 盘；原 C 盘入口仅为 junction。新终端先 `. ./scripts/use_d_storage.ps1`，后台启动使用 `. ./scripts/start_v5_on_d.ps1`。详情见 `STORAGE_D.zh.md`。首次启动的 checkpoint 为第 205 步，协议和数据指纹保持不变。

## 优先验证、基线与捷径检查

2026-10-11：按用户顺序，长训在 Step 278 完整断点暂停，由 `scripts/run_priority_validation.py` 顺序执行 CUDA 重建初始模型、同精度 raw/重建 Step 0、已封存 Step 250 的完整输入和 text-only。驱动器结束或子评测失败后移除自身暂停标记，重新启动原训练 supervisor，沿用 optimizer/RNG。运行期间不要手动删除该标记或另起 CUDA 模型。Step 0 原件已覆盖，重建结果不能称为找回原件；CPU/GPU 构造得到相同 head 哈希，RNG 与历史 Step 60 一致。

本地 Qwen3-VL-Embedding-2B 在 CPU 独立执行同协议 1688 个去重输入，FP32、65536 pixels、官方池化；三对象冒烟已通过。API 同协议和 train-only 多数类已完成，见 `VALIDATION_CONTROLS_V5.zh.md`。所有向量原子分片保存并校验来源，失败后按相同参数继续；修改代码后须新建缓存或明确迁移指纹，不能盲目复用。

`scripts/run_selected_external.py` 是有限后台队列：等待优先 GPU 验证和 Qwen2B 完成，随后 CPU 执行封存 Step 250 的官方 ScienceQA 完整输入/text-only、Urban1k、成对证据，再执行 CPU raw 迁移对照及报告。最多等待 8 小时，失败保留缓存、停止并记录。它与最终 2000 步模型评测分开，绝不据外部测试换模型。最终器也已补入同协议问答控制及 ScienceQA text-only。

```powershell
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/priority_validation/status.json
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/initial_validation/cuda/status.json
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/validation_controls/qwen2b_cpu/status.json
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/selected_external_step250/status.json
.venv/Scripts/python.exe scripts/report_validation_controls.py
```

QA text-only 通用入口已用 Qwen API 的完整 1000 条 ScienceQA 真实请求验证：32.6%，与先前独立控制脚本一致，完整输入减控制 40.5 pp，配对区间 [37.3, 43.8] pp。61 个测试通过。问题类型是固定文字规则，稀有类别只作描述。项目仍未全部完成，按用户要求不提交 Git。

## 2026-10-11 晚间完整性复核

以上优先队列、Qwen2B、Step 250 choice 及外部队列现已完成，历史进度段不代表仍在等待。新审计入口 `scripts/audit_experiment_completeness.py` 从实际向量缓存重算分数、检查分组、checkpoint/optimizer/RNG 和封存实现；结果见 `EXPERIMENT_COMPLETENESS_V5.zh.md`。主训练仍继续到预定 2000 步，不因 Step 250 外部结果调整选择规则。

补充队列 `scripts/run_external_control_completion.py` 正在完成 CPU raw OmniJev 的 ScienceQA 完整输入/text-only，对齐 Step 250 外部评测精度；失败保留缓存，状态位于 D 盘 v5/external_control_completion/status.json。未跑完时不得把这一同精度对照标为已完成。

默认 pytest 现包括 experiments/lora_freeze_v5，合计 70 项通过。原测试模块导入时改 CUDA 可见性的问题已修复，不涉及封存训练器。最终评测入口已增加 `seal_external_inputs.py` 的运行前/后核验，2396 个图片/元数据/实现资产的现有清单为事后检查，不证明历史不可变。
