# 项目存储迁移到 D 盘

2026-10-11，按用户要求将项目在 C 盘的数据、模型缓存、断点和历史发布资产迁到 `D:\DEMO\superembedding-storage`。项目代码和虚拟环境本来就在 `D:\DEMO\superembedding`，保持原位。

| 内容 | 实际位置 |
|---|---|
| 历史发布资产、V4 缓存、V5 数据和断点 | `D:\DEMO\superembedding-storage\superembedding` |
| Hugging Face 模型、数据集及认证缓存 | `D:\DEMO\superembedding-storage\huggingface` |
| pip / uv 缓存 | `D:\DEMO\superembedding-storage\pip-cache` / `uv-cache` |
| 项目进程临时目录 | `D:\DEMO\superembedding-storage\tmp` |
| Torch / 扩展编译缓存 | `D:\DEMO\superembedding-storage\torch` / `torch-extensions` |
| 其他项目运行缓存 | 同目录下 `xdg-cache`、`matplotlib`、`numba`、`triton` |

旧的 C 盘缓存目录保留为 NTFS junction，指向上述 D 盘实际目录。它们只是兼容入口：历史数据清单与缓存指纹包含旧的绝对路径，保留入口可以原样恢复。不要把这些入口当成 C 盘真实数据重复清理，也不要删除它们指向的目标。

迁移按“暂停并保存 → 跨盘复制 → 全部文件 SHA256 比对 → 确认源目录未变 → 建立并核对 junction → 清理已验证的旧副本”执行。主训练在第 205 步暂停；304 个 optimizer 状态、数据和实现指纹均已检查。CPU 提取在 615 个完成分片处暂停，迁移后逐分片核对并继续。此次是存储搬迁，不改变训练协议、初始化或随机数状态。

新终端可先导入项目环境：

```powershell
cd D:\DEMO\superembedding
. ./scripts/use_d_storage.ps1
# 在没有暂停标记且没有同一后台任务时启动/恢复
. ./scripts/start_v5_on_d.ps1
```

项目虚拟环境的 PowerShell 激活脚本也会导入这份 D 盘设置。HF、pip、uv、Torch 的缓存位置同时写入用户环境变量，供新启动的进程使用。`TEMP/TMP` 只对导入脚本的项目进程及其子进程改到 D 盘，不改变 Windows 全局临时目录。系统 Python、其他应用和非本项目文件不在迁移范围。

查看真实运行状态：

```powershell
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/general_stage1_v51/status.json
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/general_ablation/pipeline_status.json
Get-Content D:/DEMO/superembedding-storage/superembedding/v5/general_ablation_features/status.json
```

公开迁移摘要在 `artifacts/results/v5/storage_migration.json`；逐文件校验记录位于 D 盘存储根目录，不随 Git 发布，因为缓存清单可能包含本机认证文件的名称和摘要。HF 认证内容不输出或发布。后续大模型下载可用 D 盘剩余空间，GPU 仍为 8 GB，不能把磁盘扩容解释为显存扩容。

本轮迁移不会触发 Git 提交。全部研究与发布核验完成后再一次性提交和推送。

完成校验的迁移总量为 23,418,733,859 字节、77,546 个文件；迁移后 C 盘实测可用约 25.9 GB，数值会随系统运行变化。恢复后已观察到第 209 步及 712 个提取分片，工作进程环境也核实为 D 盘缓存/临时目录。

官方权重另在 D 盘后台下载：Qwen3-VL-Embedding-2B、WeMM-Embedding-2B、VLM2Vec-V2.0 adapter 及 Qwen2-VL-2B-Instruct base、GME-2B、Ovis-3B，固定清单预计约 34.15 GB。状态在 `D:/DEMO/superembedding-storage/official_models/status.json`；版本清单在 `sources/official_embedding_models_v5.json`。不执行下载的 remote code；后续审查加载实现并分时推理，避免与 GPU 长训争用显存。
