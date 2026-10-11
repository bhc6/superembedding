# Reproducing the mainstream embedding comparison

First restore the v3 datasets using `scripts/restore_research_data.py` and install the pinned project dependencies. V4 uses exactly those manifests and images. A CUDA GPU is needed for local extraction, training and scoring; the original machine has 8 GB VRAM. No upstream weights or source images are redistributed.

Create a local ignored `.env` containing `QWEN_API_KEY=...` or set `DASHSCOPE_API_KEY` in the environment. API calls are billable and require network access to DashScope. Do not put credentials in command arguments. This project never prints headers, image payloads or configuration values.

The following commands use relative cache paths; any absolute cache directory is supported. The original run used a C-drive cache because D-drive space was limited. Archives preserve portable project-relative paths.

```powershell
.venv/Scripts/python.exe scripts/extract_baselines.py --backends clip_b32 siglip2_b16 dinov2_small --root runs/v4/features --model-cache models/v4
.venv/Scripts/python.exe scripts/extract_baselines.py --backends qwen3_vl qwen3_vl_instruct tongyi_plus --root runs/v4/features
.venv/Scripts/python.exe scripts/run_baseline_training.py --features runs/v4/features
.venv/Scripts/python.exe scripts/verify_baseline_resume.py --features runs/v4/features --backend clip_b32
.venv/Scripts/python.exe scripts/verify_baseline_resume.py --features runs/v4/features --backend qwen3_vl_instruct
.venv/Scripts/python.exe scripts/evaluate_baselines.py --features runs/v4/features
.venv/Scripts/python.exe scripts/verify_baseline_inference.py --features runs/v4/features --model-cache models/v4
.venv/Scripts/python.exe scripts/account_baseline_api.py --features runs/v4/features
.venv/Scripts/python.exe scripts/report_baselines.py
```

The two extractors may run concurrently; there are at most four API requests in flight. Training can run while extraction proceeds: a bank becomes available only after its complete packed file is atomically written. The training queue waits for unavailable banks. Stop it if extraction has failed, repair that failure, and restart both commands. A repeated extraction validates existing shards and only sends missing batches. Completed banks do not trigger API calls. A successful response interrupted before the atomic cache write may be billed again on restart.

Training checkpoints save every 100 updates, validate every 250, and retain the complete optimizer/RNG/log state. Rerun the same command to continue. Changing data, features or training configuration invalidates resume. Both learning rates and all seeds remain in the release, including non-selected runs. Learning rates are chosen by the **three-seed mean validation score**, one LR per dataset/backend/training size. Each seed's checkpoint is chosen on validation. Test metrics are scored only after the complete grid has been sealed.

For a new image:

```powershell
.venv/Scripts/python.exe scripts/encode_baseline_image.py --head path/to/best_head.pt --image path/to/image.jpg --output runs/vector.pt --model-cache models/v4
```

The head metadata selects the backend and fixed attribute basis. Attribute-instruct Qwen requests one image vector per fixed direction; no test label or eventual query is transmitted. Global baselines produce identical slots after the shared head. Use the individual normalized slots for attribute retrieval, or the normalized concatenated global vector. A new attribute basis needs new features and training.

Local model revisions and preprocessing are pinned in `superembedding/baselines.py`. The Qwen service alias is not an immutable model release: results are measurements of the cached service outputs, and future re-encoding may differ. The inference audit records fresh-query drift rather than assuming server stability. API token accounting excludes preflight and fresh-inference audit calls and cannot replace the provider invoice.

V3's frozen baselines and selected raw-head results are reused unchanged by the v4 report. V4 does not rerun the earlier decision-gate or output-projection ablations. See `EXPERIMENT_PROTOCOL_V4.zh.md` for the predeclared design and `RESULTS_V4.md` for actual results.

After the audits, `scripts/package_baselines.py --output release_assets/v4` creates six checksum-verified archives: one per backend, each containing all 54 heads, all 54 optimizer/RNG recovery states, validation logs and documentation. The GitHub release is separate from packaging; `.env`, model download caches and source images are excluded by the explicit archive member list.
