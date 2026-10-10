# Reproducing the multi-domain study

Use the pinned Windows CUDA environment in `requirements-local.lock.txt`, with PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, and Transformers 5.5.0. The 8GB RTX 5070 Laptop run uses the upstream PyTorch fallback for Qwen3.5 linear attention. Start in the repository root. The model/data licenses and fixed source revisions are documented separately.

Download the upstream repositories/models with the existing `scripts/clone_dependencies.py` and `scripts/download_models.py` workflows. FLAIR downloads its pinned official weight file automatically. The public source images are downloaded from their original providers; they are not in this GitHub repository or release.

Reconstruct **the published splits**, without regenerating the pilot experiments:

```powershell
.venv\Scripts\python.exe scripts/restore_research_data.py
```

This verifies each decoded RGB image against the published manifest hash. To investigate how the splits were originally constructed, read `scripts/prepare_multidomain.py`. Its Fashion construction depends on the historical pilot manifests and refuses to overwrite a different frozen v3 manifest. Reproduction should ordinarily restore the published splits instead.

Run all encoding and the entire 432-run validation grid:

```powershell
.venv\Scripts\python.exe scripts/run_research.py --parallel
```

Omit `--parallel` for a sequential queue with lower resource contention. Both modes use the same model/data/optimization settings. Do not run two queue instances at once. Individual extraction directories and training backends have locks as well. Inspect progress with `scripts/research_status.py` and logs under `runs/v3/logs`. Restart the same command after interruption; complete runs and valid cached features are retained.

Each run keeps `last.pt` with optimizer/RNG state every 100 steps, plus `best_head.pt`. Data/model/feature fingerprints prevent resuming with changed inputs. A numerical BF16 audit compares batched encoding with the official serial branch path using training images. Atomic writes protect the last completed checkpoint. Hardware/software changes can alter numerical results; exact resume requires matching feature values.

Once the full training grid is complete:

```powershell
.venv\Scripts\python.exe scripts/verify_research_resume.py
.venv\Scripts\python.exe scripts/audit_research.py
.venv\Scripts\python.exe scripts/verify_research_exports.py
.venv\Scripts\python.exe scripts/evaluate_research.py
.venv\Scripts\python.exe scripts/report_research.py
```

The audit verifies all 432 runs and writes `selection_sealed.json`. Only then can the evaluator score the 216 checkpoints selected across learning rates. Never revise the protocol or choose another checkpoint based on these test results. Cached test evaluations validate the sealed selection before reuse. Generated results include the full grid, per-seed metrics, coverage, paired projection differences, frozen-feature controls, and analytic random-ranking expectations.

The published multi-domain head files can be used without optimizer state:

```powershell
.venv\Scripts\python.exe scripts/encode_research_image.py --head path/to/best_head.pt --image path/to/image.jpg --output runs/vector.pt
```

The backend is read from the checkpoint. Use `--adapter`, `--base`, or `--flair` to point at the pinned local model directories. Slots and the global vector are saved separately. Compare slots from the same attribute basis using cosine similarity; select and concatenate slots for an attribute subset. Use the global vector for the all-attribute task. A compressed global vector cannot be split back into the original attribute slots.

Package a completed, audited study with `scripts/package_research.py --output <archive-directory>`. Packaging verifies archive members against every original checkpoint. Publishing is separate and is not part of the training/evaluation scripts. Retain the v0.1 release as the pilot; do not mix its previously inspected test split into v3 evaluation.
