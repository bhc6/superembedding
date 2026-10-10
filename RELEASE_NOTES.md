# v0.2.0 — multi-domain attribute memory experiments

Completed local research on **28,263 images** from Fashion, CUB attributes, and DTD textures. Three nested training sizes, three seeds, two learning rates: **432 training runs**, followed by **216 validation-sealed test evaluations**.

OmniJev raw attribute mAP: **82.10 ± 0.21 / 44.36 ± 0.10 / 49.37 ± 0.25** on Fashion/CUB/DTD. Same-protocol FLAIR raw: **74.50 ± 0.39 / 43.41 ± 0.03 / 50.88 ± 0.35**. These are image-retrieval tasks, not official CUB/DTD classification benchmarks.

Under equal joint training, compressed output projection versus concat changes global mAP by **+0.95 / −0.02 / −0.98 percentage points**. Projection is task-dependent, not universally necessary. Frozen gates underperform raw projection. More data helps DTD; Fashion/CUB largely plateau at the fixed update budget. Most selected runs peak at step 1,000, so this is a budget-limited study.

All 432 heads and all final optimizer/RNG recovery checkpoints are included, including both learning rates and negative outcomes. Each dataset has one heads archive and three backend-specific resume archives. `SHA256SUMS.json` covers all archives. Head reloads exactly reproduce validation scores; verified interrupted training is bitwise equivalent to continuous training.

Single-image inference matches training batch geometry to avoid amplified BF16 batch-1 rounding. Full fresh-inference checks are included. See the complete results, Chinese conclusions, model card, data manifests, and reproduction scripts in the repository.

No source images or upstream backbone weights are bundled. CUB/DTD artifacts are for research; project licensing does not waive dataset restrictions or assert commercial clearance. The v0.1 pilot and its LoRA negative result remain unchanged. V3 trains frozen-backbone heads, not a new end-to-end LoRA model, and makes no novelty, causal-transfer, or SOTA claim.
