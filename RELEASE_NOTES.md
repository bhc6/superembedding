# v0.3.0 — mainstream embedding comparison

324 new frozen-backbone head runs, 162 validation-selected test evaluations, six new baselines, three domains, and three nested training sizes on the same 28,263 images as v3. Adds OpenAI CLIP B/32, Google SigLIP2 B/16, Meta DINOv2 Small, Qwen3-VL embedding API (global and fixed attribute instruct), and Tongyi Vision Plus 2026-03-06.

Full-training attribute mAP (%), three seeds mean ± SD:

| Model | Fashion | CUB | DTD |
|---|---:|---:|---:|
| OmniJev raw (v3) | 82.10 ± 0.21 | 44.36 ± 0.10 | 49.37 ± 0.25 |
| CLIP B/32 | 69.22 ± 0.15 | 42.48 ± 0.09 | 45.95 ± 0.50 |
| SigLIP2 B/16 | 71.29 ± 0.07 | 43.84 ± 0.06 | 62.94 ± 0.42 |
| DINOv2 Small | 67.44 ± 0.17 | 43.80 ± 0.09 | 53.83 ± 0.22 |
| Qwen3-VL API | 68.63 ± 0.14 | 44.35 ± 0.05 | 65.42 ± 0.14 |
| Qwen3-VL API + attribute instruct | 79.90 ± 0.56 | 47.38 ± 0.03 | 65.43 ± 0.17 |
| Tongyi Vision Plus | 72.12 ± 0.10 | 44.12 ± 0.23 | 65.95 ± 0.09 |

OmniJev is not universally stronger. Under the same shared raw-head/update budget, fixed Qwen attribute instructions add +11.28/+3.03/+0.01 pp on Fashion/CUB/DTD. Native frozen embeddings are reported separately. Input widths, pretraining and compute differ. General image embeddings remain identical across repeated slots; conditional backends encode distinct fixed directions. These are custom retrieval protocols, not official classification/MMEB results.

Six archives contain all 324 heads, all 324 complete optimizer/RNG recovery states, validation reports and documentation. All heads exactly reproduce stored validation mAP; global and multi-slot exact-resume checks pass. Fresh image inference across all 18 domain/backend combinations matches cached head outputs within 2.98e-8. SHA256SUMS.json verifies each archive. Negative and non-selected runs remain included.

API extraction completed 16,264 successful requests without retries. Service aliases may change; cached features remain local and are not part of this release. Regeneration from an API is not guaranteed bitwise reproducible. No images, credentials, upstream backbone weights or provider model weights are bundled. CUB/DTD research restrictions and provider terms apply; project licensing does not waive them. V3 and v0.1 releases remain intact.

See reports/RESULTS_V4.md, reports/CONCLUSIONS_V4.zh.md, reports/EXPERIMENT_PROTOCOL_V4.zh.md and reports/REPRODUCING_V4.md for measurements, limits and commands. This release does not establish decision-sensitivity transfer, unseen-attribute generalization or novelty.
