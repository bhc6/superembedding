# Mainstream embedding comparisons (v4)

Full fixed v3 test galleries; attribute macro mAP (%). Supervised rows: three seeds, mean ± sample SD. No test-based hyperparameter selection. Native frozen embeddings and supervised heads are separate tracks.

## Frozen native features

| Model | Fashion | CUB | DTD | Native width per slot |
|---|---:|---:|---:|---:|
| OmniJev raw | 60.80 | 39.42 | 23.33 | 1024 |
| Qwen3.5 base raw | 57.79 | 37.68 | 37.52 | 1024 |
| FLAIR | 58.89 | 39.86 | 35.50 | 512 |
| CLIP B/32 | 58.71 | 39.39 | 28.84 | 512 |
| SigLIP2 B/16 | 60.43 | 40.88 | 39.24 | 768 |
| DINOv2 Small | 60.23 | 40.91 | 42.19 | 384 |
| Qwen3-VL API | 61.99 | 41.37 | 47.59 | 2560 |
| Qwen3-VL API + attribute instruct | 65.22 | 42.81 | 46.14 | 2560 |
| Tongyi Vision Plus 2026-03-06 | 61.47 | 39.52 | 45.09 | 1152 |
| FLAIR global | 59.94 | 39.54 | 34.19 | 512 |

## Equal training budget, full training data

Shared LN + Linear(256), 1,000 optimizer steps, batch 64, same loss, two learning rates and three seeds. Backbones, resolutions, pretraining data and service compute differ. Global features yield identical slots after a shared head.

| Model | Fashion | CUB | DTD |
|---|---:|---:|---:|
| OmniJev raw | 82.10 ± 0.21 | 44.36 ± 0.10 | 49.37 ± 0.25 |
| Qwen3.5 base raw | 71.89 ± 1.32 | 41.02 ± 0.06 | 49.94 ± 0.63 |
| FLAIR | 74.50 ± 0.39 | 43.41 ± 0.03 | 50.88 ± 0.35 |
| CLIP B/32 | 69.22 ± 0.15 | 42.48 ± 0.09 | 45.95 ± 0.50 |
| SigLIP2 B/16 | 71.29 ± 0.07 | 43.84 ± 0.06 | 62.94 ± 0.42 |
| DINOv2 Small | 67.44 ± 0.17 | 43.80 ± 0.09 | 53.83 ± 0.22 |
| Qwen3-VL API | 68.63 ± 0.14 | 44.35 ± 0.05 | 65.42 ± 0.14 |
| Qwen3-VL API + attribute instruct | 79.90 ± 0.56 | 47.38 ± 0.03 | 65.43 ± 0.17 |
| Tongyi Vision Plus 2026-03-06 | 72.12 ± 0.10 | 44.12 ± 0.23 | 65.95 ± 0.09 |

The head architecture and update budget are shared, but input widths differ, so trainable parameter counts are not equal.

| Model | Head trainable parameters |
|---|---:|
| OmniJev raw | 264448 |
| Qwen3.5 base raw | 264448 |
| FLAIR | 132352 |
| CLIP B/32 | 132352 |
| SigLIP2 B/16 | 198400 |
| DINOv2 Small | 99328 |
| Qwen3-VL API | 660736 |
| Qwen3-VL API + attribute instruct | 660736 |
| Tongyi Vision Plus 2026-03-06 | 297472 |

## Full-data joint retrieval

| Model | Fashion | CUB | DTD |
|---|---:|---:|---:|
| OmniJev raw | 54.00 ± 0.65 | 9.60 ± 0.07 | 49.37 ± 0.25 |
| Qwen3.5 base raw | 36.86 ± 2.62 | 8.18 ± 0.10 | 49.94 ± 0.63 |
| FLAIR | 55.10 ± 0.28 | 9.98 ± 0.06 | 50.88 ± 0.35 |
| CLIP B/32 | 54.52 ± 1.61 | 9.73 ± 0.12 | 45.95 ± 0.50 |
| SigLIP2 B/16 | 62.85 ± 0.74 | 10.80 ± 0.07 | 62.94 ± 0.42 |
| DINOv2 Small | 40.69 ± 0.76 | 10.87 ± 0.10 | 53.83 ± 0.22 |
| Qwen3-VL API | 55.21 ± 0.03 | 11.16 ± 0.07 | 65.42 ± 0.14 |
| Qwen3-VL API + attribute instruct | 54.34 ± 0.76 | 11.59 ± 0.07 | 65.43 ± 0.17 |
| Tongyi Vision Plus 2026-03-06 | 67.87 ± 0.25 | 11.29 ± 0.22 | 65.95 ± 0.09 |

## Data scaling (attribute mAP)

| Dataset | Model | 25% train | 50% train | 100% train |
|---|---|---:|---:|---:|
| Fashion | OmniJev raw | 81.93 ± 0.23 | 81.83 ± 0.20 | 82.10 ± 0.21 |
| Fashion | Qwen3.5 base raw | 72.38 ± 0.37 | 72.62 ± 0.71 | 71.89 ± 1.32 |
| Fashion | FLAIR | 74.65 ± 0.14 | 74.64 ± 0.24 | 74.50 ± 0.39 |
| Fashion | CLIP B/32 | 69.09 ± 0.25 | 69.32 ± 0.04 | 69.22 ± 0.15 |
| Fashion | SigLIP2 B/16 | 70.97 ± 0.34 | 71.23 ± 0.12 | 71.29 ± 0.07 |
| Fashion | DINOv2 Small | 67.24 ± 0.07 | 67.40 ± 0.18 | 67.44 ± 0.17 |
| Fashion | Qwen3-VL API | 68.62 ± 0.26 | 68.73 ± 0.16 | 68.63 ± 0.14 |
| Fashion | Qwen3-VL API + attribute instruct | 79.64 ± 0.18 | 79.93 ± 0.34 | 79.90 ± 0.56 |
| Fashion | Tongyi Vision Plus 2026-03-06 | 71.72 ± 0.09 | 72.09 ± 0.17 | 72.12 ± 0.10 |
| CUB | OmniJev raw | 44.17 ± 0.16 | 44.40 ± 0.01 | 44.36 ± 0.10 |
| CUB | Qwen3.5 base raw | 41.21 ± 0.15 | 41.20 ± 0.19 | 41.02 ± 0.06 |
| CUB | FLAIR | 42.97 ± 0.10 | 43.31 ± 0.15 | 43.41 ± 0.03 |
| CUB | CLIP B/32 | 41.86 ± 0.07 | 42.21 ± 0.12 | 42.48 ± 0.09 |
| CUB | SigLIP2 B/16 | 43.43 ± 0.14 | 43.73 ± 0.15 | 43.84 ± 0.06 |
| CUB | DINOv2 Small | 43.00 ± 0.10 | 43.43 ± 0.11 | 43.80 ± 0.09 |
| CUB | Qwen3-VL API | 43.80 ± 0.18 | 44.12 ± 0.07 | 44.35 ± 0.05 |
| CUB | Qwen3-VL API + attribute instruct | 46.72 ± 0.05 | 47.11 ± 0.10 | 47.38 ± 0.03 |
| CUB | Tongyi Vision Plus 2026-03-06 | 43.05 ± 0.11 | 43.64 ± 0.09 | 44.12 ± 0.23 |
| DTD | OmniJev raw | 46.71 ± 0.41 | 48.26 ± 0.71 | 49.37 ± 0.25 |
| DTD | Qwen3.5 base raw | 47.67 ± 0.76 | 49.34 ± 0.61 | 49.94 ± 0.63 |
| DTD | FLAIR | 46.58 ± 0.47 | 49.21 ± 1.05 | 50.88 ± 0.35 |
| DTD | CLIP B/32 | 40.50 ± 0.28 | 43.34 ± 0.28 | 45.95 ± 0.50 |
| DTD | SigLIP2 B/16 | 55.50 ± 0.32 | 59.27 ± 0.46 | 62.94 ± 0.42 |
| DTD | DINOv2 Small | 48.25 ± 0.17 | 51.42 ± 0.51 | 53.83 ± 0.22 |
| DTD | Qwen3-VL API | 59.07 ± 0.29 | 63.49 ± 0.14 | 65.42 ± 0.14 |
| DTD | Qwen3-VL API + attribute instruct | 59.63 ± 0.22 | 63.40 ± 0.02 | 65.43 ± 0.17 |
| DTD | Tongyi Vision Plus 2026-03-06 | 58.91 ± 0.31 | 63.25 ± 0.28 | 65.95 ± 0.09 |

## Controlled Qwen instruction comparison

Same API model and native dimension; fixed attribute instructions versus global image features. Differences below are instruct minus global, in percentage points. Seed-paired training differences do not establish statistical significance.

| Dataset | Frozen attribute mAP delta | Adapted attribute mAP delta |
|---|---:|---:|
| Fashion | +3.23 | +11.28 ± 0.42 |
| CUB | +1.44 | +3.03 ± 0.05 |
| DTD | -1.44 | +0.01 ± 0.23 |

## Full-data per-attribute mAP

| Dataset | Model | Attribute | mAP (%) |
|---|---|---|---:|
| Fashion | OmniJev raw | color | 65.64 ± 0.63 |
| Fashion | OmniJev raw | article_type | 86.41 ± 0.06 |
| Fashion | OmniJev raw | audience | 94.26 ± 0.05 |
| Fashion | Qwen3.5 base raw | color | 38.98 ± 4.06 |
| Fashion | Qwen3.5 base raw | article_type | 83.79 ± 0.67 |
| Fashion | Qwen3.5 base raw | audience | 92.92 ± 0.21 |
| Fashion | FLAIR | color | 58.58 ± 0.62 |
| Fashion | FLAIR | article_type | 74.68 ± 0.56 |
| Fashion | FLAIR | audience | 90.22 ± 0.53 |
| Fashion | CLIP B/32 | color | 41.72 ± 2.08 |
| Fashion | CLIP B/32 | article_type | 77.68 ± 2.01 |
| Fashion | CLIP B/32 | audience | 88.27 ± 1.83 |
| Fashion | SigLIP2 B/16 | color | 46.68 ± 0.32 |
| Fashion | SigLIP2 B/16 | article_type | 79.28 ± 0.74 |
| Fashion | SigLIP2 B/16 | audience | 87.91 ± 0.87 |
| Fashion | DINOv2 Small | color | 34.02 ± 0.72 |
| Fashion | DINOv2 Small | article_type | 81.32 ± 0.95 |
| Fashion | DINOv2 Small | audience | 86.98 ± 0.70 |
| Fashion | Qwen3-VL API | color | 46.48 ± 0.89 |
| Fashion | Qwen3-VL API | article_type | 72.84 ± 0.76 |
| Fashion | Qwen3-VL API | audience | 86.56 ± 0.76 |
| Fashion | Qwen3-VL API + attribute instruct | color | 64.51 ± 1.01 |
| Fashion | Qwen3-VL API + attribute instruct | article_type | 82.91 ± 0.62 |
| Fashion | Qwen3-VL API + attribute instruct | audience | 92.30 ± 0.29 |
| Fashion | Tongyi Vision Plus 2026-03-06 | color | 50.08 ± 1.32 |
| Fashion | Tongyi Vision Plus 2026-03-06 | article_type | 76.39 ± 1.18 |
| Fashion | Tongyi Vision Plus 2026-03-06 | audience | 89.88 ± 0.44 |
| CUB | OmniJev raw | primary_color | 56.90 ± 0.04 |
| CUB | OmniJev raw | bill_shape | 40.64 ± 0.24 |
| CUB | OmniJev raw | wing_shape | 35.55 ± 0.06 |
| CUB | Qwen3.5 base raw | primary_color | 50.69 ± 0.27 |
| CUB | Qwen3.5 base raw | bill_shape | 37.16 ± 0.03 |
| CUB | Qwen3.5 base raw | wing_shape | 35.20 ± 0.08 |
| CUB | FLAIR | primary_color | 56.25 ± 0.45 |
| CUB | FLAIR | bill_shape | 38.78 ± 0.40 |
| CUB | FLAIR | wing_shape | 35.20 ± 0.05 |
| CUB | CLIP B/32 | primary_color | 53.88 ± 0.18 |
| CUB | CLIP B/32 | bill_shape | 38.62 ± 0.33 |
| CUB | CLIP B/32 | wing_shape | 34.93 ± 0.13 |
| CUB | SigLIP2 B/16 | primary_color | 52.41 ± 0.42 |
| CUB | SigLIP2 B/16 | bill_shape | 44.10 ± 0.27 |
| CUB | SigLIP2 B/16 | wing_shape | 35.02 ± 0.07 |
| CUB | DINOv2 Small | primary_color | 53.09 ± 0.20 |
| CUB | DINOv2 Small | bill_shape | 43.25 ± 0.24 |
| CUB | DINOv2 Small | wing_shape | 35.06 ± 0.06 |
| CUB | Qwen3-VL API | primary_color | 55.06 ± 0.14 |
| CUB | Qwen3-VL API | bill_shape | 42.93 ± 0.18 |
| CUB | Qwen3-VL API | wing_shape | 35.07 ± 0.16 |
| CUB | Qwen3-VL API + attribute instruct | primary_color | 58.69 ± 0.09 |
| CUB | Qwen3-VL API + attribute instruct | bill_shape | 47.72 ± 0.20 |
| CUB | Qwen3-VL API + attribute instruct | wing_shape | 35.72 ± 0.06 |
| CUB | Tongyi Vision Plus 2026-03-06 | primary_color | 53.06 ± 0.27 |
| CUB | Tongyi Vision Plus 2026-03-06 | bill_shape | 44.30 ± 0.23 |
| CUB | Tongyi Vision Plus 2026-03-06 | wing_shape | 35.01 ± 0.20 |
| DTD | OmniJev raw | texture | 49.37 ± 0.25 |
| DTD | Qwen3.5 base raw | texture | 49.94 ± 0.63 |
| DTD | FLAIR | texture | 50.88 ± 0.35 |
| DTD | CLIP B/32 | texture | 45.95 ± 0.50 |
| DTD | SigLIP2 B/16 | texture | 62.94 ± 0.42 |
| DTD | DINOv2 Small | texture | 53.83 ± 0.22 |
| DTD | Qwen3-VL API | texture | 65.42 ± 0.14 |
| DTD | Qwen3-VL API + attribute instruct | texture | 65.43 ± 0.17 |
| DTD | Tongyi Vision Plus 2026-03-06 | texture | 65.95 ± 0.09 |

## API accounting

Usage is measured from successful cached batches. Retries or interrupted uncached requests may additionally be billed; the provider invoice is authoritative. API latency was measured during overlapping tasks and is not a hardware-normalized speed benchmark.

| API backend | Requests | Total tokens | Extra attempts |
|---|---:|---:|---:|
| Qwen3-VL API | 3534 | 4020366 | 0 |
| Qwen3-VL API + attribute instruct | 9196 | 10516761 | 0 |
| Tongyi Vision Plus 2026-03-06 | 3534 | 11389989 | 0 |

114/162 new selected checkpoints are best at the 1,000-step budget boundary. This is a fixed-budget comparison, not a convergence claim.

CUB excludes missing attributes and uses observed positive-set overlap. DTD is split 1 attribute retrieval, not the official ten-split classification benchmark. All test query coverage is retained in test_results.json. Qwen service aliases cannot be pinned to open weights; cached vectors, dates and parameters identify the measured service.
