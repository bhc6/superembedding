# Formal local comparison — Fashion2400

mAP in percent; ± is sample standard deviation across training seeds 17/42/91. Same 469 held-out queries/gallery images; self excluded. These are local image-to-image attribute retrieval measurements, not the original papers’ benchmark results.

| Model | Attribute mAP | Two-attribute mAP | Three-attribute mAP | Selected LR | Global dim |
|---|---:|---:|---:|---:|---:|
| omnijev frozen | 53.34 | 34.94 | 23.46 | — | — |
| omnijev/gated_frozen | 45.26 ± 0.48 | 26.77 ± 0.60 | 16.25 ± 0.56 | 0.001 | 768 |
| omnijev/gated_tuned | 72.12 ± 1.31 | 51.34 ± 2.60 | 36.90 ± 3.57 | 0.001 | 768 |
| omnijev/joint_concat | 78.94 ± 0.15 | 62.41 ± 0.11 | 52.60 ± 0.09 | 0.001 | 768 |
| omnijev/joint_out256 | 78.98 ± 0.37 | 61.85 ± 0.40 | 51.97 ± 0.79 | 0.001 | 256 |
| omnijev/joint_out768 | 78.02 ± 0.02 | 59.67 ± 0.09 | 51.93 ± 0.20 | 0.0001 | 768 |
| omnijev/raw | 78.56 ± 0.29 | 61.55 ± 0.62 | 50.49 ± 0.64 | 0.001 | 768 |
| omnijev/residual | 78.96 ± 0.12 | 61.78 ± 0.38 | 51.02 ± 0.24 | 0.001 | 768 |
| base frozen | 51.90 | 33.97 | 21.81 | — | — |
| base/raw | 67.20 ± 0.17 | 44.02 ± 0.37 | 27.77 ± 0.74 | 0.0001 | 768 |
| flair frozen | 53.11 | 37.75 | 29.27 | — | — |
| flair global frozen | 54.87 | 40.26 | 32.09 | — | 512 |
| flair/raw | 70.66 ± 0.63 | 56.66 ± 0.73 | 50.10 ± 0.72 | 0.001 | 768 |
| LoRA validation-selected step 0 (1 seed) | 78.43 | 61.12 | 50.08 | 1e-5 | 768 |
