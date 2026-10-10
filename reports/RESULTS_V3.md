# V3 multi-domain test results

All learning rates and checkpoints selected on validation before test. Mean ± sample SD over seeds 17/42/91. Metrics are percentages. DTD is official split 1 image retrieval, not 10-split classification.

| Dataset | Backend | Architecture | Train N | Attribute mAP | Joint mAP | Attribute R@1 | Global dim | Best steps |
|---|---|---|---:|---:|---:|---:|---:|---|
| CUB attributes | base | raw | 4793 | 41.02 ± 0.06 | 8.18 ± 0.10 | 50.21 ± 0.27 | 768 | [750, 1000, 1000] |
| CUB attributes | flair | joint_concat | 4793 | 43.62 ± 0.04 | 10.24 ± 0.02 | 54.31 ± 0.10 | 768 | [1000, 1000, 750] |
| CUB attributes | flair | joint_out_equal | 4793 | 43.54 ± 0.08 | 9.90 ± 0.05 | 53.98 ± 0.45 | 768 | [1000, 1000, 750] |
| CUB attributes | flair | joint_out_small | 4793 | 43.53 ± 0.10 | 9.90 ± 0.06 | 54.07 ± 0.42 | 256 | [1000, 1000, 750] |
| CUB attributes | flair | raw | 4793 | 43.41 ± 0.03 | 9.98 ± 0.06 | 53.84 ± 0.33 | 768 | [1000, 1000, 1000] |
| CUB attributes | omnijev | gated_frozen | 4793 | 36.40 ± 0.08 | 6.08 ± 0.05 | 40.64 ± 0.39 | 768 | [1000, 1000, 1000] |
| CUB attributes | omnijev | gated_tuned | 4793 | 40.53 ± 1.04 | 7.85 ± 0.43 | 44.51 ± 0.51 | 768 | [1000, 1000, 1000] |
| CUB attributes | omnijev | joint_concat | 4793 | 44.64 ± 0.25 | 9.83 ± 0.10 | 52.57 ± 0.47 | 768 | [1000, 1000, 1000] |
| CUB attributes | omnijev | joint_out_equal | 4793 | 44.34 ± 0.24 | 9.84 ± 0.02 | 53.32 ± 0.20 | 768 | [1000, 1000, 1000] |
| CUB attributes | omnijev | joint_out_small | 4793 | 44.26 ± 0.27 | 9.81 ± 0.06 | 53.43 ± 0.22 | 256 | [1000, 1000, 1000] |
| CUB attributes | omnijev | raw | 4793 | 44.36 ± 0.10 | 9.60 ± 0.07 | 53.23 ± 0.34 | 768 | [1000, 1000, 1000] |
| CUB attributes | omnijev | residual | 4793 | 44.12 ± 0.10 | 9.49 ± 0.06 | 53.20 ± 0.46 | 768 | [1000, 1000, 1000] |
| DTD textures | base | raw | 1874 | 49.94 ± 0.63 | 49.94 ± 0.63 | 70.41 ± 0.50 | 256 | [1000, 1000, 1000] |
| DTD textures | flair | joint_concat | 1874 | 51.92 ± 0.34 | 51.92 ± 0.33 | 71.85 ± 0.11 | 256 | [1000, 750, 1000] |
| DTD textures | flair | joint_out_equal | 1874 | 52.07 ± 0.11 | 49.79 ± 0.46 | 72.08 ± 0.77 | 256 | [1000, 750, 1000] |
| DTD textures | flair | joint_out_small | 1874 | 52.10 ± 0.17 | 50.09 ± 0.31 | 72.92 ± 0.26 | 128 | [1000, 1000, 1000] |
| DTD textures | flair | raw | 1874 | 50.88 ± 0.35 | 50.88 ± 0.35 | 71.18 ± 0.27 | 256 | [1000, 750, 1000] |
| DTD textures | omnijev | gated_frozen | 1874 | 8.13 ± 0.18 | 8.13 ± 0.18 | 16.94 ± 0.38 | 256 | [1000, 1000, 1000] |
| DTD textures | omnijev | gated_tuned | 1874 | 26.60 ± 1.34 | 26.60 ± 1.34 | 39.78 ± 2.33 | 256 | [1000, 1000, 1000] |
| DTD textures | omnijev | joint_concat | 1874 | 51.24 ± 0.48 | 51.24 ± 0.48 | 69.14 ± 0.49 | 256 | [750, 1000, 1000] |
| DTD textures | omnijev | joint_out_equal | 1874 | 50.58 ± 0.19 | 50.75 ± 0.31 | 70.70 ± 0.24 | 256 | [1000, 1000, 1000] |
| DTD textures | omnijev | joint_out_small | 1874 | 50.18 ± 0.17 | 50.26 ± 0.11 | 70.56 ± 0.69 | 128 | [1000, 1000, 1000] |
| DTD textures | omnijev | raw | 1874 | 49.37 ± 0.25 | 49.37 ± 0.25 | 69.72 ± 0.44 | 256 | [1000, 1000, 1000] |
| DTD textures | omnijev | residual | 1874 | 49.11 ± 0.45 | 49.11 ± 0.45 | 69.14 ± 0.67 | 256 | [1000, 1000, 1000] |
| Fashion | base | raw | 7894 | 71.89 ± 1.32 | 36.86 ± 2.62 | 85.28 ± 0.65 | 768 | [1000, 1000, 750] |
| Fashion | flair | joint_concat | 7894 | 74.68 ± 0.05 | 57.63 ± 0.21 | 87.40 ± 0.16 | 768 | [1000, 1000, 750] |
| Fashion | flair | joint_out_equal | 7894 | 74.67 ± 0.19 | 57.44 ± 0.11 | 87.36 ± 0.22 | 768 | [750, 750, 750] |
| Fashion | flair | joint_out_small | 7894 | 74.49 ± 0.34 | 56.90 ± 0.17 | 87.24 ± 0.45 | 256 | [1000, 750, 750] |
| Fashion | flair | raw | 7894 | 74.50 ± 0.39 | 55.10 ± 0.28 | 87.15 ± 0.20 | 768 | [1000, 1000, 1000] |
| Fashion | omnijev | gated_frozen | 7894 | 56.02 ± 0.39 | 23.42 ± 0.40 | 69.59 ± 0.22 | 768 | [1000, 1000, 1000] |
| Fashion | omnijev | gated_tuned | 7894 | 80.08 ± 0.70 | 50.01 ± 1.48 | 83.82 ± 1.16 | 768 | [1000, 1000, 1000] |
| Fashion | omnijev | joint_concat | 7894 | 82.55 ± 0.23 | 57.65 ± 0.20 | 87.27 ± 0.18 | 768 | [1000, 1000, 1000] |
| Fashion | omnijev | joint_out_equal | 7894 | 82.10 ± 0.38 | 58.39 ± 0.66 | 87.87 ± 0.08 | 768 | [1000, 1000, 1000] |
| Fashion | omnijev | joint_out_small | 7894 | 82.15 ± 0.34 | 58.60 ± 0.30 | 88.02 ± 0.18 | 256 | [1000, 1000, 1000] |
| Fashion | omnijev | raw | 7894 | 82.10 ± 0.21 | 54.00 ± 0.65 | 87.93 ± 0.13 | 768 | [1000, 1000, 750] |
| Fashion | omnijev | residual | 7894 | 82.25 ± 0.47 | 55.67 ± 0.98 | 87.25 ± 0.42 | 768 | [1000, 1000, 1000] |

## Data scaling at a fixed optimizer-update budget

| Dataset | Backend | Architecture | Train N | Attribute mAP | Joint mAP |
|---|---|---|---:|---:|---:|
| CUB attributes | base | raw | 1198 | 41.21 ± 0.15 | 8.20 ± 0.13 |
| CUB attributes | base | raw | 2396 | 41.20 ± 0.19 | 8.25 ± 0.12 |
| CUB attributes | base | raw | 4793 | 41.02 ± 0.06 | 8.18 ± 0.10 |
| CUB attributes | flair | raw | 1198 | 42.97 ± 0.10 | 9.67 ± 0.06 |
| CUB attributes | flair | raw | 2396 | 43.31 ± 0.15 | 9.85 ± 0.10 |
| CUB attributes | flair | raw | 4793 | 43.41 ± 0.03 | 9.98 ± 0.06 |
| CUB attributes | omnijev | joint_concat | 1198 | 44.04 ± 0.19 | 9.66 ± 0.09 |
| CUB attributes | omnijev | joint_out_equal | 1198 | 43.75 ± 0.44 | 9.05 ± 0.33 |
| CUB attributes | omnijev | joint_out_small | 1198 | 43.71 ± 0.19 | 9.05 ± 0.07 |
| CUB attributes | omnijev | raw | 1198 | 44.17 ± 0.16 | 9.48 ± 0.05 |
| CUB attributes | omnijev | joint_concat | 2396 | 44.48 ± 0.42 | 9.80 ± 0.09 |
| CUB attributes | omnijev | joint_out_equal | 2396 | 44.47 ± 0.02 | 9.37 ± 0.10 |
| CUB attributes | omnijev | joint_out_small | 2396 | 44.33 ± 0.18 | 9.36 ± 0.07 |
| CUB attributes | omnijev | raw | 2396 | 44.40 ± 0.01 | 9.62 ± 0.01 |
| CUB attributes | omnijev | joint_concat | 4793 | 44.64 ± 0.25 | 9.83 ± 0.10 |
| CUB attributes | omnijev | joint_out_equal | 4793 | 44.34 ± 0.24 | 9.84 ± 0.02 |
| CUB attributes | omnijev | joint_out_small | 4793 | 44.26 ± 0.27 | 9.81 ± 0.06 |
| CUB attributes | omnijev | raw | 4793 | 44.36 ± 0.10 | 9.60 ± 0.07 |
| DTD textures | base | raw | 468 | 47.67 ± 0.76 | 47.67 ± 0.76 |
| DTD textures | base | raw | 937 | 49.34 ± 0.61 | 49.34 ± 0.61 |
| DTD textures | base | raw | 1874 | 49.94 ± 0.63 | 49.94 ± 0.63 |
| DTD textures | flair | raw | 468 | 46.58 ± 0.47 | 46.58 ± 0.47 |
| DTD textures | flair | raw | 937 | 49.21 ± 1.05 | 49.21 ± 1.05 |
| DTD textures | flair | raw | 1874 | 50.88 ± 0.35 | 50.88 ± 0.35 |
| DTD textures | omnijev | joint_concat | 468 | 47.55 ± 0.32 | 47.55 ± 0.32 |
| DTD textures | omnijev | joint_out_equal | 468 | 47.19 ± 0.16 | 46.24 ± 0.46 |
| DTD textures | omnijev | joint_out_small | 468 | 46.98 ± 0.05 | 45.69 ± 0.56 |
| DTD textures | omnijev | raw | 468 | 46.71 ± 0.41 | 46.71 ± 0.41 |
| DTD textures | omnijev | joint_concat | 937 | 49.74 ± 0.04 | 49.74 ± 0.04 |
| DTD textures | omnijev | joint_out_equal | 937 | 49.54 ± 0.53 | 49.47 ± 0.62 |
| DTD textures | omnijev | joint_out_small | 937 | 49.21 ± 0.43 | 48.85 ± 0.55 |
| DTD textures | omnijev | raw | 937 | 48.26 ± 0.71 | 48.26 ± 0.71 |
| DTD textures | omnijev | joint_concat | 1874 | 51.24 ± 0.48 | 51.24 ± 0.48 |
| DTD textures | omnijev | joint_out_equal | 1874 | 50.58 ± 0.19 | 50.75 ± 0.31 |
| DTD textures | omnijev | joint_out_small | 1874 | 50.18 ± 0.17 | 50.26 ± 0.11 |
| DTD textures | omnijev | raw | 1874 | 49.37 ± 0.25 | 49.37 ± 0.25 |
| Fashion | base | raw | 1974 | 72.38 ± 0.37 | 38.46 ± 1.03 |
| Fashion | base | raw | 3947 | 72.62 ± 0.71 | 38.66 ± 1.32 |
| Fashion | base | raw | 7894 | 71.89 ± 1.32 | 36.86 ± 2.62 |
| Fashion | flair | raw | 1974 | 74.65 ± 0.14 | 54.83 ± 0.27 |
| Fashion | flair | raw | 3947 | 74.64 ± 0.24 | 55.15 ± 0.50 |
| Fashion | flair | raw | 7894 | 74.50 ± 0.39 | 55.10 ± 0.28 |
| Fashion | omnijev | joint_concat | 1974 | 82.14 ± 0.12 | 56.68 ± 0.50 |
| Fashion | omnijev | joint_out_equal | 1974 | 82.06 ± 0.14 | 57.66 ± 0.69 |
| Fashion | omnijev | joint_out_small | 1974 | 81.91 ± 0.15 | 57.59 ± 0.66 |
| Fashion | omnijev | raw | 1974 | 81.93 ± 0.23 | 54.70 ± 0.57 |
| Fashion | omnijev | joint_concat | 3947 | 82.39 ± 0.22 | 57.05 ± 0.28 |
| Fashion | omnijev | joint_out_equal | 3947 | 81.74 ± 0.32 | 58.05 ± 0.36 |
| Fashion | omnijev | joint_out_small | 3947 | 81.80 ± 0.30 | 57.98 ± 0.30 |
| Fashion | omnijev | raw | 3947 | 81.83 ± 0.20 | 54.85 ± 0.67 |
| Fashion | omnijev | joint_concat | 7894 | 82.55 ± 0.23 | 57.65 ± 0.20 |
| Fashion | omnijev | joint_out_equal | 7894 | 82.10 ± 0.38 | 58.39 ± 0.66 |
| Fashion | omnijev | joint_out_small | 7894 | 82.15 ± 0.34 | 58.60 ± 0.30 |
| Fashion | omnijev | raw | 7894 | 82.10 ± 0.21 | 54.00 ± 0.65 |

## Output projection paired differences

Differences pair the same training seeds. No significance claim is made from three seeds.

| Dataset | Backend | Comparison | Joint mAP delta (pp) |
|---|---|---|---:|
| Fashion | omnijev | joint_out_equal minus joint_concat | +0.74 ± 0.58 |
| Fashion | omnijev | joint_out_small minus joint_concat | +0.95 ± 0.23 |
| Fashion | flair | joint_out_equal minus joint_concat | -0.19 ± 0.10 |
| Fashion | flair | joint_out_small minus joint_concat | -0.73 ± 0.37 |
| CUB attributes | omnijev | joint_out_equal minus joint_concat | +0.01 ± 0.09 |
| CUB attributes | omnijev | joint_out_small minus joint_concat | -0.02 ± 0.09 |
| CUB attributes | flair | joint_out_equal minus joint_concat | -0.34 ± 0.05 |
| CUB attributes | flair | joint_out_small minus joint_concat | -0.34 ± 0.06 |
| DTD textures | omnijev | joint_out_equal minus joint_concat | -0.49 ± 0.35 |
| DTD textures | omnijev | joint_out_small minus joint_concat | -0.98 ± 0.54 |
| DTD textures | flair | joint_out_equal minus joint_concat | -2.13 ± 0.73 |
| DTD textures | flair | joint_out_small minus joint_concat | -1.83 ± 0.64 |

## Frozen and random-ranking controls

| Dataset | Control | Attribute mAP | Joint mAP |
|---|---|---:|---:|
| Fashion | Uniform random ranking (analytic expectation) | 35.68 | 4.23 |
| CUB attributes | Uniform random ranking (analytic expectation) | 32.15 | 3.99 |
| DTD textures | Uniform random ranking (analytic expectation) | 2.45 | 2.45 |
| Fashion | omnijev frozen raw | 60.80 | 28.47 |
| Fashion | base frozen raw | 57.79 | 23.52 |
| Fashion | flair frozen raw | 58.89 | 32.41 |
| Fashion | flair frozen global | 59.94 | 34.98 |
| CUB attributes | omnijev frozen raw | 39.42 | 7.48 |
| CUB attributes | base frozen raw | 37.68 | 6.96 |
| CUB attributes | flair frozen raw | 39.86 | 8.23 |
| CUB attributes | flair frozen global | 39.54 | 8.04 |
| DTD textures | omnijev frozen raw | 23.33 | 23.33 |
| DTD textures | base frozen raw | 37.52 | 37.52 |
| DTD textures | flair frozen raw | 35.50 | 35.50 |
| DTD textures | flair frozen global | 34.19 | 34.19 |

## Query coverage

| Dataset | Attribute | Queries with positives | Observed queries | All queries |
|---|---|---:|---:|---:|
| Fashion | color | 1775 | 1775 | 1775 |
| Fashion | article_type | 1775 | 1775 | 1775 |
| Fashion | audience | 1775 | 1775 | 1775 |
| CUB attributes | primary_color | 5433 | 5433 | 5794 |
| CUB attributes | bill_shape | 5266 | 5266 | 5794 |
| CUB attributes | wing_shape | 2873 | 2873 | 5794 |
| DTD textures | texture | 1877 | 1877 | 1877 |

Compressed global vectors do not replace slots for attribute-selective retrieval. Backbone and image resolution differ across methods; these are same-data/protocol comparisons, not equal-compute architecture comparisons. CUB missing-attribute pairs are excluded and query coverage is recorded in summary.json.
