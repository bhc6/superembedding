# SuperEmbedding model cards

Current release: 28,263 images, three domains, 432 frozen-backbone head training runs, and 216 validation-selected test evaluations.

- [V3 architecture, training, evaluation, data rights, and checkpoint usage](reports/MODEL_CARD_V3.md)
- [Full results](reports/RESULTS_V3.md)
- [Chinese conclusions and limitations](reports/CONCLUSIONS_V3.zh.md)
- [Reproduction instructions](reports/REPRODUCING_V3.md)
- [Preserved v0.1 Fashion2400 model card](reports/MODEL_CARD_PILOT_V01.md)

V3 uses separate heads for Fashion, CUB attributes, and DTD textures. Backbones are frozen. Original images and upstream weights are downloaded separately. Project licensing does not waive dataset restrictions or assert commercial clearance.

All 432 exported heads exactly reproduce stored validation metrics. Training recovery verifies model, optimizer, RNG, and log accumulators. Fresh image inference is separately checked across all domains/backends plus compressed OmniJev heads; see `artifacts/results/v3/fresh_inference_verification.json`.
