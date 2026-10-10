# SuperEmbedding Fashion2400 models

## Intended use

Research on reusable visual attribute memories for fashion catalog images. Images are encoded with three fixed English directions (color, item type, intended catalog audience) and a fixed instruction. Slots enable single-attribute or subset retrieval without re-encoding gallery images. The audience label is supplied by the product catalog; this is not a person's gender inference system.

Base: Qwen/Qwen3.5-0.8B, revision `2fc06364715b967f1860aea9cf38778875588b17`.
Decision initialization: tinnel123/OmniJev-0.8B v1.1, revision `3c9aa908f3c9d3f3ccf16bb281ee705d9e98252f`.
Official code: OmniJev `14dbec4f71e194852c8d7b88ab36ef639493f400`.

## Architecture and files

Raw head: LayerNorm(1024) → Linear(1024,256), per-slot normalization.
Gated head: `P(LN(opt(u) * tanh(que(zq))))`, where both MLPs are copied from OmniJev, either frozen or trained.
Residual head adds a separate raw `P(LN(u))` path before slot normalization.
Global vector is normalized concatenation (768d), or a learned bias-free linear projection to 768/256d followed by normalization. Slots remain 3×256d regardless of global projection.

`all_trained_heads.zip` contains all 54 validation-selected heads, including both learning rates and every seed; the architecture-level LR choices are in the results files. `*_resume_checkpoints.zip` contains the full final cached-head optimizer/RNG/model state for every run. `lora_model_and_resume.zip` contains the validation-selected derived adapter, embedding head, processor/tokenizer dependencies, and best/last trainable-parameter training snapshots. Original base model weights must be downloaded separately at the pinned revision. Checksums are in `SHA256SUMS.json`.

Raw/gated/joint heads under `suite/` require the original OmniJev adapter. `suite_base/` heads were trained with adapter LoRA disabled but OmniJev special-token embeddings retained; they are only a base-network ablation. `suite_flair/` heads require the official FLAIR image features; they cannot be attached to OmniJev.

Local absolute paths inside checkpoints record provenance. `encode_image.py --adapter ... --base ...` provides explicit relocation. When resuming the released LoRA training state on a different computer, use `--relocate-source` and the original input head/adapter/base; hashes still validate identity. Hardware/library changes can prevent bitwise reproducibility even when the experiment is reproducible statistically.

The LoRA package distinguishes `adapter/` + `embedding_head.pt` (best validation) from `final_adapter/` + `final_embedding_head.pt` (100 updates). If validation selects step 0, the selected model is the unchanged starting model, not evidence of successful quality improvement. `export_manifest.json` records this explicitly; the fully trained final weights and resumable state are still included.

In this release validation **did select step 0**: initial macro mAP 80.73%, final 100-update macro mAP 80.56%. The follow-up did not improve validation quality. The selected-model test result therefore equals the seed42 starting model; it is not a LoRA gain. The final 100-update adapter is released for transparency but is not the recommended validation-selected model.

`development_models_fashion600.zip` preserves the earlier 600-image pilot heads and 3-update LoRA smoke model. That pilot predates optimizer/RNG checkpoint support, and its test subset was reused during development; it is not part of the formal comparison.

## Training and evaluation

Data: ashraq/fashion-product-images-small, revision `3859c76db2f6f3d3b9a3863345e3ccdbff75879d`. Product IDs and exact decoded images were deduplicated; dHash distance <=4 also filtered before splitting, including exclusion against the earlier 600-image development subset. The formal subset has 1,462 train / 469 validation / 469 test images, 40 joint attribute strata. Splits, labels and hashes are published without source images.

Cached training: frozen visual-language model, 500 AdamW updates, batch64, margin .2, learning rates 1e-4 and 1e-3, seeds17/42/91. Checkpoints use validation only. Architecture LR is selected using mean validation score across all three seeds. Attribute-only heads use attribute macro mAP; joint-loss variants use mean(attribute macro mAP, three-attribute mAP).

Real-image LoRA follow-up: existing rank32 adapter plus a raw embedding head, 100 updates, one triplet per update, lr1e-5, seed42, validation every25 updates. This is an exploratory one-seed experiment with a different compute budget. Original decision heads are carried for upstream-loader compatibility only and are not validated after embedding fine-tuning.

See [complete measured table](reports/FORMAL_RESULTS_TABLE.md) and machine-readable records under `artifacts/results/`. mAP and R@K are image-to-image retrieval with self excluded. Every formal test query has positives for all reported attribute combinations. Seed standard deviation measures optimizer variability, not uncertainty across datasets.

## Limitations

This is a small, single-domain study. It does not establish novel architecture, general multimodal memory quality, or causal transfer of decision sensitivity. All attribute values and joint combinations are seen in training. dHash is a conservative heuristic and can both remove distinct products and miss semantic duplicates. Pretraining overlap is unknown. No image augmentation, hard query-specific evidence intervention, unseen-composition split, video/audio evaluation or cross-dataset replication is included.

Only two learning rates were searched. Several architectures peak at the final budgeted update, so the comparison is budget-limited and does not establish each architecture's best achievable performance. The real-image LoRA follow-up uses one triplet per update; many sampled triplets already satisfy the margin and have zero loss.

FLAIR comparison uses its official merged30m checkpoint and fixed attribute prompts under our image-to-image task; it is not the authors' original benchmark. Backbones/pretraining and projection parameter counts differ. FLAIR global features are also reported because fixed generic prompts need not improve every dataset. Omni-Attribute lacks a usable released implementation/checkpoint in the inspected repository and is not an empirical comparison here.

## License

Our code and derived heads/adapters are Apache-2.0, with upstream attributions in NOTICE and original model license in each archive. Qwen and OmniJev upstream models are Apache-2.0. External FLAIR checkpoint card declares MIT and is downloaded separately. The upstream Kaggle dataset listing reports MIT; the HF mirror has no license field. See the recorded source verification. No upstream endorsement is implied.
