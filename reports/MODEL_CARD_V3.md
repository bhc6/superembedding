# SuperEmbedding v3 multi-domain heads

## Scope

Fixed-direction image memory embeddings for fashion catalog attributes, bird attributes, and surface textures. A gallery is encoded once for a chosen attribute basis. At retrieval time, the stored slot for an attribute, a concatenation of selected slots, or a global vector can be compared without running the image encoder again. New arbitrary attribute directions require re-encoding and are not supported by a checkpoint trained for another basis.

The three domains have separate heads. This is not a jointly trained universal multi-domain model. It is an architecture and data-scaling study, not a claim of causal decision-evidence transfer or new state-of-the-art performance.

## Model families

- OmniJev: Qwen/Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17`, with tinnel123/OmniJev-0.8B revision `3c9aa908f3c9d3f3ccf16bb281ee705d9e98252f`. Official OmniJev code revision `14dbec4f71e194852c8d7b88ab36ef639493f400`.
- Base ablation: same network and special-token embeddings, with OmniJev LoRA disabled. This is not the untouched base model or an official Qwen embedding model.
- FLAIR: official `xiaorui638/flair` merged30m checkpoint, revision `0a306fd7060ba3076da3c9ee75b569e6aabea6c6`. Generic fixed attribute directions serve as text queries to its visual attention pooler. This uses the official model under a custom image-to-image retrieval protocol.

All backbones are frozen in v3. The head/gate parameters are trained. The earlier 100-update real-image LoRA experiment remains in the v0.1 release and did not improve validation quality; it is not counted as a v3 result.

## Architecture

Per direction, OmniJev independently branches from an image/instruction prefix and returns option hidden vector `u_k` and prefix/question vector `zq`. Language-model candidate probability features are disabled.

Raw slot: `normalize(Linear(LayerNorm(u_k)))`.

Gated slot: `normalize(Linear(LayerNorm(opt(u_k) * tanh(que(zq)))))`, with official decision-head MLPs either frozen or trainable. Residual mode adds an independent raw projection before normalization.

All slots are 256-dimensional. The global embedding is either normalized concatenation, a learned square linear projection followed by normalization, or a compressed projection. Fashion/CUB have three slots and global dimensions 768 or 256; DTD has one slot and global dimensions 256 or 128. The bias-free output projection changes the cosine metric through `WᵀW`; it is not required simply to concatenate vectors.

Keep slots for attribute-selective retrieval. A compressed global vector alone does not retain an independently addressable vector for every attribute.

## Training data

| Domain | Total after exact deduplication | Train | Validation | Test | Attributes |
|---|---:|---:|---:|---:|---|
| Fashion expanded | 10,853 | 7,894 | 1,184 | 1,775 | color, article type, catalog audience |
| CUB-200-2011 | 11,787 | 4,793 | 1,200 | 5,794 | primary color, bill shape, wing shape |
| DTD split 1 | 5,623 | 1,874 | 1,872 | 1,877 | key texture; joint texture annotations for secondary evaluation |

Fashion catalog audience is a supplied product label, not a demographic inference about a person. All previously inspected pilot images and their detected duplicate components are restricted to the new training split. CUB preserves the official test membership; validation is drawn from official train. DTD uses official split 1 with duplicate removal, so counts differ from the unfiltered benchmark.

CUB uses all positive attribute values with human certainty ≥3. Missing high-confidence positive labels are treated as unobserved. Only pairs with observed values for all requested attributes enter that task. Overlap-based relevance does not establish that annotations exhaust every attribute present. Wing-shape coverage is much lower than color/bill-shape coverage.

The 25%/50%/100% training subsets are nested, with approximately preserved first-attribute label prevalence. Each scale starts from its own seed initialization and uses the same number of optimizer updates. This is a controlled data-availability experiment, not continual warm-start learning and not equal-epoch training.

## Training and selection

Protocol: 1,000 updates, 64 triplets per loss, margin .2, AdamW weight decay .01, gradient norm clipping at 1. Two learning rates (1e-4 and 1e-3), three seeds (17/42/91). Attribute-only and joint-loss methods have different training compute; joint methods use an additional batch of 64 joint triplets. The output-projection comparison holds this joint-loss budget constant.

Checkpoint selection uses validation every 250 updates including the untrained step 0. One learning rate per dataset/backend/scale/architecture is selected by three-seed mean validation score, before test. A step-0 selection is retained and must not be represented as a training gain. The complete matrix is 432 training runs; 216 checkpoints are selected across learning rates for test evaluation. Full grids and selections live under `artifacts/results/v3` in the completed release.

OmniJev/base image encoding uses max_pixels=65,536 and batches of up to four images with identical prefix/token/grid layouts. Image and option states stay independent. BF16 batched computation has small differences from official serial computation; 16 training-image comparisons per backend/domain check cosine similarity ≥0.999. Feature caches for the main study use the same batched implementation. FLAIR uses its official 224px preprocessing.

## Evaluation and limitations

The released OmniJev/base single-image inference entry point repeats a single image into four independent rows to match training's BF16 batch shape. A batch-1 path showed amplified rounding after the base projection and failed the initial fresh-inference check; matching batch geometry resolved that discrepancy without changing training or test features. Actual fresh-image reload differences are recorded in `fresh_inference_verification.json`.

Primary metrics are mean average precision for each attribute (macro average across attributes) and all-attribute intersection. Self matches and pairs with missing annotations are excluded; query coverage is reported. R@1 and R@5 mean at least one positive in the first 1/5 retrieved candidates. DTD joint-texture overlap is a separate secondary metric. These scores must not be presented as CUB species accuracy or the official DTD ten-split classification score.

Mean ± sample standard deviation covers three training seeds. It is not a confidence interval over datasets, a statistical significance claim, or evidence of consistent superiority. Backbones, pretraining, input resolution, and trainable parameter counts differ across OmniJev and FLAIR. Only two learning rates and a finite update budget are explored. Images may overlap with unknown upstream pretraining corpora.

No causal intervention test, unseen-composition protocol, arbitrary new-attribute transfer, video/audio benchmark, or text-query parser is included. Within-domain image retrieval does not by itself prove useful general-purpose multimodal memory. See the generated `reports/RESULTS_V3.md` and Chinese conclusions for measured outcomes and negative results.

## Files and restart

Each dataset's `*_all_heads.zip` contains every best head and validation record, including both learning rates. `*_<backend>_resume.zip` contains every final `last.pt` with model, optimizer, RNG, log accumulators, data/feature fingerprint, and step. Source images and upstream base-model weights are not included. `SHA256SUMS.json` authenticates the released archives by hash; it is not a cryptographic signature.

Run `scripts/research_suite.py` with the original configuration and prepared feature cache to resume. Moving the cache to another directory is supported when its content and pinned model identities match. Bitwise recovery requires the same feature values, compatible software, and hardware. The best head is independently loadable; optimizer state is not needed for inference. A best head's domain, backend, and fixed attribute order must match its encoder.

## License and data rights

Project code and contributed head parameters are offered under Apache-2.0, subject to upstream notices. This does not grant rights in third-party images or waive applicable dataset restrictions. CUB images are provided for non-commercial research and educational use; DTD is provided for research purposes. Treat the CUB/DTD-trained artifacts as research artifacts; this project does not assert commercial-use clearance. The original Fashion Kaggle listing reports MIT; the HF mirror lacks a license field. No source images are redistributed. See `sources/research_datasets_v3.json` and `NOTICE`.
