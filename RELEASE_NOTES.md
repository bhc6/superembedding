# Fashion2400 local research release

54 reproducible cached-feature experiments on an RTX 5070 Laptop 8GB: seven OmniJev embedding-head architectures, three seeds and two learning rates, plus official FLAIR and LoRA-disabled base-network controls. A separate 100-update real-image LoRA follow-up verifies training, interruption/resume, validation selection and export reload.

Main measurements: on 469 held-out images, direct concatenation with an attribute+joint triplet objective obtains 52.60±0.09% three-attribute mAP. A 768d output projection gives 51.93±0.20%; 256d gives 51.97±0.79%. Output projection is not necessary in this experiment; 256d offers a compression tradeoff. Multiplicative gates underperform a raw hidden projection; negative results are included.

Raw projection single-attribute mAP: OmniJev 78.56±0.29%, FLAIR 70.66±0.63%, LoRA-disabled base ablation 67.20±0.17%. FLAIR/raw has comparable joint mAP and higher attribute R@1, so this is not an across-the-board win. These are our fixed-direction image-to-image task measurements, not the original papers' benchmark scores. All values/combinations are seen in training.

Assets include all 54 trained heads, final resumable cached-training states, the derived LoRA model and best/last LoRA training states. Source code, frozen protocol, split manifests, all validation grids, held-out metrics, model card and figures are in the repository. Source product images, third-party code and original base weights are downloaded separately at pinned revisions. See SHA256SUMS.json and the model card for attribution and limitations.

Research prerelease; this study does not establish novelty or causal transfer of decision evidence sensitivity.

The real-image LoRA follow-up did not improve validation: 80.73% initially versus 80.56% at update100; the selected checkpoint is step0. Both the selected starting model and the actually trained final adapter/head are included and clearly labeled, alongside optimizer/RNG states.
