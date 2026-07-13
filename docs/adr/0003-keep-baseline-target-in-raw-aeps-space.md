---
status: accepted
---

# Keep the baseline target in raw AEPS space

The canonical baseline predicts untransformed, dimensionless AEPS and minimizes the equally weighted mean squared error over simulation cases and their 48 element points. No logarithmic transform, target standardization, clipping, or hotspot, case, or element weighting is applied. Although the observed AEPS values span roughly six orders of magnitude, a log transform was rejected because it would change the baseline from absolute physical-space accuracy toward relative-error weighting and introduce a consequential offset choice. Any log-target experiment must be identified as a separate sensitivity ablation rather than as the canonical baseline.

Each training item is one complete simulation case with its 48-point AEPS field. Mini-batches contain whole cases, and the loss is the mean over the resulting `batch × 48` prediction matrix. Element points are never sampled or shuffled independently, preserving the case-level split and the agreed equal weighting.

Training uses shuffled mini-batches of 32 complete cases. The final smaller batch is retained, yielding eight optimizer updates per epoch for the 246-case training partition. Validation and test evaluation use their complete, unshuffled partitions.
