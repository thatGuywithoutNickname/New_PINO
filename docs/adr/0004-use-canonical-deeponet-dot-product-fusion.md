---
status: accepted
---

# Use canonical DeepONet dot-product fusion

The baseline uses a branch MLP for the five operating-condition and material inputs and a trunk MLP for the normalized `(x, z)` element-point coordinates. Both emit vectors of the same latent dimension, and AEPS at a point is their dot product plus one learned scalar bias. A concatenation decoder and independent per-element output heads are excluded so the baseline remains a genuine shared neural operator over spatial queries rather than a multi-output condition regressor.

The shared latent dimension is fixed at 16. A pre-implementation spectrum inspection found that approximately 5–6 singular components capture 99.99% of the observed AEPS-field energy, so 16 provides margin for learned nonlinear spatial bases without using an unnecessarily large latent rank.

Both MLPs use `tanh` after every hidden layer and a linear final projection into the latent space. No activation is applied to either latent vector before dot-product fusion. This smooth activation is suitable for the normalized inputs.

The branch architecture is `5 → 32 → 64 → 32 → 16`, and the trunk architecture is `2 → 32 → 64 → 32 → 16`. This locked tapered profile contains 9,729 trainable parameters including the fusion bias, providing a central nonlinear expansion while keeping capacity proportionate to the small dataset.

The fused scalar is the final prediction: no positivity activation, absolute value, squaring, or clamping is applied. Negative predictions remain unchanged in evaluation and are reported as diagnostics, keeping the data-only baseline transparent rather than imposing or concealing a physical constraint.
