# Baseline training protocol

This document records the current training configuration for the canonical data-driven DeepONet baseline.

## Phase scope

The current phase covers only the design and eventual implementation and evaluation of the strictly data-driven DeepONet baseline.

## Experimental role

The baseline is deliberately a strictly data-driven experimental control. Its only training objective is the raw-AEPS data loss defined below.

## Loss and batching

Each training item is one complete 48-point simulation case. Cases are shuffled each epoch and batched with `batch_size = 32`; the final smaller batch is retained. Validation and test partitions are evaluated as complete, unshuffled batches.

The training dataset begins in the canonical training-case order stored in the split manifest. A dedicated CPU `torch.Generator` is seeded once with the run's training seed and supplies one new case permutation per epoch; it is not reseeded between epochs. Loading uses `shuffle = true`, `drop_last = false`, and `num_workers = 0`. Validation and test cases remain in their canonical manifest order with shuffling disabled.

## Optimizer

The optimizer is Adam with the following fixed configuration:

| Parameter | Value |
| --- | ---: |
| Learning rate | `1e-3` |
| Betas | `(0.9, 0.999)` |
| Epsilon | `1e-8` |
| Weight decay | `0` |

## Non-finite failure policy

Every training step checks that the scalar loss and all populated parameter gradients are finite before the optimizer update, and that all model parameters remain finite after the update. Every validation pass checks all predictions and the resulting validation MSE. The first non-finite training loss, gradient, model parameter, validation prediction, or validation MSE aborts that seed immediately.

The run does not skip the offending batch, clamp or replace a value, lower the learning rate automatically, continue from the invalid state, or substitute another seed. Its preceding completed-epoch recovery snapshot may be retained for diagnosis, but the failed trajectory cannot resume as a canonical run under the unchanged configuration. A numerical failure is not treated as an ordinary interruption.

Because the freeze gate requires all five seeds to complete with finite values, any such failure keeps test evaluation locked. A resulting protocol revision is documented, and all five seeds are rerun under the one common revised configuration.

## Training duration and checkpoint selection

Training has `max_epochs = 3000`. Raw validation AEPS MSE is evaluated after every epoch. The numerically lowest validation-MSE checkpoint is always retained.

The first finite validation MSE initializes two independent references. `best_checkpoint_mse` is updated after any strictly lower finite validation MSE and identifies the checkpoint retained for evaluation. `best_meaningful_mse` is updated only when the current validation MSE is strictly less than `best_meaningful_mse * (1 - 1e-3)`, matching the scheduler's relative-threshold comparison; only such an update resets the early-stopping patience counter.

Training stops after 300 consecutive epochs without meaningful progress, or at the 3000-epoch ceiling. Evaluation restores the checkpoint identified by `best_checkpoint_mse`; the final-epoch state is not used automatically, and test performance never participates in checkpoint selection.

## Learning-rate schedule

Adam's learning rate is controlled by `ReduceLROnPlateau` using raw validation AEPS MSE with `mode = min`, `factor = 0.5`, `patience = 75`, `threshold = 1e-3`, `threshold_mode = rel`, and `min_lr = 1e-6`. The scheduler applies the same relative meaningful-progress threshold through its own serialized internal best value and counter. Its complete state is restored rather than reconstructed from `best_checkpoint_mse` or `best_meaningful_mse`.

## Regularization

No dropout, batch normalization, layer normalization, weight decay, gradient clipping, or other explicit regularization layer, penalty, or gradient modification is used. Capacity control comes from the locked compact architecture, while generalization control comes from grouped validation and early stopping. Large finite gradients are not clipped; non-finite gradients follow the fail-fast policy.

## Parameter initialization

Weights in hidden linear layers use Xavier normal initialization with the `tanh` gain of `5/3`. The final 16-dimensional latent-projection weights use Xavier normal initialization with gain `1`. All branch and trunk biases, including the global DeepONet fusion bias, start at zero. The same initialization rule is applied for every declared random seed.

## Numerical precision

Source values, interpolated material properties, branch-feature normalization statistics, and trunk-coordinate bounds are calculated and stored in `float64`. Normalized branch inputs, normalized trunk coordinates, and raw AEPS targets are cast to `float32` before entering the model.

Model parameters, forward and backward passes, the training MSE, and Adam optimizer state use `float32`. Automatic mixed precision is disabled; `float16`, `bfloat16`, and TensorFloat-32 are not used.

Validation and test predictions are produced by the `float32` model and then cast to `float64` together with their raw targets before calculating MSE, RMSE, checkpoint-selection values, scheduler and early-stopping inputs, hotspot metrics, and reported aggregates. Checkpoints retain `float32` model and optimizer tensors. The complete dtype policy is stored in every run configuration.

## Canonical execution backend

The canonical backend is `cuda:0` on the available NVIDIA GeForce RTX 4070. All five seeded training runs, validation passes, checkpoint selection, and final test evaluation use the same physical GPU and the same Python, NumPy, PyTorch, CUDA, driver, and deterministic configuration. Results produced on different backends are not pooled into one canonical five-run package.

PyTorch deterministic algorithms are enabled in strict mode rather than warning-only mode. TensorFloat-32 and automatic mixed precision remain disabled, cuDNN benchmarking is disabled, and the deterministic cuBLAS workspace configuration is set before CUDA initialization. Encountering an operation without a deterministic CUDA implementation stops the run.

If CUDA cannot satisfy this policy, the canonical backend may be changed to CPU before test evaluation, but every seed must then be trained and evaluated again under one common CPU configuration. Existing CUDA and CPU results remain separate. CPU smoke tests are permitted but do not count as canonical evidence.

Each run records the device identifier and name, driver version, CUDA version, PyTorch build, deterministic flags, cuBLAS workspace configuration, and relevant software versions. Reproducibility is claimed only for the recorded environment; bitwise identity across unrecorded hardware or library changes is not assumed.

## Repeated runs and determinism

The baseline is trained independently with seeds `0`, `1`, `2`, `3`, and `4` on the same locked split and normalization statistics. Each seed controls Python, NumPy, PyTorch, parameter initialization, and data-loader shuffling. Every run follows the strict deterministic-backend policy above.

Each run selects its checkpoint using validation MSE only. All five test results and their mean and standard deviation are reported; no best-test-seed result is presented as the baseline.

The five selected checkpoints are separate baseline realizations rather than one implicit predictor. Runtime use must identify the exact seed-specific checkpoint and preserve its provenance according to the [baseline inference contract](baseline-inference-contract.md). No seed is silently treated as the default or selected because it performed best, and the five predictions are not silently averaged.

## Interrupted-run recovery

After validation, learning-rate scheduling, early-stopping updates, and best-checkpoint handling for each completed epoch, the run atomically replaces one rolling recovery snapshot. The recovery snapshot is distinct from the numerically best validation checkpoint and exists only to continue an interrupted training trajectory.

The snapshot contains the completed epoch, model state, Adam state, complete learning-rate scheduler state, early-stopping counter, `best_checkpoint_mse`, `best_meaningful_mse`, best-checkpoint identity, Python random state, NumPy random state, PyTorch CPU and CUDA random states, and the data-loader generator state.

At resume, the dedicated data-loader generator state is restored before the next epoch's permutation is requested. This reproduces the uninterrupted next-epoch case order; no epoch-number reseeding or reconstructed permutation is permitted.

It is bound to the source-file checksums, split-manifest identity, preprocessing statistics, complete run configuration, numerical-precision policy, execution-backend metadata, and relevant software versions. Resume is rejected if any bound value differs or if the snapshot is incomplete or corrupt. Restoring model weights without the complete bound state is not a continuation of the canonical run; that seed must restart.

An interruption during an epoch discards the partial epoch. Training resumes only from the preceding completed epoch boundary. Every interruption, resume attempt, acceptance or rejection of the snapshot, and restarted seed is recorded in the run history.

## Baseline freeze gate

The baseline must pass a validation-only sanity gate before any test predictions or test metrics are generated. The training-mean AEPS-field predictor is the 48-value vector obtained by taking, separately at each element index, the arithmetic mean of raw AEPS over the 246 training simulation cases. This same condition-independent vector is used as the prediction for every validation case, and its global validation RMSE is computed over the complete `51 × 48` validation field.

All five DeepONet runs must complete with finite training and validation losses and finite validation predictions. At least four of the five seed-specific global validation RMSE values must be strictly lower than the global validation RMSE of the training-mean AEPS-field predictor. The mean global validation RMSE across all five seeds, including any nonpassing seed, must also be strictly lower than the comparator. Passing both requirements is the operational non-collapse check; no additional absolute RMSE threshold is imposed.

A seed that does not beat the comparator is not replaced, rerun under a different seed, dropped, or hidden. If the four-out-of-five majority-and-mean gate passes, all five selected checkpoints proceed to test evaluation and remain in every five-seed aggregate. The nonpassing seed and its validation-gate status are identified explicitly in the frozen package, test report, and per-seed artifacts.

Before the baseline is frozen, the following artifacts must be preserved:

- the source-file SHA-256 checksums and passing preflight report;
- the split manifest;
- the mean and standard deviation for each of the five branch features and the minimum and maximum bound for each of the two trunk coordinates;
- the exact run configuration, numerical-precision and execution-backend policies, and reproducibility metadata;
- the selected checkpoint and training history for every seed;
- the interruption and resume history for every seed;
- validation predictions, the complete per-case validation metric table, each seed's comparator pass/fail status, and the majority-and-mean sanity-gate result.

If the gate fails, diagnosis and any resulting changes use only training and validation evidence, after which all five seeded runs are repeated under one common revised configuration. The locked test partition remains unevaluated. Once the gate passes, the data contract, split, preprocessing, architecture, training protocol, and selected checkpoints are frozen; test evaluation may then be performed, and its results cannot be used to revise or select the baseline.

## Evaluation metrics

Global RMSE over every test case and all 48 element points is the headline metric. Global MSE over the same values is reported alongside it and remains the training-aligned metric.

For each simulation case, let `k = ceil(0.05 × 48) = 3`. Let `H_true` and `H_pred` be the element indices of the three largest ground-truth and predicted AEPS values respectively; ties are resolved by ascending element index. The following case-level metrics are reported:

- **Hotspot relative L2 error:** `||prediction[H_true] - truth[H_true]||_2 / ||truth[H_true]||_2`.
- **Peak-magnitude relative error:** `|max(prediction) - max(truth)| / max(truth)`.
- **Hotspot-location error:** Euclidean distance in millimetres between the coordinate centroids of `H_true` and `H_pred`.
- **Hotspot overlap:** `|H_true ∩ H_pred| / 3`.

Because the scalar output is unconstrained, the negative-prediction fraction and most-negative prediction are retained as validity diagnostics rather than headline performance metrics.

For each seed, hotspot relative L2 error, peak-magnitude relative error, and hotspot-location error are summarized by their median and 90th percentile across the 54 test cases. Values are sorted in ascending order. The ordinary median is used, averaging the two middle values when the case count is even. The 90th percentile uses the empirical nearest-rank rule at one-based sorted rank `ceil(0.9 × N)`; for `N = 54`, this is the 49th value. No interpolation between case values is used for the 90th percentile. Hotspot overlap is summarized by its mean and by the proportion of test cases with perfect overlap (`1.0`). The complete per-case metric table is preserved.

Across the five seeds, the mean and sample standard deviation of every seed-level summary are reported. The cross-seed standard deviation uses `ddof = 1`, calculated as `sqrt(sum((v_i - mean(v))^2) / (5 - 1))`. Global RMSE and MSE are likewise reported per seed and as five-seed mean and sample standard deviation using the same convention. This reporting convention is distinct from the `ddof = 0` population standard deviation used to normalize branch features.
