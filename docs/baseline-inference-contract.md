# Baseline inference contract

This document defines the prediction boundary of the frozen data-driven DeepONet baseline. It prevents the runtime interface from implying spatial or operating-condition generalization that the baseline has not been designed to support.

## Predictor identity

The frozen baseline package contains one seed-specific predictor for each declared training seed `0`, `1`, `2`, `3`, and `4`. An official prediction request must identify the exact frozen checkpoint directly or identify a seed that resolves unambiguously to that checkpoint through the frozen package manifest. There is no implicit default seed, cross-seed best-checkpoint selection, or silent averaging across predictors.

Every returned prediction is accompanied by the training seed, checkpoint identity, that seed's validation-comparator pass/fail status, source-file checksum set, split-manifest identity, and complete run-configuration identity. These fields identify the exact baseline realization that produced the AEPS field. A seed that did not individually beat the comparator is never presented without that status, even when the frozen package passed the four-out-of-five majority-and-mean gate.

Runtime loads the checkpoint-bound branch normalization statistics, trunk-coordinate bounds, and material-property evaluation metadata from the frozen package. It never recomputes preprocessing state from current source data. Before prediction, it verifies the repository-local `data/co_ind.csv` and `data/material_properties.md` files against the checkpoint-bound source checksums; either mismatch stops prediction. The bound `data/combined_training_data.csv` checksum remains part of predictor provenance even though target data are not read to produce a runtime prediction.

The five-seed aggregate is a reporting summary rather than an implicit predictor. A predictor that ensembles the five seed-specific outputs is a separate model variant and requires its own predeclared evaluation before it can be used as an official result.

## Spatial output

One accepted operating condition produces exactly one 48-value AEPS field in ascending element-index order. The element points are the canonical `(x, z)` rows from `data/co_ind.csv`, bound to the checkpoint through the saved source-file checksum. The caller does not provide, replace, reorder, subset, or augment the spatial coordinates.

Arbitrary-coordinate evaluation is unsupported even though the internal DeepONet trunk consumes coordinates. The available evidence covers only the fixed 48 element points.

## Supported operating-condition envelope

An input is supported only when all three supplied values are finite and lie inside these inclusive intervals:

| Quantity | Supported interval |
| --- | ---: |
| Temperature | `[-40, 125]` degrees Celsius |
| Vibration displacement amplitude | `[0.2, 0.9]` millimetres |
| PCB Young's modulus | `[20, 27]` gigapascals |

Intermediate values inside all three intervals are treated as interpolation inputs. The temperature-dependent SAC305 Young's modulus and Poisson's ratio are then calculated according to the dataset contract. The wider `[-60, 150]` degrees Celsius range of the material-property table does not enlarge the learned model's supported temperature domain.

## Rejection behavior

An unsupported input stops prediction with an explicit error identifying the offending value and supported interval. The runtime does not clamp, substitute, silently extrapolate, or emit an official baseline prediction with a warning. A required canonical source-file checksum mismatch likewise stops prediction.

## Official evaluation

Official validation and test metrics are generated only for simulation cases assigned by the saved split manifest. Ad hoc in-envelope predictions are not added to those partitions or reported as held-out evidence.
