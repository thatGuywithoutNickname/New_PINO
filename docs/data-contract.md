# Dataset contract

## Canonical source location

The repository-local `data/` directory under `New_PINO` is the sole canonical source for baseline training, validation, test evaluation, and runtime coordinate binding. Baseline code must not read, search for, compare against, or fall back to `E:\study\DA TU Dresden\operator\data` or any other external copy.

An external Ansys-generated dataset is upstream provenance only. A revision affects the baseline only after it is deliberately imported into the repository-local `data/` directory, receives a new source-checksum set, and passes the complete preflight as a new dataset version. Existing split manifests, preprocessing statistics, checkpoints, and reports remain bound to their original repository-local source checksums.

## Fail-fast source preflight

Preflight validation runs before split generation, normalization, or model construction. A violation stops the run with the file, rule, and offending row, column, or value identified. Consumers must not repair a violation by imputing, clipping, dropping, reordering, deduplicating, or otherwise rewriting source values.

`data/combined_training_data.csv` must satisfy all of the following:

- its header is exactly `Temperature`, `Amplitude`, `Youngs_Modulus`, followed in numeric order by `AEPS_Element_1` through `AEPS_Element_48`;
- it contains exactly 351 simulation cases and no duplicate `(Temperature, Amplitude, Youngs_Modulus)` condition;
- every cell is numeric and finite;
- every AEPS value is non-negative, with individual zero values permitted, and every simulation case contains at least one strictly positive AEPS value so an all-zero AEPS field is invalid;
- it contains exactly 117 distinct temperature-amplitude groups; and
- every temperature-amplitude group contains exactly one case at each PCB Young's modulus in `{20, 23.5, 27}` gigapascals and no other modulus.

`data/co_ind.csv` must have exactly two header fields, `x` followed by `z`, contain exactly 48 finite and pairwise-unique coordinate rows, and retain its source row order. Coordinate rows and AEPS columns must never be sorted independently.

The SAC305 table in `data/material_properties.md` must have strictly increasing, pairwise-unique temperature knots, finite positive Young's modulus at every knot, and finite Poisson's ratio strictly between `-1` and `0.5` at every knot. Its temperature range must cover every simulation-case temperature so that no property extrapolation is required.

The SHA-256 checksum of each of the three source files is computed before parsing and stored with the split manifest and every run artifact. A checksum change identifies a different dataset version and requires a new preflight result; it is never accepted silently.

## Element-point ordering

The element index is one-based and stable across every simulation case. For each `i` from 1 through 48:

- `AEPS_Element_i` in `data/combined_training_data.csv` is the AEPS value for element index `i`.
- Data row `i` in `data/co_ind.csv`, excluding its header, contains the corresponding `(x, z)` element-point coordinates.

Both coordinate components are lengths measured in millimetres.

Consumers must preserve this positional mapping. They must not independently sort the AEPS columns or coordinate rows. The element index is a dataset-local ordering key and must not be interpreted as an Ansys element ID.

## Vibration amplitude

`Amplitude` in `data/combined_training_data.csv` is the prescribed peak vibration displacement from the equilibrium position, measured in millimetres. It is neither a peak-to-peak nor an RMS displacement.

## Material-property interpretation

`Youngs_Modulus` in `data/combined_training_data.csv` is the PCB Young's modulus in gigapascals. It is distinct from the temperature-dependent SAC305 solder Young's modulus tabulated in `data/material_properties.md` in pascals.

The SAC305 thermal-expansion coefficient and density are treated as constants carried forward from their populated table entries. Bulk and shear moduli are derived quantities rather than independent material inputs.

### Temperature-dependent property evaluation

SAC305 solder Young's modulus and Poisson's ratio are evaluated at a simulation case's temperature by piecewise-linear interpolation between the surrounding rows of `data/material_properties.md`. Values at table knots are preserved exactly. Solder Young's modulus is converted from pascals to gigapascals before branch-input normalization.

Temperatures outside the material table's inclusive range of `[-60, 150]` degrees Celsius are invalid inputs. Consumers must reject them rather than clamp or extrapolate the material properties.

## Input normalization

The branch tensor has this fixed feature order and unit schema:

| Position | Feature | Unit | Source |
| ---: | --- | --- | --- |
| 1 | Temperature | degrees Celsius | supplied operating condition |
| 2 | Vibration displacement amplitude | millimetres | supplied operating condition |
| 3 | PCB Young's modulus | gigapascals | supplied operating condition |
| 4 | Temperature-evaluated SAC305 Young's modulus | gigapascals | derived internally from temperature |
| 5 | Temperature-evaluated SAC305 Poisson's ratio | dimensionless | derived internally from temperature |

Runtime callers supply only the three operating-condition values. The two SAC305 features are derived according to the material-property rule before the ordered branch tensor is normalized. Feature names, units, order, and matching normalization statistics are stored with the run configuration and checkpoint; a mismatch is rejected rather than reordered implicitly.

Each of the five branch features is standardized independently using the mean and population standard deviation computed from the `N = 246` training simulation cases only. For each feature, preprocessing calculates `mu = sum(x_i) / N` and `sigma = sqrt(sum((x_i - mu)^2) / N)` in `float64`, equivalent to `ddof = 0`. A zero or non-finite `sigma` is a preprocessing error and stops the run. Validation and test data reuse the saved training `mu` and `sigma` values.

The trunk tensor has fixed coordinate order `(x, z)`. Each component `c` is mapped independently using `c_norm = 2 * (c - c_min) / (c_max - c_min) - 1`, where its minimum and maximum are calculated over all 48 canonical element points. Bounds and normalized coordinates are calculated and stored in `float64`, then the normalized coordinates are cast to `float32` before model input according to the numerical-precision policy. A non-finite bound or a zero coordinate range is a preflight error.

The same saved coordinate bounds are reused unchanged for every partition and training seed. The element index is not a model input. The raw AEPS target is not transformed or normalized.

The branch statistics and trunk bounds are versioned with the split manifest and model checkpoint.
