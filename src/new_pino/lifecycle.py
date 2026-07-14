"""The single public lifecycle boundary, starting with fixture prediction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from typing import Any, Mapping

from .preparation import (
    PreparedDataArtifact,
    _AEPS_ELEMENT_ORDER,
    _BRANCH_FEATURE_ORDER,
    _BRANCH_FEATURE_UNITS,
    _DTYPE_POLICY,
    _RUNTIME_FEATURE_ORDER,
    _TRUNK_FEATURE_ORDER,
    _TRUNK_FEATURE_UNITS,
    _feature_schema_content_identity,
    _interpolate_material_properties,
    _normalize_coordinate,
    _preprocessing_content_identity,
    _source_content_identity,
    prepare_sources,
    _unit_schema_content_identity,
)
from .reporting import (
    EvaluationReport,
    _evaluate_fixture,
)
from .training import (
    CpuSmokeTrainingResult,
    _BRANCH_WIDTHS,
    _EXPECTED_ARCHITECTURE,
    _TRUNK_WIDTHS,
    train_cpu_smoke,
)


class PredictionContractError(ValueError):
    """A prediction request or bound artifact violates the public contract."""


@dataclass(frozen=True)
class OperatingCondition:
    temperature_c: float
    vibration_displacement_amplitude_mm: float
    pcb_youngs_modulus_gpa: float


@dataclass(frozen=True)
class PredictorSelector:
    seed: int | None = None
    checkpoint_identity: str | None = None


@dataclass(frozen=True)
class PredictionRequest:
    operating_condition: OperatingCondition
    predictor: PredictorSelector
    element_points: object | None = None


@dataclass(frozen=True)
class PredictionProvenance:
    seed: int
    checkpoint_identity: str
    validation_comparator_status: str
    source_checksums: Mapping[str, str]
    source_identity: str
    split_identity: str
    preprocessing_identity: str
    feature_schema_identity: str
    unit_schema_identity: str
    run_configuration_identity: str


@dataclass(frozen=True)
class PredictionResult:
    aeps_field: tuple[float, ...]
    element_indices: tuple[int, ...]
    evidence_status: str
    provenance: PredictionProvenance


@dataclass(frozen=True)
class _SourceChecksums:
    training_data: str
    element_points: str
    material_properties: str

    def as_dict(self) -> dict[str, str]:
        return {
            "training_data": self.training_data,
            "element_points": self.element_points,
            "material_properties": self.material_properties,
        }


@dataclass(frozen=True)
class _MaterialMetadata:
    temperature_knots_c: tuple[float, ...]
    youngs_modulus_pa: tuple[float, ...]
    poissons_ratio: tuple[float, ...]


@dataclass(frozen=True)
class _PreprocessingState:
    branch_mean: tuple[float, ...]
    branch_population_std: tuple[float, ...]
    x_bounds_mm: tuple[float, float]
    z_bounds_mm: tuple[float, float]
    element_points_mm: tuple[tuple[float, float], ...]
    normalized_trunk_coordinates: tuple[tuple[float, float], ...]
    material: _MaterialMetadata
    feature_schema_identity: str
    unit_schema_identity: str
    content_identity: str


@dataclass(frozen=True)
class _FixtureCheckpoint:
    phase: float
    weight_scale: float
    bias_scale: float
    fusion_bias: float


@dataclass(frozen=True)
class _FixturePredictor:
    seed: int
    checkpoint_identity: str
    validation_comparator_status: str
    checkpoint: _FixtureCheckpoint


@dataclass(frozen=True)
class _FixturePackage:
    evidence_status: str
    source_checksums: _SourceChecksums
    source_identity: str
    split_identity: str
    run_configuration_identity: str
    preprocessing: _PreprocessingState
    predictors: tuple[_FixturePredictor, ...]


class BaselineLifecycle:
    """Loads one frozen package and exposes its public baseline operations."""

    def __init__(
        self,
        package: _FixturePackage,
        *,
        package_path: Path,
    ) -> None:
        self._package = package
        self._coordinate_source = package_path / "runtime_sources" / "co_ind.csv"
        self._material_source = (
            package_path / "runtime_sources" / "material_properties.md"
        )

    @classmethod
    def prepare(
        cls,
        repository_root: str | Path,
        *,
        artifact_path: str | Path | None = None,
    ) -> PreparedDataArtifact:
        """Validate and bind the repository-local canonical baseline sources."""

        return prepare_sources(repository_root, artifact_path=artifact_path)

    @classmethod
    def train(
        cls,
        prepared: PreparedDataArtifact,
        *,
        seed: int,
        artifact_directory: str | Path,
        smoke_max_epochs: int,
    ) -> CpuSmokeTrainingResult:
        """Run one explicit, noncanonical DeepONet CPU smoke seed."""

        return train_cpu_smoke(
            prepared,
            seed=seed,
            artifact_directory=artifact_directory,
            smoke_max_epochs=smoke_max_epochs,
        )

    @classmethod
    def from_package(
        cls,
        package_path: str | Path,
    ) -> BaselineLifecycle:
        resolved_package_path = Path(package_path)
        metadata_path = resolved_package_path / "package.json"
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = json.load(stream)
        except (OSError, json.JSONDecodeError) as error:
            raise PredictionContractError(
                f"fixture package metadata {metadata_path} cannot be loaded: {error}"
            ) from error
        package = _parse_fixture_package(metadata)
        return cls(
            package,
            package_path=resolved_package_path,
        )

    def predict(self, request: PredictionRequest) -> PredictionResult:
        self._validate_request(request)
        predictor = self._resolve_predictor(request.predictor)
        self._verify_runtime_sources()

        preprocessing = self._package.preprocessing
        material = preprocessing.material
        condition = request.operating_condition
        solder_modulus_pa, poisson_ratio = _interpolate_material_properties(
            condition.temperature_c,
            material.temperature_knots_c,
            material.youngs_modulus_pa,
            material.poissons_ratio,
        )
        raw_branch = (
            condition.temperature_c,
            condition.vibration_displacement_amplitude_mm,
            condition.pcb_youngs_modulus_gpa,
            solder_modulus_pa / 1_000_000_000.0,
            poisson_ratio,
        )
        branch = tuple(
            _as_float32(value)
            for value in _standardize(
                raw_branch,
                preprocessing.branch_mean,
                preprocessing.branch_population_std,
            )
        )

        recipe = predictor.checkpoint
        branch_latent = _mlp(
            branch,
            _BRANCH_WIDTHS,
            recipe,
            path_offset=0,
        )
        predictions: list[float] = []
        for point in preprocessing.normalized_trunk_coordinates:
            trunk = (
                _as_float32(point[0]),
                _as_float32(point[1]),
            )
            trunk_latent = _mlp(
                trunk,
                _TRUNK_WIDTHS,
                recipe,
                path_offset=4,
            )
            predictions.append(
                sum(left * right for left, right in zip(branch_latent, trunk_latent))
                + recipe.fusion_bias
            )
        if not all(math.isfinite(value) for value in predictions):
            raise PredictionContractError(
                "the selected fixture checkpoint produced a non-finite AEPS value"
            )

        provenance = PredictionProvenance(
            seed=predictor.seed,
            checkpoint_identity=predictor.checkpoint_identity,
            validation_comparator_status=predictor.validation_comparator_status,
            source_checksums=self._package.source_checksums.as_dict(),
            source_identity=self._package.source_identity,
            split_identity=self._package.split_identity,
            preprocessing_identity=preprocessing.content_identity,
            feature_schema_identity=preprocessing.feature_schema_identity,
            unit_schema_identity=preprocessing.unit_schema_identity,
            run_configuration_identity=self._package.run_configuration_identity,
        )
        return PredictionResult(
            aeps_field=tuple(predictions),
            element_indices=tuple(range(1, len(predictions) + 1)),
            evidence_status=self._package.evidence_status,
            provenance=provenance,
        )

    def evaluate(
        self,
        fixture_path: str | Path,
        *,
        artifact_path: str | Path | None = None,
    ) -> EvaluationReport:
        """Produce an auditable noncanonical report from an authorized fixture."""

        self._verify_runtime_sources()
        package = self._package
        return _evaluate_fixture(
            fixture_path,
            source_checksums=package.source_checksums.as_dict(),
            source_identity=package.source_identity,
            split_identity=package.split_identity,
            preprocessing_identity=package.preprocessing.content_identity,
            run_configuration_identity=package.run_configuration_identity,
            predictor_identities=tuple(
                (predictor.seed, predictor.checkpoint_identity)
                for predictor in package.predictors
            ),
            element_points_mm=package.preprocessing.element_points_mm,
            artifact_path=artifact_path,
        )

    def _validate_request(self, request: PredictionRequest) -> None:
        if request.element_points is not None:
            raise PredictionContractError(
                "caller-supplied element points are unsupported; prediction uses "
                "the checkpoint-bound 48-point element-index order"
            )
        condition = request.operating_condition
        _require_supported_value(
            condition.temperature_c,
            name="temperature",
            lower=-40.0,
            upper=125.0,
        )
        _require_supported_value(
            condition.vibration_displacement_amplitude_mm,
            name="vibration displacement amplitude",
            lower=0.2,
            upper=0.9,
        )
        _require_supported_value(
            condition.pcb_youngs_modulus_gpa,
            name="PCB Young's modulus",
            lower=20.0,
            upper=27.0,
        )

    def _resolve_predictor(self, selector: PredictorSelector) -> _FixturePredictor:
        predictors = self._package.predictors
        has_seed = selector.seed is not None
        has_checkpoint = selector.checkpoint_identity is not None
        if not has_seed and not has_checkpoint:
            raise PredictionContractError(
                "an explicit seed or checkpoint identity is required; implicit "
                "defaults, best-seed selection, and multi-seed averaging are unsupported"
            )
        if has_seed and has_checkpoint:
            raise PredictionContractError(
                "provide exactly one predictor identity: seed or checkpoint identity"
            )

        if has_seed:
            matches = [item for item in predictors if item.seed == selector.seed]
            description = f"seed {selector.seed}"
        else:
            matches = [
                item
                for item in predictors
                if item.checkpoint_identity == selector.checkpoint_identity
            ]
            description = f"checkpoint identity {selector.checkpoint_identity!r}"
        if len(matches) != 1:
            raise PredictionContractError(
                f"{description} does not identify one predictor in the frozen package"
            )
        return matches[0]

    def _verify_runtime_sources(self) -> None:
        expected = self._package.source_checksums
        _verify_bound_source(
            binding_name="element-point",
            path=self._coordinate_source,
            expected_checksum=expected.element_points,
        )
        _verify_bound_source(
            binding_name="material-property",
            path=self._material_source,
            expected_checksum=expected.material_properties,
        )
        bound_points = _load_element_points(self._coordinate_source)
        saved_points = self._package.preprocessing.element_points_mm
        if saved_points != bound_points:
            raise PredictionContractError(
                "saved element-point binding does not match the bound coordinate "
                "source order"
            )
        bound_material = _load_material_metadata(self._material_source)
        if self._package.preprocessing.material != bound_material:
            raise PredictionContractError(
                "saved material metadata does not match the bound material-property "
                "source table"
            )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_fixture_package(metadata: object) -> _FixturePackage:
    package = _require_mapping(metadata, "fixture package")
    if package.get("schema_version") != "fixture-prediction-package-v1":
        raise PredictionContractError(
            "fixture package schema_version must be 'fixture-prediction-package-v1'"
        )
    if package.get("canonical") is not False or package.get(
        "evidence_status"
    ) != "noncanonical_fixture":
        raise PredictionContractError(
            "fixture package must be machine-visibly noncanonical with "
            "evidence_status 'noncanonical_fixture'"
        )

    checksums = _require_mapping(
        package.get("source_checksums"), "fixture source_checksums"
    )
    expected_checksum_names = {
        "training_data",
        "element_points",
        "material_properties",
    }
    if set(checksums) != expected_checksum_names:
        raise PredictionContractError(
            "fixture source_checksums must bind training_data, element_points, "
            "and material_properties"
        )
    for name, checksum in checksums.items():
        if not _is_sha256(checksum):
            raise PredictionContractError(
                f"fixture source checksum {name!r} must be 64 lowercase hex characters"
            )

    source_checksums = _SourceChecksums(
        training_data=str(checksums["training_data"]),
        element_points=str(checksums["element_points"]),
        material_properties=str(checksums["material_properties"]),
    )
    source_identity = package.get("source_identity")
    expected_source_identity = _source_content_identity(source_checksums.as_dict())
    if source_identity != expected_source_identity:
        raise PredictionContractError(
            "fixture source_identity must match the bound source checksums"
        )

    split_identity = _require_nonempty_identity(package, "split_identity")
    run_configuration_identity = _require_nonempty_identity(
        package, "run_configuration_identity"
    )

    architecture = _require_mapping(
        package.get("architecture"), "fixture architecture"
    )
    for name, expected_value in _EXPECTED_ARCHITECTURE.items():
        if architecture.get(name) != expected_value:
            raise PredictionContractError(
                f"fixture architecture {name} must be {expected_value!r}"
            )

    preprocessing = _require_mapping(
        package.get("preprocessing"), "fixture preprocessing"
    )
    expected_preprocessing_keys = {
        "schema_version",
        "runtime_feature_order",
        "branch_feature_order",
        "branch_feature_units",
        "branch_mean",
        "branch_population_std",
        "trunk_feature_order",
        "trunk_feature_units",
        "trunk_bounds_mm",
        "element_points_mm",
        "normalized_trunk_coordinates",
        "aeps_element_order",
        "aeps_unit",
        "aeps_transform",
        "aeps_weighting",
        "material",
        "dtype_policy",
        "feature_schema_identity",
        "unit_schema_identity",
        "content_identity",
    }
    if set(preprocessing) != expected_preprocessing_keys:
        raise PredictionContractError(
            "fixture preprocessing must contain the complete checkpoint-bound "
            "schema, state, and identities"
        )
    if preprocessing.get("schema_version") != "baseline-preprocessing-v1":
        raise PredictionContractError(
            "fixture preprocessing schema_version must be 'baseline-preprocessing-v1'"
        )
    if preprocessing.get("runtime_feature_order") != list(_RUNTIME_FEATURE_ORDER):
        raise PredictionContractError(
            "fixture runtime_feature_order must contain only temperature, vibration "
            "displacement amplitude, and PCB Young's modulus in the accepted order"
        )
    if preprocessing.get("branch_feature_order") != list(_BRANCH_FEATURE_ORDER):
        raise PredictionContractError(
            f"fixture branch_feature_order must be {list(_BRANCH_FEATURE_ORDER)!r}"
        )
    if preprocessing.get("branch_feature_units") != list(_BRANCH_FEATURE_UNITS):
        raise PredictionContractError(
            f"fixture branch_feature_units must be {list(_BRANCH_FEATURE_UNITS)!r}"
        )
    if preprocessing.get("trunk_feature_order") != list(_TRUNK_FEATURE_ORDER):
        raise PredictionContractError(
            f"fixture trunk_feature_order must be {list(_TRUNK_FEATURE_ORDER)!r}"
        )
    if preprocessing.get("trunk_feature_units") != list(_TRUNK_FEATURE_UNITS):
        raise PredictionContractError(
            f"fixture trunk_feature_units must be {list(_TRUNK_FEATURE_UNITS)!r}"
        )
    if preprocessing.get("aeps_element_order") != list(_AEPS_ELEMENT_ORDER):
        raise PredictionContractError(
            "fixture aeps_element_order must preserve AEPS_Element_1 through "
            "AEPS_Element_48"
        )
    if (
        preprocessing.get("aeps_unit") != "dimensionless"
        or preprocessing.get("aeps_transform") != "none"
        or preprocessing.get("aeps_weighting") != "none"
    ):
        raise PredictionContractError(
            "fixture AEPS fields must remain raw, dimensionless, untransformed, and "
            "unweighted"
        )
    if preprocessing.get("dtype_policy") != _DTYPE_POLICY:
        raise PredictionContractError(
            f"fixture dtype_policy must be {_DTYPE_POLICY!r}"
        )
    branch_mean = _require_finite_numbers(
        preprocessing.get("branch_mean"),
        count=5,
        label="fixture branch_mean",
    )
    branch_std = _require_finite_numbers(
        preprocessing.get("branch_population_std"),
        count=5,
        label="fixture branch_population_std",
    )
    if any(value <= 0.0 for value in branch_std):
        raise PredictionContractError(
            "fixture branch_population_std values must all be positive"
        )

    bounds = _require_mapping(
        preprocessing.get("trunk_bounds_mm"), "fixture trunk_bounds_mm"
    )
    x_bounds = _require_bounds(bounds.get("x"), "x")
    z_bounds = _require_bounds(bounds.get("z"), "z")
    points = preprocessing.get("element_points_mm")
    if not isinstance(points, list) or len(points) != 48:
        raise PredictionContractError(
            "fixture preprocessing must bind exactly 48 ordered element points"
        )
    parsed_points = tuple(
        _require_element_point(point, index=index)
        for index, point in enumerate(points, start=1)
    )
    if (min(point[0] for point in parsed_points), max(point[0] for point in parsed_points)) != x_bounds:
        raise PredictionContractError(
            "fixture x trunk bounds must match the saved 48 element points"
        )
    if (min(point[1] for point in parsed_points), max(point[1] for point in parsed_points)) != z_bounds:
        raise PredictionContractError(
            "fixture z trunk bounds must match the saved 48 element points"
        )
    normalized_points = preprocessing.get("normalized_trunk_coordinates")
    if not isinstance(normalized_points, list) or len(normalized_points) != 48:
        raise PredictionContractError(
            "fixture preprocessing must store exactly 48 normalized trunk coordinates"
        )
    parsed_normalized_points = tuple(
        _require_element_point(point, index=index)
        for index, point in enumerate(normalized_points, start=1)
    )
    expected_normalized_points = tuple(
        (
            _normalize_coordinate(point[0], x_bounds),
            _normalize_coordinate(point[1], z_bounds),
        )
        for point in parsed_points
    )
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
        for actual_point, expected_point in zip(
            parsed_normalized_points,
            expected_normalized_points,
            strict=True,
        )
        for actual, expected in zip(actual_point, expected_point, strict=True)
    ):
        raise PredictionContractError(
            "fixture normalized trunk coordinates must use the saved x/z bounds "
            "and element-point order"
        )

    material = _require_mapping(
        preprocessing.get("material"), "fixture material metadata"
    )
    raw_knots = material.get("temperature_knots_c")
    if not isinstance(raw_knots, list) or len(raw_knots) < 2:
        raise PredictionContractError(
            "fixture material temperature knots must contain at least two values"
        )
    knots = _require_finite_numbers(
        raw_knots,
        count=len(raw_knots),
        label="fixture material temperature knots",
    )
    moduli = _require_finite_numbers(
        material.get("youngs_modulus_pa"),
        count=len(knots),
        label="fixture material Young's moduli",
    )
    ratios = _require_finite_numbers(
        material.get("poissons_ratio"),
        count=len(knots),
        label="fixture material Poisson's ratios",
    )
    if any(left >= right for left, right in zip(knots, knots[1:])):
        raise PredictionContractError(
            "fixture material temperature knots must be strictly increasing"
        )
    if knots[0] > -40.0 or knots[-1] < 125.0:
        raise PredictionContractError(
            "fixture material metadata must cover the supported temperature envelope"
        )
    if any(value <= 0.0 for value in moduli) or any(
        not -1.0 < value < 0.5 for value in ratios
    ):
        raise PredictionContractError(
            "fixture material metadata contains an invalid modulus or Poisson's ratio"
        )
    if material.get("interpolation") != "piecewise_linear" or material.get(
        "out_of_range"
    ) != "reject":
        raise PredictionContractError(
            "fixture material metadata must declare piecewise-linear interpolation "
            "and out-of-range rejection"
        )

    feature_schema_identity = preprocessing.get("feature_schema_identity")
    expected_feature_schema_identity = _feature_schema_content_identity()
    if feature_schema_identity != expected_feature_schema_identity:
        raise PredictionContractError(
            "fixture feature_schema_identity is incompatible with the saved "
            "feature order"
        )
    unit_schema_identity = preprocessing.get("unit_schema_identity")
    expected_unit_schema_identity = _unit_schema_content_identity()
    if unit_schema_identity != expected_unit_schema_identity:
        raise PredictionContractError(
            "fixture unit_schema_identity is incompatible with the saved units"
        )
    preprocessing_metadata = dict(preprocessing)
    preprocessing_identity = preprocessing_metadata.pop("content_identity")
    expected_preprocessing_identity = _preprocessing_content_identity(
        source_checksums=source_checksums.as_dict(),
        source_identity=str(source_identity),
        split_identity=split_identity,
        preprocessing=preprocessing_metadata,
    )
    if preprocessing_identity != expected_preprocessing_identity:
        raise PredictionContractError(
            "fixture preprocessing content identity does not match its "
            "checkpoint-bound state"
        )

    predictors = package.get("predictors")
    if not isinstance(predictors, list) or not predictors:
        raise PredictionContractError(
            "fixture package must contain at least one explicit predictor"
        )
    seeds: set[int] = set()
    checkpoints: set[str] = set()
    parsed_predictors: list[_FixturePredictor] = []
    for predictor_number, raw_predictor in enumerate(predictors, start=1):
        predictor = _require_mapping(
            raw_predictor, f"fixture predictor {predictor_number}"
        )
        seed = predictor.get("seed")
        checkpoint = predictor.get("checkpoint_identity")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise PredictionContractError(
                f"fixture predictor {predictor_number} seed must be an integer"
            )
        if not isinstance(checkpoint, str) or not checkpoint:
            raise PredictionContractError(
                f"fixture predictor {predictor_number} checkpoint identity is missing"
            )
        if seed in seeds or checkpoint in checkpoints:
            raise PredictionContractError(
                "fixture predictor seeds and checkpoint identities must be unambiguous"
            )
        seeds.add(seed)
        checkpoints.add(checkpoint)
        comparator_status = predictor.get("validation_comparator_status")
        if comparator_status not in {
            "passed",
            "not_passed",
        }:
            raise PredictionContractError(
                f"fixture predictor {predictor_number} validation comparator status "
                "must be 'passed' or 'not_passed'"
            )
        expected_binding = {
            "source_identity": source_identity,
            "split_identity": split_identity,
            "preprocessing_identity": preprocessing_identity,
            "feature_schema_identity": feature_schema_identity,
            "unit_schema_identity": unit_schema_identity,
            "branch_feature_order": list(_BRANCH_FEATURE_ORDER),
        }
        if predictor.get("preprocessing_binding") != expected_binding:
            raise PredictionContractError(
                f"fixture predictor {predictor_number} preprocessing binding is "
                "incompatible with the package identities or feature order"
            )
        checkpoint_state = _require_mapping(
            predictor.get("fixture_checkpoint"),
            f"fixture predictor {predictor_number} checkpoint",
        )
        if checkpoint_state.get("encoding") != "fixture_dense_formula_v1":
            raise PredictionContractError(
                "noncanonical fixture checkpoints must declare "
                "encoding 'fixture_dense_formula_v1'"
            )
        recipe_values = _require_finite_numbers(
            [
                checkpoint_state.get("phase"),
                checkpoint_state.get("weight_scale"),
                checkpoint_state.get("bias_scale"),
                checkpoint_state.get("fusion_bias"),
            ],
            count=4,
            label=f"fixture predictor {predictor_number} checkpoint recipe",
        )
        if recipe_values[1] <= 0.0:
            raise PredictionContractError(
                f"fixture predictor {predictor_number} weight_scale must be positive"
            )
        parsed_predictors.append(
            _FixturePredictor(
                seed=seed,
                checkpoint_identity=checkpoint,
                validation_comparator_status=str(comparator_status),
                checkpoint=_FixtureCheckpoint(
                    phase=recipe_values[0],
                    weight_scale=recipe_values[1],
                    bias_scale=recipe_values[2],
                    fusion_bias=recipe_values[3],
                ),
            )
        )

    return _FixturePackage(
        evidence_status="noncanonical_fixture",
        source_checksums=source_checksums,
        source_identity=str(source_identity),
        split_identity=split_identity,
        run_configuration_identity=run_configuration_identity,
        preprocessing=_PreprocessingState(
            branch_mean=branch_mean,
            branch_population_std=branch_std,
            x_bounds_mm=x_bounds,
            z_bounds_mm=z_bounds,
            element_points_mm=parsed_points,
            normalized_trunk_coordinates=parsed_normalized_points,
            material=_MaterialMetadata(
                temperature_knots_c=knots,
                youngs_modulus_pa=moduli,
                poissons_ratio=ratios,
            ),
            feature_schema_identity=str(feature_schema_identity),
            unit_schema_identity=str(unit_schema_identity),
            content_identity=str(preprocessing_identity),
        ),
        predictors=tuple(parsed_predictors),
    )


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise PredictionContractError(f"{label} must be a JSON object")
    return value


def _require_nonempty_identity(package: Mapping[str, Any], name: str) -> str:
    value = package.get(name)
    if not isinstance(value, str) or not value:
        raise PredictionContractError(f"fixture {name} must be a non-empty string")
    return value


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_finite_numbers(
    value: object, *, count: int, label: str
) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != count:
        raise PredictionContractError(f"{label} must contain exactly {count} values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(item)
        for item in value
    ):
        raise PredictionContractError(f"{label} must contain only finite numbers")
    return tuple(float(item) for item in value)


def _require_bounds(value: object, component: str) -> tuple[float, float]:
    lower, upper = _require_finite_numbers(
        value,
        count=2,
        label=f"fixture {component} trunk bounds",
    )
    if lower >= upper:
        raise PredictionContractError(
            f"fixture {component} trunk bounds must have a nonzero increasing range"
        )
    return lower, upper


def _require_element_point(
    value: object, *, index: int
) -> tuple[float, float]:
    x_coordinate, z_coordinate = _require_finite_numbers(
        value,
        count=2,
        label=f"element point {index}",
    )
    return x_coordinate, z_coordinate


def _verify_bound_source(
    *, binding_name: str, path: Path, expected_checksum: str
) -> None:
    try:
        observed_checksum = _sha256_file(path)
    except OSError as error:
        raise PredictionContractError(
            f"{binding_name} source {path} cannot be read: {error}"
        ) from error
    if observed_checksum != expected_checksum:
        raise PredictionContractError(
            f"{binding_name} source {path} checksum mismatch: expected "
            f"{expected_checksum}, observed {observed_checksum}"
        )


def _load_element_points(path: Path) -> tuple[tuple[float, float], ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.reader(stream))
    except OSError as error:
        raise PredictionContractError(
            f"bound element-point source {path} cannot be read: {error}"
        ) from error
    if not rows or rows[0] != ["x", "z"] or len(rows) != 49:
        raise PredictionContractError(
            f"bound element-point source {path} must contain header x,z and 48 rows"
        )
    try:
        points = tuple((float(row[0]), float(row[1])) for row in rows[1:])
    except (IndexError, ValueError) as error:
        raise PredictionContractError(
            f"bound element-point source {path} contains an invalid coordinate row"
        ) from error
    if any(
        len(row) != 2 or not all(math.isfinite(value) for value in point)
        for row, point in zip(rows[1:], points, strict=True)
    ):
        raise PredictionContractError(
            f"bound element-point source {path} contains an invalid coordinate row"
        )
    if len(set(points)) != len(points):
        raise PredictionContractError(
            f"bound element-point source {path} contains duplicate element points"
        )
    return points


def _load_material_metadata(path: Path) -> _MaterialMetadata:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise PredictionContractError(
            f"bound material-property source {path} cannot be read: {error}"
        ) from error
    rows: list[tuple[float, float, float]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) < 3:
            continue
        try:
            row = float(cells[0]), float(cells[1]), float(cells[2])
        except ValueError:
            continue
        rows.append(row)
    if len(rows) < 2 or any(
        not all(math.isfinite(value) for value in row) for row in rows
    ):
        raise PredictionContractError(
            f"bound material-property source {path} must contain at least two finite "
            "temperature, Young's-modulus, and Poisson-ratio rows"
        )
    return _MaterialMetadata(
        temperature_knots_c=tuple(row[0] for row in rows),
        youngs_modulus_pa=tuple(row[1] for row in rows),
        poissons_ratio=tuple(row[2] for row in rows),
    )


def _require_supported_value(
    value: float, *, name: str, lower: float, upper: float
) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise PredictionContractError(f"{name} must be finite; received {value!r}")
    if not lower <= value <= upper:
        raise PredictionContractError(
            f"{name} {value:g} is outside the supported inclusive interval "
            f"[{lower:g}, {upper:g}]"
        )


def _standardize(
    values: tuple[float, ...],
    means: tuple[float, ...],
    standard_deviations: tuple[float, ...],
) -> tuple[float, ...]:
    return tuple(
        (value - mean) / standard_deviation
        for value, mean, standard_deviation in zip(
            values, means, standard_deviations, strict=True
        )
    )


def _as_float32(value: float) -> float:
    return struct.unpack("!f", struct.pack("!f", value))[0]


def _mlp(
    inputs: tuple[float, ...],
    widths: tuple[int, ...],
    recipe: _FixtureCheckpoint,
    *,
    path_offset: int,
) -> tuple[float, ...]:
    values = inputs
    for layer_index, (input_width, output_width) in enumerate(
        zip(widths, widths[1:])
    ):
        phase = recipe.phase + (path_offset + layer_index) * 0.41
        weight_scale = recipe.weight_scale / math.sqrt(input_width)
        bias_scale = recipe.bias_scale
        outputs = []
        for output_index in range(output_width):
            total = bias_scale * math.cos(phase + (output_index + 1) * 0.19)
            for input_index, value in enumerate(values):
                parameter_index = output_index * input_width + input_index + 1
                weight = weight_scale * math.sin(
                    phase + parameter_index * 0.37
                )
                total += weight * value
            outputs.append(total)
        if layer_index < len(widths) - 2:
            values = tuple(math.tanh(value) for value in outputs)
        else:
            values = tuple(outputs)
    return values
