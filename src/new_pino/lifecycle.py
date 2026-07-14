"""The single public lifecycle boundary, starting with fixture prediction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import struct
from typing import Any, Callable, Mapping

import torch

from .preparation import (
    PreparedDataArtifact,
    PreparedPartition,
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
    _partition_content_identity,
    _preprocessing_content_identity,
    _source_content_identity,
    prepare_sources,
    _unit_schema_content_identity,
)
from .freezing import FreezeResult, _PROTOCOL_FIELDS, freeze_seed_runs
from .reporting import (
    EvaluationContractError,
    EvaluationReport,
    _content_identity,
    _evaluation_report,
    _evaluate_fixture,
    _seed_metric_report,
)
from .training import (
    SeedTrainingResult,
    _BRANCH_WIDTHS,
    _EXPECTED_ARCHITECTURE,
    _DeepONet,
    _TRUNK_WIDTHS,
    _checkpoint_identity,
    train_seed,
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
class _FrozenCheckpoint:
    path: Path


@dataclass(frozen=True)
class _LoadedPredictor:
    seed: int
    checkpoint_identity: str
    precision_identity: str
    backend_identity: str
    content_identity: str
    compatibility_identity: str
    validation_comparator_status: str
    run_configuration_identity: str
    checkpoint: _FixtureCheckpoint | _FrozenCheckpoint


@dataclass(frozen=True)
class _LoadedPackage:
    canonical: bool
    evidence_status: str
    gate_status: str
    test_partition_status: str
    source_checksums: _SourceChecksums
    source_identity: str
    split_identity: str
    run_configuration_identity: str
    preprocessing: _PreprocessingState
    predictors: tuple[_LoadedPredictor, ...]


class BaselineLifecycle:
    """Loads one frozen package and exposes its public baseline operations."""

    def __init__(
        self,
        package: _LoadedPackage,
        *,
        package_path: Path,
    ) -> None:
        self._package = package
        self._package_path = package_path.resolve()
        self._coordinate_source = (
            self._package_path / "runtime_sources" / "co_ind.csv"
        )
        self._material_source = (
            self._package_path / "runtime_sources" / "material_properties.md"
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
        smoke_max_epochs: int | None = None,
        recovery_snapshot: str | Path | None = None,
        restart: bool = False,
    ) -> SeedTrainingResult:
        """Run one canonical GPU seed or an explicit CPU smoke seed."""

        return train_seed(
            prepared,
            seed=seed,
            artifact_directory=artifact_directory,
            smoke_max_epochs=smoke_max_epochs,
            recovery_snapshot=recovery_snapshot,
            restart=restart,
        )

    @classmethod
    def freeze(
        cls,
        prepared: PreparedDataArtifact,
        seed_runs: tuple[SeedTrainingResult, ...] | list[SeedTrainingResult],
        *,
        repository_root: str | Path,
        package_directory: str | Path,
    ) -> FreezeResult:
        """Apply the validation-only five-seed freeze gate."""

        return freeze_seed_runs(
            prepared,
            seed_runs,
            repository_root=repository_root,
            package_directory=package_directory,
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
                f"package metadata {metadata_path} cannot be loaded: {error}"
            ) from error
        if isinstance(metadata, dict) and metadata.get("schema_version") == (
            "baseline-frozen-package-v1"
        ):
            package = _parse_frozen_package(metadata, resolved_package_path)
        else:
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

        if isinstance(predictor.checkpoint, _FixtureCheckpoint):
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
                    sum(
                        left * right
                        for left, right in zip(branch_latent, trunk_latent)
                    )
                    + recipe.fusion_bias
                )
        else:
            predictions = _predict_frozen_checkpoint(
                predictor,
                branch,
                preprocessing.normalized_trunk_coordinates,
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
            run_configuration_identity=predictor.run_configuration_identity,
        )
        return PredictionResult(
            aeps_field=tuple(predictions),
            element_indices=tuple(range(1, len(predictions) + 1)),
            evidence_status=self._package.evidence_status,
            provenance=provenance,
        )

    def authorize_locked_test_partition(
        self,
        prepared: PreparedDataArtifact,
    ) -> PreparedPartition:
        """Return the locked partition only for compatible canonical freeze evidence."""

        package = self._package
        if (
            not package.canonical
            or package.gate_status != "passed"
            or package.test_partition_status != "eligible_locked_test"
        ):
            raise PredictionContractError(
                "only a passing canonical frozen package can authorize the real "
                "locked test partition"
            )
        self._verify_runtime_sources()
        for predictor in package.predictors:
            _load_frozen_model(predictor, require_canonical=True)
        preprocessing = prepared.preprocessing
        expected = (
            package.source_checksums.as_dict(),
            package.source_identity,
            package.split_identity,
            package.preprocessing.content_identity,
        )
        actual = (
            dict(prepared.source_checksums),
            preprocessing.source_identity,
            preprocessing.split_identity,
            preprocessing.content_identity,
        )
        if actual != expected:
            raise PredictionContractError(
                "the prepared data is incompatible with the frozen package"
            )
        test_partition = prepared._locked_test_partition
        if (
            _partition_content_identity(test_partition)
            != test_partition.content_identity
            or test_partition.source_identity != package.source_identity
            or test_partition.split_identity != package.split_identity
            or test_partition.preprocessing_identity
            != package.preprocessing.content_identity
        ):
            raise PredictionContractError(
                "the locked test partition is incompatible with the frozen package"
            )
        return test_partition

    def evaluate(
        self,
        evaluation: str | Path | PreparedDataArtifact,
        *,
        artifact_path: str | Path | None = None,
    ) -> EvaluationReport:
        """Evaluate an authorized fixture or the gated locked test partition."""

        if artifact_path is not None:
            output = Path(artifact_path).resolve()
            if output == self._package_path or self._package_path in output.parents:
                raise EvaluationContractError(
                    "evaluation artifact_path must remain outside the loaded package"
                )

        if isinstance(evaluation, PreparedDataArtifact):
            partition = self.authorize_locked_test_partition(evaluation)
            package = self._package
            case_order = [
                f"source_row_{source_row}" for source_row in partition.source_rows
            ]
            truths = tuple(
                tuple(float(value) for value in field)
                for field in partition.raw_aeps_fields
            )
            branch_inputs = tuple(
                tuple(float(value) for value in branch)
                for branch in partition.branch_inputs
            )
            seed_reports = []
            for predictor in package.predictors:
                predictions = _predict_frozen_partition(
                    predictor,
                    branch_inputs,
                    package.preprocessing.normalized_trunk_coordinates,
                )
                seed_reports.append(
                    _seed_metric_report(
                        seed=predictor.seed,
                        checkpoint_identity=predictor.checkpoint_identity,
                        validation_comparator_status=(
                            predictor.validation_comparator_status
                        ),
                        precision_identity=predictor.precision_identity,
                        backend_identity=predictor.backend_identity,
                        content_identity=predictor.content_identity,
                        compatibility_identity=predictor.compatibility_identity,
                        case_order=case_order,
                        truths=truths,
                        predictions=predictions,
                        element_points_mm=package.preprocessing.element_points_mm,
                    )
                )
            return _evaluation_report(
                canonical=True,
                evidence_status="canonical_test_evidence",
                partition_authority_kind="locked_test_manifest",
                partition_authority_identity=partition.content_identity,
                case_order_basis="manifest",
                case_order=case_order,
                source_checksums=package.source_checksums.as_dict(),
                source_identity=package.source_identity,
                split_identity=package.split_identity,
                preprocessing_identity=package.preprocessing.content_identity,
                run_configuration_identity=package.run_configuration_identity,
                seed_reports=seed_reports,
                artifact_path=artifact_path,
            )

        self._verify_runtime_sources()
        package = self._package
        predict_fields: Callable[
            [int, str, tuple[tuple[float, ...], ...]],
            tuple[tuple[float, ...], ...],
        ] | None = None
        if not package.canonical and package.gate_status == "passed" and all(
            isinstance(predictor.checkpoint, _FrozenCheckpoint)
            for predictor in package.predictors
        ):
            predictors = {
                (predictor.seed, predictor.checkpoint_identity): predictor
                for predictor in package.predictors
            }

            def predict_fields(
                seed: int,
                checkpoint_identity: str,
                branch_inputs: tuple[tuple[float, ...], ...],
            ) -> tuple[tuple[float, ...], ...]:
                return _predict_frozen_partition(
                    predictors[(seed, checkpoint_identity)],
                    branch_inputs,
                    package.preprocessing.normalized_trunk_coordinates,
                )

        return _evaluate_fixture(
            evaluation,
            source_checksums=package.source_checksums.as_dict(),
            source_identity=package.source_identity,
            split_identity=package.split_identity,
            preprocessing_identity=package.preprocessing.content_identity,
            run_configuration_identity=package.run_configuration_identity,
            predictor_identities=tuple(
                (
                    predictor.seed,
                    predictor.checkpoint_identity,
                    predictor.validation_comparator_status,
                    predictor.precision_identity,
                    predictor.backend_identity,
                    predictor.content_identity,
                    predictor.compatibility_identity,
                )
                for predictor in package.predictors
            ),
            element_points_mm=package.preprocessing.element_points_mm,
            artifact_path=artifact_path,
            predict_fields=predict_fields,
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

    def _resolve_predictor(self, selector: PredictorSelector) -> _LoadedPredictor:
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


def _parse_package_source_binding(
    package: Mapping[str, Any],
    *,
    package_kind: str,
) -> tuple[_SourceChecksums, str]:
    checksums = _require_mapping(
        package.get("source_checksums"), f"{package_kind} source checksums"
    )
    expected_names = {
        "training_data",
        "element_points",
        "material_properties",
    }
    if set(checksums) != expected_names:
        raise PredictionContractError(
            f"{package_kind} source checksums must bind training_data, "
            "element_points, and material_properties"
        )
    for name, checksum in checksums.items():
        if not _is_sha256(checksum):
            raise PredictionContractError(
                f"{package_kind} source checksum {name!r} must be 64 lowercase "
                "hex characters"
            )
    source_checksums = _SourceChecksums(
        training_data=str(checksums["training_data"]),
        element_points=str(checksums["element_points"]),
        material_properties=str(checksums["material_properties"]),
    )
    source_identity = package.get("source_identity")
    if source_identity != _source_content_identity(source_checksums.as_dict()):
        raise PredictionContractError(
            f"{package_kind} source identity does not match its checksums"
        )
    return source_checksums, str(source_identity)


def _parse_predictor_binding(
    predictor: Mapping[str, Any],
    *,
    label: str,
    source_identity: str,
    split_identity: str,
    preprocessing: _PreprocessingState,
    run_configuration_identity: str,
) -> tuple[str, str, str, str]:
    compatibility = _require_mapping(
        predictor.get("compatibility"), f"{label} compatibility"
    )
    expected_compatibility = {
        "source_identity": source_identity,
        "split_identity": split_identity,
        "preprocessing_identity": preprocessing.content_identity,
        "configuration_identity": run_configuration_identity,
    }
    if any(
        compatibility.get(name) != value
        for name, value in expected_compatibility.items()
    ):
        raise PredictionContractError(
            f"{label} compatibility is inconsistent with the package identities"
        )
    precision_identity = _require_nonempty_identity(
        compatibility, "precision_identity"
    )
    backend_identity = _require_nonempty_identity(
        compatibility, "backend_identity"
    )
    content_identities = _require_mapping(
        compatibility.get("content_identities"), f"{label} content identities"
    )
    if set(content_identities) != {
        "training_partition",
        "validation_partition",
    } or any(
        not isinstance(identity, str) or not identity
        for identity in content_identities.values()
    ):
        raise PredictionContractError(
            f"{label} content identities must bind the training and validation "
            "partitions"
        )
    compatibility_identity = _require_nonempty_identity(
        predictor, "compatibility_identity"
    )
    if compatibility_identity != _content_identity(compatibility):
        raise PredictionContractError(
            f"{label} compatibility identity is stale"
        )
    expected_binding = {
        "source_identity": source_identity,
        "split_identity": split_identity,
        "preprocessing_identity": preprocessing.content_identity,
        "feature_schema_identity": preprocessing.feature_schema_identity,
        "unit_schema_identity": preprocessing.unit_schema_identity,
        "branch_feature_order": list(_BRANCH_FEATURE_ORDER),
    }
    if predictor.get("preprocessing_binding") != expected_binding:
        raise PredictionContractError(
            f"{label} preprocessing binding is incompatible with the package "
            "identities or feature order"
        )
    return (
        precision_identity,
        backend_identity,
        _content_identity(content_identities),
        compatibility_identity,
    )


def _parse_fixture_package(metadata: object) -> _LoadedPackage:
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

    source_checksums, source_identity = _parse_package_source_binding(
        package,
        package_kind="fixture",
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
    parsed_preprocessing = _parse_package_preprocessing(
        package.get("preprocessing"),
        package_kind="fixture",
        source_checksums=source_checksums,
        source_identity=source_identity,
        split_identity=split_identity,
    )
    predictors = package.get("predictors")
    if not isinstance(predictors, list) or not predictors:
        raise PredictionContractError(
            "fixture package must contain at least one explicit predictor"
        )
    seeds: set[int] = set()
    checkpoints: set[str] = set()
    parsed_predictors: list[_LoadedPredictor] = []
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
        (
            precision_identity,
            backend_identity,
            content_identity,
            compatibility_identity,
        ) = _parse_predictor_binding(
            predictor,
            label=f"fixture predictor {predictor_number}",
            source_identity=source_identity,
            split_identity=split_identity,
            preprocessing=parsed_preprocessing,
            run_configuration_identity=run_configuration_identity,
        )
        comparator_status = predictor.get("validation_comparator_status")
        if comparator_status not in {
            "passed",
            "not_passed",
        }:
            raise PredictionContractError(
                f"fixture predictor {predictor_number} validation comparator status "
                "must be 'passed' or 'not_passed'"
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
            _LoadedPredictor(
                seed=seed,
                checkpoint_identity=checkpoint,
                precision_identity=precision_identity,
                backend_identity=backend_identity,
                content_identity=content_identity,
                compatibility_identity=compatibility_identity,
                validation_comparator_status=str(comparator_status),
                run_configuration_identity=run_configuration_identity,
                checkpoint=_FixtureCheckpoint(
                    phase=recipe_values[0],
                    weight_scale=recipe_values[1],
                    bias_scale=recipe_values[2],
                    fusion_bias=recipe_values[3],
                ),
            )
        )

    return _LoadedPackage(
        canonical=False,
        evidence_status="noncanonical_fixture",
        gate_status="fixture_only",
        test_partition_status="locked_ineligible_noncanonical",
        source_checksums=source_checksums,
        source_identity=source_identity,
        split_identity=split_identity,
        run_configuration_identity=run_configuration_identity,
        preprocessing=parsed_preprocessing,
        predictors=tuple(parsed_predictors),
    )


def _parse_frozen_package(
    metadata: object,
    package_path: Path,
) -> _LoadedPackage:
    package = _require_mapping(metadata, "frozen package")
    package_without_identity = dict(package)
    package_content_identity = package_without_identity.pop(
        "package_content_identity", None
    )
    if package_content_identity != _content_identity(package_without_identity):
        raise PredictionContractError("frozen package content identity is stale")
    if package.get("gate_status") != "passed":
        raise PredictionContractError("a frozen package requires a passing gate")
    canonical = package.get("canonical")
    evidence_status = package.get("evidence_status")
    test_partition_status = package.get("test_partition_status")
    expected_status = (
        ("canonical_frozen_package", "eligible_locked_test")
        if canonical is True
        else (
            "noncanonical_frozen_package",
            "locked_ineligible_noncanonical",
        )
    )
    if canonical not in {True, False} or (
        evidence_status,
        test_partition_status,
    ) != expected_status:
        raise PredictionContractError(
            "frozen package canonical and locked-test statuses are inconsistent"
        )

    gate = _require_mapping(package.get("gate_evidence"), "frozen gate evidence")
    gate_without_identity = dict(gate)
    gate_identity = gate_without_identity.pop("content_identity", None)
    if (
        gate_identity != _content_identity(gate_without_identity)
        or gate.get("gate_passed") is not True
        or gate.get("canonical") is not canonical
    ):
        raise PredictionContractError("frozen package gate evidence is incompatible")

    source_checksums, source_identity = _parse_package_source_binding(
        package,
        package_kind="frozen",
    )
    split_identity = _require_nonempty_identity(package, "split_identity")
    if package.get("architecture") != dict(_EXPECTED_ARCHITECTURE):
        raise PredictionContractError("frozen architecture is incompatible")
    preprocessing = _parse_package_preprocessing(
        package.get("preprocessing"),
        package_kind="frozen",
        source_checksums=source_checksums,
        source_identity=source_identity,
        split_identity=split_identity,
    )
    protocol = _require_mapping(
        package.get("training_protocol"), "frozen training protocol"
    )
    protocol_identity = _require_nonempty_identity(protocol, "identity")
    protocol_configuration = _require_mapping(
        protocol.get("configuration"), "frozen training protocol configuration"
    )
    if protocol_identity != _content_identity(protocol_configuration):
        raise PredictionContractError("frozen training protocol identity is stale")

    raw_predictors = package.get("predictors")
    if not isinstance(raw_predictors, list) or len(raw_predictors) != 5:
        raise PredictionContractError(
            "frozen package must contain exactly five explicit predictors"
        )
    predictors: list[_LoadedPredictor] = []
    pooling_identities: set[tuple[str, str, str, str]] = set()
    for index, raw_predictor in enumerate(raw_predictors, start=1):
        predictor = _require_mapping(raw_predictor, f"frozen predictor {index}")
        seed = predictor.get("seed")
        checkpoint_identity = predictor.get("checkpoint_identity")
        if seed != index - 1 or not isinstance(checkpoint_identity, str) or not checkpoint_identity:
            raise PredictionContractError(
                "frozen predictors must contain seeds 0 through 4 in order with "
                "distinct checkpoint identities"
            )
        comparator_status = predictor.get("validation_comparator_status")
        if comparator_status not in {"passed", "not_passed"}:
            raise PredictionContractError(
                f"frozen predictor {index} comparator status is invalid"
            )
        run_configuration_identity = _require_nonempty_identity(
            predictor, "run_configuration_identity"
        )
        compatibility = _require_mapping(
            predictor.get("compatibility"),
            f"frozen predictor {index} compatibility",
        )
        if set(compatibility) != {
            "source_identity",
            "split_identity",
            "preprocessing_identity",
            "configuration_identity",
            "precision_identity",
            "backend_identity",
            "software_identity",
            "content_identities",
        }:
            raise PredictionContractError(
                f"frozen predictor {index} compatibility metadata is incomplete"
            )
        software_identity = _require_nonempty_identity(
            compatibility,
            "software_identity",
        )
        (
            precision_identity,
            backend_identity,
            content_identity,
            compatibility_identity,
        ) = _parse_predictor_binding(
            predictor,
            label=f"frozen predictor {index}",
            source_identity=source_identity,
            split_identity=split_identity,
            preprocessing=preprocessing,
            run_configuration_identity=run_configuration_identity,
        )
        pooling_identities.add(
            (
                precision_identity,
                backend_identity,
                software_identity,
                content_identity,
            )
        )
        artifacts = _require_mapping(
            predictor.get("artifacts"), f"frozen predictor {index} artifacts"
        )
        if set(artifacts) != {
            "checkpoint",
            "validation_predictions",
            "history",
            "metadata",
            "recovery_snapshot",
            "run_history",
        }:
            raise PredictionContractError(
                f"frozen predictor {index} retained artifacts are incomplete"
            )
        artifact_paths = {
            name: _bound_package_artifact(
                package_path,
                relative_path,
                label=f"frozen predictor {index} {name}",
            )
            for name, relative_path in artifacts.items()
        }
        try:
            retained_metadata_value = json.loads(
                artifact_paths["metadata"].read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise PredictionContractError(
                f"frozen predictor {index} retained metadata cannot be loaded"
            ) from error
        retained_metadata = _require_mapping(
            retained_metadata_value,
            f"frozen predictor {index} retained metadata",
        )
        retained_without_identity = dict(retained_metadata)
        retained_identity = retained_without_identity.pop("content_identity", None)
        selected_checkpoint = retained_metadata.get("selected_checkpoint")
        retained_protocol = {
            name: retained_metadata[name]
            for name in _PROTOCOL_FIELDS
            if name in retained_metadata
        }
        if (
            retained_identity != _content_identity(retained_without_identity)
            or retained_metadata.get("seed") != seed
            or retained_metadata.get("configuration_identity")
            != run_configuration_identity
            or retained_metadata.get("compatibility") != compatibility
            or retained_metadata.get("compatibility_identity")
            != compatibility_identity
            or not isinstance(selected_checkpoint, dict)
            or selected_checkpoint.get("identity") != checkpoint_identity
            or retained_protocol != protocol_configuration
        ):
            raise PredictionContractError(
                f"frozen predictor {index} retained metadata is incompatible"
            )
        predictors.append(
            _LoadedPredictor(
                seed=seed,
                checkpoint_identity=checkpoint_identity,
                precision_identity=precision_identity,
                backend_identity=backend_identity,
                content_identity=content_identity,
                compatibility_identity=compatibility_identity,
                validation_comparator_status=str(comparator_status),
                run_configuration_identity=run_configuration_identity,
                checkpoint=_FrozenCheckpoint(artifact_paths["checkpoint"]),
            )
        )
    if len({item.checkpoint_identity for item in predictors}) != 5:
        raise PredictionContractError(
            "frozen checkpoint identities must be distinct"
        )
    if len(pooling_identities) != 1:
        raise PredictionContractError(
            "frozen predictors require compatible precision, backend, software, "
            "and partition identities"
        )
    comparator = gate.get("comparator")
    seed_evidence = gate.get("seed_evidence")
    if (
        gate.get("schema_version") != "baseline-freeze-gate-v1"
        or gate.get("gate_status") != "passed"
        or gate.get("evidence_status") != evidence_status
        or gate.get("test_partition_status") != test_partition_status
        or gate.get("protocol_identity") != protocol_identity
        or gate.get("required_passing_seed_count") != 4
        or gate.get("strict_comparison") is not True
        or gate.get("failure_reasons") != []
        or gate.get("revision_requirement") is not None
        or not isinstance(comparator, dict)
        or comparator.get("kind") != "element_wise_training_mean_aeps_field"
        or comparator.get("applied_unchanged_to_validation_case_count") != 51
        or not isinstance(seed_evidence, list)
        or len(seed_evidence) != 5
    ):
        raise PredictionContractError(
            "frozen package gate evidence is incompatible"
        )
    training_mean = comparator.get("training_mean_aeps_field")
    comparator_rmse = comparator.get("global_validation_rmse")
    if (
        not isinstance(training_mean, list)
        or len(training_mean) != 48
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in training_mean
        )
        or isinstance(comparator_rmse, bool)
        or not isinstance(comparator_rmse, (int, float))
        or not math.isfinite(float(comparator_rmse))
        or comparator_rmse < 0.0
    ):
        raise PredictionContractError(
            "frozen package gate evidence is incompatible"
        )
    seed_rmse: list[float] = []
    passing_count = 0
    for loaded_predictor, raw_evidence in zip(
        predictors,
        seed_evidence,
        strict=True,
    ):
        if not isinstance(raw_evidence, dict):
            raise PredictionContractError(
                "frozen package gate evidence is incompatible"
            )
        validation_rmse = raw_evidence.get("validation_global_rmse")
        if (
            isinstance(validation_rmse, bool)
            or not isinstance(validation_rmse, (int, float))
            or not math.isfinite(float(validation_rmse))
            or validation_rmse < 0.0
        ):
            raise PredictionContractError(
                "frozen package gate evidence is incompatible"
            )
        seed_status = (
            "passed" if validation_rmse < comparator_rmse else "not_passed"
        )
        if (
            raw_evidence.get("seed") != loaded_predictor.seed
            or raw_evidence.get("checkpoint_identity")
            != loaded_predictor.checkpoint_identity
            or raw_evidence.get("comparator_status") != seed_status
            or loaded_predictor.validation_comparator_status != seed_status
        ):
            raise PredictionContractError(
                "frozen package gate evidence is incompatible"
            )
        seed_rmse.append(float(validation_rmse))
        passing_count += seed_status == "passed"
    mean_seed_rmse = sum(seed_rmse) / 5
    if (
        passing_count < 4
        or gate.get("passing_seed_count") != passing_count
        or gate.get("mean_seed_global_rmse") != mean_seed_rmse
        or mean_seed_rmse >= comparator_rmse
    ):
        raise PredictionContractError(
            "frozen package gate evidence is incompatible"
        )
    return _LoadedPackage(
        canonical=bool(canonical),
        evidence_status=str(evidence_status),
        gate_status="passed",
        test_partition_status=str(test_partition_status),
        source_checksums=source_checksums,
        source_identity=source_identity,
        split_identity=split_identity,
        run_configuration_identity=protocol_identity,
        preprocessing=preprocessing,
        predictors=tuple(predictors),
    )


def _parse_package_preprocessing(
    value: object,
    *,
    package_kind: str,
    source_checksums: _SourceChecksums,
    source_identity: str,
    split_identity: str,
) -> _PreprocessingState:
    preprocessing = _require_mapping(value, f"{package_kind} preprocessing")
    expected_keys = {
        "schema_version",
        "runtime_feature_order",
        "branch_feature_order",
        "branch_feature_units",
        "trunk_feature_order",
        "trunk_feature_units",
        "aeps_element_order",
        "aeps_unit",
        "aeps_transform",
        "aeps_weighting",
        "branch_mean",
        "branch_population_std",
        "trunk_bounds_mm",
        "element_points_mm",
        "normalized_trunk_coordinates",
        "material",
        "dtype_policy",
        "feature_schema_identity",
        "unit_schema_identity",
        "content_identity",
    }
    feature_schema_identity = preprocessing.get("feature_schema_identity")
    unit_schema_identity = preprocessing.get("unit_schema_identity")
    if preprocessing.get("branch_feature_units") != list(_BRANCH_FEATURE_UNITS):
        raise PredictionContractError(
            f"{package_kind} branch_feature_units are incompatible"
        )
    if (
        set(preprocessing) != expected_keys
        or preprocessing.get("schema_version") != "baseline-preprocessing-v1"
        or feature_schema_identity != _feature_schema_content_identity()
        or unit_schema_identity != _unit_schema_content_identity()
        or preprocessing.get("runtime_feature_order") != list(_RUNTIME_FEATURE_ORDER)
        or preprocessing.get("branch_feature_order") != list(_BRANCH_FEATURE_ORDER)
        or preprocessing.get("trunk_feature_order") != list(_TRUNK_FEATURE_ORDER)
        or preprocessing.get("trunk_feature_units") != list(_TRUNK_FEATURE_UNITS)
        or preprocessing.get("aeps_element_order") != list(_AEPS_ELEMENT_ORDER)
        or preprocessing.get("dtype_policy") != _DTYPE_POLICY
        or preprocessing.get("aeps_unit") != "dimensionless"
        or preprocessing.get("aeps_transform") != "none"
        or preprocessing.get("aeps_weighting") != "none"
    ):
        raise PredictionContractError(
            f"{package_kind} preprocessing schemas are incompatible"
        )
    preprocessing_without_identity = dict(preprocessing)
    preprocessing_identity = preprocessing_without_identity.pop(
        "content_identity", None
    )
    branch_mean = _require_finite_numbers(
        preprocessing.get("branch_mean"),
        count=5,
        label=f"{package_kind} branch mean",
    )
    branch_std = _require_finite_numbers(
        preprocessing.get("branch_population_std"),
        count=5,
        label=f"{package_kind} branch population standard deviation",
    )
    if any(value <= 0.0 for value in branch_std):
        raise PredictionContractError(
            f"{package_kind} branch population standard deviations must be positive"
        )
    bounds = _require_mapping(
        preprocessing.get("trunk_bounds_mm"), f"{package_kind} trunk bounds"
    )
    x_bounds = _require_bounds(
        bounds.get("x"), "x", package_kind=package_kind
    )
    z_bounds = _require_bounds(
        bounds.get("z"), "z", package_kind=package_kind
    )
    raw_points = preprocessing.get("element_points_mm")
    raw_normalized = preprocessing.get("normalized_trunk_coordinates")
    if (
        not isinstance(raw_points, list)
        or len(raw_points) != 48
        or not isinstance(raw_normalized, list)
        or len(raw_normalized) != 48
    ):
        message = (
            "fixture preprocessing must bind exactly 48 ordered element points"
            if package_kind == "fixture"
            else "frozen preprocessing must contain 48 element points and trunk "
            "coordinates"
        )
        raise PredictionContractError(message)
    points = tuple(
        _require_element_point(point, index=index)
        for index, point in enumerate(raw_points, start=1)
    )
    normalized = tuple(
        _require_element_point(point, index=index)
        for index, point in enumerate(raw_normalized, start=1)
    )
    expected_normalized = tuple(
        (
            _normalize_coordinate(point[0], x_bounds),
            _normalize_coordinate(point[1], z_bounds),
        )
        for point in points
    )
    if (
        (min(point[0] for point in points), max(point[0] for point in points))
        != x_bounds
        or (min(point[1] for point in points), max(point[1] for point in points))
        != z_bounds
        or any(
            not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
            for actual_point, expected_point in zip(
                normalized, expected_normalized, strict=True
            )
            for actual, expected in zip(
                actual_point, expected_point, strict=True
            )
        )
    ):
        message = (
            "fixture normalized trunk coordinates must use the saved x/z bounds "
            "and element-point order"
            if package_kind == "fixture"
            else "frozen normalized trunk coordinates are incompatible"
        )
        raise PredictionContractError(message)
    material = _require_mapping(
        preprocessing.get("material"), f"{package_kind} material metadata"
    )
    raw_knots = material.get("temperature_knots_c")
    if not isinstance(raw_knots, list) or len(raw_knots) < 2:
        raise PredictionContractError(
            f"{package_kind} material metadata requires at least two temperature knots"
        )
    knots = _require_finite_numbers(
        raw_knots,
        count=len(raw_knots),
        label=f"{package_kind} material knots",
    )
    moduli = _require_finite_numbers(
        material.get("youngs_modulus_pa"),
        count=len(knots),
        label=f"{package_kind} material moduli",
    )
    ratios = _require_finite_numbers(
        material.get("poissons_ratio"),
        count=len(knots),
        label=f"{package_kind} material Poisson ratios",
    )
    if (
        any(left >= right for left, right in zip(knots, knots[1:]))
        or knots[0] > -40.0
        or knots[-1] < 125.0
        or any(value <= 0.0 for value in moduli)
        or any(not -1.0 < value < 0.5 for value in ratios)
        or material.get("interpolation") != "piecewise_linear"
        or material.get("out_of_range") != "reject"
    ):
        raise PredictionContractError(
            f"{package_kind} material metadata is incompatible"
        )
    if preprocessing_identity != _preprocessing_content_identity(
        source_checksums=source_checksums.as_dict(),
        source_identity=source_identity,
        split_identity=split_identity,
        preprocessing=preprocessing_without_identity,
    ):
        raise PredictionContractError(
            f"{package_kind} preprocessing content identity is stale"
        )
    return _PreprocessingState(
        branch_mean=branch_mean,
        branch_population_std=branch_std,
        x_bounds_mm=x_bounds,
        z_bounds_mm=z_bounds,
        element_points_mm=points,
        normalized_trunk_coordinates=normalized,
        material=_MaterialMetadata(
            temperature_knots_c=knots,
            youngs_modulus_pa=moduli,
            poissons_ratio=ratios,
        ),
        feature_schema_identity=str(feature_schema_identity),
        unit_schema_identity=str(unit_schema_identity),
        content_identity=str(preprocessing_identity),
    )


def _bound_package_artifact(
    package_path: Path,
    value: object,
    *,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise PredictionContractError(f"{label} path is missing")
    root = package_path.resolve()
    artifact = (root / value).resolve()
    if root not in artifact.parents or not artifact.is_file():
        raise PredictionContractError(
            f"{label} must be a file inside the frozen package"
        )
    return artifact


def _predict_frozen_checkpoint(
    predictor: _LoadedPredictor,
    branch: tuple[float, ...],
    normalized_trunk_coordinates: tuple[tuple[float, float], ...],
) -> list[float]:
    model = _load_frozen_model(predictor, require_canonical=False)
    with torch.no_grad():
        values = model(
            torch.tensor([branch], dtype=torch.float32),
            torch.tensor(normalized_trunk_coordinates, dtype=torch.float32),
        )[0]
    return [float(value) for value in values]


def _predict_frozen_partition(
    predictor: _LoadedPredictor,
    branch_inputs: tuple[tuple[float, ...], ...],
    normalized_trunk_coordinates: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, ...], ...]:
    model = _load_frozen_model(predictor, require_canonical=False)
    with torch.no_grad():
        values = model(
            torch.tensor(branch_inputs, dtype=torch.float32),
            torch.tensor(normalized_trunk_coordinates, dtype=torch.float32),
        )
    predictions = tuple(
        tuple(float(value) for value in field) for field in values
    )
    if not all(math.isfinite(value) for field in predictions for value in field):
        raise PredictionContractError(
            "the selected frozen checkpoint produced a non-finite AEPS value"
        )
    return predictions


def _load_frozen_model(
    predictor: _LoadedPredictor,
    *,
    require_canonical: bool,
) -> _DeepONet:
    if not isinstance(predictor.checkpoint, _FrozenCheckpoint):
        raise PredictionContractError(
            "fixture predictors cannot authorize the real locked test partition"
        )
    try:
        checkpoint = torch.load(
            predictor.checkpoint.path,
            map_location="cpu",
            weights_only=True,
        )
    except (OSError, RuntimeError) as error:
        raise PredictionContractError(
            f"frozen checkpoint cannot be loaded: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping) or (
        checkpoint.get("schema_version") != "baseline-checkpoint-v1"
        or checkpoint.get("seed") != predictor.seed
        or checkpoint.get("checkpoint_identity") != predictor.checkpoint_identity
        or checkpoint.get("run_configuration_identity")
        != predictor.run_configuration_identity
        or checkpoint.get("compatibility_identity")
        != predictor.compatibility_identity
    ):
        raise PredictionContractError(
            "frozen checkpoint identities do not match the selected predictor"
        )
    if require_canonical and (
        checkpoint.get("canonical") is not True
        or checkpoint.get("evidence_status") != "canonical_seed_run"
    ):
        raise PredictionContractError(
            "locked-test authorization requires five canonical seed checkpoints"
        )
    model_state = checkpoint.get("model_state")
    try:
        (
            actual_checkpoint_identity,
            actual_model_state_identity,
            actual_optimizer_state_identity,
        ) = _checkpoint_identity(
            seed=predictor.seed,
            epoch=int(checkpoint["epoch"]),
            validation_mse=float(checkpoint["validation_mse"]),
            model_state=model_state,
            optimizer_state=checkpoint["optimizer_state"],
            run_configuration_identity=predictor.run_configuration_identity,
            compatibility_identity=predictor.compatibility_identity,
            identities={
                "source": checkpoint["source_identity"],
                "split": checkpoint["split_identity"],
                "preprocessing": checkpoint["preprocessing_identity"],
                "feature_schema": checkpoint["feature_schema_identity"],
                "unit_schema": checkpoint["unit_schema_identity"],
            },
        )
    except (KeyError, TypeError, ValueError) as error:
        raise PredictionContractError("frozen checkpoint is incomplete") from error
    if (
        not isinstance(model_state, Mapping)
        or checkpoint.get("model_state_identity")
        != actual_model_state_identity
        or checkpoint.get("optimizer_state_identity")
        != actual_optimizer_state_identity
        or checkpoint.get("content_identity") != predictor.checkpoint_identity
        or actual_checkpoint_identity != predictor.checkpoint_identity
    ):
        raise PredictionContractError("frozen checkpoint content identity is stale")
    model = _DeepONet()
    try:
        model.load_state_dict(model_state)
    except RuntimeError as error:
        raise PredictionContractError(
            f"frozen checkpoint model state is incompatible: {error}"
        ) from error
    model.eval()
    return model


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


def _require_bounds(
    value: object,
    component: str,
    *,
    package_kind: str,
) -> tuple[float, float]:
    lower, upper = _require_finite_numbers(
        value,
        count=2,
        label=f"{package_kind} {component} trunk bounds",
    )
    if lower >= upper:
        raise PredictionContractError(
            f"{package_kind} {component} trunk bounds must have a nonzero increasing range"
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
