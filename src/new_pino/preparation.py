"""Canonical source preparation for the public baseline lifecycle."""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from hashlib import sha256
import io
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, NoReturn, Sequence, cast

import numpy as np


_SCHEMA_VERSION = "baseline-source-preflight-v1"
_SPLIT_SCHEMA_VERSION = "baseline-grouped-split-v1"
_PREPROCESSING_SCHEMA_VERSION = "baseline-preprocessing-v1"
_PARTITION_SCHEMA_VERSION = "baseline-prepared-partition-v1"
_SPLIT_RELATIVE_PATH = Path("data/splits/baseline_split_seed42.json")
_CANONICAL_SOURCE_STATUS = "repository_local_canonical"
_SOURCE_PATHS = {
    "training_data": "data/combined_training_data.csv",
    "element_points": "data/co_ind.csv",
    "material_properties": "data/material_properties.md",
}
_EXPECTED_TRAINING_HEADER = (
    "Temperature",
    "Amplitude",
    "Youngs_Modulus",
    *(f"AEPS_Element_{index}" for index in range(1, 49)),
)
_EXPECTED_PCB_YOUNGS_MODULUS_GPA = (20.0, 23.5, 27.0)
_EXPECTED_ELEMENT_POINT_HEADER = ("x", "z")
_RUNTIME_FEATURE_ORDER = (
    "temperature_c",
    "vibration_displacement_amplitude_mm",
    "pcb_youngs_modulus_gpa",
)
_BRANCH_FEATURE_ORDER = (
    *_RUNTIME_FEATURE_ORDER,
    "sac305_youngs_modulus_gpa",
    "sac305_poissons_ratio",
)
_BRANCH_FEATURE_UNITS = (
    "degrees_celsius",
    "millimetres",
    "gigapascals",
    "gigapascals",
    "dimensionless",
)
_TRUNK_FEATURE_ORDER = ("x", "z")
_TRUNK_FEATURE_UNITS = ("millimetres", "millimetres")
_AEPS_ELEMENT_ORDER = _EXPECTED_TRAINING_HEADER[3:]
_DTYPE_POLICY = {
    "source_values": "float64",
    "material_interpolation": "float64",
    "statistics_and_bounds": "float64",
    "stored_normalized_coordinates": "float64",
    "model_inputs": "float32",
    "raw_aeps_fields": "float32",
}
_BASE_TEMPERATURES_C = (
    -40.0,
    -19.25,
    1.5,
    22.5,
    42.0,
    62.75,
    83.5,
    104.25,
    115.0,
    120.0,
    125.0,
)
_BASE_AMPLITUDES_MM = (
    0.2,
    0.29,
    0.375,
    0.46,
    0.55,
    0.64,
    0.725,
    0.81,
    0.9,
)


@dataclass(frozen=True)
class SourcePreflightViolation:
    """The first canonical-source rule violation found by preparation."""

    source: str
    rule: str
    location: str
    value: object
    message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "rule": self.rule,
            "location": self.location,
            "value": self.value,
            "message": self.message,
        }


class SourcePreflightError(ValueError):
    """Preparation stopped at the first canonical-source violation."""

    def __init__(self, artifact: SourcePreflightArtifact) -> None:
        violation = artifact.violation
        if violation is None:
            raise ValueError("a failed preflight artifact requires a violation")
        super().__init__(
            f"{violation.source} violates {violation.rule} at "
            f"{violation.location}: {violation.message}"
        )
        self.artifact = artifact


class SplitManifestError(ValueError):
    """The canonical grouped split cannot be generated or reused."""


class PreprocessingError(ValueError):
    """Validated sources cannot produce the accepted preprocessing state."""


class _SourceViolationFound(Exception):
    def __init__(self, violation: SourcePreflightViolation) -> None:
        self.violation = violation


@dataclass(frozen=True)
class _SimulationCase:
    source_row: int
    temperature_c: float
    vibration_displacement_amplitude_mm: float
    pcb_youngs_modulus_gpa: float
    raw_aeps_field: tuple[float, ...]


@dataclass(frozen=True)
class _TrainingValidation:
    temperature_range_c: tuple[float, float]
    summary: Mapping[str, object]
    cases: tuple[_SimulationCase, ...]


@dataclass(frozen=True)
class _ElementPointValidation:
    summary: Mapping[str, object]
    points_mm: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class _MaterialValidation:
    summary: Mapping[str, object]
    temperature_knots_c: tuple[float, ...]
    youngs_modulus_pa: tuple[float, ...]
    poissons_ratio: tuple[float, ...]


@dataclass(frozen=True)
class _SplitBinding:
    cases: tuple[Mapping[str, object], ...]
    content_identity: str


@dataclass(frozen=True)
class SourcePreflightArtifact:
    """Machine-readable binding of the repository-local canonical sources."""

    schema_version: str
    status: str
    canonical_source_status: str
    source_paths: Mapping[str, str]
    source_checksums: Mapping[str, str]
    validation_summary: Mapping[str, object]
    violation: SourcePreflightViolation | None
    content_identity: str

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "canonical_source_status": self.canonical_source_status,
            "source_paths": dict(self.source_paths),
            "source_checksums": dict(self.source_checksums),
            "validation_summary": dict(self.validation_summary),
            "violation": (
                None if self.violation is None else self.violation.to_dict()
            ),
            "content_identity": self.content_identity,
        }


@dataclass(frozen=True)
class PreprocessingState:
    """Checkpoint-bound schemas, statistics, bounds, and material metadata."""

    schema_version: str
    source_checksums: Mapping[str, str]
    source_identity: str
    split_identity: str
    runtime_feature_order: tuple[str, ...]
    branch_feature_order: tuple[str, ...]
    branch_feature_units: tuple[str, ...]
    trunk_feature_order: tuple[str, ...]
    trunk_feature_units: tuple[str, ...]
    aeps_element_order: tuple[str, ...]
    aeps_unit: str
    aeps_transform: str
    aeps_weighting: str
    branch_mean: np.ndarray
    branch_population_std: np.ndarray
    x_bounds_mm: np.ndarray
    z_bounds_mm: np.ndarray
    element_points_mm: np.ndarray
    normalized_trunk_coordinates: np.ndarray
    material_temperature_knots_c: np.ndarray
    material_youngs_modulus_pa: np.ndarray
    material_poissons_ratio: np.ndarray
    dtype_policy: Mapping[str, str]
    feature_schema_identity: str
    unit_schema_identity: str
    content_identity: str

    def _checkpoint_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "runtime_feature_order": list(self.runtime_feature_order),
            "branch_feature_order": list(self.branch_feature_order),
            "branch_feature_units": list(self.branch_feature_units),
            "trunk_feature_order": list(self.trunk_feature_order),
            "trunk_feature_units": list(self.trunk_feature_units),
            "aeps_element_order": list(self.aeps_element_order),
            "aeps_unit": self.aeps_unit,
            "aeps_transform": self.aeps_transform,
            "aeps_weighting": self.aeps_weighting,
            "branch_mean": self.branch_mean.tolist(),
            "branch_population_std": self.branch_population_std.tolist(),
            "trunk_bounds_mm": {
                "x": self.x_bounds_mm.tolist(),
                "z": self.z_bounds_mm.tolist(),
            },
            "element_points_mm": self.element_points_mm.tolist(),
            "normalized_trunk_coordinates": (
                self.normalized_trunk_coordinates.tolist()
            ),
            "material": {
                "temperature_knots_c": self.material_temperature_knots_c.tolist(),
                "youngs_modulus_pa": self.material_youngs_modulus_pa.tolist(),
                "poissons_ratio": self.material_poissons_ratio.tolist(),
                "interpolation": "piecewise_linear",
                "out_of_range": "reject",
            },
            "dtype_policy": dict(self.dtype_policy),
            "feature_schema_identity": self.feature_schema_identity,
            "unit_schema_identity": self.unit_schema_identity,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "source_checksums": dict(self.source_checksums),
            "source_identity": self.source_identity,
            "split_identity": self.split_identity,
            **self._checkpoint_payload(),
            "content_identity": self.content_identity,
        }


@dataclass(frozen=True)
class PreparedPartition:
    """One canonically ordered simulation-case partition ready for consumers."""

    schema_version: str
    name: str
    source_rows: tuple[int, ...]
    raw_branch_features: np.ndarray
    branch_inputs: np.ndarray
    trunk_inputs: np.ndarray
    raw_aeps_fields: np.ndarray
    element_indices: tuple[int, ...]
    branch_feature_order: tuple[str, ...]
    source_checksums: Mapping[str, str]
    source_identity: str
    split_identity: str
    preprocessing_identity: str
    feature_schema_identity: str
    unit_schema_identity: str
    content_identity: str


@dataclass(frozen=True)
class PreparedDataArtifact(SourcePreflightArtifact):
    """A passing source preflight enriched with checkpoint-bound tensors."""

    preprocessing: PreprocessingState
    partitions: Mapping[str, PreparedPartition]


def prepare_sources(
    repository_root: str | Path,
    *,
    artifact_path: str | Path | None,
) -> PreparedDataArtifact:
    root = Path(repository_root).resolve()
    _guard_artifact_destination(root, artifact_path)
    source_checksums: dict[str, str] = {}
    source_contents: dict[str, bytes] = {}
    validation_summary: dict[str, object] = {}
    try:
        _checksum_canonical_sources(root, source_checksums, source_contents)
        training_validation = _validate_training_source(
            source_contents["training_data"]
        )
        validation_summary["training_data"] = training_validation.summary
        element_point_validation = _validate_element_point_source(
            source_contents["element_points"]
        )
        validation_summary["element_points"] = element_point_validation.summary
        material_validation = _validate_material_source(
            source_contents["material_properties"],
            simulation_temperature_range=training_validation.temperature_range_c,
        )
        validation_summary["material_properties"] = material_validation.summary
    except _SourceViolationFound as error:
        artifact = _build_artifact(
            status="failed",
            source_checksums=source_checksums,
            validation_summary=validation_summary,
            violation=error.violation,
        )
        _write_artifact(artifact, artifact_path)
        raise SourcePreflightError(artifact) from error

    artifact = _build_artifact(
        status="passed",
        source_checksums=source_checksums,
        validation_summary=validation_summary,
        violation=None,
    )
    _write_artifact(artifact, artifact_path)
    split = _prepare_split_manifest(root, training_validation.cases, source_checksums)
    return _prepare_checkpoint_bound_data(
        artifact,
        training_validation=training_validation,
        element_point_validation=element_point_validation,
        material_validation=material_validation,
        split=split,
    )


def _guard_artifact_destination(
    repository_root: Path,
    artifact_path: str | Path | None,
) -> None:
    if artifact_path is None:
        return
    output = Path(artifact_path).resolve()
    split_manifest = (repository_root / _SPLIT_RELATIVE_PATH).resolve()
    aliases_split_manifest = output == split_manifest
    if output.exists() and split_manifest.exists():
        aliases_split_manifest = (
            aliases_split_manifest or output.samefile(split_manifest)
        )
    if aliases_split_manifest:
        raise ValueError(
            "artifact_path must not overwrite the authoritative split manifest"
        )
    for relative_path in _SOURCE_PATHS.values():
        canonical_source = (repository_root / relative_path).resolve()
        aliases_source = output == canonical_source
        if output.exists() and canonical_source.exists():
            aliases_source = aliases_source or output.samefile(canonical_source)
        if aliases_source:
            raise ValueError(
                "artifact_path must not overwrite a repository-local canonical "
                f"source: {relative_path}"
            )


def _checksum_canonical_sources(
    repository_root: Path,
    source_checksums: dict[str, str],
    source_contents: dict[str, bytes],
) -> None:
    for source, relative_path in _SOURCE_PATHS.items():
        path = repository_root / relative_path
        try:
            resolved_path = path.resolve(strict=True)
        except OSError as error:
            _raise_violation(
                source=source,
                rule="canonical_source_access",
                location=relative_path,
                value={"path": relative_path, "error": str(error)},
                message=f"repository-local canonical source cannot be resolved: {error}",
            )
        if not resolved_path.is_relative_to(repository_root):
            _raise_violation(
                source=source,
                rule="repository_local_source",
                location=relative_path,
                value=str(resolved_path),
                message=(
                    "canonical source resolves outside the supplied repository root: "
                    f"{resolved_path}"
                ),
            )
        try:
            content = resolved_path.read_bytes()
        except OSError as error:
            _raise_violation(
                source=source,
                rule="canonical_source_access",
                location=relative_path,
                value={"path": relative_path, "error": str(error)},
                message=f"repository-local canonical source cannot be read: {error}",
            )
        source_contents[source] = content
        source_checksums[source] = sha256(content).hexdigest()


def _validate_training_source(content: bytes) -> _TrainingValidation:
    with io.StringIO(_decode_source(content, source="training_data")) as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if tuple(header) != _EXPECTED_TRAINING_HEADER:
            _raise_violation(
                source="training_data",
                rule="exact_ordered_schema",
                location="header",
                value=header,
                message=(
                    f"expected {list(_EXPECTED_TRAINING_HEADER)!r}; "
                    f"received {header!r}"
                ),
            )

        simulation_cases: set[tuple[float, float, float]] = set()
        groups: dict[tuple[float, float], list[float]] = {}
        temperatures: list[float] = []
        cases: list[_SimulationCase] = []
        case_count = 0
        for source_row, raw_values in enumerate(reader, start=1):
            case_count += 1
            if len(raw_values) != len(_EXPECTED_TRAINING_HEADER):
                _raise_violation(
                    source="training_data",
                    rule="exact_ordered_schema",
                    location=f"row {source_row}",
                    value=len(raw_values),
                    message=(
                        f"expected {len(_EXPECTED_TRAINING_HEADER)} columns; "
                        f"received {len(raw_values)}"
                    ),
                )

            values: list[float] = []
            for column, raw_value in zip(
                _EXPECTED_TRAINING_HEADER,
                raw_values,
                strict=True,
            ):
                values.append(
                    _parse_finite_source_value(
                        source="training_data",
                        location=f"row {source_row}, column {column}",
                        raw_value=raw_value,
                    )
                )

            aeps_field = tuple(values[3:])
            for column, value in zip(
                _EXPECTED_TRAINING_HEADER[3:],
                aeps_field,
                strict=True,
            ):
                if value < 0.0:
                    _raise_violation(
                        source="training_data",
                        rule="non_negative_aeps",
                        location=f"row {source_row}, column {column}",
                        value=value,
                        message=f"expected non-negative AEPS; received {value!r}",
                    )
            if not any(value > 0.0 for value in aeps_field):
                _raise_violation(
                    source="training_data",
                    rule="positive_aeps_field",
                    location=f"row {source_row}, AEPS field",
                    value="all_zero",
                    message="expected at least one positive AEPS value; received all zeros",
                )

            condition = values[0], values[1], values[2]
            if condition in simulation_cases:
                _raise_violation(
                    source="training_data",
                    rule="unique_simulation_cases",
                    location=f"row {source_row}, operating condition",
                    value=list(condition),
                    message=f"duplicate operating condition {condition!r}",
                )
            simulation_cases.add(condition)
            groups.setdefault(condition[:2], []).append(condition[2])
            temperatures.append(condition[0])
            cases.append(
                _SimulationCase(
                    source_row=source_row,
                    temperature_c=condition[0],
                    vibration_displacement_amplitude_mm=condition[1],
                    pcb_youngs_modulus_gpa=condition[2],
                    raw_aeps_field=aeps_field,
                )
            )

    if case_count != 351:
        _raise_violation(
            source="training_data",
            rule="exact_simulation_case_count",
            location="table",
            value=case_count,
            message=f"expected 351 simulation cases; received {case_count}",
        )
    if len(groups) != 117:
        _raise_violation(
            source="training_data",
            rule="temperature_amplitude_group_count",
            location="table",
            value=len(groups),
            message=f"expected 117 temperature-amplitude groups; received {len(groups)}",
        )
    for group, moduli in groups.items():
        sorted_moduli = sorted(moduli)
        if sorted_moduli != list(_EXPECTED_PCB_YOUNGS_MODULUS_GPA):
            _raise_violation(
                source="training_data",
                rule="pcb_youngs_modulus_coverage",
                location=f"temperature-amplitude group {group!r}",
                value=sorted_moduli,
                message=(
                    "expected exactly one case at each PCB Young's modulus "
                    f"{list(_EXPECTED_PCB_YOUNGS_MODULUS_GPA)!r}; "
                    f"received {sorted_moduli!r}"
                ),
            )
    temperature_range = min(temperatures), max(temperatures)
    return _TrainingValidation(
        temperature_range_c=temperature_range,
        summary={
            "simulation_case_count": case_count,
            "temperature_amplitude_group_count": len(groups),
            "pcb_youngs_modulus_gpa": list(_EXPECTED_PCB_YOUNGS_MODULUS_GPA),
            "aeps_element_count": 48,
            "simulation_temperature_range_c": list(temperature_range),
        },
        cases=tuple(cases),
    )


def _prepare_split_manifest(
    repository_root: Path,
    cases: tuple[_SimulationCase, ...],
    source_checksums: Mapping[str, str],
) -> _SplitBinding:
    grouped_cases: dict[tuple[float, float], list[_SimulationCase]] = {}
    for case in cases:
        group = (
            case.temperature_c,
            case.vibration_displacement_amplitude_mm,
        )
        grouped_cases.setdefault(group, []).append(case)

    base_groups = sorted(
        group
        for group in grouped_cases
        if group[0] in _BASE_TEMPERATURES_C
        and group[1] in _BASE_AMPLITUDES_MM
    )
    enrichment_groups = sorted(set(grouped_cases) - set(base_groups))
    if len(base_groups) != 99 or len(enrichment_groups) != 18:
        raise SplitManifestError(
            "expected 99 base-grid groups and 18 enrichment groups; "
            f"received {len(base_groups)} and {len(enrichment_groups)}"
        )
    generator = np.random.Generator(np.random.PCG64(42))
    permuted_base = [base_groups[index] for index in generator.permutation(99)]
    permuted_enrichment = [
        enrichment_groups[index] for index in generator.permutation(18)
    ]
    partition_groups = {
        "training": permuted_base[:69] + permuted_enrichment[:13],
        "validation": permuted_base[69:84] + permuted_enrichment[13:15],
        "test": permuted_base[84:] + permuted_enrichment[15:],
    }
    training_temperatures = {group[0] for group in partition_groups["training"]}
    training_amplitudes = {group[1] for group in partition_groups["training"]}
    held_out_groups = (
        partition_groups["validation"] + partition_groups["test"]
    )
    missing_temperatures = sorted(
        {group[0] for group in held_out_groups} - training_temperatures
    )
    missing_amplitudes = sorted(
        {group[1] for group in held_out_groups} - training_amplitudes
    )
    if missing_temperatures or missing_amplitudes:
        raise SplitManifestError(
            "validation or test levels are absent from training: temperatures "
            f"{missing_temperatures}; vibration displacement amplitudes "
            f"{missing_amplitudes}"
        )

    serialized_cases: list[dict[str, object]] = []
    base_group_set = set(base_groups)
    for partition, groups in partition_groups.items():
        for group in groups:
            for case in sorted(
                grouped_cases[group],
                key=lambda item: item.pcb_youngs_modulus_gpa,
            ):
                serialized_cases.append(
                    {
                        "source_row": case.source_row,
                        "temperature_c": case.temperature_c,
                        "vibration_displacement_amplitude_mm": (
                            case.vibration_displacement_amplitude_mm
                        ),
                        "pcb_youngs_modulus_gpa": (
                            case.pcb_youngs_modulus_gpa
                        ),
                        "stratum": (
                            "base_grid" if group in base_group_set else "enrichment"
                        ),
                        "partition": partition,
                    }
                )

    manifest = {
        "schema_version": _SPLIT_SCHEMA_VERSION,
        "source_checksums": dict(source_checksums),
        "generation": {
            "bit_generator": "PCG64",
            "split_seed": 42,
            "grouping_keys": [
                "temperature_c",
                "vibration_displacement_amplitude_mm",
            ],
            "strata": {
                "base_grid": {
                    "definition": "cartesian_product",
                    "temperature_levels_c": list(_BASE_TEMPERATURES_C),
                    "vibration_displacement_amplitude_levels_mm": list(
                        _BASE_AMPLITUDES_MM
                    ),
                    "group_count": 99,
                },
                "enrichment": {
                    "definition": "all_other_temperature_amplitude_groups",
                    "group_count": 18,
                },
            },
            "group_sort_order": [
                "temperature_c_ascending",
                "vibration_displacement_amplitude_mm_ascending",
            ],
            "generator_call_order": ["base_grid", "enrichment"],
            "allocation_slices": {
                "base_grid": {
                    "training": [0, 69],
                    "validation": [69, 84],
                    "test": [84, 99],
                },
                "enrichment": {
                    "training": [0, 13],
                    "validation": [13, 15],
                    "test": [15, 18],
                },
                "slice_semantics": "zero_based_half_open",
            },
            "partition_order": ["training", "validation", "test"],
            "expansion": {
                "pcb_youngs_modulus_gpa": list(
                    _EXPECTED_PCB_YOUNGS_MODULUS_GPA
                ),
                "order": "ascending",
            },
        },
        "cases": serialized_cases,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    output = repository_root / _SPLIT_RELATIVE_PATH
    if output.exists():
        try:
            existing_bytes = output.read_bytes()
            existing_manifest = json.loads(existing_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SplitManifestError(
                f"split manifest {output} cannot be loaded: {error}"
            ) from error
        if (
            not isinstance(existing_manifest, dict)
            or existing_manifest.get("source_checksums")
            != manifest["source_checksums"]
        ):
            raise SplitManifestError("split manifest source identity mismatch")
        if (
            existing_manifest.get("schema_version")
            != manifest["schema_version"]
            or existing_manifest.get("generation") != manifest["generation"]
        ):
            raise SplitManifestError(
                "split manifest generation metadata mismatch"
            )
        if existing_manifest.get("cases") != manifest["cases"]:
            raise SplitManifestError(
                "split manifest case assignment mismatch"
            )
        if existing_bytes != manifest_bytes:
            raise SplitManifestError(
                "split manifest identity mismatch: expected "
                f"{sha256(manifest_bytes).hexdigest()}, observed "
                f"{sha256(existing_bytes).hexdigest()}"
            )
        return _SplitBinding(
            cases=tuple(serialized_cases),
            content_identity=sha256(existing_bytes).hexdigest(),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(manifest_bytes)
    return _SplitBinding(
        cases=tuple(serialized_cases),
        content_identity=sha256(manifest_bytes).hexdigest(),
    )


def _prepare_checkpoint_bound_data(
    artifact: SourcePreflightArtifact,
    *,
    training_validation: _TrainingValidation,
    element_point_validation: _ElementPointValidation,
    material_validation: _MaterialValidation,
    split: _SplitBinding,
) -> PreparedDataArtifact:
    cases_by_source_row = {
        case.source_row: case for case in training_validation.cases
    }
    raw_branch_by_source_row = {
        source_row: _raw_branch_features(case, material_validation)
        for source_row, case in cases_by_source_row.items()
    }
    training_rows = tuple(
        cast(int, entry["source_row"])
        for entry in split.cases
        if entry["partition"] == "training"
    )
    training_branch = np.asarray(
        [raw_branch_by_source_row[source_row] for source_row in training_rows],
        dtype=np.float64,
    )
    branch_mean = np.mean(training_branch, axis=0, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        branch_population_std = np.std(
            training_branch,
            axis=0,
            ddof=0,
            dtype=np.float64,
        )
    invalid_std = (~np.isfinite(branch_population_std)) | (
        branch_population_std == 0.0
    )
    if np.any(invalid_std):
        invalid_features = [
            _BRANCH_FEATURE_ORDER[index]
            for index in np.flatnonzero(invalid_std).tolist()
        ]
        raise PreprocessingError(
            "training-only branch population standard deviation must be finite "
            f"and non-zero; invalid features: {invalid_features!r}"
        )

    element_points = np.asarray(
        element_point_validation.points_mm,
        dtype=np.float64,
    )
    x_bounds = np.asarray(
        [np.min(element_points[:, 0]), np.max(element_points[:, 0])],
        dtype=np.float64,
    )
    z_bounds = np.asarray(
        [np.min(element_points[:, 1]), np.max(element_points[:, 1])],
        dtype=np.float64,
    )
    x_normalization_bounds = (float(x_bounds[0]), float(x_bounds[1]))
    z_normalization_bounds = (float(z_bounds[0]), float(z_bounds[1]))
    normalized_trunk_coordinates = np.asarray(
        [
            (
                _normalize_coordinate(float(point[0]), x_normalization_bounds),
                _normalize_coordinate(float(point[1]), z_normalization_bounds),
            )
            for point in element_points
        ],
        dtype=np.float64,
    )

    feature_schema_identity = _feature_schema_content_identity()
    unit_schema_identity = _unit_schema_content_identity()
    source_checksums = MappingProxyType(dict(artifact.source_checksums))
    source_identity = _source_content_identity(source_checksums)
    preprocessing = PreprocessingState(
        schema_version=_PREPROCESSING_SCHEMA_VERSION,
        source_checksums=source_checksums,
        source_identity=source_identity,
        split_identity=split.content_identity,
        runtime_feature_order=_RUNTIME_FEATURE_ORDER,
        branch_feature_order=_BRANCH_FEATURE_ORDER,
        branch_feature_units=_BRANCH_FEATURE_UNITS,
        trunk_feature_order=_TRUNK_FEATURE_ORDER,
        trunk_feature_units=_TRUNK_FEATURE_UNITS,
        aeps_element_order=_AEPS_ELEMENT_ORDER,
        aeps_unit="dimensionless",
        aeps_transform="none",
        aeps_weighting="none",
        branch_mean=_readonly_array(branch_mean, dtype=np.float64),
        branch_population_std=_readonly_array(
            branch_population_std,
            dtype=np.float64,
        ),
        x_bounds_mm=_readonly_array(x_bounds, dtype=np.float64),
        z_bounds_mm=_readonly_array(z_bounds, dtype=np.float64),
        element_points_mm=_readonly_array(element_points, dtype=np.float64),
        normalized_trunk_coordinates=_readonly_array(
            normalized_trunk_coordinates,
            dtype=np.float64,
        ),
        material_temperature_knots_c=_readonly_array(
            material_validation.temperature_knots_c,
            dtype=np.float64,
        ),
        material_youngs_modulus_pa=_readonly_array(
            material_validation.youngs_modulus_pa,
            dtype=np.float64,
        ),
        material_poissons_ratio=_readonly_array(
            material_validation.poissons_ratio,
            dtype=np.float64,
        ),
        dtype_policy=MappingProxyType(dict(_DTYPE_POLICY)),
        feature_schema_identity=feature_schema_identity,
        unit_schema_identity=unit_schema_identity,
        content_identity="",
    )
    preprocessing = replace(
        preprocessing,
        content_identity=_preprocessing_content_identity(
            source_checksums=preprocessing.source_checksums,
            source_identity=preprocessing.source_identity,
            split_identity=preprocessing.split_identity,
            preprocessing=preprocessing._checkpoint_payload(),
        ),
    )

    trunk_inputs = _readonly_array(
        preprocessing.normalized_trunk_coordinates,
        dtype=np.float32,
    )
    partitions: dict[str, PreparedPartition] = {}
    for partition_name in ("training", "validation", "test"):
        source_rows = tuple(
            cast(int, entry["source_row"])
            for entry in split.cases
            if entry["partition"] == partition_name
        )
        raw_branch_features = _readonly_array(
            [raw_branch_by_source_row[source_row] for source_row in source_rows],
            dtype=np.float64,
        )
        branch_inputs = _readonly_array(
            (
                raw_branch_features - preprocessing.branch_mean
            )
            / preprocessing.branch_population_std,
            dtype=np.float32,
        )
        raw_aeps_fields = _readonly_array(
            [
                cases_by_source_row[source_row].raw_aeps_field
                for source_row in source_rows
            ],
            dtype=np.float32,
        )
        partition = PreparedPartition(
            schema_version=_PARTITION_SCHEMA_VERSION,
            name=partition_name,
            source_rows=source_rows,
            raw_branch_features=raw_branch_features,
            branch_inputs=branch_inputs,
            trunk_inputs=trunk_inputs,
            raw_aeps_fields=raw_aeps_fields,
            element_indices=tuple(range(1, 49)),
            branch_feature_order=_BRANCH_FEATURE_ORDER,
            source_checksums=source_checksums,
            source_identity=source_identity,
            split_identity=split.content_identity,
            preprocessing_identity=preprocessing.content_identity,
            feature_schema_identity=feature_schema_identity,
            unit_schema_identity=unit_schema_identity,
            content_identity="",
        )
        partitions[partition_name] = replace(
            partition,
            content_identity=_partition_content_identity(partition),
        )

    return PreparedDataArtifact(
        schema_version=artifact.schema_version,
        status=artifact.status,
        canonical_source_status=artifact.canonical_source_status,
        source_paths=artifact.source_paths,
        source_checksums=artifact.source_checksums,
        validation_summary=artifact.validation_summary,
        violation=artifact.violation,
        content_identity=artifact.content_identity,
        preprocessing=preprocessing,
        partitions=MappingProxyType(partitions),
    )


def _raw_branch_features(
    case: _SimulationCase,
    material: _MaterialValidation,
) -> tuple[float, ...]:
    solder_modulus_pa, poissons_ratio = _interpolate_material_properties(
        case.temperature_c,
        material.temperature_knots_c,
        material.youngs_modulus_pa,
        material.poissons_ratio,
    )
    return (
        case.temperature_c,
        case.vibration_displacement_amplitude_mm,
        case.pcb_youngs_modulus_gpa,
        solder_modulus_pa / 1_000_000_000.0,
        poissons_ratio,
    )


def _interpolate_material_properties(
    temperature_c: float,
    temperature_knots_c: Sequence[float],
    youngs_modulus_pa: Sequence[float],
    poissons_ratio: Sequence[float],
) -> tuple[float, float]:
    for index, knot in enumerate(temperature_knots_c):
        if temperature_c == knot:
            return youngs_modulus_pa[index], poissons_ratio[index]
        if temperature_c < knot:
            lower_index = index - 1
            fraction = (temperature_c - temperature_knots_c[lower_index]) / (
                knot - temperature_knots_c[lower_index]
            )
            return (
                youngs_modulus_pa[lower_index]
                + fraction
                * (youngs_modulus_pa[index] - youngs_modulus_pa[lower_index]),
                poissons_ratio[lower_index]
                + fraction * (poissons_ratio[index] - poissons_ratio[lower_index]),
            )
    return youngs_modulus_pa[-1], poissons_ratio[-1]


def _normalize_coordinate(value: float, bounds: Sequence[float]) -> float:
    scale = max(abs(bounds[0]), abs(bounds[1]))
    scaled_lower = bounds[0] / scale
    scaled_upper = bounds[1] / scale
    return (
        2.0
        * (value / scale - scaled_lower)
        / (scaled_upper - scaled_lower)
        - 1.0
    )


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _canonical_json_identity(payload: object) -> str:
    return sha256(_canonical_json_bytes(payload)).hexdigest()


def _source_content_identity(source_checksums: Mapping[str, str]) -> str:
    return _canonical_json_identity({"source_checksums": dict(source_checksums)})


def _preprocessing_content_identity(
    *,
    source_checksums: Mapping[str, str],
    source_identity: str,
    split_identity: str,
    preprocessing: Mapping[str, object],
) -> str:
    return _canonical_json_identity(
        {
            "source_checksums": dict(source_checksums),
            "source_identity": source_identity,
            "split_identity": split_identity,
            "preprocessing": dict(preprocessing),
        }
    )


def _feature_schema_content_identity() -> str:
    return _canonical_json_identity(
        {
            "runtime_feature_order": list(_RUNTIME_FEATURE_ORDER),
            "branch_feature_order": list(_BRANCH_FEATURE_ORDER),
            "trunk_feature_order": list(_TRUNK_FEATURE_ORDER),
            "aeps_element_order": list(_AEPS_ELEMENT_ORDER),
            "element_index_model_input": False,
            "aeps_transform": "none",
            "aeps_weighting": "none",
        }
    )


def _unit_schema_content_identity() -> str:
    return _canonical_json_identity(
        {
            "branch_feature_units": list(_BRANCH_FEATURE_UNITS),
            "trunk_feature_units": list(_TRUNK_FEATURE_UNITS),
            "aeps_unit": "dimensionless",
        }
    )


def _partition_content_identity(partition: PreparedPartition) -> str:
    arrays = {
        "raw_branch_features": partition.raw_branch_features,
        "branch_inputs": partition.branch_inputs,
        "trunk_inputs": partition.trunk_inputs,
        "raw_aeps_fields": partition.raw_aeps_fields,
    }
    metadata: dict[str, object] = {
        "schema_version": partition.schema_version,
        "name": partition.name,
        "source_rows": list(partition.source_rows),
        "element_indices": list(partition.element_indices),
        "branch_feature_order": list(partition.branch_feature_order),
        "source_checksums": dict(partition.source_checksums),
        "source_identity": partition.source_identity,
        "split_identity": partition.split_identity,
        "preprocessing_identity": partition.preprocessing_identity,
        "feature_schema_identity": partition.feature_schema_identity,
        "unit_schema_identity": partition.unit_schema_identity,
        "arrays": {
            name: {"shape": list(array.shape), "dtype": str(array.dtype)}
            for name, array in arrays.items()
        },
    }
    digest = sha256(_canonical_json_bytes(metadata))
    for name, array in arrays.items():
        digest.update(name.encode("ascii"))
        digest.update(b"\0")
        little_endian = np.ascontiguousarray(
            array,
            dtype=array.dtype.newbyteorder("<"),
        )
        digest.update(little_endian.tobytes(order="C"))
    return digest.hexdigest()


def _raise_violation(
    *,
    source: str,
    rule: str,
    location: str,
    value: object,
    message: str,
) -> NoReturn:
    raise _SourceViolationFound(
        SourcePreflightViolation(
            source=_SOURCE_PATHS[source],
            rule=rule,
            location=location,
            value=value,
            message=message,
        )
    )


def _parse_finite_source_value(
    *,
    source: str,
    location: str,
    raw_value: str,
) -> float:
    try:
        value = float(raw_value)
    except ValueError:
        _raise_violation(
            source=source,
            rule="finite_numeric_values",
            location=location,
            value=raw_value,
            message=f"expected a finite numeric value; received {raw_value!r}",
        )
    if not math.isfinite(value):
        _raise_violation(
            source=source,
            rule="finite_numeric_values",
            location=location,
            value=raw_value,
            message=f"expected a finite numeric value; received {raw_value!r}",
        )
    return value


def _validate_element_point_source(content: bytes) -> _ElementPointValidation:
    with io.StringIO(_decode_source(content, source="element_points")) as stream:
        reader = csv.reader(stream)
        header = next(reader, [])
        if tuple(header) != _EXPECTED_ELEMENT_POINT_HEADER:
            _raise_violation(
                source="element_points",
                rule="exact_ordered_schema",
                location="header",
                value=header,
                message=(
                    f"expected {list(_EXPECTED_ELEMENT_POINT_HEADER)!r}; "
                    f"received {header!r}"
                ),
            )

        points: list[tuple[float, float]] = []
        seen_points: set[tuple[float, float]] = set()
        for source_row, raw_values in enumerate(reader, start=1):
            if len(raw_values) != 2:
                _raise_violation(
                    source="element_points",
                    rule="exact_ordered_schema",
                    location=f"row {source_row}",
                    value=len(raw_values),
                    message=f"expected 2 columns; received {len(raw_values)}",
                )
            values: list[float] = []
            for column, raw_value in zip(
                _EXPECTED_ELEMENT_POINT_HEADER,
                raw_values,
                strict=True,
            ):
                values.append(
                    _parse_finite_source_value(
                        source="element_points",
                        location=f"row {source_row}, column {column}",
                        raw_value=raw_value,
                    )
                )
            point = values[0], values[1]
            if point in seen_points:
                _raise_violation(
                    source="element_points",
                    rule="pairwise_unique_element_points",
                    location=f"row {source_row}, element point",
                    value=list(point),
                    message=f"duplicate element point {point!r}",
                )
            seen_points.add(point)
            points.append(point)

    if len(points) != 48:
        _raise_violation(
            source="element_points",
            rule="exact_element_point_count",
            location="table",
            value=len(points),
            message=f"expected 48 element points; received {len(points)}",
        )
    for column_index, column in enumerate(_EXPECTED_ELEMENT_POINT_HEADER):
        coordinate_values = [point[column_index] for point in points]
        bounds = min(coordinate_values), max(coordinate_values)
        if bounds[0] == bounds[1]:
            _raise_violation(
                source="element_points",
                rule="nonzero_coordinate_range",
                location=f"column {column}",
                value=list(bounds),
                message=f"expected a nonzero {column} range; received {bounds!r}",
            )
    x_range = min(point[0] for point in points), max(point[0] for point in points)
    z_range = min(point[1] for point in points), max(point[1] for point in points)
    return _ElementPointValidation(
        summary={
            "element_point_count": len(points),
            "element_index_basis": "one_based_source_order",
            "x_range_mm": list(x_range),
            "z_range_mm": list(z_range),
            "element_index_binding": [
                {"element_index": index, "x_mm": point[0], "z_mm": point[1]}
                for index, point in enumerate(points, start=1)
            ],
        },
        points_mm=tuple(points),
    )


def _validate_material_source(
    content: bytes,
    *,
    simulation_temperature_range: tuple[float, float],
) -> _MaterialValidation:
    lines = _decode_source(content, source="material_properties").splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if _is_material_table_header(_markdown_cells(line))
        ),
        None,
    )
    if header_index is None or header_index + 2 >= len(lines):
        _raise_violation(
            source="material_properties",
            rule="material_table_schema",
            location="document",
            value="not_found",
            message="expected a SAC305 temperature-property Markdown table",
        )

    header_cells = _markdown_cells(lines[header_index])
    assert header_cells is not None
    for column_index, column, accepted_suffixes in (
        (0, "temperature", ("(c)", "(°c)")),
        (1, "Young's modulus", ("(pa)",)),
    ):
        heading = header_cells[column_index]
        normalized_heading = "".join(heading.casefold().split())
        if not normalized_heading.endswith(accepted_suffixes):
            _raise_violation(
                source="material_properties",
                rule="material_units",
                location=f"header, column {column}",
                value=heading,
                message=(
                    f"expected {column} in "
                    f"{', '.join(accepted_suffixes)!r}; received {heading!r}"
                ),
            )

    material_rows: list[tuple[int, float, float, float]] = []
    for line_index in range(header_index + 2, len(lines)):
        cells = _markdown_cells(lines[line_index])
        if cells is None:
            break
        line_number = line_index + 1
        if len(cells) < 3:
            _raise_violation(
                source="material_properties",
                rule="material_table_schema",
                location=f"line {line_number}",
                value=len(cells),
                message=f"expected at least 3 material columns; received {len(cells)}",
            )
        parsed: list[float] = []
        for column, raw_value in zip(
            ("temperature knot", "Young's modulus", "Poisson's ratio"),
            cells[:3],
            strict=True,
        ):
            parsed.append(
                _parse_finite_source_value(
                    source="material_properties",
                    location=f"line {line_number}, column {column}",
                    raw_value=raw_value,
                )
            )
        temperature, youngs_modulus, poissons_ratio = parsed
        if youngs_modulus <= 0.0:
            _raise_violation(
                source="material_properties",
                rule="positive_youngs_modulus",
                location=f"line {line_number}, column Young's modulus",
                value=youngs_modulus,
                message=(
                    "expected a finite positive Young's modulus; "
                    f"received {youngs_modulus!r}"
                ),
            )
        if not -1.0 < poissons_ratio < 0.5:
            _raise_violation(
                source="material_properties",
                rule="poissons_ratio_range",
                location=f"line {line_number}, column Poisson's ratio",
                value=poissons_ratio,
                message=(
                    "expected Poisson's ratio strictly between -1 and 0.5; "
                    f"received {poissons_ratio!r}"
                ),
            )
        material_rows.append(
            (line_number, temperature, youngs_modulus, poissons_ratio)
        )

    if len(material_rows) < 2:
        _raise_violation(
            source="material_properties",
            rule="material_table_schema",
            location="material table",
            value=len(material_rows),
            message=(
                "expected at least two material-property rows for interpolation; "
                f"received {len(material_rows)}"
            ),
        )
    for previous, current in zip(material_rows, material_rows[1:]):
        if previous[1] >= current[1]:
            _raise_violation(
                source="material_properties",
                rule="strictly_increasing_temperature_knots",
                location=f"line {current[0]}, temperature knot",
                value=current[1],
                message=(
                    "expected a unique temperature knot strictly above "
                    f"{previous[1]!r}; received {current[1]!r}"
                ),
            )

    material_temperature_range = material_rows[0][1], material_rows[-1][1]
    if (
        material_temperature_range[0] > simulation_temperature_range[0]
        or material_temperature_range[1] < simulation_temperature_range[1]
    ):
        coverage_value = {
            "material_temperature_range_c": list(material_temperature_range),
            "simulation_temperature_range_c": list(simulation_temperature_range),
        }
        _raise_violation(
            source="material_properties",
            rule="simulation_temperature_coverage",
            location="temperature range",
            value=coverage_value,
            message=(
                f"material range {material_temperature_range!r} must cover every "
                f"simulation temperature in {simulation_temperature_range!r}"
            ),
        )
    return _MaterialValidation(
        summary={
            "temperature_knot_count": len(material_rows),
            "temperature_range_c": list(material_temperature_range),
            "youngs_modulus_unit": "Pa",
            "interpolation": "piecewise_linear",
            "out_of_range": "reject",
        },
        temperature_knots_c=tuple(row[1] for row in material_rows),
        youngs_modulus_pa=tuple(row[2] for row in material_rows),
        poissons_ratio=tuple(row[3] for row in material_rows),
    )


def _markdown_cells(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _is_material_table_header(cells: list[str] | None) -> bool:
    if cells is None or len(cells) < 3:
        return False
    temperature, youngs_modulus, poissons_ratio = (
        cell.casefold() for cell in cells[:3]
    )
    return (
        "temperatur" in temperature
        and ("e-modul" in youngs_modulus or "young" in youngs_modulus)
        and ("querkontraktionszahl" in poissons_ratio or "poisson" in poissons_ratio)
    )


def _build_artifact(
    *,
    status: str,
    source_checksums: Mapping[str, str],
    validation_summary: Mapping[str, object],
    violation: SourcePreflightViolation | None,
) -> SourcePreflightArtifact:
    payload: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "status": status,
        "canonical_source_status": _CANONICAL_SOURCE_STATUS,
        "source_paths": dict(_SOURCE_PATHS),
        "source_checksums": dict(source_checksums),
        "validation_summary": dict(validation_summary),
        "violation": None if violation is None else violation.to_dict(),
    }
    return SourcePreflightArtifact(
        schema_version=_SCHEMA_VERSION,
        status=status,
        canonical_source_status=_CANONICAL_SOURCE_STATUS,
        source_paths=dict(_SOURCE_PATHS),
        source_checksums=dict(source_checksums),
        validation_summary=dict(validation_summary),
        violation=violation,
        content_identity=sha256(_canonical_json_bytes(payload)).hexdigest(),
    )


def _write_artifact(
    artifact: SourcePreflightArtifact,
    artifact_path: str | Path | None,
) -> None:
    if artifact_path is None:
        return
    output = Path(artifact_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(artifact.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _decode_source(content: bytes, *, source: str) -> str:
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError as error:
        _raise_violation(
            source=source,
            rule="utf8_text_encoding",
            location=f"byte {error.start}",
            value=content[error.start : error.end].hex(),
            message=f"expected UTF-8 source text: {error}",
        )


def _canonical_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
