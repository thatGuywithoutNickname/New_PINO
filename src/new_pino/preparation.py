"""Canonical source preparation for the public baseline lifecycle."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from hashlib import sha256
import io
import json
import math
from pathlib import Path
from typing import Mapping, NoReturn


_SCHEMA_VERSION = "baseline-source-preflight-v1"
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


class _SourceViolationFound(Exception):
    def __init__(self, violation: SourcePreflightViolation) -> None:
        self.violation = violation


@dataclass(frozen=True)
class _TrainingValidation:
    temperature_range_c: tuple[float, float]
    summary: Mapping[str, object]


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


def prepare_sources(
    repository_root: str | Path,
    *,
    artifact_path: str | Path | None,
) -> SourcePreflightArtifact:
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
        validation_summary["element_points"] = _validate_element_point_source(
            source_contents["element_points"]
        )
        validation_summary["material_properties"] = _validate_material_source(
            source_contents["material_properties"],
            simulation_temperature_range=training_validation.temperature_range_c,
        )
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
    return artifact


def _guard_artifact_destination(
    repository_root: Path,
    artifact_path: str | Path | None,
) -> None:
    if artifact_path is None:
        return
    output = Path(artifact_path).resolve()
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

            aeps_field = values[3:]
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
    )


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


def _validate_element_point_source(content: bytes) -> dict[str, object]:
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
    return {
        "element_point_count": len(points),
        "element_index_basis": "one_based_source_order",
        "x_range_mm": list(x_range),
        "z_range_mm": list(z_range),
        "element_index_binding": [
            {"element_index": index, "x_mm": point[0], "z_mm": point[1]}
            for index, point in enumerate(points, start=1)
        ],
    }


def _validate_material_source(
    content: bytes,
    *,
    simulation_temperature_range: tuple[float, float],
) -> dict[str, object]:
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
    return {
        "temperature_knot_count": len(material_rows),
        "temperature_range_c": list(material_temperature_range),
        "youngs_modulus_unit": "Pa",
        "interpolation": "piecewise_linear",
        "out_of_range": "reject",
    }


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


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
