from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
import re
from typing import Callable

import pytest

from new_pino import BaselineLifecycle, SourcePreflightError


CANONICAL_SOURCE_PATHS = {
    "training_data": "data/combined_training_data.csv",
    "element_points": "data/co_ind.csv",
    "material_properties": "data/material_properties.md",
}
SYNTHETIC_TEMPERATURES = (
    -40,
    -25,
    -10,
    5,
    20,
    35,
    50,
    65,
    80,
    95,
    110,
    120,
    125,
)
SYNTHETIC_AMPLITUDES = (0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9)
SYNTHETIC_MATERIAL_ROWS = (
    (-50, 8_000_000_000, 0.10),
    (-25, 7_000_000_000, 0.15),
    (0, 6_000_000_000, 0.20),
    (25, 5_000_000_000, 0.25),
    (50, 4_000_000_000, 0.30),
    (75, 3_000_000_000, 0.35),
    (100, 2_000_000_000, 0.40),
    (130, 1_000_000_000, 0.45),
)


def synthetic_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    data_directory = repository / "data"
    data_directory.mkdir(parents=True)

    training_path = repository / CANONICAL_SOURCE_PATHS["training_data"]
    with training_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Temperature",
                "Amplitude",
                "Youngs_Modulus",
                *(f"AEPS_Element_{index}" for index in range(1, 49)),
            ]
        )
        case_number = 0
        for temperature in SYNTHETIC_TEMPERATURES:
            for amplitude in SYNTHETIC_AMPLITUDES:
                for pcb_modulus in (20.0, 23.5, 27.0):
                    case_number += 1
                    aeps_field = [
                        case_number * element_index / 1_000_000
                        for element_index in range(1, 49)
                    ]
                    writer.writerow(
                        [temperature, amplitude, pcb_modulus, *aeps_field]
                    )

    coordinate_path = repository / CANONICAL_SOURCE_PATHS["element_points"]
    with coordinate_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x", "z"])
        writer.writerows((x, z) for x in range(8) for z in range(6))

    material_path = repository / CANONICAL_SOURCE_PATHS["material_properties"]
    material_lines = [
        "# Synthetic Test Fixture Material Properties",
        "",
        (
            "These invented values exist only to exercise automated contracts. "
            "They are not engineering data."
        ),
        "",
        "| Temperature (C) | Young's modulus (Pa) | Poisson's ratio |",
        "| --- | --- | --- |",
        *(
            f"| {temperature} | {modulus} | {poisson_ratio:.2f} |"
            for temperature, modulus, poisson_ratio in SYNTHETIC_MATERIAL_ROWS
        ),
    ]
    material_path.write_text(
        "\n".join(material_lines) + "\n",
        encoding="utf-8",
    )
    return repository


def checksums_for_repository_sources(repository: Path) -> dict[str, str]:
    return {
        name: sha256((repository / relative_path).read_bytes()).hexdigest()
        for name, relative_path in CANONICAL_SOURCE_PATHS.items()
    }


def mutate_csv_source(
    repository: Path,
    source_name: str,
    mutation: Callable[[list[list[str]]], None],
) -> None:
    path = repository / CANONICAL_SOURCE_PATHS[source_name]
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    mutation(rows)
    with path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)


def set_nonnumeric_temperature(rows: list[list[str]]) -> None:
    rows[1][0] = "not-a-number"


def set_nonfinite_temperature(rows: list[list[str]]) -> None:
    rows[1][0] = "nan"


def remove_one_simulation_case(rows: list[list[str]]) -> None:
    rows.pop()


def duplicate_one_operating_condition(rows: list[list[str]]) -> None:
    rows[2][:3] = rows[1][:3]


def set_negative_aeps(rows: list[list[str]]) -> None:
    rows[1][3] = "-0.1"


def set_all_zero_aeps_field(rows: list[list[str]]) -> None:
    rows[1][3:] = ["0"] * 48


def create_extra_temperature_amplitude_group(rows: list[list[str]]) -> None:
    rows[1][0] = "22.6"


def replace_accepted_pcb_modulus(rows: list[list[str]]) -> None:
    rows[1][2] = "21"


def reorder_element_point_header(rows: list[list[str]]) -> None:
    rows[0] = ["z", "x"]


def remove_one_element_point(rows: list[list[str]]) -> None:
    rows.pop()


def set_nonnumeric_element_coordinate(rows: list[list[str]]) -> None:
    rows[1][0] = "not-a-number"


def duplicate_one_element_point(rows: list[list[str]]) -> None:
    rows[2] = rows[1].copy()


def collapse_x_coordinate_range(rows: list[list[str]]) -> None:
    for index, row in enumerate(rows[1:], start=1):
        row[:] = ["1", str(index)]


def collapse_z_coordinate_range(rows: list[list[str]]) -> None:
    for index, row in enumerate(rows[1:], start=1):
        row[:] = [str(index), "1"]


def mutate_material_source(
    repository: Path,
    mutation: Callable[[str], str],
) -> None:
    path = repository / CANONICAL_SOURCE_PATHS["material_properties"]
    path.write_text(mutation(path.read_text(encoding="utf-8")), encoding="utf-8")


def remove_material_table(_: str) -> str:
    return "# Synthetic Material Properties\n\nNo material table is present.\n"


def set_nonnumeric_solder_modulus(text: str) -> str:
    return text.replace("7000000000 | 0.15", "not-a-number | 0.15", 1)


def relabel_material_temperature_as_kelvin(text: str) -> str:
    return text.replace("Temperature (C)", "Temperature (K)", 1)


def relabel_solder_modulus_as_gigapascals(text: str) -> str:
    return text.replace("Young's modulus (Pa)", "Young's modulus (GPa)", 1)


def set_negative_solder_modulus(text: str) -> str:
    return text.replace("7000000000 | 0.15", "-1 | 0.15", 1)


def set_invalid_poissons_ratio(text: str) -> str:
    return text.replace("7000000000 | 0.15", "7000000000 | 0.5", 1)


def duplicate_material_temperature_knot(text: str) -> str:
    return text.replace("| -25 | 7000000000", "| -50 | 7000000000", 1)


def stop_material_coverage_below_simulation_maximum(text: str) -> str:
    return text.replace("| 130 | 1000000000", "| 124 | 1000000000", 1)


def test_prepare_binds_synthetic_sources_in_a_versioned_artifact(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)
    original_bytes = {
        name: (repository / relative_path).read_bytes()
        for name, relative_path in CANONICAL_SOURCE_PATHS.items()
    }
    artifact_path = tmp_path / "baseline-source-preflight.json"

    artifact = BaselineLifecycle.prepare(
        repository,
        artifact_path=artifact_path,
    )

    assert artifact.schema_version == "baseline-source-preflight-v1"
    assert artifact.status == "passed"
    assert artifact.canonical_source_status == "repository_local_canonical"
    assert artifact.source_paths == CANONICAL_SOURCE_PATHS
    assert artifact.source_checksums == checksums_for_repository_sources(repository)
    assert re.fullmatch(r"[0-9a-f]{64}", artifact.content_identity)
    serialized_artifact = artifact.to_dict()
    identity_payload = dict(serialized_artifact)
    identity_payload.pop("content_identity")
    expected_identity = sha256(
        json.dumps(
            identity_payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert artifact.content_identity == expected_identity
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == serialized_artifact
    assert {
        name: (repository / relative_path).read_bytes()
        for name, relative_path in CANONICAL_SOURCE_PATHS.items()
    } == original_bytes


def test_training_schema_failure_keeps_all_checksum_first_bindings(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)
    training_source = repository / CANONICAL_SOURCE_PATHS["training_data"]
    training_source.write_text(
        training_source.read_text(encoding="utf-8").replace(
            "Temperature,Amplitude",
            "temperature,Amplitude",
            1,
        ),
        encoding="utf-8",
    )
    expected_checksums = checksums_for_repository_sources(repository)
    artifact_path = tmp_path / "failed-preflight.json"

    with pytest.raises(
        SourcePreflightError,
        match=(
            r"data/combined_training_data\.csv violates exact_ordered_schema "
            r"at header"
        ),
    ) as caught:
        BaselineLifecycle.prepare(repository, artifact_path=artifact_path)

    artifact = caught.value.artifact
    assert artifact.status == "failed"
    assert artifact.source_checksums == expected_checksums
    assert artifact.violation is not None
    assert artifact.violation.source == "data/combined_training_data.csv"
    assert artifact.violation.rule == "exact_ordered_schema"
    assert artifact.violation.location == "header"
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact.to_dict()


@pytest.mark.parametrize(
    ("mutation", "expected_rule", "expected_location", "expected_value"),
    [
        (
            set_nonnumeric_temperature,
            "finite_numeric_values",
            "row 1, column Temperature",
            "not-a-number",
        ),
        (
            set_nonfinite_temperature,
            "finite_numeric_values",
            "row 1, column Temperature",
            "nan",
        ),
        (
            set_negative_aeps,
            "non_negative_aeps",
            "row 1, column AEPS_Element_1",
            "-0.1",
        ),
        (
            set_all_zero_aeps_field,
            "positive_aeps_field",
            "row 1, AEPS field",
            "all_zero",
        ),
        (
            duplicate_one_operating_condition,
            "unique_simulation_cases",
            "row 2, operating condition",
            "-40.0",
        ),
        (
            remove_one_simulation_case,
            "exact_simulation_case_count",
            "table",
            "350",
        ),
        (
            create_extra_temperature_amplitude_group,
            "temperature_amplitude_group_count",
            "table",
            "118",
        ),
        (
            replace_accepted_pcb_modulus,
            "pcb_youngs_modulus_coverage",
            "temperature-amplitude group",
            "21.0",
        ),
    ],
)
def test_training_source_contract_fails_on_the_first_actionable_violation(
    tmp_path: Path,
    mutation: Callable[[list[list[str]]], None],
    expected_rule: str,
    expected_location: str,
    expected_value: str,
) -> None:
    repository = synthetic_repository(tmp_path)
    mutate_csv_source(repository, "training_data", mutation)
    expected_checksums = checksums_for_repository_sources(repository)

    with pytest.raises(SourcePreflightError) as caught:
        BaselineLifecycle.prepare(repository)

    artifact = caught.value.artifact
    assert artifact.source_checksums == expected_checksums
    assert artifact.violation is not None
    assert artifact.violation.source == "data/combined_training_data.csv"
    assert artifact.violation.rule == expected_rule
    assert expected_location in artifact.violation.location
    assert expected_value in repr(artifact.violation.value)


@pytest.mark.parametrize(
    ("mutation", "expected_rule", "expected_location", "expected_value"),
    [
        (
            reorder_element_point_header,
            "exact_ordered_schema",
            "header",
            "z",
        ),
        (
            set_nonnumeric_element_coordinate,
            "finite_numeric_values",
            "row 1, column x",
            "not-a-number",
        ),
        (
            duplicate_one_element_point,
            "pairwise_unique_element_points",
            "row 2, element point",
            "[0.0, 0.0]",
        ),
        (
            remove_one_element_point,
            "exact_element_point_count",
            "table",
            "47",
        ),
        (
            collapse_x_coordinate_range,
            "nonzero_coordinate_range",
            "column x",
            "1.0",
        ),
        (
            collapse_z_coordinate_range,
            "nonzero_coordinate_range",
            "column z",
            "1.0",
        ),
    ],
)
def test_element_point_contract_rejects_invalid_positional_bindings(
    tmp_path: Path,
    mutation: Callable[[list[list[str]]], None],
    expected_rule: str,
    expected_location: str,
    expected_value: str,
) -> None:
    repository = synthetic_repository(tmp_path)
    mutate_csv_source(repository, "element_points", mutation)

    with pytest.raises(SourcePreflightError) as caught:
        BaselineLifecycle.prepare(repository)

    violation = caught.value.artifact.violation
    assert violation is not None
    assert violation.source == "data/co_ind.csv"
    assert violation.rule == expected_rule
    assert expected_location in violation.location
    assert expected_value in repr(violation.value)


@pytest.mark.parametrize(
    ("mutation", "expected_rule", "expected_location", "expected_value"),
    [
        (
            remove_material_table,
            "material_table_schema",
            "document",
            "not_found",
        ),
        (
            relabel_material_temperature_as_kelvin,
            "material_units",
            "header, column temperature",
            "Temperature (K)",
        ),
        (
            relabel_solder_modulus_as_gigapascals,
            "material_units",
            "header, column Young's modulus",
            "Young's modulus (GPa)",
        ),
        (
            set_nonnumeric_solder_modulus,
            "finite_numeric_values",
            "Young's modulus",
            "not-a-number",
        ),
        (
            set_negative_solder_modulus,
            "positive_youngs_modulus",
            "Young's modulus",
            "-1.0",
        ),
        (
            set_invalid_poissons_ratio,
            "poissons_ratio_range",
            "Poisson's ratio",
            "0.5",
        ),
        (
            duplicate_material_temperature_knot,
            "strictly_increasing_temperature_knots",
            "temperature knot",
            "-50.0",
        ),
        (
            stop_material_coverage_below_simulation_maximum,
            "simulation_temperature_coverage",
            "temperature range",
            "124.0",
        ),
    ],
)
def test_material_contract_rejects_invalid_or_incomplete_property_tables(
    tmp_path: Path,
    mutation: Callable[[str], str],
    expected_rule: str,
    expected_location: str,
    expected_value: str,
) -> None:
    repository = synthetic_repository(tmp_path)
    mutate_material_source(repository, mutation)

    with pytest.raises(SourcePreflightError) as caught:
        BaselineLifecycle.prepare(repository)

    violation = caught.value.artifact.violation
    assert violation is not None
    assert violation.source == "data/material_properties.md"
    assert violation.rule == expected_rule
    assert expected_location in violation.location
    assert expected_value in repr(violation.value)


def test_missing_repository_source_fails_without_using_an_external_fallback(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)
    canonical_material = repository / CANONICAL_SOURCE_PATHS["material_properties"]
    external_decoy = repository.parent / "material_properties.md"
    external_decoy.write_bytes(canonical_material.read_bytes())
    canonical_material.unlink()
    artifact_path = tmp_path / "missing-source-preflight.json"

    with pytest.raises(SourcePreflightError) as caught:
        BaselineLifecycle.prepare(repository, artifact_path=artifact_path)

    artifact = caught.value.artifact
    assert artifact.status == "failed"
    assert set(artifact.source_checksums) == {"training_data", "element_points"}
    assert artifact.violation is not None
    assert artifact.violation.source == "data/material_properties.md"
    assert artifact.violation.rule == "canonical_source_access"
    assert "data/material_properties.md" in artifact.violation.location
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == artifact.to_dict()


@pytest.mark.parametrize("source_name", list(CANONICAL_SOURCE_PATHS))
def test_artifact_output_cannot_overwrite_a_canonical_source(
    tmp_path: Path,
    source_name: str,
) -> None:
    repository = synthetic_repository(tmp_path)
    source_path = repository / CANONICAL_SOURCE_PATHS[source_name]
    original_bytes = source_path.read_bytes()

    with pytest.raises(
        ValueError,
        match="artifact_path must not overwrite a repository-local canonical source",
    ):
        BaselineLifecycle.prepare(repository, artifact_path=source_path)

    assert source_path.read_bytes() == original_bytes


def test_passing_artifact_preserves_the_validated_one_based_source_order(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)

    artifact = BaselineLifecycle.prepare(repository)

    assert artifact.violation is None
    assert artifact.validation_summary["training_data"] == {
        "simulation_case_count": 351,
        "temperature_amplitude_group_count": 117,
        "pcb_youngs_modulus_gpa": [20.0, 23.5, 27.0],
        "aeps_element_count": 48,
        "simulation_temperature_range_c": [-40.0, 125.0],
    }
    element_summary = artifact.validation_summary["element_points"]
    assert isinstance(element_summary, dict)
    assert element_summary["element_point_count"] == 48
    assert element_summary["element_index_basis"] == "one_based_source_order"
    assert element_summary["x_range_mm"] == [0.0, 7.0]
    assert element_summary["z_range_mm"] == [0.0, 5.0]
    binding = element_summary["element_index_binding"]
    assert isinstance(binding, list)
    assert binding[0] == {"element_index": 1, "x_mm": 0.0, "z_mm": 0.0}
    assert binding[1] == {"element_index": 2, "x_mm": 0.0, "z_mm": 1.0}
    assert binding[-1] == {"element_index": 48, "x_mm": 7.0, "z_mm": 5.0}
    assert artifact.validation_summary["material_properties"] == {
        "temperature_knot_count": 8,
        "temperature_range_c": [-50.0, 130.0],
        "youngs_modulus_unit": "Pa",
        "interpolation": "piecewise_linear",
        "out_of_range": "reject",
    }
