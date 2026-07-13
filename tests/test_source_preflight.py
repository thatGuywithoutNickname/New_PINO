from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path
import re
from typing import Callable

import numpy as np
import pytest

from new_pino import (
    BaselineLifecycle,
    PreprocessingError,
    SourcePreflightError,
    SplitManifestError,
)


CANONICAL_SOURCE_PATHS = {
    "training_data": "data/combined_training_data.csv",
    "element_points": "data/co_ind.csv",
    "material_properties": "data/material_properties.md",
}
SYNTHETIC_BASE_TEMPERATURES = (
    -40,
    -19.25,
    1.5,
    22.5,
    42,
    62.75,
    83.5,
    104.25,
    115,
    120,
    125,
)
SYNTHETIC_BASE_AMPLITUDES = (
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
SYNTHETIC_ENRICHMENT_GROUPS = (
    (117, 0.85),
    (117, 0.9),
    (119, 0.85),
    (119, 0.9),
    (120, 0.85),
    (121, 0.81),
    (121, 0.85),
    (121, 0.9),
    (122, 0.81),
    (122, 0.85),
    (122, 0.9),
    (123, 0.81),
    (123, 0.85),
    (123, 0.9),
    (124, 0.81),
    (124, 0.85),
    (124, 0.9),
    (125, 0.85),
)
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
        groups = [
            (temperature, amplitude)
            for temperature in SYNTHETIC_BASE_TEMPERATURES
            for amplitude in SYNTHETIC_BASE_AMPLITUDES
        ]
        groups.extend(SYNTHETIC_ENRICHMENT_GROUPS)
        for temperature, amplitude in groups:
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


def canonical_identity(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


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


def move_one_base_group_to_enrichment(rows: list[list[str]]) -> None:
    for row in rows[1:]:
        if row[0] == "-40" and row[1] == "0.2":
            row[0] = "-39"


def give_a_held_out_enrichment_group_unique_levels(rows: list[list[str]]) -> None:
    for row in rows[1:]:
        if row[0] == "125" and row[1] == "0.85":
            row[0] = "126"
            row[1] = "0.86"


def change_one_valid_aeps_value(rows: list[list[str]]) -> None:
    rows[1][3] = str(float(rows[1][3]) + 0.001)


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


def use_extreme_finite_coordinate_bounds(rows: list[list[str]]) -> None:
    for index, row in enumerate(rows[1:]):
        x_coordinate = "-1e308" if index < 24 else "1e308"
        row[:] = [x_coordinate, str(index)]


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


def insert_noncollinear_material_knot(text: str) -> str:
    return text.replace(
        "| 25 | 5000000000 | 0.25 |",
        "| 22.5 | 9000000000 | 0.33 |\n| 25 | 5000000000 | 0.25 |",
        1,
    )


def make_poissons_ratio_constant(text: str) -> str:
    for value in ("0.10", "0.15", "0.20", "0.30", "0.35", "0.40", "0.45"):
        text = text.replace(f"| {value} |", "| 0.25 |")
    return text


def make_solder_modulus_variation_overflow_float64(text: str) -> str:
    for old, new in zip(
        (
            "8000000000",
            "7000000000",
            "6000000000",
            "5000000000",
            "4000000000",
            "3000000000",
            "2000000000",
            "1000000000",
        ),
        ("8e307", "7e307", "6e307", "5e307", "4e307", "3e307", "2e307", "1e307"),
        strict=True,
    ):
        text = text.replace(old, new, 1)
    return text


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
    expected_identity = canonical_identity(identity_payload)
    assert artifact.content_identity == expected_identity
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == serialized_artifact
    assert {
        name: (repository / relative_path).read_bytes()
        for name, relative_path in CANONICAL_SOURCE_PATHS.items()
    } == original_bytes


def test_prepare_fits_checkpoint_bound_manifest_ordered_tensors(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)

    prepared = BaselineLifecycle.prepare(repository)

    preprocessing = prepared.preprocessing
    assert preprocessing.runtime_feature_order == (
        "temperature_c",
        "vibration_displacement_amplitude_mm",
        "pcb_youngs_modulus_gpa",
    )
    assert preprocessing.branch_feature_order == (
        "temperature_c",
        "vibration_displacement_amplitude_mm",
        "pcb_youngs_modulus_gpa",
        "sac305_youngs_modulus_gpa",
        "sac305_poissons_ratio",
    )
    assert preprocessing.branch_feature_units == (
        "degrees_celsius",
        "millimetres",
        "gigapascals",
        "gigapascals",
        "dimensionless",
    )
    assert preprocessing.trunk_feature_order == ("x", "z")
    assert preprocessing.trunk_feature_units == ("millimetres", "millimetres")
    assert preprocessing.aeps_element_order == tuple(
        f"AEPS_Element_{index}" for index in range(1, 49)
    )
    assert preprocessing.aeps_unit == "dimensionless"
    assert preprocessing.aeps_transform == "none"
    assert preprocessing.aeps_weighting == "none"
    assert preprocessing.source_identity == canonical_identity(
        {"source_checksums": dict(prepared.source_checksums)}
    )
    checkpoint_state = preprocessing.to_dict()
    for provenance_name in (
        "source_checksums",
        "source_identity",
        "split_identity",
        "content_identity",
    ):
        checkpoint_state.pop(provenance_name)
    assert preprocessing.content_identity == canonical_identity(
        {
            "source_checksums": dict(prepared.source_checksums),
            "source_identity": preprocessing.source_identity,
            "split_identity": preprocessing.split_identity,
            "preprocessing": checkpoint_state,
        }
    )

    assert preprocessing.branch_mean.dtype == np.float64
    assert preprocessing.branch_population_std.dtype == np.float64
    np.testing.assert_allclose(
        preprocessing.branch_mean,
        [
            66.84756097560975,
            0.5959756097560981,
            23.5,
            3.3829471544715473,
            0.3308526422764223,
        ],
        rtol=1e-15,
    )
    np.testing.assert_allclose(
        preprocessing.branch_population_std,
        [
            57.063784862482166,
            0.23775556086491248,
            2.857738033247041,
            2.229298507593977,
            0.11146492537969904,
        ],
        rtol=1e-15,
    )
    assert preprocessing.x_bounds_mm.dtype == np.float64
    assert preprocessing.z_bounds_mm.dtype == np.float64
    assert preprocessing.element_points_mm.dtype == np.float64
    assert preprocessing.normalized_trunk_coordinates.dtype == np.float64
    np.testing.assert_array_equal(preprocessing.x_bounds_mm, [0.0, 7.0])
    np.testing.assert_array_equal(preprocessing.z_bounds_mm, [0.0, 5.0])
    np.testing.assert_allclose(
        preprocessing.normalized_trunk_coordinates[[0, 1, 20, 47]],
        [[-1.0, -1.0], [-1.0, -0.6], [-1.0 / 7.0, -0.2], [1.0, 1.0]],
    )

    manifest = json.loads(
        (repository / "data/splits/baseline_split_seed42.json").read_text(
            encoding="utf-8"
        )
    )
    expected_sizes = {"training": 246, "validation": 51, "test": 54}
    expected_first_inputs = {
        "training": [
            0.29182151,
            -0.57191348,
            -1.22474492,
            -0.32429355,
            0.32429355,
        ],
        "validation": [
            -0.43543485,
            1.27872670,
            -1.22474492,
            0.42033529,
            -0.42033529,
        ],
        "test": [
            -0.43543485,
            0.18516661,
            -1.22474492,
            0.42033529,
            -0.42033529,
        ],
    }
    for name, size in expected_sizes.items():
        partition = prepared.partitions[name]
        assert partition.source_rows == tuple(
            case["source_row"]
            for case in manifest["cases"]
            if case["partition"] == name
        )
        assert partition.raw_branch_features.shape == (size, 5)
        assert partition.raw_branch_features.dtype == np.float64
        assert partition.branch_inputs.shape == (size, 5)
        assert partition.branch_inputs.dtype == np.float32
        assert partition.trunk_inputs.shape == (48, 2)
        assert partition.trunk_inputs.dtype == np.float32
        assert partition.raw_aeps_fields.shape == (size, 48)
        assert partition.raw_aeps_fields.dtype == np.float32
        assert partition.element_indices == tuple(range(1, 49))
        np.testing.assert_allclose(
            partition.branch_inputs[0],
            expected_first_inputs[name],
            rtol=1e-6,
        )
        np.testing.assert_array_equal(
            partition.trunk_inputs,
            preprocessing.normalized_trunk_coordinates.astype(np.float32),
        )
        assert partition.source_checksums == prepared.source_checksums
        assert partition.source_identity == preprocessing.source_identity
        assert partition.split_identity == preprocessing.split_identity
        assert partition.preprocessing_identity == preprocessing.content_identity
        assert (
            partition.feature_schema_identity
            == preprocessing.feature_schema_identity
        )
        assert partition.unit_schema_identity == preprocessing.unit_schema_identity
        assert re.fullmatch(r"[0-9a-f]{64}", partition.content_identity)
        assert not partition.raw_branch_features.flags.writeable
        assert not partition.branch_inputs.flags.writeable
        assert not partition.trunk_inputs.flags.writeable
        assert not partition.raw_aeps_fields.flags.writeable

    training = prepared.partitions["training"]
    assert training.source_rows[0] == 172
    np.testing.assert_allclose(
        training.raw_branch_features[0],
        [83.5, 0.46, 20.0, 2.66, 0.367],
        rtol=1e-15,
    )
    np.testing.assert_array_equal(
        training.raw_aeps_fields[0],
        np.asarray(
            [172 * element_index / 1_000_000 for element_index in range(1, 49)],
            dtype=np.float32,
        ),
    )

    repeated = BaselineLifecycle.prepare(repository)
    assert repeated.preprocessing.content_identity == preprocessing.content_identity
    assert {
        name: partition.content_identity
        for name, partition in repeated.partitions.items()
    } == {
        name: partition.content_identity
        for name, partition in prepared.partitions.items()
    }


def test_prepare_preserves_exact_material_knots_in_branch_features(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)
    mutate_material_source(repository, insert_noncollinear_material_knot)

    prepared = BaselineLifecycle.prepare(repository)

    rows_at_knot = np.concatenate(
        [
            partition.raw_branch_features[
                partition.raw_branch_features[:, 0] == 22.5
            ]
            for partition in prepared.partitions.values()
        ]
    )
    assert rows_at_knot.shape == (27, 5)
    np.testing.assert_array_equal(rows_at_knot[:, 3], np.full(27, 9.0))
    np.testing.assert_array_equal(rows_at_knot[:, 4], np.full(27, 0.33))


def test_prepare_normalizes_extreme_finite_coordinates_without_overflow(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)
    mutate_csv_source(
        repository,
        "element_points",
        use_extreme_finite_coordinate_bounds,
    )

    prepared = BaselineLifecycle.prepare(repository)

    normalized = prepared.preprocessing.normalized_trunk_coordinates
    assert np.all(np.isfinite(normalized))
    np.testing.assert_array_equal(normalized[:24, 0], -1.0)
    np.testing.assert_array_equal(normalized[24:, 0], 1.0)


@pytest.mark.parametrize(
    ("mutation", "invalid_feature"),
    [
        (make_poissons_ratio_constant, "sac305_poissons_ratio"),
        (
            make_solder_modulus_variation_overflow_float64,
            "sac305_youngs_modulus_gpa",
        ),
    ],
)
def test_prepare_rejects_invalid_training_population_standard_deviation(
    tmp_path: Path,
    mutation: Callable[[str], str],
    invalid_feature: str,
) -> None:
    repository = synthetic_repository(tmp_path)
    mutate_material_source(repository, mutation)

    with pytest.raises(PreprocessingError, match=invalid_feature):
        BaselineLifecycle.prepare(repository)


def test_prepare_generates_the_authoritative_grouped_split_manifest(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)

    preflight = BaselineLifecycle.prepare(repository)

    manifest_path = repository / "data/splits/baseline_split_seed42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "baseline-grouped-split-v1"
    assert manifest["source_checksums"] == preflight.source_checksums
    cases = manifest["cases"]
    assert len(cases) == 351
    assert [
        sum(case["partition"] == partition for case in cases)
        for partition in ("training", "validation", "test")
    ] == [246, 51, 54]


def test_grouped_split_records_the_complete_generation_rule(tmp_path: Path) -> None:
    repository = synthetic_repository(tmp_path)

    BaselineLifecycle.prepare(repository)

    manifest = json.loads(
        (repository / "data/splits/baseline_split_seed42.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["generation"] == {
        "bit_generator": "PCG64",
        "split_seed": 42,
        "grouping_keys": [
            "temperature_c",
            "vibration_displacement_amplitude_mm",
        ],
        "strata": {
            "base_grid": {
                "definition": "cartesian_product",
                "temperature_levels_c": list(SYNTHETIC_BASE_TEMPERATURES),
                "vibration_displacement_amplitude_levels_mm": list(
                    SYNTHETIC_BASE_AMPLITUDES
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
            "pcb_youngs_modulus_gpa": [20.0, 23.5, 27.0],
            "order": "ascending",
        },
    }


def test_grouped_split_rejects_an_invalid_stratum_classification(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)
    mutate_csv_source(
        repository,
        "training_data",
        move_one_base_group_to_enrichment,
    )

    with pytest.raises(
        SplitManifestError,
        match=(
            "expected 99 base-grid groups and 18 enrichment groups; "
            "received 98 and 19"
        ),
    ):
        BaselineLifecycle.prepare(repository)


def test_grouped_split_rejects_held_out_levels_missing_from_training(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)
    mutate_csv_source(
        repository,
        "training_data",
        give_a_held_out_enrichment_group_unique_levels,
    )

    with pytest.raises(
        SplitManifestError,
        match=re.escape(
            "validation or test levels are absent from training: temperatures "
            "[126.0]; vibration displacement amplitudes [0.86]"
        ),
    ):
        BaselineLifecycle.prepare(repository)


def test_grouped_split_rejects_source_identity_drift(tmp_path: Path) -> None:
    repository = synthetic_repository(tmp_path)
    BaselineLifecycle.prepare(repository)
    mutate_csv_source(repository, "training_data", change_one_valid_aeps_value)

    with pytest.raises(
        SplitManifestError,
        match="split manifest source identity mismatch",
    ):
        BaselineLifecycle.prepare(repository)


def test_grouped_split_rejects_generation_metadata_drift(tmp_path: Path) -> None:
    repository = synthetic_repository(tmp_path)
    BaselineLifecycle.prepare(repository)
    manifest_path = repository / "data/splits/baseline_split_seed42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["generation"]["split_seed"] = 43
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SplitManifestError,
        match="split manifest generation metadata mismatch",
    ):
        BaselineLifecycle.prepare(repository)


def test_grouped_split_rejects_case_assignment_drift(tmp_path: Path) -> None:
    repository = synthetic_repository(tmp_path)
    BaselineLifecycle.prepare(repository)
    manifest_path = repository / "data/splits/baseline_split_seed42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["partition"] = "test"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        SplitManifestError,
        match="split manifest case assignment mismatch",
    ):
        BaselineLifecycle.prepare(repository)


def test_grouped_split_rejects_finished_file_identity_drift(tmp_path: Path) -> None:
    repository = synthetic_repository(tmp_path)
    BaselineLifecycle.prepare(repository)
    manifest_path = repository / "data/splits/baseline_split_seed42.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_path.write_text(
        json.dumps(manifest, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(
        SplitManifestError,
        match="split manifest identity mismatch: expected [0-9a-f]{64}, observed",
    ):
        BaselineLifecycle.prepare(repository)


def test_preflight_artifact_cannot_overwrite_the_locked_split_manifest(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path)
    BaselineLifecycle.prepare(repository)
    manifest_path = repository / "data/splits/baseline_split_seed42.json"
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(
        ValueError,
        match="artifact_path must not overwrite the authoritative split manifest",
    ):
        BaselineLifecycle.prepare(repository, artifact_path=manifest_path)

    assert manifest_path.read_bytes() == original_manifest


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
