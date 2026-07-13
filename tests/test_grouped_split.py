from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
import shutil

import pytest

from new_pino import BaselineLifecycle


REPOSITORY_ROOT = Path(__file__).parents[1]
MANIFEST_PATH = REPOSITORY_ROOT / "data/splits/baseline_split_seed42.json"
SOURCE_PATHS = {
    "training_data": REPOSITORY_ROOT / "data/combined_training_data.csv",
    "element_points": REPOSITORY_ROOT / "data/co_ind.csv",
    "material_properties": REPOSITORY_ROOT / "data/material_properties.md",
}
EXPECTED_SOURCE_CHECKSUMS = {
    "training_data": "d708af290ea778c44cb128d34a570b0758347323170f13587e00df9279286c6c",
    "element_points": "e0bc385f8078619fe1fec68d4e289da49f122ecb4497e6c232c30e304b689946",
    "material_properties": "1bd03993bc3620ad211254fa10357242f49b1bcc5797410c28a37fe006a5a323",
}
EXPECTED_CASE_PROJECTION_IDENTITY = (
    "675ae535cae1a33c41ea0512b5f820127a0c7b58fbee059525ae72d3ecc70927"
)
EXPECTED_MANIFEST_IDENTITY = (
    "9ede3e8b744cf608eb343cedcffd03c07332b206790424c14f2c6c963e85abc6"
)


def test_checked_in_real_split_locks_assignments_order_and_coverage() -> None:
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)

    assert sha256(manifest_bytes).hexdigest() == EXPECTED_MANIFEST_IDENTITY
    assert manifest["source_checksums"] == EXPECTED_SOURCE_CHECKSUMS
    cases = manifest["cases"]
    assert [case["partition"] for case in cases] == (
        ["training"] * 246 + ["validation"] * 51 + ["test"] * 54
    )

    groups = []
    for case_index in range(0, len(cases), 3):
        group_cases = cases[case_index : case_index + 3]
        group_keys = {
            (
                case["temperature_c"],
                case["vibration_displacement_amplitude_mm"],
                case["stratum"],
                case["partition"],
            )
            for case in group_cases
        }
        assert len(group_keys) == 1
        assert [case["pcb_youngs_modulus_gpa"] for case in group_cases] == [
            20.0,
            23.5,
            27.0,
        ]
        groups.append(next(iter(group_keys)))

    assert [
        sum(group[2] == stratum and group[3] == partition for group in groups)
        for partition, stratum in (
            ("training", "base_grid"),
            ("training", "enrichment"),
            ("validation", "base_grid"),
            ("validation", "enrichment"),
            ("test", "base_grid"),
            ("test", "enrichment"),
        )
    ] == [69, 13, 15, 2, 15, 3]

    training_groups = [group for group in groups if group[3] == "training"]
    held_out_groups = [group for group in groups if group[3] != "training"]
    assert {group[0] for group in held_out_groups} <= {
        group[0] for group in training_groups
    }
    assert {group[1] for group in held_out_groups} <= {
        group[1] for group in training_groups
    }

    assert [case["source_row"] for case in cases[:3]] == [334, 335, 336]
    assert [case["source_row"] for case in cases[243:246]] == [139, 140, 141]
    assert [case["source_row"] for case in cases[246:249]] == [295, 296, 297]
    assert [case["source_row"] for case in cases[294:297]] == [181, 182, 183]
    assert [case["source_row"] for case in cases[297:300]] == [286, 287, 288]
    assert [case["source_row"] for case in cases[-3:]] == [199, 200, 201]

    case_projection = [
        [
            case["source_row"],
            case["temperature_c"],
            case["vibration_displacement_amplitude_mm"],
            case["pcb_youngs_modulus_gpa"],
            case["stratum"],
            case["partition"],
        ]
        for case in cases
    ]
    assert sha256(
        json.dumps(case_projection, separators=(",", ":")).encode("utf-8")
    ).hexdigest() == EXPECTED_CASE_PROJECTION_IDENTITY


@pytest.mark.skipif(
    not all(path.is_file() for path in SOURCE_PATHS.values()),
    reason="authorized repository-local canonical sources are unavailable",
)
def test_real_sources_regenerate_and_reuse_the_locked_manifest(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    data_directory = repository / "data"
    data_directory.mkdir(parents=True)
    for source_path in SOURCE_PATHS.values():
        shutil.copy2(source_path, data_directory / source_path.name)

    preflight = BaselineLifecycle.prepare(repository)
    generated_path = repository / "data/splits/baseline_split_seed42.json"
    generated_bytes = generated_path.read_bytes()

    assert preflight.source_checksums == EXPECTED_SOURCE_CHECKSUMS
    assert generated_bytes == MANIFEST_PATH.read_bytes()
    assert sha256(generated_bytes).hexdigest() == sha256(
        MANIFEST_PATH.read_bytes()
    ).hexdigest()

    BaselineLifecycle.prepare(repository)
    assert generated_path.read_bytes() == generated_bytes
