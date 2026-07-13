from __future__ import annotations

import json
from hashlib import sha256
import math
from pathlib import Path
import re
import shutil

import pytest

from new_pino import (
    BaselineLifecycle,
    OperatingCondition,
    PredictionContractError,
    PredictionRequest,
    PredictorSelector,
)


FIXTURE_PACKAGE = Path(__file__).parent / "fixtures" / "prediction_package"
FIXTURE_SOURCES = FIXTURE_PACKAGE / "runtime_sources"
SYNTHETIC_TRAINING_SOURCE_CHECKSUM = (
    "fdc9c863cfd00b0d07f44940cb1e437301a5e422ed170dde9de3128c271f2497"
)


def fixture_lifecycle() -> BaselineLifecycle:
    return BaselineLifecycle.from_package(FIXTURE_PACKAGE)


def test_explicit_fixture_seed_returns_one_ordered_aeps_field_with_provenance() -> None:
    lifecycle = fixture_lifecycle()

    result = lifecycle.predict(
        PredictionRequest(
            operating_condition=OperatingCondition(
                temperature_c=22.5,
                vibration_displacement_amplitude_mm=0.55,
                pcb_youngs_modulus_gpa=23.5,
            ),
            predictor=PredictorSelector(seed=0),
        )
    )

    assert result.element_indices == tuple(range(1, 49))
    assert len(result.aeps_field) == 48
    assert all(math.isfinite(value) for value in result.aeps_field)
    assert result.evidence_status == "noncanonical_fixture"
    assert result.provenance.seed == 0
    assert result.provenance.checkpoint_identity == "synthetic-fixture-checkpoint-0"
    assert result.provenance.validation_comparator_status == "passed"
    assert set(result.provenance.source_checksums) == {
        "training_data",
        "element_points",
        "material_properties",
    }
    assert result.provenance.split_identity == "synthetic-fixture-split-v1"
    assert result.provenance.run_configuration_identity == "synthetic-fixture-run-v1"


def test_explicit_checkpoint_identity_resolves_the_same_frozen_predictor() -> None:
    lifecycle = fixture_lifecycle()
    condition = OperatingCondition(22.5, 0.55, 23.5)

    by_seed = lifecycle.predict(
        PredictionRequest(condition, PredictorSelector(seed=0))
    )
    by_checkpoint = lifecycle.predict(
        PredictionRequest(
            condition,
            PredictorSelector(checkpoint_identity="synthetic-fixture-checkpoint-0"),
        )
    )

    assert by_checkpoint == by_seed


@pytest.mark.parametrize(
    ("selector", "message"),
    [
        (PredictorSelector(), "explicit seed or checkpoint identity is required"),
        (
            PredictorSelector(
                seed=0,
                checkpoint_identity="synthetic-fixture-checkpoint-0",
            ),
            "exactly one predictor identity",
        ),
        (PredictorSelector(seed=99), "seed 99 does not identify one predictor"),
        (
            PredictorSelector(checkpoint_identity="best"),
            "checkpoint identity 'best' does not identify one predictor",
        ),
        (
            PredictorSelector(checkpoint_identity="average"),
            "checkpoint identity 'average' does not identify one predictor",
        ),
    ],
)
def test_implicit_ambiguous_best_and_aggregate_predictor_selection_are_rejected(
    selector: PredictorSelector, message: str
) -> None:
    lifecycle = fixture_lifecycle()

    with pytest.raises(PredictionContractError, match=re.escape(message)):
        lifecycle.predict(
            PredictionRequest(
                OperatingCondition(22.5, 0.55, 23.5),
                selector,
            )
        )


@pytest.mark.parametrize(
    ("condition", "message"),
    [
        (
            OperatingCondition(float("nan"), 0.55, 23.5),
            "temperature must be finite",
        ),
        (
            OperatingCondition(-40.01, 0.55, 23.5),
            "temperature -40.01 is outside the supported inclusive interval [-40, 125]",
        ),
        (
            OperatingCondition(22.5, float("inf"), 23.5),
            "vibration displacement amplitude must be finite",
        ),
        (
            OperatingCondition(22.5, 0.91, 23.5),
            (
                "vibration displacement amplitude 0.91 is outside the supported "
                "inclusive interval [0.2, 0.9]"
            ),
        ),
        (
            OperatingCondition(22.5, 0.55, float("-inf")),
            "PCB Young's modulus must be finite",
        ),
        (
            OperatingCondition(22.5, 0.55, 27.01),
            "PCB Young's modulus 27.01 is outside the supported inclusive interval [20, 27]",
        ),
    ],
)
def test_unsupported_operating_conditions_are_rejected_actionably(
    condition: OperatingCondition, message: str
) -> None:
    lifecycle = fixture_lifecycle()

    with pytest.raises(PredictionContractError, match=re.escape(message)):
        lifecycle.predict(PredictionRequest(condition, PredictorSelector(seed=0)))


@pytest.mark.parametrize(
    "condition",
    [
        OperatingCondition(-40.0, 0.2, 20.0),
        OperatingCondition(125.0, 0.9, 27.0),
    ],
)
def test_inclusive_operating_condition_boundaries_are_supported(
    condition: OperatingCondition,
) -> None:
    result = fixture_lifecycle().predict(
        PredictionRequest(condition, PredictorSelector(seed=0))
    )

    assert len(result.aeps_field) == 48


def test_caller_supplied_element_points_are_rejected() -> None:
    request = PredictionRequest(
        OperatingCondition(22.5, 0.55, 23.5),
        PredictorSelector(seed=0),
        element_points=((99.0, 99.0),),
    )

    with pytest.raises(
        PredictionContractError,
        match="caller-supplied element points are unsupported",
    ):
        fixture_lifecycle().predict(request)


@pytest.mark.parametrize(
    ("source_name", "binding_name"),
    [
        ("co_ind.csv", "element-point"),
        ("material_properties.md", "material-property"),
    ],
)
def test_bound_runtime_source_mismatches_identify_expected_and_actual_checksums(
    tmp_path: Path, source_name: str, binding_name: str
) -> None:
    package = tmp_path / "prediction_package"
    shutil.copytree(FIXTURE_PACKAGE, package)
    sources = package / "runtime_sources"
    changed_source = sources / source_name
    changed_source.write_text(
        changed_source.read_text(encoding="utf-8") + "\nchanged\n",
        encoding="utf-8",
    )
    lifecycle = BaselineLifecycle.from_package(package)

    with pytest.raises(
        PredictionContractError,
        match=(
            rf"{binding_name} source .* checksum mismatch: expected [0-9a-f]{{64}}, "
            rf"observed [0-9a-f]{{64}}"
        ),
    ):
        lifecycle.predict(
            PredictionRequest(
                OperatingCondition(22.5, 0.55, 23.5),
                PredictorSelector(seed=0),
            )
        )


def test_training_source_checksum_is_provenance_without_runtime_target_reads() -> None:
    assert not (FIXTURE_SOURCES / "combined_training_data.csv").exists()

    result = fixture_lifecycle().predict(
        PredictionRequest(
            OperatingCondition(22.5, 0.55, 23.5),
            PredictorSelector(seed=0),
        )
    )

    assert (
        result.provenance.source_checksums["training_data"]
        == SYNTHETIC_TRAINING_SOURCE_CHECKSUM
    )


def copied_package_metadata(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    package = tmp_path / "prediction_package"
    shutil.copytree(FIXTURE_PACKAGE, package)
    metadata = json.loads((package / "package.json").read_text(encoding="utf-8"))
    return package, metadata


def write_package_metadata(package: Path, metadata: dict[str, object]) -> None:
    (package / "package.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def load_copied_package(package: Path) -> BaselineLifecycle:
    return BaselineLifecycle.from_package(package)


def test_fixture_package_cannot_claim_canonical_evidence(tmp_path: Path) -> None:
    package, metadata = copied_package_metadata(tmp_path)
    metadata["canonical"] = True
    write_package_metadata(package, metadata)

    with pytest.raises(
        PredictionContractError,
        match="fixture package must be machine-visibly noncanonical",
    ):
        load_copied_package(package)


def test_fixture_package_rejects_deeponet_architecture_drift(tmp_path: Path) -> None:
    package, metadata = copied_package_metadata(tmp_path)
    architecture = metadata["architecture"]
    assert isinstance(architecture, dict)
    architecture["branch_widths"] = [5, 64, 16]
    write_package_metadata(package, metadata)

    with pytest.raises(
        PredictionContractError,
        match=re.escape(
            "fixture architecture branch_widths must be [5, 32, 64, 32, 16]"
        ),
    ):
        load_copied_package(package)


def test_fixture_package_requires_complete_saved_prediction_state(
    tmp_path: Path,
) -> None:
    package, metadata = copied_package_metadata(tmp_path)
    preprocessing = metadata["preprocessing"]
    assert isinstance(preprocessing, dict)
    preprocessing["element_points_mm"] = preprocessing["element_points_mm"][:-1]
    write_package_metadata(package, metadata)

    with pytest.raises(
        PredictionContractError,
        match="fixture preprocessing must bind exactly 48 ordered element points",
    ):
        load_copied_package(package)


def test_saved_element_point_reordering_is_rejected_against_bound_source(
    tmp_path: Path,
) -> None:
    package, metadata = copied_package_metadata(tmp_path)
    preprocessing = metadata["preprocessing"]
    assert isinstance(preprocessing, dict)
    points = preprocessing["element_points_mm"]
    assert isinstance(points, list)
    points[0], points[1] = points[1], points[0]
    write_package_metadata(package, metadata)
    lifecycle = load_copied_package(package)

    with pytest.raises(
        PredictionContractError,
        match=(
            "saved element-point binding does not match the bound coordinate "
            "source order"
        ),
    ):
        lifecycle.predict(
            PredictionRequest(
                OperatingCondition(22.5, 0.55, 23.5),
                PredictorSelector(seed=0),
            )
        )


def test_saved_material_metadata_must_match_the_bound_material_table(
    tmp_path: Path,
) -> None:
    package, metadata = copied_package_metadata(tmp_path)
    preprocessing = metadata["preprocessing"]
    assert isinstance(preprocessing, dict)
    material = preprocessing["material"]
    assert isinstance(material, dict)
    moduli = material["youngs_modulus_pa"]
    assert isinstance(moduli, list)
    moduli[0] += 1
    write_package_metadata(package, metadata)
    lifecycle = load_copied_package(package)

    with pytest.raises(
        PredictionContractError,
        match=(
            "saved material metadata does not match the bound material-property "
            "source table"
        ),
    ):
        lifecycle.predict(
            PredictionRequest(
                OperatingCondition(22.5, 0.55, 23.5),
                PredictorSelector(seed=0),
            )
        )


def test_duplicate_bound_element_points_are_rejected(tmp_path: Path) -> None:
    package, metadata = copied_package_metadata(tmp_path)
    coordinate_source = package / "runtime_sources" / "co_ind.csv"
    coordinate_lines = coordinate_source.read_text(encoding="utf-8").splitlines()
    coordinate_lines[2] = coordinate_lines[1]
    coordinate_source.write_text(
        "\n".join(coordinate_lines) + "\n",
        encoding="utf-8",
    )

    preprocessing = metadata["preprocessing"]
    assert isinstance(preprocessing, dict)
    points = preprocessing["element_points_mm"]
    assert isinstance(points, list)
    points[1] = points[0]
    checksums = metadata["source_checksums"]
    assert isinstance(checksums, dict)
    checksums["element_points"] = sha256(coordinate_source.read_bytes()).hexdigest()
    write_package_metadata(package, metadata)
    lifecycle = load_copied_package(package)

    with pytest.raises(
        PredictionContractError,
        match="bound element-point source .* contains duplicate element points",
    ):
        lifecycle.predict(
            PredictionRequest(
                OperatingCondition(22.5, 0.55, 23.5),
                PredictorSelector(seed=0),
            )
        )
