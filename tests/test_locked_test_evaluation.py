from __future__ import annotations

import csv
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import shutil
from statistics import stdev
from typing import cast

import pytest

from new_pino import (
    BaselineLifecycle,
    EvaluationContractError,
    PredictionContractError,
    PreparedDataArtifact,
    PreparedPartition,
)
from test_source_preflight import synthetic_repository


def _content_identity(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


@pytest.fixture(scope="module")
def passing_fixture_evaluation(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, PreparedDataArtifact, Path]:
    root = tmp_path_factory.mktemp("locked-test-evaluation")
    repository = synthetic_repository(root)
    initial = BaselineLifecycle.prepare(repository)
    validation_rows = set(initial.partitions["validation"].source_rows)
    source_path = repository / "data" / "combined_training_data.csv"
    with source_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    for source_row, row in enumerate(rows[1:], start=2):
        value = 1e-6 if source_row in validation_rows else 1.3
        row[3:] = [str(value)] * 48
    with source_path.open("w", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerows(rows)
    (repository / "data" / "splits" / "baseline_split_seed42.json").unlink()
    prepared = BaselineLifecycle.prepare(repository)

    runs = tuple(
        BaselineLifecycle.train(
            prepared,
            seed=seed,
            artifact_directory=root / "runs" / f"seed-{seed}",
            smoke_max_epochs=1,
        )
        for seed in range(5)
    )
    package_path = root / "frozen"
    freeze = BaselineLifecycle.freeze(
        prepared,
        runs,
        repository_root=repository,
        package_directory=package_path,
    )
    assert freeze.gate_passed is True
    assert freeze.canonical is False
    assert [item.comparator_status for item in freeze.seed_evidence] == [
        "passed",
        "passed",
        "not_passed",
        "passed",
        "passed",
    ]

    partition = prepared._locked_test_partition
    cases = [
        {
            "simulation_case_identity": f"synthetic_source_row_{source_row}",
            "branch_inputs": branch.tolist(),
            "ground_truth_aeps": truth.tolist(),
        }
        for source_row, branch, truth in zip(
            partition.source_rows,
            partition.branch_inputs,
            partition.raw_aeps_fields,
            strict=True,
        )
    ]
    partition_authority_identity = _content_identity(
        {
            "schema_version": "aeps-evaluation-partition-fixture-v1",
            "source_checksums": dict(prepared.source_checksums),
            "source_identity": prepared.preprocessing.source_identity,
            "split_identity": prepared.preprocessing.split_identity,
            "preprocessing_identity": prepared.preprocessing.content_identity,
            "case_order_basis": "manifest",
            "cases": cases,
        }
    )
    fixture = {
        "schema_version": "aeps-evaluation-partition-fixture-v1",
        "canonical": False,
        "evidence_status": "noncanonical_fixture",
        "source_checksums": dict(prepared.source_checksums),
        "source_identity": prepared.preprocessing.source_identity,
        "split_identity": prepared.preprocessing.split_identity,
        "preprocessing_identity": prepared.preprocessing.content_identity,
        "run_configuration_identity": freeze.protocol_identity,
        "partition_authority": {
            "kind": "authorized_fixture",
            "identity": partition_authority_identity,
        },
        "case_order_basis": "manifest",
        "cases": cases,
    }
    fixture_path = root / "evaluation_partition_fixture.json"
    fixture_path.write_text(
        json.dumps(fixture, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return package_path, prepared, fixture_path


def test_passing_frozen_fixture_package_evaluates_all_five_seeds(
    passing_fixture_evaluation: tuple[Path, PreparedDataArtifact, Path],
    tmp_path: Path,
) -> None:
    package_path, prepared, fixture_path = passing_fixture_evaluation
    report_path = tmp_path / "fixture_test_report.json"
    package_before = {
        path.relative_to(package_path): sha256(path.read_bytes()).hexdigest()
        for path in package_path.rglob("*")
        if path.is_file()
    }
    fixture_before = fixture_path.read_bytes()

    report = BaselineLifecycle.from_package(package_path).evaluate(
        fixture_path,
        artifact_path=report_path,
    )

    assert report.canonical is False
    assert report.evidence_status == "noncanonical_fixture"
    assert report.partition_authority_kind == "authorized_fixture"
    assert report.partition_authority_identity != (
        prepared.locked_test_binding.content_identity
    )
    assert len(report.case_order) == 54
    assert all(case.startswith("synthetic_source_row_") for case in report.case_order)
    assert [seed.seed for seed in report.seed_reports] == [0, 1, 2, 3, 4]
    assert [seed.validation_comparator_status for seed in report.seed_reports] == [
        "passed",
        "passed",
        "not_passed",
        "passed",
        "passed",
    ]
    assert all(len(seed.case_metrics) == 54 for seed in report.seed_reports)
    seed_rmse = [seed.global_rmse for seed in report.seed_reports]
    assert report.cross_seed_summary["global_rmse"].mean == pytest.approx(
        sum(seed_rmse) / 5
    )
    assert report.cross_seed_summary[
        "global_rmse"
    ].sample_standard_deviation == pytest.approx(stdev(seed_rmse))
    assert "predictions" not in json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture_path.read_bytes() == fixture_before
    assert {
        path.relative_to(package_path): sha256(path.read_bytes()).hexdigest()
        for path in package_path.rglob("*")
        if path.is_file()
    } == package_before

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    identity = payload.pop("report_content_identity")
    assert identity == _content_identity(payload)


def test_generated_fixture_authority_is_bound_to_its_exact_payload(
    passing_fixture_evaluation: tuple[Path, PreparedDataArtifact, Path],
    tmp_path: Path,
) -> None:
    package_path, _, fixture_path = passing_fixture_evaluation
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    cases = fixture["cases"]
    assert isinstance(cases, list)
    first_case = cases[0]
    assert isinstance(first_case, dict)
    truth = first_case["ground_truth_aeps"]
    assert isinstance(truth, list)
    truth[0] = float(truth[0]) + 1e-7
    changed_fixture = tmp_path / "changed_fixture.json"
    changed_fixture.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(
        EvaluationContractError,
        match="partition-authority identity does not match",
    ):
        BaselineLifecycle.from_package(package_path).evaluate(changed_fixture)


def test_generated_fixture_requires_a_noncanonical_frozen_package(
    passing_fixture_evaluation: tuple[Path, PreparedDataArtifact, Path],
    tmp_path: Path,
) -> None:
    package_path, _, fixture_path = passing_fixture_evaluation
    canonical_path = tmp_path / "forged-canonical"
    shutil.copytree(package_path, canonical_path)
    metadata_path = canonical_path / "package.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["canonical"] = True
    metadata["evidence_status"] = "canonical_frozen_package"
    metadata["test_partition_status"] = "eligible_locked_test"
    gate = metadata["gate_evidence"]
    gate["canonical"] = True
    gate["evidence_status"] = "canonical_frozen_package"
    gate["test_partition_status"] = "eligible_locked_test"
    gate_without_identity = dict(gate)
    gate_without_identity.pop("content_identity")
    gate["content_identity"] = _content_identity(gate_without_identity)
    metadata_without_identity = dict(metadata)
    metadata_without_identity.pop("package_content_identity")
    metadata["package_content_identity"] = _content_identity(
        metadata_without_identity
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        EvaluationContractError,
        match="noncanonical passing frozen package",
    ):
        BaselineLifecycle.from_package(canonical_path).evaluate(fixture_path)


def test_evaluation_report_cannot_overwrite_the_frozen_package(
    passing_fixture_evaluation: tuple[Path, PreparedDataArtifact, Path],
    tmp_path: Path,
) -> None:
    package_path, _, fixture_path = passing_fixture_evaluation
    copied_package = tmp_path / "frozen"
    shutil.copytree(package_path, copied_package)
    metadata_path = copied_package / "package.json"
    metadata_before = metadata_path.read_bytes()

    with pytest.raises(
        EvaluationContractError,
        match="outside the loaded package",
    ):
        BaselineLifecycle.from_package(copied_package).evaluate(
            fixture_path,
            artifact_path=metadata_path,
        )

    assert metadata_path.read_bytes() == metadata_before


def test_frozen_package_rejects_cross_seed_pooling_mismatch(
    passing_fixture_evaluation: tuple[Path, PreparedDataArtifact, Path],
    tmp_path: Path,
) -> None:
    package_path, _, _ = passing_fixture_evaluation
    mismatched_path = tmp_path / "mixed-software"
    shutil.copytree(package_path, mismatched_path)
    package_metadata_path = mismatched_path / "package.json"
    package = json.loads(package_metadata_path.read_text(encoding="utf-8"))
    predictor = package["predictors"][4]
    compatibility = predictor["compatibility"]
    compatibility["software_identity"] = "different-software"
    predictor["compatibility_identity"] = _content_identity(compatibility)
    predictor_metadata_path = mismatched_path / predictor["artifacts"]["metadata"]
    predictor_metadata = json.loads(
        predictor_metadata_path.read_text(encoding="utf-8")
    )
    predictor_metadata["compatibility"] = compatibility
    predictor_metadata["compatibility_identity"] = predictor[
        "compatibility_identity"
    ]
    predictor_metadata_without_identity = dict(predictor_metadata)
    predictor_metadata_without_identity.pop("content_identity")
    predictor_metadata["content_identity"] = _content_identity(
        predictor_metadata_without_identity
    )
    predictor_metadata_path.write_text(
        json.dumps(predictor_metadata),
        encoding="utf-8",
    )
    package_without_identity = dict(package)
    package_without_identity.pop("package_content_identity")
    package["package_content_identity"] = _content_identity(
        package_without_identity
    )
    package_metadata_path.write_text(json.dumps(package), encoding="utf-8")

    with pytest.raises(
        PredictionContractError,
        match="compatible precision, backend, software, and partition identities",
    ):
        BaselineLifecycle.from_package(mismatched_path)


def test_frozen_predictor_metadata_must_match_the_package_protocol(
    passing_fixture_evaluation: tuple[Path, PreparedDataArtifact, Path],
    tmp_path: Path,
) -> None:
    package_path, _, _ = passing_fixture_evaluation
    mismatched_path = tmp_path / "mixed-protocol"
    shutil.copytree(package_path, mismatched_path)
    package = json.loads(
        (mismatched_path / "package.json").read_text(encoding="utf-8")
    )
    predictor = package["predictors"][4]
    metadata_path = mismatched_path / predictor["artifacts"]["metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["optimizer"]["learning_rate"] = 0.5
    metadata_without_identity = dict(metadata)
    metadata_without_identity.pop("content_identity")
    metadata["content_identity"] = _content_identity(metadata_without_identity)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(
        PredictionContractError,
        match="retained metadata is incompatible",
    ):
        BaselineLifecycle.from_package(mismatched_path)


class _UnreadableLockedTestPartition:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError("the locked test partition was accessed")


def test_noncanonical_freeze_is_rejected_before_locked_partition_access(
    passing_fixture_evaluation: tuple[Path, PreparedDataArtifact, Path],
) -> None:
    package_path, prepared, _ = passing_fixture_evaluation
    guarded = replace(
        prepared,
        _locked_test_partition=cast(
            PreparedPartition,
            _UnreadableLockedTestPartition(),
        ),
    )

    with pytest.raises(
        PredictionContractError,
        match="only a passing canonical frozen package can authorize",
    ):
        BaselineLifecycle.from_package(package_path).evaluate(guarded)


@pytest.mark.parametrize(
    ("evidence_case", "message"),
    [
        ("missing", "frozen gate evidence must be a JSON object"),
        ("failed", "a frozen package requires a passing gate"),
        ("incomplete", "frozen package must contain exactly five"),
        ("identity_incompatible", "frozen package gate evidence is incompatible"),
        ("rehashed_failed", "frozen package gate evidence is incompatible"),
        ("rehashed_incomplete", "frozen package gate evidence is incompatible"),
        ("rehashed_protocol", "frozen package gate evidence is incompatible"),
    ],
)
def test_invalid_freeze_evidence_is_rejected_before_partition_access(
    passing_fixture_evaluation: tuple[Path, PreparedDataArtifact, Path],
    tmp_path: Path,
    evidence_case: str,
    message: str,
) -> None:
    package_path, prepared, _ = passing_fixture_evaluation
    tampered_path = tmp_path / evidence_case
    shutil.copytree(package_path, tampered_path)
    metadata_path = tampered_path / "package.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if evidence_case == "missing":
        metadata.pop("gate_evidence")
    elif evidence_case == "failed":
        metadata["gate_status"] = "failed"
    elif evidence_case == "incomplete":
        metadata["predictors"].pop()
    elif evidence_case == "identity_incompatible":
        metadata["gate_evidence"]["content_identity"] = "stale"
    else:
        gate = metadata["gate_evidence"]
        if evidence_case == "rehashed_failed":
            gate["gate_status"] = "failed"
        elif evidence_case == "rehashed_incomplete":
            gate["seed_evidence"].pop()
        else:
            gate["protocol_identity"] = "different-protocol"
        gate_without_identity = dict(gate)
        gate_without_identity.pop("content_identity")
        gate["content_identity"] = _content_identity(gate_without_identity)
    metadata_without_identity = dict(metadata)
    metadata_without_identity.pop("package_content_identity")
    metadata["package_content_identity"] = _content_identity(
        metadata_without_identity
    )
    metadata_path.write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    guarded = replace(
        prepared,
        _locked_test_partition=cast(
            PreparedPartition,
            _UnreadableLockedTestPartition(),
        ),
    )

    with pytest.raises(PredictionContractError, match=message):
        BaselineLifecycle.from_package(tampered_path).evaluate(guarded)
