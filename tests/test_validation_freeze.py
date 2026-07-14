from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from new_pino import (
    BaselineLifecycle,
    FreezeContractError,
    OperatingCondition,
    PredictionContractError,
    PredictionRequest,
    PredictorSelector,
    SeedTrainingResult,
)
from new_pino.preparation import PreparedDataArtifact, PreparedPartition
from test_source_preflight import synthetic_repository


EXPECTED_ARCHITECTURE = {
    "kind": "dot_product_deeponet",
    "branch_widths": [5, 32, 64, 32, 16],
    "trunk_widths": [2, 32, 64, 32, 16],
    "hidden_activation": "tanh",
    "latent_activation": "linear",
    "fusion": "dot_product_plus_scalar_bias",
    "output_repair": "none",
    "trainable_parameter_count": 9729,
}


def _identity(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _comparator_rmse(prepared: PreparedDataArtifact) -> float:
    training_mean = np.mean(
        prepared.partitions["training"].raw_aeps_fields,
        axis=0,
        dtype=np.float64,
    )
    validation_truth = prepared.partitions["validation"].raw_aeps_fields.astype(
        np.float64
    )
    return float(np.sqrt(np.mean((validation_truth - training_mean) ** 2)))


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _fake_seed_run(
    root: Path,
    prepared: PreparedDataArtifact,
    *,
    seed: int,
    rmse_ratio: float,
    finite: bool = True,
) -> SeedTrainingResult:
    run = BaselineLifecycle.train(
        prepared,
        seed=seed,
        artifact_directory=root / f"seed-{seed}",
        smoke_max_epochs=1,
    )
    comparator_rmse = _comparator_rmse(prepared)
    predictions = (
        prepared.partitions["validation"].raw_aeps_fields.astype(np.float64)
        + comparator_rmse * rmse_ratio
    )
    if not finite:
        predictions[0, 0] = np.nan
    np.save(run.validation_predictions_path, predictions)

    validation_mse = float((comparator_rmse * rmse_ratio) ** 2)
    metadata = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    metadata["selected_checkpoint"]["validation_mse"] = validation_mse
    metadata["validation_predictions"] = {
        "shape": [51, 48],
        "dtype": "float64",
        "content_identity": sha256(predictions.tobytes()).hexdigest(),
    }
    metadata["content_identity"] = _identity(
        {key: value for key, value in metadata.items() if key != "content_identity"}
    )
    _write_json(run.metadata_path, metadata)
    return replace(run, best_validation_mse=validation_mse)


def _runs(
    root: Path,
    prepared: PreparedDataArtifact,
    ratios: list[float],
) -> tuple[SeedTrainingResult, ...]:
    return tuple(
        _fake_seed_run(root, prepared, seed=seed, rmse_ratio=ratio)
        for seed, ratio in enumerate(ratios)
    )


class _LockedTestPartitions(Mapping[str, PreparedPartition]):
    def __init__(self, partitions: Mapping[str, PreparedPartition]) -> None:
        self._partitions = partitions

    def __getitem__(self, name: str) -> PreparedPartition:
        if name == "test":
            raise AssertionError("the locked test partition was accessed")
        return self._partitions[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._partitions)

    def __len__(self) -> int:
        return len(self._partitions)


def test_preparation_does_not_publicly_expose_the_locked_test_partition(
    tmp_path: Path,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path / "repository"))

    assert set(prepared.partitions) == {"training", "validation"}
    with pytest.raises(KeyError):
        prepared.partitions["test"]
    assert prepared.locked_test_binding.name == "test"
    assert prepared.locked_test_binding.case_count == 54


def test_full_validation_gate_pass_freezes_five_disclosed_predictors(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)
    locked = replace(prepared, partitions=_LockedTestPartitions(prepared.partitions))
    package_directory = tmp_path / "frozen"
    runs = _runs(tmp_path / "runs", prepared, [0.5] * 5)

    result = BaselineLifecycle.freeze(
        locked,
        runs,
        repository_root=repository,
        package_directory=package_directory,
    )

    assert result.gate_passed is True
    assert result.canonical is False
    assert result.evidence_status == "noncanonical_frozen_package"
    assert result.test_partition_status == "locked_ineligible_noncanonical"
    assert result.comparator_global_rmse == pytest.approx(_comparator_rmse(prepared))
    assert result.mean_seed_global_rmse == pytest.approx(
        result.comparator_global_rmse * 0.5
    )
    assert result.passing_seed_count == 5
    assert [seed.comparator_status for seed in result.seed_evidence] == ["passed"] * 5
    assert result.package_path == package_directory

    package = json.loads((package_directory / "package.json").read_text(encoding="utf-8"))
    assert package["schema_version"] == "baseline-frozen-package-v1"
    assert package["canonical"] is False
    assert package["gate_evidence"]["gate_passed"] is True
    assert package["source_preflight"]["status"] == "passed"
    assert package["source_preflight"]["source_checksums"] == dict(
        prepared.source_checksums
    )
    assert package["frozen_contract"] == {
        "source_identity": prepared.preprocessing.source_identity,
        "split_identity": prepared.preprocessing.split_identity,
        "preprocessing_identity": prepared.preprocessing.content_identity,
        "feature_schema_identity": prepared.preprocessing.feature_schema_identity,
        "unit_schema_identity": prepared.preprocessing.unit_schema_identity,
        "architecture": EXPECTED_ARCHITECTURE,
        "training_protocol_identity": result.protocol_identity,
        "selected_checkpoint_identities": [
            run.checkpoint_identity for run in runs
        ],
    }
    assert [item["seed"] for item in package["predictors"]] == [0, 1, 2, 3, 4]
    assert all(
        item["validation_comparator_status"] == "passed"
        for item in package["predictors"]
    )
    assert "default_predictor" not in package
    assert "best_seed" not in package
    assert "ensemble" not in package
    assert len(package["validation_report"]["case_order"]) == 51
    assert len(package["validation_report"]["seed_reports"]) == 5
    assert (package_directory / "runtime_sources" / "co_ind.csv").is_file()
    assert (
        package_directory / "runtime_sources" / "material_properties.md"
    ).is_file()
    for predictor in package["predictors"]:
        for path in predictor["artifacts"].values():
            assert (package_directory / path).is_file()


def test_one_nonpassing_seed_is_retained_when_majority_and_mean_pass(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)

    result = BaselineLifecycle.freeze(
        prepared,
        _runs(tmp_path / "runs", prepared, [0.5, 0.6, 0.7, 0.8, 1.1]),
        repository_root=repository,
        package_directory=tmp_path / "frozen",
    )

    assert result.gate_passed is True
    assert result.passing_seed_count == 4
    assert [seed.comparator_status for seed in result.seed_evidence] == [
        "passed",
        "passed",
        "passed",
        "passed",
        "not_passed",
    ]
    package = json.loads((result.package_path / "package.json").read_text(encoding="utf-8"))
    assert len(package["predictors"]) == 5
    assert package["predictors"][4]["validation_comparator_status"] == "not_passed"


@pytest.mark.parametrize(
    ("ratios", "reason"),
    [
        ([0.5, 0.6, 0.7, 1.1, 1.2], "insufficient_seed_majority"),
        ([0.9, 0.9, 0.9, 0.9, 2.0], "five_seed_mean_not_better"),
    ],
)
def test_nonpassing_gate_keeps_test_locked_and_requires_all_five_to_rerun(
    tmp_path: Path,
    ratios: list[float],
    reason: str,
) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)
    locked = replace(prepared, partitions=_LockedTestPartitions(prepared.partitions))
    output = tmp_path / "failed-freeze"

    result = BaselineLifecycle.freeze(
        locked,
        _runs(tmp_path / "runs", prepared, ratios),
        repository_root=repository,
        package_directory=output,
    )

    assert result.gate_passed is False
    assert result.failure_reasons == (reason,)
    assert result.test_partition_status == "locked_gate_failed"
    assert result.package_path is None
    assert result.revision_requirement == (
        "rerun_all_five_seeds_under_one_revised_common_protocol"
    )
    assert list(output.iterdir()) == [result.gate_artifact_path]
    assert not list(output.rglob("*test*"))


def test_incomplete_seed_set_is_rejected_without_test_artifacts(tmp_path: Path) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)
    output = tmp_path / "invalid-freeze"

    with pytest.raises(FreezeContractError, match="exactly seeds 0 through 4") as raised:
        BaselineLifecycle.freeze(
            prepared,
            _runs(tmp_path / "runs", prepared, [0.5] * 4),
            repository_root=repository,
            package_directory=output,
        )

    assert raised.value.gate_artifact_path == output / "freeze_gate.json"
    assert list(output.iterdir()) == [raised.value.gate_artifact_path]


def test_incompatible_seed_identities_are_rejected(tmp_path: Path) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)
    runs = list(_runs(tmp_path / "runs", prepared, [0.5] * 5))
    incompatible = json.loads(runs[4].metadata_path.read_text(encoding="utf-8"))
    incompatible["environment"]["protocol"] = "different-protocol"
    incompatible["content_identity"] = _identity(
        {key: value for key, value in incompatible.items() if key != "content_identity"}
    )
    _write_json(runs[4].metadata_path, incompatible)

    with pytest.raises(FreezeContractError, match="common training protocol"):
        BaselineLifecycle.freeze(
            prepared,
            runs,
            repository_root=repository,
            package_directory=tmp_path / "invalid-freeze",
        )


def test_different_backend_compatibility_identity_is_rejected(tmp_path: Path) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)
    runs = list(_runs(tmp_path / "runs", prepared, [0.5] * 5))
    incompatible = json.loads(runs[4].metadata_path.read_text(encoding="utf-8"))
    incompatible["compatibility"]["backend_identity"] = "different-backend"
    incompatible["compatibility_identity"] = _identity(
        incompatible["compatibility"]
    )
    incompatible["content_identity"] = _identity(
        {key: value for key, value in incompatible.items() if key != "content_identity"}
    )
    _write_json(runs[4].metadata_path, incompatible)
    history = json.loads(runs[4].history_path.read_text(encoding="utf-8"))
    history["compatibility"] = incompatible["compatibility"]
    history["compatibility_identity"] = incompatible["compatibility_identity"]
    history["content_identity"] = _identity(
        {key: value for key, value in history.items() if key != "content_identity"}
    )
    _write_json(runs[4].history_path, history)
    checkpoint = torch.load(
        runs[4].checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["compatibility"] = incompatible["compatibility"]
    checkpoint["compatibility_identity"] = incompatible["compatibility_identity"]
    torch.save(checkpoint, runs[4].checkpoint_path)

    with pytest.raises(FreezeContractError, match="compatible precision, backend"):
        BaselineLifecycle.freeze(
            prepared,
            runs,
            repository_root=repository,
            package_directory=tmp_path / "invalid-freeze",
        )


def test_corrupt_selected_checkpoint_is_rejected_before_package_eligibility(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)
    runs = list(_runs(tmp_path / "runs", prepared, [0.5] * 5))
    runs[4].checkpoint_path.write_bytes(b"corrupt checkpoint")

    with pytest.raises(FreezeContractError, match="checkpoint cannot be loaded"):
        BaselineLifecycle.freeze(
            prepared,
            runs,
            repository_root=repository,
            package_directory=tmp_path / "invalid-freeze",
        )


def test_incompatible_model_state_is_rejected_before_package_eligibility(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)
    runs = list(_runs(tmp_path / "runs", prepared, [0.5] * 5))
    checkpoint = torch.load(
        runs[4].checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    checkpoint["model_state"] = {}
    checkpoint["model_state_identity"] = sha256(b"mapping\0").hexdigest()
    torch.save(checkpoint, runs[4].checkpoint_path)

    with pytest.raises(FreezeContractError, match="model state is incompatible"):
        BaselineLifecycle.freeze(
            prepared,
            runs,
            repository_root=repository,
            package_directory=tmp_path / "invalid-freeze",
        )


def test_non_finite_validation_evidence_is_rejected(tmp_path: Path) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)
    runs = list(_runs(tmp_path / "runs", prepared, [0.5] * 4))
    runs.append(
        _fake_seed_run(
            tmp_path / "runs",
            prepared,
            seed=4,
            rmse_ratio=0.5,
            finite=False,
        )
    )

    with pytest.raises(FreezeContractError, match="finite validation predictions"):
        BaselineLifecycle.freeze(
            prepared,
            runs,
            repository_root=repository,
            package_directory=tmp_path / "invalid-freeze",
        )


def test_frozen_cpu_smoke_package_uses_explicit_prediction_contract_and_stays_locked(
    tmp_path: Path,
) -> None:
    repository = synthetic_repository(tmp_path / "repository")
    prepared = BaselineLifecycle.prepare(repository)
    runs = _runs(tmp_path / "training", prepared, [0.5] * 5)

    package_path = tmp_path / "frozen"
    freeze = BaselineLifecycle.freeze(
        prepared,
        runs,
        repository_root=repository,
        package_directory=package_path,
    )
    lifecycle = BaselineLifecycle.from_package(package_path)
    prediction = lifecycle.predict(
        PredictionRequest(
            OperatingCondition(22.5, 0.55, 23.5),
            PredictorSelector(seed=4),
        )
    )

    assert freeze.canonical is False
    assert len(prediction.aeps_field) == 48
    assert prediction.provenance.seed == 4
    assert prediction.provenance.checkpoint_identity == runs[4].checkpoint_identity
    assert prediction.provenance.validation_comparator_status == "passed"
    with pytest.raises(
        PredictionContractError,
        match="only a passing canonical frozen package can authorize",
    ):
        lifecycle.authorize_locked_test_partition(
            replace(prepared, partitions=_LockedTestPartitions(prepared.partitions))
        )
