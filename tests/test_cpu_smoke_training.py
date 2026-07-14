from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

import new_pino.training as training_module
from new_pino import BaselineLifecycle, TrainingContractError
from new_pino.preparation import PreparedPartition, _partition_content_identity
from test_source_preflight import canonical_identity, synthetic_repository


class _TestLockedPartitions(Mapping[str, PreparedPartition]):
    def __init__(self, partitions: Mapping[str, PreparedPartition]) -> None:
        self._partitions = partitions

    def __getitem__(self, name: str) -> PreparedPartition:
        if name == "test":
            raise AssertionError("training accessed the locked test partition")
        return self._partitions[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._partitions)

    def __len__(self) -> int:
        return len(self._partitions)


def test_cpu_smoke_training_retains_the_best_validation_checkpoint(
    tmp_path: Path,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path))
    prepared_with_locked_test = replace(
        prepared,
        partitions=_TestLockedPartitions(prepared.partitions),
    )

    result = BaselineLifecycle.train(
        prepared_with_locked_test,
        seed=0,
        artifact_directory=tmp_path / "smoke-run",
        smoke_max_epochs=3,
    )

    assert result.evidence_status == "noncanonical_cpu_smoke"
    assert result.test_partition_status == "locked_not_accessed"
    assert result.completed_epochs == 3
    assert result.best_epoch < result.completed_epochs
    assert result.checkpoint_path.is_file()
    assert result.validation_predictions_path.is_file()
    assert result.history_path.is_file()
    assert result.metadata_path.is_file()

    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == "baseline-training-run-v1"
    assert metadata["canonical"] is False
    assert metadata["evidence_status"] == "noncanonical_cpu_smoke"
    assert metadata["test_partition_status"] == "locked_not_accessed"
    assert metadata["architecture"] == {
        "kind": "dot_product_deeponet",
        "branch_widths": [5, 32, 64, 32, 16],
        "trunk_widths": [2, 32, 64, 32, 16],
        "hidden_activation": "tanh",
        "latent_activation": "linear",
        "fusion": "dot_product_plus_scalar_bias",
        "output_repair": "none",
        "trainable_parameter_count": 9729,
    }
    assert metadata["initialization"] == {
        "hidden_weights": "xavier_normal_tanh_gain_5_over_3",
        "latent_projection_weights": "xavier_normal_gain_1",
        "all_biases": "zero",
    }
    assert metadata["loss"] == {
        "kind": "equally_weighted_raw_aeps_mse",
        "target_transform": "none",
        "target_weighting": "none",
    }
    assert metadata["batching"] == {
        "item": "complete_48_point_simulation_case",
        "batch_size": 32,
        "drop_last": False,
        "shuffle": True,
        "generator_device": "cpu",
        "generator_seeded_once": True,
        "num_workers": 0,
    }
    assert metadata["optimizer"] == {
        "kind": "Adam",
        "learning_rate": 1e-3,
        "betas": [0.9, 0.999],
        "epsilon": 1e-8,
        "weight_decay": 0.0,
    }
    canonical = metadata["canonical_configuration"]
    assert canonical["backend"] == "cuda:0"
    assert canonical["seeds"] == [0, 1, 2, 3, 4]
    assert canonical["max_epochs"] == 3000
    assert canonical["deterministic_execution"] == {
        "algorithms": "strict",
        "cudnn_benchmark": False,
        "automatic_mixed_precision": False,
        "tensorfloat32": False,
        "cublas_workspace_configuration": ":4096:8",
        "cublas_configuration_timing": "before_cuda_initialization",
    }
    assert canonical["early_stopping"] == {
        "meaningful_progress_relative_threshold": 1e-3,
        "patience_epochs": 300,
    }
    assert canonical["scheduler"] == {
        "kind": "ReduceLROnPlateau",
        "mode": "min",
        "factor": 0.5,
        "patience": 75,
        "threshold": 1e-3,
        "threshold_mode": "rel",
        "min_learning_rate": 1e-6,
    }
    assert metadata["smoke_override"] == {
        "backend": "cpu",
        "max_epochs": 3,
    }
    assert metadata["precision"] == {
        "source_and_preprocessing": "float64",
        "model_inputs_targets_parameters_and_adam": "float32",
        "validation_predictions_targets_and_mse": "float64",
        "automatic_mixed_precision": False,
        "float16": False,
        "bfloat16": False,
        "tensorfloat32": False,
    }
    assert metadata["identities"] == {
        "source": prepared.preprocessing.source_identity,
        "split": prepared.preprocessing.split_identity,
        "preprocessing": prepared.preprocessing.content_identity,
        "feature_schema": prepared.preprocessing.feature_schema_identity,
        "unit_schema": prepared.preprocessing.unit_schema_identity,
        "run_configuration": result.run_configuration_identity,
    }
    assert metadata["selected_checkpoint"]["identity"] == result.checkpoint_identity
    assert metadata["selected_checkpoint"]["epoch"] == result.best_epoch
    assert (
        metadata["selected_checkpoint"]["validation_mse"] == result.best_validation_mse
    )
    assert metadata["environment"]["device_identifier"] == "cpu"
    assert metadata["environment"]["device_name"]
    assert metadata["environment"]["driver_version"] is None
    assert metadata["environment"]["python_version"]
    assert metadata["environment"]["numpy_version"] == np.__version__
    assert metadata["environment"]["pytorch_build"] == str(torch.__version__)
    assert metadata["environment"]["cuda_version"] == torch.version.cuda
    assert metadata["environment"]["deterministic_algorithms"] is True
    assert metadata["environment"]["deterministic_algorithms_warn_only"] is False
    assert metadata["environment"]["cudnn_benchmark"] is False
    assert metadata["environment"]["cudnn_tensorfloat32"] is False
    assert metadata["environment"]["matmul_tensorfloat32"] is False
    assert metadata["environment"]["automatic_mixed_precision"] is False
    assert (
        metadata["environment"]["cublas_workspace_configuration"] == ":4096:8"
    )
    assert metadata["configuration_identity"] == result.run_configuration_identity

    compatibility = metadata["compatibility"]
    assert compatibility == {
        "source_identity": prepared.preprocessing.source_identity,
        "split_identity": prepared.preprocessing.split_identity,
        "preprocessing_identity": prepared.preprocessing.content_identity,
        "configuration_identity": result.run_configuration_identity,
        "precision_identity": canonical_identity(metadata["precision"]),
        "backend_identity": canonical_identity(metadata["environment"]),
        "content_identities": {
            "training_partition": prepared.partitions["training"].content_identity,
            "validation_partition": prepared.partitions["validation"].content_identity,
        },
    }
    assert metadata["compatibility_identity"] == canonical_identity(compatibility)
    metadata_without_identity = dict(metadata)
    metadata_content_identity = metadata_without_identity.pop("content_identity")
    assert metadata_content_identity == canonical_identity(metadata_without_identity)

    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    assert history["compatibility"] == compatibility
    assert history["compatibility_identity"] == metadata["compatibility_identity"]
    history_without_identity = dict(history)
    history_content_identity = history_without_identity.pop("content_identity")
    assert history_content_identity == canonical_identity(history_without_identity)
    epochs = history["epochs"]
    assert len(epochs) == 3
    assert history["stopping_reason"] == "smoke_epoch_ceiling"
    assert history["optimizer_steps"] == 24
    assert history["initial_parameter_identity"] != history["final_parameter_identity"]
    assert [epoch["batch_sizes"] for epoch in epochs] == [[32] * 7 + [22]] * 3
    for epoch in epochs:
        shuffled_rows = [
            source_row for batch in epoch["batch_source_rows"] for source_row in batch
        ]
        assert sorted(shuffled_rows) == sorted(
            prepared.partitions["training"].source_rows
        )
        assert epoch["scheduler"]["learning_rate"] >= 1e-6
        assert epoch["early_stopping"]["epochs_without_meaningful_progress"] >= 0
    assert epochs[0]["batch_source_rows"] != epochs[1]["batch_source_rows"]
    expected_best = min(epochs, key=lambda epoch: epoch["validation_mse"])
    assert result.best_epoch == expected_best["epoch"]
    assert result.best_validation_mse == expected_best["validation_mse"]

    validation_predictions = np.load(result.validation_predictions_path)
    assert validation_predictions.dtype == np.float64
    assert validation_predictions.shape == (51, 48)
    expected_validation_mse = np.mean(
        (
            validation_predictions
            - prepared.partitions["validation"].raw_aeps_fields.astype(np.float64)
        )
        ** 2,
        dtype=np.float64,
    )
    assert expected_validation_mse == pytest.approx(
        result.best_validation_mse,
        rel=0.0,
        abs=1e-15,
    )

    checkpoint = torch.load(
        result.checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    assert checkpoint["checkpoint_identity"] == result.checkpoint_identity
    assert checkpoint["compatibility"] == compatibility
    assert checkpoint["compatibility_identity"] == metadata["compatibility_identity"]
    assert checkpoint["content_identity"] == checkpoint["checkpoint_identity"]
    assert checkpoint["model_state_identity"] == history["selected_parameter_identity"]
    assert len(checkpoint["optimizer_state_identity"]) == 64
    assert checkpoint["seed"] == 0
    assert checkpoint["epoch"] == result.best_epoch
    assert checkpoint["run_configuration_identity"] == result.run_configuration_identity
    assert (
        checkpoint["preprocessing_identity"] == prepared.preprocessing.content_identity
    )
    assert sum(tensor.numel() for tensor in checkpoint["model_state"].values()) == 9729
    assert all(
        tensor.dtype == torch.float32 for tensor in checkpoint["model_state"].values()
    )
    optimizer_state = checkpoint["optimizer_state"]
    optimizer_group = optimizer_state["param_groups"][0]
    assert optimizer_group["betas"] == (0.9, 0.999)
    assert optimizer_group["eps"] == 1e-8
    assert optimizer_group["weight_decay"] == 0.0
    expected_optimizer_steps = sum(
        len(epoch["batch_sizes"]) for epoch in epochs[: result.best_epoch]
    )
    assert {
        int(parameter_state["step"].item())
        for parameter_state in optimizer_state["state"].values()
    } == {expected_optimizer_steps}
    optimizer_tensors = [
        value
        for parameter_state in optimizer_state["state"].values()
        for value in parameter_state.values()
        if isinstance(value, torch.Tensor)
    ]
    assert optimizer_tensors
    assert all(tensor.dtype == torch.float32 for tensor in optimizer_tensors)


def test_cpu_smoke_shuffle_and_training_are_repeatable_for_one_seed(
    tmp_path: Path,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path))

    first = BaselineLifecycle.train(
        prepared,
        seed=3,
        artifact_directory=tmp_path / "first",
        smoke_max_epochs=2,
    )
    second = BaselineLifecycle.train(
        prepared,
        seed=3,
        artifact_directory=tmp_path / "second",
        smoke_max_epochs=2,
    )

    first_history = json.loads(first.history_path.read_text(encoding="utf-8"))
    second_history = json.loads(second.history_path.read_text(encoding="utf-8"))
    assert first_history == second_history
    assert first.checkpoint_identity == second.checkpoint_identity
    np.testing.assert_array_equal(
        np.load(first.validation_predictions_path),
        np.load(second.validation_predictions_path),
    )


def test_canonical_seed_execution_rejects_an_undeclared_seed_before_cuda_access(
    tmp_path: Path,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path))

    with pytest.raises(
        TrainingContractError,
        match="canonical training seed must be one of 0, 1, 2, 3, or 4",
    ):
        BaselineLifecycle.train(
            prepared,
            seed=5,
            artifact_directory=tmp_path / "canonical",
        )


def test_canonical_seed_execution_rejects_cublas_configuration_set_after_cuda(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path))
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(training_module, "_CUBLAS_CONFIGURED_BEFORE_CUDA", False)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)

    with pytest.raises(
        TrainingContractError,
        match="requires CUBLAS_WORKSPACE_CONFIG before CUDA initialization",
    ):
        BaselineLifecycle.train(
            prepared,
            seed=0,
            artifact_directory=tmp_path / "late-cublas",
        )


def test_seed_execution_rejects_an_incompatible_prepared_partition_binding(
    tmp_path: Path,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path))
    incompatible_training = replace(
        prepared.partitions["training"],
        split_identity="different-split",
    )
    incompatible = replace(
        prepared,
        partitions=MappingProxyType(
            {
                **prepared.partitions,
                "training": incompatible_training,
            }
        ),
    )
    artifact_directory = tmp_path / "incompatible"

    with pytest.raises(
        TrainingContractError,
        match="training partition split identity is incompatible",
    ):
        BaselineLifecycle.train(
            incompatible,
            seed=0,
            artifact_directory=artifact_directory,
            smoke_max_epochs=1,
        )

    assert not artifact_directory.exists()


def test_cpu_smoke_rejects_stale_partition_content_identity(
    tmp_path: Path,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path))
    changed_inputs = np.array(
        prepared.partitions["training"].branch_inputs,
        copy=True,
    )
    changed_inputs[0, 0] += 1.0
    stale_training = replace(
        prepared.partitions["training"],
        branch_inputs=changed_inputs,
    )
    stale = replace(
        prepared,
        partitions=MappingProxyType(
            {
                **prepared.partitions,
                "training": stale_training,
            }
        ),
    )

    with pytest.raises(
        TrainingContractError,
        match="training partition content identity is incompatible",
    ):
        BaselineLifecycle.train(
            stale,
            seed=0,
            artifact_directory=tmp_path / "stale-content",
            smoke_max_epochs=1,
        )


def test_cpu_smoke_validation_control_reduces_lr_and_stops_at_300_bad_epochs(
    tmp_path: Path,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path))
    training = replace(
        prepared.partitions["training"],
        source_rows=(prepared.partitions["training"].source_rows[0],),
        branch_inputs=np.zeros((1, 5), dtype=np.float32),
        trunk_inputs=np.zeros((48, 2), dtype=np.float32),
        raw_aeps_fields=np.ones((1, 48), dtype=np.float32),
    )
    training = replace(
        training,
        content_identity=_partition_content_identity(training),
    )
    validation = replace(
        prepared.partitions["validation"],
        source_rows=(prepared.partitions["validation"].source_rows[0],),
        branch_inputs=np.zeros((1, 5), dtype=np.float32),
        trunk_inputs=np.zeros((48, 2), dtype=np.float32),
        raw_aeps_fields=np.zeros((1, 48), dtype=np.float32),
    )
    validation = replace(
        validation,
        content_identity=_partition_content_identity(validation),
    )
    controlled = replace(
        prepared,
        partitions=MappingProxyType(
            {
                **prepared.partitions,
                "training": training,
                "validation": validation,
            }
        ),
    )

    result = BaselineLifecycle.train(
        controlled,
        seed=0,
        artifact_directory=tmp_path / "validation-control",
        smoke_max_epochs=400,
    )

    history = json.loads(result.history_path.read_text(encoding="utf-8"))
    epochs = history["epochs"]
    assert history["stopping_reason"] == "canonical_early_stopping"
    assert result.completed_epochs == 301
    assert result.best_epoch == 1
    assert epochs[0]["checkpoint"]["selected_this_epoch"] is True
    assert epochs[0]["early_stopping"]["meaningful_progress"] is True
    assert all(
        epoch["checkpoint"]["selected_this_epoch"] is False
        and epoch["early_stopping"]["meaningful_progress"] is False
        for epoch in epochs[1:]
    )
    assert min(epoch["scheduler"]["learning_rate"] for epoch in epochs) < 1e-3
    assert epochs[-1]["early_stopping"]["epochs_without_meaningful_progress"] == 300
    assert history["final_parameter_identity"] != history["selected_parameter_identity"]

    restored_predictions = np.load(result.validation_predictions_path)
    restored_mse = np.mean(restored_predictions**2, dtype=np.float64)
    assert restored_mse == pytest.approx(
        result.best_validation_mse,
        rel=0.0,
        abs=1e-15,
    )


@pytest.mark.parametrize(
    ("failed_stage", "expected_attempted_updates", "expected_completed_updates"),
    [
        ("training_loss", 0, 0),
        ("training_gradient", 0, 0),
        ("training_parameter", 1, 0),
        ("validation_prediction", 8, 8),
        ("validation_mse", 8, 8),
    ],
)
def test_cpu_smoke_controlled_numerical_faults_abort_and_record_the_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_stage: str,
    expected_attempted_updates: int,
    expected_completed_updates: int,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path))
    prepared_with_locked_test = replace(
        prepared,
        partitions=_TestLockedPartitions(prepared.partitions),
    )
    artifact_directory = tmp_path / failed_stage
    monkeypatch.setattr(training_module, "_CONTROLLED_FAULT_STAGE", failed_stage)

    with pytest.raises(TrainingContractError) as raised:
        BaselineLifecycle.train(
            prepared_with_locked_test,
            seed=2,
            artifact_directory=artifact_directory,
            smoke_max_epochs=2,
        )

    error = raised.value
    assert error.failed_stage == failed_stage
    assert error.failure_artifact_path is not None
    failure_artifact_path = error.failure_artifact_path
    assert failure_artifact_path.is_file()
    assert list(artifact_directory.iterdir()) == [failure_artifact_path]

    failure = json.loads(failure_artifact_path.read_text(encoding="utf-8"))
    assert failure["schema_version"] == "baseline-training-failure-v1"
    assert failure["status"] == "failed"
    assert failure["canonical"] is False
    assert failure["evidence_status"] == "noncanonical_cpu_smoke"
    assert failure["seed"] == 2
    assert failure["failed_stage"] == failed_stage
    assert failure["epoch"] == 1
    assert failure["optimizer_updates_attempted"] == expected_attempted_updates
    assert failure["optimizer_updates_completed"] == expected_completed_updates
    assert failure["test_partition_status"] == "locked_not_accessed"
    assert failure["run_configuration"]["controlled_fault_stage"] == failed_stage
    assert failure["compatibility_identity"] == canonical_identity(
        failure["compatibility"]
    )
    payload_without_identity = dict(failure)
    content_identity = payload_without_identity.pop("content_identity")
    assert content_identity == canonical_identity(payload_without_identity)
