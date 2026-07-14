from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch

from new_pino import BaselineLifecycle
from test_source_preflight import synthetic_repository


def test_cpu_smoke_training_retains_the_best_validation_checkpoint(
    tmp_path: Path,
) -> None:
    prepared = BaselineLifecycle.prepare(synthetic_repository(tmp_path))
    inaccessible_test = replace(
        prepared.partitions["test"],
        raw_aeps_fields=np.full_like(
            prepared.partitions["test"].raw_aeps_fields,
            np.nan,
        ),
    )
    prepared_with_locked_test = replace(
        prepared,
        partitions=MappingProxyType(
            {
                **prepared.partitions,
                "test": inaccessible_test,
            }
        ),
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
    assert canonical["max_epochs"] == 3000
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
    assert metadata["environment"]["deterministic_algorithms"] is True
    assert metadata["environment"]["cudnn_benchmark"] is False

    history = json.loads(result.history_path.read_text(encoding="utf-8"))
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
    validation = replace(
        prepared.partitions["validation"],
        source_rows=(prepared.partitions["validation"].source_rows[0],),
        branch_inputs=np.zeros((1, 5), dtype=np.float32),
        trunk_inputs=np.zeros((48, 2), dtype=np.float32),
        raw_aeps_fields=np.zeros((1, 48), dtype=np.float32),
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
