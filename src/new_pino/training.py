"""Deterministic seed training through the public baseline boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
import pickle
from pathlib import Path
import platform
import random
import subprocess
import time
from typing import Any, Literal, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .preparation import (
    PreparedDataArtifact,
    _partition_content_identity,
)


_BRANCH_WIDTHS = (5, 32, 64, 32, 16)
_TRUNK_WIDTHS = (2, 32, 64, 32, 16)
_BATCH_SIZE = 32
_SHUFFLE = True
_DROP_LAST = False
_NUM_WORKERS = 0
_LEARNING_RATE = 1e-3
_ADAM_BETAS = (0.9, 0.999)
_ADAM_EPSILON = 1e-8
_WEIGHT_DECAY = 0.0
_SCHEDULER_MODE: Literal["min"] = "min"
_SCHEDULER_FACTOR = 0.5
_SCHEDULER_PATIENCE = 75
_MEANINGFUL_PROGRESS_THRESHOLD = 1e-3
_THRESHOLD_MODE: Literal["rel"] = "rel"
_MIN_LEARNING_RATE = 1e-6
_EARLY_STOPPING_PATIENCE = 300
_CANONICAL_MAX_EPOCHS = 3000
_CANONICAL_SEEDS = (0, 1, 2, 3, 4)
_EXPECTED_ARCHITECTURE: Mapping[str, object] = {
    "kind": "dot_product_deeponet",
    "branch_widths": list(_BRANCH_WIDTHS),
    "trunk_widths": list(_TRUNK_WIDTHS),
    "hidden_activation": "tanh",
    "latent_activation": "linear",
    "fusion": "dot_product_plus_scalar_bias",
    "output_repair": "none",
    "trainable_parameter_count": 9729,
}
_INITIALIZATION = {
    "hidden_weights": "xavier_normal_tanh_gain_5_over_3",
    "latent_projection_weights": "xavier_normal_gain_1",
    "all_biases": "zero",
}
_LOSS = {
    "kind": "equally_weighted_raw_aeps_mse",
    "target_transform": "none",
    "target_weighting": "none",
}
_BATCHING = {
    "item": "complete_48_point_simulation_case",
    "batch_size": _BATCH_SIZE,
    "drop_last": _DROP_LAST,
    "shuffle": _SHUFFLE,
    "generator_device": "cpu",
    "generator_seeded_once": True,
    "num_workers": _NUM_WORKERS,
}
_OPTIMIZER = {
    "kind": "Adam",
    "learning_rate": _LEARNING_RATE,
    "betas": list(_ADAM_BETAS),
    "epsilon": _ADAM_EPSILON,
    "weight_decay": _WEIGHT_DECAY,
}
_SCHEDULER = {
    "kind": "ReduceLROnPlateau",
    "mode": _SCHEDULER_MODE,
    "factor": _SCHEDULER_FACTOR,
    "patience": _SCHEDULER_PATIENCE,
    "threshold": _MEANINGFUL_PROGRESS_THRESHOLD,
    "threshold_mode": _THRESHOLD_MODE,
    "min_learning_rate": _MIN_LEARNING_RATE,
}
_EARLY_STOPPING = {
    "meaningful_progress_relative_threshold": _MEANINGFUL_PROGRESS_THRESHOLD,
    "patience_epochs": _EARLY_STOPPING_PATIENCE,
}
_CUBLAS_WORKSPACE_CONFIGURATION = ":4096:8"
_DETERMINISTIC_EXECUTION = {
    "algorithms": "strict",
    "cudnn_benchmark": False,
    "automatic_mixed_precision": False,
    "tensorfloat32": False,
    "cublas_workspace_configuration": _CUBLAS_WORKSPACE_CONFIGURATION,
    "cublas_configuration_timing": "before_cuda_initialization",
}
_CANONICAL_CONFIGURATION = {
    "backend": "cuda:0",
    "seeds": list(_CANONICAL_SEEDS),
    "max_epochs": _CANONICAL_MAX_EPOCHS,
    "deterministic_execution": _DETERMINISTIC_EXECUTION,
    "early_stopping": _EARLY_STOPPING,
    "scheduler": _SCHEDULER,
}
_PRECISION = {
    "source_and_preprocessing": "float64",
    "model_inputs_targets_parameters_and_adam": "float32",
    "validation_predictions_targets_and_mse": "float64",
    "automatic_mixed_precision": False,
    "float16": False,
    "bfloat16": False,
    "tensorfloat32": False,
}
_REGULARIZATION = {
    "dropout": False,
    "batch_normalization": False,
    "layer_normalization": False,
    "weight_decay": _WEIGHT_DECAY,
    "gradient_clipping": False,
}
_CONTROLLED_FAULT_STAGE: str | None = None
_CONTROLLED_FAULT_EPOCH = 1
_CONTROLLED_INTERRUPTION_POINT: tuple[int, int | None] | None = None
_CUBLAS_CONFIGURED_BEFORE_CUDA = False


class TrainingContractError(RuntimeError):
    """A deterministic seed run violated the accepted training contract."""

    def __init__(
        self,
        message: str,
        *,
        failed_stage: str | None = None,
        failure_artifact_path: Path | None = None,
    ) -> None:
        super().__init__(message)
        self.failed_stage = failed_stage
        self.failure_artifact_path = failure_artifact_path


class _NonFiniteTrainingState(Exception):
    def __init__(self, stage: str, detail: str) -> None:
        self.stage = stage
        self.detail = detail


@dataclass(frozen=True)
class SeedTrainingResult:
    """Paths and identities retained by one deterministic seed run."""

    seed: int
    evidence_status: str
    test_partition_status: str
    completed_epochs: int
    best_epoch: int
    best_validation_mse: float
    checkpoint_identity: str
    run_configuration_identity: str
    checkpoint_path: Path
    validation_predictions_path: Path
    history_path: Path
    metadata_path: Path
    recovery_snapshot_path: Path
    run_history_path: Path


class _DeepONet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.branch = _tapered_mlp(_BRANCH_WIDTHS)
        self.trunk = _tapered_mlp(_TRUNK_WIDTHS)
        self.fusion_bias = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self._initialize()

    def _initialize(self) -> None:
        for network in (self.branch, self.trunk):
            layers = [module for module in network if isinstance(module, nn.Linear)]
            for index, layer in enumerate(layers):
                gain = 1.0 if index == len(layers) - 1 else 5.0 / 3.0
                nn.init.xavier_normal_(layer.weight, gain=gain)
                nn.init.zeros_(layer.bias)

    def forward(
        self,
        branch_inputs: torch.Tensor,
        trunk_inputs: torch.Tensor,
    ) -> torch.Tensor:
        return (
            self.branch(branch_inputs) @ self.trunk(trunk_inputs).T + self.fusion_bias
        )


def _tapered_mlp(widths: tuple[int, ...]) -> nn.Sequential:
    modules: list[nn.Module] = []
    for index, (input_width, output_width) in enumerate(zip(widths, widths[1:])):
        modules.append(nn.Linear(input_width, output_width))
        if index < len(widths) - 2:
            modules.append(nn.Tanh())
    return nn.Sequential(*modules)


CpuSmokeTrainingResult = SeedTrainingResult


def train_seed(
    prepared: PreparedDataArtifact,
    *,
    seed: int,
    artifact_directory: str | Path,
    smoke_max_epochs: int | None = None,
    recovery_snapshot: str | Path | None = None,
    restart: bool = False,
) -> SeedTrainingResult:
    """Train one canonical GPU seed or an explicitly bounded CPU smoke seed."""

    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise TrainingContractError(
            "training seed must be an integer from 0 to 2^32 - 1"
        )
    canonical_backend = smoke_max_epochs is None
    if canonical_backend and seed not in _CANONICAL_SEEDS:
        raise TrainingContractError(
            "canonical training seed must be one of 0, 1, 2, 3, or 4"
        )
    if not canonical_backend and (
        not isinstance(smoke_max_epochs, int)
        or isinstance(smoke_max_epochs, bool)
        or smoke_max_epochs < 1
    ):
        raise TrainingContractError("smoke_max_epochs must be a positive integer")
    training = prepared.partitions["training"]
    validation = prepared.partitions["validation"]
    preprocessing = prepared.preprocessing
    controlled_fault_stage = _CONTROLLED_FAULT_STAGE
    expected_bindings: tuple[tuple[str, object], ...] = (
        ("source checksums", prepared.source_checksums),
        ("source identity", preprocessing.source_identity),
        ("split identity", preprocessing.split_identity),
        ("preprocessing identity", preprocessing.content_identity),
        ("feature schema identity", preprocessing.feature_schema_identity),
        ("unit schema identity", preprocessing.unit_schema_identity),
    )
    content_identities: dict[str, str] = {}
    for expected_name, partition in (
        ("training", training),
        ("validation", validation),
    ):
        if partition.name != expected_name:
            raise TrainingContractError(
                f"{expected_name} partition name is incompatible"
            )
        actual_bindings = (
            partition.source_checksums,
            partition.source_identity,
            partition.split_identity,
            partition.preprocessing_identity,
            partition.feature_schema_identity,
            partition.unit_schema_identity,
        )
        for (binding_name, expected), actual in zip(
            expected_bindings,
            actual_bindings,
            strict=True,
        ):
            if actual != expected:
                raise TrainingContractError(
                    f"{expected_name} partition {binding_name} is incompatible"
                )
        actual_content_identity = _partition_content_identity(partition)
        if actual_content_identity != partition.content_identity:
            raise TrainingContractError(
                f"{expected_name} partition content identity is incompatible"
            )
        content_identities[f"{expected_name}_partition"] = actual_content_identity
    output = Path(artifact_directory)
    output.mkdir(parents=True, exist_ok=True)

    device_identifier = "cuda:0" if canonical_backend else "cpu"
    canonical = canonical_backend and controlled_fault_stage is None
    if canonical:
        evidence_status = "canonical_seed_run"
    elif canonical_backend:
        evidence_status = "noncanonical_controlled_gpu_failure"
    else:
        evidence_status = "noncanonical_cpu_smoke"
    max_epochs = (
        _CANONICAL_MAX_EPOCHS if smoke_max_epochs is None else smoke_max_epochs
    )
    device = _configure_determinism(seed, device_identifier=device_identifier)
    model = _DeepONet().to(device=device, dtype=torch.float32)
    if sum(parameter.numel() for parameter in model.parameters()) != 9729:
        raise TrainingContractError(
            "the DeepONet must contain exactly 9,729 parameters"
        )
    initial_parameter_identity = _torch_state_identity(model.state_dict())

    training_branch = torch.from_numpy(np.array(training.branch_inputs, copy=True))
    training_targets = torch.from_numpy(np.array(training.raw_aeps_fields, copy=True))
    training_rows = torch.tensor(training.source_rows, dtype=torch.int64)
    validation_branch = torch.from_numpy(
        np.array(validation.branch_inputs, copy=True)
    ).to(device)
    validation_targets = torch.from_numpy(
        np.array(validation.raw_aeps_fields, copy=True)
    ).to(device)
    trunk_inputs = torch.from_numpy(
        np.array(training.trunk_inputs, copy=True)
    ).to(device)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(training_branch, training_targets, training_rows),
        batch_size=_BATCH_SIZE,
        shuffle=_SHUFFLE,
        drop_last=_DROP_LAST,
        num_workers=_NUM_WORKERS,
        generator=loader_generator,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=_LEARNING_RATE,
        betas=_ADAM_BETAS,
        eps=_ADAM_EPSILON,
        weight_decay=_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=_SCHEDULER_MODE,
        factor=_SCHEDULER_FACTOR,
        patience=_SCHEDULER_PATIENCE,
        threshold=_MEANINGFUL_PROGRESS_THRESHOLD,
        threshold_mode=_THRESHOLD_MODE,
        min_lr=_MIN_LEARNING_RATE,
    )

    identities = {
        "source": preprocessing.source_identity,
        "split": preprocessing.split_identity,
        "preprocessing": preprocessing.content_identity,
        "feature_schema": preprocessing.feature_schema_identity,
        "unit_schema": preprocessing.unit_schema_identity,
    }
    run_configuration = {
        "schema_version": "baseline-training-configuration-v1",
        "canonical": canonical,
        "evidence_status": evidence_status,
        "seed": seed,
        "architecture": dict(_EXPECTED_ARCHITECTURE),
        "initialization": _INITIALIZATION,
        "loss": _LOSS,
        "batching": _BATCHING,
        "optimizer": _OPTIMIZER,
        "regularization": _REGULARIZATION,
        "precision": _PRECISION,
        "canonical_configuration": _CANONICAL_CONFIGURATION,
        "identities": identities,
    }
    if smoke_max_epochs is not None:
        run_configuration["smoke_override"] = {
            "backend": "cpu",
            "max_epochs": smoke_max_epochs,
        }
    if controlled_fault_stage is not None:
        run_configuration["controlled_fault_stage"] = controlled_fault_stage
        if _CONTROLLED_FAULT_EPOCH != 1:
            run_configuration["controlled_fault_epoch"] = _CONTROLLED_FAULT_EPOCH
    run_configuration_identity = _canonical_json_identity(run_configuration)
    environment = _environment_metadata(device_identifier)
    software_identity = _canonical_json_identity(
        {
            name: environment[name]
            for name in (
                "python_version",
                "numpy_version",
                "pytorch_build",
                "pytorch_cuda_build",
                "cudnn_version",
            )
        }
    )
    compatibility = {
        "source_identity": preprocessing.source_identity,
        "split_identity": preprocessing.split_identity,
        "preprocessing_identity": preprocessing.content_identity,
        "configuration_identity": run_configuration_identity,
        "precision_identity": _canonical_json_identity(_PRECISION),
        "backend_identity": _canonical_json_identity(environment),
        "software_identity": software_identity,
        "content_identities": content_identities,
    }
    compatibility_identity = _canonical_json_identity(compatibility)
    run_kind = "canonical" if canonical_backend else "cpu_smoke"
    stem = f"seed_{seed}_{run_kind}_{run_configuration_identity[:12]}"
    validation_predictions_path = output / f"{stem}_validation_predictions.npy"
    history_path = output / f"{stem}_history.json"
    metadata_path = output / f"{stem}_metadata.json"
    failure_artifact_path = output / f"{stem}_failure.json"
    recovery_snapshot_path = output / f"{stem}_recovery.pt"
    run_history_path = output / f"{stem}_run_history.jsonl"
    finished_artifact_paths = (
        validation_predictions_path,
        history_path,
        metadata_path,
    )
    all_artifact_paths = (
        *finished_artifact_paths,
        failure_artifact_path,
        recovery_snapshot_path,
        run_history_path,
    )
    requested_recovery = (
        None if recovery_snapshot is None else Path(recovery_snapshot)
    )
    def reject_recovery(message: str) -> None:
        _append_run_event(
            run_history_path,
            "resume_rejected",
            reason=message,
        )
        raise TrainingContractError(message)

    def record_interruption(
        *,
        completed_epoch: int,
        interrupted_epoch: int | None,
        inferred_on_resume: bool = False,
    ) -> None:
        details: dict[str, object] = {
            "completed_epoch": completed_epoch,
            "interrupted_epoch": interrupted_epoch,
            "batch_index": None,
        }
        if inferred_on_resume:
            details["inferred_on_resume"] = True
        _append_run_event(run_history_path, "interruption", **details)
        if interrupted_epoch is not None:
            _append_run_event(
                run_history_path,
                "discarded_partial_epoch",
                completed_epoch=completed_epoch,
                discarded_epoch=interrupted_epoch,
            )

    prior_run_event = _last_run_event(run_history_path)
    prior_run_event_name = (
        None if prior_run_event is None else prior_run_event["event"]
    )
    resuming = requested_recovery is not None and not restart
    if resuming:
        _append_run_event(run_history_path, "resume_attempt")
        if prior_run_event_name == "run_completed":
            reject_recovery("the training run is already complete")
        if failure_artifact_path.exists():
            reject_recovery(
                "a non-finite failed trajectory cannot resume under the unchanged "
                "configuration"
            )
    elif restart:
        if prior_run_event_name == "epoch_started":
            assert prior_run_event is not None
            record_interruption(
                completed_epoch=int(prior_run_event["completed_epoch"]),
                interrupted_epoch=int(prior_run_event["epoch"]),
                inferred_on_resume=True,
            )
        restart_details: dict[str, object] = {}
        if requested_recovery is not None:
            restart_details["abandoned_recovery_snapshot"] = str(
                requested_recovery
            )
        _append_run_event(run_history_path, "restart", **restart_details)
        if prior_run_event_name == "run_completed":
            raise TrainingContractError("the training run is already complete")
        _append_run_event(run_history_path, "run_started")
    else:
        existing_path = next(
            (
                path
                for path in (
                    *output.glob(f"{stem}_checkpoint_*.pt"),
                    *all_artifact_paths,
                )
                if path.exists()
            ),
            None,
        )
        if existing_path is not None:
            raise TrainingContractError(
                f"training artifact path already exists: {existing_path}"
            )
        _append_run_event(run_history_path, "run_started")

    best_validation_mse = math.inf
    best_meaningful_mse = math.inf
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, object] | None = None
    epochs_without_meaningful_progress = 0
    optimizer_steps = 0
    optimizer_updates_attempted = 0
    epoch_history: list[dict[str, object]] = []
    checkpoint_identity: str | None = None
    checkpoint_path: Path | None = None
    completed_epoch = 0
    stopping_reason = (
        "canonical_epoch_ceiling" if canonical_backend else "smoke_epoch_ceiling"
    )

    if resuming:
        assert requested_recovery is not None
        try:
            recovery = torch.load(
                requested_recovery,
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as error:
            reject_recovery(f"recovery snapshot is corrupt: {error}")
        if not isinstance(recovery, Mapping):
            reject_recovery("recovery snapshot is incomplete")
        required_recovery_fields = {
            "schema_version",
            "seed",
            "completed_epoch",
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "epochs_without_meaningful_progress",
            "best_checkpoint_mse",
            "best_meaningful_mse",
            "best_epoch",
            "best_checkpoint_identity",
            "python_random_state",
            "numpy_random_state",
            "torch_cpu_random_state",
            "torch_cuda_random_states",
            "loader_generator_state",
            "epoch_history",
            "optimizer_steps",
            "optimizer_updates_attempted",
            "initial_parameter_identity",
            "run_configuration_identity",
            "compatibility",
            "compatibility_identity",
            "content_identity",
        }
        if not required_recovery_fields.issubset(recovery):
            reject_recovery("recovery snapshot is incomplete")
        recovery_without_identity = dict(recovery)
        recovery_content_identity = recovery_without_identity.pop("content_identity")
        if recovery_content_identity != _torch_state_identity(
            recovery_without_identity
        ):
            reject_recovery("recovery snapshot content identity is invalid")
        if recovery["schema_version"] != "baseline-recovery-v1":
            reject_recovery("recovery snapshot schema is incompatible")
        if recovery["seed"] != seed:
            reject_recovery("recovery snapshot seed is incompatible")
        if (
            recovery["run_configuration_identity"]
            != run_configuration_identity
            or recovery["compatibility"] != compatibility
            or recovery["compatibility_identity"] != compatibility_identity
        ):
            reject_recovery("recovery snapshot compatibility bindings do not match")
        if requested_recovery.resolve() != recovery_snapshot_path.resolve():
            reject_recovery(
                "recovery snapshot does not belong to this artifact directory"
            )
        checkpoint_identity = str(recovery["best_checkpoint_identity"])
        checkpoint_path = _checkpoint_path(output, stem, checkpoint_identity)
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except (OSError, EOFError, RuntimeError, pickle.UnpicklingError) as error:
            reject_recovery(f"best checkpoint is missing or corrupt: {error}")
        if not isinstance(checkpoint, Mapping):
            reject_recovery("best checkpoint does not match the recovery snapshot")
        try:
            (
                actual_checkpoint_identity,
                actual_model_state_identity,
                actual_optimizer_state_identity,
            ) = _checkpoint_identity(
                seed=seed,
                epoch=int(checkpoint["epoch"]),
                validation_mse=float(checkpoint["validation_mse"]),
                model_state=checkpoint["model_state"],
                optimizer_state=checkpoint["optimizer_state"],
                run_configuration_identity=run_configuration_identity,
                compatibility_identity=compatibility_identity,
                identities=identities,
            )
        except (KeyError, TypeError, ValueError) as error:
            reject_recovery(f"best checkpoint is incomplete: {error}")
        if (
            checkpoint.get("checkpoint_identity") != checkpoint_identity
            or checkpoint.get("content_identity") != checkpoint_identity
            or checkpoint.get("compatibility_identity") != compatibility_identity
            or checkpoint.get("model_state_identity")
            != actual_model_state_identity
            or checkpoint.get("optimizer_state_identity")
            != actual_optimizer_state_identity
            or actual_checkpoint_identity != checkpoint_identity
        ):
            reject_recovery("best checkpoint does not match the recovery snapshot")
        try:
            model.load_state_dict(recovery["model_state"])
            optimizer.load_state_dict(recovery["optimizer_state"])
            scheduler.load_state_dict(recovery["scheduler_state"])
            loader_generator.set_state(recovery["loader_generator_state"])
            _restore_random_states(recovery, device_identifier=device_identifier)
        except (KeyError, RuntimeError, TypeError, ValueError) as error:
            reject_recovery(f"recovery snapshot state is incomplete: {error}")
        initial_parameter_identity = recovery["initial_parameter_identity"]
        completed_epoch = int(recovery["completed_epoch"])
        best_validation_mse = float(recovery["best_checkpoint_mse"])
        best_meaningful_mse = float(recovery["best_meaningful_mse"])
        best_epoch = int(recovery["best_epoch"])
        best_model_state = dict(checkpoint["model_state"])
        best_optimizer_state = dict(checkpoint["optimizer_state"])
        epochs_without_meaningful_progress = int(
            recovery["epochs_without_meaningful_progress"]
        )
        optimizer_steps = int(recovery["optimizer_steps"])
        optimizer_updates_attempted = int(
            recovery["optimizer_updates_attempted"]
        )
        epoch_history = list(recovery["epoch_history"])
        if prior_run_event_name == "epoch_started":
            assert prior_run_event is not None
            started_epoch = int(prior_run_event["epoch"])
            record_interruption(
                completed_epoch=completed_epoch,
                interrupted_epoch=(
                    started_epoch if started_epoch > completed_epoch else None
                ),
                inferred_on_resume=True,
            )
        _append_run_event(
            run_history_path,
            "resume_accepted",
            completed_epoch=completed_epoch,
        )

    def fail(
        stage: str,
        *,
        epoch: int,
        batch_index: int | None,
        detail: str,
    ) -> None:
        location = f"epoch {epoch}"
        if batch_index is not None:
            location += f", batch {batch_index}"
        message = f"seed {seed} failed at {stage} during {location}: {detail}"
        failure = {
            "schema_version": "baseline-training-failure-v1",
            "status": "failed",
            "canonical": canonical,
            "evidence_status": evidence_status,
            "seed": seed,
            "failed_stage": stage,
            "epoch": epoch,
            "batch_index": batch_index,
            "optimizer_updates_attempted": optimizer_updates_attempted,
            "optimizer_updates_completed": optimizer_steps,
            "test_partition_status": "locked_not_accessed",
            "message": message,
            "run_configuration": run_configuration,
            "environment": environment,
            "source_checksums": dict(prepared.source_checksums),
            "compatibility": compatibility,
            "compatibility_identity": compatibility_identity,
        }
        failure["content_identity"] = _canonical_json_identity(failure)
        _write_json(failure_artifact_path, failure)
        _append_run_event(
            run_history_path,
            "numerical_failure",
            failed_stage=stage,
            failed_epoch=epoch,
            batch_index=batch_index,
            completed_epoch=completed_epoch,
        )
        raise TrainingContractError(
            message,
            failed_stage=stage,
            failure_artifact_path=failure_artifact_path,
        )

    current_epoch = completed_epoch
    next_epoch = completed_epoch + 1
    if epochs_without_meaningful_progress >= _EARLY_STOPPING_PATIENCE:
        stopping_reason = "canonical_early_stopping"
        next_epoch = max_epochs + 1
    try:
        for epoch in range(next_epoch, max_epochs + 1):
            current_epoch = epoch
            _append_run_event(
                run_history_path,
                "epoch_started",
                epoch=epoch,
                completed_epoch=completed_epoch,
            )
            active_fault_stage = (
                controlled_fault_stage
                if epoch == _CONTROLLED_FAULT_EPOCH
                else None
            )
            model.train()
            epoch_squared_error = 0.0
            epoch_value_count = 0
            batch_sizes: list[int] = []
            batch_source_rows: list[list[int]] = []
            for batch_index, (branch_batch, target_batch, source_row_batch) in enumerate(
                loader,
                start=1,
            ):
                branch_batch = branch_batch.to(device)
                target_batch = target_batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=False):
                    predictions = model(branch_batch, trunk_inputs)
                    loss = torch.nn.functional.mse_loss(predictions, target_batch)
                if active_fault_stage == "training_loss":
                    loss = loss * torch.tensor(float("nan"), device=device)
                if not bool(torch.isfinite(loss)):
                    fail(
                        "training_loss",
                        epoch=epoch,
                        batch_index=batch_index,
                        detail="the scalar loss is non-finite",
                    )
                loss.backward()
                if active_fault_stage == "training_gradient":
                    first_parameter = next(model.parameters())
                    assert first_parameter.grad is not None
                    first_parameter.grad.view(-1)[0] = float("nan")
                for name, parameter in model.named_parameters():
                    if parameter.grad is None or not bool(
                        torch.all(torch.isfinite(parameter.grad))
                    ):
                        fail(
                            "training_gradient",
                            epoch=epoch,
                            batch_index=batch_index,
                            detail=f"the gradient for {name} is missing or non-finite",
                        )
                optimizer_updates_attempted += 1
                optimizer.step()
                if active_fault_stage == "training_parameter":
                    with torch.no_grad():
                        next(model.parameters()).view(-1)[0] = float("nan")
                if any(
                    not bool(torch.all(torch.isfinite(parameter)))
                    for parameter in model.parameters()
                ):
                    fail(
                        "training_parameter",
                        epoch=epoch,
                        batch_index=batch_index,
                        detail="a model parameter is non-finite after the optimizer update",
                    )

                batch_value_count = target_batch.numel()
                epoch_squared_error += float(loss.detach()) * batch_value_count
                epoch_value_count += batch_value_count
                optimizer_steps += 1
                batch_sizes.append(len(target_batch))
                batch_source_rows.append(source_row_batch.tolist())
                if _CONTROLLED_INTERRUPTION_POINT == (epoch, batch_index):
                    raise KeyboardInterrupt

            try:
                validation_mse, _ = _validation_mse(
                    model,
                    validation_branch,
                    trunk_inputs,
                    validation_targets,
                    controlled_fault_stage=active_fault_stage,
                )
            except _NonFiniteTrainingState as failure:
                fail(
                    failure.stage,
                    epoch=epoch,
                    batch_index=None,
                    detail=failure.detail,
                )
            checkpoint_selected = validation_mse < best_validation_mse
            if checkpoint_selected:
                best_validation_mse = validation_mse
                best_epoch = epoch
                best_model_state = {
                    name: tensor.detach().clone()
                    for name, tensor in model.state_dict().items()
                }
                best_optimizer_state = deepcopy(optimizer.state_dict())

            meaningful_progress = validation_mse < best_meaningful_mse * (
                1.0 - _MEANINGFUL_PROGRESS_THRESHOLD
            )
            if meaningful_progress:
                best_meaningful_mse = validation_mse
                epochs_without_meaningful_progress = 0
            else:
                epochs_without_meaningful_progress += 1

            scheduler.step(validation_mse)
            scheduler_state = scheduler.state_dict()
            epoch_history.append(
                {
                    "epoch": epoch,
                    "training_mse": epoch_squared_error / epoch_value_count,
                    "validation_mse": validation_mse,
                    "batch_sizes": batch_sizes,
                    "batch_source_rows": batch_source_rows,
                    "checkpoint": {
                        "selected_this_epoch": checkpoint_selected,
                        "best_epoch": best_epoch,
                        "best_validation_mse": best_validation_mse,
                    },
                    "early_stopping": {
                        "meaningful_progress": meaningful_progress,
                        "best_meaningful_validation_mse": best_meaningful_mse,
                        "epochs_without_meaningful_progress": (
                            epochs_without_meaningful_progress
                        ),
                    },
                    "scheduler": {
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "best_validation_mse": float(scheduler_state["best"]),
                        "num_bad_epochs": int(scheduler_state["num_bad_epochs"]),
                        "cooldown_counter": int(scheduler_state["cooldown_counter"]),
                        "last_epoch": int(scheduler_state["last_epoch"]),
                    },
                }
            )
            should_stop = (
                epochs_without_meaningful_progress >= _EARLY_STOPPING_PATIENCE
            )
            if should_stop:
                stopping_reason = "canonical_early_stopping"
            obsolete_checkpoint_path: Path | None = None
            if checkpoint_selected:
                assert best_model_state is not None
                assert best_optimizer_state is not None
                obsolete_checkpoint_path = checkpoint_path
                (
                    checkpoint_identity,
                    selected_parameter_identity,
                    selected_optimizer_identity,
                ) = _checkpoint_identity(
                    seed=seed,
                    epoch=best_epoch,
                    validation_mse=best_validation_mse,
                    model_state=best_model_state,
                    optimizer_state=best_optimizer_state,
                    run_configuration_identity=run_configuration_identity,
                    compatibility_identity=compatibility_identity,
                    identities=identities,
                )
                checkpoint_path = _checkpoint_path(output, stem, checkpoint_identity)
                _atomic_torch_save(
                    checkpoint_path,
                    {
                        "schema_version": "baseline-checkpoint-v1",
                        "canonical": canonical,
                        "evidence_status": evidence_status,
                        "seed": seed,
                        "epoch": best_epoch,
                        "validation_mse": best_validation_mse,
                        "checkpoint_identity": checkpoint_identity,
                        "run_configuration_identity": run_configuration_identity,
                        "compatibility": compatibility,
                        "compatibility_identity": compatibility_identity,
                        "content_identity": checkpoint_identity,
                        "source_checksums": dict(prepared.source_checksums),
                        "source_identity": preprocessing.source_identity,
                        "split_identity": preprocessing.split_identity,
                        "preprocessing_identity": preprocessing.content_identity,
                        "feature_schema_identity": preprocessing.feature_schema_identity,
                        "unit_schema_identity": preprocessing.unit_schema_identity,
                        "architecture": dict(_EXPECTED_ARCHITECTURE),
                        "model_state_identity": selected_parameter_identity,
                        "optimizer_state_identity": selected_optimizer_identity,
                        "model_state": best_model_state,
                        "optimizer_state": best_optimizer_state,
                    },
                )
                if _CONTROLLED_INTERRUPTION_POINT == (epoch, 0):
                    raise KeyboardInterrupt
            assert checkpoint_identity is not None
            numpy_random_state: Any = np.random.get_state()
            recovery_payload: dict[str, object] = {
                "schema_version": "baseline-recovery-v1",
                "seed": seed,
                "completed_epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "epochs_without_meaningful_progress": epochs_without_meaningful_progress,
                "best_checkpoint_mse": best_validation_mse,
                "best_meaningful_mse": best_meaningful_mse,
                "best_epoch": best_epoch,
                "best_checkpoint_identity": checkpoint_identity,
                "python_random_state": random.getstate(),
                "numpy_random_state": {
                    "bit_generator": numpy_random_state[0],
                    "state": numpy_random_state[1].tolist(),
                    "position": numpy_random_state[2],
                    "has_gauss": numpy_random_state[3],
                    "cached_gaussian": numpy_random_state[4],
                },
                "torch_cpu_random_state": torch.get_rng_state(),
                "torch_cuda_random_states": (
                    torch.cuda.get_rng_state_all()
                    if device_identifier == "cuda:0"
                    else []
                ),
                "loader_generator_state": loader_generator.get_state(),
                "epoch_history": epoch_history,
                "optimizer_steps": optimizer_steps,
                "optimizer_updates_attempted": optimizer_updates_attempted,
                "initial_parameter_identity": initial_parameter_identity,
                "run_configuration_identity": run_configuration_identity,
                "compatibility": compatibility,
                "compatibility_identity": compatibility_identity,
            }
            recovery_payload["content_identity"] = _torch_state_identity(
                recovery_payload
            )
            _atomic_torch_save(recovery_snapshot_path, recovery_payload)
            completed_epoch = epoch
            if (
                obsolete_checkpoint_path is not None
                and obsolete_checkpoint_path != checkpoint_path
            ):
                obsolete_checkpoint_path.unlink(missing_ok=True)
            if _CONTROLLED_INTERRUPTION_POINT == (epoch, None):
                raise KeyboardInterrupt
            if should_stop:
                break

    except KeyboardInterrupt:
        interrupted_epoch = (
            current_epoch if current_epoch > completed_epoch else None
        )
        record_interruption(
            completed_epoch=completed_epoch,
            interrupted_epoch=interrupted_epoch,
        )
        raise
    try:
        if best_model_state is None or best_optimizer_state is None:
            raise TrainingContractError("training completed without a finite checkpoint")
        final_parameter_identity = _torch_state_identity(model.state_dict())
        model.load_state_dict(best_model_state)
        try:
            restored_validation_mse, validation_predictions = _validation_mse(
                model,
                validation_branch,
                trunk_inputs,
                validation_targets,
            )
        except _NonFiniteTrainingState as failure:
            fail(
                failure.stage,
                epoch=best_epoch,
                batch_index=None,
                detail=failure.detail,
            )
        if restored_validation_mse != best_validation_mse:
            raise TrainingContractError(
                "restored best checkpoint does not reproduce its validation MSE"
            )

        selected_parameter_identity = _torch_state_identity(best_model_state)
        selected_optimizer_identity = _torch_state_identity(best_optimizer_state)
        assert checkpoint_identity is not None
        assert checkpoint_path is not None
        np.save(validation_predictions_path, validation_predictions)

        history = {
            "schema_version": "baseline-training-history-v1",
            "seed": seed,
            "evidence_status": evidence_status,
            "run_configuration_identity": run_configuration_identity,
            "initial_parameter_identity": initial_parameter_identity,
            "final_parameter_identity": final_parameter_identity,
            "selected_parameter_identity": selected_parameter_identity,
            "optimizer_steps": optimizer_steps,
            "completed_epochs": len(epoch_history),
            "stopping_reason": stopping_reason,
            "epochs": epoch_history,
            "compatibility": compatibility,
            "compatibility_identity": compatibility_identity,
        }
        history["content_identity"] = _canonical_json_identity(history)
        _write_json(history_path, history)

        metadata = {
            **run_configuration,
            "schema_version": "baseline-training-run-v1",
            "environment": environment,
            "configuration_identity": run_configuration_identity,
            "compatibility": compatibility,
            "compatibility_identity": compatibility_identity,
            "source_checksums": dict(prepared.source_checksums),
            "identities": {
                **identities,
                "run_configuration": run_configuration_identity,
            },
            "selected_checkpoint": {
                "identity": checkpoint_identity,
                "epoch": best_epoch,
                "validation_mse": best_validation_mse,
            },
            "validation_predictions": {
                "shape": list(validation_predictions.shape),
                "dtype": str(validation_predictions.dtype),
                "content_identity": sha256(validation_predictions.tobytes()).hexdigest(),
            },
            "test_partition_status": "locked_not_accessed",
        }
        metadata["content_identity"] = _canonical_json_identity(metadata)
        _write_json(metadata_path, metadata)
        _append_run_event(
            run_history_path,
            "run_completed",
            completed_epoch=len(epoch_history),
            stopping_reason=stopping_reason,
        )

        return SeedTrainingResult(
            seed=seed,
            evidence_status=evidence_status,
            test_partition_status="locked_not_accessed",
            completed_epochs=len(epoch_history),
            best_epoch=best_epoch,
            best_validation_mse=best_validation_mse,
            checkpoint_identity=checkpoint_identity,
            run_configuration_identity=run_configuration_identity,
            checkpoint_path=checkpoint_path,
            validation_predictions_path=validation_predictions_path,
            history_path=history_path,
            metadata_path=metadata_path,
            recovery_snapshot_path=recovery_snapshot_path,
            run_history_path=run_history_path,
        )

    except KeyboardInterrupt:
        record_interruption(
            completed_epoch=completed_epoch,
            interrupted_epoch=None,
        )
        raise


def _atomic_torch_save(path: Path, payload: object) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary_path)
        for attempt in range(20):
            try:
                os.replace(temporary_path, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.01)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _checkpoint_path(output: Path, stem: str, checkpoint_identity: str) -> Path:
    return output / f"{stem}_checkpoint_{checkpoint_identity}.pt"


def _checkpoint_identity(
    *,
    seed: int,
    epoch: int,
    validation_mse: float,
    model_state: object,
    optimizer_state: object,
    run_configuration_identity: str,
    compatibility_identity: str,
    identities: Mapping[str, object],
) -> tuple[str, str, str]:
    model_state_identity = _torch_state_identity(model_state)
    optimizer_state_identity = _torch_state_identity(optimizer_state)
    checkpoint_identity = _canonical_json_identity(
        {
            "seed": seed,
            "epoch": epoch,
            "validation_mse": validation_mse,
            "model_state_identity": model_state_identity,
            "optimizer_state_identity": optimizer_state_identity,
            "run_configuration_identity": run_configuration_identity,
            "compatibility_identity": compatibility_identity,
            **identities,
        }
    )
    return checkpoint_identity, model_state_identity, optimizer_state_identity


def _append_run_event(path: Path, event: str, **details: object) -> None:
    sequence = 1
    if path.exists():
        with path.open("r", encoding="utf-8") as stream:
            sequence += sum(1 for line in stream if line.strip())
    payload = {"sequence": sequence, "event": event, **details}
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )


def _last_run_event(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return None if not lines else dict(json.loads(lines[-1]))


def _restore_random_states(
    recovery: Mapping[str, Any],
    *,
    device_identifier: str,
) -> None:
    numpy_state = recovery["numpy_random_state"]
    if not isinstance(numpy_state, Mapping):
        raise TypeError("NumPy random state is not a mapping")
    random.setstate(recovery["python_random_state"])
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(recovery["torch_cpu_random_state"])
    cuda_states = recovery["torch_cuda_random_states"]
    if device_identifier == "cuda:0":
        torch.cuda.set_rng_state_all(cuda_states)
    elif cuda_states:
        raise ValueError("CPU recovery snapshot contains CUDA random state")


def _configure_determinism(seed: int, *, device_identifier: str) -> torch.device:
    global _CUBLAS_CONFIGURED_BEFORE_CUDA

    cuda_initialized = torch.cuda.is_initialized()
    if (
        device_identifier == "cuda:0"
        and cuda_initialized
        and (
            not _CUBLAS_CONFIGURED_BEFORE_CUDA
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG")
            != _CUBLAS_WORKSPACE_CONFIGURATION
        )
    ):
        raise TrainingContractError(
            "canonical CUDA execution requires CUBLAS_WORKSPACE_CONFIG before "
            "CUDA initialization"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = _CUBLAS_WORKSPACE_CONFIGURATION
    if not cuda_initialized:
        _CUBLAS_CONFIGURED_BEFORE_CUDA = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if device_identifier == "cuda:0":
        if not torch.cuda.is_available():
            raise TrainingContractError(
                "canonical training requires CUDA device cuda:0"
            )
        torch.cuda.manual_seed_all(seed)
    return torch.device(device_identifier)


def _validation_mse(
    model: _DeepONet,
    branch_inputs: torch.Tensor,
    trunk_inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    controlled_fault_stage: str | None = None,
) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad(), torch.autocast(
        device_type=branch_inputs.device.type,
        enabled=False,
    ):
        predictions = model(branch_inputs, trunk_inputs).to(dtype=torch.float64)
        targets_float64 = targets.to(dtype=torch.float64)
        if controlled_fault_stage == "validation_prediction":
            predictions.view(-1)[0] = float("nan")
        if not bool(torch.all(torch.isfinite(predictions))):
            raise _NonFiniteTrainingState(
                "validation_prediction",
                "at least one validation prediction is non-finite",
            )
        mse = float(torch.mean((predictions - targets_float64) ** 2))
    if controlled_fault_stage == "validation_mse":
        mse = math.inf
    if not math.isfinite(mse):
        raise _NonFiniteTrainingState(
            "validation_mse",
            "the validation MSE is non-finite",
        )
    return mse, predictions.cpu().numpy()


def _torch_state_identity(state: object) -> str:
    digest = sha256()

    def update(value: object) -> None:
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
            return
        if isinstance(value, dict):
            digest.update(b"mapping\0")
            for key in sorted(
                value,
                key=lambda item: (type(item).__name__, repr(item)),
            ):
                update(key)
                update(value[key])
            return
        if isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode("ascii") + b"\0")
            for item in value:
                update(item)
            return
        if value is None or isinstance(value, (bool, int, float, str)):
            digest.update(type(value).__name__.encode("ascii") + b"\0")
            if isinstance(value, float) and not math.isfinite(value):
                digest.update(repr(value).encode("ascii"))
            else:
                digest.update(
                    json.dumps(
                        value,
                        allow_nan=False,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
            return
        raise TrainingContractError(
            f"unsupported checkpoint state value {type(value).__name__}"
        )

    update(state)
    return digest.hexdigest()


def _canonical_json_identity(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _environment_metadata(device_identifier: str) -> dict[str, object]:
    if device_identifier == "cuda:0":
        device_name = torch.cuda.get_device_name(0)
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader",
                    "--id=0",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise TrainingContractError(
                "canonical training could not record the NVIDIA driver version"
            ) from error
        driver_version: str | None = completed.stdout.strip()
    else:
        device_name = platform.processor() or platform.machine()
        driver_version = None
    return {
        "device_identifier": device_identifier,
        "device_name": device_name,
        "driver_version": driver_version,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pytorch_version": str(torch.__version__),
        "pytorch_build": str(torch.__version__),
        "pytorch_cuda_build": torch.version.cuda,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_tensorfloat32": torch.backends.cudnn.allow_tf32,
        "matmul_tensorfloat32": torch.backends.cuda.matmul.allow_tf32,
        "automatic_mixed_precision": False,
        "cublas_workspace_configuration": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
