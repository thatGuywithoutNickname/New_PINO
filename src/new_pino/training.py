"""Noncanonical CPU smoke training through the public baseline boundary."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import platform
import random
from typing import Literal, Mapping

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .preparation import PreparedDataArtifact


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
_CANONICAL_CONFIGURATION = {
    "backend": "cuda:0",
    "seeds": [0, 1, 2, 3, 4],
    "max_epochs": _CANONICAL_MAX_EPOCHS,
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


class TrainingContractError(RuntimeError):
    """A CPU smoke run violated the accepted training contract."""


@dataclass(frozen=True)
class CpuSmokeTrainingResult:
    """Paths and identities retained by one noncanonical CPU smoke run."""

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


def train_cpu_smoke(
    prepared: PreparedDataArtifact,
    *,
    seed: int,
    artifact_directory: str | Path,
    smoke_max_epochs: int,
) -> CpuSmokeTrainingResult:
    """Train and retain one explicitly noncanonical DeepONet CPU smoke seed."""

    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise TrainingContractError(
            "training seed must be an integer from 0 to 2^32 - 1"
        )
    if (
        not isinstance(smoke_max_epochs, int)
        or isinstance(smoke_max_epochs, bool)
        or smoke_max_epochs < 1
    ):
        raise TrainingContractError("smoke_max_epochs must be a positive integer")

    training = prepared.partitions["training"]
    validation = prepared.partitions["validation"]
    preprocessing = prepared.preprocessing
    output = Path(artifact_directory)
    output.mkdir(parents=True, exist_ok=True)

    _configure_cpu_determinism(seed)
    model = _DeepONet().to(device="cpu", dtype=torch.float32)
    if sum(parameter.numel() for parameter in model.parameters()) != 9729:
        raise TrainingContractError(
            "the DeepONet must contain exactly 9,729 parameters"
        )
    initial_parameter_identity = _model_state_identity(model.state_dict())

    training_branch = torch.from_numpy(np.array(training.branch_inputs, copy=True))
    training_targets = torch.from_numpy(np.array(training.raw_aeps_fields, copy=True))
    training_rows = torch.tensor(training.source_rows, dtype=torch.int64)
    validation_branch = torch.from_numpy(np.array(validation.branch_inputs, copy=True))
    validation_targets = torch.from_numpy(
        np.array(validation.raw_aeps_fields, copy=True)
    )
    trunk_inputs = torch.from_numpy(np.array(training.trunk_inputs, copy=True))
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
        "canonical": False,
        "evidence_status": "noncanonical_cpu_smoke",
        "seed": seed,
        "architecture": dict(_EXPECTED_ARCHITECTURE),
        "initialization": _INITIALIZATION,
        "loss": _LOSS,
        "batching": _BATCHING,
        "optimizer": _OPTIMIZER,
        "regularization": _REGULARIZATION,
        "precision": _PRECISION,
        "canonical_configuration": _CANONICAL_CONFIGURATION,
        "smoke_override": {"backend": "cpu", "max_epochs": smoke_max_epochs},
        "identities": identities,
    }
    run_configuration_identity = _canonical_json_identity(run_configuration)

    best_validation_mse = math.inf
    best_meaningful_mse = math.inf
    best_epoch = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, object] | None = None
    epochs_without_meaningful_progress = 0
    optimizer_steps = 0
    epoch_history: list[dict[str, object]] = []
    stopping_reason = "smoke_epoch_ceiling"

    for epoch in range(1, smoke_max_epochs + 1):
        model.train()
        epoch_squared_error = 0.0
        epoch_value_count = 0
        batch_sizes: list[int] = []
        batch_source_rows: list[list[int]] = []
        for branch_batch, target_batch, source_row_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            predictions = model(branch_batch, trunk_inputs)
            loss = torch.nn.functional.mse_loss(predictions, target_batch)
            if not bool(torch.isfinite(loss)):
                raise TrainingContractError(
                    f"seed {seed} produced a non-finite training loss at epoch {epoch}"
                )
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None or not bool(
                    torch.all(torch.isfinite(parameter.grad))
                ):
                    raise TrainingContractError(
                        f"seed {seed} produced an invalid gradient for {name} "
                        f"at epoch {epoch}"
                    )
            optimizer.step()
            if any(
                not bool(torch.all(torch.isfinite(parameter)))
                for parameter in model.parameters()
            ):
                raise TrainingContractError(
                    f"seed {seed} produced a non-finite parameter at epoch {epoch}"
                )

            batch_value_count = target_batch.numel()
            epoch_squared_error += float(loss.detach()) * batch_value_count
            epoch_value_count += batch_value_count
            optimizer_steps += 1
            batch_sizes.append(len(target_batch))
            batch_source_rows.append(source_row_batch.tolist())

        validation_mse, _ = _validation_mse(
            model,
            validation_branch,
            trunk_inputs,
            validation_targets,
            seed=seed,
            epoch=epoch,
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
        if epochs_without_meaningful_progress >= _EARLY_STOPPING_PATIENCE:
            stopping_reason = "canonical_early_stopping"
            break

    if best_model_state is None or best_optimizer_state is None:
        raise TrainingContractError("training completed without a finite checkpoint")
    final_parameter_identity = _model_state_identity(model.state_dict())
    model.load_state_dict(best_model_state)
    restored_validation_mse, validation_predictions = _validation_mse(
        model,
        validation_branch,
        trunk_inputs,
        validation_targets,
        seed=seed,
        epoch=best_epoch,
    )
    if restored_validation_mse != best_validation_mse:
        raise TrainingContractError(
            "restored best checkpoint does not reproduce its validation MSE"
        )

    selected_parameter_identity = _model_state_identity(best_model_state)
    checkpoint_identity = _canonical_json_identity(
        {
            "seed": seed,
            "epoch": best_epoch,
            "validation_mse": best_validation_mse,
            "model_state_identity": selected_parameter_identity,
            "run_configuration_identity": run_configuration_identity,
            **identities,
        }
    )
    stem = f"seed_{seed}_cpu_smoke"
    checkpoint_path = output / f"{stem}_checkpoint.pt"
    validation_predictions_path = output / f"{stem}_validation_predictions.npy"
    history_path = output / f"{stem}_history.json"
    metadata_path = output / f"{stem}_metadata.json"

    torch.save(
        {
            "schema_version": "baseline-checkpoint-v1",
            "canonical": False,
            "evidence_status": "noncanonical_cpu_smoke",
            "seed": seed,
            "epoch": best_epoch,
            "validation_mse": best_validation_mse,
            "checkpoint_identity": checkpoint_identity,
            "run_configuration_identity": run_configuration_identity,
            "source_checksums": dict(prepared.source_checksums),
            "source_identity": preprocessing.source_identity,
            "split_identity": preprocessing.split_identity,
            "preprocessing_identity": preprocessing.content_identity,
            "feature_schema_identity": preprocessing.feature_schema_identity,
            "unit_schema_identity": preprocessing.unit_schema_identity,
            "architecture": dict(_EXPECTED_ARCHITECTURE),
            "model_state": best_model_state,
            "optimizer_state": best_optimizer_state,
        },
        checkpoint_path,
    )
    np.save(validation_predictions_path, validation_predictions)

    history = {
        "schema_version": "baseline-training-history-v1",
        "seed": seed,
        "evidence_status": "noncanonical_cpu_smoke",
        "run_configuration_identity": run_configuration_identity,
        "initial_parameter_identity": initial_parameter_identity,
        "final_parameter_identity": final_parameter_identity,
        "selected_parameter_identity": selected_parameter_identity,
        "optimizer_steps": optimizer_steps,
        "completed_epochs": len(epoch_history),
        "stopping_reason": stopping_reason,
        "epochs": epoch_history,
    }
    _write_json(history_path, history)

    metadata = {
        **run_configuration,
        "schema_version": "baseline-training-run-v1",
        "environment": _environment_metadata(),
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
    _write_json(metadata_path, metadata)

    return CpuSmokeTrainingResult(
        seed=seed,
        evidence_status="noncanonical_cpu_smoke",
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
    )


def _configure_cpu_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def _validation_mse(
    model: _DeepONet,
    branch_inputs: torch.Tensor,
    trunk_inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    seed: int,
    epoch: int,
) -> tuple[float, np.ndarray]:
    model.eval()
    with torch.no_grad():
        predictions = model(branch_inputs, trunk_inputs).to(dtype=torch.float64)
        targets_float64 = targets.to(dtype=torch.float64)
    if not bool(torch.all(torch.isfinite(predictions))):
        raise TrainingContractError(
            f"seed {seed} produced non-finite validation predictions at epoch {epoch}"
        )
    mse = float(torch.mean((predictions - targets_float64) ** 2))
    if not math.isfinite(mse):
        raise TrainingContractError(
            f"seed {seed} produced a non-finite validation MSE at epoch {epoch}"
        )
    return mse, predictions.numpy()


def _model_state_identity(state: Mapping[str, torch.Tensor]) -> str:
    digest = sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
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


def _environment_metadata() -> dict[str, object]:
    return {
        "device_identifier": "cpu",
        "device_name": platform.processor() or platform.machine(),
        "driver_version": None,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pytorch_version": str(torch.__version__),
        "pytorch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_tensorfloat32": torch.backends.cudnn.allow_tf32,
        "matmul_tensorfloat32": torch.backends.cuda.matmul.allow_tf32,
        "cublas_workspace_configuration": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
