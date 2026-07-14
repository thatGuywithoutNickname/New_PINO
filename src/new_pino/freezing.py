"""Validation-only freezing of five explicit seed predictors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from .preparation import (
    PreparedDataArtifact,
    _partition_content_identity,
)
from .reporting import (
    SeedMetricReport,
    _case_metrics,
    _cross_seed_summary,
    _median,
    _nearest_rank_p90,
)
from .training import SeedTrainingResult, _CANONICAL_SEEDS, _EXPECTED_ARCHITECTURE


_RERUN_REQUIREMENT = (
    "rerun_all_five_seeds_under_one_revised_common_protocol"
)
_PROTOCOL_FIELDS = (
    "canonical",
    "evidence_status",
    "architecture",
    "initialization",
    "loss",
    "batching",
    "optimizer",
    "regularization",
    "precision",
    "canonical_configuration",
    "smoke_override",
    "environment",
)


class FreezeContractError(ValueError):
    """Five seed artifacts cannot form one compatible validation gate."""

    def __init__(self, message: str, *, gate_artifact_path: Path) -> None:
        super().__init__(message)
        self.gate_artifact_path = gate_artifact_path


@dataclass(frozen=True)
class SeedComparatorEvidence:
    seed: int
    checkpoint_identity: str
    validation_global_rmse: float
    comparator_status: str


@dataclass(frozen=True)
class FreezeResult:
    gate_passed: bool
    canonical: bool
    evidence_status: str
    comparator_global_rmse: float
    mean_seed_global_rmse: float
    passing_seed_count: int
    seed_evidence: tuple[SeedComparatorEvidence, ...]
    failure_reasons: tuple[str, ...]
    revision_requirement: str | None
    test_partition_status: str
    protocol_identity: str
    gate_artifact_path: Path
    package_path: Path | None


@dataclass(frozen=True)
class _ValidatedSeedRun:
    result: SeedTrainingResult
    metadata: Mapping[str, Any]
    history: Mapping[str, Any]
    predictions: np.ndarray
    compatibility: Mapping[str, Any]
    protocol_identity: str


def freeze_seed_runs(
    prepared: PreparedDataArtifact,
    seed_runs: Sequence[SeedTrainingResult],
    *,
    repository_root: str | Path,
    package_directory: str | Path,
) -> FreezeResult:
    """Apply the validation comparator and package five predictors only on pass."""

    output = Path(package_directory)
    if output.exists() and any(output.iterdir()):
        raise FreezeContractError(
            f"freeze package directory is not empty: {output}",
            gate_artifact_path=output / "freeze_gate.json",
        )
    output.mkdir(parents=True, exist_ok=True)
    gate_path = output / "freeze_gate.json"

    def reject(reason: str, message: str) -> None:
        payload: dict[str, object] = {
            "schema_version": "baseline-freeze-gate-v1",
            "gate_status": "invalid",
            "reason": reason,
            "message": message,
            "test_partition_status": "locked_invalid_evidence",
            "revision_requirement": _RERUN_REQUIREMENT,
        }
        payload["content_identity"] = _identity(payload)
        _write_json(gate_path, payload)
        raise FreezeContractError(message, gate_artifact_path=gate_path)

    seeds = [run.seed for run in seed_runs]
    if len(seed_runs) != 5 or sorted(seeds) != list(_CANONICAL_SEEDS):
        reject(
            "incomplete_seed_set",
            "validation freezing requires exactly seeds 0 through 4 once each",
        )

    training = prepared.partitions["training"]
    validation = prepared.partitions["validation"]
    if training.name != "training" or validation.name != "validation":
        reject(
            "incompatible_prepared_partitions",
            "prepared training and validation partition names are incompatible",
        )
    if (
        _partition_content_identity(training) != training.content_identity
        or _partition_content_identity(validation) != validation.content_identity
    ):
        reject(
            "incompatible_prepared_partitions",
            "prepared training or validation partition content identity is stale",
        )
    if training.raw_aeps_fields.shape != (246, 48) or validation.raw_aeps_fields.shape != (
        51,
        48,
    ):
        reject(
            "incompatible_prepared_partitions",
            "the validation freeze gate requires the accepted 246/51 case partitions",
        )

    validated_runs: list[_ValidatedSeedRun] = []
    for run in sorted(seed_runs, key=lambda item: item.seed):
        try:
            validated_runs.append(_validate_seed_run(prepared, run))
        except ValueError as error:
            reject("invalid_seed_evidence", str(error))

    protocol_identities = {run.protocol_identity for run in validated_runs}
    if len(protocol_identities) != 1:
        reject(
            "incompatible_seed_identities",
            "all five seeds must share one common training protocol and environment",
        )
    protocol_identity = next(iter(protocol_identities))
    root = Path(repository_root).resolve()
    try:
        _verify_repository_sources(root, prepared)
    except ValueError as error:
        reject("incompatible_canonical_sources", str(error))

    training_mean = np.mean(training.raw_aeps_fields, axis=0, dtype=np.float64)
    validation_truth = validation.raw_aeps_fields.astype(np.float64)
    comparator_mse = float(np.mean((validation_truth - training_mean) ** 2))
    comparator_rmse = math.sqrt(comparator_mse)
    seed_evidence: list[SeedComparatorEvidence] = []
    for validated in validated_runs:
        mse = float(np.mean((validated.predictions - validation_truth) ** 2))
        rmse = math.sqrt(mse)
        seed_evidence.append(
            SeedComparatorEvidence(
                seed=validated.result.seed,
                checkpoint_identity=validated.result.checkpoint_identity,
                validation_global_rmse=rmse,
                comparator_status=("passed" if rmse < comparator_rmse else "not_passed"),
            )
        )
    passing_count = sum(
        evidence.comparator_status == "passed" for evidence in seed_evidence
    )
    mean_seed_rmse = sum(
        evidence.validation_global_rmse for evidence in seed_evidence
    ) / 5
    failure_reasons: list[str] = []
    if passing_count < 4:
        failure_reasons.append("insufficient_seed_majority")
    if mean_seed_rmse >= comparator_rmse:
        failure_reasons.append("five_seed_mean_not_better")
    gate_passed = not failure_reasons

    canonical = all(
        run.metadata.get("canonical") is True
        and run.result.evidence_status == "canonical_seed_run"
        for run in validated_runs
    )
    evidence_status = (
        "canonical_frozen_package" if canonical else "noncanonical_frozen_package"
    )
    test_partition_status = (
        "eligible_locked_test"
        if gate_passed and canonical
        else (
            "locked_ineligible_noncanonical"
            if gate_passed
            else "locked_gate_failed"
        )
    )
    gate_payload: dict[str, object] = {
        "schema_version": "baseline-freeze-gate-v1",
        "gate_status": "passed" if gate_passed else "failed",
        "gate_passed": gate_passed,
        "canonical": canonical,
        "evidence_status": evidence_status,
        "comparator": {
            "kind": "element_wise_training_mean_aeps_field",
            "training_mean_aeps_field": training_mean.tolist(),
            "applied_unchanged_to_validation_case_count": 51,
            "global_validation_rmse": comparator_rmse,
        },
        "seed_evidence": [asdict(evidence) for evidence in seed_evidence],
        "passing_seed_count": passing_count,
        "required_passing_seed_count": 4,
        "mean_seed_global_rmse": mean_seed_rmse,
        "strict_comparison": True,
        "failure_reasons": failure_reasons,
        "test_partition_status": test_partition_status,
        "protocol_identity": protocol_identity,
        "revision_requirement": None if gate_passed else _RERUN_REQUIREMENT,
    }
    gate_payload["content_identity"] = _identity(gate_payload)
    _write_json(gate_path, gate_payload)

    result = FreezeResult(
        gate_passed=gate_passed,
        canonical=canonical,
        evidence_status=evidence_status,
        comparator_global_rmse=comparator_rmse,
        mean_seed_global_rmse=mean_seed_rmse,
        passing_seed_count=passing_count,
        seed_evidence=tuple(seed_evidence),
        failure_reasons=tuple(failure_reasons),
        revision_requirement=None if gate_passed else _RERUN_REQUIREMENT,
        test_partition_status=test_partition_status,
        protocol_identity=protocol_identity,
        gate_artifact_path=gate_path,
        package_path=output if gate_passed else None,
    )
    if not gate_passed:
        return result

    validation_report = _validation_report(
        prepared,
        validated_runs,
        seed_evidence,
        canonical=canonical,
    )
    predictors = _copy_seed_artifacts(
        output,
        prepared,
        validated_runs,
        seed_evidence,
    )
    runtime_sources = output / "runtime_sources"
    runtime_sources.mkdir()
    shutil.copy2(root / "data" / "co_ind.csv", runtime_sources / "co_ind.csv")
    shutil.copy2(
        root / "data" / "material_properties.md",
        runtime_sources / "material_properties.md",
    )
    _write_json(output / "validation_report.json", validation_report)

    preprocessing = prepared.preprocessing.to_dict()
    preprocessing.pop("source_checksums")
    preprocessing.pop("source_identity")
    preprocessing.pop("split_identity")
    package: dict[str, object] = {
        "schema_version": "baseline-frozen-package-v1",
        "canonical": canonical,
        "evidence_status": evidence_status,
        "gate_status": "passed",
        "test_partition_status": test_partition_status,
        "source_checksums": dict(prepared.source_checksums),
        "source_identity": prepared.preprocessing.source_identity,
        "split_identity": prepared.preprocessing.split_identity,
        "preprocessing": preprocessing,
        "architecture": dict(_EXPECTED_ARCHITECTURE),
        "training_protocol": {
            "identity": protocol_identity,
            "configuration": {
                name: validated_runs[0].metadata[name]
                for name in _PROTOCOL_FIELDS
                if name in validated_runs[0].metadata
            },
        },
        "source_preflight": prepared.to_dict(),
        "gate_evidence": gate_payload,
        "validation_report": validation_report,
        "frozen_contract": {
            "source_identity": prepared.preprocessing.source_identity,
            "split_identity": prepared.preprocessing.split_identity,
            "preprocessing_identity": prepared.preprocessing.content_identity,
            "feature_schema_identity": (
                prepared.preprocessing.feature_schema_identity
            ),
            "unit_schema_identity": prepared.preprocessing.unit_schema_identity,
            "architecture": dict(_EXPECTED_ARCHITECTURE),
            "training_protocol_identity": protocol_identity,
            "selected_checkpoint_identities": [
                run.result.checkpoint_identity for run in validated_runs
            ],
        },
        "runtime_sources": {
            "element_points": "runtime_sources/co_ind.csv",
            "material_properties": "runtime_sources/material_properties.md",
        },
        "predictors": predictors,
    }
    package["package_content_identity"] = _identity(package)
    _write_json(output / "package.json", package)
    return result


def _validate_seed_run(
    prepared: PreparedDataArtifact,
    run: SeedTrainingResult,
) -> _ValidatedSeedRun:
    artifact_paths = (
        run.checkpoint_path,
        run.validation_predictions_path,
        run.history_path,
        run.metadata_path,
        run.recovery_snapshot_path,
        run.run_history_path,
    )
    missing = next((path for path in artifact_paths if not path.is_file()), None)
    if missing is not None:
        raise ValueError(f"seed {run.seed} artifact is missing: {missing}")
    if run.test_partition_status != "locked_not_accessed":
        raise ValueError(f"seed {run.seed} accessed the locked test partition")

    metadata = _load_json(run.metadata_path, f"seed {run.seed} metadata")
    history = _load_json(run.history_path, f"seed {run.seed} history")
    _verify_json_identity(metadata, f"seed {run.seed} metadata")
    _verify_json_identity(history, f"seed {run.seed} history")
    if metadata.get("schema_version") != "baseline-training-run-v1":
        raise ValueError(f"seed {run.seed} metadata schema is incompatible")
    required_protocol_fields = set(_PROTOCOL_FIELDS) - {"smoke_override"}
    if not required_protocol_fields.issubset(metadata):
        raise ValueError(
            f"seed {run.seed} configuration or environment metadata is incomplete"
        )
    if (
        metadata.get("seed") != run.seed
        or metadata.get("evidence_status") != run.evidence_status
        or metadata.get("configuration_identity")
        != run.run_configuration_identity
        or metadata.get("test_partition_status") != "locked_not_accessed"
    ):
        raise ValueError(f"seed {run.seed} result and metadata identities disagree")
    selected = _require_mapping(
        metadata.get("selected_checkpoint"), f"seed {run.seed} selected checkpoint"
    )
    if (
        selected.get("identity") != run.checkpoint_identity
        or selected.get("epoch") != run.best_epoch
        or not math.isclose(
            float(selected.get("validation_mse", math.nan)),
            run.best_validation_mse,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise ValueError(f"seed {run.seed} selected checkpoint metadata is stale")
    if metadata.get("architecture") != dict(_EXPECTED_ARCHITECTURE):
        raise ValueError(f"seed {run.seed} architecture is incompatible")

    expected_identities = {
        "source": prepared.preprocessing.source_identity,
        "split": prepared.preprocessing.split_identity,
        "preprocessing": prepared.preprocessing.content_identity,
        "feature_schema": prepared.preprocessing.feature_schema_identity,
        "unit_schema": prepared.preprocessing.unit_schema_identity,
        "run_configuration": run.run_configuration_identity,
    }
    if metadata.get("source_checksums") != dict(prepared.source_checksums) or metadata.get(
        "identities"
    ) != expected_identities:
        raise ValueError(f"seed {run.seed} source or preprocessing identities are incompatible")
    compatibility = _require_mapping(
        metadata.get("compatibility"), f"seed {run.seed} compatibility"
    )
    expected_compatibility = {
        "source_identity": prepared.preprocessing.source_identity,
        "split_identity": prepared.preprocessing.split_identity,
        "preprocessing_identity": prepared.preprocessing.content_identity,
        "configuration_identity": run.run_configuration_identity,
    }
    if any(compatibility.get(name) != value for name, value in expected_compatibility.items()):
        raise ValueError(f"seed {run.seed} compatibility identities are incompatible")
    expected_content_identities = {
        "training_partition": prepared.partitions["training"].content_identity,
        "validation_partition": prepared.partitions["validation"].content_identity,
    }
    if compatibility.get("content_identities") != expected_content_identities:
        raise ValueError(f"seed {run.seed} partition identities are incompatible")
    if metadata.get("compatibility_identity") != _identity(compatibility):
        raise ValueError(f"seed {run.seed} compatibility identity is stale")
    if (
        history.get("compatibility") != compatibility
        or history.get("compatibility_identity") != metadata.get("compatibility_identity")
    ):
        raise ValueError(f"seed {run.seed} history compatibility is stale")

    try:
        predictions = np.load(run.validation_predictions_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"seed {run.seed} validation predictions cannot be loaded") from error
    if predictions.shape != (51, 48) or predictions.dtype != np.float64:
        raise ValueError(
            f"seed {run.seed} validation predictions must have shape (51, 48) and dtype float64"
        )
    if not np.isfinite(predictions).all():
        raise ValueError(f"seed {run.seed} must contain only finite validation predictions")
    prediction_metadata = _require_mapping(
        metadata.get("validation_predictions"),
        f"seed {run.seed} validation prediction metadata",
    )
    if prediction_metadata != {
        "shape": [51, 48],
        "dtype": "float64",
        "content_identity": sha256(predictions.tobytes()).hexdigest(),
    }:
        raise ValueError(f"seed {run.seed} validation prediction identity is stale")
    validation_truth = prepared.partitions["validation"].raw_aeps_fields.astype(
        np.float64
    )
    validation_mse = float(np.mean((predictions - validation_truth) ** 2))
    if not math.isclose(
        validation_mse,
        run.best_validation_mse,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"seed {run.seed} validation MSE does not match its predictions")

    events = [
        json.loads(line)
        for line in run.run_history_path.read_text(encoding="utf-8").splitlines()
    ]
    if not events or events[-1].get("event") != "run_completed":
        raise ValueError(f"seed {run.seed} interruption history is incomplete")

    protocol = {
        name: metadata[name] for name in _PROTOCOL_FIELDS if name in metadata
    }
    return _ValidatedSeedRun(
        result=run,
        metadata=metadata,
        history=history,
        predictions=predictions,
        compatibility=compatibility,
        protocol_identity=_identity(protocol),
    )


def _validation_report(
    prepared: PreparedDataArtifact,
    runs: list[_ValidatedSeedRun],
    comparator_evidence: list[SeedComparatorEvidence],
    *,
    canonical: bool,
) -> dict[str, object]:
    validation = prepared.partitions["validation"]
    truths = validation.raw_aeps_fields.astype(np.float64)
    points: tuple[tuple[float, float], ...] = tuple(
        (float(point[0]), float(point[1]))
        for point in prepared.preprocessing.element_points_mm
    )
    comparator_by_seed = {item.seed: item for item in comparator_evidence}
    seed_reports: list[SeedMetricReport] = []
    for run in runs:
        predictions = run.predictions
        case_metrics = tuple(
            _case_metrics(
                f"source_row_{source_row}",
                tuple(float(value) for value in truth),
                tuple(float(value) for value in prediction),
                points,
            )
            for source_row, truth, prediction in zip(
                validation.source_rows,
                truths,
                predictions,
                strict=True,
            )
        )
        squared_errors = (predictions - truths) ** 2
        hotspot_relative_l2 = [metric.hotspot_relative_l2 for metric in case_metrics]
        peak_error = [metric.peak_magnitude_relative_error for metric in case_metrics]
        location_error = [metric.hotspot_location_error_mm for metric in case_metrics]
        overlap = [metric.hotspot_overlap for metric in case_metrics]
        compatibility = run.compatibility
        content_identities = _require_mapping(
            compatibility.get("content_identities"), "seed content identities"
        )
        seed_reports.append(
            SeedMetricReport(
                seed=run.result.seed,
                checkpoint_identity=run.result.checkpoint_identity,
                precision_identity=str(compatibility["precision_identity"]),
                backend_identity=str(compatibility["backend_identity"]),
                content_identity=_identity(content_identities),
                compatibility_identity=str(
                    run.metadata["compatibility_identity"]
                ),
                global_mse=float(np.mean(squared_errors)),
                global_rmse=comparator_by_seed[
                    run.result.seed
                ].validation_global_rmse,
                hotspot_relative_l2_median=_median(hotspot_relative_l2),
                hotspot_relative_l2_p90=_nearest_rank_p90(hotspot_relative_l2),
                peak_magnitude_relative_error_median=_median(peak_error),
                peak_magnitude_relative_error_p90=_nearest_rank_p90(peak_error),
                hotspot_location_error_mm_median=_median(location_error),
                hotspot_location_error_mm_p90=_nearest_rank_p90(location_error),
                hotspot_overlap_mean=sum(overlap) / len(overlap),
                perfect_hotspot_overlap_fraction=(
                    sum(value == 1.0 for value in overlap) / len(overlap)
                ),
                negative_prediction_fraction=float(np.mean(predictions < 0.0)),
                most_negative_prediction=float(np.min(predictions)),
                case_metrics=case_metrics,
            )
        )
    report: dict[str, object] = {
        "schema_version": "aeps-metric-report-v1",
        "canonical": canonical,
        "evidence_status": (
            "canonical_validation_evidence"
            if canonical
            else "noncanonical_validation_evidence"
        ),
        "partition_authority_kind": "validation_manifest",
        "partition_authority_identity": validation.content_identity,
        "case_order_basis": "manifest",
        "case_order": [f"source_row_{row}" for row in validation.source_rows],
        "source_checksums": dict(prepared.source_checksums),
        "source_identity": prepared.preprocessing.source_identity,
        "split_identity": prepared.preprocessing.split_identity,
        "preprocessing_identity": prepared.preprocessing.content_identity,
        "seed_reports": [asdict(seed_report) for seed_report in seed_reports],
        "cross_seed_summary": {
            name: asdict(summary)
            for name, summary in _cross_seed_summary(seed_reports).items()
        },
    }
    report["report_content_identity"] = _identity(report)
    return report


def _copy_seed_artifacts(
    output: Path,
    prepared: PreparedDataArtifact,
    runs: list[_ValidatedSeedRun],
    comparator_evidence: list[SeedComparatorEvidence],
) -> list[dict[str, object]]:
    comparator_by_seed = {item.seed: item for item in comparator_evidence}
    predictors: list[dict[str, object]] = []
    for run in runs:
        seed_directory = output / "predictors" / f"seed_{run.result.seed}"
        seed_directory.mkdir(parents=True)
        sources = {
            "checkpoint": run.result.checkpoint_path,
            "validation_predictions": run.result.validation_predictions_path,
            "history": run.result.history_path,
            "metadata": run.result.metadata_path,
            "recovery_snapshot": run.result.recovery_snapshot_path,
            "run_history": run.result.run_history_path,
        }
        artifacts: dict[str, str] = {}
        for name, source in sources.items():
            suffix = source.suffix or ".bin"
            destination = seed_directory / f"{name}{suffix}"
            shutil.copy2(source, destination)
            artifacts[name] = destination.relative_to(output).as_posix()
        evidence = comparator_by_seed[run.result.seed]
        predictors.append(
            {
                "seed": run.result.seed,
                "checkpoint_identity": run.result.checkpoint_identity,
                "run_configuration_identity": run.result.run_configuration_identity,
                "validation_global_rmse": evidence.validation_global_rmse,
                "validation_comparator_status": evidence.comparator_status,
                "compatibility": dict(run.compatibility),
                "compatibility_identity": run.metadata["compatibility_identity"],
                "preprocessing_binding": {
                    "source_identity": prepared.preprocessing.source_identity,
                    "split_identity": prepared.preprocessing.split_identity,
                    "preprocessing_identity": (
                        prepared.preprocessing.content_identity
                    ),
                    "feature_schema_identity": (
                        prepared.preprocessing.feature_schema_identity
                    ),
                    "unit_schema_identity": (
                        prepared.preprocessing.unit_schema_identity
                    ),
                    "branch_feature_order": list(
                        prepared.preprocessing.branch_feature_order
                    ),
                },
                "artifacts": artifacts,
            }
        )
    return predictors


def _verify_repository_sources(root: Path, prepared: PreparedDataArtifact) -> None:
    paths = {
        "training_data": root / "data" / "combined_training_data.csv",
        "element_points": root / "data" / "co_ind.csv",
        "material_properties": root / "data" / "material_properties.md",
    }
    for name, path in paths.items():
        try:
            checksum = sha256(path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(
                f"canonical source cannot be read before freezing: {path}"
            ) from error
        if checksum != prepared.source_checksums[name]:
            raise ValueError(
                f"canonical source checksum changed before freezing: {name}"
            )


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be loaded") from error
    return _require_mapping(value, label)


def _verify_json_identity(payload: Mapping[str, Any], label: str) -> None:
    value = dict(payload)
    content_identity = value.pop("content_identity", None)
    if content_identity != _identity(value):
        raise ValueError(f"{label} content identity is stale")


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


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


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
