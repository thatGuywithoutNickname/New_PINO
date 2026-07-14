"""Exact AEPS metric reporting from noncanonical evaluation fixtures."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping


class EvaluationContractError(ValueError):
    """An evaluation fixture violates the public reporting contract."""


@dataclass(frozen=True)
class CaseMetricReport:
    simulation_case_identity: str
    ground_truth_hotspot_element_indices: tuple[int, int, int]
    predicted_hotspot_element_indices: tuple[int, int, int]
    hotspot_relative_l2: float
    peak_magnitude_relative_error: float
    hotspot_location_error_mm: float
    hotspot_overlap: float

@dataclass(frozen=True)
class SeedMetricReport:
    seed: int
    checkpoint_identity: str
    global_mse: float
    global_rmse: float
    hotspot_relative_l2_median: float
    hotspot_relative_l2_p90: float
    peak_magnitude_relative_error_median: float
    peak_magnitude_relative_error_p90: float
    hotspot_location_error_mm_median: float
    hotspot_location_error_mm_p90: float
    hotspot_overlap_mean: float
    perfect_hotspot_overlap_fraction: float
    negative_prediction_fraction: float
    most_negative_prediction: float
    case_metrics: tuple[CaseMetricReport, ...]

@dataclass(frozen=True)
class MeanAndSampleStandardDeviation:
    mean: float
    sample_standard_deviation: float


_CROSS_SEED_METRICS = (
    "global_mse",
    "global_rmse",
    "hotspot_relative_l2_median",
    "hotspot_relative_l2_p90",
    "peak_magnitude_relative_error_median",
    "peak_magnitude_relative_error_p90",
    "hotspot_location_error_mm_median",
    "hotspot_location_error_mm_p90",
    "hotspot_overlap_mean",
    "perfect_hotspot_overlap_fraction",
    "negative_prediction_fraction",
    "most_negative_prediction",
)


@dataclass(frozen=True)
class EvaluationReport:
    schema_version: str
    canonical: bool
    evidence_status: str
    partition_authority_kind: str
    partition_authority_identity: str
    case_order_basis: str
    case_order: tuple[str, ...]
    source_checksums: Mapping[str, str]
    source_identity: str
    split_identity: str
    preprocessing_identity: str
    run_configuration_identity: str
    seed_reports: tuple[SeedMetricReport, ...]
    cross_seed_summary: Mapping[str, MeanAndSampleStandardDeviation]
    report_content_identity: str


def _evaluate_fixture(
    fixture_path: str | Path,
    *,
    source_checksums: Mapping[str, str],
    source_identity: str,
    split_identity: str,
    preprocessing_identity: str,
    run_configuration_identity: str,
    predictor_identities: tuple[tuple[int, str], ...],
    element_points_mm: tuple[tuple[float, float], ...],
    artifact_path: str | Path | None,
) -> EvaluationReport:
    fixture = _load_fixture(fixture_path)
    if fixture.get("schema_version") != "aeps-evaluation-fixture-v1":
        raise EvaluationContractError(
            "evaluation fixture schema_version must be "
            "'aeps-evaluation-fixture-v1'"
        )
    if fixture.get("canonical") is not False:
        raise EvaluationContractError(
            "evaluation fixtures must be machine-visibly noncanonical"
        )

    expected_identities: Mapping[str, object] = {
        "source_checksums": dict(source_checksums),
        "source_identity": source_identity,
        "split_identity": split_identity,
        "preprocessing_identity": preprocessing_identity,
        "run_configuration_identity": run_configuration_identity,
    }
    for name, expected in expected_identities.items():
        if fixture.get(name) != expected:
            raise EvaluationContractError(
                f"evaluation fixture {name} does not match the loaded package"
            )

    authority = _require_mapping(
        fixture.get("partition_authority"), "evaluation partition_authority"
    )
    authority_kind = authority.get("kind")
    reporting_modes = {
        "authorized_fixture": ("noncanonical_fixture", "manifest"),
        "ad_hoc": ("noncanonical_ad_hoc", "request"),
    }
    if authority_kind not in reporting_modes:
        raise EvaluationContractError(
            "evaluation partition_authority kind must be 'authorized_fixture' "
            "or 'ad_hoc'; official held-out evidence is unsupported"
        )
    evidence_status, case_order_basis = reporting_modes[str(authority_kind)]
    if fixture.get("evidence_status") != evidence_status:
        raise EvaluationContractError(
            f"evaluation {authority_kind} reports must use evidence_status "
            f"{evidence_status!r}"
        )
    authority_identity = _require_identity(
        authority.get("identity"), "evaluation partition-authority identity"
    )
    if fixture.get("case_order_basis") != case_order_basis:
        raise EvaluationContractError(
            f"evaluation {authority_kind} reports must declare "
            f"{case_order_basis} case order"
        )

    cases = fixture.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvaluationContractError(
            "evaluation fixture must contain at least one simulation case"
        )
    case_order: list[str] = []
    truths: list[tuple[float, ...]] = []
    for case_number, raw_case in enumerate(cases, start=1):
        case = _require_mapping(raw_case, f"evaluation case {case_number}")
        case_identity = _require_identity(
            case.get("simulation_case_identity"),
            f"evaluation case {case_number} identity",
        )
        if case_identity in case_order:
            raise EvaluationContractError(
                "evaluation simulation-case identities must be unique"
            )
        truth = _require_aeps_field(
            case.get("ground_truth_aeps"),
            f"evaluation case {case_number} ground truth",
        )
        if any(value < 0.0 for value in truth) or not any(
            value > 0.0 for value in truth
        ):
            raise EvaluationContractError(
                f"evaluation case {case_number} ground truth must be non-negative "
                "with at least one positive AEPS value"
            )
        case_order.append(case_identity)
        truths.append(truth)

    predictions = fixture.get("predictions")
    if not isinstance(predictions, list) or len(predictions) < 2:
        raise EvaluationContractError(
            "evaluation fixture must contain at least two predictors for sample "
            "standard deviation"
        )
    seed_reports: list[SeedMetricReport] = []
    seen_seeds: set[int] = set()
    seen_checkpoints: set[str] = set()
    for prediction_number, raw_prediction in enumerate(predictions, start=1):
        prediction = _require_mapping(
            raw_prediction, f"evaluation predictor {prediction_number}"
        )
        seed = prediction.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise EvaluationContractError(
                f"evaluation predictor {prediction_number} seed must be an integer"
            )
        checkpoint_identity = _require_identity(
            prediction.get("checkpoint_identity"),
            f"evaluation predictor {prediction_number} checkpoint identity",
        )
        if seed in seen_seeds or checkpoint_identity in seen_checkpoints:
            raise EvaluationContractError(
                "evaluation seeds and checkpoint identities must be unique"
            )
        seen_seeds.add(seed)
        seen_checkpoints.add(checkpoint_identity)
        if (seed, checkpoint_identity) not in predictor_identities:
            raise EvaluationContractError(
                f"seed {seed} and checkpoint identity {checkpoint_identity!r} do not "
                "identify one predictor in the loaded package"
            )

        raw_fields = prediction.get("aeps_fields")
        if not isinstance(raw_fields, list) or len(raw_fields) != len(truths):
            raise EvaluationContractError(
                f"evaluation predictor {prediction_number} must contain one AEPS "
                "field per simulation case"
            )
        squared_error_sum = 0.0
        prediction_values: list[float] = []
        case_metrics: list[CaseMetricReport] = []
        for case_number, (truth, raw_field) in enumerate(
            zip(truths, raw_fields), start=1
        ):
            field = _require_aeps_field(
                raw_field,
                f"evaluation predictor {prediction_number} case {case_number}",
            )
            squared_error_sum += sum(
                (predicted - expected) ** 2
                for predicted, expected in zip(field, truth)
            )
            prediction_values.extend(field)
            case_metrics.append(
                _case_metrics(
                    case_order[case_number - 1],
                    truth,
                    field,
                    element_points_mm,
                )
            )
        global_mse = squared_error_sum / (len(truths) * 48)
        hotspot_relative_l2_values = [
            metric.hotspot_relative_l2 for metric in case_metrics
        ]
        peak_error_values = [
            metric.peak_magnitude_relative_error for metric in case_metrics
        ]
        location_error_values = [
            metric.hotspot_location_error_mm for metric in case_metrics
        ]
        overlap_values = [metric.hotspot_overlap for metric in case_metrics]
        seed_reports.append(
            SeedMetricReport(
                seed=seed,
                checkpoint_identity=checkpoint_identity,
                global_mse=global_mse,
                global_rmse=math.sqrt(global_mse),
                hotspot_relative_l2_median=_median(
                    hotspot_relative_l2_values
                ),
                hotspot_relative_l2_p90=_nearest_rank_p90(
                    hotspot_relative_l2_values
                ),
                peak_magnitude_relative_error_median=_median(peak_error_values),
                peak_magnitude_relative_error_p90=_nearest_rank_p90(
                    peak_error_values
                ),
                hotspot_location_error_mm_median=_median(location_error_values),
                hotspot_location_error_mm_p90=_nearest_rank_p90(
                    location_error_values
                ),
                hotspot_overlap_mean=sum(overlap_values) / len(overlap_values),
                perfect_hotspot_overlap_fraction=(
                    sum(value == 1.0 for value in overlap_values)
                    / len(overlap_values)
                ),
                negative_prediction_fraction=(
                    sum(value < 0.0 for value in prediction_values)
                    / len(prediction_values)
                ),
                most_negative_prediction=min(prediction_values),
                case_metrics=tuple(case_metrics),
            )
        )

    report_without_identity = EvaluationReport(
        schema_version="aeps-metric-report-v1",
        canonical=False,
        evidence_status=evidence_status,
        partition_authority_kind=str(authority_kind),
        partition_authority_identity=authority_identity,
        case_order_basis=case_order_basis,
        case_order=tuple(case_order),
        source_checksums=dict(source_checksums),
        source_identity=source_identity,
        split_identity=split_identity,
        preprocessing_identity=preprocessing_identity,
        run_configuration_identity=run_configuration_identity,
        seed_reports=tuple(seed_reports),
        cross_seed_summary=_cross_seed_summary(seed_reports),
        report_content_identity="",
    )
    content = asdict(report_without_identity)
    content.pop("report_content_identity")
    report = replace(
        report_without_identity,
        report_content_identity=_content_identity(content),
    )
    if artifact_path is not None:
        Path(artifact_path).write_text(
            json.dumps(asdict(report), indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def _load_fixture(path: str | Path) -> Mapping[str, Any]:
    fixture_path = Path(path)
    try:
        with fixture_path.open("r", encoding="utf-8") as stream:
            return _require_mapping(
                json.load(stream), "evaluation fixture"
            )
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationContractError(
            f"evaluation fixture {fixture_path} cannot be loaded: {error}"
        ) from error


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise EvaluationContractError(f"{label} must be an object")
    return value


def _require_identity(value: object, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise EvaluationContractError(f"{label} must be a non-empty string")
    return value


def _require_aeps_field(value: object, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != 48:
        raise EvaluationContractError(f"{label} must contain exactly 48 values")
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in value
    ):
        raise EvaluationContractError(f"{label} values must be finite numbers")
    return tuple(float(item) for item in value)


def _case_metrics(
    simulation_case_identity: str,
    truth: tuple[float, ...],
    prediction: tuple[float, ...],
    element_points_mm: tuple[tuple[float, float], ...],
) -> CaseMetricReport:
    ground_truth_hotspot = _hotspot_indices(truth)
    predicted_hotspot = _hotspot_indices(prediction)
    truth_hotspot_norm = math.sqrt(
        sum(truth[index - 1] ** 2 for index in ground_truth_hotspot)
    )
    hotspot_error_norm = math.sqrt(
        sum(
            (prediction[index - 1] - truth[index - 1]) ** 2
            for index in ground_truth_hotspot
        )
    )
    truth_centroid = _centroid(ground_truth_hotspot, element_points_mm)
    prediction_centroid = _centroid(predicted_hotspot, element_points_mm)
    return CaseMetricReport(
        simulation_case_identity=simulation_case_identity,
        ground_truth_hotspot_element_indices=ground_truth_hotspot,
        predicted_hotspot_element_indices=predicted_hotspot,
        hotspot_relative_l2=hotspot_error_norm / truth_hotspot_norm,
        peak_magnitude_relative_error=(
            abs(max(prediction) - max(truth)) / max(truth)
        ),
        hotspot_location_error_mm=math.dist(
            truth_centroid, prediction_centroid
        ),
        hotspot_overlap=(
            len(set(ground_truth_hotspot) & set(predicted_hotspot)) / 3
        ),
    )


def _hotspot_indices(field: tuple[float, ...]) -> tuple[int, int, int]:
    ranked = sorted(
        enumerate(field, start=1),
        key=lambda item: (-item[1], item[0]),
    )
    return (ranked[0][0], ranked[1][0], ranked[2][0])


def _centroid(
    indices: tuple[int, int, int],
    element_points_mm: tuple[tuple[float, float], ...],
) -> tuple[float, float]:
    return (
        sum(element_points_mm[index - 1][0] for index in indices) / 3,
        sum(element_points_mm[index - 1][1] for index in indices) / 3,
    )


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def _nearest_rank_p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def _cross_seed_summary(
    seed_reports: list[SeedMetricReport],
) -> dict[str, MeanAndSampleStandardDeviation]:
    return {
        metric: _mean_and_sample_standard_deviation(
            [getattr(report, metric) for report in seed_reports]
        )
        for metric in _CROSS_SEED_METRICS
    }


def _mean_and_sample_standard_deviation(
    values: list[float],
) -> MeanAndSampleStandardDeviation:
    mean = sum(values) / len(values)
    return MeanAndSampleStandardDeviation(
        mean=mean,
        sample_standard_deviation=math.sqrt(
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        ),
    )


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
