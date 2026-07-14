from __future__ import annotations

from hashlib import sha256
import json
import math
from pathlib import Path
import re

import pytest

from new_pino import BaselineLifecycle, EvaluationContractError


FIXTURE_PACKAGE = Path(__file__).parent / "fixtures" / "prediction_package"
EVALUATION_FIXTURE = FIXTURE_PACKAGE / "evaluation_fixture.json"


def test_authorized_fixture_reports_exact_global_metrics_in_manifest_order(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "aeps_metric_report.json"

    report = BaselineLifecycle.from_package(FIXTURE_PACKAGE).evaluate(
        EVALUATION_FIXTURE,
        artifact_path=artifact_path,
    )

    assert report.schema_version == "aeps-metric-report-v1"
    assert report.canonical is False
    assert report.evidence_status == "noncanonical_fixture"
    assert report.partition_authority_kind == "authorized_fixture"
    assert report.partition_authority_identity == "hand-checkable-aeps-fixture-v1"
    assert report.case_order_basis == "manifest"
    assert report.case_order == ("manifest-case-002", "manifest-case-001")
    assert report.source_identity == (
        "c2ec12703f9236eacd827b046af43f4bda1bb5dca5bf60f6b5e683768934cd74"
    )
    assert report.split_identity == "synthetic-fixture-split-v1"
    assert report.preprocessing_identity == (
        "df1d4363ad2237ef198a3dca3e425844754a2ed7a84a98d0cbb1eb678b9d3469"
    )
    assert report.run_configuration_identity == "synthetic-fixture-run-v1"

    seed_0, seed_1 = report.seed_reports
    assert (seed_0.seed, seed_0.checkpoint_identity) == (
        0,
        "synthetic-fixture-checkpoint-0",
    )
    assert seed_0.global_mse == pytest.approx(3 / 16)
    assert seed_0.global_rmse == pytest.approx(math.sqrt(3) / 4)
    assert (seed_1.seed, seed_1.checkpoint_identity) == (
        1,
        "synthetic-fixture-checkpoint-1",
    )
    assert seed_1.global_mse == pytest.approx(77 / 96)
    assert seed_1.global_rmse == pytest.approx(math.sqrt(77 / 96))

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    report_content_identity = payload.pop("report_content_identity")
    expected_identity = sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert report.report_content_identity == report_content_identity
    assert report_content_identity == expected_identity
    assert re.fullmatch(r"[0-9a-f]{64}", report_content_identity)


def test_case_reports_preserve_exact_hotspot_and_negative_prediction_metrics() -> None:
    report = BaselineLifecycle.from_package(FIXTURE_PACKAGE).evaluate(
        EVALUATION_FIXTURE
    )
    seed_0, seed_1 = report.seed_reports

    seed_0_case_2, seed_0_case_1 = seed_0.case_metrics
    assert seed_0_case_2.simulation_case_identity == "manifest-case-002"
    assert seed_0_case_2.ground_truth_hotspot_element_indices == (1, 2, 3)
    assert seed_0_case_2.predicted_hotspot_element_indices == (1, 4, 2)
    assert seed_0_case_2.hotspot_relative_l2 == pytest.approx(2 / math.sqrt(11))
    assert seed_0_case_2.peak_magnitude_relative_error == 0.0
    assert seed_0_case_2.hotspot_location_error_mm == pytest.approx(1 / 3)
    assert seed_0_case_2.hotspot_overlap == pytest.approx(2 / 3)

    assert seed_0_case_1.simulation_case_identity == "manifest-case-001"
    assert seed_0_case_1.ground_truth_hotspot_element_indices == (7, 8, 9)
    assert seed_0_case_1.predicted_hotspot_element_indices == (7, 8, 9)
    assert seed_0_case_1.hotspot_relative_l2 == 0.0
    assert seed_0_case_1.peak_magnitude_relative_error == 0.0
    assert seed_0_case_1.hotspot_location_error_mm == 0.0
    assert seed_0_case_1.hotspot_overlap == 1.0
    assert seed_0.negative_prediction_fraction == pytest.approx(1 / 96)
    assert seed_0.most_negative_prediction == -1.0

    seed_1_case_2, seed_1_case_1 = seed_1.case_metrics
    assert seed_1_case_2.ground_truth_hotspot_element_indices == (1, 2, 3)
    assert seed_1_case_2.predicted_hotspot_element_indices == (1, 2, 3)
    assert seed_1_case_2.hotspot_relative_l2 == 0.0
    assert seed_1_case_1.ground_truth_hotspot_element_indices == (7, 8, 9)
    assert seed_1_case_1.predicted_hotspot_element_indices == (13, 14, 7)
    assert seed_1_case_1.hotspot_relative_l2 == pytest.approx(math.sqrt(7 / 11))
    assert seed_1_case_1.peak_magnitude_relative_error == 0.5
    assert seed_1_case_1.hotspot_location_error_mm == pytest.approx(
        2 * math.sqrt(2) / 3
    )
    assert seed_1_case_1.hotspot_overlap == pytest.approx(1 / 3)
    assert seed_1.negative_prediction_fraction == pytest.approx(1 / 96)
    assert seed_1.most_negative_prediction == -2.0


def test_seed_and_cross_seed_summaries_use_the_accepted_statistics() -> None:
    report = BaselineLifecycle.from_package(FIXTURE_PACKAGE).evaluate(
        EVALUATION_FIXTURE
    )
    seed_0, seed_1 = report.seed_reports

    assert seed_0.hotspot_relative_l2_median == pytest.approx(1 / math.sqrt(11))
    assert seed_0.hotspot_relative_l2_p90 == pytest.approx(2 / math.sqrt(11))
    assert seed_0.peak_magnitude_relative_error_median == 0.0
    assert seed_0.peak_magnitude_relative_error_p90 == 0.0
    assert seed_0.hotspot_location_error_mm_median == pytest.approx(1 / 6)
    assert seed_0.hotspot_location_error_mm_p90 == pytest.approx(1 / 3)
    assert seed_0.hotspot_overlap_mean == pytest.approx(5 / 6)
    assert seed_0.perfect_hotspot_overlap_fraction == 0.5

    assert seed_1.hotspot_relative_l2_median == pytest.approx(
        math.sqrt(7 / 11) / 2
    )
    assert seed_1.hotspot_relative_l2_p90 == pytest.approx(math.sqrt(7 / 11))
    assert seed_1.peak_magnitude_relative_error_median == 0.25
    assert seed_1.peak_magnitude_relative_error_p90 == 0.5
    assert seed_1.hotspot_location_error_mm_median == pytest.approx(
        math.sqrt(2) / 3
    )
    assert seed_1.hotspot_location_error_mm_p90 == pytest.approx(
        2 * math.sqrt(2) / 3
    )
    assert seed_1.hotspot_overlap_mean == pytest.approx(2 / 3)
    assert seed_1.perfect_hotspot_overlap_fraction == 0.5

    summary = report.cross_seed_summary
    assert summary["global_mse"].mean == pytest.approx(95 / 192)
    assert summary["global_mse"].sample_standard_deviation == pytest.approx(
        59 / (96 * math.sqrt(2))
    )
    expected_rmse = (math.sqrt(3) / 4 + math.sqrt(77 / 96)) / 2
    assert summary["global_rmse"].mean == pytest.approx(expected_rmse)
    assert summary["global_rmse"].sample_standard_deviation == pytest.approx(
        abs(math.sqrt(3) / 4 - math.sqrt(77 / 96)) / math.sqrt(2)
    )
    expected_hotspot_summaries = {
        "hotspot_relative_l2_median": (
            1 / math.sqrt(11),
            math.sqrt(7 / 11) / 2,
        ),
        "hotspot_relative_l2_p90": (2 / math.sqrt(11), math.sqrt(7 / 11)),
        "peak_magnitude_relative_error_median": (0.0, 0.25),
        "peak_magnitude_relative_error_p90": (0.0, 0.5),
        "hotspot_location_error_mm_median": (1 / 6, math.sqrt(2) / 3),
        "hotspot_location_error_mm_p90": (1 / 3, 2 * math.sqrt(2) / 3),
    }
    for metric, (seed_0_value, seed_1_value) in expected_hotspot_summaries.items():
        aggregate = summary[metric]
        assert aggregate.mean == pytest.approx((seed_0_value + seed_1_value) / 2)
        assert aggregate.sample_standard_deviation == pytest.approx(
            abs(seed_0_value - seed_1_value) / math.sqrt(2)
        )

    assert summary["hotspot_overlap_mean"].mean == pytest.approx(3 / 4)
    assert summary[
        "hotspot_overlap_mean"
    ].sample_standard_deviation == pytest.approx(1 / (6 * math.sqrt(2)))
    assert summary["perfect_hotspot_overlap_fraction"].mean == 0.5
    assert (
        summary["perfect_hotspot_overlap_fraction"].sample_standard_deviation
        == 0.0
    )
    assert summary["negative_prediction_fraction"].mean == pytest.approx(1 / 96)
    assert summary["negative_prediction_fraction"].sample_standard_deviation == 0.0
    assert summary["most_negative_prediction"].mean == -1.5
    assert summary[
        "most_negative_prediction"
    ].sample_standard_deviation == pytest.approx(
        1 / math.sqrt(2)
    )


def test_fixture_and_ad_hoc_reports_cannot_claim_official_evidence(
    tmp_path: Path,
) -> None:
    fixture = json.loads(EVALUATION_FIXTURE.read_text(encoding="utf-8"))
    fixture["evidence_status"] = "noncanonical_ad_hoc"
    fixture["partition_authority"] = {
        "kind": "ad_hoc",
        "identity": "ad-hoc-request-17",
    }
    fixture["case_order_basis"] = "request"
    ad_hoc_path = tmp_path / "ad_hoc_fixture.json"
    ad_hoc_path.write_text(json.dumps(fixture), encoding="utf-8")

    report = BaselineLifecycle.from_package(FIXTURE_PACKAGE).evaluate(ad_hoc_path)

    assert report.canonical is False
    assert report.evidence_status == "noncanonical_ad_hoc"
    assert report.partition_authority_kind == "ad_hoc"
    assert report.partition_authority_identity == "ad-hoc-request-17"
    assert report.case_order_basis == "request"

    fixture["canonical"] = True
    fixture["evidence_status"] = "official_held_out"
    fixture["partition_authority"] = {
        "kind": "locked_test",
        "identity": "forged-test-authority",
    }
    official_path = tmp_path / "official_fixture.json"
    official_path.write_text(json.dumps(fixture), encoding="utf-8")

    with pytest.raises(
        EvaluationContractError,
        match="must be machine-visibly noncanonical",
    ):
        BaselineLifecycle.from_package(FIXTURE_PACKAGE).evaluate(official_path)
