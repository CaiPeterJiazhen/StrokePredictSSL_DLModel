from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stroke_predict.phase8_1_validation import (
    apply_multiple_comparison_correction,
    audit_comparison_models,
    audit_source_mode,
    build_patient_error_audit,
    build_threshold_calibration_table,
    write_phase8_1_validation_outputs,
)


def test_source_mode_proxy_blocks_time_series_claims() -> None:
    audit = audit_source_mode(pd.DataFrame({"source_mode": ["psd_artifact_proxy"], "n_edges": [496]}))

    assert audit["source_mode"] == "psd_artifact_proxy"
    assert audit["is_real_time_series_fc"] is False
    assert "must not be called real time-series" in audit["claim_guard"]


def test_source_mode_time_series_allows_real_fc_claims() -> None:
    audit = audit_source_mode({"source_mode": "time_series", "n_edges": 496})

    assert audit["source_mode"] == "time_series"
    assert audit["is_real_time_series_fc"] is True
    assert audit["claim_guard"] == "real baseline EO/EC time-series full-edge FC"


def test_m15_audit_reports_different_features_but_identical_predictions() -> None:
    audit = audit_comparison_models(
        features=_comparison_feature_table(different=True),
        predictions=_comparison_predictions(identical=True),
        model_a="M15a_prop_roi_fc_best_ml",
        model_b="M15b_prop_summary_eeg_best_ml",
    )

    assert audit["feature_matrices_identical"] is False
    assert audit["predictions_identical"] is True
    assert "same predictions despite different feature matrices" in audit["explanation"]


def test_m15_audit_rejects_silent_shared_predictions() -> None:
    with pytest.raises(ValueError, match="silently share predictions"):
        audit_comparison_models(
            features=_comparison_feature_table(different=False),
            predictions=_comparison_predictions(identical=True),
            model_a="M15a_prop_roi_fc_best_ml",
            model_b="M15b_prop_summary_eeg_best_ml",
        )


def test_multiple_comparison_correction_adds_raw_bonferroni_and_fdr_flags() -> None:
    corrected = apply_multiple_comparison_correction(_permutation_table())

    assert {
        "model_id",
        "raw_permutation_p_value",
        "bonferroni_p_value",
        "fdr_q_value",
        "nominal_p_lt_0_05",
        "fdr_q_lt_0_05",
        "bonferroni_p_lt_0_05",
    } <= set(corrected.columns)
    assert corrected.loc[corrected["model_id"].eq("M14b"), "bonferroni_p_value"].iloc[0] == pytest.approx(0.045 * 6)
    assert corrected.loc[corrected["model_id"].eq("M14b"), "nominal_p_lt_0_05"].iloc[0] is True
    assert corrected.loc[corrected["model_id"].eq("M14b"), "bonferroni_p_lt_0_05"].iloc[0] is False


def test_threshold_calibration_outputs_fixed_inner_youden_calibration_and_distribution() -> None:
    table = build_threshold_calibration_table(_best_model_predictions(include_inner=True))

    assert set(table["analysis_type"]) >= {
        "fixed_0.5_threshold",
        "inner_cv_threshold",
        "inner_cv_youden_threshold",
        "calibration_bin",
        "score_distribution_by_group",
    }
    assert "brier_score" in table.columns
    assert table.loc[table["analysis_type"].eq("calibration_bin"), "bin_count"].sum() == 6


def test_missing_inner_thresholds_are_reported_not_recomputed() -> None:
    table = build_threshold_calibration_table(_best_model_predictions(include_inner=False))

    unavailable = table.loc[table["analysis_type"].isin(["inner_cv_threshold", "inner_cv_youden_threshold"])]
    assert set(unavailable["status"]) == {"not_available"}
    assert unavailable["threshold_source"].str.contains("outer test predictions not used").all()


def test_patient_error_audit_contains_required_columns_and_boundary_flags() -> None:
    audit = build_patient_error_audit(_labels_for_error_audit(), _best_model_predictions(include_inner=True))

    assert {
        "patient_id",
        "old_label",
        "proportional_label",
        "baseline_fma",
        "post_fma",
        "observed_delta",
        "expected_delta",
        "residual",
        "predicted_score",
        "predicted_label",
        "correct",
        "rank",
        "near_median_threshold",
        "old_new_label_disagree",
    } <= set(audit.columns)
    assert audit["near_median_threshold"].any()
    assert audit["old_new_label_disagree"].any()


def test_phase8_1_report_files_are_written_and_answer_required_questions(tmp_path: Path) -> None:
    source_audit = audit_source_mode({"source_mode": "psd_artifact_proxy", "n_edges": 496})
    duplicate_audit = audit_comparison_models(
        features=_comparison_feature_table(different=True),
        predictions=_comparison_predictions(identical=True),
        model_a="M15a_prop_roi_fc_best_ml",
        model_b="M15b_prop_summary_eeg_best_ml",
    )
    corrections = apply_multiple_comparison_correction(_permutation_table())
    threshold = build_threshold_calibration_table(_best_model_predictions(include_inner=True))
    patient_audit = build_patient_error_audit(_labels_for_error_audit(), _best_model_predictions(include_inner=True))
    paths = write_phase8_1_validation_outputs(
        output_dir=tmp_path,
        source_audit=source_audit,
        duplicate_audit=duplicate_audit,
        correction_table=corrections,
        threshold_calibration=threshold,
        patient_error_audit=patient_audit,
        no_leakage_audit=_no_leakage_audit(),
        best_model_id="M14b",
        real_time_series_reproduced=False,
    )

    expected = {
        tmp_path / "reports" / "phase8_1_validation_report.md",
        tmp_path / "evaluation" / "phase8_1_multiple_comparison_correction.csv",
        tmp_path / "evaluation" / "phase8_1_threshold_calibration.csv",
        tmp_path / "evaluation" / "phase8_1_patient_error_audit.csv",
        tmp_path / "reports" / "phase8_1_no_leakage_report.txt",
    }
    assert expected <= set(paths.values())
    assert all(path.exists() for path in expected)

    report = (tmp_path / "reports" / "phase8_1_validation_report.md").read_text(encoding="utf-8")
    for question in [
        "Was full-edge FC real time-series FC or proxy?",
        "If proxy, what must be done before claiming time-series FC evidence?",
        "Does real time-series FC reproduce the Phase 8 positive signal?",
        "Why were ROI-FC and summary EEG metrics identical?",
        "Does the best model survive FDR or Bonferroni correction?",
        "Is classification performance limited by threshold choice?",
        "Are errors concentrated near the residual median boundary?",
        "Is the result strong enough to justify no-SSL MatrixNet next?",
        "Should SSL remain blocked until real FC and MatrixNet validation?",
    ]:
        assert question in report
    assert "proxy" in report
    assert "SSL should remain blocked" in report

    no_leakage = (tmp_path / "reports" / "phase8_1_no_leakage_report.txt").read_text(encoding="utf-8")
    assert "PASS" in no_leakage


def _comparison_feature_table(*, different: bool) -> pd.DataFrame:
    subjects = [f"STK-{index:03d}" for index in range(1, 7)]
    roi_values = [0.10, 0.70, 0.20, 0.80, 0.30, 0.90]
    summary_values = [0.10, 0.70, 0.20, 0.80, 0.30, 0.90] if not different else [0.90, 0.30, 0.80, 0.20, 0.70, 0.10]
    return pd.DataFrame(
        {
            "subject_id": subjects,
            "roi_fc_alpha": roi_values,
            "summary_alpha": summary_values,
        }
    )


def _comparison_predictions(*, identical: bool) -> pd.DataFrame:
    subjects = [f"STK-{index:03d}" for index in range(1, 7)]
    rows: list[dict[str, object]] = []
    for model_id in ("M15a_prop_roi_fc_best_ml", "M15b_prop_summary_eeg_best_ml"):
        for index, subject in enumerate(subjects):
            score = [0.2, 0.7, 0.3, 0.8, 0.4, 0.9][index]
            if model_id.startswith("M15b") and not identical:
                score = 1.0 - score
            rows.append(
                {
                    "model_id": model_id,
                    "patient_id": subject,
                    "predicted_score": score,
                    "predicted_label": "ProportionalRecovery" if score >= 0.5 else "PoorRecovery",
                }
            )
    return pd.DataFrame(rows)


def _permutation_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["M14a", "M14b", "M14c", "M14d", "M15a", "M15b"],
            "permutation_p_value": [0.40, 0.045, 0.20, 0.80, 0.50, 0.50],
        }
    )


def _best_model_predictions(*, include_inner: bool) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "model_id": ["M14b"] * 6,
            "outer_fold": [1, 2, 3, 4, 5, 6],
            "patient_id": [f"STK-{index:03d}" for index in range(1, 7)],
            "y_true": [0, 1, 0, 1, 0, 1],
            "true_label": [
                "PoorRecovery",
                "ProportionalRecovery",
                "PoorRecovery",
                "ProportionalRecovery",
                "PoorRecovery",
                "ProportionalRecovery",
            ],
            "predicted_score": [0.30, 0.55, 0.45, 0.70, 0.52, 0.82],
            "predicted_label": [
                "PoorRecovery",
                "ProportionalRecovery",
                "PoorRecovery",
                "ProportionalRecovery",
                "ProportionalRecovery",
                "ProportionalRecovery",
            ],
            "threshold": [0.5] * 6,
            "prediction_unit": ["patient"] * 6,
        }
    )
    if include_inner:
        frame["inner_cv_threshold"] = [0.48, 0.50, 0.47, 0.53, 0.51, 0.49]
        frame["inner_cv_youden_threshold"] = [0.55, 0.55, 0.55, 0.60, 0.58, 0.56]
    return frame


def _labels_for_error_audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"STK-{index:03d}" for index in range(1, 7)],
            "current_clinically_meaningful": ["Poor", "Good", "Poor", "Good", "Good", "Poor"],
            "primary_label_prop_residual": [
                "PoorRecovery",
                "ProportionalRecovery",
                "PoorRecovery",
                "ProportionalRecovery",
                "PoorRecovery",
                "ProportionalRecovery",
            ],
            "baseline_fma": [40.0, 42.0, 44.0, 46.0, 48.0, 50.0],
            "post_fma": [45.0, 60.0, 50.0, 61.0, 55.0, 62.0],
            "observed_delta": [5.0, 18.0, 6.0, 15.0, 7.0, 12.0],
            "expected_delta": [18.2, 16.8, 15.4, 14.0, 12.6, 11.2],
            "residual": [13.2, -1.2, 9.4, -1.0, 5.6, -0.8],
            "median_residual": [2.4] * 6,
        }
    )


def _no_leakage_audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["M14b", "M14b"],
            "outer_fold": [1, 2],
            "test_subject": ["STK-001", "STK-002"],
            "outer_test_in_fit_subjects": [False, False],
            "outer_test_in_transform_fit_subjects": [False, False],
            "outer_test_in_inner_cv_subjects": [False, False],
        }
    )
