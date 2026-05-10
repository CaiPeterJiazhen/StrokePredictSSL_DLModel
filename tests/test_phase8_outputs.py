from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stroke_predict.phase8_reports import (
    assert_no_forbidden_git_artifacts_staged,
    assert_phase8_public_output_safe,
    write_phase8_label_audit,
    write_phase8_model_report,
)


def test_phase8_required_outputs_are_generated(tmp_path: Path) -> None:
    label_paths = write_phase8_label_audit(_toy_label_table(), _toy_label_audit(), output_dir=tmp_path)
    model_paths = write_phase8_model_report(
        _toy_predictions(),
        _toy_metrics(),
        _toy_fc_audit(),
        output_dir=tmp_path,
        label_audit=_toy_label_audit(),
    )

    required = [
        tmp_path / "reports" / "phase8_label_audit.md",
        tmp_path / "evaluation" / "phase8_label_audit.csv",
        tmp_path / "evaluation" / "phase8_label_transition_table.csv",
        tmp_path / "predictions" / "phase8_prop_full_edge_patient_predictions.csv",
        tmp_path / "evaluation" / "phase8_prop_full_edge_metrics.csv",
        tmp_path / "reports" / "phase8_proportional_full_edge_fc_report.md",
    ]
    assert all(path.exists() for path in required)
    assert set(required) <= set(label_paths.values()) | set(model_paths.values())


def test_phase8_label_audit_outputs_required_columns(tmp_path: Path) -> None:
    write_phase8_label_audit(_toy_label_table(), _toy_label_audit(), output_dir=tmp_path)

    audit = pd.read_csv(tmp_path / "evaluation" / "phase8_label_audit.csv")
    transition = pd.read_csv(tmp_path / "evaluation" / "phase8_label_transition_table.csv")

    assert {
        "subject_id",
        "baseline_fma",
        "observed_delta",
        "residual",
        "primary_label_prop_residual",
        "primary_label_int_prop_residual",
        "current_clinically_meaningful",
    } <= set(audit.columns)
    assert {"current_clinically_meaningful", "primary_label_prop_residual", "n_patients"} <= set(transition.columns)


def test_phase8_model_report_contains_required_status_and_caution(tmp_path: Path) -> None:
    write_phase8_model_report(
        _toy_predictions(),
        _toy_metrics(permutation_p=0.50),
        _toy_fc_audit(),
        output_dir=tmp_path,
        label_audit=_toy_label_audit(),
    )

    report = (tmp_path / "reports" / "phase8_proportional_full_edge_fc_report.md").read_text(encoding="utf-8")

    assert "ProportionalRecovery vs PoorRecovery" in report
    assert "proportional-residual median split" in report
    assert "no SSL started" in report
    assert "no unplanned MatrixNet training started" in report
    assert "no post-treatment EEG supervised input" in report
    assert "source mode: psd_artifact_proxy" in report
    assert "Do not claim EEG efficacy" in report


def test_phase8_public_output_privacy_guard_rejects_private_strings() -> None:
    frame = pd.DataFrame({"subject_id": ["STK-001"], "path_like": ["C:" + "/local/private"]})

    with pytest.raises(ValueError, match="private"):
        assert_phase8_public_output_safe(frame)

    extension_frame = pd.DataFrame({"artifact": ["subject." + "set"]})
    with pytest.raises(ValueError, match="private"):
        assert_phase8_public_output_safe(extension_frame)


def test_phase8_staging_guard_rejects_outputs_and_private_artifacts() -> None:
    with pytest.raises(ValueError, match="outputs"):
        assert_no_forbidden_git_artifacts_staged(["outputs/evaluation/phase8_label_audit.csv"])

    with pytest.raises(ValueError, match="forbidden"):
        assert_no_forbidden_git_artifacts_staged(["data/private_file." + "fdt"])


def _toy_label_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["STK-001", "STK-002", "STK-003", "STK-004"],
            "baseline_fma": [40.0, 45.0, 66.0, 50.0],
            "post_fma": [60.0, 48.0, 66.0, 55.0],
            "expected_delta": [18.2, 14.7, pd.NA, 11.2],
            "observed_delta": [20.0, 3.0, 0.0, 5.0],
            "residual": [-1.8, 11.7, pd.NA, 6.2],
            "phase8_label_status": ["analyzable", "analyzable", "ceiling_exclude", "analyzable"],
            "primary_label_prop_residual": ["ProportionalRecovery", "PoorRecovery", pd.NA, "ProportionalRecovery"],
            "primary_label_int_prop_residual": [1, 0, pd.NA, 1],
            "current_clinically_meaningful": ["Good", "Poor", "ceiling_exclude", "Poor"],
            "absolute_70_achieved": ["ProportionalRecoveryAchieved", "NotAchieved", pd.NA, "NotAchieved"],
            "clear_residual_tertile": ["ClearProportionalRecovery", "ClearPoorRecovery", pd.NA, "middle_exclude"],
        }
    )


def _toy_label_audit() -> dict[str, object]:
    return {
        "n_analyzable": 3,
        "n_ceiling_exclude": 1,
        "n_missing_excluded": 0,
        "median_residual": 6.2,
        "n_proportional_recovery": 2,
        "n_poor_recovery": 1,
    }


def _toy_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["M14a_prop_reduced32_fullfc_ridge_logistic"] * 4,
            "outer_fold": [1, 2, 3, 4],
            "patient_id": ["STK-001", "STK-002", "STK-003", "STK-004"],
            "true_label": ["ProportionalRecovery", "PoorRecovery", "ProportionalRecovery", "PoorRecovery"],
            "y_true": [1, 0, 1, 0],
            "predicted_score": [0.8, 0.4, 0.7, 0.6],
            "predicted_label": ["ProportionalRecovery", "PoorRecovery", "ProportionalRecovery", "ProportionalRecovery"],
            "threshold": [0.5, 0.5, 0.5, 0.5],
            "prediction_unit": ["patient"] * 4,
        }
    )


def _toy_metrics(permutation_p: float = 0.50) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_id": ["M14a_prop_reduced32_fullfc_ridge_logistic"],
            "roc_auc": [0.75],
            "pr_auc": [0.83],
            "balanced_accuracy": [0.75],
            "sensitivity": [1.0],
            "specificity": [0.5],
            "f1": [0.80],
            "brier_score": [0.23],
            "bootstrap_ci_lower": [0.25],
            "bootstrap_ci_upper": [1.0],
            "permutation_p_value": [permutation_p],
            "tn": [1],
            "fp": [1],
            "fn": [0],
            "tp": [2],
            "mean_score_proportional": [0.75],
            "mean_score_poor": [0.50],
            "auc_score": [0.75],
            "auc_one_minus_score": [0.25],
        }
    )


def _toy_fc_audit() -> dict[str, object]:
    return {
        "reduced32_n_channels": 32,
        "reduced32_n_edges": 496,
        "metrics": "coherence, imaginary_coherence, wpli",
        "bands": "delta, theta, alpha_mu, low_beta, high_beta, broad_beta",
        "conditions": "EO, EC",
        "full62_smoke_status": "not_run",
        "source_mode": "psd_artifact_proxy",
    }
