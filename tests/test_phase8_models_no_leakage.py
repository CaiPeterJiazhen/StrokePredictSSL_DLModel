from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stroke_predict.phase8_evaluation import (
    validate_phase8_no_leakage,
    validate_phase8_patient_predictions,
)
from stroke_predict.phase8_features import align_full_edge_features
from stroke_predict.phase8_models import Phase8ModelSpec, run_phase8_lopo_models


def test_full_edge_fc_inputs_align_with_matrix_subject_index_and_labels() -> None:
    subjects = [f"STK-{index:03d}" for index in range(1, 5)]
    matrix = np.arange(4 * 2 * 3 * 2, dtype=float).reshape(4, 2, 3, 2)
    matrix_subject_index = pd.DataFrame({"subject_id": subjects})
    labels = _toy_labels(subjects)

    features = align_full_edge_features(matrix, matrix_subject_index, labels, feature_prefix="reduced32")

    assert features["subject_id"].tolist() == subjects
    assert "reduced32_c0_e0_b0" in features.columns
    assert "reduced32_c1_e2_b1" in features.columns
    assert features.filter(like="reduced32_").shape[1] == 12


def test_full_edge_fc_alignment_rejects_missing_label_subjects() -> None:
    matrix = np.ones((2, 1, 2, 1), dtype=float)
    matrix_subject_index = pd.DataFrame({"subject_id": ["STK-001", "STK-002"]})
    labels = _toy_labels(["STK-001"])

    with pytest.raises(ValueError, match="labels"):
        align_full_edge_features(matrix, matrix_subject_index, labels)


def test_phase8_lopo_excludes_outer_test_from_all_fit_steps() -> None:
    subjects = [f"STK-{index:03d}" for index in range(1, 7)]
    result = run_phase8_lopo_models(
        config=_toy_model_config(models=["M14a_prop_reduced32_fullfc_ridge_logistic"]),
        features=_toy_features(subjects),
        labels=_toy_labels(subjects),
        folds=_toy_folds(subjects),
        run_mode="fast",
    )

    audit = result.no_leakage_audit
    assert not audit["outer_test_in_fit_subjects"].any()
    assert not audit["outer_test_in_transform_fit_subjects"].any()
    assert not audit["outer_test_in_inner_cv_subjects"].any()
    validate_phase8_no_leakage(audit)


def test_phase8_predictions_are_patient_level_without_duplicate_model_patient_rows() -> None:
    subjects = [f"STK-{index:03d}" for index in range(1, 7)]
    result = run_phase8_lopo_models(
        config=_toy_model_config(models=["M14b_prop_reduced32_fullfc_elasticnet"]),
        features=_toy_features(subjects),
        labels=_toy_labels(subjects),
        folds=_toy_folds(subjects),
        run_mode="fast",
    )

    predictions = result.predictions
    validate_phase8_patient_predictions(predictions, expected_patient_count=6)
    assert predictions.groupby("model_id")["patient_id"].nunique().eq(6).all()
    assert predictions.duplicated(["model_id", "patient_id"]).sum() == 0
    assert set(predictions["prediction_unit"]) == {"patient"}


def test_phase8_model_spec_defines_expected_primary_models() -> None:
    spec = Phase8ModelSpec.for_model_id("M14c_prop_reduced32_fullfc_linear_svm")

    assert spec.model_id == "M14c_prop_reduced32_fullfc_linear_svm"
    assert spec.feature_set == "reduced32_full_edge"
    assert spec.estimator == "linear_svm"


def test_full_mode_refuses_unplanned_full62_training() -> None:
    subjects = [f"STK-{index:03d}" for index in range(1, 7)]

    with pytest.raises(ValueError, match="M16 full62 full-mode"):
        run_phase8_lopo_models(
            config=_toy_model_config(models=["M16a_prop_full62_fullfc_ridge_logistic"]),
            features=_toy_features(subjects),
            labels=_toy_labels(subjects),
            folds=_toy_folds(subjects),
            run_mode="full",
            feature_set="full62",
        )


def _toy_features(subjects: list[str]) -> pd.DataFrame:
    labels = [index % 2 for index, _subject in enumerate(subjects)]
    return pd.DataFrame(
        {
            "subject_id": subjects,
            "fullfc_signal": [float(label) for label in labels],
            "fullfc_inverse": [1.0 - float(label) for label in labels],
            "roi_fc_signal": [float(label) + 0.1 for label in labels],
            "summary_signal": [float(label) + 0.2 for label in labels],
        }
    )


def _toy_labels(subjects: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": subjects,
            "phase8_label_status": ["analyzable"] * len(subjects),
            "primary_label_prop_residual": [
                "PoorRecovery" if index % 2 == 0 else "ProportionalRecovery"
                for index, _subject in enumerate(subjects)
            ],
            "primary_label_int_prop_residual": [index % 2 for index, _subject in enumerate(subjects)],
        }
    )


def _toy_folds(subjects: list[str]) -> dict[str, object]:
    folds = []
    for outer_fold, test_subject in enumerate(subjects, start=1):
        folds.append(
            {
                "outer_fold": outer_fold,
                "test_subject": test_subject,
                "supervised_train_subjects": [subject for subject in subjects if subject != test_subject],
            }
        )
    return {"n_supervised_main": len(subjects), "folds": folds}


def _toy_model_config(models: list[str]) -> dict[str, object]:
    return {
        "random_seed": 17,
        "models": models,
        "fast": {"bootstrap_resamples": 5, "permutation_resamples": 5},
        "m16_full62_full_mode_enabled": False,
    }
