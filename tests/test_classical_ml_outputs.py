from __future__ import annotations

from pathlib import Path

import pandas as pd

from stroke_predict.evaluation import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    permutation_test,
    validate_patient_predictions,
)
from stroke_predict.ml_models import REQUIRED_MODEL_IDS, run_classical_ml_baselines


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model_id": "M1_fma_only",
                "outer_fold": 1,
                "subject_id": "STK-001",
                "label_true": "Good",
                "y_true": 1,
                "prob_good": 0.90,
                "pred_label": "Good",
                "threshold": 0.50,
            },
            {
                "model_id": "M1_fma_only",
                "outer_fold": 2,
                "subject_id": "STK-002",
                "label_true": "Poor",
                "y_true": 0,
                "prob_good": 0.20,
                "pred_label": "Poor",
                "threshold": 0.50,
            },
            {
                "model_id": "M1_fma_only",
                "outer_fold": 3,
                "subject_id": "STK-003",
                "label_true": "Good",
                "y_true": 1,
                "prob_good": 0.70,
                "pred_label": "Good",
                "threshold": 0.50,
            },
            {
                "model_id": "M1_fma_only",
                "outer_fold": 4,
                "subject_id": "STK-004",
                "label_true": "Poor",
                "y_true": 0,
                "prob_good": 0.40,
                "pred_label": "Poor",
                "threshold": 0.50,
            },
        ]
    )


def test_required_phase5_model_ids_are_configured() -> None:
    assert REQUIRED_MODEL_IDS == [
        "M0_majority",
        "M1_fma_only",
        "M2_clinical_only",
        "M3_psd_ml",
        "M4_fc_ml",
        "M5_tacs_target_ml",
        "M6_all_handcrafted_eeg_ml",
        "M12_clinical_plus_eeg_ml",
    ]


def test_patient_prediction_validation_rejects_duplicate_model_subject() -> None:
    duplicate = pd.concat([_predictions(), _predictions().iloc[[0]]], ignore_index=True)

    try:
        validate_patient_predictions(duplicate, expected_subject_count=4)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate model-subject validation failure")


def test_compute_classical_metrics_has_required_columns() -> None:
    metrics = compute_classification_metrics(_predictions())

    assert list(metrics["model_id"]) == ["M1_fma_only"]
    required = {
        "roc_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "pr_auc",
        "brier_score",
        "n_subjects",
    }
    assert required <= set(metrics.columns)
    row = metrics.iloc[0]
    assert row["roc_auc"] == 1.0
    assert row["balanced_accuracy"] == 1.0
    assert row["sensitivity"] == 1.0
    assert row["specificity"] == 1.0
    assert row["n_subjects"] == 4


def test_bootstrap_and_permutation_outputs_are_patient_level() -> None:
    predictions = _predictions()

    ci = bootstrap_metric_ci(predictions, n_bootstrap=25, random_seed=7)
    perm = permutation_test(predictions, n_permutations=25, random_seed=7)

    assert {
        "model_id",
        "metric",
        "observed_value",
        "ci_lower",
        "ci_upper",
        "n_bootstrap",
        "random_seed",
    } <= set(ci.columns)
    assert {
        "model_id",
        "metric",
        "observed_value",
        "null_mean",
        "null_std",
        "p_value",
        "n_permutations",
        "random_seed",
    } <= set(perm.columns)
    assert ci["n_bootstrap"].eq(25).all()
    assert perm["n_permutations"].eq(25).all()
    assert perm["p_value"].between(0, 1).all()


def _synthetic_cohort() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "STK-001",
                "role": "supervised_main",
                "label_primary": "Good",
                "age": 61,
                "sex": "M",
                "affected_hand": "right",
                "treated_hand": "right",
                "baseline_fma": 20,
                "baseline_mbi": 45,
                "mmse": 27,
            },
            {
                "subject_id": "STK-002",
                "role": "supervised_main",
                "label_primary": "Poor",
                "age": 63,
                "sex": "F",
                "affected_hand": "left",
                "treated_hand": "left",
                "baseline_fma": 32,
                "baseline_mbi": 60,
                "mmse": 28,
            },
            {
                "subject_id": "STK-003",
                "role": "supervised_main",
                "label_primary": "Good",
                "age": 59,
                "sex": "M",
                "affected_hand": "right",
                "treated_hand": "right",
                "baseline_fma": 22,
                "baseline_mbi": 40,
                "mmse": 26,
            },
            {
                "subject_id": "STK-004",
                "role": "supervised_main",
                "label_primary": "Poor",
                "age": 66,
                "sex": "F",
                "affected_hand": "left",
                "treated_hand": "left",
                "baseline_fma": 34,
                "baseline_mbi": 62,
                "mmse": 29,
            },
            {
                "subject_id": "STK-005",
                "role": "supervised_main",
                "label_primary": "Good",
                "age": 57,
                "sex": "M",
                "affected_hand": "right",
                "treated_hand": "right",
                "baseline_fma": 24,
                "baseline_mbi": 42,
                "mmse": 25,
            },
            {
                "subject_id": "STK-006",
                "role": "supervised_main",
                "label_primary": "Poor",
                "age": 68,
                "sex": "F",
                "affected_hand": "left",
                "treated_hand": "left",
                "baseline_fma": 36,
                "baseline_mbi": 64,
                "mmse": 30,
            },
        ]
    )


def _synthetic_registries(subjects: list[str]) -> tuple[dict[str, object], list[dict[str, object]]]:
    folds = {"n_supervised_main": len(subjects), "folds": []}
    registries = []
    for fold_index, test_subject in enumerate(subjects, start=1):
        train_subjects = [subject for subject in subjects if subject != test_subject]
        folds["folds"].append(
            {
                "outer_fold": fold_index,
                "test_subject": test_subject,
                "registry_path": f"fold_{fold_index:02d}_registry.json",
            }
        )
        registries.append(
            {
                "outer_fold": fold_index,
                "test_subject": test_subject,
                "supervised_train_subjects": train_subjects,
                "normalization_fit_subjects": train_subjects,
                "feature_selection_fit_subjects": train_subjects,
                "threshold_selection_subjects": train_subjects,
                "inner_splits": [
                    {
                        "inner_fold": 1,
                        "train_subjects": train_subjects[1:],
                        "val_subjects": train_subjects[:1],
                    },
                    {
                        "inner_fold": 2,
                        "train_subjects": train_subjects[:1] + train_subjects[2:],
                        "val_subjects": train_subjects[1:2],
                    },
                ],
            }
        )
    return folds, registries


def test_run_classical_ml_baselines_writes_required_outputs(tmp_path: Path) -> None:
    cohort = _synthetic_cohort()
    handcrafted = cohort[["subject_id", "label_primary"]].assign(
        eeg_power=[2.0, 0.5, 2.2, 0.4, 2.5, 0.3],
        native_fc_roi_eo_mean=[0.8, 0.2, 0.7, 0.3, 0.9, 0.1],
    )
    folds, registries = _synthetic_registries(cohort["subject_id"].tolist())
    config = {
        "random_seed": 5,
        "models": ["M0_majority", "M1_fma_only", "M2_clinical_only", "M5_tacs_target_ml"],
        "bootstrap_resamples": 10,
        "permutation_resamples": 10,
        "output_paths": {
            "predictions": str(tmp_path / "classical_patient_predictions.csv"),
            "metrics": str(tmp_path / "classical_metrics.csv"),
            "bootstrap_ci": str(tmp_path / "classical_bootstrap_ci.csv"),
            "permutation": str(tmp_path / "classical_permutation.csv"),
            "feature_importance": str(tmp_path / "classical_feature_importance.csv"),
        },
    }

    outputs = run_classical_ml_baselines(
        config,
        cohort=cohort,
        handcrafted=handcrafted,
        tacs=handcrafted,
        folds=folds,
        registries=registries,
    )

    predictions = pd.read_csv(outputs["predictions"])
    validate_patient_predictions(predictions, expected_subject_count=6)
    assert set(predictions["model_id"]) == {
        "M0_majority",
        "M1_fma_only",
        "M2_clinical_only",
        "M5_tacs_target_ml",
    }
    assert predictions.groupby("model_id").size().eq(6).all()
    assert Path(outputs["metrics"]).exists()
    assert Path(outputs["bootstrap_ci"]).exists()
    assert Path(outputs["permutation"]).exists()
    assert Path(outputs["feature_importance"]).exists()
