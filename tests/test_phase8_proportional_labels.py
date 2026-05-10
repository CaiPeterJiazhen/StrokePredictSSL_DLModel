from __future__ import annotations

import math

import pandas as pd

from stroke_predict.phase8_labels import (
    LABEL_TO_INT_PROP_RESIDUAL,
    MAX_FMA_UE,
    build_phase8_label_table,
    compute_proportional_recovery_record,
    label_with_train_median_threshold,
)


def test_proportional_residual_formula_and_primary_label_direction() -> None:
    record = compute_proportional_recovery_record("STK-001", 40, 55, median_residual=10.0)

    assert MAX_FMA_UE == 66
    assert math.isclose(record["expected_delta"], 0.7 * (66 - 40))
    assert record["observed_delta"] == 15.0
    assert math.isclose(record["residual"], 0.7 * 26 - 15)
    assert record["primary_label_prop_residual"] == "ProportionalRecovery"
    assert record["primary_label_int_prop_residual"] == 1


def test_smaller_residual_maps_to_proportional_recovery() -> None:
    proportional = compute_proportional_recovery_record("STK-001", 40, 60, median_residual=5.0)
    poor = compute_proportional_recovery_record("STK-002", 40, 45, median_residual=5.0)

    assert proportional["residual"] < poor["residual"]
    assert proportional["primary_label_prop_residual"] == "ProportionalRecovery"
    assert poor["primary_label_prop_residual"] == "PoorRecovery"


def test_residual_tie_goes_to_proportional_and_above_median_is_poor() -> None:
    tied = compute_proportional_recovery_record("STK-002", 50, 55, median_residual=6.2)
    poor = compute_proportional_recovery_record("STK-003", 50, 53, median_residual=4.0)

    assert math.isclose(tied["residual"], 6.2)
    assert tied["primary_label_prop_residual"] == "ProportionalRecovery"
    assert poor["residual"] > 4.0
    assert poor["primary_label_prop_residual"] == "PoorRecovery"
    assert poor["primary_label_int_prop_residual"] == 0


def test_ceiling_and_missing_are_excluded_from_primary_classification() -> None:
    ceiling = compute_proportional_recovery_record("STK-004", 66, 66, median_residual=0.0)
    missing_baseline = compute_proportional_recovery_record("STK-005", None, 50, median_residual=0.0)
    missing_post = compute_proportional_recovery_record("STK-006", 40, None, median_residual=0.0)

    assert ceiling["phase8_label_status"] == "ceiling_exclude"
    assert pd.isna(ceiling["primary_label_int_prop_residual"])
    assert missing_baseline["phase8_label_status"] == "excluded_missing"
    assert missing_post["phase8_label_status"] == "excluded_missing"


def test_phase8_label_table_uses_cohort_median_and_encodes_classes() -> None:
    cohort = pd.DataFrame(
        {
            "subject_id": ["STK-001", "STK-002", "STK-003", "STK-004", "STK-005", "STK-006"],
            "baseline_fma": [40, 50, 60, 66, None, 45],
            "post_fma": [60, 55, 61, 66, 50, None],
            "label_primary": ["Good", "Poor", "Poor", "ceiling_exclude", "missing", "missing"],
        }
    )

    labels, audit = build_phase8_label_table(cohort)

    analyzable = labels.loc[labels["phase8_label_status"].eq("analyzable")]
    expected_median = float(analyzable["residual"].median())
    assert audit["n_analyzable"] == 3
    assert audit["n_ceiling_exclude"] == 1
    assert audit["n_missing_excluded"] == 2
    assert math.isclose(audit["median_residual"], expected_median)
    assert set(analyzable["primary_label_prop_residual"]) == {"ProportionalRecovery", "PoorRecovery"}
    assert set(analyzable["primary_label_int_prop_residual"].astype(int)) == {0, 1}
    assert LABEL_TO_INT_PROP_RESIDUAL == {"PoorRecovery": 0, "ProportionalRecovery": 1}


def test_sensitivity_labels_are_generated_for_analyzable_patients() -> None:
    cohort = pd.DataFrame(
        {
            "subject_id": ["STK-001", "STK-002", "STK-003", "STK-004", "STK-005"],
            "baseline_fma": [40, 40, 40, 40, 40],
            "post_fma": [62, 58, 50, 45, 41],
            "label_primary": ["Good", "Good", "Good", "Poor", "Poor"],
        }
    )

    labels, _audit = build_phase8_label_table(cohort)

    assert "absolute_70_achieved" in labels.columns
    assert "current_clinically_meaningful" in labels.columns
    assert "clear_residual_tertile" in labels.columns
    assert set(labels["current_clinically_meaningful"]) == {"Good", "Poor"}
    assert "ClearProportionalRecovery" in set(labels["clear_residual_tertile"])
    assert "ClearPoorRecovery" in set(labels["clear_residual_tertile"])


def test_train_median_threshold_labels_held_out_patient_using_train_only_median() -> None:
    label_table = pd.DataFrame(
        {
            "subject_id": ["STK-001", "STK-002", "STK-003", "STK-004"],
            "residual": [-5.0, 0.0, 5.0, 20.0],
            "phase8_label_status": ["analyzable", "analyzable", "analyzable", "analyzable"],
        }
    )

    labeled = label_with_train_median_threshold(
        label_table,
        train_subjects=["STK-001", "STK-002", "STK-003"],
        test_subject="STK-004",
    )

    assert labeled["train_median_residual"] == 0.0
    assert labeled["test_subject"] == "STK-004"
    assert labeled["primary_label_train_median_threshold"] == "PoorRecovery"
    assert labeled["primary_label_int_train_median_threshold"] == 0
