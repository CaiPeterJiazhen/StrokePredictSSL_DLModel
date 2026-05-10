from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from stroke_predict.cohort.labels import parse_optional_float


MAX_FMA_UE = 66
PROPORTIONAL_EXPECTED_FRACTION = 0.70
POOR_RECOVERY = "PoorRecovery"
PROPORTIONAL_RECOVERY = "ProportionalRecovery"
LABEL_TO_INT_PROP_RESIDUAL = {POOR_RECOVERY: 0, PROPORTIONAL_RECOVERY: 1}
INT_TO_LABEL_PROP_RESIDUAL = {value: key for key, value in LABEL_TO_INT_PROP_RESIDUAL.items()}


def compute_proportional_recovery_record(
    subject_id: Any,
    baseline_fma: Any,
    post_fma: Any,
    *,
    median_residual: float | None,
) -> dict[str, Any]:
    subject = str(subject_id)
    baseline = parse_optional_float(baseline_fma)
    post = parse_optional_float(post_fma)
    base = {
        "subject_id": subject,
        "baseline_fma": baseline,
        "post_fma": post,
        "expected_delta": np.nan,
        "observed_delta": np.nan,
        "residual": np.nan,
        "median_residual": float(median_residual) if median_residual is not None and math.isfinite(float(median_residual)) else np.nan,
        "phase8_label_status": "analyzable",
        "primary_label_prop_residual": pd.NA,
        "primary_label_int_prop_residual": pd.NA,
        "absolute_70_achieved": pd.NA,
    }
    if baseline is None or post is None:
        return {**base, "phase8_label_status": "excluded_missing"}
    if not _valid_fma_score(baseline) or not _valid_fma_score(post):
        return {**base, "phase8_label_status": "excluded_missing"}
    if baseline == MAX_FMA_UE:
        return {
            **base,
            "observed_delta": float(post - baseline),
            "phase8_label_status": "ceiling_exclude",
        }

    expected_delta = PROPORTIONAL_EXPECTED_FRACTION * (MAX_FMA_UE - baseline)
    observed_delta = post - baseline
    residual = expected_delta - observed_delta
    label = pd.NA
    label_int: int | pd._libs.missing.NAType = pd.NA
    if median_residual is not None and math.isfinite(float(median_residual)):
        label = PROPORTIONAL_RECOVERY if residual <= float(median_residual) else POOR_RECOVERY
        label_int = LABEL_TO_INT_PROP_RESIDUAL[str(label)]
    return {
        **base,
        "expected_delta": float(expected_delta),
        "observed_delta": float(observed_delta),
        "residual": float(residual),
        "primary_label_prop_residual": label,
        "primary_label_int_prop_residual": label_int,
        "absolute_70_achieved": "ProportionalRecoveryAchieved" if residual <= 0 else "NotAchieved",
    }


def build_phase8_label_table(
    cohort: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    baseline_col: str = "baseline_fma",
    post_col: str = "post_fma",
    current_label_col: str = "label_primary",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = {subject_col, baseline_col, post_col}
    missing = sorted(required - set(cohort.columns))
    if missing:
        raise ValueError(f"Missing required label columns: {missing}")

    preliminary = [
        compute_proportional_recovery_record(
            row[subject_col],
            row[baseline_col],
            row[post_col],
            median_residual=None,
        )
        for _, row in cohort.iterrows()
    ]
    preliminary_frame = pd.DataFrame(preliminary)
    analyzable_residuals = preliminary_frame.loc[
        preliminary_frame["phase8_label_status"].eq("analyzable"), "residual"
    ].astype(float)
    median_residual = float(analyzable_residuals.median()) if not analyzable_residuals.empty else np.nan

    records = [
        compute_proportional_recovery_record(
            row[subject_col],
            row[baseline_col],
            row[post_col],
            median_residual=median_residual,
        )
        for _, row in cohort.iterrows()
    ]
    labels = pd.DataFrame(records)
    if current_label_col in cohort.columns:
        labels["current_clinically_meaningful"] = cohort[current_label_col].astype("string").to_numpy()
    else:
        labels["current_clinically_meaningful"] = pd.NA
    labels["clear_residual_tertile"] = _clear_residual_tertiles(labels)

    analyzable = labels.loc[labels["phase8_label_status"].eq("analyzable")].copy()
    proportional_count = int(analyzable["primary_label_prop_residual"].eq(PROPORTIONAL_RECOVERY).sum())
    poor_count = int(analyzable["primary_label_prop_residual"].eq(POOR_RECOVERY).sum())
    audit = {
        "n_total": int(len(labels)),
        "n_analyzable": int(len(analyzable)),
        "n_ceiling_exclude": int(labels["phase8_label_status"].eq("ceiling_exclude").sum()),
        "n_missing_excluded": int(labels["phase8_label_status"].eq("excluded_missing").sum()),
        "median_residual": median_residual,
        "n_proportional_recovery": proportional_count,
        "n_poor_recovery": poor_count,
    }
    return labels, audit


def label_with_train_median_threshold(
    label_table: pd.DataFrame,
    *,
    train_subjects: Iterable[str],
    test_subject: str,
) -> dict[str, Any]:
    required = {"subject_id", "residual", "phase8_label_status"}
    missing = sorted(required - set(label_table.columns))
    if missing:
        raise ValueError(f"Missing required train-median columns: {missing}")

    train_set = {str(subject) for subject in train_subjects}
    subjects = label_table["subject_id"].astype(str)
    train = label_table.loc[
        subjects.isin(train_set) & label_table["phase8_label_status"].eq("analyzable")
    ]
    if train.empty:
        raise ValueError("Train median threshold requires at least one analyzable training patient")
    train_median = float(train["residual"].astype(float).median())
    test_rows = label_table.loc[subjects.eq(str(test_subject))]
    if len(test_rows) != 1:
        raise ValueError(f"Expected one test subject row for {test_subject}, found {len(test_rows)}")
    test = test_rows.iloc[0]
    if str(test["phase8_label_status"]) != "analyzable":
        label: str | pd._libs.missing.NAType = pd.NA
        label_int: int | pd._libs.missing.NAType = pd.NA
    else:
        residual = float(test["residual"])
        label = PROPORTIONAL_RECOVERY if residual <= train_median else POOR_RECOVERY
        label_int = LABEL_TO_INT_PROP_RESIDUAL[label]
    return {
        "test_subject": str(test_subject),
        "train_median_residual": train_median,
        "primary_label_train_median_threshold": label,
        "primary_label_int_train_median_threshold": label_int,
    }


def _clear_residual_tertiles(labels: pd.DataFrame) -> pd.Series:
    result = pd.Series(pd.NA, index=labels.index, dtype="object")
    analyzable_mask = labels["phase8_label_status"].eq("analyzable")
    residuals = labels.loc[analyzable_mask, "residual"].astype(float)
    if residuals.empty:
        return result
    low = float(residuals.quantile(1 / 3))
    high = float(residuals.quantile(2 / 3))
    result.loc[analyzable_mask & labels["residual"].astype(float).le(low)] = "ClearProportionalRecovery"
    result.loc[analyzable_mask & labels["residual"].astype(float).ge(high)] = "ClearPoorRecovery"
    result.loc[analyzable_mask & result.isna()] = "middle_exclude"
    return result


def _valid_fma_score(value: float) -> bool:
    return 0.0 <= value <= MAX_FMA_UE
