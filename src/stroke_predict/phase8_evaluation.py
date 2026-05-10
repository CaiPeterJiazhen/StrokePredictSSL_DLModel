from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


PHASE8_REQUIRED_PREDICTION_COLUMNS = {
    "model_id",
    "outer_fold",
    "patient_id",
    "true_label",
    "y_true",
    "predicted_score",
    "predicted_label",
    "threshold",
    "prediction_unit",
}
PHASE8_NO_LEAKAGE_COLUMNS = {
    "model_id",
    "outer_fold",
    "test_subject",
    "outer_test_in_fit_subjects",
    "outer_test_in_transform_fit_subjects",
    "outer_test_in_inner_cv_subjects",
}


def validate_phase8_patient_predictions(
    predictions: pd.DataFrame,
    *,
    expected_patient_count: int | None = None,
) -> None:
    missing = sorted(PHASE8_REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing Phase 8 prediction columns: {missing}")
    if not predictions["prediction_unit"].eq("patient").all():
        raise ValueError("Phase 8 predictions must be patient-level")
    duplicates = predictions.duplicated(["model_id", "patient_id"], keep=False)
    if duplicates.any():
        duplicate_rows = predictions.loc[duplicates, ["model_id", "patient_id"]].drop_duplicates()
        raise ValueError(f"Duplicate Phase 8 model-patient predictions: {duplicate_rows.to_dict('records')}")
    bad_labels = sorted(set(predictions["y_true"].dropna().astype(int)) - {0, 1})
    if bad_labels:
        raise ValueError(f"Phase 8 y_true must be 0/1, found: {bad_labels}")
    if expected_patient_count is not None:
        counts = predictions.groupby("model_id")["patient_id"].nunique()
        bad = counts[counts.ne(expected_patient_count)]
        if not bad.empty:
            raise ValueError(f"Unexpected Phase 8 patient counts per model: {bad.to_dict()}")


def validate_phase8_no_leakage(audit: pd.DataFrame) -> None:
    missing = sorted(PHASE8_NO_LEAKAGE_COLUMNS - set(audit.columns))
    if missing:
        raise ValueError(f"Missing Phase 8 no-leakage columns: {missing}")
    leak_columns = [
        "outer_test_in_fit_subjects",
        "outer_test_in_transform_fit_subjects",
        "outer_test_in_inner_cv_subjects",
    ]
    leaked = audit.loc[audit[leak_columns].any(axis=1), ["model_id", "outer_fold", *leak_columns]]
    if not leaked.empty:
        raise ValueError(f"Phase 8 no-leakage audit failed: {leaked.to_dict('records')}")


def compute_phase8_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    validate_phase8_patient_predictions(predictions)
    rows: list[dict[str, object]] = []
    for model_id, group in predictions.groupby("model_id", sort=True):
        rows.append(_metric_row_for_group(str(model_id), group))
    return pd.DataFrame(rows)


def bootstrap_phase8_ci(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
    metrics: Iterable[str] = ("roc_auc",),
) -> pd.DataFrame:
    validate_phase8_patient_predictions(predictions)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for model_id, group in predictions.groupby("model_id", sort=True):
        group = group.reset_index(drop=True)
        observed = compute_phase8_metrics(group).iloc[0].to_dict()
        for metric in metrics:
            samples = []
            for _ in range(n_bootstrap):
                sample = group.iloc[rng.integers(0, len(group), len(group))]
                samples.append(float(_metric_row_for_group(str(model_id), sample)[metric]))
            values = np.asarray(samples, dtype=float)
            values = values[np.isfinite(values)]
            rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "observed_value": float(observed[metric]),
                    "ci_lower": float(np.percentile(values, 2.5)) if values.size else np.nan,
                    "ci_upper": float(np.percentile(values, 97.5)) if values.size else np.nan,
                    "n_bootstrap": int(n_bootstrap),
                    "random_seed": int(random_seed),
                }
            )
    return pd.DataFrame(rows)


def permutation_phase8_test(
    predictions: pd.DataFrame,
    *,
    n_permutations: int,
    random_seed: int,
    metric: str = "roc_auc",
) -> pd.DataFrame:
    validate_phase8_patient_predictions(predictions)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    for model_id, group in predictions.groupby("model_id", sort=True):
        group = group.reset_index(drop=True)
        observed = float(compute_phase8_metrics(group).iloc[0][metric])
        null_values = []
        for _ in range(n_permutations):
            permuted = group.copy()
            permuted["y_true"] = rng.permutation(permuted["y_true"].astype(int).to_numpy())
            null_values.append(float(_metric_row_for_group(str(model_id), permuted)[metric]))
        null = np.asarray(null_values, dtype=float)
        null = null[np.isfinite(null)]
        if null.size and np.isfinite(observed):
            p_value = float((np.sum(null >= observed) + 1) / (len(null) + 1))
        else:
            p_value = np.nan
        rows.append(
            {
                "model_id": model_id,
                "metric": metric,
                "observed_value": observed,
                "null_mean": float(np.mean(null)) if null.size else np.nan,
                "null_std": float(np.std(null, ddof=0)) if null.size else np.nan,
                "p_value": p_value,
                "n_permutations": int(n_permutations),
                "random_seed": int(random_seed),
            }
        )
    return pd.DataFrame(rows)


def _metric_row_for_group(model_id: str, group: pd.DataFrame) -> dict[str, object]:
    y_true = group["y_true"].astype(int).to_numpy()
    score = group["predicted_score"].astype(float).to_numpy()
    y_pred = (group["predicted_label"].astype(str) == "ProportionalRecovery").astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else np.nan
    specificity = float(tn / (tn + fp)) if tn + fp else np.nan
    return {
        "model_id": model_id,
        "n_patients": int(len(group)),
        "roc_auc": _safe_auc(y_true, score),
        "pr_auc": _safe_pr_auc(y_true, score),
        "balanced_accuracy": _mean_if_finite(sensitivity, specificity),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": _safe_f1(y_true, y_pred),
        "brier_score": _safe_brier(y_true, score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "mean_score_proportional": _class_mean(score, y_true, 1),
        "mean_score_poor": _class_mean(score, y_true, 0),
        "auc_score": _safe_auc(y_true, score),
        "auc_one_minus_score": _safe_auc(y_true, 1.0 - score),
    }


def _safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(set(y_true.tolist())) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, score))
    except ValueError:
        return np.nan


def _safe_pr_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if not np.any(y_true == 1):
        return np.nan
    try:
        return float(average_precision_score(y_true, score))
    except ValueError:
        return np.nan


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(f1_score(y_true, y_pred, zero_division=0))
    except ValueError:
        return np.nan


def _safe_brier(y_true: np.ndarray, score: np.ndarray) -> float:
    try:
        return float(brier_score_loss(y_true, np.clip(score, 0.0, 1.0)))
    except ValueError:
        return np.nan


def _mean_if_finite(left: float, right: float) -> float:
    if np.isfinite(left) and np.isfinite(right):
        return float((left + right) / 2)
    return np.nan


def _class_mean(score: np.ndarray, y_true: np.ndarray, label: int) -> float:
    values = score[y_true == label]
    return float(np.mean(values)) if values.size else np.nan
