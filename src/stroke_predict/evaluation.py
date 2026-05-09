from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score


REQUIRED_PREDICTION_COLUMNS = {
    "model_id",
    "outer_fold",
    "subject_id",
    "label_true",
    "y_true",
    "prob_good",
    "pred_label",
    "threshold",
}

METRIC_NAMES = (
    "roc_auc",
    "balanced_accuracy",
    "sensitivity",
    "specificity",
    "pr_auc",
    "brier_score",
)


def validate_patient_predictions(predictions: pd.DataFrame, expected_subject_count: int | None = None) -> None:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    duplicate_mask = predictions.duplicated(["model_id", "subject_id"], keep=False)
    if duplicate_mask.any():
        duplicates = predictions.loc[duplicate_mask, ["model_id", "subject_id"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicate model-subject predictions: {duplicates}")
    if expected_subject_count is not None:
        counts = predictions.groupby("model_id")["subject_id"].nunique()
        bad = counts[counts.ne(expected_subject_count)]
        if not bad.empty:
            raise ValueError(f"Unexpected subject counts per model: {bad.to_dict()}")
    bad_y = sorted(set(predictions["y_true"].dropna().astype(int)) - {0, 1})
    if bad_y:
        raise ValueError(f"y_true must be binary 0/1, found: {bad_y}")


def compute_classification_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    validate_patient_predictions(predictions)
    rows: list[dict[str, object]] = []
    for model_id, group in predictions.groupby("model_id", sort=True):
        values = _extract_arrays(group)
        row: dict[str, object] = {
            "model_id": model_id,
            "n_subjects": int(len(group)),
        }
        for metric in METRIC_NAMES:
            row[metric] = _metric_value(metric, *values)
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_metric_ci(
    predictions: pd.DataFrame,
    *,
    n_bootstrap: int,
    random_seed: int,
    metrics: Iterable[str] = METRIC_NAMES,
) -> pd.DataFrame:
    validate_patient_predictions(predictions)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    metric_names = tuple(metrics)
    for model_id, group in predictions.groupby("model_id", sort=True):
        group = group.reset_index(drop=True)
        observed = {metric: _metric_value(metric, *_extract_arrays(group)) for metric in metric_names}
        samples: dict[str, list[float]] = {metric: [] for metric in metric_names}
        for _ in range(n_bootstrap):
            indices = rng.integers(0, len(group), len(group))
            sample = group.iloc[indices].reset_index(drop=True)
            values = _extract_arrays(sample)
            for metric in metric_names:
                samples[metric].append(_metric_value(metric, *values))
        for metric in metric_names:
            sample_values = np.asarray(samples[metric], dtype=float)
            valid = sample_values[~np.isnan(sample_values)]
            if valid.size:
                ci_lower = float(np.percentile(valid, 2.5))
                ci_upper = float(np.percentile(valid, 97.5))
            else:
                ci_lower = np.nan
                ci_upper = np.nan
            rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "observed_value": observed[metric],
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_bootstrap": int(n_bootstrap),
                    "random_seed": int(random_seed),
                }
            )
    return pd.DataFrame(rows)


def permutation_test(
    predictions: pd.DataFrame,
    *,
    n_permutations: int,
    random_seed: int,
    metrics: Iterable[str] = ("roc_auc",),
) -> pd.DataFrame:
    validate_patient_predictions(predictions)
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    metric_names = tuple(metrics)
    for model_id, group in predictions.groupby("model_id", sort=True):
        group = group.reset_index(drop=True)
        y_true, prob_good, y_pred = _extract_arrays(group)
        observed = {metric: _metric_value(metric, y_true, prob_good, y_pred) for metric in metric_names}
        nulls: dict[str, list[float]] = {metric: [] for metric in metric_names}
        for _ in range(n_permutations):
            permuted = rng.permutation(y_true)
            for metric in metric_names:
                nulls[metric].append(_metric_value(metric, permuted, prob_good, y_pred))
        for metric in metric_names:
            null_values = np.asarray(nulls[metric], dtype=float)
            valid = null_values[~np.isnan(null_values)]
            observed_value = observed[metric]
            if valid.size and not np.isnan(observed_value):
                p_value = float((np.sum(valid >= observed_value) + 1) / (valid.size + 1))
                null_mean = float(np.mean(valid))
                null_std = float(np.std(valid, ddof=0))
            else:
                p_value = np.nan
                null_mean = np.nan
                null_std = np.nan
            rows.append(
                {
                    "model_id": model_id,
                    "metric": metric,
                    "observed_value": observed_value,
                    "null_mean": null_mean,
                    "null_std": null_std,
                    "p_value": p_value,
                    "n_permutations": int(n_permutations),
                    "random_seed": int(random_seed),
                }
            )
    return pd.DataFrame(rows)


def _extract_arrays(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_true = group["y_true"].astype(int).to_numpy()
    prob_good = group["prob_good"].astype(float).to_numpy()
    y_pred = (group["pred_label"].astype(str) == "Good").astype(int).to_numpy()
    return y_true, prob_good, y_pred


def _metric_value(metric: str, y_true: np.ndarray, prob_good: np.ndarray, y_pred: np.ndarray) -> float:
    if metric == "roc_auc":
        if len(set(y_true.tolist())) < 2:
            return np.nan
        return _safe_metric(roc_auc_score, y_true, prob_good)
    if metric == "balanced_accuracy":
        sensitivity = _metric_value("sensitivity", y_true, prob_good, y_pred)
        specificity = _metric_value("specificity", y_true, prob_good, y_pred)
        if np.isnan(sensitivity) or np.isnan(specificity):
            return np.nan
        return float((sensitivity + specificity) / 2)
    if metric == "sensitivity":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return float(tp / (tp + fn)) if tp + fn else np.nan
    if metric == "specificity":
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        return float(tn / (tn + fp)) if tn + fp else np.nan
    if metric == "pr_auc":
        if not np.any(y_true == 1):
            return np.nan
        return _safe_metric(average_precision_score, y_true, prob_good)
    if metric == "brier_score":
        return _safe_metric(brier_score_loss, y_true, prob_good)
    raise ValueError(f"Unsupported metric: {metric}")


def _safe_metric(func: Callable[[np.ndarray, np.ndarray], float], y_true: np.ndarray, values: np.ndarray) -> float:
    try:
        result = float(func(y_true, values))
    except ValueError:
        return np.nan
    return result if np.isfinite(result) else np.nan
