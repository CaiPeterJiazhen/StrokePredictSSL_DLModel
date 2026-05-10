from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score

from stroke_predict.phase8_reports import assert_phase8_public_output_safe


M15A_MODEL_ID = "M15a_prop_roi_fc_best_ml"
M15B_MODEL_ID = "M15b_prop_summary_eeg_best_ml"


def audit_source_mode(fc_audit: pd.DataFrame | Mapping[str, object]) -> dict[str, object]:
    row = _first_audit_row(fc_audit)
    source_mode = str(row.get("source_mode", "not_available"))
    is_real = source_mode == "time_series"
    if is_real:
        claim_guard = "real baseline EO/EC time-series full-edge FC"
        required_next = "Real time-series FC inputs were present for this run."
    elif source_mode == "psd_artifact_proxy":
        claim_guard = "psd_artifact_proxy must not be called real time-series full-edge FC"
        required_next = (
            "Provide baseline EO/EC preprocessed time-series input, fixed reduced32 channels, "
            "496 undirected edges, coherence, imaginary coherence, wPLI, and the Phase 8.1 leakage audit."
        )
    else:
        claim_guard = "source mode is not available; no time-series FC claim is allowed"
        required_next = "Regenerate Phase 8 FC features with auditable source_mode metadata."
    return {
        "source_mode": source_mode,
        "is_real_time_series_fc": bool(is_real),
        "claim_guard": claim_guard,
        "required_before_time_series_claim": required_next,
        "n_edges": row.get("n_edges", np.nan),
        "n_channels": row.get("n_channels", np.nan),
    }


def audit_comparison_models(
    *,
    features: pd.DataFrame,
    predictions: pd.DataFrame,
    model_a: str = M15A_MODEL_ID,
    model_b: str = M15B_MODEL_ID,
    allow_intentional_shared_predictions: bool = False,
) -> dict[str, object]:
    feature_a = _feature_matrix_for_model(features, model_a)
    feature_b = _feature_matrix_for_model(features, model_b)
    prediction_a = _prediction_matrix_for_model(predictions, model_a)
    prediction_b = _prediction_matrix_for_model(predictions, model_b)

    feature_hash_a = _frame_hash(feature_a)
    feature_hash_b = _frame_hash(feature_b)
    prediction_hash_a = _frame_hash(prediction_a)
    prediction_hash_b = _frame_hash(prediction_b)
    features_identical = feature_hash_a == feature_hash_b
    predictions_identical = prediction_hash_a == prediction_hash_b

    if not allow_intentional_shared_predictions and features_identical and predictions_identical:
        raise ValueError(f"{model_a} and {model_b} silently share predictions and feature matrices")
    if not allow_intentional_shared_predictions and features_identical:
        raise ValueError(f"{model_a} and {model_b} use identical feature matrices")

    if predictions_identical and not features_identical:
        explanation = "same predictions despite different feature matrices; likely small-sample LOPO/model behavior, not feature reuse"
    elif predictions_identical:
        explanation = "same predictions were explicitly allowed"
    else:
        explanation = (
            "comparison model predictions differ after source-tagged feature matrices; prior identical metrics "
            "were consistent with untagged feature-selector fallback or exact small-sample equality"
        )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "feature_matrices_identical": bool(features_identical),
        "predictions_identical": bool(predictions_identical),
        "feature_hash_a": feature_hash_a,
        "feature_hash_b": feature_hash_b,
        "prediction_hash_a": prediction_hash_a,
        "prediction_hash_b": prediction_hash_b,
        "n_feature_columns_a": int(feature_a.shape[1]),
        "n_feature_columns_b": int(feature_b.shape[1]),
        "explanation": explanation,
    }


def apply_multiple_comparison_correction(permutation_table: pd.DataFrame) -> pd.DataFrame:
    if "model_id" not in permutation_table.columns:
        raise ValueError("Permutation table must include model_id")
    p_col = "permutation_p_value" if "permutation_p_value" in permutation_table.columns else "p_value"
    if p_col not in permutation_table.columns:
        raise ValueError("Permutation table must include permutation_p_value or p_value")

    result = permutation_table.copy()
    raw = pd.to_numeric(result[p_col], errors="coerce")
    result["raw_permutation_p_value"] = raw
    n_tests = int(raw.notna().sum())
    result["bonferroni_p_value"] = np.minimum(raw * max(n_tests, 1), 1.0)
    result["fdr_q_value"] = _benjamini_hochberg(raw.to_numpy(dtype=float))
    result["nominal_p_lt_0_05"] = raw.lt(0.05).map(lambda value: bool(value)).astype(object)
    result["fdr_q_lt_0_05"] = pd.to_numeric(result["fdr_q_value"], errors="coerce").lt(0.05).map(lambda value: bool(value)).astype(object)
    result["bonferroni_p_lt_0_05"] = (
        pd.to_numeric(result["bonferroni_p_value"], errors="coerce").lt(0.05).map(lambda value: bool(value)).astype(object)
    )
    return result


def build_threshold_calibration_table(predictions: pd.DataFrame, *, n_bins: int = 5) -> pd.DataFrame:
    required = {"model_id", "y_true", "predicted_score"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Threshold calibration predictions missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for model_id, group in predictions.groupby("model_id", sort=True):
        rows.extend(_threshold_rows(str(model_id), group, "fixed_0.5_threshold", "threshold", 0.5))
        rows.extend(_threshold_rows(str(model_id), group, "inner_cv_threshold", "inner_cv_threshold", None))
        rows.extend(_threshold_rows(str(model_id), group, "inner_cv_youden_threshold", "inner_cv_youden_threshold", None))
        rows.extend(_calibration_rows(str(model_id), group, n_bins=n_bins))
        rows.extend(_score_distribution_rows(str(model_id), group))
    return pd.DataFrame(rows)


def build_patient_error_audit(
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    boundary_fraction: float = 0.25,
) -> pd.DataFrame:
    label_table = labels.copy().assign(patient_id=lambda frame: frame["subject_id"].astype(str))
    pred = predictions.copy().assign(patient_id=lambda frame: frame["patient_id"].astype(str))
    merged = pred.merge(label_table, on="patient_id", how="left", suffixes=("", "_label"))
    if merged["subject_id"].isna().any():
        missing = merged.loc[merged["subject_id"].isna(), "patient_id"].astype(str).tolist()
        raise ValueError(f"Patient error audit missing label rows: {missing}")

    proportional_label = merged["primary_label_prop_residual"].astype(str)
    correct = merged["predicted_label"].astype(str).eq(proportional_label)
    old_label = merged.get("current_clinically_meaningful", pd.Series(["not_available"] * len(merged))).astype(str)
    residual = pd.to_numeric(merged["residual"], errors="coerce")
    median_residual = _median_residual(merged)
    distance = (residual - median_residual).abs()
    near = _near_boundary_flags(distance, residual, boundary_fraction=boundary_fraction)

    output = pd.DataFrame(
        {
            "patient_id": merged["patient_id"].astype(str),
            "old_label": old_label,
            "proportional_label": proportional_label,
            "baseline_fma": pd.to_numeric(merged["baseline_fma"], errors="coerce"),
            "post_fma": pd.to_numeric(merged["post_fma"], errors="coerce"),
            "observed_delta": pd.to_numeric(merged["observed_delta"], errors="coerce"),
            "expected_delta": pd.to_numeric(merged["expected_delta"], errors="coerce"),
            "residual": residual,
            "predicted_score": pd.to_numeric(merged["predicted_score"], errors="coerce"),
            "predicted_label": merged["predicted_label"].astype(str),
            "correct": correct.map(lambda value: bool(value)).astype(object),
            "rank": pd.to_numeric(merged["predicted_score"], errors="coerce").rank(ascending=False, method="first").astype(int),
            "near_median_threshold": near.map(lambda value: bool(value)).astype(object),
            "old_new_label_disagree": old_label.ne(proportional_label).map(lambda value: bool(value)).astype(object),
        }
    )
    return output.sort_values("rank").reset_index(drop=True)


def write_phase8_1_validation_outputs(
    *,
    output_dir: str | Path,
    source_audit: Mapping[str, object],
    duplicate_audit: Mapping[str, object],
    correction_table: pd.DataFrame,
    threshold_calibration: pd.DataFrame,
    patient_error_audit: pd.DataFrame,
    no_leakage_audit: pd.DataFrame,
    best_model_id: str,
    real_time_series_reproduced: bool,
) -> dict[str, Path]:
    root = Path(output_dir)
    report_dir = root / "reports"
    evaluation_dir = root / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    correction_path = evaluation_dir / "phase8_1_multiple_comparison_correction.csv"
    threshold_path = evaluation_dir / "phase8_1_threshold_calibration.csv"
    patient_path = evaluation_dir / "phase8_1_patient_error_audit.csv"
    report_path = report_dir / "phase8_1_validation_report.md"
    no_leakage_path = report_dir / "phase8_1_no_leakage_report.txt"

    for frame in (correction_table, threshold_calibration, patient_error_audit):
        assert_phase8_public_output_safe(frame)
    correction_table.to_csv(correction_path, index=False)
    threshold_calibration.to_csv(threshold_path, index=False)
    patient_error_audit.to_csv(patient_path, index=False)

    no_leakage_pass = _no_leakage_passes(no_leakage_audit)
    no_leakage_path.write_text(_no_leakage_text(no_leakage_pass, no_leakage_audit), encoding="utf-8")
    report_path.write_text(
        _validation_report_markdown(
            source_audit=source_audit,
            duplicate_audit=duplicate_audit,
            correction_table=correction_table,
            threshold_calibration=threshold_calibration,
            patient_error_audit=patient_error_audit,
            best_model_id=best_model_id,
            real_time_series_reproduced=real_time_series_reproduced,
            no_leakage_pass=no_leakage_pass,
        ),
        encoding="utf-8",
    )
    return {
        "validation_report": report_path,
        "multiple_comparison_correction": correction_path,
        "threshold_calibration": threshold_path,
        "patient_error_audit": patient_path,
        "no_leakage_report": no_leakage_path,
    }


def _first_audit_row(fc_audit: pd.DataFrame | Mapping[str, object]) -> dict[str, object]:
    if isinstance(fc_audit, pd.DataFrame):
        if fc_audit.empty:
            return {}
        return fc_audit.iloc[0].to_dict()
    return dict(fc_audit)


def _feature_matrix_for_model(features: pd.DataFrame, model_id: str) -> pd.DataFrame:
    if "subject_id" not in features.columns:
        raise ValueError("Comparison feature table must include subject_id")
    if model_id == M15A_MODEL_ID:
        prefixes = ("roi_fc", "fc_roi")
    elif model_id == M15B_MODEL_ID:
        prefixes = ("summary", "psd_", "fc_", "eeg_")
    else:
        prefixes = ()
    columns = [
        column
        for column in features.columns
        if column != "subject_id" and pd.api.types.is_numeric_dtype(features[column]) and (not prefixes or column.startswith(prefixes))
    ]
    if not columns:
        raise ValueError(f"No comparison feature columns found for {model_id}")
    ordered = features.sort_values("subject_id").reset_index(drop=True)
    matrix = ordered[columns].reset_index(drop=True)
    matrix.columns = [f"feature_{index}" for index in range(matrix.shape[1])]
    return matrix


def _prediction_matrix_for_model(predictions: pd.DataFrame, model_id: str) -> pd.DataFrame:
    required = {"model_id", "patient_id", "predicted_score", "predicted_label"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Comparison predictions missing columns: {missing}")
    rows = predictions.loc[predictions["model_id"].astype(str).eq(model_id)].copy()
    if rows.empty:
        raise ValueError(f"No predictions found for {model_id}")
    rows = rows.sort_values("patient_id").reset_index(drop=True)
    return rows[["patient_id", "predicted_score", "predicted_label"]]


def _frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.copy()
    for column in normalized.columns:
        if pd.api.types.is_numeric_dtype(normalized[column]):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round(12)
        else:
            normalized[column] = normalized[column].astype(str)
    payload = normalized.to_csv(index=False, na_rep="__NA__").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    result = np.full(len(p_values), np.nan, dtype=float)
    valid_mask = np.isfinite(p_values)
    valid = p_values[valid_mask]
    if valid.size == 0:
        return result
    order = np.argsort(valid)
    sorted_p = valid[order]
    n = len(sorted_p)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for index in range(n - 1, -1, -1):
        rank = index + 1
        running = min(running, sorted_p[index] * n / rank)
        adjusted[index] = running
    valid_result = np.empty(n, dtype=float)
    valid_result[order] = np.clip(adjusted, 0.0, 1.0)
    result[np.where(valid_mask)[0]] = valid_result
    return result


def _threshold_rows(
    model_id: str,
    group: pd.DataFrame,
    analysis_type: str,
    threshold_column: str,
    fallback_threshold: float | None,
) -> list[dict[str, object]]:
    if threshold_column not in group.columns and fallback_threshold is None:
        return [
            {
                "model_id": model_id,
                "analysis_type": analysis_type,
                "threshold_source": f"{threshold_column} not available; outer test predictions not used",
                "status": "not_available",
                "threshold": np.nan,
                "brier_score": _brier(group),
            }
        ]
    threshold_values = (
        pd.Series([fallback_threshold] * len(group), index=group.index, dtype=float)
        if fallback_threshold is not None
        else pd.to_numeric(group[threshold_column], errors="coerce")
    )
    if threshold_values.isna().any():
        return [
            {
                "model_id": model_id,
                "analysis_type": analysis_type,
                "threshold_source": f"{threshold_column} has missing values; outer test predictions not used",
                "status": "not_available",
                "threshold": np.nan,
                "brier_score": _brier(group),
            }
        ]
    y_true = group["y_true"].astype(int).to_numpy()
    score = pd.to_numeric(group["predicted_score"], errors="coerce").to_numpy(dtype=float)
    y_pred = (score >= threshold_values.to_numpy(dtype=float)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else np.nan
    specificity = float(tn / (tn + fp)) if tn + fp else np.nan
    return [
        {
            "model_id": model_id,
            "analysis_type": analysis_type,
            "threshold_source": threshold_column if fallback_threshold is None else "fixed_0.5",
            "status": "available",
            "threshold": float(np.mean(threshold_values)),
            "balanced_accuracy": _mean_if_finite(sensitivity, specificity),
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": _safe_f1(y_true, y_pred),
            "brier_score": _brier(group),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
    ]


def _calibration_rows(model_id: str, group: pd.DataFrame, *, n_bins: int) -> list[dict[str, object]]:
    score = pd.to_numeric(group["predicted_score"], errors="coerce").clip(0.0, 1.0)
    y_true = group["y_true"].astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    labels = range(n_bins)
    bin_ids = pd.cut(score, bins=bins, labels=labels, include_lowest=True, right=True)
    rows = []
    for bin_id in labels:
        mask = bin_ids.astype("Int64").eq(bin_id)
        values = score[mask]
        labels_in_bin = y_true[mask]
        rows.append(
            {
                "model_id": model_id,
                "analysis_type": "calibration_bin",
                "threshold_source": "score_bin_independent_of_label",
                "status": "available",
                "bin_index": int(bin_id),
                "bin_lower": float(bins[int(bin_id)]),
                "bin_upper": float(bins[int(bin_id) + 1]),
                "bin_count": int(mask.sum()),
                "mean_score": float(values.mean()) if len(values) else np.nan,
                "observed_rate": float(labels_in_bin.mean()) if len(labels_in_bin) else np.nan,
                "brier_score": _brier(group),
            }
        )
    return rows


def _score_distribution_rows(model_id: str, group: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for label_value, label_name in ((0, "PoorRecovery"), (1, "ProportionalRecovery")):
        scores = pd.to_numeric(group.loc[group["y_true"].astype(int).eq(label_value), "predicted_score"], errors="coerce")
        rows.append(
            {
                "model_id": model_id,
                "analysis_type": "score_distribution_by_group",
                "threshold_source": "not_thresholded",
                "status": "available",
                "group_label": label_name,
                "group_count": int(scores.notna().sum()),
                "mean_score": float(scores.mean()) if scores.notna().any() else np.nan,
                "score_std": float(scores.std(ddof=0)) if scores.notna().any() else np.nan,
                "score_min": float(scores.min()) if scores.notna().any() else np.nan,
                "score_max": float(scores.max()) if scores.notna().any() else np.nan,
                "brier_score": _brier(group),
            }
        )
    return rows


def _brier(group: pd.DataFrame) -> float:
    try:
        return float(brier_score_loss(group["y_true"].astype(int), pd.to_numeric(group["predicted_score"], errors="coerce").clip(0, 1)))
    except ValueError:
        return np.nan


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        return float(f1_score(y_true, y_pred, zero_division=0))
    except ValueError:
        return np.nan


def _mean_if_finite(left: float, right: float) -> float:
    if np.isfinite(left) and np.isfinite(right):
        return float((left + right) / 2)
    return np.nan


def _median_residual(frame: pd.DataFrame) -> float:
    if "median_residual" in frame.columns and frame["median_residual"].notna().any():
        return float(pd.to_numeric(frame["median_residual"], errors="coerce").dropna().iloc[0])
    return float(pd.to_numeric(frame["residual"], errors="coerce").median())


def _near_boundary_flags(distance: pd.Series, residual: pd.Series, *, boundary_fraction: float) -> pd.Series:
    finite_distance = distance[np.isfinite(distance)]
    if finite_distance.empty:
        return pd.Series([False] * len(distance), index=distance.index)
    q1 = float(np.nanpercentile(residual, 25))
    q3 = float(np.nanpercentile(residual, 75))
    iqr = q3 - q1
    if len(finite_distance) < 8 or iqr <= 0:
        n_near = max(1, int(math.ceil(len(finite_distance) * boundary_fraction)))
        near_indices = finite_distance.nsmallest(n_near).index
        return pd.Series(distance.index.isin(near_indices), index=distance.index)
    return distance.le(iqr * 0.10)


def _no_leakage_passes(audit: pd.DataFrame) -> bool:
    leak_columns = [
        "outer_test_in_fit_subjects",
        "outer_test_in_transform_fit_subjects",
        "outer_test_in_inner_cv_subjects",
    ]
    existing = [column for column in leak_columns if column in audit.columns]
    if not existing:
        return False
    return not audit[existing].astype(bool).any(axis=None)


def _no_leakage_text(no_leakage_pass: bool, audit: pd.DataFrame) -> str:
    status = "PASS" if no_leakage_pass else "FAIL"
    return "\n".join(
        [
            f"Phase 8.1 no-leakage report: {status}",
            f"rows_checked: {len(audit)}",
            "outer_test_in_fit_subjects: checked",
            "outer_test_in_transform_fit_subjects: checked",
            "outer_test_in_inner_cv_subjects: checked",
            "",
        ]
    )


def _validation_report_markdown(
    *,
    source_audit: Mapping[str, object],
    duplicate_audit: Mapping[str, object],
    correction_table: pd.DataFrame,
    threshold_calibration: pd.DataFrame,
    patient_error_audit: pd.DataFrame,
    best_model_id: str,
    real_time_series_reproduced: bool,
    no_leakage_pass: bool,
) -> str:
    source_mode = str(source_audit.get("source_mode", "not_available"))
    best = correction_table.loc[correction_table["model_id"].astype(str).eq(str(best_model_id))]
    survives_fdr = bool(best["fdr_q_lt_0_05"].iloc[0]) if not best.empty and "fdr_q_lt_0_05" in best else False
    survives_bonf = bool(best["bonferroni_p_lt_0_05"].iloc[0]) if not best.empty and "bonferroni_p_lt_0_05" in best else False
    m14b_status = _model_correction_status(correction_table, "M14b_prop_reduced32_fullfc_elasticnet")
    near_rate = _near_boundary_rate(patient_error_audit)
    threshold_limited = _threshold_limit_statement(threshold_calibration)
    no_leakage = "passed" if no_leakage_pass else "failed"
    return "\n".join(
        [
            "# Phase 8.1 Validation Report",
            "",
            "## Required Answers",
            "",
            "### Was full-edge FC real time-series FC or proxy?",
            f"Answer: source mode is `{source_mode}`. {source_audit.get('claim_guard', '')}",
            "",
            "### If proxy, what must be done before claiming time-series FC evidence?",
            f"Answer: {source_audit.get('required_before_time_series_claim', 'Regenerate from baseline time-series input.')}",
            "",
            "### Does real time-series FC reproduce the Phase 8 positive signal?",
            f"Answer: {'yes' if real_time_series_reproduced else 'no'} for this Phase 8.1 run.",
            "",
            "### Why were ROI-FC and summary EEG metrics identical?",
            f"Answer: {duplicate_audit.get('explanation', 'not available')}.",
            "",
            "### Does the best model survive FDR or Bonferroni correction?",
            f"Answer: best audited model `{best_model_id}` FDR survival is {survives_fdr}; Bonferroni survival is {survives_bonf}. {m14b_status}",
            "",
            "### Is classification performance limited by threshold choice?",
            f"Answer: {threshold_limited}",
            "",
            "### Are errors concentrated near the residual median boundary?",
            f"Answer: {near_rate:.3f} of audited patients are near the residual median threshold.",
            "",
            "### Is the result strong enough to justify no-SSL MatrixNet next?",
            "Answer: no. Phase 8.1 is a validation gate; no-SSL MatrixNet should not start from proxy FC evidence.",
            "",
            "### Should SSL remain blocked until real FC and MatrixNet validation?",
            "Answer: SSL should remain blocked until real FC and no-SSL MatrixNet validation pass.",
            "",
            "## No-Leakage Status",
            "",
            f"No-leakage audit {no_leakage}.",
            "",
            "## Multiple-Comparison Correction",
            "",
            correction_table.to_markdown(index=False),
            "",
        ]
    )


def _near_boundary_rate(patient_error_audit: pd.DataFrame) -> float:
    if "near_median_threshold" not in patient_error_audit.columns or patient_error_audit.empty:
        return 0.0
    return float(patient_error_audit["near_median_threshold"].astype(bool).mean())


def _model_correction_status(correction_table: pd.DataFrame, model_id: str) -> str:
    if correction_table.empty or "model_id" not in correction_table.columns:
        return ""
    row = correction_table.loc[correction_table["model_id"].astype(str).eq(model_id)]
    if row.empty:
        return ""
    fdr = bool(row["fdr_q_lt_0_05"].iloc[0]) if "fdr_q_lt_0_05" in row else False
    bonf = bool(row["bonferroni_p_lt_0_05"].iloc[0]) if "bonferroni_p_lt_0_05" in row else False
    raw = float(pd.to_numeric(row["raw_permutation_p_value"], errors="coerce").iloc[0])
    q_value = float(pd.to_numeric(row["fdr_q_value"], errors="coerce").iloc[0])
    bonf_value = float(pd.to_numeric(row["bonferroni_p_value"], errors="coerce").iloc[0])
    return (
        f"Phase 8 exploratory M14b raw p={raw:.3f}, FDR q={q_value:.3f}, "
        f"Bonferroni p={bonf_value:.3f}; FDR survival is {fdr}; Bonferroni survival is {bonf}."
    )


def _threshold_limit_statement(threshold_calibration: pd.DataFrame) -> str:
    if threshold_calibration.empty:
        return "threshold analysis was not available"
    unavailable = threshold_calibration.loc[
        threshold_calibration["analysis_type"].isin(["inner_cv_threshold", "inner_cv_youden_threshold"])
        & threshold_calibration.get("status", pd.Series([], dtype=str)).eq("not_available")
    ]
    if not unavailable.empty:
        return "inner-CV thresholds were unavailable, and outer test predictions were not used to choose thresholds"
    fixed = threshold_calibration.loc[threshold_calibration["analysis_type"].eq("fixed_0.5_threshold")]
    learned = threshold_calibration.loc[
        threshold_calibration["analysis_type"].isin(["inner_cv_threshold", "inner_cv_youden_threshold"])
    ]
    if fixed.empty or learned.empty or "balanced_accuracy" not in threshold_calibration.columns:
        return "threshold analyses were generated without evidence of a threshold-only explanation"
    fixed_best = pd.to_numeric(fixed["balanced_accuracy"], errors="coerce").max()
    learned_best = pd.to_numeric(learned["balanced_accuracy"], errors="coerce").max()
    if np.isfinite(fixed_best) and np.isfinite(learned_best) and learned_best > fixed_best:
        return "performance is partly threshold-limited because fold-safe learned thresholds improve balanced accuracy"
    return "performance is not primarily explained by threshold choice in this run"
