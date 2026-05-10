from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FORBIDDEN_PUBLIC_COLUMNS = {
    "subject_name",
    "file_path",
    "set_path",
    "fdt_path",
    "姓名",
    "姓名写法",
    "EEG文件夹",
}
PRIVATE_VALUE_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|" + "\\." + "set\\b|" + "\\." + "fdt\\b|" + "\\." + "xlsx\\b|" + "\\." + "xls\\b)",
    re.IGNORECASE,
)


def write_phase8_label_audit(
    label_table: pd.DataFrame,
    audit: dict[str, object],
    *,
    output_dir: str | Path,
) -> dict[str, Path]:
    root = Path(output_dir)
    report_dir = root / "reports"
    evaluation_dir = root / "evaluation"
    report_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    audit_csv = evaluation_dir / "phase8_label_audit.csv"
    transition_csv = evaluation_dir / "phase8_label_transition_table.csv"
    audit_md = report_dir / "phase8_label_audit.md"

    public_table = label_table.copy()
    assert_phase8_public_output_safe(public_table)
    public_table.to_csv(audit_csv, index=False)

    transition = _label_transition_table(public_table)
    assert_phase8_public_output_safe(transition)
    transition.to_csv(transition_csv, index=False)

    audit_md.write_text(_label_audit_markdown(public_table, audit, transition), encoding="utf-8")
    return {
        "label_audit_md": audit_md,
        "label_audit_csv": audit_csv,
        "label_transition_table": transition_csv,
    }


def write_phase8_model_report(
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
    fc_audit: dict[str, object],
    *,
    output_dir: str | Path,
    label_audit: dict[str, object] | None = None,
) -> dict[str, Path]:
    root = Path(output_dir)
    prediction_dir = root / "predictions"
    evaluation_dir = root / "evaluation"
    report_dir = root / "reports"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    prediction_csv = prediction_dir / "phase8_prop_full_edge_patient_predictions.csv"
    metrics_csv = evaluation_dir / "phase8_prop_full_edge_metrics.csv"
    report_md = report_dir / "phase8_proportional_full_edge_fc_report.md"

    assert_phase8_public_output_safe(predictions)
    assert_phase8_public_output_safe(metrics)
    predictions.to_csv(prediction_csv, index=False)
    metrics.to_csv(metrics_csv, index=False)
    report_md.write_text(_model_report_markdown(metrics, fc_audit, label_audit or {}), encoding="utf-8")
    return {
        "predictions": prediction_csv,
        "metrics": metrics_csv,
        "report": report_md,
    }


def assert_phase8_public_output_safe(frame: pd.DataFrame) -> None:
    forbidden_columns = sorted(FORBIDDEN_PUBLIC_COLUMNS.intersection(frame.columns))
    if forbidden_columns:
        raise ValueError(f"Phase 8 public output contains private columns: {forbidden_columns}")
    for column in frame.columns:
        values = frame[column].dropna().astype(str)
        if values.str.contains(PRIVATE_VALUE_RE, regex=True).any():
            raise ValueError(f"Phase 8 public output contains private path-like values in column {column}")


def assert_no_forbidden_git_artifacts_staged(staged_paths: Iterable[str]) -> None:
    forbidden_suffixes = ("." + "xlsx", "." + "xls", "." + "set", "." + "fdt")
    for path in staged_paths:
        normalized = str(path).replace("\\", "/")
        lowered = normalized.lower()
        if lowered == "outputs" or lowered.startswith("outputs/") or "/outputs/" in lowered:
            raise ValueError(f"outputs artifact must not be staged: {path}")
        if lowered.endswith(forbidden_suffixes):
            raise ValueError(f"forbidden private artifact must not be staged: {path}")
        if "checkpoint" in lowered:
            raise ValueError(f"forbidden model weight artifact must not be staged: {path}")


def _label_transition_table(label_table: pd.DataFrame) -> pd.DataFrame:
    required = {"current_clinically_meaningful", "primary_label_prop_residual"}
    if not required <= set(label_table.columns):
        return pd.DataFrame(columns=["current_clinically_meaningful", "primary_label_prop_residual", "n_patients"])
    transition = (
        label_table.loc[label_table["phase8_label_status"].eq("analyzable")]
        .groupby(["current_clinically_meaningful", "primary_label_prop_residual"], dropna=False)
        .size()
        .reset_index(name="n_patients")
    )
    return transition


def _label_audit_markdown(
    label_table: pd.DataFrame,
    audit: dict[str, object],
    transition: pd.DataFrame,
) -> str:
    changed = []
    if {"current_clinically_meaningful", "primary_label_prop_residual"} <= set(label_table.columns):
        changed_frame = label_table.loc[
            label_table["phase8_label_status"].eq("analyzable")
            & label_table["current_clinically_meaningful"].notna()
            & label_table["primary_label_prop_residual"].notna()
            & label_table["current_clinically_meaningful"].astype(str).ne(label_table["primary_label_prop_residual"].astype(str))
        ]
        changed = changed_frame["subject_id"].astype(str).tolist()
    return "\n".join(
        [
            "# Phase 8 Label Audit",
            "",
            "Primary label: proportional-residual median split.",
            f"n analyzable patients: {audit.get('n_analyzable', 0)}",
            f"n ceiling_exclude: {audit.get('n_ceiling_exclude', 0)}",
            f"n missing excluded: {audit.get('n_missing_excluded', 0)}",
            f"ProportionalRecovery count: {audit.get('n_proportional_recovery', 0)}",
            f"PoorRecovery count: {audit.get('n_poor_recovery', 0)}",
            f"median_residual: {audit.get('median_residual', np.nan)}",
            "",
            "The proportional-residual label is cohort-relative and is not an absolute achievement of 70% recovery.",
            "",
            "Label transition table rows:",
            str(len(transition)),
            "",
            "De-identified subject IDs with changed labels:",
            ", ".join(changed) if changed else "none",
            "",
        ]
    )


def _model_report_markdown(
    metrics: pd.DataFrame,
    fc_audit: dict[str, object],
    label_audit: dict[str, object],
) -> str:
    best_auc = _best_row(metrics, "roc_auc")
    best_perm = _best_permutation_row(metrics)
    any_significant = _any_permutation_significant(metrics)
    source_mode = str(fc_audit.get("source_mode", "not_available"))
    caution = ""
    if not any_significant or source_mode != "time_series":
        caution = (
            "Do not claim EEG efficacy. Proportional recovery label and full-edge FC did not produce "
            "stable time-series full-edge FC evidence in this phase."
        )
    return "\n".join(
        [
            "# Phase 8 Proportional Full-Edge FC Report",
            "",
            "## Phase Status",
            "",
            "Primary task is ProportionalRecovery vs PoorRecovery.",
            "Label definition: proportional-residual median split.",
            "no SSL started",
            "no unplanned MatrixNet training started",
            "no post-treatment EEG supervised input",
            "",
            "## Label Audit",
            "",
            "Formula: expected_delta = 0.7 * (66 - baseline_FMA_UE); observed_delta = post_FMA_UE - baseline_FMA_UE; residual = expected_delta - observed_delta.",
            f"median residual: {label_audit.get('median_residual', 'not_available')}",
            f"ProportionalRecovery count: {label_audit.get('n_proportional_recovery', 'not_available')}",
            f"PoorRecovery count: {label_audit.get('n_poor_recovery', 'not_available')}",
            "This label is cohort-relative.",
            "",
            "## Full-Edge FC Audit",
            "",
            f"reduced32 selected channels: {fc_audit.get('reduced32_n_channels', 'not_available')}",
            f"reduced32 edge count: {fc_audit.get('reduced32_n_edges', 'not_available')}",
            f"metrics included: {fc_audit.get('metrics', 'not_available')}",
            f"bands included: {fc_audit.get('bands', 'not_available')}",
            f"EO/EC handling: {fc_audit.get('conditions', 'not_available')}",
            f"source mode: {source_mode}",
            f"full62 smoke status: {fc_audit.get('full62_smoke_status', 'not_available')}",
            "",
            "## Model Results",
            "",
            metrics.to_markdown(index=False) if not metrics.empty else "No model metrics available.",
            "",
            "## Key Comparisons",
            "",
            "Compare proportional label against current clinically meaningful label, reduced32 full-edge FC against ROI-FC and summary EEG, and ridge, elastic-net, SVM, and PLS-DA model families.",
            "",
            "## Decision Answers",
            "",
            f"Best model by ROC-AUC: {best_auc}",
            f"Best model by permutation p-value: {best_perm}",
            f"Any model permutation-significant: {any_significant}",
            "MatrixNet on full-edge FC should only be considered in a later phase if patient-level LOPO performance is stable with acceptable uncertainty and permutation significance.",
            "62-channel full-edge FC should remain exploratory unless smoke results and runtime justify a written full-mode subphase.",
            "The previous current label remains a sensitivity comparator.",
            "",
            "## Scientific Caution",
            "",
            caution or "At least one model reached the configured permutation significance threshold; interpret only with uncertainty and leakage audit context.",
            "",
        ]
    )


def _best_row(metrics: pd.DataFrame, column: str) -> str:
    if column not in metrics.columns or metrics.empty:
        return "not_available"
    values = pd.to_numeric(metrics[column], errors="coerce")
    if values.notna().sum() == 0:
        return "not_available"
    row = metrics.loc[values.idxmax()]
    return f"{row['model_id']} ({column}={values.max():.3f})"


def _best_permutation_row(metrics: pd.DataFrame) -> str:
    column = "permutation_p_value"
    if column not in metrics.columns or metrics.empty:
        return "not_available"
    values = pd.to_numeric(metrics[column], errors="coerce")
    if values.notna().sum() == 0:
        return "not_available"
    row = metrics.loc[values.idxmin()]
    return f"{row['model_id']} ({column}={values.min():.3f})"


def _any_permutation_significant(metrics: pd.DataFrame) -> bool:
    if "permutation_p_value" not in metrics.columns:
        return False
    values = pd.to_numeric(metrics["permutation_p_value"], errors="coerce")
    return bool((values < 0.05).any())
