from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stroke_predict.matrixnet import MatrixNet, MatrixNetConfig
from stroke_predict.matrixnet_data import load_matrixnet_inputs

SSL_VARIANTS = (
    "stroke_baseline",
    "stroke_healthy_baseline",
    "stroke_all_stage",
    "stroke_all_stage_healthy",
)
STROKE_SOURCES = {"stroke_supervised", "stroke_ssl_only"}
HEALTHY_SOURCES = {"healthy"}
BASELINE_STAGES = {"baseline"}
ALL_STAGES = {"baseline", "immediate", "mid", "final"}
CONDITIONS = {"eo", "ec"}
REQUIRED_SSL_INDEX_COLUMNS = {"subject_id", "source", "stage", "condition"}
RAW_PATH_MARKERS = ("Patient_" + "tACS", "Health_" + "tACS", "RestingStateEEG_" + "afterProcess")
PRIVATE_PATTERN = re.compile(
    r"([A-Za-z]:[\\/]|\.set\b|\.fdt\b|\.xlsx\b|" + "|".join(map(re.escape, RAW_PATH_MARKERS)) + r")",
    re.IGNORECASE,
)


def assert_no_private_strings(value: pd.DataFrame | str) -> None:
    if isinstance(value, pd.DataFrame):
        text = value.astype(str).to_csv(index=False)
    else:
        text = str(value)
    if PRIVATE_PATTERN.search(text):
        raise ValueError("public SSL output contains private or raw path strings")


def validate_ssl_matrix_index(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_SSL_INDEX_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"SSL matrix index missing columns: {sorted(missing)}")
    validated = frame.copy()
    for column in ("subject_id", "source", "stage", "condition"):
        validated[column] = validated[column].astype(str).str.strip()
    validated["source"] = validated["source"].str.lower()
    validated["stage"] = validated["stage"].str.lower()
    validated["condition"] = validated["condition"].str.lower()
    unknown_sources = sorted(set(validated["source"]) - STROKE_SOURCES - HEALTHY_SOURCES)
    if unknown_sources:
        raise ValueError(f"Unknown SSL sources: {unknown_sources}")
    unknown_stages = sorted(set(validated["stage"]) - ALL_STAGES)
    if unknown_stages:
        raise ValueError(f"Unknown SSL stages: {unknown_stages}")
    unknown_conditions = sorted(set(validated["condition"]) - CONDITIONS)
    if unknown_conditions:
        raise ValueError(f"Unknown SSL conditions: {unknown_conditions}")
    if "row_index" not in validated.columns:
        validated["row_index"] = np.arange(len(validated), dtype=int)
    validated["row_index"] = validated["row_index"].astype(int)
    if validated["row_index"].duplicated().any():
        raise ValueError("SSL matrix row_index values must be unique")
    validated = _add_variant_eligibility(validated)
    assert_no_private_strings(validated)
    return validated.reset_index(drop=True)


def validate_ssl_matrix_arrays(index: pd.DataFrame, *, psd: np.ndarray, fc: np.ndarray) -> None:
    n_rows = len(index)
    if int(psd.shape[0]) != n_rows:
        raise ValueError(f"PSD row count {psd.shape[0]} does not match SSL metadata {n_rows}")
    if int(fc.shape[0]) != n_rows:
        raise ValueError(f"FC row count {fc.shape[0]} does not match SSL metadata {n_rows}")
    if not np.isfinite(psd).all():
        raise ValueError("PSD SSL matrix contains NaN or Inf")
    if not np.isfinite(fc).all():
        raise ValueError("FC SSL matrix contains NaN or Inf")


def build_ssl_fold_pools(
    index: pd.DataFrame,
    outer_folds: dict[str, Any],
    *,
    ssl_variant: str,
    fold_limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ssl_variant not in SSL_VARIANTS:
        raise ValueError(f"Unsupported ssl_variant: {ssl_variant}")
    validated = validate_ssl_matrix_index(index)
    folds = list(outer_folds.get("folds", []))
    if fold_limit is not None:
        folds = folds[: int(fold_limit)]
    pool_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    eligible_column = f"eligible_{ssl_variant}"
    for fold in folds:
        outer_fold = int(fold["outer_fold"])
        test_subject = str(fold["test_subject"])
        eligible = validated[validated[eligible_column].astype(bool)].copy()
        included = eligible[~eligible["subject_id"].astype(str).eq(test_subject)].copy()
        excluded_test = eligible[eligible["subject_id"].astype(str).eq(test_subject)]
        for _, row in included.iterrows():
            row_dict = row.to_dict()
            row_dict.update(
                {
                    "outer_fold": outer_fold,
                    "test_subject": test_subject,
                    "ssl_variant": ssl_variant,
                }
            )
            pool_rows.append(row_dict)
        audit_rows.append(
            {
                "outer_fold": outer_fold,
                "test_subject": test_subject,
                "ssl_variant": ssl_variant,
                "test_subject_records_in_pool": int(included["subject_id"].astype(str).eq(test_subject).sum()),
                "test_subject_records_excluded": int(len(excluded_test)),
                "healthy_records_in_pool": int(included["source"].isin(HEALTHY_SOURCES).sum()),
                "unlabeled_stroke_records_in_pool": int(included["source"].eq("stroke_ssl_only").sum()),
                "ssl_pool_size": int(len(included)),
                "leakage_passed": bool(not included["subject_id"].astype(str).eq(test_subject).any()),
            }
        )
    pool = pd.DataFrame(pool_rows)
    audit = pd.DataFrame(audit_rows)
    assert_no_private_strings(pool if not pool.empty else "")
    assert_no_private_strings(audit if not audit.empty else "")
    return pool, audit


def build_ssl_matrix_index_from_baseline_outputs(output_dir: str | Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    inputs = load_matrixnet_inputs(output_dir)
    psd_eo = _canonicalize_psd_array(inputs.psd_eo)
    psd_ec = _canonicalize_psd_array(inputs.psd_ec)
    fc_eo = _canonicalize_fc_array(inputs.fc_eo)
    fc_ec = _canonicalize_fc_array(inputs.fc_ec)
    rows: list[dict[str, object]] = []
    psd_rows: list[np.ndarray] = []
    fc_rows: list[np.ndarray] = []
    row_index = 0
    for subject_index, subject_id in enumerate(inputs.subject_ids):
        for condition, psd_source, fc_source in (
            ("eo", psd_eo, fc_eo),
            ("ec", psd_ec, fc_ec),
        ):
            rows.append(
                {
                    "row_index": row_index,
                    "subject_id": subject_id,
                    "source": "stroke_supervised",
                    "stage": "baseline",
                    "condition": condition,
                }
            )
            psd_rows.append(psd_source[subject_index])
            fc_rows.append(fc_source[subject_index])
            row_index += 1
    index = validate_ssl_matrix_index(pd.DataFrame(rows))
    psd = np.stack(psd_rows).astype(np.float32)
    fc = np.stack(fc_rows).astype(np.float32)
    validate_ssl_matrix_arrays(index, psd=psd, fc=fc)
    return index, psd, fc


def _add_variant_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame["source"].astype(str)
    stage = frame["stage"].astype(str)
    is_stroke = source.isin(STROKE_SOURCES)
    is_healthy = source.isin(HEALTHY_SOURCES)
    is_baseline = stage.isin(BASELINE_STAGES)
    is_all_stage = stage.isin(ALL_STAGES)
    frame = frame.copy()
    frame["eligible_stroke_baseline"] = is_stroke & is_baseline
    frame["eligible_stroke_healthy_baseline"] = (is_stroke & is_baseline) | (is_healthy & is_baseline)
    frame["eligible_stroke_all_stage"] = is_stroke & is_all_stage
    frame["eligible_stroke_all_stage_healthy"] = (is_stroke & is_all_stage) | (is_healthy & is_baseline)
    return frame


def _canonicalize_psd_array(array: np.ndarray) -> np.ndarray:
    tensor = MatrixNet(MatrixNetConfig(use_psd=True, use_fc=False, dropout=0.0)).canonicalize_psd(
        _to_torch(array)
    )
    return tensor.detach().cpu().numpy().astype(np.float32)


def _canonicalize_fc_array(array: np.ndarray) -> np.ndarray:
    tensor = MatrixNet(MatrixNetConfig(use_psd=False, use_fc=True, dropout=0.0)).canonicalize_fc(
        _to_torch(array)
    )
    return tensor.detach().cpu().numpy().astype(np.float32)


def _to_torch(array: np.ndarray):
    import torch

    return torch.from_numpy(array).float()
