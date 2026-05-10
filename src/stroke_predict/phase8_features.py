from __future__ import annotations

import numpy as np
import pandas as pd


def align_full_edge_features(
    matrix: np.ndarray,
    matrix_subject_index: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    feature_prefix: str = "fullfc",
) -> pd.DataFrame:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 4:
        raise ValueError(f"Full-edge matrix must have shape N x C x edges x bands, got {array.shape}")
    if "subject_id" not in matrix_subject_index.columns:
        raise ValueError("matrix_subject_index must include subject_id")
    if "subject_id" not in labels.columns:
        raise ValueError("labels must include subject_id")
    subjects = matrix_subject_index["subject_id"].astype(str).tolist()
    if len(subjects) != array.shape[0]:
        raise ValueError("matrix_subject_index row count must match matrix axis 0")
    label_subjects = set(labels["subject_id"].astype(str))
    missing_labels = sorted(set(subjects) - label_subjects)
    if missing_labels:
        raise ValueError(f"Full-edge matrix subjects missing from labels: {missing_labels}")

    n_subjects, n_channels, n_edges, n_bands = array.shape
    flat = array.reshape(n_subjects, n_channels * n_edges * n_bands)
    columns = [
        f"{feature_prefix}_c{channel}_e{edge}_b{band}"
        for channel in range(n_channels)
        for edge in range(n_edges)
        for band in range(n_bands)
    ]
    return pd.DataFrame(flat, columns=columns).assign(subject_id=subjects)[["subject_id", *columns]]


def merge_feature_tables(base: pd.DataFrame, *feature_tables: pd.DataFrame) -> pd.DataFrame:
    if "subject_id" not in base.columns:
        raise ValueError("Base feature table must include subject_id")
    merged = base.copy()
    for table in feature_tables:
        if table is None or table.empty:
            continue
        if "subject_id" not in table.columns:
            raise ValueError("Feature table must include subject_id")
        merged = merged.merge(table, on="subject_id", how="left")
    return merged


def tag_feature_table(table: pd.DataFrame, source_prefix: str) -> pd.DataFrame:
    if "subject_id" not in table.columns:
        raise ValueError("Feature table must include subject_id")
    prefix = str(source_prefix).strip()
    if not prefix:
        raise ValueError("source_prefix must not be empty")
    rename = {
        column: f"{prefix}__{column}"
        for column in table.columns
        if column != "subject_id" and not str(column).startswith(f"{prefix}__")
    }
    return table.rename(columns=rename)
