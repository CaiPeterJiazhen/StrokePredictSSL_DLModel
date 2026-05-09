from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from stroke_predict.features.outputs import dictionary_frame


IDENTIFIER_COLUMNS = {"subject_id", "label_primary", "treated_hand", "affected_hand"}


def build_psd_summary_features(
    subjects: pd.DataFrame,
    channels: list[str],
    freqs: np.ndarray,
    bands: dict[str, tuple[float, float]],
    rois: dict[str, list[str]],
    views: list[str],
    psd_eo: np.ndarray,
    psd_ec: np.ndarray,
    *,
    values_are_log: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    channel_lookup = {channel.upper(): index for index, channel in enumerate(channels)}
    roi_indices = {
        roi: [channel_lookup[channel.upper()] for channel in labels if channel.upper() in channel_lookup]
        for roi, labels in rois.items()
    }
    for subject_index, subject in subjects.reset_index(drop=True).iterrows():
        row: dict[str, object] = {"subject_id": str(subject["subject_id"])}
        for condition, matrix in (("eyes_open", psd_eo), ("eyes_closed", psd_ec)):
            for view_index, view in enumerate(views):
                for band, (low, high) in bands.items():
                    freq_mask = (freqs >= low) & (freqs < high)
                    roi_values: dict[str, float | None] = {}
                    row[f"{view}_{condition}_global_{band}_power"] = _nanmean_or_none(
                        matrix[subject_index, view_index, :, :][:, freq_mask]
                    )
                    for roi, indices in roi_indices.items():
                        value = _nanmean_or_none(matrix[subject_index, view_index, indices, :][:, freq_mask]) if indices else None
                        roi_values[roi] = value
                        row[f"{view}_{condition}_{roi}_{band}_power"] = value
                    for left, right in _paired_rois(rois):
                        left_value = roi_values.get(left)
                        right_value = roi_values.get(right)
                        row[f"{view}_{condition}_{left}_minus_{right}_{band}_power"] = _diff(left_value, right_value)
                        row[f"{view}_{condition}_{left}_div_{right}_{band}_power"] = _ratio(left_value, right_value)
                        row[f"{view}_{condition}_log_{left}_minus_log_{right}_{band}_power"] = _log_diff(
                            left_value,
                            right_value,
                            values_are_log=values_are_log,
                        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_fc_summary_features(
    subjects: pd.DataFrame,
    roi_edges: list[tuple[str, str]],
    bands: dict[str, tuple[float, float]],
    methods: list[str],
    views: list[str],
    fc_eo: np.ndarray,
    fc_ec: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    band_names = list(bands)
    for subject_index, subject in subjects.reset_index(drop=True).iterrows():
        row: dict[str, object] = {"subject_id": str(subject["subject_id"])}
        for condition, matrix in (("eyes_open", fc_eo), ("eyes_closed", fc_ec)):
            for view_index, view in enumerate(views):
                for edge_index, edge in enumerate(roi_edges):
                    edge_name = f"{edge[0]}__{edge[1]}"
                    for band_index, band in enumerate(band_names):
                        for method_index, method in enumerate(methods):
                            row[f"{view}_{condition}_{edge_name}_{band}_{method}"] = float(
                                matrix[subject_index, view_index, edge_index, band_index, method_index]
                            )
                for band_index, band in enumerate(band_names):
                    for method_index, method in enumerate(methods):
                        row[f"{view}_{condition}_global_fc_{band}_{method}"] = _nanmean_or_none(
                            matrix[subject_index, view_index, :, band_index, method_index]
                        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_eo_ec_reactivity_features(*tables: pd.DataFrame) -> pd.DataFrame:
    subject_ids: list[str] = []
    for table in tables:
        if "subject_id" in table.columns:
            subject_ids = table["subject_id"].astype(str).tolist()
            break
    rows = [{"subject_id": subject_id} for subject_id in subject_ids]
    by_subject = {subject_id: row for subject_id, row in zip(subject_ids, rows)}
    for table in tables:
        if "subject_id" not in table.columns:
            continue
        indexed = table.set_index("subject_id", drop=False)
        for open_column in [column for column in table.columns if "_eyes_open_" in column]:
            closed_column = open_column.replace("_eyes_open_", "_eyes_closed_")
            if closed_column not in table.columns:
                continue
            if not pd.api.types.is_numeric_dtype(table[open_column]) or not pd.api.types.is_numeric_dtype(table[closed_column]):
                continue
            minus_column = open_column.replace("_eyes_open_", "_ec_minus_eo_")
            ratio_column = open_column.replace("_eyes_open_", "_ec_div_eo_")
            for subject_id in subject_ids:
                eo = indexed.at[subject_id, open_column]
                ec = indexed.at[subject_id, closed_column]
                by_subject[subject_id][minus_column] = _diff(ec, eo)
                by_subject[subject_id][ratio_column] = _ratio(ec, eo)
    return pd.DataFrame(rows)


def build_all_summary_features(*tables: pd.DataFrame) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for table in tables:
        if table.empty:
            continue
        if merged is None:
            merged = table.copy()
            continue
        new_columns = [column for column in table.columns if column == "subject_id" or column not in merged.columns]
        merged = merged.merge(table[new_columns], on="subject_id", how="left")
    return pd.DataFrame(columns=["subject_id"]) if merged is None else merged


def build_summary_dictionary(table_specs: Mapping[str, tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_file, (feature_group, table) in table_specs.items():
        for column in table.columns:
            if column in IDENTIFIER_COLUMNS:
                continue
            rows.append(
                {
                    "feature_name": column,
                    "feature_group": feature_group,
                    "condition": _condition_from_name(column),
                    "band": _band_from_name(column),
                    "channel": None,
                    "roi": _roi_from_name(column),
                    "metric": _metric_from_name(column),
                    "hemisphere_space": _space_from_name(column),
                    "matrix_file": source_file,
                    "axis0_subject_index": None,
                    "axis1_view_index": None,
                    "axis2_feature_index": None,
                    "axis3_feature_index": None,
                }
            )
    return dictionary_frame(rows)


def _paired_rois(rois: dict[str, list[str]]) -> list[tuple[str, str]]:
    pairs = []
    for suffix in ("motor", "frontal", "parietal"):
        left = f"left_{suffix}"
        right = f"right_{suffix}"
        if left in rois and right in rois:
            pairs.append((left, right))
    return pairs


def _nanmean_or_none(values: np.ndarray) -> float | None:
    array = np.asarray(values, dtype=float)
    if array.size == 0 or np.isnan(array).all():
        return None
    return float(np.nanmean(array))


def _diff(left: object, right: object) -> float | None:
    if _is_missing(left) or _is_missing(right):
        return None
    return float(left) - float(right)


def _ratio(left: object, right: object) -> float | None:
    if _is_missing(left) or _is_missing(right) or float(right) == 0:
        return None
    return float(left) / float(right)


def _log_diff(left: object, right: object, *, values_are_log: bool) -> float | None:
    if _is_missing(left) or _is_missing(right):
        return None
    if values_are_log:
        return float(left) - float(right)
    if float(left) <= 0 or float(right) <= 0:
        return None
    return float(np.log(float(left)) - np.log(float(right)))


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False


def _condition_from_name(name: str) -> str | None:
    if "_eyes_open_" in name:
        return "eyes_open"
    if "_eyes_closed_" in name:
        return "eyes_closed"
    if "_ec_minus_eo_" in name or "_ec_div_eo_" in name:
        return "eyes_closed_vs_eyes_open"
    return None


def _space_from_name(name: str) -> str | None:
    if name.startswith("native_"):
        return "native"
    if name.startswith("lesion_normalized_"):
        return "lesion_normalized"
    return None


def _band_from_name(name: str) -> str | None:
    for band in ("low_gamma_optional", "high_beta", "low_beta", "alpha_mu", "theta", "delta"):
        if f"_{band}_" in name or name.endswith(f"_{band}"):
            return band
    return None


def _metric_from_name(name: str) -> str:
    if name.endswith("_coherence"):
        return "coherence"
    if name.endswith("_wpli"):
        return "wpli"
    if "_div_" in name:
        return "ratio"
    if "_minus_" in name:
        return "difference"
    return "summary"


def _roi_from_name(name: str) -> str | None:
    for roi in (
        "left_motor",
        "right_motor",
        "midline_motor",
        "left_frontal",
        "right_frontal",
        "left_parietal",
        "right_parietal",
        "occipital",
        "target_roi",
        "homologous_roi",
    ):
        if roi in name:
            return roi
    return None
