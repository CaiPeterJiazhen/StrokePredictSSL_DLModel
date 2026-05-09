from __future__ import annotations

import math
from typing import Any

import pandas as pd


def map_tacs_target(treated_or_affected_hand: str | None) -> dict[str, Any]:
    hand = str(treated_or_affected_hand or "").strip().lower()
    if hand == "right":
        return {
            "target_channel": "C3",
            "homologous_channel": "C4",
            "target_roi": ["FC3", "C1", "C3", "CP3"],
            "homologous_roi": ["FC4", "C2", "C4", "CP4"],
        }
    if hand == "left":
        return {
            "target_channel": "C4",
            "homologous_channel": "C3",
            "target_roi": ["FC4", "C2", "C4", "CP4"],
            "homologous_roi": ["FC3", "C1", "C3", "CP3"],
        }
    return {"target_channel": None, "homologous_channel": None, "target_roi": [], "homologous_roi": []}


def build_tacs_features(
    cohort: pd.DataFrame,
    *,
    band_power: dict[tuple[str, str, str, str, str], float],
    connectivity: dict[tuple[str, str, str, str, str, str, str], float],
    band_power_is_log: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bands = sorted({key[4] for key in band_power})
    methods = sorted({key[6] for key in connectivity}) or ["coherence", "wpli"]
    for subject in cohort.to_dict("records"):
        subject_id = str(subject["subject_id"])
        hand = subject.get("treated_hand") or subject.get("affected_hand")
        native = map_tacs_target(hand)
        normalized = map_tacs_target("right")
        row: dict[str, Any] = {
            "subject_id": subject_id,
            "treated_hand": subject.get("treated_hand"),
            "affected_hand": subject.get("affected_hand"),
        }
        for condition in ("eyes_open", "eyes_closed"):
            for view, mapping in (("native", native), ("lesion_normalized", normalized)):
                target = mapping["target_channel"]
                homologous = mapping["homologous_channel"]
                row[f"{view}_{condition}_target_channel"] = target
                row[f"{view}_{condition}_homologous_channel"] = homologous
                target_roi_labels = [str(label) for label in mapping["target_roi"]]
                homologous_roi_labels = [str(label) for label in mapping["homologous_roi"]]
                roi_mapping = _connectivity_roi_mapping(hand, view)
                for band in bands:
                    target_value = band_power.get((subject_id, condition, view, target, band))
                    homologous_value = band_power.get((subject_id, condition, view, homologous, band))
                    target_roi_value = _mean_power(subject_id, condition, view, target_roi_labels, band, band_power)
                    homologous_roi_value = _mean_power(subject_id, condition, view, homologous_roi_labels, band, band_power)
                    row[f"{view}_{condition}_target_{band}_power"] = target_value
                    row[f"{view}_{condition}_homologous_{band}_power"] = homologous_value
                    row[f"{view}_{condition}_target_roi_mean_{band}_power"] = target_roi_value
                    row[f"{view}_{condition}_homologous_roi_mean_{band}_power"] = homologous_roi_value
                    row[f"{view}_{condition}_target_minus_homologous_{band}_power"] = _diff(target_value, homologous_value)
                    row[f"{view}_{condition}_target_div_homologous_{band}_power"] = _ratio(target_value, homologous_value)
                    row[f"{view}_{condition}_log_target_minus_log_homologous_{band}_power"] = _log_diff(
                        target_value,
                        homologous_value,
                        values_are_log=band_power_is_log,
                    )
                    row[f"{view}_{condition}_target_roi_minus_homologous_roi_{band}_power"] = _diff(
                        target_roi_value, homologous_roi_value
                    )
                    row[f"{view}_{condition}_target_roi_div_homologous_roi_{band}_power"] = _ratio(
                        target_roi_value, homologous_roi_value
                    )
                    row[f"{view}_{condition}_log_target_roi_minus_log_homologous_roi_{band}_power"] = _log_diff(
                        target_roi_value,
                        homologous_roi_value,
                        values_are_log=band_power_is_log,
                    )
                    for method in methods:
                        for label, roi in (
                            ("target_homologous", roi_mapping["homologous"]),
                            ("target_to_midline", roi_mapping["midline"]),
                            ("target_to_frontal", roi_mapping["frontal"]),
                            ("target_to_parietal", roi_mapping["parietal"]),
                        ):
                            row[f"{view}_{condition}_{label}_{band}_{method}"] = _connectivity_value(
                                connectivity,
                                subject_id,
                                condition,
                                view,
                                roi_mapping["target"],
                                roi,
                                band,
                                method,
                            )
        for view in ("native", "lesion_normalized"):
            for band in bands:
                for label in ("target", "homologous", "target_roi_mean", "homologous_roi_mean"):
                    eo = row.get(f"{view}_eyes_open_{label}_{band}_power")
                    ec = row.get(f"{view}_eyes_closed_{label}_{band}_power")
                    row[f"{view}_ec_minus_eo_{label}_{band}_power"] = _diff(ec, eo)
                    row[f"{view}_ec_div_eo_{label}_{band}_power"] = _ratio(ec, eo)
                for method in methods:
                    for label in ("target_homologous", "target_to_midline", "target_to_frontal", "target_to_parietal"):
                        eo = row.get(f"{view}_eyes_open_{label}_{band}_{method}")
                        ec = row.get(f"{view}_eyes_closed_{label}_{band}_{method}")
                        row[f"{view}_ec_minus_eo_{label}_{band}_{method}"] = _diff(ec, eo)
                        row[f"{view}_ec_div_eo_{label}_{band}_{method}"] = _ratio(ec, eo)
        rows.append(row)
    return pd.DataFrame(rows)


def _diff(left: float | None, right: float | None) -> float | None:
    if _is_missing(left) or _is_missing(right):
        return None
    return float(left - right)


def _ratio(left: float | None, right: float | None) -> float | None:
    if _is_missing(left) or _is_missing(right) or float(right) == 0:
        return None
    return float(left / right)


def _log_diff(left: float | None, right: float | None, *, values_are_log: bool) -> float | None:
    if _is_missing(left) or _is_missing(right):
        return None
    if values_are_log:
        return float(left - right)
    if float(left) <= 0 or float(right) <= 0:
        return None
    return float(math.log(float(left)) - math.log(float(right)))


def _mean_power(
    subject_id: str,
    condition: str,
    view: str,
    channels: list[str],
    band: str,
    band_power: dict[tuple[str, str, str, str, str], float],
) -> float | None:
    values = [
        band_power.get((subject_id, condition, view, channel, band))
        for channel in channels
        if not _is_missing(band_power.get((subject_id, condition, view, channel, band)))
    ]
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def _connectivity_roi_mapping(hand: str | None, view: str) -> dict[str, str]:
    normalized = view == "lesion_normalized"
    hand_text = str(hand or "").strip().lower()
    if normalized or hand_text == "right":
        side = "left"
        opposite = "right"
    elif hand_text == "left":
        side = "right"
        opposite = "left"
    else:
        side = "left"
        opposite = "right"
    return {
        "target": f"{side}_motor",
        "homologous": f"{opposite}_motor",
        "midline": "midline_motor",
        "frontal": f"{side}_frontal",
        "parietal": f"{side}_parietal",
    }


def _connectivity_value(
    connectivity: dict[tuple[str, str, str, str, str, str, str], float],
    subject_id: str,
    condition: str,
    view: str,
    left_roi: str,
    right_roi: str,
    band: str,
    method: str,
) -> float | None:
    value = connectivity.get((subject_id, condition, view, left_roi, right_roi, band, method))
    if _is_missing(value):
        value = connectivity.get((subject_id, condition, view, right_roi, left_roi, band, method))
    return None if _is_missing(value) else float(value)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except TypeError:
        return False

