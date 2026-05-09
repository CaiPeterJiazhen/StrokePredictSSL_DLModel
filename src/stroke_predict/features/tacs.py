from __future__ import annotations

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
    connectivity: dict[tuple[str, str, str, str, str], float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    bands = sorted({key[4] for key in band_power})
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
                for band in bands:
                    target_value = band_power.get((subject_id, condition, view, target, band))
                    homologous_value = band_power.get((subject_id, condition, view, homologous, band))
                    row[f"{view}_{condition}_target_{band}_power"] = target_value
                    row[f"{view}_{condition}_homologous_{band}_power"] = homologous_value
                    row[f"{view}_{condition}_target_minus_homologous_{band}_power"] = _diff(target_value, homologous_value)
                    row[f"{view}_{condition}_target_div_homologous_{band}_power"] = _ratio(target_value, homologous_value)
                    row[f"{view}_{condition}_target_homologous_{band}_coherence"] = connectivity.get(
                        (subject_id, condition, view, target, homologous, band, "coherence")
                    )
                    row[f"{view}_{condition}_target_homologous_{band}_wpli"] = connectivity.get(
                        (subject_id, condition, view, target, homologous, band, "wpli")
                    )
        for view in ("native", "lesion_normalized"):
            for band in bands:
                eo = row.get(f"{view}_eyes_open_target_{band}_power")
                ec = row.get(f"{view}_eyes_closed_target_{band}_power")
                row[f"{view}_ec_minus_eo_target_{band}_power"] = _diff(ec, eo)
                row[f"{view}_ec_div_eo_target_{band}_power"] = _ratio(ec, eo)
        rows.append(row)
    return pd.DataFrame(rows)


def _diff(left: float | None, right: float | None) -> float | None:
    if left is None or right is None:
        return None
    return float(left - right)


def _ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right in (None, 0):
        return None
    return float(left / right)

