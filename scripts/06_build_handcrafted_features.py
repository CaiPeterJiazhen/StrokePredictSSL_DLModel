from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_project_config
from stroke_predict.features.config import load_feature_config
from stroke_predict.features.outputs import dictionary_frame, write_public_csv
from stroke_predict.features.psd import make_frequency_grid
from stroke_predict.features.tacs import build_tacs_features


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    feature_config = load_feature_config(args.config)
    project = load_project_config(feature_config.project_config_path)
    output_dir = project.output_dir / "features"
    matrix_dir = output_dir / "matrices"
    cohort = pd.read_csv(project.output_dir / "cohort" / "cohort_master.csv")
    subjects = cohort.loc[cohort["role"].eq("supervised_main")].sort_values("subject_id").reset_index(drop=True)
    psd_eo = np.load(matrix_dir / "psd_eo.npy")
    psd_ec = np.load(matrix_dir / "psd_ec.npy")
    fc_eo = np.load(matrix_dir / "fc_roi_eo.npy")
    fc_ec = np.load(matrix_dir / "fc_roi_ec.npy")
    freqs = make_frequency_grid(feature_config.freq_min_hz, feature_config.freq_max_hz, feature_config.freq_resolution_hz)

    channels = _channels_from_dictionary(output_dir / "feature_dictionary.csv")
    band_power = _build_band_power(subjects, channels, freqs, feature_config.bands, psd_eo, psd_ec, feature_config.views)
    tacs = build_tacs_features(subjects, band_power=band_power, connectivity={})
    handcrafted = _build_handcrafted(subjects, tacs, fc_eo, fc_ec)
    write_public_csv(tacs, output_dir / "tacs_target_features.csv")
    write_public_csv(handcrafted, output_dir / "handcrafted_features.csv")

    existing = pd.read_csv(output_dir / "feature_dictionary.csv")
    rows = []
    for column in handcrafted.columns:
        if column == "subject_id":
            continue
        rows.append(
            {
                "feature_name": column,
                "feature_group": "handcrafted_summary",
                "condition": None,
                "band": _band_from_name(column, feature_config.bands),
                "channel": None,
                "roi": None,
                "metric": "summary",
                "hemisphere_space": "native" if column.startswith("native_") else ("lesion_normalized" if column.startswith("lesion_normalized_") else None),
                "matrix_file": None,
                "axis0_subject_index": None,
                "axis1_view_index": None,
                "axis2_feature_index": None,
                "axis3_feature_index": None,
            }
        )
    dictionary = pd.concat([existing, dictionary_frame(rows)], ignore_index=True)
    write_public_csv(dictionary, output_dir / "feature_dictionary.csv")
    print("HANDCRAFTED_FEATURES_OK")
    print(f"n_handcrafted_rows={len(handcrafted)}")
    print(f"n_tacs_rows={len(tacs)}")
    return 0


def _channels_from_dictionary(path: Path) -> list[str]:
    dictionary = pd.read_csv(path)
    psd = dictionary[dictionary["feature_group"].eq("psd_matrix")]
    first_view = psd[psd["axis1_view_index"].eq(0)]
    channels = first_view.sort_values("axis2_feature_index")["channel"].dropna().unique().tolist()
    return [str(channel) for channel in channels]


def _build_band_power(
    subjects: pd.DataFrame,
    channels: list[str],
    freqs: np.ndarray,
    bands: dict[str, tuple[float, float]],
    psd_eo: np.ndarray,
    psd_ec: np.ndarray,
    views: list[str],
) -> dict[tuple[str, str, str, str, str], float]:
    values: dict[tuple[str, str, str, str, str], float] = {}
    for subject_index, subject in subjects.iterrows():
        subject_id = str(subject["subject_id"])
        for condition, matrix in (("eyes_open", psd_eo), ("eyes_closed", psd_ec)):
            for view_index, view in enumerate(views):
                for channel_index, channel in enumerate(channels):
                    for band, (low, high) in bands.items():
                        mask = (freqs >= low) & (freqs < high)
                        values[(subject_id, condition, view, channel, band)] = float(np.nanmean(matrix[subject_index, view_index, channel_index, mask]))
    return values


def _build_handcrafted(subjects: pd.DataFrame, tacs: pd.DataFrame, fc_eo: np.ndarray, fc_ec: np.ndarray) -> pd.DataFrame:
    base = subjects[["subject_id", "label_primary", "treated_hand", "affected_hand"]].reset_index(drop=True)
    summaries = []
    for subject_index, subject in subjects.iterrows():
        summaries.append(
            {
                "subject_id": subject["subject_id"],
                "native_fc_roi_eo_mean": float(np.nanmean(fc_eo[subject_index, 0])),
                "native_fc_roi_ec_mean": float(np.nanmean(fc_ec[subject_index, 0])),
                "lesion_normalized_fc_roi_eo_mean": float(np.nanmean(fc_eo[subject_index, 1])),
                "lesion_normalized_fc_roi_ec_mean": float(np.nanmean(fc_ec[subject_index, 1])),
            }
        )
    return base.merge(pd.DataFrame(summaries), on="subject_id").merge(tacs, on=["subject_id", "treated_hand", "affected_hand"])


def _band_from_name(name: str, bands: dict[str, tuple[float, float]]) -> str | None:
    for band in bands:
        if band in name:
            return band
    return None


if __name__ == "__main__":
    raise SystemExit(main())

