from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_project_config
from stroke_predict.eeg.config import load_eeg_config
from stroke_predict.features.channels import flip_psd_matrix, should_flip_for_hand
from stroke_predict.features.config import load_feature_config
from stroke_predict.features.io import assert_single_channel_order, read_phase_inputs, read_record_data, select_baseline_record, supervised_subjects
from stroke_predict.features.outputs import dictionary_frame, save_matrix, write_public_csv
from stroke_predict.features.psd import band_power_from_psd, compute_psd_matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    feature_config = load_feature_config(args.config)
    eeg_config = load_eeg_config(feature_config.eeg_config_path)
    project = load_project_config(feature_config.project_config_path)
    cohort, qc, _record_index, private_by_record = read_phase_inputs(project)
    assert_single_channel_order(qc)
    subjects = supervised_subjects(cohort)

    matrices: dict[str, list[np.ndarray]] = {"eyes_open": [], "eyes_closed": []}
    dictionary_rows: list[dict[str, object]] = []
    freq_grid = None
    channels: list[str] | None = None

    for _subject_index, subject in subjects.iterrows():
        subject_views: dict[str, list[np.ndarray]] = {"eyes_open": [], "eyes_closed": []}
        for condition in ("eyes_open", "eyes_closed"):
            record_id = select_baseline_record(qc, str(subject["subject_id"]), condition)
            data, header = read_record_data(record_id, private_by_record)
            channel_labels = header.channel_labels
            if channels is None:
                channels = channel_labels
            psd, freq_grid = compute_psd_matrix(
                data,
                sfreq=float(header.sfreq),
                freq_min_hz=feature_config.freq_min_hz,
                freq_max_hz=feature_config.freq_max_hz,
                freq_resolution_hz=feature_config.freq_resolution_hz,
                window_length_sec=eeg_config.window_length_sec,
                overlap=eeg_config.window_overlap,
                log_transform=feature_config.log_transform,
            )
            subject_views[condition].append(psd)
            hand = subject.get("treated_hand") or subject.get("affected_hand")
            normalized = flip_psd_matrix(psd, channel_labels, feature_config.channel_pair_map) if should_flip_for_hand(hand) else psd.copy()
            subject_views[condition].append(normalized)
        matrices["eyes_open"].append(np.stack(subject_views["eyes_open"], axis=0))
        matrices["eyes_closed"].append(np.stack(subject_views["eyes_closed"], axis=0))

    output_dir = project.output_dir / "features"
    matrix_dir = output_dir / "matrices"
    psd_eo = np.stack(matrices["eyes_open"], axis=0)
    psd_ec = np.stack(matrices["eyes_closed"], axis=0)
    save_matrix(matrix_dir / "psd_eo.npy", psd_eo)
    save_matrix(matrix_dir / "psd_ec.npy", psd_ec)

    assert channels is not None and freq_grid is not None
    for matrix_file, condition in (("psd_eo.npy", "eyes_open"), ("psd_ec.npy", "eyes_closed")):
        for view_index, view in enumerate(feature_config.views):
            for channel_index, channel in enumerate(channels):
                for freq_index, freq in enumerate(freq_grid):
                    dictionary_rows.append(
                        {
                            "feature_name": f"{view}_{condition}_{channel}_{freq:g}Hz_psd",
                            "feature_group": "psd_matrix",
                            "condition": condition,
                            "band": None,
                            "channel": channel,
                            "roi": None,
                            "metric": "log_psd",
                            "hemisphere_space": view,
                            "matrix_file": matrix_file,
                            "axis0_subject_index": None,
                            "axis1_view_index": view_index,
                            "axis2_feature_index": channel_index,
                            "axis3_feature_index": freq_index,
                        }
                    )
    dictionary = dictionary_frame(dictionary_rows)
    write_public_csv(dictionary, output_dir / "feature_dictionary.csv")
    print("PSD_FEATURES_OK")
    print(f"n_subjects={psd_eo.shape[0]}")
    print(f"psd_eo_shape={psd_eo.shape}")
    print(f"psd_ec_shape={psd_ec.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

