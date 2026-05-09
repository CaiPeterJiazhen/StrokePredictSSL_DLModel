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
from stroke_predict.features.channels import should_flip_for_hand
from stroke_predict.features.config import load_feature_config
from stroke_predict.features.fc import compute_roi_fc_matrix
from stroke_predict.features.io import assert_single_channel_order, read_phase_inputs, read_record_data, select_baseline_record, supervised_subjects
from stroke_predict.features.outputs import dictionary_frame, save_matrix, write_public_csv


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    feature_config = load_feature_config(args.config)
    project = load_project_config(feature_config.project_config_path)
    cohort, qc, _record_index, private_by_record = read_phase_inputs(project)
    assert_single_channel_order(qc)
    subjects = supervised_subjects(cohort)

    matrices: dict[str, list[np.ndarray]] = {"eyes_open": [], "eyes_closed": []}
    roi_edges = None
    methods = None
    for _subject_index, subject in subjects.iterrows():
        hand = subject.get("treated_hand") or subject.get("affected_hand")
        for condition in ("eyes_open", "eyes_closed"):
            record_id = select_baseline_record(qc, str(subject["subject_id"]), condition)
            data, header = read_record_data(record_id, private_by_record)
            native, roi_edges, methods = compute_roi_fc_matrix(
                data,
                header.channel_labels,
                sfreq=float(header.sfreq),
                rois=feature_config.rois,
                bands=feature_config.bands,
                methods=feature_config.connectivity_methods,
            )
            normalized_data = _flip_data_if_needed(data, header.channel_labels, hand, feature_config.channel_pair_map)
            normalized, _edges, _methods = compute_roi_fc_matrix(
                normalized_data,
                header.channel_labels,
                sfreq=float(header.sfreq),
                rois=feature_config.rois,
                bands=feature_config.bands,
                methods=feature_config.connectivity_methods,
            )
            matrices[condition].append(np.stack([native, normalized], axis=0))

    output_dir = project.output_dir / "features"
    matrix_dir = output_dir / "matrices"
    fc_eo = np.stack(matrices["eyes_open"], axis=0)
    fc_ec = np.stack(matrices["eyes_closed"], axis=0)
    save_matrix(matrix_dir / "fc_roi_eo.npy", fc_eo)
    save_matrix(matrix_dir / "fc_roi_ec.npy", fc_ec)

    assert roi_edges is not None and methods is not None
    existing = _read_existing_dictionary(output_dir / "feature_dictionary.csv")
    rows: list[dict[str, object]] = []
    for matrix_file, condition in (("fc_roi_eo.npy", "eyes_open"), ("fc_roi_ec.npy", "eyes_closed")):
        for view_index, view in enumerate(feature_config.views):
            for edge_index, edge in enumerate(roi_edges):
                for band_index, band in enumerate(feature_config.bands):
                    for method_index, method in enumerate(methods):
                        rows.append(
                            {
                                "feature_name": f"{view}_{condition}_{edge[0]}__{edge[1]}_{band}_{method}",
                                "feature_group": "fc_roi_matrix",
                                "condition": condition,
                                "band": band,
                                "channel": None,
                                "roi": f"{edge[0]}__{edge[1]}",
                                "metric": method,
                                "hemisphere_space": view,
                                "matrix_file": matrix_file,
                                "axis0_subject_index": None,
                                "axis1_view_index": view_index,
                                "axis2_feature_index": edge_index,
                                "axis3_feature_index": band_index,
                            }
                        )
    dictionary = pd.concat([existing, dictionary_frame(rows)], ignore_index=True)
    write_public_csv(dictionary, output_dir / "feature_dictionary.csv")
    print("FC_FEATURES_OK")
    print(f"fc_roi_eo_shape={fc_eo.shape}")
    print(f"fc_roi_ec_shape={fc_ec.shape}")
    return 0


def _flip_data_if_needed(data: np.ndarray, channels: list[str], hand: str | None, pair_map: dict[str, str]) -> np.ndarray:
    if not should_flip_for_hand(hand):
        return data.copy()
    from stroke_predict.features.channels import build_flip_indices

    return data[build_flip_indices(channels, pair_map), :]


def _read_existing_dictionary(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


if __name__ == "__main__":
    raise SystemExit(main())

