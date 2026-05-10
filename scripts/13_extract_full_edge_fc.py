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

from stroke_predict.config import load_yaml_mapping
from stroke_predict.full_edge_fc import (
    PHASE8_BANDS,
    PHASE8_FC_METHODS,
    build_canonical_full_edge_matrix,
    build_full_edge_index,
    compute_full_edge_fc,
    select_reduced32_channels,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--feature-set", choices=["reduced32", "full62"], default="reduced32")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_mapping(config_path)
    input_paths = dict(config.get("input_paths", {}))
    output_dir = _resolve(config_path, str(config.get("output_dir", "outputs")))
    _ensure_dirs(output_dir)

    labels = _load_labels(output_dir)
    subjects = labels.loc[labels["phase8_label_status"].eq("analyzable"), "subject_id"].astype(str).tolist()
    if not subjects:
        raise ValueError("Phase 8 FC extraction requires analyzable labels")

    if "toy_eeg_npz" in input_paths:
        fc_outputs = _extract_from_toy_timeseries(config_path, input_paths, output_dir, subjects, args.run_mode, args.feature_set)
    else:
        fc_outputs = _extract_from_psd_artifacts(config_path, input_paths, output_dir, subjects, args.feature_set)

    print("PHASE8_FULL_EDGE_FC_OK")
    print(f"run_mode={args.run_mode}")
    print(f"feature_set={args.feature_set}")
    print(f"n_subjects={len(subjects)}")
    print(f"n_channels={fc_outputs['n_channels']}")
    print(f"n_edges={fc_outputs['n_edges']}")
    print(f"source_mode={fc_outputs['source_mode']}")
    return 0


def _extract_from_toy_timeseries(
    config_path: Path,
    input_paths: dict[str, object],
    output_dir: Path,
    subjects: list[str],
    run_mode: str,
    feature_set: str,
) -> dict[str, object]:
    npz_path = _resolve(config_path, str(input_paths["toy_eeg_npz"]))
    index_path = _resolve(config_path, str(input_paths["toy_eeg_index"]))
    channel_path = npz_path.with_name("toy_channels.csv")
    arrays = np.load(npz_path)
    index = pd.read_csv(index_path)
    channels = pd.read_csv(channel_path)["channel"].astype(str).tolist()
    selected = _select_channels(channels, output_dir, feature_set)
    selected_indices = [channels.index(channel) for channel in selected.selected_channels]
    bands = _bands_for_run_mode(run_mode)

    condition_matrices: dict[str, list[np.ndarray]] = {"eo": [], "ec": []}
    edge_index = build_full_edge_index(selected.selected_channels)
    for subject in subjects:
        for condition in ("eo", "ec"):
            row = index.loc[index["subject_id"].astype(str).eq(subject) & index["condition"].astype(str).eq(condition)]
            if len(row) != 1:
                raise ValueError(f"Expected one toy EEG row for {subject} {condition}, found {len(row)}")
            array_key = str(row.iloc[0]["array_key"])
            data = np.asarray(arrays[array_key], dtype=float)[selected_indices, :]
            matrix, edge_index, _methods, _bands = compute_full_edge_fc(
                data,
                selected.selected_channels,
                sfreq=float(row.iloc[0]["sfreq"]),
                bands=bands,
                methods=PHASE8_FC_METHODS,
            )
            condition_matrices[condition].append(matrix)

    _write_outputs(output_dir, feature_set, selected.selected_channels, edge_index, subjects, condition_matrices, "time_series")
    return {
        "n_channels": selected.n_channels,
        "n_edges": selected.n_edges,
        "source_mode": "time_series",
    }


def _extract_from_psd_artifacts(
    config_path: Path,
    input_paths: dict[str, object],
    output_dir: Path,
    subjects: list[str],
    feature_set: str,
) -> dict[str, object]:
    psd_eo = np.load(_resolve(config_path, str(input_paths["psd_eo"])))
    psd_ec = np.load(_resolve(config_path, str(input_paths["psd_ec"])))
    dictionary = pd.read_csv(_resolve(config_path, str(input_paths["feature_dictionary"])))
    channels = _channels_from_dictionary(dictionary)
    selected = _select_channels(channels, output_dir, feature_set)
    selected_indices = [channels.index(channel) for channel in selected.selected_channels]
    edge_index = build_full_edge_index(selected.selected_channels)
    condition_matrices = {
        "eo": _psd_proxy_full_edge(psd_eo[:, 0, selected_indices, :], edge_index, selected.selected_channels),
        "ec": _psd_proxy_full_edge(psd_ec[:, 0, selected_indices, :], edge_index, selected.selected_channels),
    }
    _write_outputs(output_dir, feature_set, selected.selected_channels, edge_index, subjects, condition_matrices, "psd_artifact_proxy")
    return {
        "n_channels": selected.n_channels,
        "n_edges": selected.n_edges,
        "source_mode": "psd_artifact_proxy",
    }


def _psd_proxy_full_edge(psd: np.ndarray, edge_index: pd.DataFrame, channels: list[str]) -> list[np.ndarray]:
    bands = np.array_split(np.arange(psd.shape[-1]), len(PHASE8_BANDS))
    channel_positions = {channel: index for index, channel in enumerate(channels)}
    subject_matrices: list[np.ndarray] = []
    for subject_index in range(psd.shape[0]):
        matrix = np.zeros((len(PHASE8_FC_METHODS), len(edge_index), len(bands)), dtype=float)
        for edge in edge_index.itertuples(index=False):
            left = psd[subject_index, channel_positions[str(edge.ch_i)], :]
            right = psd[subject_index, channel_positions[str(edge.ch_j)], :]
            for band_index, freq_indices in enumerate(bands):
                x = left[freq_indices]
                y = right[freq_indices]
                corr = _safe_corr(x, y)
                matrix[0, int(edge.edge_index), band_index] = (corr + 1.0) / 2.0
                matrix[1, int(edge.edge_index), band_index] = abs(float(np.mean(x - y))) / (abs(float(np.mean(x))) + abs(float(np.mean(y))) + 1e-8)
                matrix[2, int(edge.edge_index), band_index] = abs(float(np.mean(np.sign(x - y))))
        subject_matrices.append(np.clip(matrix, 0.0, 1.0))
    return subject_matrices


def _write_outputs(
    output_dir: Path,
    feature_set: str,
    channels: list[str],
    edge_index: pd.DataFrame,
    subjects: list[str],
    condition_matrices: dict[str, list[np.ndarray]],
    source_mode: str,
) -> None:
    features_dir = output_dir / "features"
    matrices_dir = output_dir / "matrices"
    features_dir.mkdir(parents=True, exist_ok=True)
    matrices_dir.mkdir(parents=True, exist_ok=True)
    if feature_set == "reduced32":
        edge_path = features_dir / "phase8_reduced32_full_edge_index.csv"
        eo_path = matrices_dir / "phase8_fc_full_reduced32_eo.npy"
        ec_path = matrices_dir / "phase8_fc_full_reduced32_ec.npy"
    else:
        edge_path = features_dir / "phase8_full62_full_edge_index.csv"
        eo_path = matrices_dir / "phase8_fc_full62_eo.npy"
        ec_path = matrices_dir / "phase8_fc_full62_ec.npy"
    edge_index.to_csv(edge_path, index=False)
    np.save(eo_path, build_canonical_full_edge_matrix(condition_matrices["eo"]))
    np.save(ec_path, build_canonical_full_edge_matrix(condition_matrices["ec"]))
    pd.DataFrame({"subject_id": subjects}).to_csv(matrices_dir / "phase8_matrix_subject_index.csv", index=False)
    pd.DataFrame({"channel": channels}).to_csv(features_dir / f"phase8_{feature_set}_channels.csv", index=False)
    pd.DataFrame(
        [
            {
                "feature_set": feature_set,
                "source_mode": source_mode,
                "n_subjects": len(subjects),
                "n_channels": len(channels),
                "n_edges": len(edge_index),
            }
        ]
    ).to_csv(features_dir / f"phase8_{feature_set}_full_edge_audit.csv", index=False)


def _select_channels(channels: list[str], output_dir: Path, feature_set: str):
    if feature_set == "full62":
        selected_channels = list(channels)
        edge_index = build_full_edge_index(selected_channels)
        metadata = pd.DataFrame(
            {
                "selection_rank": range(len(selected_channels)),
                "selected_channel": selected_channels,
                "selection_source": "full62_available",
                "n_channels": len(selected_channels),
                "n_edges": len(edge_index),
            }
        )
        from stroke_predict.full_edge_fc import ChannelSelection

        return ChannelSelection(selected_channels, len(selected_channels), len(edge_index), metadata)
    return select_reduced32_channels(
        channels,
        output_csv=output_dir / "features" / "phase8_reduced32_channel_selection.csv",
    )


def _load_labels(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "evaluation" / "phase8_label_audit.csv"
    if not path.exists():
        raise ValueError(f"Phase 8 labels not found; run script 12 first: {path}")
    return pd.read_csv(path)


def _channels_from_dictionary(dictionary: pd.DataFrame) -> list[str]:
    rows = dictionary.loc[dictionary["feature_group"].eq("psd_matrix"), "channel"].dropna().astype(str)
    channels = []
    for channel in rows:
        if channel not in channels:
            channels.append(channel)
    if not channels:
        raise ValueError("Could not recover channel labels from feature dictionary")
    return channels


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else 0.0


def _bands_for_run_mode(run_mode: str) -> dict[str, tuple[float, float]]:
    if run_mode == "fast":
        return {name: PHASE8_BANDS[name] for name in ("theta", "alpha_mu")}
    return PHASE8_BANDS


def _ensure_dirs(output_dir: Path) -> None:
    for name in ("features", "matrices", "reports", "evaluation", "predictions"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
