from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import coherence, csd, welch


PHASE8_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha_mu": (8.0, 13.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "broad_beta": (13.0, 30.0),
}
PHASE8_FC_METHODS = ("coherence", "imaginary_coherence", "wpli")
REDUCED32_MIN_CHANNELS = 24
REDUCED32_TARGET_CHANNELS = 32
REDUCED32_PREFERRED = (
    "Fp1",
    "Fp2",
    "F7",
    "F3",
    "Fz",
    "F4",
    "F8",
    "FC5",
    "FC1",
    "FC2",
    "FC6",
    "T7",
    "C3",
    "Cz",
    "C4",
    "T8",
    "CP5",
    "CP1",
    "CP2",
    "CP6",
    "P7",
    "P3",
    "Pz",
    "P4",
    "P8",
    "POz",
    "O1",
    "Oz",
    "O2",
)
REDUCED32_FALLBACK = (
    "AF3",
    "AF4",
    "F5",
    "F1",
    "F2",
    "F6",
    "FT7",
    "FC3",
    "FC4",
    "FT8",
    "C5",
    "C1",
    "C2",
    "C6",
    "TP7",
    "CP3",
    "CP4",
    "TP8",
    "P5",
    "P1",
    "P2",
    "P6",
    "PO3",
    "PO4",
    "PO7",
    "PO8",
    "Iz",
)


@dataclass(frozen=True)
class ChannelSelection:
    selected_channels: list[str]
    n_channels: int
    n_edges: int
    metadata: pd.DataFrame


def select_reduced32_channels(
    available_channels: Iterable[str],
    *,
    output_csv: str | Path | None = None,
) -> ChannelSelection:
    lookup = {_normalize_channel(channel): str(channel) for channel in available_channels}
    selected: list[str] = []
    sources: list[str] = []

    for channel in REDUCED32_PREFERRED:
        key = _normalize_channel(channel)
        if key in lookup and lookup[key] not in selected:
            selected.append(lookup[key])
            sources.append("preferred")

    for channel in REDUCED32_FALLBACK:
        if len(selected) >= REDUCED32_TARGET_CHANNELS:
            break
        key = _normalize_channel(channel)
        if key in lookup and lookup[key] not in selected:
            selected.append(lookup[key])
            sources.append("fallback")

    if len(selected) < REDUCED32_MIN_CHANNELS:
        raise ValueError(f"Reduced32 selection found fewer than 24 valid channels: {len(selected)}")

    selected = selected[:REDUCED32_TARGET_CHANNELS]
    sources = sources[:REDUCED32_TARGET_CHANNELS]
    n_edges = len(selected) * (len(selected) - 1) // 2
    metadata = pd.DataFrame(
        {
            "selection_rank": np.arange(len(selected), dtype=int),
            "selected_channel": selected,
            "selection_source": sources,
            "n_channels": len(selected),
            "n_edges": n_edges,
        }
    )
    if output_csv is not None:
        output = Path(output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(output, index=False)
    return ChannelSelection(
        selected_channels=selected,
        n_channels=len(selected),
        n_edges=n_edges,
        metadata=metadata,
    )


def build_full_edge_index(channels: Iterable[str]) -> pd.DataFrame:
    channel_list = [str(channel) for channel in channels]
    rows: list[dict[str, object]] = []
    for edge_index, (left, right) in enumerate(combinations(channel_list, 2)):
        hemisphere_i = _hemisphere(left)
        hemisphere_j = _hemisphere(right)
        rows.append(
            {
                "edge_index": edge_index,
                "ch_i": left,
                "ch_j": right,
                "roi_i": _roi(left),
                "roi_j": _roi(right),
                "hemisphere_i": hemisphere_i,
                "hemisphere_j": hemisphere_j,
                "edge_type": _edge_type(hemisphere_i, hemisphere_j),
            }
        )
    return pd.DataFrame(rows)


def compute_full_edge_fc(
    data: np.ndarray,
    channels: list[str],
    *,
    sfreq: float,
    bands: dict[str, tuple[float, float]] | None = None,
    methods: Iterable[str] = PHASE8_FC_METHODS,
) -> tuple[np.ndarray, pd.DataFrame, list[str], list[str]]:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"Full-edge FC input must be 2D channel x time, got shape {array.shape}")
    if array.shape[0] != len(channels):
        raise ValueError("Number of channel labels must match data rows")
    band_map = bands or PHASE8_BANDS
    band_names = list(band_map)
    method_list = [str(method) for method in methods]
    unsupported = sorted(set(method_list) - set(PHASE8_FC_METHODS))
    if unsupported:
        raise ValueError(f"Unsupported Phase 8 FC methods: {unsupported}")

    edge_index = build_full_edge_index(channels)
    matrix = np.full((len(method_list), len(edge_index), len(band_names)), np.nan, dtype=float)
    for edge_row in edge_index.itertuples(index=False):
        left = int(channels.index(edge_row.ch_i))
        right = int(channels.index(edge_row.ch_j))
        for band_index, band_name in enumerate(band_names):
            band = band_map[band_name]
            for method_index, method in enumerate(method_list):
                matrix[method_index, int(edge_row.edge_index), band_index] = _connectivity_value(
                    array[left],
                    array[right],
                    sfreq=float(sfreq),
                    band=band,
                    method=method,
                )
    return matrix, edge_index, method_list, band_names


def build_canonical_full_edge_matrix(subject_matrices: Iterable[np.ndarray]) -> np.ndarray:
    matrices = [np.asarray(matrix, dtype=float) for matrix in subject_matrices]
    if not matrices:
        raise ValueError("At least one subject matrix is required")
    first_shape = matrices[0].shape
    if len(first_shape) != 3:
        raise ValueError(f"Each subject matrix must have shape C x edges x bands, got {first_shape}")
    for matrix in matrices:
        if matrix.shape != first_shape:
            raise ValueError(f"Subject matrix shapes differ: {matrix.shape} != {first_shape}")
    return np.stack(matrices, axis=0)


def _connectivity_value(
    x: np.ndarray,
    y: np.ndarray,
    *,
    sfreq: float,
    band: tuple[float, float],
    method: str,
) -> float:
    if method == "coherence":
        return _band_coherence(x, y, sfreq, band)
    if method == "imaginary_coherence":
        return _band_imaginary_coherence(x, y, sfreq, band)
    if method == "wpli":
        return _band_wpli(x, y, sfreq, band)
    raise ValueError(f"Unsupported connectivity method: {method}")


def _band_coherence(x: np.ndarray, y: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    freqs, values = coherence(x, y, fs=sfreq, nperseg=_nperseg(x, sfreq))
    mask = _band_mask(freqs, band)
    return _finite_or_zero(np.nanmean(values[mask]) if mask.any() else np.nan)


def _band_imaginary_coherence(x: np.ndarray, y: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    nperseg = _nperseg(x, sfreq)
    freqs, cross = csd(x, y, fs=sfreq, nperseg=nperseg)
    _freqs_x, pxx = welch(x, fs=sfreq, nperseg=nperseg)
    _freqs_y, pyy = welch(y, fs=sfreq, nperseg=nperseg)
    length = min(len(freqs), len(pxx), len(pyy))
    freqs = freqs[:length]
    cross = cross[:length]
    denom = np.sqrt(np.maximum(pxx[:length], 0) * np.maximum(pyy[:length], 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        imag_coh = np.abs(np.imag(cross / denom))
    mask = _band_mask(freqs, band)
    return _finite_or_zero(np.nanmean(imag_coh[mask]) if mask.any() else np.nan)


def _band_wpli(x: np.ndarray, y: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    freqs, cross = csd(x, y, fs=sfreq, nperseg=_nperseg(x, sfreq))
    mask = _band_mask(freqs, band)
    if not mask.any():
        return 0.0
    imag = np.imag(cross[mask])
    denominator = float(np.sum(np.abs(imag)))
    if denominator == 0.0:
        return 0.0
    return _finite_or_zero(abs(float(np.sum(imag))) / denominator)


def _nperseg(x: np.ndarray, sfreq: float) -> int:
    return min(len(x), max(8, int(round(4 * sfreq))))


def _band_mask(freqs: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (freqs >= band[0]) & (freqs < band[1])


def _finite_or_zero(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def _normalize_channel(channel: str) -> str:
    return str(channel).strip().upper()


def _hemisphere(channel: str) -> str:
    label = str(channel).strip()
    if label.lower().endswith("z"):
        return "midline"
    digits = "".join(character for character in label if character.isdigit())
    if digits:
        return "left" if int(digits[-1]) % 2 else "right"
    return "unknown"


def _edge_type(hemisphere_i: str, hemisphere_j: str) -> str:
    if "midline" in {hemisphere_i, hemisphere_j}:
        return "midline"
    if hemisphere_i == "left" and hemisphere_j == "left":
        return "intra_left"
    if hemisphere_i == "right" and hemisphere_j == "right":
        return "intra_right"
    if {hemisphere_i, hemisphere_j} == {"left", "right"}:
        return "interhemispheric"
    return "other"


def _roi(channel: str) -> str:
    label = str(channel).upper()
    if label.startswith(("FP", "AF", "F")):
        return "frontal"
    if label.startswith(("FC", "C")):
        return "central"
    if label.startswith(("CP", "P")):
        return "parietal"
    if label.startswith("T"):
        return "temporal"
    if label.startswith(("PO", "O", "I")):
        return "occipital"
    return "other"
