from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Iterable

import numpy as np
from scipy.signal import coherence, csd, welch


def build_roi_edges(rois: dict[str, list[str]]) -> list[tuple[str, str]]:
    names = sorted(rois)
    return list(combinations_with_replacement(names, 2))


def compute_roi_fc_matrix(
    data: np.ndarray,
    channels: list[str],
    *,
    sfreq: float,
    rois: dict[str, list[str]],
    bands: dict[str, tuple[float, float]],
    methods: Iterable[str] = ("coherence", "wpli"),
) -> tuple[np.ndarray, list[tuple[str, str]], list[str]]:
    array = np.asarray(data, dtype=float)
    roi_series = _roi_mean_series(array, channels, rois)
    edges = build_roi_edges(rois)
    method_list = [str(method) for method in methods]
    matrix = np.full((len(edges), len(bands), len(method_list)), np.nan, dtype=float)
    for edge_index, (left, right) in enumerate(edges):
        x = roi_series.get(left)
        y = roi_series.get(right)
        if x is None or y is None:
            continue
        for band_index, (_band, bounds) in enumerate(bands.items()):
            for method_index, method in enumerate(method_list):
                if method == "coherence":
                    matrix[edge_index, band_index, method_index] = _band_coherence(x, y, sfreq, bounds)
                elif method == "wpli":
                    matrix[edge_index, band_index, method_index] = _band_wpli(x, y, sfreq, bounds)
                else:
                    raise ValueError(f"Unsupported connectivity method: {method}")
    return matrix, edges, method_list


def _roi_mean_series(data: np.ndarray, channels: list[str], rois: dict[str, list[str]]) -> dict[str, np.ndarray]:
    lookup = {channel.upper(): index for index, channel in enumerate(channels)}
    series: dict[str, np.ndarray] = {}
    for roi, labels in rois.items():
        indices = [lookup[label.upper()] for label in labels if label.upper() in lookup]
        if indices:
            series[roi] = np.nanmean(data[indices, :], axis=0)
    return series


def _band_coherence(x: np.ndarray, y: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    freqs, coh = coherence(x, y, fs=sfreq, nperseg=min(len(x), int(round(4 * sfreq))))
    mask = (freqs >= band[0]) & (freqs < band[1])
    return float(np.nanmean(coh[mask])) if mask.any() else np.nan


def _band_wpli(x: np.ndarray, y: np.ndarray, sfreq: float, band: tuple[float, float]) -> float:
    nperseg = min(len(x), int(round(4 * sfreq)))
    freqs, cross = csd(x, y, fs=sfreq, nperseg=nperseg)
    _freqs, px = welch(x, fs=sfreq, nperseg=nperseg)
    if len(freqs) != len(px):
        return np.nan
    mask = (freqs >= band[0]) & (freqs < band[1])
    if not mask.any():
        return np.nan
    imag = np.imag(cross[mask])
    denom = np.sum(np.abs(imag))
    if denom == 0:
        return 0.0
    return float(abs(np.sum(imag)) / denom)

