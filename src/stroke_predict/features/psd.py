from __future__ import annotations

import numpy as np
from scipy.signal import welch


def make_frequency_grid(freq_min_hz: float = 0.5, freq_max_hz: float = 45.0, resolution_hz: float = 0.5) -> np.ndarray:
    count = int(round((freq_max_hz - freq_min_hz) / resolution_hz)) + 1
    return np.round(freq_min_hz + np.arange(count) * resolution_hz, 10)


def compute_psd_matrix(
    data: np.ndarray,
    *,
    sfreq: float,
    freq_min_hz: float = 0.5,
    freq_max_hz: float = 45.0,
    freq_resolution_hz: float = 0.5,
    window_length_sec: float = 4.0,
    overlap: float = 0.5,
    log_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError("EEG data must be shaped [n_channels, n_samples]")
    grid = make_frequency_grid(freq_min_hz, freq_max_hz, freq_resolution_hz)
    nperseg = int(round(window_length_sec * sfreq))
    noverlap = int(round(nperseg * overlap))
    if array.shape[1] < nperseg:
        raise ValueError("EEG data is shorter than one PSD window")
    freqs, power = welch(array, fs=sfreq, nperseg=nperseg, noverlap=noverlap, axis=1)
    selected = np.empty((array.shape[0], len(grid)), dtype=float)
    for channel_index in range(array.shape[0]):
        selected[channel_index] = np.interp(grid, freqs, power[channel_index])
    if log_transform:
        selected = np.log10(selected + np.finfo(float).eps)
    return selected, grid


def band_power_from_psd(psd: np.ndarray, freqs: np.ndarray, bands: dict[str, tuple[float, float]]) -> dict[tuple[str, int], float]:
    values: dict[tuple[str, int], float] = {}
    for band, (low, high) in bands.items():
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            continue
        for channel_index in range(psd.shape[0]):
            values[(band, channel_index)] = float(np.nanmean(psd[channel_index, mask]))
    return values

