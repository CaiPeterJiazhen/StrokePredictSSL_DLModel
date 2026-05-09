import numpy as np

from stroke_predict.features.psd import compute_psd_matrix, make_frequency_grid


def test_frequency_grid_has_expected_bins() -> None:
    freqs = make_frequency_grid(0.5, 45.0, 0.5)

    assert len(freqs) == 90
    assert freqs[0] == 0.5
    assert freqs[-1] == 45.0


def test_compute_psd_matrix_returns_channel_by_frequency() -> None:
    sfreq = 250
    t = np.arange(0, 8, 1 / sfreq)
    signal = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 20 * t)])

    psd, freqs = compute_psd_matrix(signal, sfreq=sfreq)

    assert psd.shape == (2, 90)
    assert freqs.shape == (90,)
    assert np.isfinite(psd).all()
    assert freqs[np.argmax(psd[0])] == 10.0

