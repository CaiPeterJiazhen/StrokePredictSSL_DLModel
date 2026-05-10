from __future__ import annotations

import numpy as np

from stroke_predict.full_edge_fc import (
    PHASE8_BANDS,
    PHASE8_FC_METHODS,
    build_canonical_full_edge_matrix,
    build_full_edge_index,
    compute_full_edge_fc,
    select_reduced32_channels,
)


def test_full_edge_count_matches_channel_pairs() -> None:
    channels = ["Fp1", "Fp2", "F3", "F4", "C3"]

    edge_index = build_full_edge_index(channels)

    assert len(edge_index) == len(channels) * (len(channels) - 1) // 2
    assert {"edge_index", "ch_i", "ch_j", "edge_type"} <= set(edge_index.columns)
    assert edge_index["edge_index"].tolist() == list(range(len(edge_index)))


def test_reduced32_selector_is_deterministic_and_writes_metadata(tmp_path) -> None:
    channels = [
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
        "AF3",
        "AF4",
        "PO3",
    ]

    first = select_reduced32_channels(channels, output_csv=tmp_path / "selection.csv")
    second = select_reduced32_channels(list(reversed(channels)), output_csv=tmp_path / "selection2.csv")

    assert first.selected_channels == second.selected_channels
    assert first.n_channels == 32
    assert first.n_edges == 496
    metadata = (tmp_path / "selection.csv").read_text(encoding="utf-8")
    assert "selected_channel" in metadata
    assert "local_path" not in metadata


def test_reduced32_selector_fails_with_too_few_valid_channels() -> None:
    channels = ["C3", "C4", "Cz", "Pz"]

    try:
        select_reduced32_channels(channels)
    except ValueError as exc:
        assert "fewer than 24" in str(exc)
    else:
        raise AssertionError("Expected reduced32 selection to fail with too few channels")


def test_edge_index_contains_metadata_without_private_paths() -> None:
    edge_index = build_full_edge_index(["C3", "C4", "Cz", "P3"])

    assert {"roi_i", "roi_j", "hemisphere_i", "hemisphere_j"} <= set(edge_index.columns)
    assert set(edge_index["edge_type"]) >= {"interhemispheric", "midline", "intra_left"}
    text = edge_index.astype(str).to_string()
    assert "local_path" not in text
    assert "private" not in text.lower()


def test_compute_full_edge_fc_returns_finite_required_metrics() -> None:
    sfreq = 250
    t = np.arange(0, 8, 1 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 10 * t + 0.3),
            np.sin(2 * np.pi * 18 * t),
            np.sin(2 * np.pi * 6 * t),
        ]
    )
    channels = ["C3", "C4", "P3", "P4"]
    bands = {"theta": PHASE8_BANDS["theta"], "alpha_mu": PHASE8_BANDS["alpha_mu"]}

    matrix, edge_index, methods, band_names = compute_full_edge_fc(
        data,
        channels,
        sfreq=sfreq,
        bands=bands,
        methods=PHASE8_FC_METHODS,
    )

    assert matrix.shape == (len(PHASE8_FC_METHODS), 6, 2)
    assert len(edge_index) == 6
    assert methods == list(PHASE8_FC_METHODS)
    assert band_names == ["theta", "alpha_mu"]
    assert np.isfinite(matrix).all()


def test_canonical_matrix_shape_is_subject_channel_edge_band() -> None:
    subject_a = np.ones((3, 6, 2))
    subject_b = np.zeros((3, 6, 2))

    matrix = build_canonical_full_edge_matrix([subject_a, subject_b])

    assert matrix.shape == (2, 3, 6, 2)
