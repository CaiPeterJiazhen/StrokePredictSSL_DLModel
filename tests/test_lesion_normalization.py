import numpy as np

from stroke_predict.features.channels import (
    DEFAULT_CHANNEL_PAIR_MAP,
    build_flip_indices,
    flip_fc_edges,
    flip_psd_matrix,
)


CHANNELS = ["FC3", "FC4", "C3", "C4", "CP3", "CP4", "Cz"]


def test_pair_map_contains_required_motor_pairs() -> None:
    assert DEFAULT_CHANNEL_PAIR_MAP["C3"] == "C4"
    assert DEFAULT_CHANNEL_PAIR_MAP["FC3"] == "FC4"
    assert DEFAULT_CHANNEL_PAIR_MAP["CP3"] == "CP4"


def test_flip_indices_swap_motor_channels_and_keep_midline() -> None:
    indices = build_flip_indices(CHANNELS, DEFAULT_CHANNEL_PAIR_MAP)
    flipped = [CHANNELS[i] for i in indices]

    assert flipped == ["FC4", "FC3", "C4", "C3", "CP4", "CP3", "Cz"]


def test_flip_psd_matrix_swaps_channel_axis() -> None:
    psd = np.arange(len(CHANNELS) * 2).reshape(len(CHANNELS), 2)
    flipped = flip_psd_matrix(psd, CHANNELS, DEFAULT_CHANNEL_PAIR_MAP)

    assert np.array_equal(flipped[CHANNELS.index("C3")], psd[CHANNELS.index("C4")])
    assert np.array_equal(flipped[CHANNELS.index("FC4")], psd[CHANNELS.index("FC3")])
    assert np.array_equal(flipped[CHANNELS.index("Cz")], psd[CHANNELS.index("Cz")])


def test_flip_fc_edges_maps_edge_endpoints() -> None:
    edges = [("C3", "FC3"), ("C4", "FC4"), ("C3", "Cz")]
    values = np.array([[1.0], [2.0], [3.0]])

    flipped_edges, flipped_values = flip_fc_edges(edges, values, DEFAULT_CHANNEL_PAIR_MAP)

    assert ("C4", "FC4") in flipped_edges
    assert ("C3", "FC3") in flipped_edges
    assert ("C4", "Cz") in flipped_edges
    assert float(flipped_values[flipped_edges.index(("C4", "FC4")), 0]) == 1.0

