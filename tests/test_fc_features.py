import numpy as np

from stroke_predict.features.fc import build_roi_edges, compute_roi_fc_matrix


def test_build_roi_edges_is_stable() -> None:
    rois = {"left_motor": ["C3", "FC3"], "right_motor": ["C4", "FC4"], "midline": ["Cz"]}

    edges = build_roi_edges(rois)

    assert edges == [
        ("left_motor", "left_motor"),
        ("left_motor", "midline"),
        ("left_motor", "right_motor"),
        ("midline", "midline"),
        ("midline", "right_motor"),
        ("right_motor", "right_motor"),
    ]


def test_compute_roi_fc_matrix_shape() -> None:
    sfreq = 250
    t = np.arange(0, 8, 1 / sfreq)
    data = np.vstack(
        [
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 10 * t),
            np.sin(2 * np.pi * 20 * t),
        ]
    )
    channels = ["C3", "C4", "Cz"]
    rois = {"left": ["C3"], "right": ["C4"], "midline": ["Cz"]}
    bands = {"alpha_mu": (8, 13), "low_beta": (13, 20)}

    matrix, edges, methods = compute_roi_fc_matrix(
        data,
        channels,
        sfreq=sfreq,
        rois=rois,
        bands=bands,
        methods=("coherence", "wpli"),
    )

    assert matrix.shape == (6, 2, 2)
    assert len(edges) == 6
    assert methods == ["coherence", "wpli"]
    assert np.isfinite(matrix[:, :, 0]).all()

