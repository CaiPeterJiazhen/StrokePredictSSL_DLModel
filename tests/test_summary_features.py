from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stroke_predict.features.summary import (
    build_all_summary_features,
    build_eo_ec_reactivity_features,
    build_fc_summary_features,
    build_psd_summary_features,
    build_summary_dictionary,
)


def test_build_psd_summary_features_includes_roi_asymmetry() -> None:
    subjects = pd.DataFrame({"subject_id": ["STK-001"], "label_primary": ["Good"]})
    channels = ["C3", "C4", "Cz", "F3", "F4", "P3", "P4"]
    freqs = np.asarray([2.0, 10.0])
    bands = {"delta": (1.0, 4.0), "alpha_mu": (8.0, 13.0)}
    rois = {
        "left_motor": ["C3"],
        "right_motor": ["C4"],
        "midline_motor": ["Cz"],
        "left_frontal": ["F3"],
        "right_frontal": ["F4"],
        "left_parietal": ["P3"],
        "right_parietal": ["P4"],
    }
    psd_eo = np.zeros((1, 2, len(channels), len(freqs)), dtype=float)
    psd_ec = np.zeros_like(psd_eo)
    psd_eo[0, 0, channels.index("C3"), 0] = 4.0
    psd_eo[0, 0, channels.index("C4"), 0] = 2.0
    psd_ec[0, 0, channels.index("C3"), 0] = 8.0
    psd_ec[0, 0, channels.index("C4"), 0] = 4.0

    summary = build_psd_summary_features(subjects, channels, freqs, bands, rois, ["native", "lesion_normalized"], psd_eo, psd_ec)

    assert summary.loc[0, "native_eyes_open_left_motor_delta_power"] == pytest.approx(4.0)
    assert summary.loc[0, "native_eyes_open_left_motor_minus_right_motor_delta_power"] == pytest.approx(2.0)
    assert summary.loc[0, "native_eyes_open_left_motor_div_right_motor_delta_power"] == pytest.approx(2.0)
    assert summary.loc[0, "native_eyes_open_log_left_motor_minus_log_right_motor_delta_power"] == pytest.approx(2.0)


def test_build_fc_summary_features_uses_roi_edges_and_methods() -> None:
    subjects = pd.DataFrame({"subject_id": ["STK-001"]})
    edges = [("left_motor", "right_motor"), ("left_motor", "midline_motor")]
    bands = {"alpha_mu": (8.0, 13.0)}
    methods = ["coherence", "wpli"]
    fc_eo = np.zeros((1, 2, len(edges), len(bands), len(methods)), dtype=float)
    fc_ec = np.zeros_like(fc_eo)
    fc_eo[0, 0, 0, 0, 0] = 0.6
    fc_eo[0, 0, 0, 0, 1] = 0.4
    fc_ec[0, 0, 0, 0, 0] = 0.7

    summary = build_fc_summary_features(subjects, edges, bands, methods, ["native", "lesion_normalized"], fc_eo, fc_ec)

    assert summary.loc[0, "native_eyes_open_left_motor__right_motor_alpha_mu_coherence"] == pytest.approx(0.6)
    assert summary.loc[0, "native_eyes_open_left_motor__right_motor_alpha_mu_wpli"] == pytest.approx(0.4)


def test_reactivity_all_summary_and_dictionary_outputs() -> None:
    psd = pd.DataFrame(
        {
            "subject_id": ["STK-001"],
            "native_eyes_open_left_motor_delta_power": [4.0],
            "native_eyes_closed_left_motor_delta_power": [8.0],
        }
    )
    fc = pd.DataFrame(
        {
            "subject_id": ["STK-001"],
            "native_eyes_open_left_motor__right_motor_alpha_mu_coherence": [0.6],
            "native_eyes_closed_left_motor__right_motor_alpha_mu_coherence": [0.9],
        }
    )
    tacs = pd.DataFrame(
        {
            "subject_id": ["STK-001"],
            "native_eyes_open_target_alpha_mu_power": [2.0],
            "native_eyes_closed_target_alpha_mu_power": [6.0],
        }
    )

    reactivity = build_eo_ec_reactivity_features(psd, fc, tacs)
    all_summary = build_all_summary_features(psd, fc, tacs, reactivity)
    dictionary = build_summary_dictionary(
        {
            "features_psd_summary.csv": ("psd_summary", psd),
            "features_fc_summary.csv": ("fc_summary", fc),
            "features_tacs_target_summary.csv": ("tacs_target_summary", tacs),
            "features_eo_ec_reactivity.csv": ("eo_ec_reactivity", reactivity),
        }
    )

    assert reactivity.loc[0, "native_ec_minus_eo_left_motor_delta_power"] == pytest.approx(4.0)
    assert reactivity.loc[0, "native_ec_div_eo_left_motor_delta_power"] == pytest.approx(2.0)
    assert reactivity.loc[0, "native_ec_minus_eo_left_motor__right_motor_alpha_mu_coherence"] == pytest.approx(0.3)
    assert "native_ec_minus_eo_target_alpha_mu_power" in all_summary.columns
    assert set(dictionary["feature_group"]) == {"psd_summary", "fc_summary", "tacs_target_summary", "eo_ec_reactivity"}


def test_reactivity_ignores_non_numeric_eye_condition_columns() -> None:
    tacs = pd.DataFrame(
        {
            "subject_id": ["STK-001"],
            "native_eyes_open_target_channel": ["C3"],
            "native_eyes_closed_target_channel": ["C3"],
            "native_eyes_open_target_alpha_mu_power": [2.0],
            "native_eyes_closed_target_alpha_mu_power": [6.0],
        }
    )

    reactivity = build_eo_ec_reactivity_features(tacs)

    assert "native_ec_minus_eo_target_channel" not in reactivity.columns
    assert reactivity.loc[0, "native_ec_minus_eo_target_alpha_mu_power"] == pytest.approx(4.0)
