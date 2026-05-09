import pandas as pd
import pytest

from stroke_predict.features.tacs import build_tacs_features, map_tacs_target


def test_map_tacs_target_uses_c3_for_right_and_c4_for_left() -> None:
    assert map_tacs_target("right")["target_channel"] == "C3"
    assert map_tacs_target("right")["homologous_channel"] == "C4"
    assert map_tacs_target("left")["target_channel"] == "C4"
    assert map_tacs_target("left")["homologous_channel"] == "C3"


def test_build_tacs_features_includes_native_and_normalized_names() -> None:
    cohort = pd.DataFrame({"subject_id": ["STK-001"], "treated_hand": ["left"], "affected_hand": ["left"]})
    band_power = {
        ("STK-001", "eyes_open", "native", "C4", "alpha_mu"): 4.0,
        ("STK-001", "eyes_open", "native", "C3", "alpha_mu"): 2.0,
        ("STK-001", "eyes_open", "lesion_normalized", "C3", "alpha_mu"): 4.0,
        ("STK-001", "eyes_open", "lesion_normalized", "C4", "alpha_mu"): 2.0,
    }

    features = build_tacs_features(cohort, band_power=band_power, connectivity={})

    assert features.loc[0, "native_eyes_open_target_channel"] == "C4"
    assert features.loc[0, "lesion_normalized_eyes_open_target_channel"] == "C3"
    assert features.loc[0, "native_eyes_open_target_alpha_mu_power"] == 4.0
    assert features.loc[0, "lesion_normalized_eyes_open_target_alpha_mu_power"] == 4.0


def test_build_tacs_features_uses_roi_power_connectivity_and_reactivity() -> None:
    cohort = pd.DataFrame({"subject_id": ["STK-001"], "treated_hand": ["right"], "affected_hand": ["right"]})
    band_power = {
        ("STK-001", "eyes_open", "native", "C3", "alpha_mu"): 4.0,
        ("STK-001", "eyes_open", "native", "C4", "alpha_mu"): 2.0,
        ("STK-001", "eyes_open", "native", "FC3", "alpha_mu"): 5.0,
        ("STK-001", "eyes_open", "native", "C1", "alpha_mu"): 3.0,
        ("STK-001", "eyes_open", "native", "CP3", "alpha_mu"): 4.0,
        ("STK-001", "eyes_open", "native", "FC4", "alpha_mu"): 1.0,
        ("STK-001", "eyes_open", "native", "C2", "alpha_mu"): 2.0,
        ("STK-001", "eyes_open", "native", "CP4", "alpha_mu"): 3.0,
        ("STK-001", "eyes_closed", "native", "C3", "alpha_mu"): 8.0,
        ("STK-001", "eyes_closed", "native", "C4", "alpha_mu"): 4.0,
        ("STK-001", "eyes_closed", "native", "FC3", "alpha_mu"): 9.0,
        ("STK-001", "eyes_closed", "native", "C1", "alpha_mu"): 7.0,
        ("STK-001", "eyes_closed", "native", "CP3", "alpha_mu"): 8.0,
        ("STK-001", "eyes_closed", "native", "FC4", "alpha_mu"): 3.0,
        ("STK-001", "eyes_closed", "native", "C2", "alpha_mu"): 4.0,
        ("STK-001", "eyes_closed", "native", "CP4", "alpha_mu"): 5.0,
        ("STK-001", "eyes_open", "lesion_normalized", "C3", "alpha_mu"): 4.0,
        ("STK-001", "eyes_open", "lesion_normalized", "C4", "alpha_mu"): 2.0,
        ("STK-001", "eyes_closed", "lesion_normalized", "C3", "alpha_mu"): 8.0,
        ("STK-001", "eyes_closed", "lesion_normalized", "C4", "alpha_mu"): 4.0,
    }
    connectivity = {
        ("STK-001", "eyes_open", "native", "left_motor", "right_motor", "alpha_mu", "coherence"): 0.61,
        ("STK-001", "eyes_open", "native", "left_motor", "right_motor", "alpha_mu", "wpli"): 0.62,
        ("STK-001", "eyes_open", "native", "left_motor", "midline_motor", "alpha_mu", "coherence"): 0.71,
        ("STK-001", "eyes_open", "native", "left_motor", "left_frontal", "alpha_mu", "coherence"): 0.81,
        ("STK-001", "eyes_open", "native", "left_motor", "left_parietal", "alpha_mu", "coherence"): 0.91,
        ("STK-001", "eyes_closed", "native", "left_motor", "right_motor", "alpha_mu", "coherence"): 0.66,
    }

    features = build_tacs_features(
        cohort,
        band_power=band_power,
        connectivity=connectivity,
        band_power_is_log=False,
    )

    assert features.loc[0, "native_eyes_open_target_roi_mean_alpha_mu_power"] == pytest.approx(4.0)
    assert features.loc[0, "native_eyes_open_homologous_roi_mean_alpha_mu_power"] == pytest.approx(2.0)
    assert features.loc[0, "native_eyes_open_target_roi_minus_homologous_roi_alpha_mu_power"] == pytest.approx(2.0)
    assert features.loc[0, "native_eyes_open_log_target_minus_log_homologous_alpha_mu_power"] == pytest.approx(0.693147, abs=1e-6)
    assert features.loc[0, "native_eyes_open_target_homologous_alpha_mu_coherence"] == pytest.approx(0.61)
    assert features.loc[0, "native_eyes_open_target_homologous_alpha_mu_wpli"] == pytest.approx(0.62)
    assert features.loc[0, "native_eyes_open_target_to_midline_alpha_mu_coherence"] == pytest.approx(0.71)
    assert features.loc[0, "native_eyes_open_target_to_frontal_alpha_mu_coherence"] == pytest.approx(0.81)
    assert features.loc[0, "native_eyes_open_target_to_parietal_alpha_mu_coherence"] == pytest.approx(0.91)
    assert features.loc[0, "native_ec_minus_eo_target_alpha_mu_power"] == pytest.approx(4.0)
    assert features.loc[0, "native_ec_minus_eo_target_homologous_alpha_mu_coherence"] == pytest.approx(0.05)

