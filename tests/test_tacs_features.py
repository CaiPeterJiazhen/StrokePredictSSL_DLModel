import pandas as pd

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

