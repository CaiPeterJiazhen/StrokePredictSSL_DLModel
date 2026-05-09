from pathlib import Path

from stroke_predict.features.config import load_feature_config


def test_load_feature_config_defaults(tmp_path: Path) -> None:
    project = tmp_path / "project.yaml"
    eeg = tmp_path / "eeg.yaml"
    features = tmp_path / "features.yaml"
    project.write_text("paths_config: paths.yaml\n", encoding="utf-8")
    eeg.write_text("project_config: project.yaml\n", encoding="utf-8")
    features.write_text("project_config: project.yaml\neeg_config: eeg.yaml\n", encoding="utf-8")

    config = load_feature_config(features)

    assert config.path == features.resolve()
    assert config.project_config_path == project.resolve()
    assert config.eeg_config_path == eeg.resolve()
    assert config.freq_min_hz == 0.5
    assert config.freq_max_hz == 45.0
    assert config.freq_resolution_hz == 0.5
    assert config.views == ["native", "lesion_normalized"]
    assert "alpha_mu" in config.bands

