from __future__ import annotations

from pathlib import Path

from stroke_predict.eeg.config import load_eeg_config


def test_loads_eeg_config_defaults(tmp_path: Path) -> None:
    project = tmp_path / "project.yaml"
    project.write_text(
        "paths_config: paths.yaml\n"
        "sheets:\n"
        "  summary: S\n"
        "privacy:\n"
        "  pii_columns: []\n",
        encoding="utf-8",
    )
    eeg = tmp_path / "eeg.yaml"
    eeg.write_text("project_config: project.yaml\n", encoding="utf-8")

    config = load_eeg_config(eeg)

    assert config.path == eeg.resolve()
    assert config.project_config_path == project.resolve()
    assert config.required_channels == 62
    assert config.allowed_sampling_rate_hz == 250
    assert config.min_duration_sec_main == 60
    assert config.min_duration_sec_ssl == 30
    assert config.window_length_sec == 4.0
    assert config.window_overlap == 0.5
    assert config.min_valid_windows_per_condition == 10


def test_loads_eeg_config_overrides(tmp_path: Path) -> None:
    project = tmp_path / "custom_project.yaml"
    project.write_text("paths_config: paths.yaml\n", encoding="utf-8")
    eeg = tmp_path / "eeg.yaml"
    eeg.write_text(
        "project_config: custom_project.yaml\n"
        "qc:\n"
        "  min_duration_sec_main: 90\n"
        "  min_duration_sec_ssl: 45\n"
        "  allowed_sampling_rate_hz: 500\n"
        "  required_channels: 64\n"
        "window:\n"
        "  length_sec: 8\n"
        "  overlap: 0.25\n"
        "  min_valid_windows_per_condition: 5\n",
        encoding="utf-8",
    )

    config = load_eeg_config(eeg)

    assert config.project_config_path == project.resolve()
    assert config.required_channels == 64
    assert config.allowed_sampling_rate_hz == 500
    assert config.min_duration_sec_main == 90
    assert config.min_duration_sec_ssl == 45
    assert config.window_length_sec == 8
    assert config.window_overlap == 0.25
    assert config.min_valid_windows_per_condition == 5
