from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stroke_predict.config import load_yaml_mapping


@dataclass(frozen=True)
class EEGConfig:
    path: Path
    project_config_path: Path
    raw: dict[str, Any]

    @property
    def required_channels(self) -> int:
        return int(self.raw.get("qc", {}).get("required_channels", 62))

    @property
    def allowed_sampling_rate_hz(self) -> float:
        return float(self.raw.get("qc", {}).get("allowed_sampling_rate_hz", 250))

    @property
    def min_duration_sec_main(self) -> float:
        return float(self.raw.get("qc", {}).get("min_duration_sec_main", 60))

    @property
    def min_duration_sec_ssl(self) -> float:
        return float(self.raw.get("qc", {}).get("min_duration_sec_ssl", 30))

    @property
    def window_length_sec(self) -> float:
        return float(self.raw.get("window", {}).get("length_sec", 4))

    @property
    def window_overlap(self) -> float:
        return float(self.raw.get("window", {}).get("overlap", 0.5))

    @property
    def min_valid_windows_per_condition(self) -> int:
        return int(self.raw.get("window", {}).get("min_valid_windows_per_condition", 10))


def load_eeg_config(path: str | Path) -> EEGConfig:
    config_path = Path(path).resolve()
    raw = load_yaml_mapping(config_path)
    project_name = str(raw.get("project_config", "project.yaml"))
    project_path = Path(project_name)
    if not project_path.is_absolute():
        project_path = (config_path.parent / project_path).resolve()
    return EEGConfig(path=config_path, project_config_path=project_path, raw=raw)
