from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stroke_predict.config import load_yaml_mapping


DEFAULT_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha_mu": (8.0, 13.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "low_gamma_optional": (30.0, 45.0),
}

DEFAULT_ROIS = {
    "left_motor": ["FC3", "FC1", "C3", "C1", "CP3", "CP1"],
    "right_motor": ["FC4", "FC2", "C4", "C2", "CP4", "CP2"],
    "midline_motor": ["FCz", "Cz", "CPz"],
    "left_frontal": ["F3", "F5", "F7", "FC3", "FC5"],
    "right_frontal": ["F4", "F6", "F8", "FC4", "FC6"],
    "left_parietal": ["P3", "P5", "P7", "CP3", "CP5"],
    "right_parietal": ["P4", "P6", "P8", "CP4", "CP6"],
    "occipital": ["O1", "Oz", "O2", "PO3", "POz", "PO4"],
}


@dataclass(frozen=True)
class FeatureConfig:
    path: Path
    project_config_path: Path
    eeg_config_path: Path
    raw: dict[str, Any]

    @property
    def freq_min_hz(self) -> float:
        return float(self.raw.get("psd", {}).get("freq_min_hz", 0.5))

    @property
    def freq_max_hz(self) -> float:
        return float(self.raw.get("psd", {}).get("freq_max_hz", 45.0))

    @property
    def freq_resolution_hz(self) -> float:
        return float(self.raw.get("psd", {}).get("freq_resolution_hz", 0.5))

    @property
    def log_transform(self) -> bool:
        return bool(self.raw.get("psd", {}).get("log_transform", True))

    @property
    def bands(self) -> dict[str, tuple[float, float]]:
        configured = self.raw.get("bands", DEFAULT_BANDS)
        return {str(name): (float(bounds[0]), float(bounds[1])) for name, bounds in configured.items()}

    @property
    def rois(self) -> dict[str, list[str]]:
        configured = self.raw.get("connectivity", {}).get("rois", DEFAULT_ROIS)
        return {str(name): [str(channel) for channel in channels] for name, channels in configured.items()}

    @property
    def connectivity_methods(self) -> list[str]:
        return [str(method) for method in self.raw.get("connectivity", {}).get("methods", ["coherence", "wpli"])]

    @property
    def views(self) -> list[str]:
        return [str(view) for view in self.raw.get("lesion_normalization", {}).get("views", ["native", "lesion_normalized"])]

    @property
    def channel_pair_map(self) -> dict[str, str]:
        from stroke_predict.features.channels import DEFAULT_CHANNEL_PAIR_MAP

        configured = self.raw.get("channel_pair_map", DEFAULT_CHANNEL_PAIR_MAP)
        pairs: dict[str, str] = {}
        for left, right in configured.items():
            pairs[str(left)] = str(right)
            pairs[str(right)] = str(left)
        return pairs


def load_feature_config(path: str | Path) -> FeatureConfig:
    config_path = Path(path).resolve()
    raw = load_yaml_mapping(config_path)
    project_path = _resolve(config_path, str(raw.get("project_config", "project.yaml")))
    eeg_path = _resolve(config_path, str(raw.get("eeg_config", "eeg.yaml")))
    return FeatureConfig(path=config_path, project_config_path=project_path, eeg_config_path=eeg_path, raw=raw)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()

