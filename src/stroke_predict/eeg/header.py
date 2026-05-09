from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EEGHeader:
    n_channels: int | None
    sfreq: float | None
    pnts: int | None
    trials: int | None
    channel_labels: list[str]
    datfile: str | None = None

    @property
    def duration_sec(self) -> float | None:
        if self.sfreq in (None, 0) or self.pnts is None or self.trials is None:
            return None
        return float(self.pnts * self.trials) / float(self.sfreq)


def read_eeglab_set_header(path: str | Path) -> EEGHeader:
    import scipy.io

    mat = scipy.io.loadmat(path, simplify_cells=True, verify_compressed_data_integrity=False)
    return EEGHeader(
        n_channels=_optional_int(mat.get("nbchan")),
        sfreq=_optional_float(mat.get("srate")),
        pnts=_optional_int(mat.get("pnts")),
        trials=_optional_int(mat.get("trials")),
        channel_labels=_extract_channel_labels(mat.get("chanlocs")),
        datfile=_optional_str(mat.get("datfile") or mat.get("data")),
    )


def _extract_channel_labels(chanlocs: Any) -> list[str]:
    if chanlocs is None:
        return []
    if isinstance(chanlocs, dict):
        return [_optional_str(chanlocs.get("labels")) or ""]
    labels: list[str] = []
    try:
        iterator = iter(chanlocs)
    except TypeError:
        return labels
    for item in iterator:
        if isinstance(item, dict):
            labels.append(_optional_str(item.get("labels")) or "")
    return labels


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None
