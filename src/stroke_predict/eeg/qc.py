from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stroke_predict.eeg.config import EEGConfig
from stroke_predict.eeg.header import EEGHeader, read_eeglab_set_header


QC_COLUMNS = [
    "record_id",
    "subject_id",
    "source",
    "stage",
    "condition",
    "exists",
    "readable",
    "n_channels",
    "sfreq",
    "channel_order_hash",
    "duration_sec",
    "n_valid_samples",
    "n_valid_windows_2s",
    "n_valid_windows_4s",
    "n_valid_windows_8s",
    "bad_channel_count",
    "artifact_ratio_if_available",
    "passes_qc",
    "qc_reason",
]


def count_windows(duration_sec: float | None, length_sec: float, overlap: float) -> int:
    if duration_sec is None or duration_sec < length_sec:
        return 0
    step = length_sec * (1.0 - overlap)
    if step <= 0:
        raise ValueError("window overlap must be less than 1.0")
    return int((duration_sec - length_sec) // step) + 1


def channel_order_hash(labels: Iterable[str]) -> str:
    normalized = "|".join(str(label).strip().upper() for label in labels)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def evaluate_qc(
    header: EEGHeader | None,
    *,
    source: str,
    stage: str,
    condition: str,
    set_exists: bool,
    fdt_exists: bool,
    readable: bool,
    config: EEGConfig,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not set_exists:
        reasons.append("missing_set")
    if not fdt_exists:
        reasons.append("missing_fdt")
    if not readable or header is None:
        reasons.append("unreadable_set")

    n_channels = header.n_channels if header else None
    sfreq = header.sfreq if header else None
    duration = header.duration_sec if header else None
    if n_channels != config.required_channels:
        reasons.append("bad_channel_count")
    if sfreq != config.allowed_sampling_rate_hz:
        reasons.append("bad_sampling_rate")

    min_duration = (
        config.min_duration_sec_main
        if source == "stroke" and stage == "baseline" and condition in {"eyes_open", "eyes_closed"}
        else config.min_duration_sec_ssl
    )
    if duration is None or duration < min_duration:
        reasons.append("short_duration")
    windows_4s = count_windows(duration, config.window_length_sec, config.window_overlap)
    if windows_4s < config.min_valid_windows_per_condition:
        reasons.append("too_few_4s_windows")

    labels = header.channel_labels if header else []
    pnts = header.pnts if header and header.pnts is not None else None
    trials = header.trials if header and header.trials is not None else None
    return {
        "exists": bool(set_exists and fdt_exists),
        "readable": bool(readable),
        "n_channels": n_channels,
        "sfreq": sfreq,
        "channel_order_hash": channel_order_hash(labels) if labels else None,
        "duration_sec": duration,
        "n_valid_samples": pnts * trials if pnts is not None and trials is not None else None,
        "n_valid_windows_2s": count_windows(duration, 2, config.window_overlap),
        "n_valid_windows_4s": windows_4s,
        "n_valid_windows_8s": count_windows(duration, 8, config.window_overlap),
        "bad_channel_count": 0,
        "artifact_ratio_if_available": None,
        "passes_qc": not reasons,
        "qc_reason": "pass" if not reasons else ";".join(dict.fromkeys(reasons)),
    }


def run_qc(
    public_index: pd.DataFrame,
    *,
    private_records: list[dict[str, Any]],
    config: EEGConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for public_row, private in zip(public_index.to_dict("records"), private_records, strict=True):
        set_path = private.get("set_path")
        set_exists = bool(public_row.get("set_exists", False))
        fdt_exists = bool(public_row.get("fdt_exists", False))
        header = None
        readable = False
        if set_exists and set_path:
            try:
                header = read_eeglab_set_header(Path(str(set_path)))
                readable = True
            except Exception:
                readable = False
        qc = evaluate_qc(
            header,
            source=str(public_row["source"]),
            stage=str(public_row["stage"]),
            condition=str(public_row["condition"]),
            set_exists=set_exists,
            fdt_exists=fdt_exists,
            readable=readable,
            config=config,
        )
        public = {
            key: public_row[key]
            for key in ["record_id", "subject_id", "source", "stage", "condition"]
        }
        rows.append({**public, **qc})
    return pd.DataFrame(rows, columns=QC_COLUMNS)
