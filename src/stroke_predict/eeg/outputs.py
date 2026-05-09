from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


FORBIDDEN_PUBLIC_COLUMNS = {
    "subject_name",
    "set_path",
    "fdt_path",
    "file_path",
    "_source_key",
    "姓名",
    "姓名写法",
    "EEG文件夹",
}


def assert_public_eeg_output(frame: pd.DataFrame) -> None:
    leaked = sorted(FORBIDDEN_PUBLIC_COLUMNS.intersection(frame.columns))
    if leaked:
        raise ValueError(f"EEG public output contains forbidden columns: {leaked}")
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        values = frame[column].dropna().astype(str)
        if values.str.contains(r"[A-Za-z]:[\\/]|\.set\b|\.fdt\b", regex=True).any():
            raise ValueError(f"EEG public output contains path-like values in column {column}")


def write_record_index_output(record_index: pd.DataFrame, output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    assert_public_eeg_output(record_index)
    path = output / "eeg_record_index.csv"
    record_index.to_csv(path, index=False)
    return path


def write_qc_outputs(
    *,
    record_index: pd.DataFrame,
    qc_summary: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    channel_report = build_channel_order_report(qc_summary)
    for frame in (record_index, qc_summary, channel_report):
        assert_public_eeg_output(frame)
    paths = {
        "record_index": output / "eeg_record_index.csv",
        "qc_summary": output / "eeg_qc_summary.csv",
        "channel_order_report": output / "channel_order_report.csv",
    }
    record_index.to_csv(paths["record_index"], index=False)
    qc_summary.to_csv(paths["qc_summary"], index=False)
    channel_report.to_csv(paths["channel_order_report"], index=False)
    return paths


def build_channel_order_report(qc_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if "channel_order_hash" not in qc_summary.columns:
        return pd.DataFrame(
            columns=[
                "channel_order_hash",
                "n_records",
                "n_channels",
                "example_subject_id",
                "example_record_id",
            ]
        )
    valid = qc_summary.dropna(subset=["channel_order_hash"])
    for hash_value, group in valid.groupby("channel_order_hash", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "channel_order_hash": hash_value,
                "n_records": int(len(group)),
                "n_channels": int(first["n_channels"]) if pd.notna(first["n_channels"]) else None,
                "example_subject_id": first["subject_id"],
                "example_record_id": first["record_id"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "channel_order_hash",
            "n_records",
            "n_channels",
            "example_subject_id",
            "example_record_id",
        ],
    )
