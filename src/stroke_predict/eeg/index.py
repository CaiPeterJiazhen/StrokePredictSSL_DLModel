from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stroke_predict.cohort.ids import build_subject_id_map, normalize_source_key
from stroke_predict.io.excel_status import StatusWorkbook


PUBLIC_INDEX_COLUMNS = [
    "record_id",
    "subject_id",
    "source",
    "stage",
    "condition",
    "record_index",
    "set_exists",
    "fdt_exists",
]


def build_eeg_record_index(status: StatusWorkbook) -> pd.DataFrame:
    rows, _private = _build_eeg_rows(status)
    return pd.DataFrame(rows, columns=PUBLIC_INDEX_COLUMNS)


def build_eeg_private_records(status: StatusWorkbook) -> list[dict[str, Any]]:
    _rows, private = _build_eeg_rows(status)
    return private


def _build_eeg_rows(status: StatusWorkbook) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    clinical = status.clinical_overview.copy()
    files = status.preprocessed_files.copy()
    clinical["_source_key"] = clinical["患者编号"].map(normalize_source_key)
    files["_source_key"] = files["subject_id"].map(normalize_source_key)

    stroke_keys = sorted(
        set(clinical["_source_key"].dropna()).union(
            set(files.loc[files["source"].eq("stroke"), "_source_key"].dropna())
        )
    )
    healthy_keys = sorted(set(files.loc[files["source"].eq("healthy"), "_source_key"].dropna()))
    stroke_ids = build_subject_id_map(stroke_keys, source="stroke", prefix="STK")
    healthy_ids = build_subject_id_map(healthy_keys, source="healthy", prefix="HC")

    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    counters: dict[tuple[str, str, str, str], int] = {}
    sorted_files = files.sort_values(["source", "_source_key", "stage", "condition"]).reset_index(drop=True)

    for row in sorted_files.to_dict("records"):
        source = str(row.get("source", "")).strip()
        source_key = normalize_source_key(row.get("_source_key"))
        subject_id = stroke_ids[source_key] if source == "stroke" else healthy_ids[source_key]
        stage = normalize_stage(row.get("stage"))
        condition = normalize_condition(row.get("condition"))
        counter_key = (subject_id, source, stage, condition)
        counters[counter_key] = counters.get(counter_key, 0) + 1
        record_index = counters[counter_key]
        record_id = f"{subject_id}_{stage}_{condition}_{record_index:02d}"
        public_rows.append(
            {
                "record_id": record_id,
                "subject_id": subject_id,
                "source": source,
                "stage": stage,
                "condition": condition,
                "record_index": record_index,
                "set_exists": _path_exists(row.get("set_path")),
                "fdt_exists": _path_exists(row.get("fdt_path")),
            }
        )
        private_rows.append(
            {
                "record_id": record_id,
                "set_path": row.get("set_path"),
                "fdt_path": row.get("fdt_path"),
            }
        )

    combined = sorted(
        zip(public_rows, private_rows, strict=True),
        key=lambda pair: (
            str(pair[0]["subject_id"]),
            str(pair[0]["stage"]),
            str(pair[0]["condition"]),
            str(pair[0]["record_id"]),
        ),
    )
    if not combined:
        return [], []
    public_sorted, private_sorted = zip(*combined, strict=True)
    return list(public_sorted), list(private_sorted)


def normalize_stage(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    mapping = {
        "基线": "baseline",
        "即时": "immediate",
        "阶段": "mid",
        "最终": "final",
    }
    return mapping.get(text, text)


def normalize_condition(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower().replace(" ", "_")
    mapping = {
        "任务_1": "eyes_open",
        "任务1": "eyes_open",
        "任务_2": "eyes_closed",
        "任务2": "eyes_closed",
    }
    return mapping.get(text, text)


def _path_exists(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return Path(str(value)).exists()
