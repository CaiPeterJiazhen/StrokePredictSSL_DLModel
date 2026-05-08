from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd


DEFAULT_SHEETS = {
    "summary": "02_统计汇总",
    "clinical_overview": "01_患者数据总览",
    "clinical_raw": "03_临床量表原始",
    "preprocessed_summary": "06_预处理静息态阶段汇总",
    "preprocessed_files": "07_预处理静息态文件明细",
}

REQUIRED_COLUMNS = {
    "clinical_overview": ("患者编号", "治疗前FMA", "治疗后FMA", "FMA前后完整"),
    "clinical_raw": ("患者编号", "治疗前FMA", "治疗后FMA", "FMA前后完整"),
    "preprocessed_files": ("source", "subject_id", "stage", "condition"),
}


@dataclass(frozen=True)
class StatusWorkbook:
    summary: pd.DataFrame
    clinical_overview: pd.DataFrame
    clinical_raw: pd.DataFrame
    preprocessed_summary: pd.DataFrame
    preprocessed_files: pd.DataFrame


def read_status_workbook(
    path: str | Path,
    sheets: Mapping[str, str] | None = None,
) -> StatusWorkbook:
    sheet_names = dict(DEFAULT_SHEETS)
    if sheets is not None:
        sheet_names.update(sheets)

    frames = {
        key: pd.read_excel(path, sheet_name=sheet_name)
        for key, sheet_name in sheet_names.items()
    }
    for key, required in REQUIRED_COLUMNS.items():
        require_columns(frames[key], required, frame_name=key)

    return StatusWorkbook(
        summary=frames["summary"],
        clinical_overview=frames["clinical_overview"],
        clinical_raw=frames["clinical_raw"],
        preprocessed_summary=frames["preprocessed_summary"],
        preprocessed_files=frames["preprocessed_files"],
    )


def require_columns(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    frame_name: str = "frame",
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} missing columns: {missing}")
