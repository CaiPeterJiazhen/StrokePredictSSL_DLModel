from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stroke_predict.config import ProjectConfig
from stroke_predict.eeg.header import EEGHeader, read_eeglab_set_header
from stroke_predict.eeg.index import build_eeg_private_records, build_eeg_record_index
from stroke_predict.io.excel_status import read_status_workbook


def read_phase_inputs(project: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    cohort_path = project.output_dir / "cohort" / "cohort_master.csv"
    qc_path = project.output_dir / "qc" / "eeg_qc_summary.csv"
    index_path = project.output_dir / "qc" / "eeg_record_index.csv"
    for path in (cohort_path, qc_path, index_path):
        if not path.exists():
            raise FileNotFoundError(f"Required Phase 3 input missing: {path}")
    cohort = pd.read_csv(cohort_path)
    qc = pd.read_csv(qc_path)
    record_index = pd.read_csv(index_path)
    status = read_status_workbook(
        project.workbook_path,
        sheets={
            "summary": project.sheet("summary"),
            "clinical_overview": project.sheet("clinical_overview"),
            "clinical_raw": project.sheet("clinical_raw"),
            "preprocessed_summary": project.sheet("preprocessed_summary"),
            "preprocessed_files": project.sheet("preprocessed_files"),
        },
    )
    rebuilt_index = build_eeg_record_index(status)
    private_records = build_eeg_private_records(status)
    private_by_record = {
        str(public["record_id"]): private
        for public, private in zip(rebuilt_index.to_dict("records"), private_records, strict=True)
    }
    return cohort, qc, record_index, private_by_record


def supervised_subjects(cohort: pd.DataFrame) -> pd.DataFrame:
    subjects = cohort.loc[cohort["role"].eq("supervised_main")].copy()
    subjects = subjects.sort_values("subject_id").reset_index(drop=True)
    if subjects.empty:
        raise ValueError("No supervised_main subjects found")
    return subjects


def select_baseline_record(qc: pd.DataFrame, subject_id: str, condition: str) -> str:
    rows = qc[
        qc["subject_id"].eq(subject_id)
        & qc["stage"].eq("baseline")
        & qc["condition"].eq(condition)
        & qc["passes_qc"].astype(bool)
    ].sort_values("record_id")
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one passing baseline {condition} record for {subject_id}, found {len(rows)}")
    return str(rows.iloc[0]["record_id"])


def assert_single_channel_order(qc: pd.DataFrame) -> None:
    hashes = qc.loc[qc["passes_qc"].astype(bool), "channel_order_hash"].dropna().unique()
    if len(hashes) != 1:
        raise ValueError(f"Expected one passing channel_order_hash, found {len(hashes)}")


def read_record_data(record_id: str, private_by_record: dict[str, dict[str, Any]]) -> tuple[np.ndarray, EEGHeader]:
    private = private_by_record[record_id]
    set_path = Path(str(private["set_path"]))
    fdt_path = Path(str(private["fdt_path"]))
    header = read_eeglab_set_header(set_path)
    if header.n_channels is None or header.pnts is None or header.trials is None:
        raise ValueError(f"Header missing dimensions for {record_id}")
    samples = int(header.pnts) * int(header.trials)
    expected = int(header.n_channels) * samples
    data = _read_fdt(fdt_path, expected)
    matrix = data.reshape((samples, int(header.n_channels))).T
    return matrix, header


def _read_fdt(path: Path, expected: int) -> np.ndarray:
    data = np.fromfile(path, dtype="<f4")
    if data.size == expected:
        return data
    data64 = np.fromfile(path, dtype="<f8")
    if data64.size == expected:
        return data64.astype(float)
    raise ValueError(f"Unexpected FDT sample count for private EEG record: expected {expected}, got {data.size}")

