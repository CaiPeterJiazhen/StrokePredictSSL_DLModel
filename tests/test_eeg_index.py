from __future__ import annotations

from pathlib import Path

import pandas as pd

from stroke_predict.eeg.index import build_eeg_private_records, build_eeg_record_index
from stroke_predict.io.excel_status import StatusWorkbook


PRIVATE_COLUMNS = {"subject_name", "set_path", "fdt_path", "file_path", "_source_key", "姓名"}


def _status_workbook() -> StatusWorkbook:
    clinical = pd.DataFrame(
        {
            "患者编号": ["p01"],
            "姓名": ["Private Name"],
            "治疗前FMA": [40],
            "治疗后FMA": [45],
            "FMA前后完整": [True],
        }
    )
    files = pd.DataFrame(
        {
            "source": ["stroke", "stroke", "healthy"],
            "subject_id": ["p01", "p01", "h01"],
            "subject_name": ["Private Name", "Private Name", "Healthy Name"],
            "stage": ["基线", "baseline", "baseline"],
            "condition": ["任务 1", "eyes_closed", "eyes_open"],
            "set_path": ["private-a.set", "private-b.set", "private-h.set"],
            "fdt_path": ["private-a.fdt", "private-b.fdt", "private-h.fdt"],
        }
    )
    return StatusWorkbook(
        summary=pd.DataFrame(),
        clinical_overview=clinical,
        clinical_raw=clinical.copy(),
        preprocessed_summary=pd.DataFrame(),
        preprocessed_files=files,
    )


def test_builds_deidentified_record_index() -> None:
    index = build_eeg_record_index(_status_workbook())

    assert list(index["subject_id"]) == ["HC-001", "STK-001", "STK-001"]
    assert set(index["stage"]) == {"baseline"}
    assert set(index["condition"]) == {"eyes_open", "eyes_closed"}
    assert index["record_id"].is_unique
    assert PRIVATE_COLUMNS.isdisjoint(index.columns)
    assert set(index.columns) == {
        "record_id",
        "subject_id",
        "source",
        "stage",
        "condition",
        "record_index",
        "set_exists",
        "fdt_exists",
    }


def test_record_index_keeps_file_existence_flags(tmp_path: Path) -> None:
    status = _status_workbook()
    set_file = tmp_path / "x.set"
    fdt_file = tmp_path / "x.fdt"
    set_file.write_text("placeholder", encoding="utf-8")
    fdt_file.write_bytes(b"")
    status.preprocessed_files.loc[0, "set_path"] = str(set_file)
    status.preprocessed_files.loc[0, "fdt_path"] = str(fdt_file)

    index = build_eeg_record_index(status)

    row = index.loc[index["record_id"].str.contains("eyes_open") & index["subject_id"].eq("STK-001")].iloc[0]
    assert bool(row["set_exists"]) is True
    assert bool(row["fdt_exists"]) is True


def test_private_records_keep_paths_out_of_public_index() -> None:
    status = _status_workbook()

    private_records = build_eeg_private_records(status)
    public_index = build_eeg_record_index(status)

    assert len(private_records) == len(public_index)
    assert {"set_path", "fdt_path"}.issubset(private_records[0])
    assert PRIVATE_COLUMNS.isdisjoint(public_index.columns)
