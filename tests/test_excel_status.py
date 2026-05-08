from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stroke_predict.io.excel_status import read_status_workbook


def _write_status_workbook(path: Path, *, omit_column: str | None = None) -> None:
    clinical = pd.DataFrame(
        {
            "患者编号": ["sub01"],
            "治疗前FMA": [40],
            "治疗后FMA": [45],
            "FMA前后完整": [True],
        }
    )
    files = pd.DataFrame(
        {
            "source": ["stroke"],
            "subject_id": ["sub01"],
            "stage": ["baseline"],
            "condition": ["eyes_open"],
        }
    )
    if omit_column:
        if omit_column in clinical:
            clinical = clinical.drop(columns=[omit_column])
        if omit_column in files:
            files = files.drop(columns=[omit_column])

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"统计项": ["临床表患者数"], "数值": [1]}).to_excel(
            writer, sheet_name="02_统计汇总", index=False
        )
        clinical.to_excel(writer, sheet_name="01_患者数据总览", index=False)
        clinical.to_excel(writer, sheet_name="03_临床量表原始", index=False)
        pd.DataFrame({"来源": ["stroke"], "受试者编号": ["sub01"]}).to_excel(
            writer, sheet_name="06_预处理静息态阶段汇总", index=False
        )
        files.to_excel(writer, sheet_name="07_预处理静息态文件明细", index=False)


def test_reads_status_workbook_clinical_and_preprocessed_frames(tmp_path: Path) -> None:
    workbook = tmp_path / "status.xlsx"
    _write_status_workbook(workbook)

    status = read_status_workbook(workbook)

    assert list(status.clinical_overview["患者编号"]) == ["sub01"]
    assert list(status.clinical_raw["治疗前FMA"]) == [40]
    assert list(status.preprocessed_files["subject_id"]) == ["sub01"]


def test_missing_required_columns_raise_value_error(tmp_path: Path) -> None:
    workbook = tmp_path / "status.xlsx"
    _write_status_workbook(workbook, omit_column="condition")

    with pytest.raises(ValueError, match="missing columns"):
        read_status_workbook(workbook)
