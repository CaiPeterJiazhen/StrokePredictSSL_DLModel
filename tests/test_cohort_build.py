from __future__ import annotations

import pandas as pd

from stroke_predict.cohort.build import build_cohort_tables
from stroke_predict.io.excel_status import StatusWorkbook


PRIVATE_COLUMNS = {"姓名", "subject_name", "set_path", "fdt_path", "_source_key"}


def _status_workbook() -> StatusWorkbook:
    clinical = pd.DataFrame(
        {
            "患者编号": ["sub01", "sub02", "sub03", "sub04"],
            "姓名": ["甲", "乙", "丙", "丁"],
            "年龄": ["60岁", "61岁", "62岁", "63岁"],
            "性别": ["男", "女", "男", "女"],
            "患侧": ["右手", "左手", "右手", "左手"],
            "治疗前FMA": [40, 66, 50, 40],
            "治疗后FMA": [45, 66, None, 44],
            "FMA前后完整": [True, True, False, True],
            "治疗前MBI": [80, 90, None, 70],
            "治疗后MBI": [90, 90, None, 75],
            "MMSE": [28, 29, 27, 26],
        }
    )
    preprocessed = pd.DataFrame(
        {
            "source": ["stroke", "stroke", "stroke", "stroke", "healthy", "healthy"],
            "subject_id": ["sub01", "sub01", "sub02", "sub03", "sub001", "sub001"],
            "subject_name": ["甲", "甲", "乙", "丙", "健康甲", "健康甲"],
            "stage": ["baseline", "baseline", "baseline", "baseline", "baseline", "baseline"],
            "condition": [
                "eyes_open",
                "eyes_closed",
                "eyes_open",
                "eyes_open",
                "eyes_open",
                "eyes_closed",
            ],
            "set_path": ["F:/private/file.set"] * 6,
            "fdt_path": ["F:/private/file.fdt"] * 6,
        }
    )
    return StatusWorkbook(
        summary=pd.DataFrame(),
        clinical_overview=clinical,
        clinical_raw=clinical.copy(),
        preprocessed_summary=pd.DataFrame(),
        preprocessed_files=preprocessed,
    )


def test_builds_deidentified_cohort_and_roles() -> None:
    tables = build_cohort_tables(
        _status_workbook(),
        pii_columns=["姓名", "subject_name", "set_path", "fdt_path", "_source_key"],
    )

    cohort = tables.cohort_master.sort_values("subject_id").reset_index(drop=True)
    roles = dict(zip(cohort["subject_id"], cohort["role"]))

    assert roles["HC-001"] == "healthy_ssl"
    assert roles["STK-001"] == "supervised_main"
    assert roles["STK-002"] == "ceiling_exclude"
    assert roles["STK-003"] == "ssl_only_stroke"
    assert roles["STK-004"] == "excluded_no_eeg"
    assert PRIVATE_COLUMNS.isdisjoint(cohort.columns)
    assert PRIVATE_COLUMNS.isdisjoint(tables.label_audit.columns)


def test_label_audit_contains_required_fields_and_allowed_labels() -> None:
    tables = build_cohort_tables(
        _status_workbook(),
        pii_columns=["姓名", "subject_name", "set_path", "fdt_path", "_source_key"],
    )

    audit = tables.label_audit
    expected = {
        "subject_id",
        "baseline_fma",
        "post_fma",
        "delta_fma",
        "possible_recovery",
        "recovery_ratio",
        "label_primary",
        "label_delta5_all",
        "label_prop70",
        "label_low_baseline_only",
        "label_reason",
    }
    assert expected.issubset(set(audit.columns))
    assert set(audit["label_primary"]).issubset({"Good", "Poor", "ceiling_exclude", "missing"})
