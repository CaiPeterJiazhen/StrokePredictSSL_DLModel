from __future__ import annotations

import pandas as pd

from stroke_predict.splits import build_outer_folds


def test_ssl_registry_excludes_outer_test_all_stage_records() -> None:
    cohort = pd.DataFrame(
        [
            {"subject_id": "STK-001", "source": "stroke", "role": "supervised_main", "label_primary": "Poor"},
            {"subject_id": "STK-002", "source": "stroke", "role": "supervised_main", "label_primary": "Good"},
            {"subject_id": "STK-003", "source": "stroke", "role": "supervised_main", "label_primary": "Poor"},
            {"subject_id": "STK-004", "source": "stroke", "role": "supervised_main", "label_primary": "Good"},
            {"subject_id": "STK-010", "source": "stroke", "role": "ssl_only_stroke", "label_primary": "missing"},
            {"subject_id": "HC-001", "source": "healthy", "role": "healthy_ssl", "label_primary": "missing"},
        ]
    )
    qc = pd.DataFrame(
        [
            {
                "record_id": "STK-001_baseline_eyes_open_01",
                "subject_id": "STK-001",
                "source": "stroke",
                "stage": "baseline",
                "condition": "eyes_open",
                "passes_qc": True,
            },
            {
                "record_id": "STK-001_final_eyes_closed_01",
                "subject_id": "STK-001",
                "source": "stroke",
                "stage": "final",
                "condition": "eyes_closed",
                "passes_qc": True,
            },
            {
                "record_id": "STK-002_baseline_eyes_open_01",
                "subject_id": "STK-002",
                "source": "stroke",
                "stage": "baseline",
                "condition": "eyes_open",
                "passes_qc": True,
            },
            {
                "record_id": "STK-010_baseline_eyes_open_01",
                "subject_id": "STK-010",
                "source": "stroke",
                "stage": "baseline",
                "condition": "eyes_open",
                "passes_qc": True,
            },
            {
                "record_id": "HC-001_baseline_eyes_open_01",
                "subject_id": "HC-001",
                "source": "healthy",
                "stage": "baseline",
                "condition": "eyes_open",
                "passes_qc": True,
            },
        ]
    )
    features = pd.DataFrame({"subject_id": ["STK-001", "STK-002", "STK-003", "STK-004"]})

    result = build_outer_folds(cohort, qc, features, inner_k=2)
    registry = result["registries"][0]

    assert registry["test_subject"] == "STK-001"
    assert "STK-001" not in registry["ssl_train_subjects"]
    assert "STK-001" in registry["ssl_excluded_subjects"]
    assert {record["subject_id"] for record in registry["ssl_train_records"]} == {"STK-002", "STK-010", "HC-001"}
    assert {record["stage"] for record in registry["ssl_excluded_records"]} == {"baseline", "final"}
    for record in registry["ssl_train_records"] + registry["ssl_excluded_records"]:
        assert set(record) == {"record_id", "subject_id", "source", "stage", "condition"}
        assert ".set" not in str(record)
        assert ".fdt" not in str(record)
