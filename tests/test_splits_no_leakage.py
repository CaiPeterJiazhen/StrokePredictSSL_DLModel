from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stroke_predict.splits import build_outer_folds, write_fold_outputs


def _cohort() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subject_id": "STK-001", "source": "stroke", "role": "supervised_main", "label_primary": "Poor"},
            {"subject_id": "STK-002", "source": "stroke", "role": "supervised_main", "label_primary": "Good"},
            {"subject_id": "STK-003", "source": "stroke", "role": "supervised_main", "label_primary": "Poor"},
            {"subject_id": "STK-004", "source": "stroke", "role": "supervised_main", "label_primary": "Good"},
            {"subject_id": "STK-005", "source": "stroke", "role": "supervised_main", "label_primary": "Poor"},
            {"subject_id": "STK-006", "source": "stroke", "role": "supervised_main", "label_primary": "Good"},
            {"subject_id": "STK-020", "source": "stroke", "role": "ssl_only_stroke", "label_primary": "missing"},
            {"subject_id": "HC-001", "source": "healthy", "role": "healthy_ssl", "label_primary": "missing"},
        ]
    )


def _qc() -> pd.DataFrame:
    rows = []
    for subject_id, source in [
        ("STK-001", "stroke"),
        ("STK-002", "stroke"),
        ("STK-003", "stroke"),
        ("STK-004", "stroke"),
        ("STK-005", "stroke"),
        ("STK-006", "stroke"),
        ("STK-020", "stroke"),
        ("HC-001", "healthy"),
    ]:
        for stage in ("baseline", "final"):
            rows.append(
                {
                    "record_id": f"{subject_id}_{stage}_eyes_open_01",
                    "subject_id": subject_id,
                    "source": source,
                    "stage": stage,
                    "condition": "eyes_open",
                    "passes_qc": True,
                }
            )
    return pd.DataFrame(rows)


def test_lopo_outer_and_inner_splits_are_patient_level(tmp_path: Path) -> None:
    features = pd.DataFrame({"subject_id": [f"STK-{index:03d}" for index in range(1, 7)]})

    result = build_outer_folds(_cohort(), _qc(), features, inner_k=3)

    assert [fold["test_subject"] for fold in result["folds"]] == [f"STK-{index:03d}" for index in range(1, 7)]
    assert sorted(fold["test_subject"] for fold in result["folds"]) == result["supervised_subjects"]
    for fold in result["registries"]:
        test_subject = fold["test_subject"]
        train_subjects = set(fold["supervised_train_subjects"])
        assert test_subject not in train_subjects
        assert set(fold["normalization_fit_subjects"]) <= train_subjects
        assert set(fold["feature_selection_fit_subjects"]) <= train_subjects
        assert set(fold["threshold_selection_subjects"]) <= train_subjects
        for inner in fold["inner_splits"]:
            assert test_subject not in inner["train_subjects"]
            assert test_subject not in inner["val_subjects"]
            assert set(inner["train_subjects"]) <= train_subjects
            assert set(inner["val_subjects"]) <= train_subjects

    write_fold_outputs(result, tmp_path)
    assert (tmp_path / "outer_folds.json").exists()
    assert (tmp_path / "fold_01_registry.json").exists()
    outer = json.loads((tmp_path / "outer_folds.json").read_text(encoding="utf-8"))
    assert outer["n_supervised_main"] == 6
