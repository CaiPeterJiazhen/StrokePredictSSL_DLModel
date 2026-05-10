from __future__ import annotations

import pandas as pd

from stroke_predict.ssl_matrixnet_data import build_ssl_fold_pools, validate_ssl_matrix_index


def test_outer_test_patient_all_stages_are_excluded_from_ssl_pool() -> None:
    index = validate_ssl_matrix_index(_stage_rich_index())
    outer_folds = _outer_folds()

    pool, audit = build_ssl_fold_pools(index, outer_folds, ssl_variant="stroke_all_stage", fold_limit=1)

    assert "STK-001" not in set(pool["subject_id"])
    assert not pool["stage"].isin(["baseline", "immediate", "mid", "final"]).where(pool["subject_id"].eq("STK-001")).any()
    assert audit["test_subject_records_in_pool"].eq(0).all()
    assert audit["leakage_passed"].all()


def test_healthy_records_may_be_included_for_healthy_variants() -> None:
    index = validate_ssl_matrix_index(_stage_rich_index())
    outer_folds = _outer_folds()

    pool, audit = build_ssl_fold_pools(index, outer_folds, ssl_variant="stroke_healthy_baseline", fold_limit=1)

    assert "HC-001" in set(pool["subject_id"])
    assert audit["healthy_records_in_pool"].gt(0).all()


def test_unlabeled_stroke_records_do_not_require_supervised_labels() -> None:
    index = validate_ssl_matrix_index(_stage_rich_index().drop(columns=["label"], errors="ignore"))
    outer_folds = _outer_folds()

    pool, audit = build_ssl_fold_pools(index, outer_folds, ssl_variant="stroke_all_stage", fold_limit=1)

    assert "STK-099" in set(pool["subject_id"])
    assert audit["unlabeled_stroke_records_in_pool"].gt(0).all()


def _outer_folds() -> dict[str, object]:
    return {
        "folds": [
            {
                "outer_fold": 1,
                "test_subject": "STK-001",
                "supervised_train_subjects": ["STK-002"],
            }
        ]
    }


def _stage_rich_index() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    row_index = 0
    for stage in ["baseline", "immediate", "mid", "final"]:
        for condition in ["eo", "ec"]:
            rows.append(
                {
                    "row_index": row_index,
                    "subject_id": "STK-001",
                    "source": "stroke_supervised",
                    "stage": stage,
                    "condition": condition,
                }
            )
            row_index += 1
    for subject_id, source, stage in [
        ("STK-002", "stroke_supervised", "baseline"),
        ("STK-099", "stroke_ssl_only", "final"),
        ("HC-001", "healthy", "baseline"),
    ]:
        for condition in ["eo", "ec"]:
            rows.append(
                {
                    "row_index": row_index,
                    "subject_id": subject_id,
                    "source": source,
                    "stage": stage,
                    "condition": condition,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)
