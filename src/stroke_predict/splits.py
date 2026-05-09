from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

ANON_RECORD_FIELDS = ("record_id", "subject_id", "source", "stage", "condition")


def build_outer_folds(cohort: pd.DataFrame, qc: pd.DataFrame, features: pd.DataFrame, inner_k: int = 3) -> dict[str, Any]:
    if inner_k < 2:
        raise ValueError("inner_k must be at least 2")
    supervised = _supervised_subjects(cohort)
    if len(supervised) <= inner_k:
        raise ValueError("Number of supervised subjects must be greater than inner_k")
    _validate_feature_subjects(supervised, features)

    registries = []
    folds = []
    for outer_index, test_subject in enumerate(supervised, start=1):
        train_subjects = [subject for subject in supervised if subject != test_subject]
        registry = _build_registry(
            outer_fold=outer_index,
            test_subject=test_subject,
            train_subjects=train_subjects,
            cohort=cohort,
            qc=qc,
            inner_k=inner_k,
        )
        registries.append(registry)
        folds.append(
            {
                "outer_fold": outer_index,
                "test_subject": test_subject,
                "supervised_train_subjects": train_subjects,
                "registry_path": f"fold_{outer_index:02d}_registry.json",
            }
        )

    return {
        "schema_version": 1,
        "outer_cv": "leave_one_patient_out",
        "unit": "subject_id",
        "inner_cv": "stratified_kfold",
        "inner_k": inner_k,
        "n_supervised_main": len(supervised),
        "supervised_subjects": supervised,
        "folds": folds,
        "registries": registries,
    }


def write_fold_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outer = {key: value for key, value in result.items() if key != "registries"}
    _write_json(outer, output_path / "outer_folds.json")
    for registry in result["registries"]:
        _write_json(registry, output_path / f"fold_{int(registry['outer_fold']):02d}_registry.json")


def _supervised_subjects(cohort: pd.DataFrame) -> list[str]:
    _require_columns(cohort, ["subject_id", "role"])
    subjects = cohort.loc[cohort["role"].eq("supervised_main"), "subject_id"].dropna().astype(str).tolist()
    subjects = sorted(set(subjects))
    if not subjects:
        raise ValueError("No supervised_main subjects found")
    return subjects


def _validate_feature_subjects(supervised: list[str], features: pd.DataFrame) -> None:
    _require_columns(features, ["subject_id"])
    feature_subjects = set(features["subject_id"].dropna().astype(str))
    missing = [subject for subject in supervised if subject not in feature_subjects]
    if missing:
        raise ValueError(f"Missing Phase 3 features for supervised_main subjects: {missing}")


def _build_registry(
    outer_fold: int,
    test_subject: str,
    train_subjects: list[str],
    cohort: pd.DataFrame,
    qc: pd.DataFrame,
    inner_k: int,
) -> dict[str, Any]:
    train_set = set(train_subjects)
    inner_splits = _make_inner_splits(cohort, train_subjects, inner_k)
    ssl_train_records = _qc_records(qc, exclude_subject=test_subject)
    ssl_excluded_records = _qc_records(qc, include_subject=test_subject)
    ssl_train_subjects = sorted({record["subject_id"] for record in ssl_train_records})
    healthy_subjects = _healthy_ssl_subjects(cohort, ssl_train_subjects)
    stages = sorted({record["stage"] for record in ssl_train_records})
    conditions = sorted({record["condition"] for record in ssl_train_records})
    return {
        "outer_fold": outer_fold,
        "test_subject": test_subject,
        "supervised_train_subjects": train_subjects,
        "inner_splits": inner_splits,
        "ssl_train_subjects": ssl_train_subjects,
        "ssl_train_records": ssl_train_records,
        "ssl_excluded_subjects": [test_subject],
        "ssl_excluded_records": ssl_excluded_records,
        "healthy_ssl_subjects": healthy_subjects,
        "normalization_fit_subjects": train_subjects,
        "feature_selection_fit_subjects": train_subjects,
        "threshold_selection_subjects": train_subjects,
        "stages_used": stages,
        "conditions_used": conditions,
    }


def _make_inner_splits(cohort: pd.DataFrame, train_subjects: list[str], inner_k: int) -> list[dict[str, Any]]:
    _require_columns(cohort, ["subject_id", "label_primary"])
    labels = (
        cohort.loc[cohort["subject_id"].astype(str).isin(train_subjects), ["subject_id", "label_primary"]]
        .drop_duplicates("subject_id")
        .assign(subject_id=lambda frame: frame["subject_id"].astype(str))
    )
    label_groups: dict[str, list[str]] = {}
    for _, row in labels.sort_values(["label_primary", "subject_id"]).iterrows():
        label_groups.setdefault(str(row["label_primary"]), []).append(str(row["subject_id"]))

    val_by_fold: list[list[str]] = [[] for _ in range(inner_k)]
    for subjects in label_groups.values():
        for index, subject in enumerate(subjects):
            val_by_fold[index % inner_k].append(subject)

    train_set = set(train_subjects)
    splits = []
    for index, val_subjects in enumerate(val_by_fold, start=1):
        val_subjects = sorted(val_subjects)
        inner_train = sorted(train_set - set(val_subjects))
        splits.append({"inner_fold": index, "train_subjects": inner_train, "val_subjects": val_subjects})
    return splits


def _qc_records(qc: pd.DataFrame, include_subject: str | None = None, exclude_subject: str | None = None) -> list[dict[str, str]]:
    _require_columns(qc, [*ANON_RECORD_FIELDS, "passes_qc"])
    records = []
    for _, row in qc.iterrows():
        subject_id = str(row["subject_id"])
        if include_subject is not None and subject_id != include_subject:
            continue
        if exclude_subject is not None and subject_id == exclude_subject:
            continue
        if not _is_true(row["passes_qc"]):
            continue
        records.append({field: str(row[field]) for field in ANON_RECORD_FIELDS})
    return sorted(records, key=lambda record: (record["subject_id"], record["stage"], record["condition"], record["record_id"]))


def _healthy_ssl_subjects(cohort: pd.DataFrame, ssl_train_subjects: list[str]) -> list[str]:
    _require_columns(cohort, ["subject_id", "role"])
    ssl_set = set(ssl_train_subjects)
    healthy = cohort.loc[cohort["role"].eq("healthy_ssl"), "subject_id"].dropna().astype(str)
    return sorted(subject for subject in healthy if subject in ssl_set)


def _is_true(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
