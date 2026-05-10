from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABEL_TO_INT = {"Poor": 0, "Good": 1}
MATRIX_FILES = ("psd_eo.npy", "psd_ec.npy", "fc_roi_eo.npy", "fc_roi_ec.npy")
MATRIX_SUBJECT_INDEX = "matrix_subject_index.csv"


@dataclass(frozen=True)
class MatrixNetInputs:
    output_dir: Path
    subject_ids: list[str]
    labels: np.ndarray
    label_names: list[str]
    cohort: pd.DataFrame
    outer_folds: dict[str, Any]
    registries: list[dict[str, Any]]
    psd_eo: np.ndarray
    psd_ec: np.ndarray
    fc_eo: np.ndarray
    fc_ec: np.ndarray
    tacs: pd.DataFrame | None
    clinical: pd.DataFrame
    ml_metrics: pd.DataFrame | None


def load_matrixnet_inputs(output_dir: str | Path) -> MatrixNetInputs:
    root = Path(output_dir)
    cohort = pd.read_csv(root / "cohort" / "cohort_master.csv")
    supervised = _supervised_main(cohort)
    subject_ids = supervised["subject_id"].astype(str).tolist()
    label_names = supervised["label_primary"].astype(str).tolist()
    unknown = sorted(set(label_names) - set(LABEL_TO_INT))
    if unknown:
        raise ValueError(f"Labels must be Good/Poor, found: {unknown}")

    matrix_dir = _matrix_dir(root)
    matrix_subjects = load_or_create_matrix_subject_index(root, subject_ids, allow_generate=True)
    if matrix_subjects != subject_ids:
        raise ValueError("matrix_subject_index.csv does not match sorted supervised_main subject order")

    matrices = {name: np.load(matrix_dir / name) for name in MATRIX_FILES}
    for name, array in matrices.items():
        if array.shape[0] != len(matrix_subjects):
            raise ValueError(f"{name} first dimension {array.shape[0]} does not match matrix_subject_index {len(matrix_subjects)}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")

    outer_folds = json.loads((root / "folds" / "outer_folds.json").read_text(encoding="utf-8"))
    _validate_outer_folds(outer_folds, subject_ids)
    registries = []
    for fold in outer_folds["folds"]:
        registry = json.loads((root / "folds" / str(fold["registry_path"])).read_text(encoding="utf-8"))
        validate_fold_registry(registry)
        registries.append(registry)

    tacs_path = root / "features" / "features_tacs_target_summary.csv"
    tacs = pd.read_csv(tacs_path) if tacs_path.exists() else None
    ml_metrics_path = root / "evaluation" / "ml_metrics_all.csv"
    ml_metrics = pd.read_csv(ml_metrics_path) if ml_metrics_path.exists() else None
    return MatrixNetInputs(
        output_dir=root,
        subject_ids=subject_ids,
        labels=np.asarray([LABEL_TO_INT[label] for label in label_names], dtype=np.int64),
        label_names=label_names,
        cohort=cohort,
        outer_folds=outer_folds,
        registries=registries,
        psd_eo=matrices["psd_eo.npy"],
        psd_ec=matrices["psd_ec.npy"],
        fc_eo=matrices["fc_roi_eo.npy"],
        fc_ec=matrices["fc_roi_ec.npy"],
        tacs=tacs,
        clinical=_clinical_frame(supervised),
        ml_metrics=ml_metrics,
    )


def ensure_matrix_subject_index(output_dir: str | Path, *, allow_generate: bool) -> Path:
    root = Path(output_dir)
    cohort = pd.read_csv(root / "cohort" / "cohort_master.csv")
    subject_ids = _supervised_main(cohort)["subject_id"].astype(str).tolist()
    matrix_dir = _matrix_dir(root)
    path = matrix_dir / MATRIX_SUBJECT_INDEX
    if path.exists():
        _load_matrix_subjects(path, expected_n=len(subject_ids))
        return path
    if not allow_generate:
        raise FileNotFoundError("matrix_subject_index.csv is required to verify matrix row order")
    _validate_matrix_row_counts(matrix_dir, expected_n=len(subject_ids))
    pd.DataFrame({"row_index": list(range(len(subject_ids))), "subject_id": subject_ids}).to_csv(path, index=False)
    return path


def load_or_create_matrix_subject_index(root: Path, subject_ids: list[str], *, allow_generate: bool) -> list[str]:
    path = ensure_matrix_subject_index(root, allow_generate=allow_generate)
    return _load_matrix_subjects(path, expected_n=len(subject_ids))


def validate_fold_registry(registry: dict[str, Any]) -> None:
    test_subject = str(registry["test_subject"])
    train = set(map(str, registry.get("supervised_train_subjects", [])))
    if test_subject in train:
        raise ValueError("outer test subject appears in supervised_train_subjects")
    for key in ("normalization_fit_subjects", "feature_selection_fit_subjects", "threshold_selection_subjects"):
        values = set(map(str, registry.get(key, [])))
        if test_subject in values:
            raise ValueError(f"outer test subject appears in {key}")
    for split in registry.get("inner_splits", []):
        for key in ("train_subjects", "val_subjects"):
            if test_subject in set(map(str, split.get(key, []))):
                raise ValueError(f"outer test subject appears in inner {key}")


def _supervised_main(cohort: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "role", "label_primary"}
    missing = required - set(cohort.columns)
    if missing:
        raise ValueError(f"cohort_master.csv missing columns: {sorted(missing)}")
    supervised = cohort.loc[cohort["role"].eq("supervised_main")].copy()
    supervised = supervised.sort_values("subject_id").reset_index(drop=True)
    if supervised.empty:
        raise ValueError("No supervised_main patients found")
    for column in ("has_baseline_eo", "has_baseline_ec"):
        if column in supervised.columns and not supervised[column].astype(bool).all():
            raise ValueError(f"Not every supervised_main patient has {column}")
    return supervised


def _matrix_dir(root: Path) -> Path:
    canonical = root / "matrices"
    legacy = root / "features" / "matrices"
    if all((canonical / name).exists() for name in MATRIX_FILES):
        return canonical
    if all((legacy / name).exists() for name in MATRIX_FILES):
        return legacy
    missing = [name for name in MATRIX_FILES if not (canonical / name).exists() and not (legacy / name).exists()]
    raise FileNotFoundError(f"Missing matrix inputs: {missing}")


def _validate_matrix_row_counts(matrix_dir: Path, *, expected_n: int) -> None:
    for name in MATRIX_FILES:
        array = np.load(matrix_dir / name, mmap_mode="r")
        if array.shape[0] != expected_n:
            raise ValueError(f"{name} first dimension {array.shape[0]} does not match supervised_main {expected_n}")


def _load_matrix_subjects(path: Path, *, expected_n: int) -> list[str]:
    frame = pd.read_csv(path)
    missing = {"row_index", "subject_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"matrix_subject_index.csv missing columns: {sorted(missing)}")
    frame = frame.sort_values("row_index").reset_index(drop=True)
    if len(frame) != expected_n:
        raise ValueError(f"matrix_subject_index.csv row count {len(frame)} does not match expected {expected_n}")
    if frame["row_index"].astype(int).tolist() != list(range(expected_n)):
        raise ValueError("matrix_subject_index.csv row_index must be contiguous from 0")
    return frame["subject_id"].astype(str).tolist()


def _validate_outer_folds(outer_folds: dict[str, Any], subject_ids: list[str]) -> None:
    fold_subjects = [str(fold.get("test_subject")) for fold in outer_folds.get("folds", [])]
    if sorted(fold_subjects) != sorted(subject_ids):
        raise ValueError("outer_folds test subjects do not match matrix_subject_index subjects")
    if len(fold_subjects) != len(set(fold_subjects)):
        raise ValueError("outer_folds contains duplicated test_subject")
    supervised_subjects = [str(subject) for subject in outer_folds.get("supervised_subjects", [])]
    if sorted(supervised_subjects) != sorted(subject_ids):
        raise ValueError("outer_folds supervised_subjects do not match matrix_subject_index subjects")


def _clinical_frame(supervised: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [
        "subject_id",
        "baseline_fma",
        "baseline_mbi",
        "mmse",
        "age",
        "sex",
        "affected_side",
        "affected_hand",
        "treated_hand",
        "disease_duration",
        "disease_duration_days",
        "time_since_stroke",
    ]
    return supervised[[column for column in candidate_columns if column in supervised.columns]].copy()
