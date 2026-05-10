from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stroke_predict.matrixnet_data import ensure_matrix_subject_index, load_matrixnet_inputs, validate_fold_registry
from stroke_predict.matrixnet_preprocessing import FoldPreprocessor, fit_vector_preprocessor


def _write_minimal_inputs(root: Path) -> None:
    (root / "cohort").mkdir(parents=True)
    (root / "folds").mkdir(parents=True)
    (root / "matrices").mkdir(parents=True)
    (root / "features").mkdir(parents=True)
    cohort = pd.DataFrame(
        {
            "subject_id": ["S01", "S02", "S03"],
            "role": ["supervised_main", "supervised_main", "supervised_main"],
            "label_primary": ["Good", "Poor", "Good"],
            "baseline_fma": [50, 40, 55],
            "age": [60, 61, 62],
            "sex": ["F", "M", "F"],
            "baseline_mbi": [80, 70, 90],
            "mmse": [28, 27, 29],
            "affected_hand": ["left", "right", "left"],
            "treated_hand": ["left", "right", "left"],
            "has_baseline_eo": [True, True, True],
            "has_baseline_ec": [True, True, True],
        }
    )
    cohort.to_csv(root / "cohort" / "cohort_master.csv", index=False)
    folds = {
        "n_supervised_main": 3,
        "supervised_subjects": ["S01", "S02", "S03"],
        "folds": [
            {
                "outer_fold": 1,
                "test_subject": "S01",
                "supervised_train_subjects": ["S02", "S03"],
                "registry_path": "fold_01_registry.json",
            },
            {
                "outer_fold": 2,
                "test_subject": "S02",
                "supervised_train_subjects": ["S01", "S03"],
                "registry_path": "fold_02_registry.json",
            },
            {
                "outer_fold": 3,
                "test_subject": "S03",
                "supervised_train_subjects": ["S01", "S02"],
                "registry_path": "fold_03_registry.json",
            },
        ],
    }
    (root / "folds" / "outer_folds.json").write_text(json.dumps(folds), encoding="utf-8")
    for fold in folds["folds"]:
        train_subjects = fold["supervised_train_subjects"]
        registry = {
            "outer_fold": fold["outer_fold"],
            "test_subject": fold["test_subject"],
            "supervised_train_subjects": train_subjects,
            "inner_splits": [{"inner_fold": 1, "train_subjects": train_subjects[:1], "val_subjects": train_subjects[1:]}],
            "normalization_fit_subjects": train_subjects,
            "threshold_selection_subjects": train_subjects,
        }
        (root / "folds" / fold["registry_path"]).write_text(json.dumps(registry), encoding="utf-8")
    for name, shape in {
        "psd_eo.npy": (3, 2, 4, 5),
        "psd_ec.npy": (3, 2, 4, 5),
        "fc_roi_eo.npy": (3, 2, 3, 2, 2),
        "fc_roi_ec.npy": (3, 2, 3, 2, 2),
    }.items():
        np.save(root / "matrices" / name, np.ones(shape, dtype=np.float32))
    pd.DataFrame({"row_index": [0, 1, 2], "subject_id": ["S01", "S02", "S03"]}).to_csv(
        root / "matrices" / "matrix_subject_index.csv",
        index=False,
    )
    pd.DataFrame({"subject_id": ["S01", "S02", "S03"], "tacs_a": [1.0, None, 3.0]}).to_csv(
        root / "features" / "features_tacs_target_summary.csv",
        index=False,
    )


def test_load_matrixnet_inputs_aligns_rows_to_matrix_subject_index(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    inputs = load_matrixnet_inputs(tmp_path)
    assert inputs.subject_ids == ["S01", "S02", "S03"]
    assert inputs.labels.tolist() == [1, 0, 1]
    assert inputs.psd_eo.shape == (3, 2, 4, 5)
    assert inputs.fc_ec.shape == (3, 2, 3, 2, 2)


def test_load_matrixnet_inputs_fails_when_matrix_rows_do_not_match_subjects(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    np.save(tmp_path / "matrices" / "psd_eo.npy", np.ones((2, 2, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="psd_eo.npy first dimension"):
        load_matrixnet_inputs(tmp_path)


def test_load_matrixnet_inputs_generates_verifiable_matrix_subject_index(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    (tmp_path / "matrices" / "matrix_subject_index.csv").unlink()
    index_path = ensure_matrix_subject_index(tmp_path, allow_generate=True)
    assert index_path.exists()
    loaded = pd.read_csv(index_path)
    assert loaded["subject_id"].tolist() == ["S01", "S02", "S03"]


def test_load_matrixnet_inputs_fails_on_mismatched_matrix_subject_index(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    pd.DataFrame({"row_index": [0, 1, 2], "subject_id": ["S01", "S03", "S02"]}).to_csv(
        tmp_path / "matrices" / "matrix_subject_index.csv",
        index=False,
    )
    with pytest.raises(ValueError, match="matrix_subject_index"):
        load_matrixnet_inputs(tmp_path)


def test_validate_fold_registry_rejects_test_subject_in_fit_sets() -> None:
    registry = {
        "outer_fold": 1,
        "test_subject": "S01",
        "supervised_train_subjects": ["S02", "S03"],
        "inner_splits": [{"inner_fold": 1, "train_subjects": ["S01"], "val_subjects": ["S02"]}],
        "normalization_fit_subjects": ["S02", "S03"],
        "threshold_selection_subjects": ["S02", "S03"],
    }
    with pytest.raises(ValueError, match="outer test subject"):
        validate_fold_registry(registry)


def test_matrix_scaler_uses_only_outer_training_subjects() -> None:
    subject_ids = ["S01", "S02", "S03"]
    matrix = np.asarray([[[1.0, 1.0]], [[3.0, 3.0]], [[100.0, 100.0]]], dtype=np.float32)
    preprocessor = FoldPreprocessor.fit(subject_ids, train_subjects=["S01", "S02"], matrices={"psd": matrix})
    transformed = preprocessor.transform_matrix("psd", matrix)
    assert np.allclose(preprocessor.matrix_stats["psd"].mean, 2.0)
    assert np.allclose(preprocessor.matrix_stats["psd"].std, 1.0)
    assert np.allclose(transformed[0], -1.0)
    assert np.allclose(transformed[1], 1.0)
    assert np.allclose(transformed[2], 98.0)


def test_vector_preprocessor_imputes_and_scales_from_training_rows_only() -> None:
    frame = pd.DataFrame({"subject_id": ["S01", "S02", "S03"], "a": [1.0, 3.0, None], "b": [10.0, 14.0, 100.0]})
    processed = fit_vector_preprocessor(frame, subject_ids=["S01", "S02", "S03"], train_subjects=["S01", "S02"])
    values = processed.transform(frame, ["S01", "S02", "S03"])
    assert values.shape == (3, 2)
    assert np.isfinite(values).all()
    assert np.allclose(processed.medians["a"], 2.0)
    assert np.allclose(processed.means["b"], 12.0)
