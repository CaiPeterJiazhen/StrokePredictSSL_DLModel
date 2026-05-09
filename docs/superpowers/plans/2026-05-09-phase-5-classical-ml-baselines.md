# Phase 5 Classical ML Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build leakage-controlled classical ML baselines for Phase 5 and write patient-level predictions, metrics, bootstrap CI, permutation tests, and fold-derived feature importance.

**Architecture:** Add a focused `stroke_predict.evaluation` module for metric/statistical routines and a `stroke_predict.ml_models` module for feature assembly, registry-driven inner/outer fold training, threshold selection, predictions, and feature importance. `scripts/08_train_ml_baselines.py` is a thin CLI that reads `configs/models_ml.yaml`, calls the training module, and writes only ignored `outputs/` artifacts.

**Tech Stack:** Python 3.10+, pandas, numpy, scipy, scikit-learn, PyYAML/project YAML loader, pytest.

---

## File Structure

- Create `configs/models_ml.yaml`: Phase 5 model list, hyperparameters, feature paths, fold paths, output paths, bootstrap/permutation settings.
- Create `src/stroke_predict/evaluation.py`: patient-level metrics, bootstrap confidence intervals, permutation tests, threshold application helpers.
- Create `src/stroke_predict/ml_models.py`: model specs, feature tables, sklearn pipelines, inner CV hyperparameter selection, threshold selection, LOPO training loop, feature importance.
- Create `scripts/08_train_ml_baselines.py`: CLI entrypoint.
- Create `tests/test_feature_pipeline_no_leakage.py`: synthetic registry tests that prove fold-specific fit subjects and threshold subjects exclude outer test.
- Create `tests/test_classical_ml_outputs.py`: synthetic output/metric tests for one prediction per subject per model and required schemas.
- Modify `requirements.txt`: add `scikit-learn` so the sklearn dependency is explicit.

### Task 1: Evaluation RED-GREEN

**Files:**
- Create: `tests/test_classical_ml_outputs.py`
- Create: `src/stroke_predict/evaluation.py`

- [ ] **Step 1: Write the failing metrics test**

Add to `tests/test_classical_ml_outputs.py`:

```python
from __future__ import annotations

import pandas as pd

from stroke_predict.evaluation import (
    bootstrap_metric_ci,
    compute_classification_metrics,
    permutation_test,
    validate_patient_predictions,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model_id": "M1_fma_only", "outer_fold": 1, "subject_id": "STK-001", "label_true": "Good", "y_true": 1, "prob_good": 0.90, "pred_label": "Good", "threshold": 0.50},
            {"model_id": "M1_fma_only", "outer_fold": 2, "subject_id": "STK-002", "label_true": "Poor", "y_true": 0, "prob_good": 0.20, "pred_label": "Poor", "threshold": 0.50},
            {"model_id": "M1_fma_only", "outer_fold": 3, "subject_id": "STK-003", "label_true": "Good", "y_true": 1, "prob_good": 0.70, "pred_label": "Good", "threshold": 0.50},
            {"model_id": "M1_fma_only", "outer_fold": 4, "subject_id": "STK-004", "label_true": "Poor", "y_true": 0, "prob_good": 0.40, "pred_label": "Poor", "threshold": 0.50},
        ]
    )


def test_patient_prediction_validation_rejects_duplicate_model_subject() -> None:
    duplicate = pd.concat([_predictions(), _predictions().iloc[[0]]], ignore_index=True)

    try:
        validate_patient_predictions(duplicate, expected_subject_count=4)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Expected duplicate model-subject validation failure")


def test_compute_classical_metrics_has_required_columns() -> None:
    metrics = compute_classification_metrics(_predictions())

    assert list(metrics["model_id"]) == ["M1_fma_only"]
    required = {"roc_auc", "balanced_accuracy", "sensitivity", "specificity", "pr_auc", "brier_score", "n_subjects"}
    assert required <= set(metrics.columns)
    row = metrics.iloc[0]
    assert row["roc_auc"] == 1.0
    assert row["balanced_accuracy"] == 1.0
    assert row["sensitivity"] == 1.0
    assert row["specificity"] == 1.0
    assert row["n_subjects"] == 4


def test_bootstrap_and_permutation_outputs_are_patient_level() -> None:
    predictions = _predictions()

    ci = bootstrap_metric_ci(predictions, n_bootstrap=25, random_seed=7)
    perm = permutation_test(predictions, n_permutations=25, random_seed=7)

    assert {"model_id", "metric", "observed_value", "ci_lower", "ci_upper", "n_bootstrap", "random_seed"} <= set(ci.columns)
    assert {"model_id", "metric", "observed_value", "null_mean", "null_std", "p_value", "n_permutations", "random_seed"} <= set(perm.columns)
    assert ci["n_bootstrap"].eq(25).all()
    assert perm["n_permutations"].eq(25).all()
    assert perm["p_value"].between(0, 1).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_classical_ml_outputs.py --basetemp=.codex_pytest_tmp -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'stroke_predict.evaluation'`.

- [ ] **Step 3: Implement minimal evaluation module**

Create `src/stroke_predict/evaluation.py` with these public functions:

```python
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


REQUIRED_PREDICTION_COLUMNS = {
    "model_id",
    "outer_fold",
    "subject_id",
    "label_true",
    "y_true",
    "prob_good",
    "pred_label",
    "threshold",
}


def validate_patient_predictions(predictions: pd.DataFrame, expected_subject_count: int | None = None) -> None:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS - set(predictions.columns))
    if missing:
        raise ValueError(f"Missing prediction columns: {missing}")
    duplicate_mask = predictions.duplicated(["model_id", "subject_id"], keep=False)
    if duplicate_mask.any():
        duplicates = predictions.loc[duplicate_mask, ["model_id", "subject_id"]].drop_duplicates().to_dict("records")
        raise ValueError(f"Duplicate model-subject predictions: {duplicates}")
    if expected_subject_count is not None:
        counts = predictions.groupby("model_id")["subject_id"].nunique()
        bad = counts[counts.ne(expected_subject_count)]
        if not bad.empty:
            raise ValueError(f"Unexpected subject counts per model: {bad.to_dict()}")


def compute_classification_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    validate_patient_predictions(predictions)
    rows: list[dict[str, object]] = []
    for model_id, group in predictions.groupby("model_id", sort=True):
        y_true = group["y_true"].astype(int).to_numpy()
        prob = group["prob_good"].astype(float).to_numpy()
        y_pred = (group["pred_label"].astype(str) == "Good").astype(int).to_numpy()
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "model_id": model_id,
                "n_subjects": int(len(group)),
                "roc_auc": _safe_metric(roc_auc_score, y_true, prob),
                "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, y_pred),
                "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
                "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
                "pr_auc": _safe_metric(average_precision_score, y_true, prob),
                "brier_score": _safe_metric(brier_score_loss, y_true, prob),
            }
        )
    return pd.DataFrame(rows)
```

The same file also implements `bootstrap_metric_ci`, `permutation_test`, `_metric_value`, and `_safe_metric` using patient-level rows only.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_classical_ml_outputs.py --basetemp=.codex_pytest_tmp -q
```

Expected: PASS.

### Task 2: Fold Leakage RED-GREEN

**Files:**
- Create: `tests/test_feature_pipeline_no_leakage.py`
- Create: `src/stroke_predict/ml_models.py`

- [ ] **Step 1: Write the failing fold-training test**

Add to `tests/test_feature_pipeline_no_leakage.py`:

```python
from __future__ import annotations

import pandas as pd

from stroke_predict.ml_models import ModelSpec, train_model_on_outer_fold


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subject_id": "STK-001", "label_primary": "Good", "baseline_fma": 20.0, "x": 2.0},
            {"subject_id": "STK-002", "label_primary": "Poor", "baseline_fma": 30.0, "x": 1.0},
            {"subject_id": "STK-003", "label_primary": "Good", "baseline_fma": 21.0, "x": 3.0},
            {"subject_id": "STK-004", "label_primary": "Poor", "baseline_fma": 31.0, "x": 0.5},
            {"subject_id": "STK-005", "label_primary": "Good", "baseline_fma": 22.0, "x": 2.5},
            {"subject_id": "STK-006", "label_primary": "Poor", "baseline_fma": 32.0, "x": 0.2},
        ]
    )


def _registry() -> dict[str, object]:
    return {
        "outer_fold": 1,
        "test_subject": "STK-001",
        "supervised_train_subjects": ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"],
        "normalization_fit_subjects": ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"],
        "feature_selection_fit_subjects": ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"],
        "threshold_selection_subjects": ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"],
        "inner_splits": [
            {"inner_fold": 1, "train_subjects": ["STK-002", "STK-003", "STK-004"], "val_subjects": ["STK-005", "STK-006"]},
            {"inner_fold": 2, "train_subjects": ["STK-003", "STK-005", "STK-006"], "val_subjects": ["STK-002", "STK-004"]},
        ],
    }


def test_outer_fold_training_excludes_test_subject_from_fit_and_threshold() -> None:
    spec = ModelSpec(
        model_id="M1_fma_only",
        feature_columns=["baseline_fma"],
        feature_groups={"baseline_fma": "clinical"},
        estimator="ridge_logistic",
        c_values=[0.1, 1.0],
        l1_ratios=[0.0],
    )

    result = train_model_on_outer_fold(spec, _features(), _registry(), random_seed=11)

    assert result.prediction["subject_id"] == "STK-001"
    assert result.prediction["n_train_subjects"] == 5
    assert result.fit_subjects == ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"]
    assert "STK-001" not in result.threshold_subjects
    assert set(result.threshold_subjects) <= set(result.fit_subjects)
    assert result.prediction["prob_good"] >= 0.0
    assert result.prediction["prob_good"] <= 1.0
    assert result.importance
    assert result.importance[0]["feature_name"] == "baseline_fma"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_feature_pipeline_no_leakage.py --basetemp=.codex_pytest_tmp -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'stroke_predict.ml_models'`.

- [ ] **Step 3: Implement fold-level ML training**

Create `src/stroke_predict/ml_models.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


LABEL_TO_INT = {"Poor": 0, "Good": 1}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    feature_columns: list[str]
    feature_groups: dict[str, str]
    estimator: str
    c_values: list[float]
    l1_ratios: list[float]


@dataclass(frozen=True)
class FoldTrainingResult:
    prediction: dict[str, object]
    importance: list[dict[str, object]]
    fit_subjects: list[str]
    threshold_subjects: list[str]
```

Implement `train_model_on_outer_fold`, `_fit_pipeline`, `_inner_cv_predictions`, `_select_threshold`, `_pipeline_for_spec`, `_encode_labels`, and `_linear_importance`. The implementation must subset rows by `subject_id`, fit only on registry train subjects, use inner validation predictions for threshold, then refit on full outer train subjects.

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
python -m pytest tests/test_feature_pipeline_no_leakage.py --basetemp=.codex_pytest_tmp -q
```

Expected: PASS.

### Task 3: Feature Assembly and CLI RED-GREEN

**Files:**
- Modify: `tests/test_classical_ml_outputs.py`
- Modify: `src/stroke_predict/ml_models.py`
- Create: `configs/models_ml.yaml`
- Create: `scripts/08_train_ml_baselines.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing model-output orchestration test**

Append to `tests/test_classical_ml_outputs.py`:

```python
from pathlib import Path

from stroke_predict.ml_models import run_classical_ml_baselines


def test_run_classical_ml_baselines_writes_required_outputs(tmp_path: Path) -> None:
    cohort = pd.DataFrame(
        [
            {"subject_id": "STK-001", "role": "supervised_main", "label_primary": "Good", "age": 61, "sex": "M", "affected_hand": "right", "treated_hand": "right", "baseline_fma": 20, "baseline_mbi": 45, "mmse": 27},
            {"subject_id": "STK-002", "role": "supervised_main", "label_primary": "Poor", "age": 63, "sex": "F", "affected_hand": "left", "treated_hand": "left", "baseline_fma": 32, "baseline_mbi": 60, "mmse": 28},
            {"subject_id": "STK-003", "role": "supervised_main", "label_primary": "Good", "age": 59, "sex": "M", "affected_hand": "right", "treated_hand": "right", "baseline_fma": 22, "baseline_mbi": 40, "mmse": 26},
            {"subject_id": "STK-004", "role": "supervised_main", "label_primary": "Poor", "age": 66, "sex": "F", "affected_hand": "left", "treated_hand": "left", "baseline_fma": 34, "baseline_mbi": 62, "mmse": 29},
        ]
    )
    handcrafted = cohort[["subject_id", "label_primary"]].assign(eeg_power=[2.0, 0.5, 2.2, 0.4], native_fc_roi_eo_mean=[0.8, 0.2, 0.7, 0.3])
    folds = {
        "n_supervised_main": 4,
        "folds": [
            {"outer_fold": 1, "test_subject": "STK-001", "registry_path": "fold_01_registry.json"},
            {"outer_fold": 2, "test_subject": "STK-002", "registry_path": "fold_02_registry.json"},
            {"outer_fold": 3, "test_subject": "STK-003", "registry_path": "fold_03_registry.json"},
            {"outer_fold": 4, "test_subject": "STK-004", "registry_path": "fold_04_registry.json"},
        ],
    }
    registries = []
    subjects = cohort["subject_id"].tolist()
    for fold in folds["folds"]:
        test_subject = fold["test_subject"]
        train_subjects = [subject for subject in subjects if subject != test_subject]
        registries.append(
            {
                "outer_fold": fold["outer_fold"],
                "test_subject": test_subject,
                "supervised_train_subjects": train_subjects,
                "normalization_fit_subjects": train_subjects,
                "feature_selection_fit_subjects": train_subjects,
                "threshold_selection_subjects": train_subjects,
                "inner_splits": [
                    {"inner_fold": 1, "train_subjects": train_subjects[:2], "val_subjects": train_subjects[2:]},
                    {"inner_fold": 2, "train_subjects": train_subjects[1:], "val_subjects": train_subjects[:1]},
                ],
            }
        )
    config = {
        "random_seed": 5,
        "models": ["M0_majority", "M1_fma_only", "M2_clinical_only", "M5_tacs_target_ml"],
        "bootstrap_resamples": 10,
        "permutation_resamples": 10,
        "output_paths": {
            "predictions": str(tmp_path / "classical_patient_predictions.csv"),
            "metrics": str(tmp_path / "classical_metrics.csv"),
            "bootstrap_ci": str(tmp_path / "classical_bootstrap_ci.csv"),
            "permutation": str(tmp_path / "classical_permutation.csv"),
            "feature_importance": str(tmp_path / "classical_feature_importance.csv"),
        },
    }

    outputs = run_classical_ml_baselines(config, cohort=cohort, handcrafted=handcrafted, tacs=handcrafted, folds=folds, registries=registries)

    predictions = pd.read_csv(outputs["predictions"])
    validate_patient_predictions(predictions, expected_subject_count=4)
    assert set(predictions["model_id"]) == {"M0_majority", "M1_fma_only", "M2_clinical_only", "M5_tacs_target_ml"}
    assert predictions.groupby("model_id").size().eq(4).all()
    assert Path(outputs["metrics"]).exists()
    assert Path(outputs["bootstrap_ci"]).exists()
    assert Path(outputs["permutation"]).exists()
    assert Path(outputs["feature_importance"]).exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
python -m pytest tests/test_classical_ml_outputs.py::test_run_classical_ml_baselines_writes_required_outputs --basetemp=.codex_pytest_tmp -q
```

Expected: FAIL because `run_classical_ml_baselines` is not implemented.

- [ ] **Step 3: Implement feature assembly and runner**

Extend `src/stroke_predict/ml_models.py` with:

```python
def run_classical_ml_baselines(
    config: dict[str, Any],
    *,
    cohort: pd.DataFrame,
    handcrafted: pd.DataFrame,
    tacs: pd.DataFrame,
    folds: dict[str, Any],
    registries: list[dict[str, Any]],
    psd: pd.DataFrame | None = None,
    fc: pd.DataFrame | None = None,
) -> dict[str, str]:
    supervised = cohort.loc[cohort["role"].eq("supervised_main")].copy()
    feature_tables = build_feature_tables(supervised, handcrafted, tacs, psd=psd, fc=fc)
    specs = build_model_specs(config, feature_tables)
    predictions, importance = [], []
    registry_by_fold = {int(registry["outer_fold"]): registry for registry in registries}
    for spec in specs:
        table = feature_tables[spec.model_id]
        if spec.model_id == "M0_majority":
            predictions.extend(_run_majority_model(table, folds, registry_by_fold))
            continue
        for fold in folds["folds"]:
            result = train_model_on_outer_fold(spec, table, registry_by_fold[int(fold["outer_fold"])], random_seed=int(config.get("random_seed", 42)))
            predictions.append(result.prediction)
            importance.extend(result.importance)
    return write_classical_outputs(config, pd.DataFrame(predictions), pd.DataFrame(importance))
```

Create `configs/models_ml.yaml` with all 8 required model IDs and default output paths under `outputs/`.

Create `scripts/08_train_ml_baselines.py` that resolves config paths, loads real CSV/JSON/NPY inputs, calls `flatten_psd_matrices` and `flatten_fc_matrices`, runs `run_classical_ml_baselines`, and prints:

```text
CLASSICAL_ML_OK
n_models=8
n_predictions=152
```

Add `scikit-learn` to `requirements.txt`.

- [ ] **Step 4: Run orchestration test to verify it passes**

Run:

```bash
python -m pytest tests/test_classical_ml_outputs.py::test_run_classical_ml_baselines_writes_required_outputs --basetemp=.codex_pytest_tmp -q
```

Expected: PASS.

### Task 4: Output Schema and No-Leakage Regression

**Files:**
- Modify: `tests/test_feature_pipeline_no_leakage.py`
- Modify: `tests/test_classical_ml_outputs.py`
- Modify: `src/stroke_predict/ml_models.py`

- [ ] **Step 1: Add tests for all required model IDs and privacy-safe outputs**

Append assertions to `tests/test_classical_ml_outputs.py`:

```python
def test_required_phase5_model_ids_are_configured() -> None:
    from stroke_predict.ml_models import REQUIRED_MODEL_IDS

    assert REQUIRED_MODEL_IDS == [
        "M0_majority",
        "M1_fma_only",
        "M2_clinical_only",
        "M3_psd_ml",
        "M4_fc_ml",
        "M5_tacs_target_ml",
        "M6_all_handcrafted_eeg_ml",
        "M12_clinical_plus_eeg_ml",
    ]
```

Append to `tests/test_feature_pipeline_no_leakage.py`:

```python
def test_outer_test_subject_in_feature_table_does_not_change_train_threshold() -> None:
    spec = ModelSpec(
        model_id="M1_fma_only",
        feature_columns=["baseline_fma"],
        feature_groups={"baseline_fma": "clinical"},
        estimator="ridge_logistic",
        c_values=[1.0],
        l1_ratios=[0.0],
    )
    baseline = train_model_on_outer_fold(spec, _features(), _registry(), random_seed=17)
    poisoned = _features()
    poisoned.loc[poisoned["subject_id"].eq("STK-001"), "baseline_fma"] = 999999.0
    changed = train_model_on_outer_fold(spec, poisoned, _registry(), random_seed=17)

    assert baseline.threshold == changed.threshold
    assert baseline.fit_subjects == changed.fit_subjects
    assert baseline.threshold_subjects == changed.threshold_subjects
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
python -m pytest tests/test_feature_pipeline_no_leakage.py tests/test_classical_ml_outputs.py --basetemp=.codex_pytest_tmp -q
```

Expected: FAIL until `REQUIRED_MODEL_IDS`, `threshold`, and stable leakage behavior are implemented.

- [ ] **Step 3: Implement required IDs and threshold result field**

Update `FoldTrainingResult` with `threshold: float`, define `REQUIRED_MODEL_IDS` in order, and ensure model specs are emitted exactly in that order unless config narrows models for a synthetic test.

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
python -m pytest tests/test_feature_pipeline_no_leakage.py tests/test_classical_ml_outputs.py --basetemp=.codex_pytest_tmp -q
```

Expected: PASS.

### Task 5: Real Data Acceptance

**Files:**
- No production code changes unless the real command reveals a tested defect.

- [ ] **Step 1: Make real ignored outputs visible in the worktree**

Because Git worktrees do not copy ignored `outputs/`, create a worktree-local junction or directory mapping to the original ignored `outputs/` without staging it. If junction creation is blocked, run the CLI from the main checkout after merging the code.

Run:

```powershell
New-Item -ItemType Junction -Path outputs -Target F:\CJZProjectFile\StrokePredictSSL-DLModel\outputs
```

Expected: worktree path `outputs/` resolves to the original ignored run outputs.

- [ ] **Step 2: Run Phase 5 command**

Run:

```bash
python scripts/08_train_ml_baselines.py --config configs/models_ml.yaml
```

Expected output includes:

```text
CLASSICAL_ML_OK
n_models=8
n_predictions=152
```

- [ ] **Step 3: Validate real prediction counts**

Run:

```bash
python -c "import pandas as pd; p=pd.read_csv('outputs/predictions/classical_patient_predictions.csv'); print(p.groupby('model_id').size().to_dict()); assert p.groupby('model_id').size().eq(19).all(); assert not p.duplicated(['model_id','subject_id']).any()"
```

Expected: eight model IDs each mapped to `19`.

- [ ] **Step 4: Run targeted Phase 5 and leakage tests**

Run:

```bash
python -m pytest tests/test_feature_pipeline_no_leakage.py tests/test_classical_ml_outputs.py tests/test_splits_no_leakage.py tests/test_ssl_no_leakage.py --basetemp=.codex_pytest_tmp -q
```

Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run:

```bash
python -m pytest --basetemp=.codex_pytest_tmp
```

Expected: PASS.

- [ ] **Step 6: Verify ignored artifacts are not staged or tracked**

Run:

```bash
git status --short
```

Expected: no `outputs/`, `.xlsx`, `.set`, or `.fdt` entries.

### Task 6: Commit, Merge, Push, and Cleanup

**Files:**
- Git metadata only after all verification passes.

- [ ] **Step 1: Commit implementation**

Run:

```bash
git add configs/models_ml.yaml requirements.txt scripts/08_train_ml_baselines.py src/stroke_predict/evaluation.py src/stroke_predict/ml_models.py tests/test_feature_pipeline_no_leakage.py tests/test_classical_ml_outputs.py docs/superpowers/plans/2026-05-09-phase-5-classical-ml-baselines.md
git commit -m "feat: add phase 5 classical ml baselines"
```

Expected: commit succeeds and does not include `outputs/`.

- [ ] **Step 2: Merge into main**

Run from the main checkout:

```bash
git checkout main
git merge codex/phase-5-ml-baselines
```

Expected: fast-forward or clean merge.

- [ ] **Step 3: Run post-merge verification**

Run:

```bash
python -m pytest --basetemp=.codex_pytest_tmp
python scripts/08_train_ml_baselines.py --config configs/models_ml.yaml
```

Expected: tests pass and Phase 5 command writes the required ignored outputs.

- [ ] **Step 4: Push main**

Run:

```bash
git push origin main
```

Expected: remote main receives the Phase 5 commits.

- [ ] **Step 5: Clean owned worktree**

Run from the main checkout:

```bash
git worktree remove F:\CJZProjectFile\StrokePredictSSL-DLModel\.worktrees\phase-5-ml-baselines
git worktree prune
```

Expected: worktree is removed. Original ignored real data outputs under `F:\CJZProjectFile\StrokePredictSSL-DLModel\outputs` remain intact.

## Self-Review

This plan covers all Phase 5 model IDs, fold-registry usage, patient-level prediction uniqueness, fold-local preprocessing and threshold selection, metrics, bootstrap CI, permutation tests, feature importance, real-data acceptance, and the requested merge/push/cleanup workflow. It does not include MatrixNet, SSL, deep learning, figures, manuscript generation, Excel files, EEG files, or tracked `outputs/` artifacts.
