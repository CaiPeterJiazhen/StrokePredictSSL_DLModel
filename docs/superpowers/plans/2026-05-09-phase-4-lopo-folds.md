# Phase 4 LOPO Folds Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build patient-level LOPO fold registries and fold-specific SSL pool registries with tests that prove no supervised, inner, normalization, or SSL leakage.

**Architecture:** Add a small `stroke_predict.splits` module responsible for deterministic patient-level fold construction, registry validation, and JSON writing. Add `configs/cv.yaml` and `scripts/07_make_folds.py` as the command entrypoint. Tests use synthetic cohort/QC/feature data first, then the real-data command verifies produced registries.

**Tech Stack:** Python 3.10+, pandas, standard-library json/pathlib/dataclasses, pytest. No scikit-learn dependency is added.

---

### Task 1: Split Builder Unit Tests

**Files:**
- Create: `tests/test_splits_no_leakage.py`
- Create: `src/stroke_predict/splits.py`

- [ ] **Step 1: Write the failing test**

```python
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
    features = pd.DataFrame({"subject_id": [f"STK-{i:03d}" for i in range(1, 7)]})

    result = build_outer_folds(_cohort(), _qc(), features, inner_k=3)

    assert [fold["test_subject"] for fold in result["folds"]] == [f"STK-{i:03d}" for i in range(1, 7)]
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_splits_no_leakage.py --basetemp=.codex_pytest_tmp -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'stroke_predict.splits'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/stroke_predict/splits.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ANON_RECORD_FIELDS = ("record_id", "subject_id", "source", "stage", "condition")


def build_outer_folds(cohort: pd.DataFrame, qc: pd.DataFrame, features: pd.DataFrame, inner_k: int = 3) -> dict[str, Any]:
    supervised = _supervised_subjects(cohort)
    _validate_feature_subjects(supervised, features)
    registries = []
    folds = []
    for outer_index, test_subject in enumerate(supervised, start=1):
        train_subjects = [subject for subject in supervised if subject != test_subject]
        registry = _build_registry(outer_index, test_subject, train_subjects, cohort, qc, inner_k)
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
```

Also implement helpers `_supervised_subjects`, `_validate_feature_subjects`, `_build_registry`, `_make_inner_splits`, `_ssl_records`, and `write_fold_outputs`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_splits_no_leakage.py --basetemp=.codex_pytest_tmp -q`
Expected: PASS.

### Task 2: SSL Leakage Tests

**Files:**
- Create: `tests/test_ssl_no_leakage.py`
- Modify: `src/stroke_predict/splits.py`

- [ ] **Step 1: Write the failing test**

```python
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
            {"record_id": "STK-001_baseline_eyes_open_01", "subject_id": "STK-001", "source": "stroke", "stage": "baseline", "condition": "eyes_open", "passes_qc": True},
            {"record_id": "STK-001_final_eyes_closed_01", "subject_id": "STK-001", "source": "stroke", "stage": "final", "condition": "eyes_closed", "passes_qc": True},
            {"record_id": "STK-002_baseline_eyes_open_01", "subject_id": "STK-002", "source": "stroke", "stage": "baseline", "condition": "eyes_open", "passes_qc": True},
            {"record_id": "STK-010_baseline_eyes_open_01", "subject_id": "STK-010", "source": "stroke", "stage": "baseline", "condition": "eyes_open", "passes_qc": True},
            {"record_id": "HC-001_baseline_eyes_open_01", "subject_id": "HC-001", "source": "healthy", "stage": "baseline", "condition": "eyes_open", "passes_qc": True},
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ssl_no_leakage.py --basetemp=.codex_pytest_tmp -q`
Expected: FAIL if SSL exclusion fields or anonymous record filtering are missing.

- [ ] **Step 3: Implement SSL registry fields**

Update `_build_registry` so `ssl_train_records` comes from QC pass records where `subject_id != test_subject`, and `ssl_excluded_records` contains QC pass records where `subject_id == test_subject`. Derive `ssl_train_subjects`, `healthy_ssl_subjects`, `stages_used`, and `conditions_used` from those anonymous records.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_splits_no_leakage.py tests/test_ssl_no_leakage.py --basetemp=.codex_pytest_tmp -q`
Expected: PASS.

### Task 3: Config and CLI

**Files:**
- Create: `configs/cv.yaml`
- Create: `scripts/07_make_folds.py`
- Modify: `tests/test_scripts.py`

- [ ] **Step 1: Write failing script existence/config test**

Append to `tests/test_scripts.py`:

```python
def test_phase4_fold_script_and_config_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "configs" / "cv.yaml").exists()
    assert (root / "scripts" / "07_make_folds.py").exists()
```

Run: `pytest tests/test_scripts.py::test_phase4_fold_script_and_config_exist --basetemp=.codex_pytest_tmp -q`
Expected: FAIL because files do not exist.

- [ ] **Step 2: Add config and CLI**

Create `configs/cv.yaml`:

```yaml
project_config: "project.yaml"
outer_cv: "leave_one_patient_out"
inner_cv: "stratified_kfold"
inner_k: 3
unit: "subject_id"
```

Create `scripts/07_make_folds.py` that loads YAML, resolves `project_config`, reads `cohort_master.csv`, `eeg_qc_summary.csv`, and `handcrafted_features.csv`, then writes fold outputs under `project.output_dir / "folds"`.

- [ ] **Step 3: Run script existence test**

Run: `pytest tests/test_scripts.py::test_phase4_fold_script_and_config_exist --basetemp=.codex_pytest_tmp -q`
Expected: PASS.

### Task 4: Real Data Acceptance

**Files:**
- No production code changes unless acceptance reveals a defect.

- [ ] **Step 1: Generate real fold outputs**

Run from the project root:

```bash
python scripts/07_make_folds.py --config configs/cv.yaml
```

Expected output includes:

```text
FOLDS_OK
n_outer_folds=19
```

- [ ] **Step 2: Run Phase 4 tests**

Run:

```bash
pytest tests/test_splits_no_leakage.py tests/test_ssl_no_leakage.py --basetemp=.codex_pytest_tmp -q
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
pytest --basetemp=.codex_pytest_tmp
```

Expected: PASS.

- [ ] **Step 4: Verify outputs remain untracked**

Run:

```bash
git status --short
```

Expected: no `outputs/` files listed. `outputs/folds/` may exist in the real project but must be ignored by Git.

## Self-Review

This plan covers Phase 4 spec requirements: patient-level LOPO, inner patient-level splits, fold-specific SSL pool, feature coverage checks without feature-value decisions, no-leakage tests, CLI output, and real-data acceptance. It does not add model training, MatrixNet, SSL pretraining, evaluation outputs, Excel files, EEG files, or tracked `outputs/` artifacts.
