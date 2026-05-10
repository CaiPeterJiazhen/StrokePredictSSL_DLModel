# Phase 6.2 Score-Direction Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a tested Phase 6.2 audit path that proves MatrixNet label/score direction, calibrates score orientation using inner validation only, writes detailed consistency metrics, and reruns no-SSL MatrixNet without starting SSL.

**Architecture:** Keep the existing MatrixNet modules, but add explicit score contract fields and Phase 6.2 output names in `matrixnet_training.py`. Add small pure helper functions for orientation calibration and metric aggregation so they can be tested without running full training. Extend the CLI/config for `device`, `require_cuda`, and `--phase6-2-audit` output naming.

**Tech Stack:** Python, PyTorch, NumPy, pandas, scikit-learn ROC-AUC, PyYAML, pytest.

---

## File Structure

- Modify `src/stroke_predict/matrixnet_data.py`
  - Keep `LABEL_TO_INT = {"Poor": 0, "Good": 1}` and expose assertions through tests.
- Modify `src/stroke_predict/matrixnet_training.py`
  - Add `label_int`, `logit`, `sigmoid_score`, `score_orientation`.
  - Add inner-validation-only orientation calibration.
  - Add Phase 6.2 file names and report writer.
  - Add seed-wise, pooled, and patient-averaged metric helpers.
  - Add device/require_cuda handling.
- Modify `scripts/09_train_matrixnet.py`
  - Parse `device`, `require_cuda`, and `--phase6-2-audit`.
- Modify `configs/matrixnet.yaml`
  - Add full-mode `device: cuda`, `require_cuda: true`, and optional `orientation_calibration: inner_val_auc`.
- Create `tests/test_matrixnet_score_direction.py`
  - Unit tests for label contract, BCE/sigmoid semantics, orientation calibration, prediction rows, and metric consistency.
- Modify `tests/test_matrixnet_outputs.py`
  - Assert Phase 6.2 output names and required columns.
- Modify `tests/test_matrixnet_training_smoke.py`
  - Assert `predicted_score` remains Good probability after orientation handling.
- Create docs:
  - `docs/superpowers/specs/2026-05-10-phase6-2-score-direction-audit-design.md`
  - `docs/superpowers/plans/2026-05-10-phase6-2-score-direction-audit.md`

## Task 1: RED tests for label and score contract

**Files:**
- Create: `tests/test_matrixnet_score_direction.py`
- Modify later: `src/stroke_predict/matrixnet_training.py`

- [ ] **Step 1: Write the failing tests**

Add:

```python
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from stroke_predict.matrixnet_data import LABEL_TO_INT
from stroke_predict.matrixnet_training import (
    INT_TO_LABEL,
    _apply_score_orientation,
    _calibrate_score_orientation,
    _phase6_2_metric_frames,
)


def test_label_encoding_contract_is_good_positive_class() -> None:
    assert LABEL_TO_INT == {"Poor": 0, "Good": 1}
    assert INT_TO_LABEL == {0: "Poor", 1: "Good"}
    labels = torch.tensor([0.0, 1.0])
    logits = torch.tensor([-2.0, 2.0])
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
    swapped_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(logits, 1.0 - labels)
    assert float(loss.mean()) < float(swapped_loss.mean())
    assert torch.sigmoid(logits).tolist()[1] > torch.sigmoid(logits).tolist()[0]


def test_inner_validation_auc_below_half_inverts_without_outer_scores() -> None:
    inner_y = np.asarray([0, 0, 1, 1], dtype=int)
    inner_scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=float)
    orientation = _calibrate_score_orientation(inner_y, inner_scores)
    assert orientation == "inverted_by_inner_val"
    oriented = _apply_score_orientation(inner_scores, orientation)
    assert math.isclose(float(roc_auc_score(inner_y, oriented)), 1.0)


def test_inner_validation_single_class_does_not_invert() -> None:
    inner_y = np.asarray([1, 1], dtype=int)
    inner_scores = np.asarray([0.2, 0.1], dtype=float)
    orientation = _calibrate_score_orientation(inner_y, inner_scores)
    assert orientation == "normal_insufficient_inner_classes"
    assert np.allclose(_apply_score_orientation(inner_scores, orientation), inner_scores)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_score_direction.py -q
```

Expected: FAIL because `tests/test_matrixnet_score_direction.py` is new and helper functions are not implemented.

- [ ] **Step 3: Implement minimal score helpers**

In `src/stroke_predict/matrixnet_training.py`, add:

```python
def _calibrate_score_orientation(y_true: np.ndarray, sigmoid_scores: np.ndarray) -> str:
    if y_true.size == 0 or sigmoid_scores.size == 0 or len(set(y_true.astype(int).tolist())) < 2:
        return "normal_insufficient_inner_classes"
    auc = float(roc_auc_score(y_true.astype(int), sigmoid_scores.astype(float)))
    return "inverted_by_inner_val" if auc < 0.5 else "normal"


def _apply_score_orientation(sigmoid_scores: np.ndarray, orientation: str) -> np.ndarray:
    scores = sigmoid_scores.astype(float)
    if orientation == "inverted_by_inner_val":
        return 1.0 - scores
    if orientation in {"normal", "normal_insufficient_inner_classes"}:
        return scores
    raise ValueError(f"Unsupported score_orientation: {orientation}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_score_direction.py::test_label_encoding_contract_is_good_positive_class tests/test_matrixnet_score_direction.py::test_inner_validation_auc_below_half_inverts_without_outer_scores tests/test_matrixnet_score_direction.py::test_inner_validation_single_class_does_not_invert -q
```

Expected: 3 passed.

## Task 2: RED tests for Phase 6.2 metric consistency frames

**Files:**
- Modify: `tests/test_matrixnet_score_direction.py`
- Modify later: `src/stroke_predict/matrixnet_training.py`

- [ ] **Step 1: Add failing metric consistency test**

Append:

```python
def test_phase6_2_metric_frames_compare_seed_pooled_and_patient_averaged_auc() -> None:
    predictions = pd.DataFrame(
        [
            {"model_name": "M8b_matrixnet_fc_only", "seed": 0, "outer_fold": 1, "patient_id": "S01", "true_label": "Poor", "label_int": 0, "predicted_score": 0.10, "predicted_label": "Poor", "threshold": 0.5, "score_orientation": "normal", "run_mode": "full"},
            {"model_name": "M8b_matrixnet_fc_only", "seed": 0, "outer_fold": 2, "patient_id": "S02", "true_label": "Poor", "label_int": 0, "predicted_score": 0.20, "predicted_label": "Poor", "threshold": 0.5, "score_orientation": "normal", "run_mode": "full"},
            {"model_name": "M8b_matrixnet_fc_only", "seed": 0, "outer_fold": 3, "patient_id": "S03", "true_label": "Good", "label_int": 1, "predicted_score": 0.80, "predicted_label": "Good", "threshold": 0.5, "score_orientation": "normal", "run_mode": "full"},
            {"model_name": "M8b_matrixnet_fc_only", "seed": 0, "outer_fold": 4, "patient_id": "S04", "true_label": "Good", "label_int": 1, "predicted_score": 0.90, "predicted_label": "Good", "threshold": 0.5, "score_orientation": "normal", "run_mode": "full"},
            {"model_name": "M8b_matrixnet_fc_only", "seed": 1, "outer_fold": 1, "patient_id": "S01", "true_label": "Poor", "label_int": 0, "predicted_score": 0.40, "predicted_label": "Poor", "threshold": 0.5, "score_orientation": "normal", "run_mode": "full"},
            {"model_name": "M8b_matrixnet_fc_only", "seed": 1, "outer_fold": 2, "patient_id": "S02", "true_label": "Poor", "label_int": 0, "predicted_score": 0.30, "predicted_label": "Poor", "threshold": 0.5, "score_orientation": "normal", "run_mode": "full"},
            {"model_name": "M8b_matrixnet_fc_only", "seed": 1, "outer_fold": 3, "patient_id": "S03", "true_label": "Good", "label_int": 1, "predicted_score": 0.60, "predicted_label": "Good", "threshold": 0.5, "score_orientation": "normal", "run_mode": "full"},
            {"model_name": "M8b_matrixnet_fc_only", "seed": 1, "outer_fold": 4, "patient_id": "S04", "true_label": "Good", "label_int": 1, "predicted_score": 0.70, "predicted_label": "Good", "threshold": 0.5, "score_orientation": "normal", "run_mode": "full"},
        ]
    )
    metrics, seed_metrics, patient_metrics = _phase6_2_metric_frames(predictions)
    row = metrics.iloc[0]
    assert row["roc_auc_mean"] == 1.0
    assert row["pooled_auc"] == 1.0
    assert row["patient_averaged_auc"] == 1.0
    assert row["auc_score"] == 1.0
    assert row["auc_one_minus_score"] == 0.0
    assert row["mean_score_good"] > row["mean_score_poor"]
    assert bool(row["direction_correct"]) is True
    assert len(seed_metrics) == 2
    assert len(patient_metrics) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_score_direction.py::test_phase6_2_metric_frames_compare_seed_pooled_and_patient_averaged_auc -q
```

Expected: FAIL because `_phase6_2_metric_frames` is missing.

- [ ] **Step 3: Implement metric frames**

Add `_phase6_2_metric_frames`, `_phase6_2_seed_metric_rows`, `_safe_auc`, and patient averaging helpers in `matrixnet_training.py`. Use `label_int` as y_true when present, otherwise derive from `true_label == "Good"`.

- [ ] **Step 4: Run metric tests**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_score_direction.py -q
```

Expected: all score direction tests pass.

## Task 3: RED tests for prediction row audit columns and inner-val orientation

**Files:**
- Modify: `tests/test_matrixnet_training_smoke.py`
- Modify later: `src/stroke_predict/matrixnet_training.py`

- [ ] **Step 1: Add failing prediction-column assertions**

In `test_matrixnet_fast_smoke_produces_finite_predictions`, assert:

```python
    assert {
        "label_int",
        "logit",
        "sigmoid_score",
        "score_orientation",
    } <= set(predictions.columns)
    assert set(predictions["label_int"].astype(int)) <= {0, 1}
    assert predictions["sigmoid_score"].between(0, 1).all()
    assert predictions["predicted_score"].between(0, 1).all()
    assert predictions["score_orientation"].isin(
        ["normal", "inverted_by_inner_val", "normal_insufficient_inner_classes"]
    ).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_training_smoke.py::test_matrixnet_fast_smoke_produces_finite_predictions -q
```

Expected: FAIL because the new columns are absent.

- [ ] **Step 3: Implement prediction row columns**

Change `_eval_epoch` to return logits as well as scores and labels, or add `_predict_epoch` for inference. In `_run_one_fold`:

1. Evaluate inner validation logits/sigmoid scores.
2. Call `_calibrate_score_orientation(val_true, val_sigmoid_scores)` when enabled.
3. Apply orientation to validation scores before threshold selection.
4. Evaluate outer test logit/sigmoid score.
5. Apply the same orientation to produce `predicted_score`.
6. Add `label_int`, `logit`, `sigmoid_score`, and `score_orientation` to the prediction row.

- [ ] **Step 4: Run smoke test**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_training_smoke.py::test_matrixnet_fast_smoke_produces_finite_predictions -q
```

Expected: 1 passed.

## Task 4: RED tests for Phase 6.2 output files and report

**Files:**
- Modify: `tests/test_matrixnet_outputs.py`
- Modify later: `src/stroke_predict/matrixnet_training.py`

- [ ] **Step 1: Add failing Phase 6.2 output test**

Add a new test that builds a synthetic full-mode `MatrixNetRunResult`, calls `write_matrixnet_outputs` with `phase6_2_audit=True`, and asserts these file names exist:

```python
matrixnet_patient_predictions_phase6_2.csv
matrixnet_metrics_phase6_2.csv
seed_wise_metrics_phase6_2.csv
patient_averaged_metrics_phase6_2.csv
no_leakage_report_phase6_2.txt
phase6_2_score_direction_audit_report.md
```

Also assert report text contains:

```text
If label/score direction bug is found, fix it before SSL.
exploratory representation-learning experiment
Phase 6.2 did not start SSL
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_outputs.py::test_phase6_2_outputs_are_named_and_report_decision_rule -q
```

Expected: FAIL because Phase 6.2 output naming is not implemented.

- [ ] **Step 3: Implement Phase 6.2 writer path**

Extend `MatrixNetRunConfig` with:

```python
phase6_2_audit: bool = False
orientation_calibration: str = "inner_val_auc"
device: str = "cpu"
require_cuda: bool = False
```

Update `write_matrixnet_outputs`:

- If `phase6_2_audit`, use Phase 6.2 file names.
- Write `seed_wise_metrics_phase6_2.csv` and `patient_averaged_metrics_phase6_2.csv`.
- Write `_phase6_2_report`.
- Keep existing fast/full output names when `phase6_2_audit=False`.

- [ ] **Step 4: Run output tests**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_outputs.py -q
```

Expected: all output tests pass.

## Task 5: RED tests for CLI config device and Phase 6.2 flag

**Files:**
- Modify: `tests/test_matrixnet_outputs.py`
- Modify: `scripts/09_train_matrixnet.py`
- Modify: `configs/matrixnet.yaml`

- [ ] **Step 1: Add failing CLI smoke assertion**

Extend the temporary config in `test_matrixnet_script_fast_mode_with_fold_limit`:

```yaml
    device: cpu
    require_cuda: false
    orientation_calibration: inner_val_auc
```

Run script with:

```powershell
--phase6-2-audit
```

Assert stdout includes `phase6_2_audit=True` and the Phase 6.2 predictions path.

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_outputs.py::test_matrixnet_script_fast_mode_with_fold_limit -q
```

Expected: FAIL because CLI flag and config fields are missing.

- [ ] **Step 3: Implement CLI/config support**

Update `scripts/09_train_matrixnet.py`:

- Add `--phase6-2-audit` action flag.
- Read mode keys `device`, `require_cuda`, `orientation_calibration`.
- Pass them to `MatrixNetRunConfig`.
- Print `phase6_2_audit=True/False`.

Update `configs/matrixnet.yaml` full mode:

```yaml
    device: cuda
    require_cuda: true
    orientation_calibration: inner_val_auc
```

Update fast mode:

```yaml
    device: cpu
    require_cuda: false
    orientation_calibration: inner_val_auc
```

- [ ] **Step 4: Run CLI tests**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest tests/test_matrixnet_outputs.py::test_matrixnet_script_fast_mode_with_fold_limit -q
```

Expected: 1 passed.

## Task 6: Full unit test suite

**Files:**
- No planned production changes unless tests expose a bug.

- [ ] **Step 1: Run all tests**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest
```

Expected: all tests pass.

- [ ] **Step 2: Verify forbidden files are not tracked**

Run:

```powershell
git -C .worktrees/phase6-2-score-audit status --short
```

Expected: only code/docs/config/tests are modified; no `outputs/`, `.xlsx`, `.set`, `.fdt`.

## Task 7: Real-data Phase 6.2 no-SSL MatrixNet acceptance

**Files:**
- No code changes expected unless acceptance reveals a reproducible bug; if so, add a failing test before fixing.

- [ ] **Step 1: Prepare ignored outputs in the worktree**

If `.worktrees/phase6-2-score-audit/outputs` is absent, copy only ignored generated artifacts from the main checkout:

```powershell
Copy-Item -Recurse -Force outputs .worktrees/phase6-2-score-audit/
```

Do not stage copied outputs.

- [ ] **Step 2: Run Phase 6.2 full mode on CUDA**

Run:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python scripts/09_train_matrixnet.py --config configs/matrixnet.yaml --run-mode full --phase6-2-audit
```

Expected:

- The script refuses to continue if CUDA is unavailable because `require_cuda: true`.
- If CUDA is available, stdout contains `MATRIXNET_OK`, `run_mode=full`, `phase6_2_audit=True`.
- Prediction rows count is `5 models * 5 seeds * 19 folds = 475`.

- [ ] **Step 3: Inspect Phase 6.2 output files**

Run checks that verify:

- `outputs/predictions/matrixnet_patient_predictions_phase6_2.csv` exists.
- `outputs/evaluation/matrixnet_metrics_phase6_2.csv` exists.
- `outputs/evaluation/seed_wise_metrics_phase6_2.csv` exists.
- `outputs/evaluation/patient_averaged_metrics_phase6_2.csv` exists.
- `outputs/reports/no_leakage_report_phase6_2.txt` contains only PASS lines.
- `outputs/reports/phase6_2_score_direction_audit_report.md` contains the Phase 7 decision rules.

- [ ] **Step 4: Confirm no SSL or forbidden artifacts**

Run:

```powershell
git -C .worktrees/phase6-2-score-audit status --short
```

Expected: no `outputs/`, raw EEG, `.xlsx`, `.set`, `.fdt` staged.

## Task 8: Finish branch, merge main, push

**Files:**
- No new code expected.

- [ ] **Step 1: Use verification-before-completion**

Run fresh:

```powershell
Set-Location .worktrees/phase6-2-score-audit
python -m pytest
python scripts/09_train_matrixnet.py --config configs/matrixnet.yaml --run-mode full --phase6-2-audit
```

Only claim completion if both commands exit 0 and outputs pass acceptance checks.

- [ ] **Step 2: Commit code/docs/tests only**

Run:

```powershell
git -C .worktrees/phase6-2-score-audit add configs/matrixnet.yaml scripts/09_train_matrixnet.py src/stroke_predict/matrixnet_data.py src/stroke_predict/matrixnet_training.py tests/test_matrixnet_score_direction.py tests/test_matrixnet_outputs.py tests/test_matrixnet_training_smoke.py docs/superpowers/specs/2026-05-10-phase6-2-score-direction-audit-design.md docs/superpowers/plans/2026-05-10-phase6-2-score-direction-audit.md
git -C .worktrees/phase6-2-score-audit commit -m "feat: audit MatrixNet score direction"
```

- [ ] **Step 3: Merge and push main**

Run after verification:

```powershell
git -C .worktrees/phase6-main-merge checkout main
git -C .worktrees/phase6-main-merge merge codex/phase6-2-score-audit
python -m pytest
git -C .worktrees/phase6-main-merge push origin main
```

Do not merge or push if tests fail, if real-data acceptance fails, or if forbidden artifacts appear.

## Self-Review

- Spec coverage: label contract, score direction, prediction table, metrics consistency, orientation calibration, row alignment, real-data rerun, and Phase 7 decision rules are covered.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: helper names and config fields are consistent across tasks.
- Scope check: no SSL implementation or unplanned model family is included.
