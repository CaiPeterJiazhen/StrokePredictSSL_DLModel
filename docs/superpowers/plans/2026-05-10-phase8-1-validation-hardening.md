# Phase 8.1 Validation Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate and harden Phase 8 proportional-recovery full-edge FC results without starting MatrixNet, SSL, or manuscript figure generation.

**Architecture:** Add a focused `phase8_1_validation.py` module for source-mode audit, duplicate model audit, multiple-comparison correction, threshold/calibration analysis, patient error audit, and no-leakage summary. Extend Phase 8 extraction and reporting only where needed to fail loudly on missing real time-series input and to write the Phase 8.1 report files. Keep generated validation artifacts under ignored `outputs/`.

**Tech Stack:** Python, NumPy, pandas, scikit-learn metrics, pytest, existing Phase 8 scripts and report helpers.

---

## File Structure

- Create: `src/stroke_predict/phase8_1_validation.py`
- Create: `tests/test_phase8_1_validation.py`
- Modify: `scripts/13_extract_full_edge_fc.py`
- Modify: `scripts/14_train_phase8_full_edge_models.py`
- Modify: `src/stroke_predict/phase8_reports.py`
- Modify: `configs/phase8.yaml`
- Create: `docs/superpowers/specs/2026-05-10-phase8-1-validation-hardening-design.md`
- Create: `docs/superpowers/plans/2026-05-10-phase8-1-validation-hardening.md`

## Task 1: Source Mode Audit Tests

**Files:**
- Test: `tests/test_phase8_1_validation.py`
- Create: `src/stroke_predict/phase8_1_validation.py`

- [ ] **Step 1: Write failing tests**

Add tests for `audit_source_mode`:

```python
def test_source_mode_proxy_blocks_time_series_claims():
    audit = audit_source_mode(pd.DataFrame({"source_mode": ["psd_artifact_proxy"], "n_edges": [496]}))
    assert audit["source_mode"] == "psd_artifact_proxy"
    assert audit["is_real_time_series_fc"] is False
    assert "must not be called real time-series" in audit["claim_guard"]
```

Also add a CLI-level test that running script 13 with `--require-real-timeseries` and no time-series input exits nonzero with a missing time-series message.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_source_mode_proxy_blocks_time_series_claims -q
```

Expected: import failure for `phase8_1_validation`.

- [ ] **Step 3: Implement minimal source audit**

Implement `audit_source_mode(fc_audit: pd.DataFrame | dict[str, object]) -> dict[str, object]` and add `--require-real-timeseries` to script 13. If the flag is present and neither `toy_eeg_npz` nor `baseline_timeseries_index` is configured, raise `ValueError("Real time-series full-edge FC requires baseline EO/EC time-series input")`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_source_mode_proxy_blocks_time_series_claims tests/test_phase8_cli.py -q
```

Expected: pass.

## Task 2: M15 Duplicate Audit Tests

**Files:**
- Test: `tests/test_phase8_1_validation.py`
- Modify: `src/stroke_predict/phase8_1_validation.py`

- [ ] **Step 1: Write failing tests**

Add tests for `audit_comparison_models`:

```python
def test_m15_audit_reports_different_features_but_identical_predictions():
    audit = audit_comparison_models(
        features=_comparison_feature_table(different=True),
        predictions=_comparison_predictions(identical=True),
        model_a="M15a_prop_roi_fc_best_ml",
        model_b="M15b_prop_summary_eeg_best_ml",
    )
    assert audit["feature_matrices_identical"] is False
    assert audit["predictions_identical"] is True
    assert "same predictions despite different feature matrices" in audit["explanation"]
```

Add a second test where identical features and identical predictions raise unless `allow_intentional_shared_predictions=True`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_m15_audit_reports_different_features_but_identical_predictions -q
```

Expected: missing function failure.

- [ ] **Step 3: Implement duplicate audit**

Hash numeric feature matrices by model prefix, hash model predictions by patient order, compare M15a/M15b, and raise on silent shared feature/prediction reuse.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_m15_audit_reports_different_features_but_identical_predictions tests/test_phase8_1_validation.py::test_m15_audit_rejects_silent_shared_predictions -q
```

Expected: pass.

## Task 3: Multiple-Comparison Correction Tests

**Files:**
- Test: `tests/test_phase8_1_validation.py`
- Modify: `src/stroke_predict/phase8_1_validation.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
def test_multiple_comparison_correction_adds_raw_bonferroni_and_fdr_flags():
    corrected = apply_multiple_comparison_correction(_permutation_table())
    assert {
        "raw_permutation_p_value",
        "bonferroni_p_value",
        "fdr_q_value",
        "nominal_p_lt_0_05",
        "fdr_q_lt_0_05",
        "bonferroni_p_lt_0_05",
    } <= set(corrected.columns)
    assert corrected.loc[corrected["model_id"].eq("M14b"), "bonferroni_p_value"].iloc[0] == 0.045 * 6
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_multiple_comparison_correction_adds_raw_bonferroni_and_fdr_flags -q
```

Expected: missing function failure.

- [ ] **Step 3: Implement correction**

Implement Bonferroni and Benjamini-Hochberg on non-missing raw p-values. Preserve model order.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_multiple_comparison_correction_adds_raw_bonferroni_and_fdr_flags -q
```

Expected: pass.

## Task 4: Threshold And Calibration Tests

**Files:**
- Test: `tests/test_phase8_1_validation.py`
- Modify: `src/stroke_predict/phase8_1_validation.py`

- [ ] **Step 1: Write failing tests**

Add tests for `build_threshold_calibration_table` using predictions with fold-safe threshold columns:

```python
def test_threshold_calibration_outputs_fixed_inner_youden_calibration_and_distribution():
    table = build_threshold_calibration_table(_best_model_predictions())
    assert set(table["analysis_type"]) >= {
        "fixed_0.5_threshold",
        "inner_cv_threshold",
        "inner_cv_youden_threshold",
        "calibration_bin",
        "score_distribution_by_group",
    }
    assert "brier_score" in table.columns
```

Also test that missing inner threshold columns produce `not_available` rows rather than recomputing from outer test predictions.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_threshold_calibration_outputs_fixed_inner_youden_calibration_and_distribution -q
```

Expected: missing function failure.

- [ ] **Step 3: Implement threshold/calibration table**

Use supplied `threshold`, `inner_cv_threshold`, and `inner_cv_youden_threshold` columns only. For calibration bins, use score bins independent of labels and compute observed rate per bin. Compute Brier score from scores and `y_true`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_threshold_calibration_outputs_fixed_inner_youden_calibration_and_distribution tests/test_phase8_1_validation.py::test_missing_inner_thresholds_are_reported_not_recomputed -q
```

Expected: pass.

## Task 5: Patient Error Audit Tests

**Files:**
- Test: `tests/test_phase8_1_validation.py`
- Modify: `src/stroke_predict/phase8_1_validation.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
def test_patient_error_audit_contains_required_columns_and_boundary_flags():
    audit = build_patient_error_audit(_labels_for_error_audit(), _best_model_predictions())
    assert {
        "patient_id",
        "old_label",
        "proportional_label",
        "baseline_fma",
        "post_fma",
        "observed_delta",
        "expected_delta",
        "residual",
        "predicted_score",
        "predicted_label",
        "correct",
        "rank",
        "near_median_threshold",
        "old_new_label_disagree",
    } <= set(audit.columns)
    assert audit["near_median_threshold"].any()
```

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_patient_error_audit_contains_required_columns_and_boundary_flags -q
```

Expected: missing function failure.

- [ ] **Step 3: Implement patient audit**

Merge predictions with Phase 8 label audit on patient ID, rank by predicted score descending, compute correctness, label disagreement, and boundary proximity.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_patient_error_audit_contains_required_columns_and_boundary_flags -q
```

Expected: pass.

## Task 6: Phase 8.1 Report Writer Tests

**Files:**
- Test: `tests/test_phase8_1_validation.py`
- Modify: `src/stroke_predict/phase8_1_validation.py`
- Modify: `src/stroke_predict/phase8_reports.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
def test_phase8_1_report_files_are_written_and_answer_required_questions(tmp_path):
    paths = write_phase8_1_validation_outputs(..., output_dir=tmp_path)
    assert (tmp_path / "reports" / "phase8_1_validation_report.md").exists()
    assert (tmp_path / "evaluation" / "phase8_1_multiple_comparison_correction.csv").exists()
    assert (tmp_path / "evaluation" / "phase8_1_threshold_calibration.csv").exists()
    assert (tmp_path / "evaluation" / "phase8_1_patient_error_audit.csv").exists()
    assert (tmp_path / "reports" / "phase8_1_no_leakage_report.txt").exists()
```

Assert the report includes all nine required answer prompts and the no-leakage report includes `PASS`.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_phase8_1_report_files_are_written_and_answer_required_questions -q
```

Expected: missing function failure.

- [ ] **Step 3: Implement output writer**

Write CSV/text/markdown outputs under `output_dir`, run existing public-output privacy guard on CSV frames, and generate conservative decision text when source mode is proxy or corrected p-values are not significant.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py::test_phase8_1_report_files_are_written_and_answer_required_questions -q
```

Expected: pass.

## Task 7: Integrate Script 14

**Files:**
- Modify: `scripts/14_train_phase8_full_edge_models.py`
- Test: `tests/test_phase8_cli.py`

- [ ] **Step 1: Write failing integration test**

Extend the Phase 8 model CLI toy test to assert all Phase 8.1 output files exist after script 14 completes.

- [ ] **Step 2: Verify RED**

Run:

```bash
python -m pytest tests/test_phase8_cli.py::test_phase8_model_cli_runs_fast_fold_limit -q
```

Expected: fail because Phase 8.1 outputs are not yet generated.

- [ ] **Step 3: Implement script integration**

After metrics are produced, call Phase 8.1 validation helpers with labels, features, predictions, corrected p-values, no-leakage audit, and FC audit. Write all Phase 8.1 outputs.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
python -m pytest tests/test_phase8_cli.py::test_phase8_model_cli_runs_fast_fold_limit tests/test_phase8_1_validation.py -q
```

Expected: pass.

## Task 8: Full Tests And Acceptance

**Files:**
- All code, docs, config, tests.

- [ ] **Step 1: Run targeted tests**

Run:

```bash
python -m pytest tests/test_phase8_1_validation.py tests/test_phase8_cli.py tests/test_phase8_outputs.py tests/test_phase8_models_no_leakage.py -q
```

Expected: pass.

- [ ] **Step 2: Run full test suite**

Run:

```bash
python -m pytest
```

Expected: all tests pass.

- [ ] **Step 3: Run real-data acceptance without MatrixNet or SSL**

Run:

```bash
python scripts/12_build_phase8_labels.py --config configs/phase8.yaml --run-mode full
python scripts/13_extract_full_edge_fc.py --config configs/phase8.yaml --run-mode full --feature-set reduced32
python scripts/14_train_phase8_full_edge_models.py --config configs/phase8.yaml --run-mode full
```

Expected: Phase 8.1 outputs exist, source mode is audited, no-leakage report passes, and no MatrixNet/SSL scripts are invoked.

- [ ] **Step 4: Privacy and staging guard**

Run:

```bash
git status --short
git diff --check
```

Expected: only code/config/docs/tests changed, no `outputs/` or private artifacts staged.

## Task 9: Merge And Push Gate

**Files:**
- Git branches only.

- [ ] **Step 1: Stage allowed files only**

Stage only:

```bash
git add configs/phase8.yaml scripts/13_extract_full_edge_fc.py scripts/14_train_phase8_full_edge_models.py src/stroke_predict/phase8_1_validation.py src/stroke_predict/phase8_reports.py tests/test_phase8_1_validation.py tests/test_phase8_cli.py docs/superpowers/specs/2026-05-10-phase8-1-validation-hardening-design.md docs/superpowers/plans/2026-05-10-phase8-1-validation-hardening.md
```

Expected: no outputs, Excel, EEG binaries, checkpoints, local paths, or private identifiers staged.

- [ ] **Step 2: Commit only after all gates pass**

Run:

```bash
git commit -m "feat: add Phase 8.1 validation hardening"
```

Expected: one commit on the Phase 8.1 branch.

- [ ] **Step 3: Merge to main and push only after validation passes**

Run:

```bash
git checkout main
git merge codex/phase8-1-validation-hardening
python -m pytest
git push origin main
```

Expected: `main` contains the validated code, tests pass after merge, and remote push succeeds.
