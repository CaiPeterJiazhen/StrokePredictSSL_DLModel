# Phase 5.1 Feature Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair tACS and summary EEG features, split summary-vs-matrix ML baselines, and rerun LOPO ML evaluation before MatrixNet or SSL.

**Architecture:** Add focused summary-building helpers under `stroke_predict.features.summary`, expand `stroke_predict.features.tacs` to consume ROI-FC connectivity, and keep scripts thin. The ML layer receives explicit summary and flattened feature tables so model IDs cannot silently mix feature families.

**Tech Stack:** Python, NumPy, pandas, scikit-learn, pytest, existing project YAML configs.

---

### Task 1: tACS Summary Feature Tests

**Files:**
- Modify: `tests/test_tacs_features.py`
- Modify: `src/stroke_predict/features/tacs.py`

- [ ] **Step 1: Write failing tests** for target ROI PSD, log target-homologous features, target-homologous connectivity, target-to-midline/frontal/parietal connectivity, and EO/EC reactivity.
- [ ] **Step 2: Run** `python -m pytest tests/test_tacs_features.py -q` and confirm the new tests fail because columns are missing.
- [ ] **Step 3: Implement** the tACS feature expansion and keep existing target channel fields stable.
- [ ] **Step 4: Re-run** `python -m pytest tests/test_tacs_features.py -q` and confirm pass.

### Task 2: Summary Table Builders

**Files:**
- Create: `src/stroke_predict/features/summary.py`
- Create: `tests/test_summary_features.py`
- Modify: `scripts/06_build_handcrafted_features.py`

- [ ] **Step 1: Write failing tests** for PSD summary, FC summary, EO/EC reactivity, merged all-summary output, and dictionary rows.
- [ ] **Step 2: Run** `python -m pytest tests/test_summary_features.py -q` and confirm failure.
- [ ] **Step 3: Implement** summary builders using existing matrix arrays and feature config metadata.
- [ ] **Step 4: Re-run** `python -m pytest tests/test_summary_features.py -q` and confirm pass.

### Task 3: ML Baseline Split

**Files:**
- Modify: `src/stroke_predict/ml_models.py`
- Modify: `scripts/08_train_ml_baselines.py`
- Modify: `configs/models_ml.yaml`
- Modify: `tests/test_classical_ml_outputs.py`

- [ ] **Step 1: Update tests** so required model IDs include M3a/M4a/M5/M6 summary models and M3b/M4b/M6b flattened models.
- [ ] **Step 2: Run** `python -m pytest tests/test_classical_ml_outputs.py -q` and confirm failure.
- [ ] **Step 3: Implement** explicit feature table inputs for summary and flattened models.
- [ ] **Step 4: Re-run** `python -m pytest tests/test_classical_ml_outputs.py -q` and confirm pass.

### Task 4: Integration Verification

**Files:**
- No new files unless tests expose a missing assertion.

- [ ] **Step 1: Run** `python -m pytest tests --basetemp=F:\CJZProjectFile\StrokePredictSSL-DLModel\.codex_pytest_tmp -p no:cacheprovider`.
- [ ] **Step 2: Run** `python scripts\06_build_handcrafted_features.py --config configs\features.yaml`.
- [ ] **Step 3: Check** tACS connectivity columns have non-null values and required summary files exist.
- [ ] **Step 4: Run** `python scripts\07_make_folds.py --config configs\cv.yaml`.
- [ ] **Step 5: Run** `python scripts\08_train_ml_baselines.py --config configs\models_ml.yaml`.
- [ ] **Step 6: Check** every configured model has 19 LOPO predictions and no output CSV leaks private paths.
