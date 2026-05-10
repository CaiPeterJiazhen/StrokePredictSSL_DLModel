# Phase 8 Proportional Full-Edge FC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 8 proportional-residual FMA-UE labels, reduced32/full62 full-edge FC features, and leakage-safe LOPO classical classifiers for ProportionalRecovery vs PoorRecovery.

**Architecture:** Add focused Phase 8 modules beside the existing `stroke_predict` package. `phase8_labels.py` owns outcome definitions and label audit tables, `full_edge_fc.py` owns channel selection, edge metadata, and full-edge FC matrices, `phase8_models.py` owns fold-safe M14/M15/M16 model pipelines, `phase8_evaluation.py` owns patient-level metrics and no-leakage validation, and `phase8_reports.py` owns markdown reports and privacy checks. Scripts `12` through `14` expose label, FC, and model CLI entry points.

**Tech Stack:** Python, NumPy, pandas, SciPy signal functions, scikit-learn, pytest, YAML config, existing project config and privacy helpers.

---

## Required File Structure

- Create: `configs/phase8.yaml` for run modes, labels, FC settings, model grids, output paths, and M16 full-mode guard.
- Create: `src/stroke_predict/phase8_labels.py` for FMA-UE proportional residual records, sensitivity labels, train-median sensitivity, and label audit outputs.
- Create: `src/stroke_predict/full_edge_fc.py` for reduced32 channel selection, full62 channel handling, edge metadata, FC metrics, and canonical matrix building.
- Create: `src/stroke_predict/phase8_features.py` for matrix flattening, EO/EC feature table assembly, ROI-FC and summary comparison feature alignment.
- Create: `src/stroke_predict/phase8_models.py` for M14/M15/M16 model specs, fold-safe pipelines, inner CV, prediction writing, and full62 run guard.
- Create: `src/stroke_predict/phase8_evaluation.py` for Phase 8 metric rows, bootstrap CI, permutation, score-direction audit, confusion matrix fields, and no-leakage checks.
- Create: `src/stroke_predict/phase8_reports.py` for label audit markdown, final Phase 8 report, and public-output privacy checks.
- Create: `scripts/12_build_phase8_labels.py`.
- Create: `scripts/13_extract_full_edge_fc.py`.
- Create: `scripts/14_train_phase8_full_edge_models.py`.
- Create: `tests/test_phase8_proportional_labels.py`.
- Create: `tests/test_phase8_full_edge_fc.py`.
- Create: `tests/test_phase8_models_no_leakage.py`.
- Create: `tests/test_phase8_outputs.py`.
- Create: `tests/test_phase8_cli.py`.
- Created: `docs/superpowers/specs/2026-05-10-phase8-proportional-full-edge-fc-design.md`.
- Create: `docs/superpowers/plans/2026-05-10-phase8-proportional-full-edge-fc.md`.

## Global TDD Rule

For every behavior below, first write or extend a failing test, run the exact targeted pytest command, confirm the failure is caused by missing Phase 8 code or missing behavior, implement only the smallest code required, then rerun the targeted tests and keep them green. No production code is written before a failing test exists for that behavior.

## Task 1: Isolated Worktree And Baseline

**Files:**
- No code files changed.

- [x] **Step 1: Create isolated worktree**

Run:

```bash
git worktree add .worktrees/phase8-proportional-full-edge-fc -b codex/phase8-proportional-full-edge-fc main
```

Expected: worktree exists at `.worktrees/phase8-proportional-full-edge-fc` on branch `codex/phase8-proportional-full-edge-fc`.

- [x] **Step 2: Run baseline tests before edits**

Run from the Phase 8 worktree:

```bash
python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider
```

Expected: all existing tests pass before any Phase 8 edit.

## Task 2: Chinese Spec

**Files:**
- Create: `docs/superpowers/specs/2026-05-10-phase8-proportional-full-edge-fc-design.md`

- [x] **Step 1: Write the Chinese design spec**

The spec includes background, motivation, proportional-residual label formula, reduced32/full62 FC definitions, model families, leakage rules, evaluation plan, privacy rules, real-data acceptance, and explicit phase boundaries.

- [x] **Step 2: Run spec self-review**

Run:

```bash
Select-String -Path docs/superpowers/specs/2026-05-10-phase8-proportional-full-edge-fc-design.md -Pattern @('T'+'BD','T'+'ODO') -CaseSensitive:$false
$privateMarkers = @('\.'+'xlsx','\.'+'set\b','\.'+'fdt\b','[A-Za-z]:[\\/]')
Select-String -Path docs/superpowers/specs/2026-05-10-phase8-proportional-full-edge-fc-design.md -Pattern $privateMarkers -CaseSensitive:$false
git diff --check
```

Expected: no unfinished markers, no forbidden path-like strings, no whitespace errors.

## Task 3: TDD Plan

**Files:**
- Create: `docs/superpowers/plans/2026-05-10-phase8-proportional-full-edge-fc.md`

- [x] **Step 1: Write this plan**

The plan contains checkbox tasks, red-green order, exact commands, acceptance gates, commit scope, merge, and push steps.

- [ ] **Step 2: Run plan self-review**

Run:

```bash
Select-String -Path docs/superpowers/plans/2026-05-10-phase8-proportional-full-edge-fc.md -Pattern @('T'+'BD','T'+'ODO') -CaseSensitive:$false
$privateMarkers = @('\.'+'xlsx','\.'+'set\b','\.'+'fdt\b','[A-Za-z]:[\\/]')
Select-String -Path docs/superpowers/plans/2026-05-10-phase8-proportional-full-edge-fc.md -Pattern $privateMarkers -CaseSensitive:$false
git diff --check
```

Expected: no unfinished markers, no forbidden path-like strings, no whitespace errors.

## Task 4: RED Tests For Proportional Labels

**Files:**
- Test: `tests/test_phase8_proportional_labels.py`
- Create: `src/stroke_predict/phase8_labels.py`

- [ ] **Step 1: Write failing label tests**

Add tests that import the planned API:

```python
from stroke_predict.phase8_labels import (
    MAX_FMA_UE,
    build_phase8_label_table,
    compute_proportional_recovery_record,
    label_with_train_median_threshold,
)
```

Required assertions:

```python
def test_proportional_residual_formula_and_primary_label_direction():
    record = compute_proportional_recovery_record("STK-001", 40, 55, median_residual=10.0)
    assert record["expected_delta"] == 0.7 * (MAX_FMA_UE - 40)
    assert record["observed_delta"] == 15.0
    assert record["residual"] == 0.7 * 26 - 15
    assert record["primary_label_prop_residual"] == "ProportionalRecovery"
    assert record["primary_label_int_prop_residual"] == 1


def test_residual_tie_goes_to_proportional_and_above_median_is_poor():
    tied = compute_proportional_recovery_record("STK-002", 50, 55, median_residual=6.2)
    poor = compute_proportional_recovery_record("STK-003", 50, 53, median_residual=4.0)
    assert tied["residual"] == 6.2
    assert tied["primary_label_prop_residual"] == "ProportionalRecovery"
    assert poor["primary_label_prop_residual"] == "PoorRecovery"
    assert poor["primary_label_int_prop_residual"] == 0
```

Also cover ceiling exclusion, missing baseline/post exclusion, label integer encoding, sensitivity labels, and train-only median threshold for a held-out patient.

- [ ] **Step 2: Run RED label tests**

Run:

```bash
python -m pytest tests/test_phase8_proportional_labels.py -q
```

Expected: fail because `stroke_predict.phase8_labels` is not implemented.

- [ ] **Step 3: Implement minimal label builder**

Implement constants and functions:

```python
MAX_FMA_UE = 66
LABEL_TO_INT_PROP_RESIDUAL = {"PoorRecovery": 0, "ProportionalRecovery": 1}

def compute_proportional_recovery_record(subject_id, baseline_fma, post_fma, *, median_residual):
    ...

def build_phase8_label_table(cohort, *, subject_col="subject_id", baseline_col="baseline_fma", post_col="post_fma", current_label_col="label_primary"):
    ...

def label_with_train_median_threshold(label_table, train_subjects, test_subject):
    ...
```

Use deterministic median calculation on analyzable non-ceiling rows, mark `ceiling_exclude` and `excluded_missing`, and keep old current label only in `current_clinically_meaningful`.

- [ ] **Step 4: Run GREEN label tests**

Run:

```bash
python -m pytest tests/test_phase8_proportional_labels.py -q
```

Expected: pass.

## Task 5: RED Tests For Full-Edge FC

**Files:**
- Test: `tests/test_phase8_full_edge_fc.py`
- Create: `src/stroke_predict/full_edge_fc.py`

- [ ] **Step 1: Write failing full-edge tests**

Add tests that import:

```python
from stroke_predict.full_edge_fc import (
    PHASE8_BANDS,
    PHASE8_FC_METHODS,
    build_canonical_full_edge_matrix,
    build_full_edge_index,
    compute_full_edge_fc,
    select_reduced32_channels,
)
```

Required assertions:

```python
def test_full_edge_count_matches_channel_pairs():
    channels = ["Fp1", "Fp2", "F3", "F4", "C3"]
    edge_index = build_full_edge_index(channels)
    assert len(edge_index) == len(channels) * (len(channels) - 1) // 2
    assert {"edge_index", "ch_i", "ch_j"} <= set(edge_index.columns)


def test_reduced32_selector_is_deterministic_and_writes_metadata(tmp_path):
    channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC1", "FC2", "FC6", "T7", "C3", "Cz", "C4", "T8", "CP5", "CP1", "CP2", "CP6", "P7", "P3", "Pz", "P4", "P8", "POz", "O1", "Oz", "O2", "AF3", "AF4", "PO3"]
    first = select_reduced32_channels(channels, output_csv=tmp_path / "selection.csv")
    second = select_reduced32_channels(list(reversed(channels)), output_csv=tmp_path / "selection2.csv")
    assert first.selected_channels == second.selected_channels
    assert first.n_channels == 32
```

Also cover canonical matrix shape `[N, C, edges, bands]`, absence of private local paths in edge metadata, and finite coherence, imaginary coherence, and wPLI values or explicit missing reasons.

- [ ] **Step 2: Run RED full-edge tests**

Run:

```bash
python -m pytest tests/test_phase8_full_edge_fc.py -q
```

Expected: fail because `stroke_predict.full_edge_fc` is not implemented.

- [ ] **Step 3: Implement minimal full-edge FC**

Implement:

```python
PHASE8_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha_mu": (8.0, 13.0),
    "low_beta": (13.0, 20.0),
    "high_beta": (20.0, 30.0),
    "broad_beta": (13.0, 30.0),
}
PHASE8_FC_METHODS = ("coherence", "imaginary_coherence", "wpli")
```

Use deterministic channel normalization, fail if fewer than 24 reduced32 channels are available, build edge metadata with `edge_type`, compute spectral FC with SciPy, and build canonical matrices with condition-metric channels.

- [ ] **Step 4: Run GREEN full-edge tests**

Run:

```bash
python -m pytest tests/test_phase8_full_edge_fc.py -q
```

Expected: pass.

## Task 6: RED Tests For Fold-Safe Models And No Leakage

**Files:**
- Test: `tests/test_phase8_models_no_leakage.py`
- Create: `src/stroke_predict/phase8_features.py`
- Create: `src/stroke_predict/phase8_models.py`
- Create: `src/stroke_predict/phase8_evaluation.py`

- [ ] **Step 1: Write failing model/no-leakage tests**

Add tests that import:

```python
from stroke_predict.phase8_features import align_full_edge_features
from stroke_predict.phase8_models import Phase8ModelSpec, run_phase8_lopo_models
from stroke_predict.phase8_evaluation import validate_phase8_no_leakage, validate_phase8_patient_predictions
```

Required assertions:

```python
def test_phase8_lopo_excludes_outer_test_from_all_fit_steps():
    result = run_phase8_lopo_models(
        config=_toy_model_config(models=["M14a_prop_reduced32_fullfc_ridge_logistic"], bootstrap=5, permutations=5),
        features=_toy_features(),
        labels=_toy_labels(),
        folds=_toy_folds(),
        run_mode="fast",
    )
    audit = result.no_leakage_audit
    assert not audit["outer_test_in_fit_subjects"].any()
    assert not audit["outer_test_in_transform_fit_subjects"].any()
    assert not audit["outer_test_in_inner_cv_subjects"].any()
```

Also cover no duplicated model-patient predictions, patient-level rows, matrix subject index alignment, label subject alignment, variance filter/scaler/imputer fit only on outer train, and full-mode refusal for unplanned M16 full62 training.

- [ ] **Step 2: Run RED model tests**

Run:

```bash
python -m pytest tests/test_phase8_models_no_leakage.py -q
```

Expected: fail because Phase 8 model modules are not implemented.

- [ ] **Step 3: Implement minimal fold-safe model stack**

Implement data classes and functions:

```python
@dataclass(frozen=True)
class Phase8ModelSpec:
    model_id: str
    feature_set: str
    estimator: str
    c_values: tuple[float, ...] = (0.1, 1.0)
    l1_ratios: tuple[float, ...] = (0.0,)
    pls_components: tuple[int, ...] = (1, 2)

def run_phase8_lopo_models(config, features, labels, folds, *, run_mode, fold_limit=None, feature_set="reduced32"):
    ...
```

Use `Pipeline` with `SimpleImputer`, `VarianceThreshold`, `StandardScaler`, and estimator. Select hyperparameters inside outer train with inner leave-one-out or deterministic small folds. Use `LogisticRegression` for ridge and elastic-net, `LinearSVC` decision function converted to min-max train-side score if needed, and PLS scores with a logistic head. Record every subject used by each fit step in `no_leakage_audit`.

- [ ] **Step 4: Run GREEN model tests**

Run:

```bash
python -m pytest tests/test_phase8_models_no_leakage.py -q
```

Expected: pass.

## Task 7: RED Tests For Outputs And Reports

**Files:**
- Test: `tests/test_phase8_outputs.py`
- Create: `src/stroke_predict/phase8_reports.py`
- Modify: `src/stroke_predict/phase8_evaluation.py`

- [ ] **Step 1: Write failing output/report tests**

Required assertions:

```python
from stroke_predict.phase8_reports import write_phase8_label_audit, write_phase8_model_report

def test_phase8_required_outputs_are_generated(tmp_path):
    paths = write_phase8_label_audit(_toy_label_table(), output_dir=tmp_path)
    model_paths = write_phase8_model_report(_toy_predictions(), _toy_metrics(), _toy_fc_audit(), output_dir=tmp_path)
    required = [
        tmp_path / "reports" / "phase8_label_audit.md",
        tmp_path / "evaluation" / "phase8_label_audit.csv",
        tmp_path / "evaluation" / "phase8_label_transition_table.csv",
        tmp_path / "predictions" / "phase8_prop_full_edge_patient_predictions.csv",
        tmp_path / "evaluation" / "phase8_prop_full_edge_metrics.csv",
        tmp_path / "reports" / "phase8_proportional_full_edge_fc_report.md",
    ]
    assert all(path.exists() for path in required)
```

Also cover required columns, no forbidden private strings in Phase 8 public outputs, no output path selected for git staging, score direction audit fields, and scientific caution text when permutation p-value is not significant.

- [ ] **Step 2: Run RED output tests**

Run:

```bash
python -m pytest tests/test_phase8_outputs.py -q
```

Expected: fail because report writer is not implemented.

- [ ] **Step 3: Implement output and report writers**

Implement label audit CSV, transition table, markdown audit, prediction CSV, metrics CSV, final markdown report, privacy guard, and scientific caution text. Keep all run artifacts under ignored `outputs/`.

- [ ] **Step 4: Run GREEN output tests**

Run:

```bash
python -m pytest tests/test_phase8_outputs.py -q
```

Expected: pass.

## Task 8: RED Tests For CLI Scripts

**Files:**
- Test: `tests/test_phase8_cli.py`
- Create: `scripts/12_build_phase8_labels.py`
- Create: `scripts/13_extract_full_edge_fc.py`
- Create: `scripts/14_train_phase8_full_edge_models.py`
- Create: `configs/phase8.yaml`

- [ ] **Step 1: Write failing CLI tests**

Use toy data under `tmp_path` and run scripts with subprocess:

```python
def test_phase8_label_cli_runs_on_toy_data(tmp_path):
    config = _write_toy_phase8_config(tmp_path)
    result = subprocess.run(
        [sys.executable, "scripts/12_build_phase8_labels.py", "--config", str(config), "--run-mode", "fast"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "PHASE8_LABELS_OK" in result.stdout
```

Also cover FC extraction fast mode on synthetic matrices, training fast mode with `--fold-limit 2`, and full-mode refusal for M16 full62 unless config explicitly enables it.

- [ ] **Step 2: Run RED CLI tests**

Run:

```bash
python -m pytest tests/test_phase8_cli.py -q
```

Expected: fail because scripts and config do not exist.

- [ ] **Step 3: Implement CLI scripts and config**

Scripts must resolve config-relative paths, load project config when real outputs are used, support toy paths for tests, accept `--run-mode fast|full`, and expose:

```bash
python scripts/12_build_phase8_labels.py --config configs/phase8.yaml --run-mode fast
python scripts/13_extract_full_edge_fc.py --config configs/phase8.yaml --run-mode fast --feature-set reduced32
python scripts/14_train_phase8_full_edge_models.py --config configs/phase8.yaml --run-mode fast --fold-limit 2
```

- [ ] **Step 4: Run GREEN CLI tests**

Run:

```bash
python -m pytest tests/test_phase8_cli.py -q
```

Expected: pass.

## Task 9: Targeted Phase 8 Test Group

**Files:**
- All Phase 8 tests and modules.

- [ ] **Step 1: Run all Phase 8 tests**

Run:

```bash
python -m pytest tests/test_phase8_proportional_labels.py tests/test_phase8_full_edge_fc.py tests/test_phase8_models_no_leakage.py tests/test_phase8_outputs.py tests/test_phase8_cli.py -q
```

Expected: pass.

- [ ] **Step 2: Run whitespace and import checks**

Run:

```bash
git diff --check
python -m pytest tests/test_privacy_ids.py tests/test_feature_pipeline_no_leakage.py tests/test_splits_no_leakage.py -q
```

Expected: pass.

## Task 10: Real-Data Fast Acceptance

**Files:**
- Runtime outputs only under ignored `outputs/`.

- [ ] **Step 1: Build fast labels**

Run:

```bash
python scripts/12_build_phase8_labels.py --config configs/phase8.yaml --run-mode fast
```

Expected: `PHASE8_LABELS_OK`, label audit generated, no private strings in public outputs.

- [ ] **Step 2: Build fast reduced32 full-edge FC**

Run:

```bash
python scripts/13_extract_full_edge_fc.py --config configs/phase8.yaml --run-mode fast --feature-set reduced32
```

Expected: reduced32 channel selection, edge index, EO matrix, EC matrix, and matrix subject index generated.

- [ ] **Step 3: Train fast models with fold limit**

Run:

```bash
python scripts/14_train_phase8_full_edge_models.py --config configs/phase8.yaml --run-mode fast --fold-limit 2
```

Expected: predictions, metrics, no-leakage audit, and final report generated with no scientific interpretation.

## Task 11: Real-Data Full Acceptance

**Files:**
- Runtime outputs only under ignored `outputs/`.

- [ ] **Step 1: Build full labels**

Run:

```bash
python scripts/12_build_phase8_labels.py --config configs/phase8.yaml --run-mode full
```

Expected: full label audit generated with analyzable patient count, median residual, and class counts.

- [ ] **Step 2: Build full reduced32 full-edge FC**

Run:

```bash
python scripts/13_extract_full_edge_fc.py --config configs/phase8.yaml --run-mode full --feature-set reduced32
```

Expected: reduced32 full-edge matrices complete for EO and EC.

- [ ] **Step 3: Train full primary models**

Run:

```bash
python scripts/14_train_phase8_full_edge_models.py --config configs/phase8.yaml --run-mode full
```

Expected: M14a, M14b, M14c, M14d, M15a, and M15b complete with patient-level LOPO, bootstrap 1000, permutation 1000, no-leakage audit pass, metrics, predictions, and report generated.

## Task 12: Optional 62-Channel Smoke

**Files:**
- Runtime outputs only under ignored `outputs/`.

- [ ] **Step 1: Generate full62 fast FC**

Run:

```bash
python scripts/13_extract_full_edge_fc.py --config configs/phase8.yaml --run-mode fast --feature-set full62
```

Expected: full62 edge index and smoke matrix outputs generated or explicit documented deferral if runtime/data are unavailable.

- [ ] **Step 2: Train full62 fast smoke models**

Run:

```bash
python scripts/14_train_phase8_full_edge_models.py --config configs/phase8.yaml --run-mode fast --feature-set full62 --fold-limit 2
```

Expected: M16 fast smoke completes or exits with a documented data availability reason. Full-mode M16 remains blocked unless explicitly enabled in config.

## Task 13: Full Test Suite

**Files:**
- All tests.

- [ ] **Step 1: Run full suite**

Run:

```bash
python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider
```

Expected: all tests pass.

## Task 14: Privacy Scan And Git Staging Guard

**Files:**
- Git index and tracked Phase 8 files.

- [ ] **Step 1: Verify outputs are not staged**

Run:

```bash
git status --short
git diff --cached --name-only
```

Expected: no path under `outputs/` appears in staged files.

- [ ] **Step 2: Verify forbidden artifacts are absent from staged files**

Run:

```bash
$privateArtifactMarkers = @('(^|/)outputs/','\.'+'xlsx$','\.'+'xls$','\.'+'set$','\.'+'fdt$','checkpoint','r'+'aw')
git diff --cached --name-only | Select-String -Pattern $privateArtifactMarkers -CaseSensitive:$false
```

Expected: no matches.

- [ ] **Step 3: Verify new tracked Phase 8 files contain no private path-like strings**

Run:

```bash
git diff --name-only -- configs/phase8.yaml src/stroke_predict/phase8_labels.py src/stroke_predict/full_edge_fc.py src/stroke_predict/phase8_features.py src/stroke_predict/phase8_models.py src/stroke_predict/phase8_evaluation.py src/stroke_predict/phase8_reports.py scripts/12_build_phase8_labels.py scripts/13_extract_full_edge_fc.py scripts/14_train_phase8_full_edge_models.py tests/test_phase8_proportional_labels.py tests/test_phase8_full_edge_fc.py tests/test_phase8_models_no_leakage.py tests/test_phase8_outputs.py tests/test_phase8_cli.py docs/superpowers/specs/2026-05-10-phase8-proportional-full-edge-fc-design.md docs/superpowers/plans/2026-05-10-phase8-proportional-full-edge-fc.md
```

Expected: only intended code/docs/tests/config files are modified. Public output content privacy is enforced by tests and report writers.

## Task 15: Commit Code, Docs, And Tests Only

**Files:**
- Allowed commit set only.

- [ ] **Step 1: Stage allowed files**

Run:

```bash
git add configs/phase8.yaml src/stroke_predict/phase8_labels.py src/stroke_predict/full_edge_fc.py src/stroke_predict/phase8_features.py src/stroke_predict/phase8_models.py src/stroke_predict/phase8_evaluation.py src/stroke_predict/phase8_reports.py scripts/12_build_phase8_labels.py scripts/13_extract_full_edge_fc.py scripts/14_train_phase8_full_edge_models.py tests/test_phase8_proportional_labels.py tests/test_phase8_full_edge_fc.py tests/test_phase8_models_no_leakage.py tests/test_phase8_outputs.py tests/test_phase8_cli.py docs/superpowers/specs/2026-05-10-phase8-proportional-full-edge-fc-design.md docs/superpowers/plans/2026-05-10-phase8-proportional-full-edge-fc.md
```

Expected: no outputs, private artifacts, data files, or model weight artifacts staged.

- [ ] **Step 2: Commit**

Run:

```bash
git commit -m "feat: add Phase 8 proportional full-edge FC pipeline"
```

Expected: one commit on `codex/phase8-proportional-full-edge-fc`.

## Task 16: Merge Main And Push Remote

**Files:**
- Git branches.

- [ ] **Step 1: Merge only after all gates pass**

Run:

```bash
git checkout main
git pull --ff-only
git merge codex/phase8-proportional-full-edge-fc
python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider
```

Expected: merge succeeds and full suite passes on `main`.

- [ ] **Step 2: Push main**

Run:

```bash
git push origin main
```

Expected: remote `main` updated. Do not push if tests, real-data acceptance, no-leakage, privacy scan, or staging guard fails.
