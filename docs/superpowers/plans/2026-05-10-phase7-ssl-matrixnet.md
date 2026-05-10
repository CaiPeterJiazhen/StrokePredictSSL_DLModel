# Phase 7 SSL-MatrixNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build leakage-safe masked-matrix SSL pretraining for MatrixNet PSD/FC encoders and fine-tune SSL-A on the existing baseline EO/EC Good/Poor prognosis task.

**Architecture:** Add new Phase 7 modules beside the existing Phase 6 MatrixNet stack. `ssl_matrixnet_data.py` owns de-identified SSL matrix indexes and fold pools, `ssl_matrixnet.py` owns masking and encoder/reconstruction modules, and `ssl_matrixnet_training.py` owns pretraining, checkpoint loading, fine-tuning orchestration, metrics, reports, and privacy guards. Existing Phase 6 supervised MatrixNet code is reused where practical and extended only at the encoder-loading seam.

**Tech Stack:** Python, NumPy, pandas, PyTorch, scikit-learn metrics, pytest, YAML config, existing `stroke_predict.matrixnet*` modules.

---

## Required File Structure

- Create: `src/stroke_predict/ssl_matrixnet.py` for mask generation, masked MSE, SSL encoder/reconstruction model, and MatrixNet checkpoint loading helpers.
- Create: `src/stroke_predict/ssl_matrixnet_data.py` for SSL matrix index validation, variant eligibility, fold-safe pool building, audit tables, and privacy scanning.
- Create: `src/stroke_predict/ssl_matrixnet_training.py` for SSL pretraining, SSL fine-tuning, output writing, Phase 7 metrics, reports, and CUDA guard logic.
- Create: `scripts/10_pretrain_ssl_matrixnet.py` for pretraining CLI.
- Create: `scripts/11_train_ssl_matrixnet.py` for supervised fine-tuning CLI.
- Create: `configs/ssl_matrixnet.yaml` for fast/full Phase 7 defaults.
- Create: `tests/test_ssl_matrixnet_masking.py`.
- Create: `tests/test_ssl_matrixnet_pooling_no_leakage.py`.
- Create: `tests/test_ssl_matrixnet_data.py`.
- Create: `tests/test_ssl_matrixnet_pretrain.py`.
- Create: `tests/test_ssl_matrixnet_finetune.py`.
- Create: `tests/test_ssl_matrixnet_outputs.py`.
- Create: `tests/test_ssl_matrixnet_cli.py`.
- Modify only if needed: `src/stroke_predict/matrixnet.py` to expose branch state loading without changing Phase 6 behavior.

## Global TDD Rule

For each implementation task below, write or extend the listed test first, run the named test and verify it fails for the expected missing symbol or missing behavior, implement the smallest code that satisfies it, then rerun that test and the targeted Phase 7 group. No production code is written before a failing test exists for that behavior.

### Task 1: Chinese Spec

**Files:**
- Created: `docs/superpowers/specs/2026-05-10-phase7-ssl-matrixnet-design.md`

- [x] **Step 1: Write the Chinese design spec**

The spec records objective, exploratory status, SSL data sources, leakage rules, method, model families, training/evaluation, outputs, privacy rules, real-data acceptance, and the statement that post-treatment EEG is never supervised input.

- [x] **Step 2: Review the spec**

Run: `$markers=@('T'+'BD','T'+'ODO'); Select-String -Path docs/superpowers/specs/2026-05-10-phase7-ssl-matrixnet-design.md -Pattern $markers -CaseSensitive:$false`

Expected: no actual placeholder markers in unfinished sections.

- [x] **Step 3: Commit the spec**

Run:

```bash
git add docs/superpowers/specs/2026-05-10-phase7-ssl-matrixnet-design.md
git commit -m "docs: add Phase 7 SSL MatrixNet spec"
```

### Task 2: TDD Plan

**Files:**
- Create: `docs/superpowers/plans/2026-05-10-phase7-ssl-matrixnet.md`

- [ ] **Step 1: Write this plan**

Include every Phase 7 task in red-green order and include real-data acceptance, merge, and push gates.

- [ ] **Step 2: Run plan self-review**

Run:

```bash
$markers=@('T'+'BD','T'+'ODO'); Select-String -Path docs/superpowers/plans/2026-05-10-phase7-ssl-matrixnet.md -Pattern $markers -CaseSensitive:$false
git diff --check
```

Expected: no unfinished plan markers and no whitespace errors.

- [ ] **Step 3: Commit the plan**

Run:

```bash
git add docs/superpowers/plans/2026-05-10-phase7-ssl-matrixnet.md
git commit -m "docs: add Phase 7 SSL MatrixNet TDD plan"
```

### Task 3: SSL Data Index And No-Leakage Pools

**Files:**
- Test: `tests/test_ssl_matrixnet_data.py`
- Test: `tests/test_ssl_matrixnet_pooling_no_leakage.py`
- Create: `src/stroke_predict/ssl_matrixnet_data.py`

- [ ] **Step 1: Write failing data-index tests**

Add tests that create synthetic de-identified records with columns `subject_id`, `source`, `stage`, `condition`, `psd_path_redacted`, `fc_path_redacted`, and variant eligibility. Assertions:

```python
from stroke_predict.ssl_matrixnet_data import (
    SSL_VARIANTS,
    build_ssl_fold_pools,
    validate_ssl_matrix_index,
)

def test_ssl_matrix_index_has_required_public_columns(tmp_path):
    index = _synthetic_ssl_index()
    validated = validate_ssl_matrix_index(index)
    assert {"subject_id", "source", "stage", "condition"} <= set(validated.columns)
    for variant in SSL_VARIANTS:
        assert f"eligible_{variant}" in validated.columns
    assert not validated.astype(str).stack().str.contains(r"[A-Za-z]:[\\/]|\\.set\\b|\\.fdt\\b|\\.xlsx\\b", regex=True).any()
```

- [ ] **Step 2: Write failing leakage tests**

Add tests that build one fold with test patient `STK-001` and assert `STK-001` baseline/immediate/mid/final EO/EC rows are excluded, healthy rows remain eligible for B/D, and unlabeled stroke rows do not need labels.

- [ ] **Step 3: Run RED tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_data.py tests/test_ssl_matrixnet_pooling_no_leakage.py -q
```

Expected: fail with `ModuleNotFoundError: No module named 'stroke_predict.ssl_matrixnet_data'`.

- [ ] **Step 4: Implement data-index and pool builder**

Implement:

```python
SSL_VARIANTS = ("stroke_baseline", "stroke_healthy_baseline", "stroke_all_stage", "stroke_all_stage_healthy")
STROKE_SOURCES = {"stroke_supervised", "stroke_ssl_only"}
HEALTHY_SOURCES = {"healthy"}
BASELINE_STAGES = {"baseline"}
ALL_STAGES = {"baseline", "immediate", "mid", "final"}
CONDITIONS = {"eo", "ec"}
```

`validate_ssl_matrix_index(frame)` enforces required columns, legal stages/conditions, de-identified fields, and variant eligibility booleans. `build_ssl_fold_pools(index, outer_folds, ssl_variant, fold_limit=None)` returns `(pool_frame, audit_frame)` with one row per included fold-record and audit columns `outer_fold`, `test_subject`, `ssl_variant`, `test_subject_records_in_pool`, `healthy_records_in_pool`, `unlabeled_stroke_records_in_pool`, and `leakage_passed`.

- [ ] **Step 5: Run GREEN tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_data.py tests/test_ssl_matrixnet_pooling_no_leakage.py -q
```

Expected: pass.

- [ ] **Step 6: Commit data task**

Run:

```bash
git add src/stroke_predict/ssl_matrixnet_data.py tests/test_ssl_matrixnet_data.py tests/test_ssl_matrixnet_pooling_no_leakage.py
git commit -m "feat: build leakage-safe SSL MatrixNet pools"
```

### Task 4: Masked Matrix Modeling Primitives

**Files:**
- Test: `tests/test_ssl_matrixnet_masking.py`
- Create: `src/stroke_predict/ssl_matrixnet.py`

- [ ] **Step 1: Write failing masking tests**

Add tests for deterministic mask shape, approximate ratio, masked-only MSE, and unmasked entries not contributing:

```python
from stroke_predict.ssl_matrixnet import generate_mask, masked_mse_loss

def test_masked_mse_uses_only_masked_entries():
    prediction = torch.tensor([[1.0, 100.0], [3.0, 4.0]])
    target = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    mask = torch.tensor([[True, False], [True, False]])
    loss = masked_mse_loss(prediction, target, mask)
    assert torch.isclose(loss, torch.tensor(((1.0 - 0.0) ** 2 + (3.0 - 1.0) ** 2) / 2))
```

- [ ] **Step 2: Run RED masking tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_masking.py -q
```

Expected: fail because `stroke_predict.ssl_matrixnet` does not exist.

- [ ] **Step 3: Implement minimal masking primitives**

Implement `generate_mask(shape, mask_ratio, seed=None, device=None)` and `masked_mse_loss(prediction, target, mask)` using torch tensors. Validate mask ratios are one of `0.15`, `0.25`, `0.40` unless tests pass an explicit valid value.

- [ ] **Step 4: Run GREEN masking tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_masking.py -q
```

Expected: pass.

- [ ] **Step 5: Commit masking task**

Run:

```bash
git add src/stroke_predict/ssl_matrixnet.py tests/test_ssl_matrixnet_masking.py
git commit -m "feat: add masked matrix modeling primitives"
```

### Task 5: SSL Encoder Pretraining

**Files:**
- Test: `tests/test_ssl_matrixnet_pretrain.py`
- Modify: `src/stroke_predict/ssl_matrixnet.py`
- Create: `src/stroke_predict/ssl_matrixnet_training.py`
- Create: `scripts/10_pretrain_ssl_matrixnet.py`
- Create: `configs/ssl_matrixnet.yaml`

- [ ] **Step 1: Write failing pretraining tests**

Create a tiny PSD/FC dataset and assert one epoch returns finite loss, writes a checkpoint, checkpoint contains `psd_encoder` and `fc_encoder`, and those weights can load into MatrixNet branches.

- [ ] **Step 2: Run RED pretraining tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_pretrain.py -q
```

Expected: fail on missing `pretrain_ssl_matrixnet`.

- [ ] **Step 3: Implement minimal SSL model and trainer**

Implement `SSLMatrixNetPretrainConfig`, `SSLMatrixDataset`, `SSLMatrixAutoencoder`, `pretrain_ssl_matrixnet`, `save_ssl_checkpoint`, and `load_ssl_encoder_checkpoint`. The autoencoder reuses `MatrixBranch` shape contracts and adds lightweight linear decoders over flattened reconstructed PSD/FC shapes for masked MSE.

- [ ] **Step 4: Add pretraining CLI**

`scripts/10_pretrain_ssl_matrixnet.py` parses `--config`, `--run-mode`, `--ssl-variant`, `--fold-limit`, `--device`, and `--require-cuda`, prints `SSL_MATRIXNET_PRETRAIN_OK`, writes `pretrain_log_phase7.csv`, `ssl_fold_pool_audit_phase7.csv`, `no_leakage_report_phase7.txt`, config snapshot, and fold-specific checkpoints under ignored `outputs/`.

- [ ] **Step 5: Run GREEN pretraining tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_pretrain.py -q
```

Expected: pass.

- [ ] **Step 6: Commit pretraining task**

Run:

```bash
git add src/stroke_predict/ssl_matrixnet.py src/stroke_predict/ssl_matrixnet_training.py scripts/10_pretrain_ssl_matrixnet.py configs/ssl_matrixnet.yaml tests/test_ssl_matrixnet_pretrain.py
git commit -m "feat: pretrain SSL MatrixNet encoders"
```

### Task 6: Load Pretrained Encoders Into Supervised MatrixNet

**Files:**
- Test: `tests/test_ssl_matrixnet_pretrain.py`
- Modify: `src/stroke_predict/ssl_matrixnet.py`
- Modify only if required: `src/stroke_predict/matrixnet.py`

- [ ] **Step 1: Write failing checkpoint-loading test**

Add an assertion that a saved SSL checkpoint changes a new MatrixNet branch state after calling `load_pretrained_matrixnet_branches(model, checkpoint_path, load_psd=True, load_fc=True)`.

- [ ] **Step 2: Run RED loading test**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_pretrain.py::test_checkpoint_can_load_into_matrixnet_branches -q
```

Expected: fail on missing loader function.

- [ ] **Step 3: Implement branch loader**

Implement a narrow loader that copies matching `features` and `projection` tensors into `model.psd_branch` and `model.fc_branch`, skips disabled branches, and raises on missing requested branch weights.

- [ ] **Step 4: Run GREEN loading test**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_pretrain.py -q
```

Expected: pass.

- [ ] **Step 5: Commit loader task**

Run:

```bash
git add src/stroke_predict/ssl_matrixnet.py src/stroke_predict/matrixnet.py tests/test_ssl_matrixnet_pretrain.py
git commit -m "feat: load SSL encoders into MatrixNet"
```

### Task 7: SSL Fine-Tuning

**Files:**
- Test: `tests/test_ssl_matrixnet_finetune.py`
- Modify: `src/stroke_predict/ssl_matrixnet_training.py`
- Create: `scripts/11_train_ssl_matrixnet.py`

- [ ] **Step 1: Write failing fine-tune tests**

Use `_write_minimal_inputs(tmp_path)` from existing MatrixNet tests, generate a tiny checkpoint, run one fold, and assert patient-level output has `sigmoid_score`, `predicted_score`, `score_orientation`, `ssl_checkpoint_path_redacted`, and labels Poor=0/Good=1.

- [ ] **Step 2: Run RED fine-tune tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_finetune.py -q
```

Expected: fail on missing `run_ssl_matrixnet_lopo`.

- [ ] **Step 3: Implement fine-tune orchestration**

Implement `SSLMatrixNetRunConfig`, model mapping for `M9a_sslA_fc_only`, `M9b_sslA_psd_fc`, `M9c_sslA_psd_fc_tacs`, `M13_sslA_clinical_eeg`, and fast variant aliases. Reuse Phase 6 normalization, inner validation orientation, threshold selection, and patient-level prediction columns while loading fold-specific SSL checkpoints before supervised training.

- [ ] **Step 4: Add fine-tune CLI**

`scripts/11_train_ssl_matrixnet.py` parses the same run flags, prints `SSL_MATRIXNET_TRAIN_OK`, refuses CUDA full mode when required CUDA is unavailable, and writes Phase 7 predictions, metrics, seed-wise metrics, patient-averaged metrics, report, and config snapshot.

- [ ] **Step 5: Run GREEN fine-tune tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_finetune.py -q
```

Expected: pass.

- [ ] **Step 6: Commit fine-tune task**

Run:

```bash
git add src/stroke_predict/ssl_matrixnet_training.py scripts/11_train_ssl_matrixnet.py tests/test_ssl_matrixnet_finetune.py
git commit -m "feat: fine-tune SSL MatrixNet models"
```

### Task 8: Phase 7 Metrics And Report

**Files:**
- Test: `tests/test_ssl_matrixnet_outputs.py`
- Modify: `src/stroke_predict/ssl_matrixnet_training.py`

- [ ] **Step 1: Write failing output and metric tests**

Assert required output files are created; required columns exist; duplicate `(model_name, patient_id, seed)` rows are absent; metrics include seed mean/std AUC, pooled AUC, patient-averaged AUC, CI, permutation p-value, PR-AUC, balanced accuracy, sensitivity, specificity, F1, Brier score, direction fields, and comparison columns to Phase 6.2 matched models when baseline CSVs exist.

- [ ] **Step 2: Run RED output tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_outputs.py -q
```

Expected: fail on missing output writer fields.

- [ ] **Step 3: Implement metrics and report writer**

Implement `compute_ssl_matrixnet_metrics`, `write_ssl_matrixnet_outputs`, patient averaging by model and patient across seeds, bootstrap/permutation only in full mode, matched baseline lookup for M8b/M8c/M8d, and report sections matching the Phase 7 report requirements.

- [ ] **Step 4: Run GREEN output tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_outputs.py -q
```

Expected: pass.

- [ ] **Step 5: Commit metrics task**

Run:

```bash
git add src/stroke_predict/ssl_matrixnet_training.py tests/test_ssl_matrixnet_outputs.py
git commit -m "feat: report Phase 7 SSL MatrixNet metrics"
```

### Task 9: CLI And Privacy Guards

**Files:**
- Test: `tests/test_ssl_matrixnet_cli.py`
- Test: `tests/test_ssl_matrixnet_outputs.py`
- Modify: `src/stroke_predict/ssl_matrixnet_data.py`
- Modify: `src/stroke_predict/ssl_matrixnet_training.py`
- Modify: `scripts/10_pretrain_ssl_matrixnet.py`
- Modify: `scripts/11_train_ssl_matrixnet.py`

- [ ] **Step 1: Write failing CLI tests**

Add subprocess tests for:

```bash
python scripts/10_pretrain_ssl_matrixnet.py --config <tmp>/ssl_matrixnet.yaml --run-mode fast --fold-limit 1 --ssl-variant stroke_baseline --device cpu
python scripts/11_train_ssl_matrixnet.py --config <tmp>/ssl_matrixnet.yaml --run-mode fast --fold-limit 1 --ssl-variant stroke_baseline --device cpu
```

Also assert full CUDA mode refuses to run when `--require-cuda` is set and CUDA is unavailable.

- [ ] **Step 2: Write failing privacy tests**

Scan Phase 7 public outputs and staged files for absolute local paths, raw EEG path strings, `.set`, `.fdt`, `.xlsx`, personal-name columns, and staged `outputs/`.

- [ ] **Step 3: Run RED CLI/privacy tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_cli.py tests/test_ssl_matrixnet_outputs.py -q
```

Expected: fail until CLI and privacy guard behavior exists.

- [ ] **Step 4: Implement CLI and privacy guards**

Implement `assert_no_private_strings(frame_or_text)`, output path redaction, checkpoint redaction, `git diff --cached --name-only` helper for tests, and CLI device validation. Full mode with `require_cuda=true` raises a clear `RuntimeError` if CUDA is unavailable.

- [ ] **Step 5: Run GREEN CLI/privacy tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_cli.py tests/test_ssl_matrixnet_outputs.py -q
```

Expected: pass.

- [ ] **Step 6: Commit CLI/privacy task**

Run:

```bash
git add src/stroke_predict/ssl_matrixnet_data.py src/stroke_predict/ssl_matrixnet_training.py scripts/10_pretrain_ssl_matrixnet.py scripts/11_train_ssl_matrixnet.py tests/test_ssl_matrixnet_cli.py tests/test_ssl_matrixnet_outputs.py
git commit -m "feat: add Phase 7 CLI and privacy guards"
```

### Task 10: Targeted Phase 7 Test Suite

**Files:**
- All Phase 7 source, tests, scripts, and config files.

- [ ] **Step 1: Run targeted SSL tests**

Run:

```bash
python -m pytest tests/test_ssl_matrixnet_* -q
```

Expected: all Phase 7 tests pass.

- [ ] **Step 2: Fix with TDD if any regression appears**

For each failure, write or tighten a failing test that captures the expected behavior, verify it fails, implement the smallest fix, and rerun the failing command.

- [ ] **Step 3: Commit test-suite stabilization if files changed**

Run:

```bash
git status --short
git add <changed Phase 7 code/docs/tests only>
git commit -m "test: stabilize Phase 7 SSL MatrixNet suite"
```

Only run the commit command if there are changed allowed files.

### Task 11: Full Repository Test Suite

**Files:**
- All tracked tests.

- [ ] **Step 1: Run full tests**

Run:

```bash
python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider
```

Expected: all repository tests pass.

- [ ] **Step 2: Fix failures through TDD**

If an existing behavior regresses, add a focused failing test or extend the relevant Phase 7 test, verify RED, implement minimal GREEN, rerun the full suite.

- [ ] **Step 3: Commit stabilization if needed**

Run a code/docs/tests-only commit if the fix changed tracked files.

### Task 12: Real-Data Fast Acceptance

**Files:**
- Ignored runtime outputs under `outputs/`.

- [ ] **Step 1: Ensure ignored input artifacts are available in worktree**

If the linked worktree lacks ignored `outputs/`, copy required ignored inputs from the main checkout into `.worktrees/phase7-ssl-matrixnet/outputs/` without staging them.

- [ ] **Step 2: Run CUDA fast SSL-A pretraining**

Run:

```bash
python scripts/10_pretrain_ssl_matrixnet.py --config configs/ssl_matrixnet.yaml --run-mode fast --ssl-variant stroke_baseline --fold-limit 2 --device cuda --require-cuda
```

Expected: exits 0, prints `SSL_MATRIXNET_PRETRAIN_OK`, reports CUDA device, writes audit and checkpoints under ignored `outputs/`.

- [ ] **Step 3: Run CUDA fast SSL-A fine-tuning**

Run:

```bash
python scripts/11_train_ssl_matrixnet.py --config configs/ssl_matrixnet.yaml --run-mode fast --ssl-variant stroke_baseline --fold-limit 2 --device cuda --require-cuda
```

Expected: exits 0, prints `SSL_MATRIXNET_TRAIN_OK`, writes patient-level predictions and report under ignored `outputs/`.

- [ ] **Step 4: Run CUDA fast B/C/D pretraining leakage checks**

Run:

```bash
python scripts/10_pretrain_ssl_matrixnet.py --config configs/ssl_matrixnet.yaml --run-mode fast --ssl-variant stroke_healthy_baseline --fold-limit 2 --device cuda --require-cuda
python scripts/10_pretrain_ssl_matrixnet.py --config configs/ssl_matrixnet.yaml --run-mode fast --ssl-variant stroke_all_stage --fold-limit 2 --device cuda --require-cuda
python scripts/10_pretrain_ssl_matrixnet.py --config configs/ssl_matrixnet.yaml --run-mode fast --ssl-variant stroke_all_stage_healthy --fold-limit 2 --device cuda --require-cuda
```

Expected: each exits 0 and audit shows `leakage_passed=True` for every fold.

- [ ] **Step 5: Inspect fast acceptance artifacts**

Run:

```bash
Get-Content outputs/reports/no_leakage_report_phase7.txt
Get-Content outputs/reports/phase7_ssl_matrixnet_report.md
git status --short
```

Expected: no leakage; no `outputs/` staged.

### Task 13: Real-Data Full SSL-A Acceptance

**Files:**
- Ignored runtime outputs under `outputs/`.

- [ ] **Step 1: Run full SSL-A pretraining**

Run:

```bash
python scripts/10_pretrain_ssl_matrixnet.py --config configs/ssl_matrixnet.yaml --run-mode full --ssl-variant stroke_baseline --device cuda --require-cuda
```

Expected: exits 0 with CUDA and 19 fold-specific SSL-A checkpoints.

- [ ] **Step 2: Run full SSL-A fine-tuning**

Run:

```bash
python scripts/11_train_ssl_matrixnet.py --config configs/ssl_matrixnet.yaml --run-mode full --ssl-variant stroke_baseline --device cuda --require-cuda
```

Expected: exits 0. Prediction row count is 380 if M13 has complete clinical features, otherwise 285 and the report states why M13 was skipped.

- [ ] **Step 3: Inspect full report and counts**

Run:

```bash
python -c "import pandas as pd; p='outputs/predictions/ssl_matrixnet_patient_predictions_phase7.csv'; df=pd.read_csv(p); print(len(df)); print(df.groupby(['model_name','seed']).patient_id.nunique())"
Get-Content outputs/reports/no_leakage_report_phase7.txt
Get-Content outputs/reports/phase7_ssl_matrixnet_report.md
```

Expected: patient-level counts, no leakage, report answers all Phase 7 decision questions.

### Task 14: Verification Before Completion

**Files:**
- Entire repository.

- [ ] **Step 1: Use verification-before-completion skill**

Apply the gate: identify commands, run them fresh, read exit codes, and only then report status.

- [ ] **Step 2: Run full tests**

Run:

```bash
python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 3: Run privacy and staged-artifact scan**

Run:

```bash
git diff --cached --name-only
git status --short
git ls-files --stage | Select-String -Pattern 'outputs/|\\.xlsx$|\\.set$|\\.fdt$|checkpoint|\\.pt$|\\.pth$'
$pathMarkers=@('[A-Za-z]:[\\/]','RestingStateEEG_'+'afterProcess','Patient_'+'tACS','Health_'+'tACS'); Get-ChildItem -Recurse -File src,scripts,configs,tests,docs | Select-String -Pattern $pathMarkers
```

Expected: no staged or tracked forbidden artifacts and no raw local path strings in tracked code/docs.

- [ ] **Step 4: Inspect reports**

Run:

```bash
Get-Content outputs/reports/no_leakage_report_phase7.txt
Get-Content outputs/reports/phase7_ssl_matrixnet_report.md
```

Expected: leakage report passes and report includes exploratory caution.

### Task 15: Commit Code/Docs/Tests Only

**Files:**
- Allowed tracked files only.

- [ ] **Step 1: Confirm allowed diff**

Run:

```bash
git status --short
git diff --name-only main...HEAD
```

Expected: only Phase 7 source, scripts, config, tests, spec, plan, and any small MatrixNet loader seam changes.

- [ ] **Step 2: Commit remaining allowed files**

Run:

```bash
git add src/stroke_predict/ssl_matrixnet.py src/stroke_predict/ssl_matrixnet_data.py src/stroke_predict/ssl_matrixnet_training.py scripts/10_pretrain_ssl_matrixnet.py scripts/11_train_ssl_matrixnet.py configs/ssl_matrixnet.yaml tests/test_ssl_matrixnet_*.py docs/superpowers/specs/2026-05-10-phase7-ssl-matrixnet-design.md docs/superpowers/plans/2026-05-10-phase7-ssl-matrixnet.md
git commit -m "feat: implement Phase 7 SSL MatrixNet"
```

Only include `src/stroke_predict/matrixnet.py` if it changed for checkpoint loading.

### Task 16: Merge Main And Push

**Files:**
- Git branches.

- [ ] **Step 1: Confirm all acceptance gates passed**

Required before merge: full tests pass, real-data fast acceptance passes, SSL-A full acceptance passes, no-leakage report passes, privacy scan passes, no forbidden artifacts staged.

- [ ] **Step 2: Merge to main**

Run:

```bash
git checkout main
git pull --ff-only
git merge codex/phase7-ssl-matrixnet
python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider
```

Expected: merge succeeds and full tests pass on `main`.

- [ ] **Step 3: Push main**

Run:

```bash
git push origin main
```

Expected: remote push succeeds.

- [ ] **Step 4: Final status**

Report branch, commit hash, Superpowers flow, spec/plan existence, tests, CUDA/GPU, variants implemented, variants trained in fast/full mode, prediction counts, best patient-averaged AUC model, permutation significance, no-leakage, privacy scan, excluded artifacts, merge status, push status, and next-phase recommendation.
