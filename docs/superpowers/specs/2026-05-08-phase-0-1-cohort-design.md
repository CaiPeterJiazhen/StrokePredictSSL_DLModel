# Phase 0+1 Cohort And Labeling Design

## Goal

Build the first reproducible slice of the tACS stroke recovery prediction project: repository hygiene, configuration, environment validation, cohort assembly, de-identified subject IDs, and FMA-UE responder labels.

This phase intentionally does not train models, extract EEG matrices, or run full EEG signal QC. It creates a reliable patient-level cohort and label audit that later phases can trust.

## Inputs

Primary structured input:

`F:\CJZProjectFile\StrokePredictSSL-DLModel\current_data_status_overview_data_only.xlsx`

Relevant workbook sheets:

- `01_患者数据总览`: patient-level clinical and data availability overview.
- `03_临床量表原始`: raw clinical scale values and completeness flags.
- `06_预处理静息态阶段汇总`: preprocessed resting-state EEG summary.
- `07_预处理静息态文件明细`: preprocessed resting-state EEG file index.
- `02_统计汇总`: sanity-check counts from the source audit.

External EEG roots for later phases:

- Stroke EEG: `F:\CJZFile\EEG_M1\Patient_tACS_M1_RestingStateEEG_afterProcess`
- Healthy EEG: `F:\CJZFile\EEG_M1\Health_tACS_M1_RestingStateEEG_afterProcess`

These roots are configuration values only in Phase 0+1. Raw `.set` and `.fdt` files are not copied into the repository.

## Privacy Boundary

The workbook and EEG folders contain direct identifiers such as names and name-like folder labels. Phase 0+1 treats those fields as private input only.

Committed code and generated public outputs must not contain:

- `姓名`
- `姓名写法`
- `EEG文件夹`
- `subject_name`
- raw file paths that include a person's name
- medical record numbers, if present in future inputs

The only stable public identifier is a deterministic de-identified `subject_id`. A private mapping file may be generated only if needed and must be ignored by Git.

## Repository Structure

Phase 0+1 will create:

- `configs/paths.yaml`: local input and output paths.
- `configs/project.yaml`: cohort and label settings.
- `src/stroke_predict/config.py`: config loading and path resolution.
- `src/stroke_predict/io/excel_status.py`: workbook sheet reading with explicit schemas.
- `src/stroke_predict/cohort/labels.py`: FMA label rules.
- `src/stroke_predict/cohort/ids.py`: deterministic de-identified ID creation.
- `src/stroke_predict/cohort/build.py`: cohort assembly and audit table generation.
- `scripts/00_validate_environment.py`: environment and path validation.
- `scripts/01_build_cohort.py`: command-line cohort builder.
- `tests/`: TDD tests for label rules, PII filtering, config loading, output schemas, and cohort role assignment.

Generated runtime outputs:

- `outputs/cohort/cohort_master.csv`
- `outputs/cohort/label_audit.csv`
- `outputs/cohort/label_distribution.json`
- `outputs/cohort/cohort_summary.json`
- `outputs/figures/fig_label_distribution.png`

`outputs/` is ignored by Git.

## Label Rules

The primary label follows the PRD ceiling-adjusted clinically meaningful FMA-UE response rule:

- Missing baseline or post FMA: `missing`
- Baseline FMA equals 66: `ceiling_exclude`
- Baseline FMA <= 61: `Good` if delta FMA >= 5, else `Poor`
- Baseline FMA from 62 to 65: `Good` if delta FMA >= `min(3, 66 - baseline_fma)`, else `Poor`

Sensitivity fields:

- `label_delta5_all`
- `label_prop70`
- `label_low_baseline_only`
- `outcome_delta_fma`
- `outcome_post_fma`

The audit table records the numeric ingredients behind every label:

- `subject_id`
- `baseline_fma`
- `post_fma`
- `delta_fma`
- `possible_recovery`
- `recovery_ratio`
- `label_primary`
- `label_delta5_all`
- `label_prop70`
- `label_low_baseline_only`
- `label_reason`

## Role Assignment

Phase 0+1 assigns roles from currently available metadata:

- `supervised_main`: stroke patient with complete FMA pre/post, baseline eyes-open EEG, baseline eyes-closed EEG, non-missing Good/Poor primary label, and not ceiling excluded.
- `ceiling_exclude`: stroke patient with baseline FMA equal to 66.
- `ssl_only_stroke`: stroke participant with at least one valid preprocessed resting-state EEG record but not eligible for `supervised_main`.
- `healthy_ssl`: healthy participant with at least one valid preprocessed resting-state EEG record.
- `excluded_no_eeg`: clinical patient without usable baseline EO/EC metadata in the preprocessed index.
- `excluded_bad_qc`: reserved for Phase 2 signal-level QC; Phase 0+1 does not mark records bad on signal quality.

If multiple roles apply, the exported role field uses the most analysis-relevant role in this priority order:

`supervised_main`, `ceiling_exclude`, `ssl_only_stroke`, `healthy_ssl`, `excluded_no_eeg`, `excluded_bad_qc`.

## Data Flow

1. Validate that Python can import required packages and that the workbook exists.
2. Read the relevant workbook sheets with stable column names.
3. Normalize obvious stage and condition labels already present in the workbook.
4. Create deterministic de-identified IDs separately for stroke and healthy sources.
5. Apply FMA primary and sensitivity label rules.
6. Join clinical rows to preprocessed baseline EEG metadata by source subject key.
7. Assign cohort roles.
8. Write de-identified cohort, label audit, distribution JSON, and summary JSON.
9. Generate a label distribution figure from de-identified data only.

## Error Handling

The scripts fail fast with clear messages when:

- the workbook path does not exist;
- a required sheet is missing;
- a required column is missing;
- FMA fields cannot be parsed as numeric values where a completeness flag says they are complete;
- a public output would contain a blocked PII column;
- generated `subject_id` values are duplicated within a source.

Warnings are acceptable when:

- optional clinical fields such as MBI, BBT, or MMSE are missing;
- a patient has clinical data but no matching preprocessed EEG metadata;
- healthy subjects lack clinical fields, as expected.

## Testing Strategy

All production code in this phase is implemented test-first.

Minimum tests:

- Label rule tests for missing values, baseline 66, baseline <= 61 with delta 4/5, baseline 64/65 with the ceiling-adjusted threshold, and proportional recovery labels.
- ID tests confirming deterministic, unique, source-prefixed de-identified IDs.
- PII tests confirming public output schemas exclude blocked columns.
- Config tests confirming paths can be loaded and resolved.
- Cohort builder tests using small synthetic data frames for `supervised_main`, `ceiling_exclude`, `ssl_only_stroke`, `healthy_ssl`, and `excluded_no_eeg`.
- Script smoke tests for `00_validate_environment.py` and `01_build_cohort.py` on synthetic fixtures.

Acceptance commands:

```bash
python scripts/00_validate_environment.py
python scripts/01_build_cohort.py --config configs/project.yaml
pytest tests -q
```

## Acceptance Criteria

- `current_data_status_overview_data_only.xlsx` is not tracked by Git.
- Raw EEG files are not tracked by Git.
- `cohort_master.csv` and `label_audit.csv` contain only de-identified subject IDs.
- `label_primary` contains only `Good`, `Poor`, `ceiling_exclude`, or `missing`.
- Baseline FMA 64/65 cases can be labeled Good when they meet the adjusted threshold.
- Baseline FMA 66 cases do not enter the main binary supervised cohort.
- The supervised cohort count is derived from data and reported in `cohort_summary.json`.
- Tests pass before moving to Phase 2.

## Out Of Scope

- Reading raw EEG signal data.
- Signal-level EEG QC.
- PSD, FC, tACS-informed feature extraction.
- LOPO fold generation.
- Classical ML, MatrixNet, SSL, interpretability, and manuscript generation.

Those items begin in later phase-specific specs and plans.
