# Phase 5.1 Feature Repair Design

## Goal

Repair the pre-MatrixNet feature layer so Phase 5 ML baselines use complete, non-empty summary EEG features and keep flattened matrix baselines separate.

## Scope

- Keep MatrixNet and SSL out of scope.
- Reuse existing PSD matrices, ROI-FC matrices, lesion-normalized views, and LOPO registries.
- Generate tACS connectivity features from ROI-FC outputs instead of passing an empty connectivity dictionary.
- Add summary tables for PSD, FC, tACS target, EO/EC reactivity, and all summary EEG features.
- Split ML baselines into summary-feature and flattened-matrix model IDs.

## Design

Phase 3 feature repair remains in `scripts/06_build_handcrafted_features.py`, but the script will stop acting as a minimal handcrafted writer. It will build reusable summary tables from the existing `psd_eo/ec.npy` and `fc_roi_eo/ec.npy` matrices, write the requested CSV outputs, and rebuild the summary portion of `feature_dictionary.csv` without duplicating rows on reruns.

The tACS feature builder will receive:

- channel-level band power from PSD matrices;
- ROI-level band power derived from the configured tACS target/homologous channel sets;
- ROI-FC connectivity values derived from the existing ROI-FC matrices.

For target/homologous connectivity, the native view maps the target hand to left/right motor ROIs. The lesion-normalized view always maps the target side to the left motor ROI after channel flipping. Target-to-frontal and target-to-parietal use the ipsilateral frontal/parietal ROI in each view; target-to-midline uses `midline_motor`.

The ML layer will treat summary and flattened features as separate input families:

- `M3a_psd_summary_ml`
- `M4a_fc_summary_ml`
- `M5_tacs_target_summary_ml`
- `M6_all_summary_eeg_ml`
- `M3b_psd_matrix_flatten_ml`
- `M4b_fc_matrix_flatten_ml`
- `M6b_psd_fc_matrix_flatten_ml`

`M12_clinical_plus_eeg_ml` stays available and uses clinical features plus `features_all_summary.csv`, not flattened matrices.

## Acceptance Checks

- tACS coherence/wPLI columns are not all NaN.
- Required ROI, asymmetry, log-ratio, and EO/EC reactivity columns exist.
- Summary CSVs and feature dictionary are written.
- LOPO/no-leakage tests still pass.
- Phase 5 ML evaluation runs on the repaired feature tables.
