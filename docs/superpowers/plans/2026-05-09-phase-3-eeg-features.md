# Phase 3 EEG Feature Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 3 PSD, ROI-FC, handcrafted, and tACS-informed EEG feature outputs with native and lesion-normalized views.

**Architecture:** Add a focused `stroke_predict.features` package for configuration, channel mapping, matrix extraction, tACS feature construction, output privacy checks, and script orchestration. Scripts remain thin; tests use synthetic EEG arrays first, then real-data acceptance verifies output shapes and privacy.

**Tech Stack:** Python 3.12, NumPy, pandas, SciPy signal processing, PyYAML, pytest.

---

## File Structure

- Create `configs/features.yaml`: Phase 3 feature parameters, ROI definitions, bands, and lesion-normalization map.
- Create `src/stroke_predict/features/__init__.py`: package marker.
- Create `src/stroke_predict/features/config.py`: load feature config and resolve linked project/eeg configs.
- Create `src/stroke_predict/features/channels.py`: channel-pair map, flip indices, PSD and FC edge remapping.
- Create `src/stroke_predict/features/psd.py`: windowing and Welch PSD matrix extraction.
- Create `src/stroke_predict/features/fc.py`: ROI edge definitions, coherence and wPLI computation.
- Create `src/stroke_predict/features/tacs.py`: target mapping and native / lesion-normalized target features.
- Create `src/stroke_predict/features/outputs.py`: matrix/CSV writers and privacy validation.
- Create `scripts/04_extract_psd_matrices.py`: write PSD EO/EC matrices and dictionary rows.
- Create `scripts/05_extract_fc_matrices.py`: write ROI-FC EO/EC matrices and dictionary rows.
- Create `scripts/06_build_handcrafted_features.py`: write handcrafted and tACS feature CSVs plus consolidated dictionary.
- Create `tests/test_feature_config.py`: config defaults.
- Create `tests/test_lesion_normalization.py`: C3/C4, FC3/FC4, CP3/CP4, midline, PSD axis and FC edge mapping.
- Create `tests/test_psd_features.py`: PSD shape, frequency grid, log transform.
- Create `tests/test_fc_features.py`: ROI edge order, coherence/wPLI shape.
- Create `tests/test_tacs_features.py`: left/right target mapping and native/normalized feature names.
- Create `tests/test_feature_outputs.py`: dictionary schema and privacy validation.
- Create `tests/test_feature_scripts.py`: script existence and smoke-level import checks.

## Task 1: Feature Config

**Files:**
- Create: `configs/features.yaml`
- Create: `src/stroke_predict/features/__init__.py`
- Create: `src/stroke_predict/features/config.py`
- Test: `tests/test_feature_config.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from stroke_predict.features.config import load_feature_config


def test_load_feature_config_defaults(tmp_path: Path) -> None:
    project = tmp_path / "project.yaml"
    eeg = tmp_path / "eeg.yaml"
    features = tmp_path / "features.yaml"
    project.write_text("paths_config: paths.yaml\n", encoding="utf-8")
    eeg.write_text("project_config: project.yaml\n", encoding="utf-8")
    features.write_text("project_config: project.yaml\neeg_config: eeg.yaml\n", encoding="utf-8")

    config = load_feature_config(features)

    assert config.path == features.resolve()
    assert config.project_config_path == project.resolve()
    assert config.eeg_config_path == eeg.resolve()
    assert config.freq_min_hz == 0.5
    assert config.freq_max_hz == 45.0
    assert config.freq_resolution_hz == 0.5
    assert config.views == ["native", "lesion_normalized"]
    assert "alpha_mu" in config.bands
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feature_config.py -q`

Expected: FAIL because `stroke_predict.features` does not exist.

- [ ] **Step 3: Implement minimal config loader**

Create a dataclass `FeatureConfig` with `path`, `project_config_path`, `eeg_config_path`, `raw`, and properties for PSD, bands, ROI, connectivity methods, views, and channel pair map. Defaults must match the spec.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_feature_config.py -q`

Expected: PASS.

## Task 2: Lesion-normalized Channel Mapping

**Files:**
- Create: `src/stroke_predict/features/channels.py`
- Test: `tests/test_lesion_normalization.py`

- [ ] **Step 1: Write failing mapping tests**

```python
import numpy as np

from stroke_predict.features.channels import (
    DEFAULT_CHANNEL_PAIR_MAP,
    build_flip_indices,
    flip_fc_edges,
    flip_psd_matrix,
)


CHANNELS = ["FC3", "FC4", "C3", "C4", "CP3", "CP4", "Cz"]


def test_pair_map_contains_required_motor_pairs() -> None:
    assert DEFAULT_CHANNEL_PAIR_MAP["C3"] == "C4"
    assert DEFAULT_CHANNEL_PAIR_MAP["FC3"] == "FC4"
    assert DEFAULT_CHANNEL_PAIR_MAP["CP3"] == "CP4"


def test_flip_indices_swap_motor_channels_and_keep_midline() -> None:
    indices = build_flip_indices(CHANNELS, DEFAULT_CHANNEL_PAIR_MAP)
    flipped = [CHANNELS[i] for i in indices]

    assert flipped == ["FC4", "FC3", "C4", "C3", "CP4", "CP3", "Cz"]


def test_flip_psd_matrix_swaps_channel_axis() -> None:
    psd = np.arange(len(CHANNELS) * 2).reshape(len(CHANNELS), 2)
    flipped = flip_psd_matrix(psd, CHANNELS, DEFAULT_CHANNEL_PAIR_MAP)

    assert np.array_equal(flipped[CHANNELS.index("C3")], psd[CHANNELS.index("C4")])
    assert np.array_equal(flipped[CHANNELS.index("FC4")], psd[CHANNELS.index("FC3")])
    assert np.array_equal(flipped[CHANNELS.index("Cz")], psd[CHANNELS.index("Cz")])


def test_flip_fc_edges_maps_edge_endpoints() -> None:
    edges = [("C3", "FC3"), ("C4", "FC4"), ("C3", "Cz")]
    values = np.array([[1.0], [2.0], [3.0]])

    flipped_edges, flipped_values = flip_fc_edges(edges, values, DEFAULT_CHANNEL_PAIR_MAP)

    assert ("C4", "FC4") in flipped_edges
    assert ("C3", "FC3") in flipped_edges
    assert ("C4", "Cz") in flipped_edges
    assert float(flipped_values[flipped_edges.index(("C4", "FC4")), 0]) == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lesion_normalization.py -q`

Expected: FAIL with missing module/functions.

- [ ] **Step 3: Implement channel mapping**

Implement `DEFAULT_CHANNEL_PAIR_MAP`, `build_flip_indices`, `flip_psd_matrix`, and `flip_fc_edges`. Edge labels must be canonicalized with sorted endpoint tuples after mapping so undirected edges remain stable.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lesion_normalization.py -q`

Expected: PASS.

## Task 3: PSD Feature Matrices

**Files:**
- Create: `src/stroke_predict/features/psd.py`
- Test: `tests/test_psd_features.py`

- [ ] **Step 1: Write failing PSD tests**

```python
import numpy as np

from stroke_predict.features.psd import compute_psd_matrix, make_frequency_grid


def test_frequency_grid_has_expected_bins() -> None:
    freqs = make_frequency_grid(0.5, 45.0, 0.5)

    assert len(freqs) == 90
    assert freqs[0] == 0.5
    assert freqs[-1] == 45.0


def test_compute_psd_matrix_returns_channel_by_frequency() -> None:
    sfreq = 250
    t = np.arange(0, 8, 1 / sfreq)
    signal = np.vstack([np.sin(2 * np.pi * 10 * t), np.sin(2 * np.pi * 20 * t)])

    psd, freqs = compute_psd_matrix(signal, sfreq=sfreq)

    assert psd.shape == (2, 90)
    assert freqs.shape == (90,)
    assert np.isfinite(psd).all()
    assert freqs[np.argmax(psd[0])] == 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_psd_features.py -q`

Expected: FAIL with missing module/functions.

- [ ] **Step 3: Implement PSD helpers**

Implement frequency grid creation, 4s overlapping windowing, Welch PSD, interpolation to the configured grid, window average, and log10 transform with epsilon.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_psd_features.py -q`

Expected: PASS.

## Task 4: ROI Functional Connectivity

**Files:**
- Create: `src/stroke_predict/features/fc.py`
- Test: `tests/test_fc_features.py`

- [ ] **Step 1: Write failing FC tests**

```python
import numpy as np

from stroke_predict.features.fc import build_roi_edges, compute_roi_fc_matrix


def test_build_roi_edges_is_stable() -> None:
    rois = {"left_motor": ["C3", "FC3"], "right_motor": ["C4", "FC4"], "midline": ["Cz"]}

    edges = build_roi_edges(rois)

    assert edges == [("left_motor", "left_motor"), ("left_motor", "midline"), ("left_motor", "right_motor"), ("midline", "midline"), ("midline", "right_motor"), ("right_motor", "right_motor")]


def test_compute_roi_fc_matrix_shape() -> None:
    sfreq = 250
    t = np.arange(0, 8, 1 / sfreq)
    data = np.vstack([
        np.sin(2 * np.pi * 10 * t),
        np.sin(2 * np.pi * 10 * t),
        np.sin(2 * np.pi * 20 * t),
    ])
    channels = ["C3", "C4", "Cz"]
    rois = {"left": ["C3"], "right": ["C4"], "midline": ["Cz"]}
    bands = {"alpha_mu": (8, 13), "low_beta": (13, 20)}

    matrix, edges, methods = compute_roi_fc_matrix(data, channels, sfreq=sfreq, rois=rois, bands=bands, methods=("coherence", "wpli"))

    assert matrix.shape == (6, 2, 2)
    assert len(edges) == 6
    assert methods == ["coherence", "wpli"]
    assert np.isfinite(matrix[:, :, 0]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_fc_features.py -q`

Expected: FAIL with missing module/functions.

- [ ] **Step 3: Implement ROI FC**

Implement stable ROI edge generation, ROI mean time series, band coherence, and band wPLI. Keep matrix positions stable and fill NaN for invalid ROI pairs.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_fc_features.py -q`

Expected: PASS.

## Task 5: tACS-informed Features

**Files:**
- Create: `src/stroke_predict/features/tacs.py`
- Test: `tests/test_tacs_features.py`

- [ ] **Step 1: Write failing tACS tests**

```python
import pandas as pd

from stroke_predict.features.tacs import build_tacs_features, map_tacs_target


def test_map_tacs_target_uses_c3_for_right_and_c4_for_left() -> None:
    assert map_tacs_target("right")["target_channel"] == "C3"
    assert map_tacs_target("right")["homologous_channel"] == "C4"
    assert map_tacs_target("left")["target_channel"] == "C4"
    assert map_tacs_target("left")["homologous_channel"] == "C3"


def test_build_tacs_features_includes_native_and_normalized_names() -> None:
    cohort = pd.DataFrame({"subject_id": ["STK-001"], "treated_hand": ["left"], "affected_hand": ["left"]})
    band_power = {
        ("STK-001", "eyes_open", "native", "C4", "alpha_mu"): 4.0,
        ("STK-001", "eyes_open", "native", "C3", "alpha_mu"): 2.0,
        ("STK-001", "eyes_open", "lesion_normalized", "C3", "alpha_mu"): 4.0,
        ("STK-001", "eyes_open", "lesion_normalized", "C4", "alpha_mu"): 2.0,
    }

    features = build_tacs_features(cohort, band_power=band_power, connectivity={})

    assert features.loc[0, "native_eyes_open_target_channel"] == "C4"
    assert features.loc[0, "lesion_normalized_eyes_open_target_channel"] == "C3"
    assert features.loc[0, "native_eyes_open_target_alpha_mu_power"] == 4.0
    assert features.loc[0, "lesion_normalized_eyes_open_target_alpha_mu_power"] == 4.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tacs_features.py -q`

Expected: FAIL with missing module/functions.

- [ ] **Step 3: Implement tACS mapping and feature builder**

Implement target mapping, native/lesion-normalized target channel fields, target/homologous band power, ratios, differences, and EO/EC reactivity when both conditions exist.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tacs_features.py -q`

Expected: PASS.

## Task 6: Outputs and Scripts

**Files:**
- Create: `src/stroke_predict/features/outputs.py`
- Create: `scripts/04_extract_psd_matrices.py`
- Create: `scripts/05_extract_fc_matrices.py`
- Create: `scripts/06_build_handcrafted_features.py`
- Test: `tests/test_feature_outputs.py`
- Test: `tests/test_feature_scripts.py`

- [ ] **Step 1: Write failing output/script tests**

```python
from pathlib import Path

import pandas as pd

from stroke_predict.features.outputs import assert_public_feature_output, validate_feature_dictionary


def test_feature_output_rejects_path_like_values() -> None:
    frame = pd.DataFrame({"subject_id": ["STK-001"], "bad": ["private_raw_record.set"]})

    try:
        assert_public_feature_output(frame)
    except ValueError as exc:
        assert "path-like" in str(exc)
    else:
        raise AssertionError("Expected privacy rejection")


def test_feature_dictionary_requires_core_columns() -> None:
    dictionary = pd.DataFrame({"feature_name": ["x"]})

    try:
        validate_feature_dictionary(dictionary)
    except ValueError as exc:
        assert "feature_group" in str(exc)
    else:
        raise AssertionError("Expected schema rejection")


def test_phase3_scripts_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "04_extract_psd_matrices.py").exists()
    assert (root / "scripts" / "05_extract_fc_matrices.py").exists()
    assert (root / "scripts" / "06_build_handcrafted_features.py").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_feature_outputs.py tests/test_feature_scripts.py -q`

Expected: FAIL with missing modules/scripts.

- [ ] **Step 3: Implement output validation and scripts**

Implement privacy checks shared with Phase 2 behavior, feature dictionary validation, `.npy` saving, CSV writing, and scripts that orchestrate real-data extraction. Scripts must print `PSD_FEATURES_OK`, `FC_FEATURES_OK`, and `HANDCRAFTED_FEATURES_OK`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_feature_outputs.py tests/test_feature_scripts.py -q`

Expected: PASS.

## Task 7: Integration and Real Data Acceptance

**Files:**
- Modify tests as needed for integration gaps.

- [ ] **Step 1: Run targeted Phase 3 tests**

Run:

```bash
python -m pytest tests/test_feature_config.py tests/test_lesion_normalization.py tests/test_psd_features.py tests/test_fc_features.py tests/test_tacs_features.py tests/test_feature_outputs.py tests/test_feature_scripts.py -q
```

Expected: PASS.

- [ ] **Step 2: Run all tests**

Run: `python -m pytest tests -q`

Expected: PASS.

- [ ] **Step 3: Run real data prerequisites**

Run:

```bash
python scripts/00_validate_environment.py
python scripts/01_build_cohort.py --config configs/project.yaml
python scripts/02_index_eeg.py --config configs/eeg.yaml
python scripts/03_run_eeg_qc.py --config configs/eeg.yaml
```

Expected: `ENVIRONMENT_OK`, `COHORT_BUILD_OK`, `EEG_INDEX_OK`, `EEG_QC_OK`.

- [ ] **Step 4: Run Phase 3 real data scripts**

Run:

```bash
python scripts/04_extract_psd_matrices.py --config configs/features.yaml
python scripts/05_extract_fc_matrices.py --config configs/features.yaml
python scripts/06_build_handcrafted_features.py --config configs/features.yaml
```

Expected: `PSD_FEATURES_OK`, `FC_FEATURES_OK`, `HANDCRAFTED_FEATURES_OK`.

- [ ] **Step 5: Verify real outputs**

Run a read-only check that asserts:

```python
assert psd_eo.shape[0] == 19
assert psd_ec.shape[0] == 19
assert psd_eo.shape[1] == 2
assert psd_ec.shape[1] == 2
assert fc_roi_eo.shape[0] == 19
assert fc_roi_ec.shape[0] == 19
assert fc_roi_eo.shape[1] == 2
assert len(handcrafted) == 19
assert len(tacs) == 19
assert {"native", "lesion_normalized"}.issubset(set(dictionary["hemisphere_space"].dropna()))
assert no_public_output_contains_pii_or_paths
```

- [ ] **Step 6: Final git integration**

Run:

```bash
git status --short
git add configs/features.yaml src/stroke_predict/features scripts/04_extract_psd_matrices.py scripts/05_extract_fc_matrices.py scripts/06_build_handcrafted_features.py tests docs/superpowers
git commit -m "feat: extract phase 3 eeg features"
git switch main
git merge --no-ff codex/phase3-eeg-features
git push origin main
```

Expected: merge and push succeed without staging `outputs/`, Excel, `.set`, `.fdt`, names, or paths.

## Self-Review Checklist

- Spec coverage: PSD, ROI-FC, tACS-informed features, EO/EC reactivity, dictionary, outputs, privacy, and lesion-normalized view are covered.
- Placeholder scan: no unresolved placeholders.
- Type consistency: config, channel mapping, PSD, FC, tACS, and scripts use stable names.
- Scope guard: no LOPO folds, ML baseline, MatrixNet, SSL, or model training.
