from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_phase8_label_cli_runs_on_toy_data(tmp_path: Path) -> None:
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
    assert (tmp_path / "outputs" / "evaluation" / "phase8_label_audit.csv").exists()


def test_phase8_full_edge_fc_cli_runs_on_toy_matrix_input(tmp_path: Path) -> None:
    config = _write_toy_phase8_config(tmp_path)

    label_result = subprocess.run(
        [sys.executable, "scripts/12_build_phase8_labels.py", "--config", str(config), "--run-mode", "fast"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert label_result.returncode == 0, label_result.stderr

    result = subprocess.run(
        [
            sys.executable,
            "scripts/13_extract_full_edge_fc.py",
            "--config",
            str(config),
            "--run-mode",
            "fast",
            "--feature-set",
            "reduced32",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PHASE8_FULL_EDGE_FC_OK" in result.stdout
    assert (tmp_path / "outputs" / "features" / "phase8_reduced32_full_edge_index.csv").exists()
    assert (tmp_path / "outputs" / "matrices" / "phase8_fc_full_reduced32_eo.npy").exists()


def test_phase8_model_cli_runs_fast_fold_limit(tmp_path: Path) -> None:
    config = _write_toy_phase8_config(tmp_path)
    _run_cli([sys.executable, "scripts/12_build_phase8_labels.py", "--config", str(config), "--run-mode", "fast"])
    _run_cli(
        [
            sys.executable,
            "scripts/13_extract_full_edge_fc.py",
            "--config",
            str(config),
            "--run-mode",
            "fast",
            "--feature-set",
            "reduced32",
        ]
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/14_train_phase8_full_edge_models.py",
            "--config",
            str(config),
            "--run-mode",
            "fast",
            "--fold-limit",
            "2",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "PHASE8_MODELS_OK" in result.stdout
    predictions = pd.read_csv(tmp_path / "outputs" / "predictions" / "phase8_prop_full_edge_patient_predictions.csv")
    assert predictions["outer_fold"].nunique() == 2


def test_phase8_model_cli_refuses_unplanned_full62_full_mode(tmp_path: Path) -> None:
    config = _write_toy_phase8_config(tmp_path, models=["M16a_prop_full62_fullfc_ridge_logistic"])

    result = subprocess.run(
        [
            sys.executable,
            "scripts/14_train_phase8_full_edge_models.py",
            "--config",
            str(config),
            "--run-mode",
            "full",
            "--feature-set",
            "full62",
        ],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "M16 full62 full-mode" in result.stderr


def _run_cli(command: list[str]) -> None:
    result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def _write_toy_phase8_config(
    tmp_path: Path,
    *,
    models: list[str] | None = None,
) -> Path:
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    outputs.mkdir()
    subjects = [f"STK-{index:03d}" for index in range(1, 7)]
    cohort = pd.DataFrame(
        {
            "subject_id": subjects,
            "baseline_fma": [40, 40, 45, 45, 50, 50],
            "post_fma": [60, 45, 62, 48, 60, 52],
            "label_primary": ["Good", "Poor", "Good", "Poor", "Good", "Poor"],
        }
    )
    cohort.to_csv(inputs / "cohort.csv", index=False)
    _write_toy_eeg(inputs, subjects)
    folds = {
        "n_supervised_main": len(subjects),
        "folds": [
            {
                "outer_fold": index,
                "test_subject": subject,
                "supervised_train_subjects": [other for other in subjects if other != subject],
            }
            for index, subject in enumerate(subjects, start=1)
        ],
    }
    (inputs / "folds.json").write_text(json.dumps(folds), encoding="utf-8")
    summary = pd.DataFrame({"subject_id": subjects, "summary_signal": [0, 1, 0, 1, 0, 1]})
    roi = pd.DataFrame({"subject_id": subjects, "roi_fc_signal": [0, 1, 0, 1, 0, 1]})
    summary.to_csv(inputs / "summary_features.csv", index=False)
    roi.to_csv(inputs / "roi_features.csv", index=False)
    config = {
        "input_paths": {
            "cohort": str(inputs / "cohort.csv"),
            "toy_eeg_npz": str(inputs / "toy_eeg.npz"),
            "toy_eeg_index": str(inputs / "toy_eeg_index.csv"),
            "folds": str(inputs / "folds.json"),
            "summary_features": str(inputs / "summary_features.csv"),
            "roi_features": str(inputs / "roi_features.csv"),
        },
        "output_dir": str(outputs),
        "random_seed": 17,
        "models": models
        or [
            "M14a_prop_reduced32_fullfc_ridge_logistic",
            "M15a_prop_roi_fc_best_ml",
            "M15b_prop_summary_eeg_best_ml",
        ],
        "fast": {"bootstrap_resamples": 5, "permutation_resamples": 5},
        "full": {"bootstrap_resamples": 10, "permutation_resamples": 10},
        "m16_full62_full_mode_enabled": False,
    }
    config_path = tmp_path / "phase8.yaml"
    _write_simple_yaml(config_path, config)
    return config_path


def _write_toy_eeg(inputs: Path, subjects: list[str]) -> None:
    sfreq = 128
    t = np.arange(0, 4, 1 / sfreq)
    channels = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "POz",
        "O1",
        "Oz",
        "O2",
        "AF3",
        "AF4",
        "PO3",
    ]
    arrays = {}
    rows = []
    for subject_index, subject in enumerate(subjects):
        for condition in ("eo", "ec"):
            base_frequency = 10 + subject_index
            data = np.vstack(
                [
                    np.sin(2 * np.pi * (base_frequency + channel_index % 3) * t + channel_index * 0.1)
                    for channel_index, _channel in enumerate(channels)
                ]
            )
            key = f"{subject}_{condition}"
            arrays[key] = data
            rows.append({"subject_id": subject, "condition": condition, "array_key": key, "sfreq": sfreq})
    np.savez(inputs / "toy_eeg.npz", **arrays)
    pd.DataFrame(rows).to_csv(inputs / "toy_eeg_index.csv", index=False)
    pd.DataFrame({"channel": channels}).to_csv(inputs / "toy_channels.csv", index=False)


def _write_simple_yaml(path: Path, data: dict[str, object]) -> None:
    import yaml

    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
