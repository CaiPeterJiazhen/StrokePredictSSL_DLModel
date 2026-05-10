from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from stroke_predict.matrixnet_data import load_matrixnet_inputs
from stroke_predict.matrixnet_training import (
    MatrixNetRunConfig,
    MatrixNetRunResult,
    compute_matrixnet_metrics,
    run_matrixnet_lopo,
    write_matrixnet_outputs,
)

from tests.test_matrixnet_no_leakage import _write_minimal_inputs


def test_write_matrixnet_outputs_creates_required_files_and_columns(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    inputs = load_matrixnet_inputs(tmp_path)
    config = MatrixNetRunConfig(
        run_mode="fast",
        models=["M8a_matrixnet_psd_only"],
        seeds=[0],
        max_epochs=2,
        patience=1,
        batch_size=2,
        learning_rates=[1e-3],
        weight_decays=[1e-2],
        dropouts=[0.3],
        embedding_dims=[8],
        hidden_dims=[16],
        write_outputs=True,
    )
    result = run_matrixnet_lopo(inputs, config)
    write_matrixnet_outputs(tmp_path, result, config)

    predictions_path = tmp_path / "predictions" / "matrixnet_patient_predictions.csv"
    metrics_path = tmp_path / "evaluation" / "matrixnet_metrics.csv"
    report_path = tmp_path / "reports" / "phase6_matrixnet_report.md"
    audit_path = tmp_path / "reports" / "matrixnet_fold_audit.csv"
    leakage_path = tmp_path / "reports" / "matrixnet_no_leakage_report.txt"
    assert predictions_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert audit_path.exists()
    assert leakage_path.exists()

    predictions = pd.read_csv(predictions_path)
    assert {
        "model_name",
        "outer_fold",
        "patient_id",
        "true_label",
        "predicted_score",
        "predicted_label",
        "threshold",
        "threshold_source",
        "seed",
        "run_mode",
        "input_family",
        "best_epoch",
        "best_inner_metric",
        "train_loss_final",
        "val_loss_best",
    } <= set(predictions.columns)
    assert not predictions.duplicated(["model_name", "patient_id", "seed"]).any()
    assert predictions.groupby(["model_name", "seed"])["patient_id"].nunique().eq(3).all()

    metrics = pd.read_csv(metrics_path)
    assert {
        "model_name",
        "input_family",
        "run_mode",
        "n_patients",
        "n_good",
        "n_poor",
        "n_seeds",
        "roc_auc_mean",
        "roc_auc_std_across_seeds",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "brier_score",
        "permutation_p_value",
        "comparison_to_best_ml_auc",
        "comparison_to_fma_only_auc",
        "comparison_to_clinical_only_auc",
    } <= set(metrics.columns)
    report_text = report_path.read_text(encoding="utf-8")
    assert "supervised no-SSL" in report_text
    assert "smoke-only" in report_text
    assert "Phase 5.2" in report_text
    assert "Do not claim EEG efficacy" in report_text
    assert "M12" in report_text and "secondary" in report_text


def test_matrixnet_script_fast_mode_with_fold_limit(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    config_path = tmp_path / "matrixnet.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {tmp_path.as_posix()}",
                "run_modes:",
                "  fast:",
                "    seeds: [0]",
                "    max_epochs: 2",
                "    patience: 1",
                "    batch_size: 2",
                "    learning_rates: [0.001]",
                "    weight_decays: [0.01]",
                "    dropouts: [0.3]",
                "    embedding_dims: [8]",
                "    hidden_dims: [16]",
                "models:",
                "  fast:",
                "    - M8a_matrixnet_psd_only",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/09_train_matrixnet.py",
            "--config",
            str(config_path),
            "--run-mode",
            "fast",
            "--fold-limit",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "MATRIXNET_OK" in completed.stdout


def test_fast_mode_refuses_to_overwrite_full_mode_outputs(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "phase6_matrixnet_report.md").write_text("Run mode: **full**\n", encoding="utf-8")
    inputs = load_matrixnet_inputs(tmp_path)
    config = MatrixNetRunConfig(
        run_mode="fast",
        models=["M8a_matrixnet_psd_only"],
        seeds=[0],
        max_epochs=1,
        patience=1,
        batch_size=2,
        learning_rates=[1e-3],
        weight_decays=[1e-2],
        dropouts=[0.3],
        embedding_dims=[8],
        hidden_dims=[16],
        fold_limit=1,
        write_outputs=True,
    )
    result = run_matrixnet_lopo(inputs, config)
    with pytest.raises(FileExistsError, match="Refusing to overwrite full-mode"):
        write_matrixnet_outputs(tmp_path, result, config)


def test_full_mode_outputs_are_separate_and_include_inference_answers(tmp_path: Path) -> None:
    predictions = _full_mode_prediction_frame()
    ml_metrics = pd.DataFrame(
        {
            "model_name": ["M3b_psd_matrix_flatten_ml", "M4b_fc_matrix_flatten_ml", "M6b_psd_fc_matrix_flatten_ml", "M1_fma_only", "M2_clinical_only"],
            "roc_auc": [0.60, 0.55, 0.58, 0.52, 0.50],
        }
    )
    config = MatrixNetRunConfig(
        run_mode="full",
        models=sorted(predictions["model_name"].unique().tolist()),
        seeds=[0, 1, 2, 3, 4],
        max_epochs=2,
        patience=1,
        batch_size=2,
        learning_rates=[1e-3],
        weight_decays=[1e-2],
        dropouts=[0.3],
        embedding_dims=[8],
        hidden_dims=[16],
        bootstrap_resamples=25,
        permutation_resamples=25,
        random_seed=123,
    )
    metrics = compute_matrixnet_metrics(predictions, ml_metrics, config=config)
    result = MatrixNetRunResult(
        predictions=predictions,
        metrics=metrics,
        training_log=pd.DataFrame({"model_name": ["M8b_matrixnet_fc_only"], "seed": [0], "outer_fold": [1], "epoch": [1]}),
        fold_audit=pd.DataFrame(
            {
                "model_name": ["M8b_matrixnet_fc_only"],
                "seed": [0],
                "outer_fold": [1],
                "test_excluded_from_train": [True],
                "test_excluded_from_val": [True],
            }
        ),
    )
    paths = write_matrixnet_outputs(tmp_path, result, config)

    assert Path(paths["predictions"]).name == "matrixnet_patient_predictions_full.csv"
    assert Path(paths["metrics"]).name == "matrixnet_metrics_full.csv"
    assert Path(paths["report"]).name == "phase6_matrixnet_full_report.md"
    assert (tmp_path / "predictions" / "matrixnet_patient_predictions.csv").exists() is False

    metrics_written = pd.read_csv(paths["metrics"])
    assert metrics_written["roc_auc_ci_low"].notna().all()
    assert metrics_written["roc_auc_ci_high"].notna().all()
    assert metrics_written["permutation_p_value"].notna().all()
    assert "comparison_to_best_flattened_ml_auc" in metrics_written.columns

    report_text = Path(paths["report"]).read_text(encoding="utf-8")
    for question in [
        "Is M8b_matrixnet_fc_only stable across seeds?",
        "Does MatrixNet outperform flattened-matrix ML controls?",
        "Does PSD+FC improve over FC-only?",
        "Does adding tACS summary improve or hurt?",
        "Does clinical+EEG improve over EEG-only?",
        "Are any results permutation-significant?",
    ]:
        assert question in report_text


def _full_mode_prediction_frame() -> pd.DataFrame:
    patients = ["S01", "S02", "S03", "S04"]
    labels = {"S01": "Poor", "S02": "Poor", "S03": "Good", "S04": "Good"}
    base_scores = {
        "M8a_matrixnet_psd_only": [0.30, 0.45, 0.55, 0.60],
        "M8b_matrixnet_fc_only": [0.15, 0.25, 0.75, 0.85],
        "M8c_matrixnet_psd_fc": [0.20, 0.30, 0.70, 0.80],
        "M8d_matrixnet_psd_fc_tacs": [0.25, 0.35, 0.65, 0.75],
        "M12_matrixnet_clinical_eeg": [0.10, 0.20, 0.80, 0.90],
    }
    rows: list[dict[str, object]] = []
    for model_name, scores in base_scores.items():
        for seed in [0, 1, 2, 3, 4]:
            for fold, (patient, score) in enumerate(zip(patients, scores, strict=True), start=1):
                jittered = min(max(score + seed * 0.005, 0.001), 0.999)
                rows.append(
                    {
                        "model_name": model_name,
                        "outer_fold": fold,
                        "patient_id": patient,
                        "true_label": labels[patient],
                        "predicted_score": jittered,
                        "predicted_label": "Good" if jittered >= 0.5 else "Poor",
                        "threshold": 0.5,
                        "threshold_source": "inner_validation_balanced_accuracy",
                        "seed": seed,
                        "run_mode": "full",
                        "input_family": "synthetic",
                        "best_epoch": 1,
                        "best_inner_metric": -0.5,
                        "train_loss_final": 0.5,
                        "val_loss_best": 0.5,
                    }
                )
    return pd.DataFrame(rows)
