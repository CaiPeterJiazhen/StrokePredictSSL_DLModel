from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stroke_predict.matrixnet_data import load_matrixnet_inputs
from stroke_predict.ssl_matrixnet_data import assert_no_private_strings
from stroke_predict.ssl_matrixnet_training import (
    SSLMatrixNetRunConfig,
    SSLPretrainConfig,
    pretrain_ssl_matrixnet,
    run_ssl_matrixnet_lopo,
    write_ssl_matrixnet_outputs,
)

from tests.test_matrixnet_no_leakage import _write_minimal_inputs


def test_write_phase7_outputs_creates_required_files_and_columns(tmp_path: Path) -> None:
    result, config = _tiny_ssl_result(tmp_path)

    paths = write_ssl_matrixnet_outputs(tmp_path, result, config)

    expected_files = {
        "ssl_matrix_index",
        "ssl_fold_pool_audit",
        "no_leakage_report",
        "pretrain_log",
        "predictions",
        "metrics",
        "seed_wise_metrics",
        "patient_averaged_metrics",
        "report",
        "config_used",
    }
    assert expected_files <= set(paths)
    for key in expected_files:
        assert Path(paths[key]).exists(), key

    predictions = pd.read_csv(paths["predictions"])
    required_columns = {
        "model_name",
        "ssl_variant",
        "seed",
        "outer_fold",
        "patient_id",
        "true_label",
        "label_int",
        "logit",
        "sigmoid_score",
        "predicted_score",
        "predicted_label",
        "threshold",
        "threshold_source",
        "score_orientation",
        "run_mode",
        "ssl_checkpoint_path_redacted",
        "device",
        "best_epoch",
        "train_loss_final",
        "val_loss_best",
    }
    assert required_columns <= set(predictions.columns)
    assert not predictions.duplicated(["model_name", "patient_id", "seed"]).any()


def test_phase7_outputs_do_not_contain_private_paths_or_forbidden_artifacts(tmp_path: Path) -> None:
    result, config = _tiny_ssl_result(tmp_path)
    paths = write_ssl_matrixnet_outputs(tmp_path, result, config)

    for path in paths.values():
        path_obj = Path(path)
        if path_obj.is_file():
            assert_no_private_strings(path_obj.read_text(encoding="utf-8", errors="ignore"))


def test_no_duplicate_model_patient_seed_predictions(tmp_path: Path) -> None:
    result, config = _tiny_ssl_result(tmp_path)
    paths = write_ssl_matrixnet_outputs(tmp_path, result, config)
    predictions = pd.read_csv(paths["predictions"])

    assert not predictions.duplicated(["model_name", "patient_id", "seed"]).any()


def test_phase7_report_records_configured_mask_ratio(tmp_path: Path) -> None:
    result, config = _tiny_ssl_result(tmp_path)
    paths = write_ssl_matrixnet_outputs(tmp_path, result, config)

    report_text = Path(paths["report"]).read_text(encoding="utf-8")
    assert "mask_ratio=0.25" in report_text
    assert "mask_ratio=fast" not in report_text


def _tiny_ssl_result(tmp_path: Path):
    _write_minimal_inputs(tmp_path)
    checkpoint_path = tmp_path / "ssl_matrixnet" / "checkpoints" / "unit" / "stroke_baseline" / "fold_01" / "ssl_encoder.pt"
    checkpoint_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(13)
    pretrain_ssl_matrixnet(
        psd=rng.normal(size=(4, 2, 4, 5)).astype(np.float32),
        fc=rng.normal(size=(4, 4, 3, 2)).astype(np.float32),
        checkpoint_path=checkpoint_path,
        config=SSLPretrainConfig(
            ssl_variant="stroke_baseline",
            run_mode="fast",
            epochs=1,
            batch_size=2,
            mask_ratio=0.25,
            embedding_dim=8,
            hidden_dim=16,
            seed=0,
            device="cpu",
        ),
    )
    inputs = load_matrixnet_inputs(tmp_path)
    config = SSLMatrixNetRunConfig(
        run_mode="fast",
        ssl_variant="stroke_baseline",
        models=["M9a_sslA_fc_only"],
        seeds=[0],
        max_epochs=2,
        patience=1,
        batch_size=2,
        learning_rates=[1e-3],
        weight_decays=[1e-2],
        dropouts=[0.3],
        embedding_dims=[8],
        hidden_dims=[16],
        fold_limit=1,
        checkpoint_path_override=checkpoint_path,
        device="cpu",
    )
    return run_ssl_matrixnet_lopo(inputs, config), config
