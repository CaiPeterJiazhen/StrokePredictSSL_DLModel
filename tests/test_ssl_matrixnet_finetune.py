from __future__ import annotations

from pathlib import Path

import numpy as np

from stroke_predict.matrixnet_data import load_matrixnet_inputs
from stroke_predict.ssl_matrixnet_training import (
    SSLMatrixNetRunConfig,
    SSLPretrainConfig,
    pretrain_ssl_matrixnet,
    run_ssl_matrixnet_lopo,
)

from tests.test_matrixnet_no_leakage import _write_minimal_inputs


def test_finetuning_runs_one_fold_with_tiny_pretrained_checkpoint(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    checkpoint_path = tmp_path / "ssl_matrixnet" / "checkpoints" / "unit" / "stroke_baseline" / "fold_01" / "ssl_encoder.pt"
    checkpoint_path.parent.mkdir(parents=True)
    rng = np.random.default_rng(42)
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

    result = run_ssl_matrixnet_lopo(inputs, config)

    predictions = result.predictions
    assert len(predictions) == 1
    assert {
        "sigmoid_score",
        "predicted_score",
        "score_orientation",
        "ssl_checkpoint_path_redacted",
    } <= set(predictions.columns)
    assert predictions["label_int"].isin([0, 1]).all()
    label_map = dict(zip(predictions["true_label"], predictions["label_int"], strict=False))
    assert label_map.get("Poor", 0) == 0
