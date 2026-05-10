from __future__ import annotations

from pathlib import Path

import numpy as np

from stroke_predict.matrixnet_data import load_matrixnet_inputs
from stroke_predict.matrixnet_training import MatrixNetRunConfig, run_matrixnet_lopo

from tests.test_matrixnet_no_leakage import _write_minimal_inputs


def test_matrixnet_fast_smoke_produces_finite_predictions(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    inputs = load_matrixnet_inputs(tmp_path)
    config = MatrixNetRunConfig(
        run_mode="fast",
        models=["M8a_matrixnet_psd_only"],
        seeds=[0],
        max_epochs=3,
        patience=2,
        batch_size=2,
        learning_rates=[1e-3],
        weight_decays=[1e-2],
        dropouts=[0.3],
        embedding_dims=[8],
        hidden_dims=[16],
        fold_limit=2,
        write_outputs=False,
    )
    result = run_matrixnet_lopo(inputs, config)
    predictions = result.predictions
    assert len(predictions) == 2
    assert predictions["model_name"].eq("M8a_matrixnet_psd_only").all()
    assert {
        "label_int",
        "logit",
        "sigmoid_score",
        "score_orientation",
    } <= set(predictions.columns)
    assert set(predictions["label_int"].astype(int)) <= {0, 1}
    assert predictions["sigmoid_score"].between(0, 1).all()
    assert predictions["predicted_score"].between(0, 1).all()
    assert predictions["score_orientation"].isin(
        ["normal", "inverted_by_inner_val", "normal_insufficient_inner_classes"]
    ).all()
    assert np.isfinite(predictions["train_loss_final"]).all()
