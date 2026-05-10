from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from stroke_predict.matrixnet_data import LABEL_TO_INT
from stroke_predict.matrixnet_training import (
    INT_TO_LABEL,
    _apply_score_orientation,
    _calibrate_score_orientation,
    _phase6_2_metric_frames,
)


def test_label_encoding_contract_is_good_positive_class() -> None:
    assert LABEL_TO_INT == {"Poor": 0, "Good": 1}
    assert INT_TO_LABEL == {0: "Poor", 1: "Good"}

    labels = torch.tensor([0.0, 1.0])
    logits = torch.tensor([-2.0, 2.0])
    loss = torch.nn.BCEWithLogitsLoss(reduction="none")(logits, labels)
    swapped_loss = torch.nn.BCEWithLogitsLoss(reduction="none")(logits, 1.0 - labels)

    assert float(loss.mean()) < float(swapped_loss.mean())
    assert torch.sigmoid(logits).tolist()[1] > torch.sigmoid(logits).tolist()[0]


def test_inner_validation_auc_below_half_inverts_without_outer_scores() -> None:
    inner_y = np.asarray([0, 0, 1, 1], dtype=int)
    inner_scores = np.asarray([0.9, 0.8, 0.2, 0.1], dtype=float)

    orientation = _calibrate_score_orientation(inner_y, inner_scores)
    assert orientation == "inverted_by_inner_val"
    oriented = _apply_score_orientation(inner_scores, orientation)

    assert math.isclose(float(roc_auc_score(inner_y, oriented)), 1.0)


def test_inner_validation_single_class_does_not_invert() -> None:
    inner_y = np.asarray([1, 1], dtype=int)
    inner_scores = np.asarray([0.2, 0.1], dtype=float)

    orientation = _calibrate_score_orientation(inner_y, inner_scores)

    assert orientation == "normal_insufficient_inner_classes"
    assert np.allclose(_apply_score_orientation(inner_scores, orientation), inner_scores)


def test_phase6_2_metric_frames_compare_seed_pooled_and_patient_averaged_auc() -> None:
    predictions = pd.DataFrame(
        [
            _prediction("S01", "Poor", 0, 0.10),
            _prediction("S02", "Poor", 0, 0.20),
            _prediction("S03", "Good", 0, 0.80),
            _prediction("S04", "Good", 0, 0.90),
            _prediction("S01", "Poor", 1, 0.40),
            _prediction("S02", "Poor", 1, 0.30),
            _prediction("S03", "Good", 1, 0.60),
            _prediction("S04", "Good", 1, 0.70),
        ]
    )

    metrics, seed_metrics, patient_metrics = _phase6_2_metric_frames(predictions)
    row = metrics.iloc[0]

    assert row["roc_auc_mean"] == 1.0
    assert row["pooled_auc"] == 1.0
    assert row["patient_averaged_auc"] == 1.0
    assert row["auc_score"] == 1.0
    assert row["auc_one_minus_score"] == 0.0
    assert row["mean_score_good"] > row["mean_score_poor"]
    assert bool(row["direction_correct"]) is True
    assert len(seed_metrics) == 2
    assert len(patient_metrics) == 4


def _prediction(patient_id: str, label: str, seed: int, score: float) -> dict[str, object]:
    label_int = 1 if label == "Good" else 0
    return {
        "model_name": "M8b_matrixnet_fc_only",
        "seed": seed,
        "outer_fold": int(patient_id[-1]),
        "patient_id": patient_id,
        "true_label": label,
        "label_int": label_int,
        "predicted_score": score,
        "predicted_label": "Good" if score >= 0.5 else "Poor",
        "threshold": 0.5,
        "score_orientation": "normal",
        "run_mode": "full",
    }
