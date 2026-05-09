from __future__ import annotations

import pandas as pd

from stroke_predict.ml_models import ModelSpec, train_model_on_outer_fold


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"subject_id": "STK-001", "label_primary": "Good", "baseline_fma": 20.0, "x": 2.0},
            {"subject_id": "STK-002", "label_primary": "Poor", "baseline_fma": 30.0, "x": 1.0},
            {"subject_id": "STK-003", "label_primary": "Good", "baseline_fma": 21.0, "x": 3.0},
            {"subject_id": "STK-004", "label_primary": "Poor", "baseline_fma": 31.0, "x": 0.5},
            {"subject_id": "STK-005", "label_primary": "Good", "baseline_fma": 22.0, "x": 2.5},
            {"subject_id": "STK-006", "label_primary": "Poor", "baseline_fma": 32.0, "x": 0.2},
        ]
    )


def _registry() -> dict[str, object]:
    return {
        "outer_fold": 1,
        "test_subject": "STK-001",
        "supervised_train_subjects": ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"],
        "normalization_fit_subjects": ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"],
        "feature_selection_fit_subjects": ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"],
        "threshold_selection_subjects": ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"],
        "inner_splits": [
            {
                "inner_fold": 1,
                "train_subjects": ["STK-002", "STK-003", "STK-004"],
                "val_subjects": ["STK-005", "STK-006"],
            },
            {
                "inner_fold": 2,
                "train_subjects": ["STK-003", "STK-005", "STK-006"],
                "val_subjects": ["STK-002", "STK-004"],
            },
        ],
    }


def _spec(c_values: list[float] | None = None) -> ModelSpec:
    return ModelSpec(
        model_id="M1_fma_only",
        feature_columns=["baseline_fma"],
        feature_groups={"baseline_fma": "clinical"},
        estimator="ridge_logistic",
        c_values=c_values or [0.1, 1.0],
        l1_ratios=[0.0],
    )


def test_high_dimensional_spec_selects_features_inside_training_pipeline() -> None:
    features = _features()
    for index in range(20):
        features[f"x_{index:02d}"] = features["x"] + (index * 0.01)
    spec = ModelSpec(
        model_id="M6b_psd_fc_matrix_flatten_ml",
        feature_columns=[f"x_{index:02d}" for index in range(20)],
        feature_groups={f"x_{index:02d}": "psd_matrix_flatten" for index in range(20)},
        estimator="ridge_logistic",
        c_values=[0.1],
        l1_ratios=[0.0],
        max_importance_features=20,
        max_selected_features=3,
    )

    result = train_model_on_outer_fold(spec, features, _registry(), random_seed=23)

    assert len(result.importance) <= 3
    assert {row["feature_name"] for row in result.importance} <= set(spec.feature_columns)


def test_outer_fold_training_excludes_test_subject_from_fit_and_threshold() -> None:
    result = train_model_on_outer_fold(_spec(), _features(), _registry(), random_seed=11)

    assert result.prediction["subject_id"] == "STK-001"
    assert result.prediction["n_train_subjects"] == 5
    assert result.fit_subjects == ["STK-002", "STK-003", "STK-004", "STK-005", "STK-006"]
    assert "STK-001" not in result.threshold_subjects
    assert set(result.threshold_subjects) <= set(result.fit_subjects)
    assert 0.0 <= result.prediction["prob_good"] <= 1.0
    assert result.importance
    assert result.importance[0]["feature_name"] == "baseline_fma"


def test_outer_test_subject_in_feature_table_does_not_change_train_threshold() -> None:
    baseline = train_model_on_outer_fold(_spec([1.0]), _features(), _registry(), random_seed=17)
    poisoned = _features()
    poisoned.loc[poisoned["subject_id"].eq("STK-001"), "baseline_fma"] = 999999.0
    changed = train_model_on_outer_fold(_spec([1.0]), poisoned, _registry(), random_seed=17)

    assert baseline.threshold == changed.threshold
    assert baseline.fit_subjects == changed.fit_subjects
    assert baseline.threshold_subjects == changed.threshold_subjects
