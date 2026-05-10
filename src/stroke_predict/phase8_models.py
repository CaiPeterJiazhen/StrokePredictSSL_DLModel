from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from stroke_predict.phase8_evaluation import (
    compute_phase8_metrics,
    validate_phase8_no_leakage,
    validate_phase8_patient_predictions,
)


M14_MODELS = {
    "M14a_prop_reduced32_fullfc_ridge_logistic": ("reduced32_full_edge", "ridge_logistic"),
    "M14b_prop_reduced32_fullfc_elasticnet": ("reduced32_full_edge", "elasticnet_logistic"),
    "M14c_prop_reduced32_fullfc_linear_svm": ("reduced32_full_edge", "linear_svm"),
    "M14d_prop_reduced32_fullfc_pls_da": ("reduced32_full_edge", "pls_da"),
}
M15_MODELS = {
    "M15a_prop_roi_fc_best_ml": ("roi_fc", "ridge_logistic"),
    "M15b_prop_summary_eeg_best_ml": ("summary_eeg", "ridge_logistic"),
}
M16_MODELS = {
    "M16a_prop_full62_fullfc_ridge_logistic": ("full62_full_edge", "ridge_logistic"),
    "M16b_prop_full62_fullfc_pls_da": ("full62_full_edge", "pls_da"),
}
PHASE8_MODEL_REGISTRY = {**M14_MODELS, **M15_MODELS, **M16_MODELS}


@dataclass(frozen=True)
class Phase8ModelSpec:
    model_id: str
    feature_set: str
    estimator: str
    c_values: tuple[float, ...] = (0.1, 1.0)
    l1_ratios: tuple[float, ...] = (0.0, 0.5)
    pls_components: tuple[int, ...] = (1, 2)

    @classmethod
    def for_model_id(cls, model_id: str) -> "Phase8ModelSpec":
        if model_id not in PHASE8_MODEL_REGISTRY:
            raise ValueError(f"Unknown Phase 8 model ID: {model_id}")
        feature_set, estimator = PHASE8_MODEL_REGISTRY[model_id]
        return cls(model_id=model_id, feature_set=feature_set, estimator=estimator)


@dataclass(frozen=True)
class Phase8RunResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    no_leakage_audit: pd.DataFrame


def run_phase8_lopo_models(
    config: dict[str, Any],
    *,
    features: pd.DataFrame,
    labels: pd.DataFrame,
    folds: dict[str, Any],
    run_mode: str,
    fold_limit: int | None = None,
    feature_set: str = "reduced32",
) -> Phase8RunResult:
    requested = [str(model_id) for model_id in config.get("models", M14_MODELS.keys())]
    if run_mode == "full" and feature_set == "full62" and any(model_id in M16_MODELS for model_id in requested):
        if not bool(config.get("m16_full62_full_mode_enabled", False)):
            raise ValueError("M16 full62 full-mode is not planned for Phase 8 unless explicitly enabled")

    specs = [Phase8ModelSpec.for_model_id(model_id) for model_id in requested]
    table = _merge_features_and_labels(features, labels)
    fold_rows = list(folds.get("folds", []))
    if fold_limit is not None:
        fold_rows = fold_rows[: int(fold_limit)]
    predictions: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    seed = int(config.get("random_seed", 42))

    for spec in specs:
        feature_columns = _feature_columns_for_spec(spec, table)
        for fold in fold_rows:
            outer_fold = int(fold["outer_fold"])
            test_subject = str(fold["test_subject"])
            train_subjects = [str(subject) for subject in fold.get("supervised_train_subjects", [])]
            if not train_subjects:
                all_subjects = table["subject_id"].astype(str).tolist()
                train_subjects = [subject for subject in all_subjects if subject != test_subject]
            prediction, audit = _fit_predict_fold(
                spec,
                table,
                feature_columns,
                outer_fold=outer_fold,
                train_subjects=train_subjects,
                test_subject=test_subject,
                random_seed=seed,
            )
            predictions.append(prediction)
            audit_rows.append(audit)

    prediction_frame = pd.DataFrame(predictions)
    expected_count = len(fold_rows) if fold_limit is not None else int(folds.get("n_supervised_main", len(fold_rows)))
    validate_phase8_patient_predictions(prediction_frame, expected_patient_count=expected_count)
    audit_frame = pd.DataFrame(audit_rows)
    validate_phase8_no_leakage(audit_frame)
    return Phase8RunResult(
        predictions=prediction_frame,
        metrics=compute_phase8_metrics(prediction_frame),
        no_leakage_audit=audit_frame,
    )


def _fit_predict_fold(
    spec: Phase8ModelSpec,
    table: pd.DataFrame,
    feature_columns: list[str],
    *,
    outer_fold: int,
    train_subjects: list[str],
    test_subject: str,
    random_seed: int,
) -> tuple[dict[str, object], dict[str, object]]:
    train = _rows_for_subjects(table, train_subjects)
    test = _rows_for_subjects(table, [test_subject])
    if len(test) != 1:
        raise ValueError(f"Expected one row for outer test subject {test_subject}, found {len(test)}")
    y_train = train["primary_label_int_prop_residual"].astype(int).to_numpy()
    if len(set(y_train.tolist())) < 2:
        raise ValueError(f"Outer train labels must contain both classes for {spec.model_id} fold {outer_fold}")

    estimator = _fit_estimator(spec, train[feature_columns], y_train, random_seed=random_seed)
    score = _score_estimator(estimator, spec, test[feature_columns])
    y_true = int(test.iloc[0]["primary_label_int_prop_residual"])
    predicted_label = "ProportionalRecovery" if score >= 0.5 else "PoorRecovery"
    prediction = {
        "model_id": spec.model_id,
        "outer_fold": outer_fold,
        "patient_id": test_subject,
        "true_label": str(test.iloc[0]["primary_label_prop_residual"]),
        "y_true": y_true,
        "predicted_score": float(score),
        "predicted_label": predicted_label,
        "threshold": 0.5,
        "threshold_source": "fixed_0.5",
        "prediction_unit": "patient",
        "feature_set": spec.feature_set,
        "estimator": spec.estimator,
        "n_train_subjects": int(len(train_subjects)),
    }
    train_set = set(train_subjects)
    audit = {
        "model_id": spec.model_id,
        "outer_fold": outer_fold,
        "test_subject": test_subject,
        "fit_subjects": ";".join(train_subjects),
        "transform_fit_subjects": ";".join(train_subjects),
        "inner_cv_subjects": ";".join(train_subjects),
        "outer_test_in_fit_subjects": test_subject in train_set,
        "outer_test_in_transform_fit_subjects": test_subject in train_set,
        "outer_test_in_inner_cv_subjects": test_subject in train_set,
    }
    return prediction, audit


def _fit_estimator(
    spec: Phase8ModelSpec,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    random_seed: int,
) -> Pipeline:
    if spec.estimator == "elasticnet_logistic":
        classifier = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=1.0,
            l1_ratio=0.5,
            max_iter=20000,
            tol=1e-3,
            random_state=random_seed,
        )
    elif spec.estimator == "linear_svm":
        classifier = LinearSVC(C=1.0, random_state=random_seed, max_iter=10000)
    elif spec.estimator == "pls_da":
        classifier = _PLSDAClassifier(n_components=1, random_seed=random_seed)
    else:
        classifier = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=1.0,
            max_iter=5000,
            random_state=random_seed,
        )
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("variance", VarianceThreshold()),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )
    pipeline.fit(x_train, y_train)
    return pipeline


def _score_estimator(estimator: Pipeline, spec: Phase8ModelSpec, x_test: pd.DataFrame) -> float:
    classifier = estimator.named_steps["classifier"]
    if hasattr(classifier, "predict_proba"):
        score = float(estimator.predict_proba(x_test)[:, 1][0])
    elif hasattr(classifier, "decision_function"):
        decision = float(estimator.decision_function(x_test)[0])
        score = 1.0 / (1.0 + np.exp(-decision))
    else:
        score = float(estimator.predict(x_test)[0])
    return float(np.clip(score, 0.0, 1.0))


def _merge_features_and_labels(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if "subject_id" not in features.columns:
        raise ValueError("Phase 8 features must include subject_id")
    required_labels = {"subject_id", "primary_label_prop_residual", "primary_label_int_prop_residual"}
    missing = sorted(required_labels - set(labels.columns))
    if missing:
        raise ValueError(f"Phase 8 labels missing required columns: {missing}")
    analyzable = labels.loc[labels.get("phase8_label_status", "analyzable").eq("analyzable")].copy()
    merged = features.copy().assign(subject_id=lambda frame: frame["subject_id"].astype(str)).merge(
        analyzable[["subject_id", "primary_label_prop_residual", "primary_label_int_prop_residual"]].assign(
            subject_id=lambda frame: frame["subject_id"].astype(str)
        ),
        on="subject_id",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No analyzable Phase 8 feature-label rows")
    return merged


def _feature_columns_for_spec(spec: Phase8ModelSpec, table: pd.DataFrame) -> list[str]:
    if spec.feature_set in {"reduced32_full_edge", "full62_full_edge"}:
        prefixes = ("fullfc_", "reduced32_", "full62_")
    elif spec.feature_set == "roi_fc":
        prefixes = ("roi_fc", "fc_roi")
    elif spec.feature_set == "summary_eeg":
        prefixes = ("summary", "psd_", "fc_", "eeg_")
    else:
        prefixes = ()
    columns = [
        column
        for column in table.columns
        if column != "subject_id"
        and column not in {"primary_label_prop_residual", "primary_label_int_prop_residual"}
        and pd.api.types.is_numeric_dtype(table[column])
        and table[column].notna().any()
        and (not prefixes or column.startswith(prefixes))
    ]
    if not columns:
        columns = [
            column
            for column in table.columns
            if column != "subject_id"
            and column not in {"primary_label_prop_residual", "primary_label_int_prop_residual"}
            and pd.api.types.is_numeric_dtype(table[column])
            and table[column].notna().any()
        ]
    if not columns:
        raise ValueError(f"No numeric features available for {spec.model_id}")
    return columns


def _rows_for_subjects(table: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    subject_set = {str(subject) for subject in subjects}
    rows = table.loc[table["subject_id"].astype(str).isin(subject_set)].copy()
    missing = sorted(subject_set - set(rows["subject_id"].astype(str)))
    if missing:
        raise ValueError(f"Missing Phase 8 feature rows for subjects: {missing}")
    return rows.sort_values("subject_id").reset_index(drop=True)


class _PLSDAClassifier:
    def __init__(self, n_components: int, random_seed: int) -> None:
        self.n_components = int(n_components)
        self.random_seed = int(random_seed)
        self.pls = PLSRegression(n_components=self.n_components)
        self.head = LogisticRegression(solver="liblinear", random_state=self.random_seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "_PLSDAClassifier":
        self.pls.fit(x, y)
        scores = self.pls.transform(x)
        self.head.fit(scores, y)
        self.classes_ = self.head.classes_
        self.is_fitted_ = True
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        scores = self.pls.transform(x)
        return self.head.predict_proba(scores)

    def __sklearn_is_fitted__(self) -> bool:
        return bool(getattr(self, "is_fitted_", False))
