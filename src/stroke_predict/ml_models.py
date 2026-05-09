from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from stroke_predict.evaluation import bootstrap_metric_ci, compute_classification_metrics, permutation_test, validate_patient_predictions
from stroke_predict.features.outputs import assert_public_feature_output


LABEL_TO_INT = {"Poor": 0, "Good": 1}
INT_TO_LABEL = {0: "Poor", 1: "Good"}
REQUIRED_MODEL_IDS = [
    "M0_majority",
    "M1_fma_only",
    "M2_clinical_only",
    "M3_psd_ml",
    "M4_fc_ml",
    "M5_tacs_target_ml",
    "M6_all_handcrafted_eeg_ml",
    "M12_clinical_plus_eeg_ml",
]
CLINICAL_NUMERIC_COLUMNS = ["age", "baseline_fma", "baseline_mbi", "mmse"]
CLINICAL_CATEGORICAL_COLUMNS = ["sex", "affected_hand", "treated_hand"]
IDENTIFIER_COLUMNS = {"subject_id", "label_primary", "treated_hand", "affected_hand"}


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    feature_columns: list[str]
    feature_groups: dict[str, str]
    estimator: str
    c_values: list[float]
    l1_ratios: list[float]
    categorical_columns: tuple[str, ...] = ()
    max_importance_features: int = 50


@dataclass(frozen=True)
class FoldTrainingResult:
    prediction: dict[str, object]
    importance: list[dict[str, object]]
    fit_subjects: list[str]
    threshold_subjects: list[str]
    threshold: float


@dataclass(frozen=True)
class FeatureTables:
    tables: dict[str, pd.DataFrame]
    groups: dict[str, dict[str, str]]
    categorical: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def __getitem__(self, model_id: str) -> pd.DataFrame:
        return self.tables[model_id]


def train_model_on_outer_fold(
    spec: ModelSpec,
    features: pd.DataFrame,
    registry: dict[str, Any],
    *,
    random_seed: int,
) -> FoldTrainingResult:
    _validate_registry(registry)
    train_subjects = [str(subject) for subject in registry["supervised_train_subjects"]]
    test_subject = str(registry["test_subject"])
    outer_fold = int(registry["outer_fold"])
    _require_subjects(features, [*train_subjects, test_subject])

    train = _rows_for_subjects(features, train_subjects)
    test = _rows_for_subjects(features, [test_subject])
    y_train = _encode_labels(train["label_primary"])
    if len(set(y_train.tolist())) < 2:
        raise ValueError(f"Outer train labels must contain both classes for {spec.model_id} fold {outer_fold}")

    best_params, threshold, threshold_subjects = _select_hyperparameters_and_threshold(
        spec,
        features,
        registry,
        random_seed=random_seed,
    )
    pipeline = _fit_pipeline(spec, train, y_train, best_params, random_seed=random_seed)
    prob_good = float(pipeline.predict_proba(test[spec.feature_columns])[:, 1][0])
    y_true = int(LABEL_TO_INT[str(test.iloc[0]["label_primary"])])
    pred_int = int(prob_good >= threshold)
    prediction = {
        "model_id": spec.model_id,
        "outer_fold": outer_fold,
        "subject_id": test_subject,
        "label_true": str(test.iloc[0]["label_primary"]),
        "y_true": y_true,
        "prob_good": prob_good,
        "pred_label": INT_TO_LABEL[pred_int],
        "threshold": float(threshold),
        "n_train_subjects": int(len(train_subjects)),
        "selected_c": float(best_params["C"]),
        "selected_l1_ratio": float(best_params.get("l1_ratio", 0.0)),
    }
    importance = _linear_importance(spec, pipeline, outer_fold=outer_fold)
    return FoldTrainingResult(
        prediction=prediction,
        importance=importance,
        fit_subjects=train_subjects,
        threshold_subjects=threshold_subjects,
        threshold=float(threshold),
    )


def run_classical_ml_baselines(
    config: dict[str, Any],
    *,
    cohort: pd.DataFrame,
    handcrafted: pd.DataFrame,
    tacs: pd.DataFrame,
    folds: dict[str, Any],
    registries: list[dict[str, Any]],
    psd: pd.DataFrame | None = None,
    fc: pd.DataFrame | None = None,
) -> dict[str, str]:
    supervised = cohort.loc[cohort["role"].eq("supervised_main")].copy()
    supervised = supervised.sort_values("subject_id").reset_index(drop=True)
    feature_tables = build_feature_tables(supervised, handcrafted, tacs, psd=psd, fc=fc)
    specs = build_model_specs(config, feature_tables)
    registry_by_fold = {int(registry["outer_fold"]): registry for registry in registries}

    predictions: list[dict[str, object]] = []
    importance: list[dict[str, object]] = []
    seed = int(config.get("random_seed", 42))
    requested = [str(model_id) for model_id in config.get("models", REQUIRED_MODEL_IDS)]
    if "M0_majority" in requested:
        predictions.extend(_run_majority_model(feature_tables["M0_majority"], folds, registry_by_fold))
    for spec in specs:
        table = feature_tables[spec.model_id]
        for fold in folds["folds"]:
            registry = registry_by_fold[int(fold["outer_fold"])]
            result = train_model_on_outer_fold(spec, table, registry, random_seed=seed)
            predictions.append(result.prediction)
            importance.extend(result.importance)

    prediction_frame = pd.DataFrame(predictions)
    validate_patient_predictions(prediction_frame, expected_subject_count=int(folds["n_supervised_main"]))
    return write_classical_outputs(config, prediction_frame, pd.DataFrame(importance))


def build_feature_tables(
    supervised: pd.DataFrame,
    handcrafted: pd.DataFrame,
    tacs: pd.DataFrame,
    *,
    psd: pd.DataFrame | None = None,
    fc: pd.DataFrame | None = None,
) -> FeatureTables:
    base = supervised[["subject_id", "label_primary"]].copy()
    clinical_numeric = [column for column in CLINICAL_NUMERIC_COLUMNS if column in supervised.columns]
    clinical_categorical = [column for column in CLINICAL_CATEGORICAL_COLUMNS if column in supervised.columns]
    clinical = base.merge(supervised[["subject_id", *clinical_numeric, *clinical_categorical]], on="subject_id", how="left")

    tacs_numeric = _numeric_feature_columns(tacs)
    handcrafted_numeric = _numeric_feature_columns(handcrafted)
    tacs_table = base.merge(tacs[["subject_id", *tacs_numeric]], on="subject_id", how="left")
    handcrafted_table = base.merge(handcrafted[["subject_id", *handcrafted_numeric]], on="subject_id", how="left")

    tables: dict[str, pd.DataFrame] = {
        "M0_majority": base,
        "M1_fma_only": clinical[["subject_id", "label_primary", "baseline_fma"]],
        "M2_clinical_only": clinical,
        "M5_tacs_target_ml": tacs_table,
    }
    groups: dict[str, dict[str, str]] = {
        "M0_majority": {},
        "M1_fma_only": {"baseline_fma": "clinical"},
        "M2_clinical_only": {column: "clinical" for column in [*clinical_numeric, *clinical_categorical]},
        "M5_tacs_target_ml": {column: "tacs_target" for column in tacs_numeric},
    }
    categorical: dict[str, tuple[str, ...]] = {
        "M1_fma_only": (),
        "M2_clinical_only": tuple(clinical_categorical),
        "M5_tacs_target_ml": (),
    }

    if psd is not None:
        psd_columns = [column for column in psd.columns if column != "subject_id"]
        psd_table = base.merge(psd, on="subject_id", how="left")
        tables["M3_psd_ml"] = psd_table
        groups["M3_psd_ml"] = {column: "psd_matrix" for column in psd_columns}
        categorical["M3_psd_ml"] = ()
    if fc is not None:
        fc_columns = [column for column in fc.columns if column != "subject_id"]
        fc_table = base.merge(fc, on="subject_id", how="left")
        tables["M4_fc_ml"] = fc_table
        groups["M4_fc_ml"] = {column: "fc_roi_matrix" for column in fc_columns}
        categorical["M4_fc_ml"] = ()

    eeg_parts = [handcrafted_table]
    eeg_groups = {column: _eeg_group(column, set(tacs_numeric)) for column in handcrafted_numeric}
    if psd is not None:
        eeg_parts.append(psd.drop(columns=["subject_id"]))
        eeg_groups.update({column: "psd_matrix" for column in psd.columns if column != "subject_id"})
    if fc is not None:
        eeg_parts.append(fc.drop(columns=["subject_id"]))
        eeg_groups.update({column: "fc_roi_matrix" for column in fc.columns if column != "subject_id"})
    all_eeg = pd.concat(eeg_parts, axis=1)
    all_eeg = all_eeg.loc[:, ~all_eeg.columns.duplicated()]
    tables["M6_all_handcrafted_eeg_ml"] = all_eeg
    groups["M6_all_handcrafted_eeg_ml"] = eeg_groups
    categorical["M6_all_handcrafted_eeg_ml"] = ()

    clinical_plus_eeg = clinical.merge(all_eeg.drop(columns=["label_primary"], errors="ignore"), on="subject_id", how="left")
    clinical_plus_eeg = clinical_plus_eeg.loc[:, ~clinical_plus_eeg.columns.duplicated()]
    tables["M12_clinical_plus_eeg_ml"] = clinical_plus_eeg
    groups["M12_clinical_plus_eeg_ml"] = {
        **{column: "clinical" for column in [*clinical_numeric, *clinical_categorical]},
        **eeg_groups,
    }
    categorical["M12_clinical_plus_eeg_ml"] = tuple(clinical_categorical)

    return FeatureTables(tables=tables, groups=groups, categorical=categorical)


def build_model_specs(config: dict[str, Any], feature_tables: FeatureTables) -> list[ModelSpec]:
    requested = [str(model_id) for model_id in config.get("models", REQUIRED_MODEL_IDS)]
    unknown = sorted(set(requested) - set(REQUIRED_MODEL_IDS))
    if unknown:
        raise ValueError(f"Unknown model IDs: {unknown}")
    missing = [model_id for model_id in requested if model_id not in feature_tables.tables]
    if missing:
        raise ValueError(f"Requested models missing feature tables: {missing}")

    hyper = config.get("hyperparameters", {})
    specs = []
    for model_id in REQUIRED_MODEL_IDS:
        if model_id not in requested or model_id == "M0_majority":
            continue
        table = feature_tables[model_id]
        feature_columns = [column for column in table.columns if column not in {"subject_id", "label_primary"}]
        estimator = "elasticnet_logistic" if model_id in {"M2_clinical_only", "M6_all_handcrafted_eeg_ml", "M12_clinical_plus_eeg_ml"} else "ridge_logistic"
        model_hyper = hyper.get(model_id, {})
        specs.append(
            ModelSpec(
                model_id=model_id,
                feature_columns=feature_columns,
                feature_groups=feature_tables.groups[model_id],
                estimator=str(model_hyper.get("estimator", estimator)),
                c_values=[float(value) for value in model_hyper.get("c_values", [0.1, 1.0])],
                l1_ratios=[float(value) for value in model_hyper.get("l1_ratios", [0.0])],
                categorical_columns=feature_tables.categorical.get(model_id, ()),
                max_importance_features=int(config.get("max_importance_features", 50)),
            )
        )
    ordered_specs = []
    for model_id in requested:
        if model_id == "M0_majority":
            continue
        ordered_specs.extend(spec for spec in specs if spec.model_id == model_id)
    return ordered_specs


def flatten_psd_matrices(subjects: list[str], psd_eo: np.ndarray, psd_ec: np.ndarray, dictionary: pd.DataFrame) -> pd.DataFrame:
    rows = {"subject_id": subjects}
    for matrix_file, matrix in (("psd_eo.npy", psd_eo), ("psd_ec.npy", psd_ec)):
        subset = dictionary[dictionary["matrix_file"].eq(matrix_file)].sort_values(
            ["axis1_view_index", "axis2_feature_index", "axis3_feature_index", "feature_name"]
        )
        for _, item in subset.iterrows():
            view = int(item["axis1_view_index"])
            channel = int(item["axis2_feature_index"])
            freq = int(item["axis3_feature_index"])
            rows[str(item["feature_name"])] = matrix[:, view, channel, freq]
    return pd.DataFrame(rows)


def flatten_fc_matrices(subjects: list[str], fc_eo: np.ndarray, fc_ec: np.ndarray, dictionary: pd.DataFrame) -> pd.DataFrame:
    metric_to_index = {"coherence": 0, "wpli": 1}
    rows = {"subject_id": subjects}
    for matrix_file, matrix in (("fc_roi_eo.npy", fc_eo), ("fc_roi_ec.npy", fc_ec)):
        subset = dictionary[dictionary["matrix_file"].eq(matrix_file)].sort_values(
            ["axis1_view_index", "axis2_feature_index", "axis3_feature_index", "metric", "feature_name"]
        )
        for _, item in subset.iterrows():
            view = int(item["axis1_view_index"])
            edge = int(item["axis2_feature_index"])
            band = int(item["axis3_feature_index"])
            metric = metric_to_index[str(item["metric"])]
            rows[str(item["feature_name"])] = matrix[:, view, edge, band, metric]
    return pd.DataFrame(rows)


def load_fold_registries(fold_dir: str | Path, outer_folds: dict[str, Any]) -> list[dict[str, Any]]:
    directory = Path(fold_dir)
    registries = []
    for fold in outer_folds["folds"]:
        path = directory / str(fold["registry_path"])
        registries.append(json.loads(path.read_text(encoding="utf-8")))
    return registries


def write_classical_outputs(config: dict[str, Any], predictions: pd.DataFrame, importance: pd.DataFrame) -> dict[str, str]:
    output_paths = {key: str(value) for key, value in config["output_paths"].items()}
    expected_subject_count = int(predictions.groupby("model_id")["subject_id"].nunique().max())
    validate_patient_predictions(predictions, expected_subject_count=expected_subject_count)
    metrics = compute_classification_metrics(predictions)
    bootstrap = bootstrap_metric_ci(
        predictions,
        n_bootstrap=int(config.get("bootstrap_resamples", 1000)),
        random_seed=int(config.get("random_seed", 42)),
    )
    permutation = permutation_test(
        predictions,
        n_permutations=int(config.get("permutation_resamples", 1000)),
        random_seed=int(config.get("random_seed", 42)),
        metrics=tuple(config.get("permutation_metrics", ["roc_auc"])),
    )
    frames = {
        "predictions": predictions.sort_values(["model_id", "outer_fold"]).reset_index(drop=True),
        "metrics": metrics,
        "bootstrap_ci": bootstrap,
        "permutation": permutation,
        "feature_importance": importance.sort_values(["model_id", "outer_fold", "rank_in_fold"]).reset_index(drop=True)
        if not importance.empty
        else _empty_importance_frame(),
    }
    for key, frame in frames.items():
        assert_public_feature_output(frame)
        path = Path(output_paths[key])
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        output_paths[key] = str(path)
    return output_paths


def _run_majority_model(
    table: pd.DataFrame,
    folds: dict[str, Any],
    registry_by_fold: dict[int, dict[str, Any]],
) -> list[dict[str, object]]:
    predictions = []
    for fold in folds["folds"]:
        outer_fold = int(fold["outer_fold"])
        registry = registry_by_fold[outer_fold]
        train_subjects = [str(subject) for subject in registry["supervised_train_subjects"]]
        test_subject = str(registry["test_subject"])
        train = _rows_for_subjects(table, train_subjects)
        test = _rows_for_subjects(table, [test_subject])
        y_train = _encode_labels(train["label_primary"])
        prob_good = float(np.mean(y_train))
        pred_int = int(prob_good >= 0.5)
        predictions.append(
            {
                "model_id": "M0_majority",
                "outer_fold": outer_fold,
                "subject_id": test_subject,
                "label_true": str(test.iloc[0]["label_primary"]),
                "y_true": int(LABEL_TO_INT[str(test.iloc[0]["label_primary"])]),
                "prob_good": prob_good,
                "pred_label": INT_TO_LABEL[pred_int],
                "threshold": 0.5,
                "n_train_subjects": int(len(train_subjects)),
                "selected_c": np.nan,
                "selected_l1_ratio": np.nan,
            }
        )
    return predictions


def _select_hyperparameters_and_threshold(
    spec: ModelSpec,
    features: pd.DataFrame,
    registry: dict[str, Any],
    *,
    random_seed: int,
) -> tuple[dict[str, float], float, list[str]]:
    candidates = [{"C": c_value, "l1_ratio": l1_ratio} for c_value in spec.c_values for l1_ratio in spec.l1_ratios]
    best_params = candidates[0]
    best_score = -np.inf
    best_y: list[int] = []
    best_prob: list[float] = []
    best_subjects: list[str] = []

    for candidate in candidates:
        all_y: list[int] = []
        all_prob: list[float] = []
        all_subjects: list[str] = []
        for inner in registry.get("inner_splits", []):
            inner_train_subjects = [str(subject) for subject in inner["train_subjects"]]
            inner_val_subjects = [str(subject) for subject in inner["val_subjects"]]
            train = _rows_for_subjects(features, inner_train_subjects)
            val = _rows_for_subjects(features, inner_val_subjects)
            y_train = _encode_labels(train["label_primary"])
            if len(set(y_train.tolist())) < 2 or val.empty:
                continue
            pipeline = _fit_pipeline(spec, train, y_train, candidate, random_seed=random_seed)
            prob = pipeline.predict_proba(val[spec.feature_columns])[:, 1]
            all_y.extend(_encode_labels(val["label_primary"]).tolist())
            all_prob.extend(prob.astype(float).tolist())
            all_subjects.extend(inner_val_subjects)
        score = _score_inner_predictions(all_y, all_prob)
        if score > best_score:
            best_score = score
            best_params = candidate
            best_y = all_y
            best_prob = all_prob
            best_subjects = all_subjects

    threshold = _select_threshold(np.asarray(best_y, dtype=int), np.asarray(best_prob, dtype=float))
    if not best_subjects:
        best_subjects = [str(subject) for subject in registry["threshold_selection_subjects"]]
    return best_params, threshold, sorted(set(best_subjects))


def _fit_pipeline(
    spec: ModelSpec,
    train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict[str, float],
    *,
    random_seed: int,
) -> Pipeline:
    pipeline = _pipeline_for_spec(spec, params, random_seed=random_seed)
    pipeline.fit(train[spec.feature_columns], y_train)
    return pipeline


def _pipeline_for_spec(spec: ModelSpec, params: dict[str, float], *, random_seed: int) -> Pipeline:
    categorical = [column for column in spec.categorical_columns if column in spec.feature_columns]
    numeric = [column for column in spec.feature_columns if column not in categorical]
    transformers = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)
    if spec.estimator == "elasticnet_logistic" and float(params.get("l1_ratio", 0.0)) > 0:
        classifier = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            C=float(params["C"]),
            l1_ratio=float(params["l1_ratio"]),
            max_iter=5000,
            tol=1e-3,
            class_weight="balanced",
            random_state=random_seed,
        )
    else:
        classifier = LogisticRegression(
            penalty="l2",
            solver="liblinear",
            C=float(params["C"]),
            max_iter=1000,
            class_weight="balanced",
            random_state=random_seed,
        )
    return Pipeline([("preprocess", preprocessor), ("classifier", classifier)])


def _linear_importance(spec: ModelSpec, pipeline: Pipeline, *, outer_fold: int) -> list[dict[str, object]]:
    classifier = pipeline.named_steps["classifier"]
    coefs = classifier.coef_.ravel()
    names = _clean_feature_names(pipeline.named_steps["preprocess"].get_feature_names_out())
    order = np.argsort(-np.abs(coefs))[: spec.max_importance_features]
    rows = []
    for rank, index in enumerate(order, start=1):
        feature_name = str(names[index])
        rows.append(
            {
                "model_id": spec.model_id,
                "outer_fold": int(outer_fold),
                "feature_name": feature_name,
                "feature_group": _feature_group_for_name(feature_name, spec.feature_groups),
                "coefficient": float(coefs[index]),
                "abs_coefficient": float(abs(coefs[index])),
                "rank_in_fold": int(rank),
            }
        )
    return rows


def _clean_feature_names(names: np.ndarray) -> list[str]:
    cleaned = []
    for name in names:
        text = str(name)
        for prefix in ("numeric__", "categorical__"):
            if text.startswith(prefix):
                text = text[len(prefix) :]
        cleaned.append(text)
    return cleaned


def _feature_group_for_name(feature_name: str, groups: dict[str, str]) -> str:
    if feature_name in groups:
        return groups[feature_name]
    for raw_name, group in groups.items():
        if feature_name.startswith(f"{raw_name}_") or feature_name.startswith(raw_name):
            return group
    return "unknown"


def _select_threshold(y_true: np.ndarray, prob_good: np.ndarray) -> float:
    if y_true.size == 0 or prob_good.size == 0 or len(set(y_true.tolist())) < 2:
        return 0.5
    thresholds = np.unique(np.concatenate([prob_good, np.asarray([0.5])]))
    best_threshold = 0.5
    best_score = -np.inf
    for threshold in thresholds:
        pred = (prob_good >= threshold).astype(int)
        positives = y_true == 1
        negatives = y_true == 0
        sensitivity = float(np.mean(pred[positives] == 1)) if positives.any() else 0.0
        specificity = float(np.mean(pred[negatives] == 0)) if negatives.any() else 0.0
        score = sensitivity + specificity - 1.0
        if score > best_score or (score == best_score and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)):
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _score_inner_predictions(y_true: list[int], prob_good: list[float]) -> float:
    if not y_true or len(set(y_true)) < 2:
        return -np.inf
    try:
        return float(roc_auc_score(y_true, prob_good))
    except ValueError:
        return -np.inf


def _validate_registry(registry: dict[str, Any]) -> None:
    test_subject = str(registry["test_subject"])
    train_subjects = {str(subject) for subject in registry["supervised_train_subjects"]}
    if test_subject in train_subjects:
        raise ValueError(f"Outer test subject appears in train subjects: {test_subject}")
    for key in ("normalization_fit_subjects", "feature_selection_fit_subjects", "threshold_selection_subjects"):
        fit_subjects = {str(subject) for subject in registry.get(key, [])}
        if test_subject in fit_subjects or not fit_subjects <= train_subjects:
            raise ValueError(f"{key} must be a subset of outer train subjects")
    for inner in registry.get("inner_splits", []):
        inner_subjects = {str(subject) for subject in inner.get("train_subjects", [])}
        inner_subjects.update(str(subject) for subject in inner.get("val_subjects", []))
        if test_subject in inner_subjects or not inner_subjects <= train_subjects:
            raise ValueError("Inner CV subjects must be a subset of outer train subjects")


def _rows_for_subjects(frame: pd.DataFrame, subjects: list[str]) -> pd.DataFrame:
    rows = frame.set_index("subject_id", drop=False).reindex(subjects)
    if rows["subject_id"].isna().any():
        missing = rows.index[rows["subject_id"].isna()].tolist()
        raise ValueError(f"Missing subjects in feature table: {missing}")
    return rows.reset_index(drop=True)


def _require_subjects(frame: pd.DataFrame, subjects: list[str]) -> None:
    available = set(frame["subject_id"].astype(str))
    missing = [subject for subject in subjects if subject not in available]
    if missing:
        raise ValueError(f"Missing subjects in feature table: {missing}")


def _encode_labels(labels: pd.Series) -> np.ndarray:
    unknown = sorted(set(labels.astype(str)) - set(LABEL_TO_INT))
    if unknown:
        raise ValueError(f"Labels must be Good/Poor, found: {unknown}")
    return labels.astype(str).map(LABEL_TO_INT).astype(int).to_numpy()


def _numeric_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in IDENTIFIER_COLUMNS and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _eeg_group(column: str, tacs_columns: set[str]) -> str:
    if column in tacs_columns or "target" in column or "homologous" in column:
        return "tacs_target"
    if "fc" in column.lower():
        return "fc_roi_matrix"
    return "handcrafted_summary"


def _empty_importance_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_id",
            "outer_fold",
            "feature_name",
            "feature_group",
            "coefficient",
            "abs_coefficient",
            "rank_in_fold",
        ]
    )
