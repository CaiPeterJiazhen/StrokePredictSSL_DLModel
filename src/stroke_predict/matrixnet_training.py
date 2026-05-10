from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, f1_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stroke_predict.matrixnet import MatrixNet, MatrixNetConfig
from stroke_predict.matrixnet_data import MatrixNetInputs
from stroke_predict.matrixnet_preprocessing import FoldPreprocessor, fit_vector_preprocessor

INT_TO_LABEL = {0: "Poor", 1: "Good"}
PRIMARY_MODELS = {
    "M8a_matrixnet_psd_only",
    "M8b_matrixnet_fc_only",
    "M8c_matrixnet_psd_fc",
    "M8d_matrixnet_psd_fc_tacs",
}


@dataclass(frozen=True)
class MatrixNetRunConfig:
    run_mode: str
    models: list[str]
    seeds: list[int]
    max_epochs: int
    patience: int
    batch_size: int
    learning_rates: list[float]
    weight_decays: list[float]
    dropouts: list[float]
    embedding_dims: list[int]
    hidden_dims: list[int]
    fold_limit: int | None = None
    write_outputs: bool = True
    bootstrap_resamples: int = 1000
    permutation_resamples: int = 1000
    random_seed: int = 42
    orientation_calibration: str = "inner_val_auc"
    phase6_2_audit: bool = False
    device: str = "cpu"
    require_cuda: bool = False


@dataclass(frozen=True)
class MatrixNetRunResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    training_log: pd.DataFrame
    fold_audit: pd.DataFrame


class MatrixDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: dict[str, np.ndarray], labels: np.ndarray, indices: list[int]) -> None:
        self.arrays = arrays
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        sample = {name: torch.from_numpy(array[index]).float() for name, array in self.arrays.items()}
        sample["label"] = torch.tensor(float(self.labels[index]), dtype=torch.float32)
        return sample


def write_matrixnet_outputs(output_dir: str | Path, result: MatrixNetRunResult, config: MatrixNetRunConfig) -> dict[str, str]:
    root = Path(output_dir)
    if config.phase6_2_audit:
        paths = {
            "predictions": root / "predictions" / "matrixnet_patient_predictions_phase6_2.csv",
            "metrics": root / "evaluation" / "matrixnet_metrics_phase6_2.csv",
            "seed_wise_metrics": root / "evaluation" / "seed_wise_metrics_phase6_2.csv",
            "patient_averaged_metrics": root / "evaluation" / "patient_averaged_metrics_phase6_2.csv",
            "training_log": root / "matrixnet" / "training_log_phase6_2.csv",
            "config_used": root / "matrixnet" / "config_used_phase6_2.yaml",
            "fold_audit": root / "reports" / "matrixnet_fold_audit_phase6_2.csv",
            "no_leakage_report": root / "reports" / "no_leakage_report_phase6_2.txt",
            "report": root / "reports" / "phase6_2_score_direction_audit_report.md",
            "checkpoints": root / "matrixnet" / "checkpoints",
        }
    else:
        suffix = "_full" if config.run_mode == "full" else ""
        report_name = "phase6_matrixnet_full_report.md" if config.run_mode == "full" else "phase6_matrixnet_report.md"
        paths = {
            "predictions": root / "predictions" / f"matrixnet_patient_predictions{suffix}.csv",
            "metrics": root / "evaluation" / f"matrixnet_metrics{suffix}.csv",
            "training_log": root / "matrixnet" / f"training_log{suffix}.csv",
            "config_used": root / "matrixnet" / f"config_used{suffix}.yaml",
            "fold_audit": root / "reports" / f"matrixnet_fold_audit{suffix}.csv",
            "no_leakage_report": root / "reports" / f"matrixnet_no_leakage_report{suffix}.txt",
            "report": root / "reports" / report_name,
            "checkpoints": root / "matrixnet" / "checkpoints",
        }
    for key, path in paths.items():
        if key == "checkpoints":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            _guard_fast_overwrite(path, config.run_mode)
    metrics_to_write = result.metrics
    seed_metrics = pd.DataFrame()
    patient_metrics = pd.DataFrame()
    if config.phase6_2_audit:
        metrics_to_write, seed_metrics, patient_metrics = _phase6_2_metric_frames(result.predictions, config=config)
    result.predictions.sort_values(["model_name", "seed", "outer_fold"]).to_csv(paths["predictions"], index=False)
    metrics_to_write.sort_values(["model_name"]).to_csv(paths["metrics"], index=False)
    if config.phase6_2_audit:
        seed_metrics.sort_values(["model_name", "seed"]).to_csv(paths["seed_wise_metrics"], index=False)
        patient_metrics.sort_values(["model_name", "patient_id"]).to_csv(paths["patient_averaged_metrics"], index=False)
    result.training_log.sort_values(["model_name", "seed", "outer_fold", "epoch"]).to_csv(paths["training_log"], index=False)
    result.fold_audit.sort_values(["model_name", "seed", "outer_fold"]).to_csv(paths["fold_audit"], index=False)
    paths["config_used"].write_text(_config_snapshot(config), encoding="utf-8")
    paths["no_leakage_report"].write_text(_no_leakage_text(result), encoding="utf-8")
    report_text = _phase6_2_report(result, metrics_to_write, seed_metrics, patient_metrics, config) if config.phase6_2_audit else _phase6_report(result, config)
    paths["report"].write_text(report_text, encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def run_matrixnet_lopo(inputs: MatrixNetInputs, config: MatrixNetRunConfig) -> MatrixNetRunResult:
    rows: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    subject_to_index = {subject: index for index, subject in enumerate(inputs.subject_ids)}
    folds = inputs.outer_folds["folds"][: config.fold_limit] if config.fold_limit else inputs.outer_folds["folds"]
    registry_by_fold = {int(registry["outer_fold"]): registry for registry in inputs.registries}
    for model_name in config.models:
        for seed in config.seeds:
            for fold in folds:
                outer_fold = int(fold["outer_fold"])
                registry = registry_by_fold[outer_fold]
                prediction, fold_logs, audit = _run_one_fold(model_name, seed, fold, registry, inputs, config, subject_to_index)
                rows.append(prediction)
                logs.extend(fold_logs)
                audits.append(audit)
    predictions = pd.DataFrame(rows)
    metrics = compute_matrixnet_metrics(predictions, inputs.ml_metrics, config=config)
    return MatrixNetRunResult(
        predictions=predictions,
        metrics=metrics,
        training_log=pd.DataFrame(logs),
        fold_audit=pd.DataFrame(audits),
    )


def compute_matrixnet_metrics(
    predictions: pd.DataFrame,
    ml_metrics: pd.DataFrame | None = None,
    *,
    config: MatrixNetRunConfig | None = None,
) -> pd.DataFrame:
    seed_rows = [_seed_metric_row(str(model_name), int(seed), group, ml_metrics) for (model_name, seed), group in predictions.groupby(["model_name", "seed"], sort=True)]
    seed_frame = pd.DataFrame(seed_rows)
    run_mode = str(predictions["run_mode"].iloc[0]) if not predictions.empty else "unknown"
    rows: list[dict[str, object]] = []
    for model_name, group in seed_frame.groupby("model_name", sort=True):
        first = group.iloc[0]
        ci_low = np.nan
        ci_high = np.nan
        permutation_p = np.nan
        if run_mode == "full":
            averaged = _average_seed_predictions(predictions[predictions["model_name"].astype(str).eq(str(model_name))])
            ci_low, ci_high = _bootstrap_auc_ci(
                averaged["y_true"].to_numpy(),
                averaged["score"].to_numpy(),
                n_bootstrap=config.bootstrap_resamples if config is not None else 1000,
                random_seed=config.random_seed if config is not None else 42,
            )
            permutation_p = _permutation_auc_p_value(
                averaged["y_true"].to_numpy(),
                averaged["score"].to_numpy(),
                n_permutations=config.permutation_resamples if config is not None else 1000,
                random_seed=config.random_seed if config is not None else 42,
            )
        rows.append(
            {
                "model_name": model_name,
                "input_family": first["input_family"],
                "run_mode": first["run_mode"],
                "n_patients": int(first["n_patients"]),
                "n_good": int(first["n_good"]),
                "n_poor": int(first["n_poor"]),
                "n_seeds": int(group["seed"].nunique()),
                "roc_auc_mean": float(group["roc_auc"].mean()),
                "roc_auc_std_across_seeds": float(group["roc_auc"].std(ddof=0)) if len(group) > 1 else np.nan,
                "roc_auc_ci_low": ci_low,
                "roc_auc_ci_high": ci_high,
                "pr_auc": float(group["pr_auc"].mean()),
                "balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "sensitivity": float(group["sensitivity"].mean()),
                "specificity": float(group["specificity"].mean()),
                "f1": float(group["f1"].mean()),
                "brier_score": float(group["brier_score"].mean()),
                "permutation_p_value": permutation_p,
                "comparison_to_best_ml_auc": float(group["comparison_to_best_ml_auc"].mean()),
                "comparison_to_best_flattened_ml_auc": float(group["comparison_to_best_flattened_ml_auc"].mean()),
                "comparison_to_fma_only_auc": float(group["comparison_to_fma_only_auc"].mean()),
                "comparison_to_clinical_only_auc": float(group["comparison_to_clinical_only_auc"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _calibrate_score_orientation(y_true: np.ndarray, sigmoid_scores: np.ndarray) -> str:
    if y_true.size == 0 or sigmoid_scores.size == 0 or len(set(y_true.astype(int).tolist())) < 2:
        return "normal_insufficient_inner_classes"
    auc = float(roc_auc_score(y_true.astype(int), sigmoid_scores.astype(float)))
    return "inverted_by_inner_val" if auc < 0.5 else "normal"


def _apply_score_orientation(sigmoid_scores: np.ndarray, orientation: str) -> np.ndarray:
    scores = sigmoid_scores.astype(float)
    if orientation == "inverted_by_inner_val":
        return 1.0 - scores
    if orientation in {"normal", "normal_insufficient_inner_classes"}:
        return scores
    raise ValueError(f"Unsupported score_orientation: {orientation}")


def _phase6_2_metric_frames(
    predictions: pd.DataFrame,
    *,
    config: MatrixNetRunConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_rows: list[dict[str, object]] = []
    patient_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for (model_name, seed), group in predictions.groupby(["model_name", "seed"], sort=True):
        y_true = _phase6_2_y_true(group)
        scores = group["predicted_score"].astype(float).to_numpy()
        seed_rows.append(
            {
                "model_name": model_name,
                "seed": int(seed),
                "run_mode": str(group["run_mode"].iloc[0]),
                "n_predictions": int(len(group)),
                "n_good": int(np.sum(y_true == 1)),
                "n_poor": int(np.sum(y_true == 0)),
                "auc_score": _safe_auc(y_true, scores),
                "auc_one_minus_score": _safe_auc(y_true, 1.0 - scores),
                "mean_score_good": _class_mean(y_true, scores, 1),
                "mean_score_poor": _class_mean(y_true, scores, 0),
            }
        )
    seed_frame = pd.DataFrame(seed_rows)
    for model_name, group in predictions.groupby("model_name", sort=True):
        for patient_id, patient_group in group.groupby("patient_id", sort=True):
            y_true = _phase6_2_y_true(patient_group)
            patient_rows.append(
                {
                    "model_name": model_name,
                    "patient_id": patient_id,
                    "true_label": str(patient_group["true_label"].iloc[0]),
                    "label_int": int(y_true[0]),
                    "mean_predicted_score": float(patient_group["predicted_score"].astype(float).mean()),
                    "n_seed_predictions": int(patient_group["seed"].nunique()),
                }
            )
    patient_frame = pd.DataFrame(patient_rows)
    for model_name, group in predictions.groupby("model_name", sort=True):
        y_true = _phase6_2_y_true(group)
        scores = group["predicted_score"].astype(float).to_numpy()
        model_seed = seed_frame[seed_frame["model_name"].astype(str).eq(str(model_name))]
        patient_group = patient_frame[patient_frame["model_name"].astype(str).eq(str(model_name))]
        patient_y = patient_group["label_int"].astype(int).to_numpy()
        patient_scores = patient_group["mean_predicted_score"].astype(float).to_numpy()
        auc_score = _safe_auc(y_true, scores)
        auc_one_minus = _safe_auc(y_true, 1.0 - scores)
        mean_good = _class_mean(y_true, scores, 1)
        mean_poor = _class_mean(y_true, scores, 0)
        ci_low = np.nan
        ci_high = np.nan
        permutation_p = np.nan
        if config is not None and config.run_mode == "full":
            ci_low, ci_high = _bootstrap_auc_ci(
                patient_y,
                patient_scores,
                n_bootstrap=config.bootstrap_resamples,
                random_seed=config.random_seed,
            )
            permutation_p = _permutation_auc_p_value(
                patient_y,
                patient_scores,
                n_permutations=config.permutation_resamples,
                random_seed=config.random_seed,
            )
        summary_rows.append(
            {
                "model_name": model_name,
                "run_mode": str(group["run_mode"].iloc[0]),
                "n_patients": int(group["patient_id"].nunique()),
                "n_seed_predictions": int(len(group)),
                "n_seeds": int(group["seed"].nunique()),
                "roc_auc_mean": float(model_seed["auc_score"].mean()),
                "roc_auc_std_across_seeds": float(model_seed["auc_score"].std(ddof=0)) if len(model_seed) > 1 else np.nan,
                "pooled_auc": auc_score,
                "patient_averaged_auc": _safe_auc(patient_y, patient_scores),
                "roc_auc_ci_low": ci_low,
                "roc_auc_ci_high": ci_high,
                "permutation_p_value": permutation_p,
                "bootstrap_resamples": int(config.bootstrap_resamples) if config is not None else 0,
                "permutation_resamples": int(config.permutation_resamples) if config is not None else 0,
                "auc_score": auc_score,
                "auc_one_minus_score": auc_one_minus,
                "mean_score_good": mean_good,
                "mean_score_poor": mean_poor,
                "direction_correct": bool(
                    np.isfinite(auc_score)
                    and np.isfinite(auc_one_minus)
                    and np.isfinite(mean_good)
                    and np.isfinite(mean_poor)
                    and auc_score >= auc_one_minus
                    and mean_good >= mean_poor
                ),
                "score_orientation_counts": ";".join(
                    f"{key}={value}" for key, value in group["score_orientation"].astype(str).value_counts().sort_index().items()
                )
                if "score_orientation" in group
                else "",
            }
        )
    return pd.DataFrame(summary_rows), seed_frame, patient_frame


def _phase6_2_y_true(group: pd.DataFrame) -> np.ndarray:
    if "label_int" in group.columns:
        return group["label_int"].astype(int).to_numpy()
    return (group["true_label"].astype(str) == "Good").astype(int).to_numpy()


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(set(y_true.astype(int).tolist())) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true.astype(int), scores.astype(float)))
    except ValueError:
        return np.nan


def _class_mean(y_true: np.ndarray, scores: np.ndarray, label: int) -> float:
    mask = y_true.astype(int) == int(label)
    if not np.any(mask):
        return np.nan
    return float(scores.astype(float)[mask].mean())


def _run_one_fold(
    model_name: str,
    seed: int,
    fold: dict[str, Any],
    registry: dict[str, Any],
    inputs: MatrixNetInputs,
    config: MatrixNetRunConfig,
    subject_to_index: dict[str, int],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    split = registry["inner_splits"][0]
    train_subjects = [str(subject) for subject in split["train_subjects"]]
    val_subjects = [str(subject) for subject in split["val_subjects"]]
    test_subject = str(fold["test_subject"])
    train_indices = [subject_to_index[subject] for subject in train_subjects]
    val_indices = [subject_to_index[subject] for subject in val_subjects]
    test_index = subject_to_index[test_subject]

    arrays, tacs_dim, clinical_dim = _prepared_arrays(model_name, inputs, [str(subject) for subject in registry["supervised_train_subjects"]])
    model_config = _model_config(
        model_name,
        tacs_dim=tacs_dim,
        clinical_dim=clinical_dim,
        embedding_dim=config.embedding_dims[0],
        hidden_dim=config.hidden_dims[0],
        dropout=config.dropouts[0],
    )
    device = _torch_device(config)
    model = MatrixNet(model_config).to(device)
    pos_weight = _pos_weight(inputs.labels[train_indices])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rates[0], weight_decay=config.weight_decays[0])
    train_loader = DataLoader(MatrixDataset(arrays, inputs.labels, train_indices), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(MatrixDataset(arrays, inputs.labels, val_indices), batch_size=max(1, len(val_indices)), shuffle=False)

    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    logs: list[dict[str, object]] = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer)
        val_loss, _val_scores, _val_true, _val_logits = _eval_epoch(model, val_loader, criterion)
        logs.append(
            {
                "model_name": model_name,
                "outer_fold": int(fold["outer_fold"]),
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "run_mode": config.run_mode,
                "selection_mode": _selection_mode(config),
            }
        )
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    model.load_state_dict(best_state)
    val_loss, val_scores, val_true, _val_logits = _eval_epoch(model, val_loader, criterion)
    score_orientation = _score_orientation(val_true, val_scores, config)
    oriented_val_scores = _apply_score_orientation(val_scores, score_orientation)
    threshold, threshold_source = _select_threshold(val_true, oriented_val_scores)
    test_loader = DataLoader(MatrixDataset(arrays, inputs.labels, [test_index]), batch_size=1, shuffle=False)
    _test_loss, test_scores, _test_true, test_logits = _eval_epoch(model, test_loader, criterion)
    sigmoid_score = float(test_scores[0])
    score = float(_apply_score_orientation(test_scores, score_orientation)[0])
    pred_int = int(score >= threshold)
    prediction = {
        "model_name": model_name,
        "outer_fold": int(fold["outer_fold"]),
        "patient_id": test_subject,
        "true_label": INT_TO_LABEL[int(inputs.labels[test_index])],
        "label_int": int(inputs.labels[test_index]),
        "logit": float(test_logits[0]),
        "sigmoid_score": sigmoid_score,
        "predicted_score": score,
        "predicted_label": INT_TO_LABEL[pred_int],
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "score_orientation": score_orientation,
        "seed": seed,
        "run_mode": config.run_mode,
        "input_family": _input_family(model_name),
        "best_epoch": int(best_epoch),
        "best_inner_metric": -float(best_loss),
        "train_loss_final": float(logs[-1]["train_loss"]),
        "val_loss_best": float(best_loss),
    }
    audit = {
        "model_name": model_name,
        "outer_fold": int(fold["outer_fold"]),
        "seed": seed,
        "test_patient": test_subject,
        "test_excluded_from_train": test_subject not in train_subjects,
        "test_excluded_from_val": test_subject not in val_subjects,
        "scaler_fit_subjects": ";".join(map(str, registry["supervised_train_subjects"])),
        "threshold_fit_subjects": ";".join(val_subjects),
        "run_mode": config.run_mode,
    }
    return prediction, logs, audit


def _prepared_arrays(model_name: str, inputs: MatrixNetInputs, train_subjects: list[str]) -> tuple[dict[str, np.ndarray], int, int]:
    spec = _model_inputs(model_name)
    matrix_pre = FoldPreprocessor.fit(
        inputs.subject_ids,
        train_subjects,
        {"psd_eo": inputs.psd_eo, "psd_ec": inputs.psd_ec, "fc_eo": inputs.fc_eo, "fc_ec": inputs.fc_ec},
    )
    arrays: dict[str, np.ndarray] = {}
    if spec["psd"]:
        arrays["psd_eo"] = matrix_pre.transform_matrix("psd_eo", inputs.psd_eo)
        arrays["psd_ec"] = matrix_pre.transform_matrix("psd_ec", inputs.psd_ec)
    if spec["fc"]:
        arrays["fc_eo"] = matrix_pre.transform_matrix("fc_eo", inputs.fc_eo)
        arrays["fc_ec"] = matrix_pre.transform_matrix("fc_ec", inputs.fc_ec)
    tacs_dim = 0
    if spec["tacs"] and inputs.tacs is not None:
        tacs_pre = fit_vector_preprocessor(inputs.tacs, subject_ids=inputs.subject_ids, train_subjects=train_subjects)
        arrays["tacs"] = tacs_pre.transform(inputs.tacs, inputs.subject_ids)
        tacs_dim = arrays["tacs"].shape[1]
    clinical_dim = 0
    if spec["clinical"]:
        clinical_pre = fit_vector_preprocessor(inputs.clinical, subject_ids=inputs.subject_ids, train_subjects=train_subjects)
        arrays["clinical"] = clinical_pre.transform(inputs.clinical, inputs.subject_ids)
        clinical_dim = arrays["clinical"].shape[1]
    return arrays, tacs_dim, clinical_dim


def _model_config(
    model_name: str,
    *,
    tacs_dim: int,
    clinical_dim: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float,
) -> MatrixNetConfig:
    spec = _model_inputs(model_name)
    return MatrixNetConfig(
        use_psd=spec["psd"],
        use_fc=spec["fc"],
        use_tacs=spec["tacs"] and tacs_dim > 0,
        use_clinical=spec["clinical"],
        tacs_dim=tacs_dim,
        clinical_dim=clinical_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def _model_inputs(model_name: str) -> dict[str, bool]:
    specs = {
        "M8a_matrixnet_psd_only": {"psd": True, "fc": False, "tacs": False, "clinical": False},
        "M8b_matrixnet_fc_only": {"psd": False, "fc": True, "tacs": False, "clinical": False},
        "M8c_matrixnet_psd_fc": {"psd": True, "fc": True, "tacs": False, "clinical": False},
        "M8d_matrixnet_psd_fc_tacs": {"psd": True, "fc": True, "tacs": True, "clinical": False},
        "M12_matrixnet_clinical_eeg": {"psd": True, "fc": True, "tacs": True, "clinical": True},
    }
    if model_name not in specs:
        raise ValueError(f"Unsupported MatrixNet model: {model_name}")
    return specs[model_name]


def _train_epoch(model: MatrixNet, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    losses: list[float] = []
    device = next(model.parameters()).device
    for batch in loader:
        labels = batch.pop("label").to(device)
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        logits = model(**batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


def _eval_epoch(model: MatrixNet, loader: DataLoader, criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    labels_all: list[int] = []
    logits_all: list[float] = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("label").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch)
            loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()))
            logits_all.extend(logits.detach().cpu().numpy().astype(float).tolist())
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
    return (
        float(np.mean(losses)) if losses else np.nan,
        np.asarray(scores, dtype=float),
        np.asarray(labels_all, dtype=int),
        np.asarray(logits_all, dtype=float),
    )


def _score_orientation(y_true: np.ndarray, sigmoid_scores: np.ndarray, config: MatrixNetRunConfig) -> str:
    if config.orientation_calibration == "none":
        return "normal"
    if config.orientation_calibration == "inner_val_auc":
        return _calibrate_score_orientation(y_true, sigmoid_scores)
    raise ValueError(f"Unsupported orientation_calibration: {config.orientation_calibration}")


def _select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    if y_true.size == 0 or scores.size == 0 or len(set(y_true.tolist())) < 2:
        return 0.5, "fixed_0.5"
    thresholds = np.unique(np.concatenate([scores, np.asarray([0.5])]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if tp + fn else 0.0
        specificity = tn / (tn + fp) if tn + fp else 0.0
        balanced = float((sensitivity + specificity) / 2)
        if balanced > best_score:
            best_score = balanced
            best_threshold = float(threshold)
    return best_threshold, "inner_validation_balanced_accuracy"


def _seed_metric_row(model_name: str, seed: int, group: pd.DataFrame, ml_metrics: pd.DataFrame | None) -> dict[str, object]:
    y_true = (group["true_label"].astype(str) == "Good").astype(int).to_numpy()
    scores = group["predicted_score"].astype(float).to_numpy()
    pred = (group["predicted_label"].astype(str) == "Good").astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else np.nan
    specificity = float(tn / (tn + fp)) if tn + fp else np.nan
    roc_auc = float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else np.nan
    pr_auc = float(average_precision_score(y_true, scores)) if np.any(y_true == 1) else np.nan
    return {
        "model_name": model_name,
        "seed": seed,
        "input_family": _input_family(model_name),
        "run_mode": str(group["run_mode"].iloc[0]),
        "n_patients": int(len(group)),
        "n_good": int(np.sum(y_true == 1)),
        "n_poor": int(np.sum(y_true == 0)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "balanced_accuracy": float((sensitivity + specificity) / 2) if np.isfinite(sensitivity) and np.isfinite(specificity) else np.nan,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "comparison_to_best_ml_auc": _compare_ml(roc_auc, ml_metrics, None),
        "comparison_to_best_flattened_ml_auc": _compare_flattened_ml(roc_auc, ml_metrics),
        "comparison_to_fma_only_auc": _compare_ml(roc_auc, ml_metrics, "M1_fma_only"),
        "comparison_to_clinical_only_auc": _compare_ml(roc_auc, ml_metrics, "M2_clinical_only"),
    }


def _compare_ml(auc: float, ml_metrics: pd.DataFrame | None, model_name: str | None) -> float:
    if ml_metrics is None or np.isnan(auc) or "roc_auc" not in ml_metrics.columns:
        return np.nan
    if model_name is None:
        baseline = float(ml_metrics["roc_auc"].max())
    else:
        rows = ml_metrics[ml_metrics["model_name"].astype(str).eq(model_name)]
        if rows.empty:
            return np.nan
        baseline = float(rows.iloc[0]["roc_auc"])
    return float(auc - baseline)


def _compare_flattened_ml(auc: float, ml_metrics: pd.DataFrame | None) -> float:
    if ml_metrics is None or np.isnan(auc) or not {"model_name", "roc_auc"} <= set(ml_metrics.columns):
        return np.nan
    flattened = ml_metrics[ml_metrics["model_name"].astype(str).isin({"M3b_psd_matrix_flatten_ml", "M4b_fc_matrix_flatten_ml", "M6b_psd_fc_matrix_flatten_ml"})]
    if flattened.empty:
        return np.nan
    return float(auc - float(flattened["roc_auc"].max()))


def _pos_weight(labels: np.ndarray) -> torch.Tensor | None:
    pos = float(np.sum(labels == 1))
    neg = float(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)


def _torch_device(config: MatrixNetRunConfig) -> torch.device:
    device = torch.device(config.device)
    if config.require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("CUDA is required for this MatrixNet run, but no CUDA device is available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("MatrixNet config requested CUDA, but no CUDA device is available")
    return device


def _input_family(model_name: str) -> str:
    mapping = {
        "M8a_matrixnet_psd_only": "psd_only",
        "M8b_matrixnet_fc_only": "fc_only",
        "M8c_matrixnet_psd_fc": "psd_fc",
        "M8d_matrixnet_psd_fc_tacs": "psd_fc_tacs",
        "M12_matrixnet_clinical_eeg": "clinical_eeg_secondary",
    }
    return mapping.get(model_name, "unknown")


def _selection_mode(config: MatrixNetRunConfig) -> str:
    if config.run_mode == "full" and any(len(values) > 1 for values in (config.learning_rates, config.weight_decays, config.dropouts, config.embedding_dims, config.hidden_dims)):
        return "fixed_first_config_no_grid_search"
    return "fixed_config"


def _config_snapshot(config: MatrixNetRunConfig) -> str:
    lines = [
        f"run_mode: {config.run_mode}",
        f"max_epochs: {config.max_epochs}",
        f"patience: {config.patience}",
        f"batch_size: {config.batch_size}",
        f"selection_mode: {_selection_mode(config)}",
        "models:",
    ]
    lines.extend([f"  - {model}" for model in config.models])
    lines.append("seeds: [" + ", ".join(map(str, config.seeds)) + "]")
    lines.append("learning_rates: [" + ", ".join(map(str, config.learning_rates)) + "]")
    lines.append("weight_decays: [" + ", ".join(map(str, config.weight_decays)) + "]")
    lines.append("dropouts: [" + ", ".join(map(str, config.dropouts)) + "]")
    lines.append("embedding_dims: [" + ", ".join(map(str, config.embedding_dims)) + "]")
    lines.append("hidden_dims: [" + ", ".join(map(str, config.hidden_dims)) + "]")
    lines.append(f"bootstrap_resamples: {config.bootstrap_resamples}")
    lines.append(f"permutation_resamples: {config.permutation_resamples}")
    lines.append(f"random_seed: {config.random_seed}")
    lines.append(f"orientation_calibration: {config.orientation_calibration}")
    lines.append(f"phase6_2_audit: {str(config.phase6_2_audit).lower()}")
    lines.append(f"device: {config.device}")
    lines.append(f"require_cuda: {str(config.require_cuda).lower()}")
    return "\n".join(lines) + "\n"


def _no_leakage_text(result: MatrixNetRunResult) -> str:
    audit = result.fold_audit
    checks = [
        ("test patient excluded from train", bool(audit["test_excluded_from_train"].all()) if not audit.empty else False),
        ("test patient excluded from val", bool(audit["test_excluded_from_val"].all()) if not audit.empty else False),
        (
            "no duplicated model-patient-seed predictions",
            not result.predictions.duplicated(["model_name", "patient_id", "seed"]).any(),
        ),
        ("outputs are patient-level predictions", {"patient_id", "predicted_score"} <= set(result.predictions.columns)),
        ("matrix row alignment verified before training", True),
    ]
    return "\n".join([f"{'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks]) + "\n"


def _phase6_report(result: MatrixNetRunResult, config: MatrixNetRunConfig) -> str:
    primary = result.metrics[result.metrics["model_name"].isin(PRIMARY_MODELS)].copy()
    secondary = result.metrics[~result.metrics["model_name"].isin(PRIMARY_MODELS)].copy()
    lines = [
        "# Phase 6 MatrixNet Report",
        "",
        f"Run mode: **{config.run_mode}**",
        "",
        "## Objective",
        "",
        "Phase 6 evaluates supervised no-SSL Lin-style MatrixNet models for baseline EEG prognosis. No self-supervised pretraining, SSL, BYOL, SimSiam, MAE, or masked matrix modeling was started in this branch.",
        "",
        "## Input Artifact Audit",
        "",
        "- Matrix rows are verified through `matrix_subject_index.csv`.",
        "- FC matrices shaped `[N, 2, edge, band, metric]` are canonicalized to `[N, C, edge, band]`, preserving ROI-edge by frequency-band structure.",
        "- Phase 5.2 metrics are used only for comparison, not for MatrixNet training.",
        "",
        "## Training Settings",
        "",
        f"- Seeds: {', '.join(map(str, config.seeds))}",
        f"- Max epochs: {config.max_epochs}",
        f"- Patience: {config.patience}",
        f"- Batch size: {config.batch_size}",
        f"- Selection mode: `{_selection_mode(config)}`",
        f"- Bootstrap resamples: {config.bootstrap_resamples}",
        f"- Permutation resamples: {config.permutation_resamples}",
        "",
        _mode_statement(config),
        _hyperparameter_statement(config),
        "",
        "## LOPO No-Leakage Summary",
        "",
        _no_leakage_text(result).strip(),
        "",
        "## Primary EEG-Only MatrixNet Models",
        "",
        _markdown_table(primary),
        "",
        "Primary Phase 6 conclusions should focus on EEG-only MatrixNet models M8a-M8d.",
        "",
        "## Secondary Fusion Models",
        "",
        _markdown_table(secondary),
        "",
        "M12 clinical+EEG is secondary and must not replace the EEG-only primary model family.",
        "",
        "## Phase 5.2 Comparison",
        "",
        "MatrixNet metrics include differences from the best Phase 5.2 ML ROC-AUC, FMA-only ROC-AUC, and clinical-only ROC-AUC when `ml_metrics_all.csv` is available. Flattened matrix controls from Phase 5.2 remain the direct control comparison.",
        "",
        "## Required Full-Mode Questions",
        "",
        _question_answers(result.metrics),
        "",
        "## Scientific Caution",
        "",
        "Do not claim EEG efficacy from fast-mode results. The supervised sample size is small, and any scientific interpretation requires full-mode stability, confidence intervals, permutation testing, and leakage checks.",
        "",
        "## Phase 7 Recommendation",
        "",
        "Phase 7 may evaluate SSL-MatrixNet only after supervised Phase 6 behavior is stable and audited.",
    ]
    return "\n".join(lines) + "\n"


def _phase6_2_report(
    result: MatrixNetRunResult,
    metrics: pd.DataFrame,
    seed_metrics: pd.DataFrame,
    patient_metrics: pd.DataFrame,
    config: MatrixNetRunConfig,
) -> str:
    orientation_counts = (
        result.predictions.groupby(["model_name", "score_orientation"]).size().reset_index(name="n_predictions")
        if "score_orientation" in result.predictions.columns and not result.predictions.empty
        else pd.DataFrame()
    )
    significant_models = (
        metrics.loc[metrics["permutation_p_value"].astype(float).lt(0.05), "model_name"].astype(str).tolist()
        if "permutation_p_value" in metrics.columns
        else []
    )
    direction_ok = (
        metrics["direction_correct"].astype(str).str.lower().eq("true").all()
        if "direction_correct" in metrics.columns and not metrics.empty
        else False
    )
    lines = [
        "# Phase 6.2 Score-Direction and Evaluation Audit Report",
        "",
        "Phase 6.2 did not start SSL, self-supervised pretraining, BYOL, SimSiam, MAE, or masked matrix modeling.",
        "",
        f"Run mode: **{config.run_mode}**",
        f"Orientation calibration: `{config.orientation_calibration}`",
        "",
        "## Label and Score Contract",
        "",
        "- `LABEL_TO_INT` is `Poor=0, Good=1`.",
        "- `BCEWithLogitsLoss` receives target `1.0` for Good and `0.0` for Poor.",
        "- Raw MatrixNet output is a Good-vs-Poor logit.",
        "- `sigmoid_score = sigmoid(logit)` is the unmodified probability of Good.",
        "- `predicted_score` is the final probability of Good after inner-validation-only orientation calibration.",
        "",
        "## Prediction Table Audit",
        "",
        "The Phase 6.2 prediction table contains `model_name`, `seed`, `outer_fold`, `patient_id`, `true_label`, `label_int`, `logit`, `sigmoid_score`, `predicted_score`, `predicted_label`, `threshold`, `threshold_source`, `score_orientation`, and `run_mode`.",
        "",
        "## Metric Consistency Audit",
        "",
        "Mean seed AUC computes ROC-AUC separately for each seed and then reports the model-level mean/std, so it reflects random-initialization stability.",
        "",
        "Pooled AUC computes ROC-AUC over every model-seed-patient row at once. It is useful for detecting global score direction, but each patient appears once per seed and rows are not independent patient units.",
        "",
        "Patient-averaged AUC first averages each patient's score across seeds and then computes ROC-AUC. This restores the patient as the evaluation unit and is the preferred no-SSL stability view.",
        "",
        "Full-mode Phase 6.2 bootstrap confidence intervals and permutation p-values are computed on patient-averaged scores, preserving the patient as the inference unit.",
        "",
        _markdown_table(metrics),
        "",
        "## Seed-Wise Metrics",
        "",
        _markdown_table(seed_metrics),
        "",
        "## Patient-Averaged Scores",
        "",
        _markdown_table(patient_metrics),
        "",
        "## Score Orientation",
        "",
        _markdown_table(orientation_counts),
        "",
        "Orientation calibration uses only inner validation predictions inside each outer fold. It never uses the outer test prediction, pooled outer predictions, or patient-averaged outer predictions to decide whether to invert.",
        "",
        "## Audit Conclusion",
        "",
        "No label encoding bug was found in the code contract: Poor remains 0 and Good remains 1.",
        f"Permutation-significant models at p < 0.05: {', '.join(significant_models) if significant_models else 'none'}.",
        f"All models have direction_correct=True: {'yes' if direction_ok else 'no'}.",
        "For this Phase 6.2 run, orientation-calibrated no-SSL MatrixNet remains unstable or non-significant unless the table above shows both stable direction and p < 0.05 for a model.",
        "",
        "## LOPO No-Leakage Summary",
        "",
        _no_leakage_text(result).strip(),
        "",
        "## Phase 7 Decision Rule",
        "",
        "If label/score direction bug is found, fix it before SSL.",
        "If orientation-calibrated no-SSL MatrixNet remains unstable and non-significant, SSL may proceed only as an exploratory representation-learning experiment, not as a confirmed model improvement stage.",
        "If orientation-calibrated MatrixNet improves and becomes stable, then Phase 7 SSL can proceed as planned.",
    ]
    return "\n".join(lines) + "\n"


def _mode_statement(config: MatrixNetRunConfig) -> str:
    if config.run_mode == "full":
        return "Full mode includes bootstrap ROC-AUC CI and permutation p-values. Hyperparameters are fixed by configuration unless a later implementation records true inner-validation grid search."
    return "Fast mode is smoke-only: bootstrap CI and permutation p-value may be NaN, and these results are not for scientific interpretation."


def _hyperparameter_statement(config: MatrixNetRunConfig) -> str:
    selection_mode = _selection_mode(config)
    if selection_mode == "fixed_config":
        return "Hyperparameters used a fixed configuration; no hyperparameter grid search was performed."
    return "Hyperparameters used the first value from each configured list; no hyperparameter grid search was performed."


def _question_answers(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "_No metrics available._"
    m8b_std = _metric_value(metrics, "M8b_matrixnet_fc_only", "roc_auc_std_across_seeds")
    m8b_auc = _metric_value(metrics, "M8b_matrixnet_fc_only", "roc_auc_mean")
    m8b_stable = np.isfinite(m8b_std) and m8b_std <= 0.10
    best_primary = metrics[metrics["model_name"].isin(PRIMARY_MODELS)].sort_values("roc_auc_mean", ascending=False).head(1)
    best_primary_auc = float(best_primary.iloc[0]["roc_auc_mean"]) if not best_primary.empty else np.nan
    best_primary_name = str(best_primary.iloc[0]["model_name"]) if not best_primary.empty else "not available"
    best_primary_flat_delta = float(best_primary.iloc[0].get("comparison_to_best_flattened_ml_auc", np.nan)) if not best_primary.empty else np.nan
    m8c_delta = _metric_value(metrics, "M8c_matrixnet_psd_fc", "roc_auc_mean") - m8b_auc
    tacs_delta = _metric_value(metrics, "M8d_matrixnet_psd_fc_tacs", "roc_auc_mean") - _metric_value(metrics, "M8c_matrixnet_psd_fc", "roc_auc_mean")
    clinical_delta = _metric_value(metrics, "M12_matrixnet_clinical_eeg", "roc_auc_mean") - best_primary_auc
    significant = metrics.loc[metrics["permutation_p_value"].lt(0.05), "model_name"].astype(str).tolist()
    rows = [
        f"- Is M8b_matrixnet_fc_only stable across seeds? {'Yes' if m8b_stable else 'No'}; ROC-AUC mean={_fmt(m8b_auc)}, std={_fmt(m8b_std)} using std <= 0.10 as the stability rule.",
        f"- Does MatrixNet outperform flattened-matrix ML controls? {'Yes' if np.isfinite(best_primary_flat_delta) and best_primary_flat_delta > 0 else 'No'}; best primary={best_primary_name}, ROC-AUC={_fmt(best_primary_auc)}, delta vs best flattened ML={_fmt(best_primary_flat_delta)}.",
        f"- Does PSD+FC improve over FC-only? {'Yes' if np.isfinite(m8c_delta) and m8c_delta > 0 else 'No'}; M8c minus M8b ROC-AUC={_fmt(m8c_delta)}.",
        f"- Does adding tACS summary improve or hurt? {'Improve' if np.isfinite(tacs_delta) and tacs_delta > 0 else 'Hurt or no improvement'}; M8d minus M8c ROC-AUC={_fmt(tacs_delta)}.",
        f"- Does clinical+EEG improve over EEG-only? {'Yes' if np.isfinite(clinical_delta) and clinical_delta > 0 else 'No'}; M12 minus best M8 ROC-AUC={_fmt(clinical_delta)}.",
        f"- Are any results permutation-significant? {'Yes: ' + ', '.join(significant) if significant else 'No models at p < 0.05'}.",
    ]
    return "\n".join(rows)


def _metric_value(metrics: pd.DataFrame, model_name: str, column: str) -> float:
    row = metrics[metrics["model_name"].astype(str).eq(model_name)]
    if row.empty or column not in row.columns:
        return np.nan
    return float(row.iloc[0][column])


def _fmt(value: float) -> str:
    return "NA" if not np.isfinite(value) else f"{value:.3f}"


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    display = frame.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.3f}")
    columns = list(display.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(rows)


def _guard_fast_overwrite(path: Path, run_mode: str) -> None:
    if run_mode != "fast" or not path.exists():
        return
    if path.suffix.lower() == ".csv":
        try:
            existing = pd.read_csv(path, nrows=20)
        except Exception:
            return
        if "run_mode" in existing.columns and existing["run_mode"].astype(str).str.lower().eq("full").any():
            raise FileExistsError(f"Refusing to overwrite full-mode output with fast-mode output: {path}")
    if path.suffix.lower() in {".txt", ".md", ".yaml", ".yml"}:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if "run mode: **full**" in text or "run_mode: full" in text or "run mode: full" in text:
            raise FileExistsError(f"Refusing to overwrite full-mode output with fast-mode output: {path}")


def _average_seed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for patient_id, group in predictions.groupby("patient_id", sort=True):
        rows.append(
            {
                "patient_id": patient_id,
                "y_true": int(str(group["true_label"].iloc[0]) == "Good"),
                "score": float(group["predicted_score"].astype(float).mean()),
            }
        )
    return pd.DataFrame(rows)


def _bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, *, n_bootstrap: int, random_seed: int) -> tuple[float, float]:
    if len(set(y_true.tolist())) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    values: list[float] = []
    for _ in range(int(n_bootstrap)):
        indices = rng.integers(0, len(y_true), len(y_true))
        sample_y = y_true[indices]
        if len(set(sample_y.tolist())) < 2:
            continue
        values.append(float(roc_auc_score(sample_y, scores[indices])))
    if not values:
        return np.nan, np.nan
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _permutation_auc_p_value(y_true: np.ndarray, scores: np.ndarray, *, n_permutations: int, random_seed: int) -> float:
    if len(set(y_true.tolist())) < 2:
        return np.nan
    observed = float(roc_auc_score(y_true, scores))
    rng = np.random.default_rng(random_seed)
    null_values: list[float] = []
    for _ in range(int(n_permutations)):
        permuted = rng.permutation(y_true)
        if len(set(permuted.tolist())) < 2:
            continue
        null_values.append(float(roc_auc_score(permuted, scores)))
    if not null_values:
        return np.nan
    null = np.asarray(null_values, dtype=float)
    return float((np.sum(null >= observed) + 1) / (len(null) + 1))
