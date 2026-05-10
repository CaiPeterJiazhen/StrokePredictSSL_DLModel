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
    metrics = compute_matrixnet_metrics(predictions, inputs.ml_metrics)
    return MatrixNetRunResult(
        predictions=predictions,
        metrics=metrics,
        training_log=pd.DataFrame(logs),
        fold_audit=pd.DataFrame(audits),
    )


def compute_matrixnet_metrics(predictions: pd.DataFrame, ml_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    seed_rows = [_seed_metric_row(str(model_name), int(seed), group, ml_metrics) for (model_name, seed), group in predictions.groupby(["model_name", "seed"], sort=True)]
    seed_frame = pd.DataFrame(seed_rows)
    rows: list[dict[str, object]] = []
    for model_name, group in seed_frame.groupby("model_name", sort=True):
        first = group.iloc[0]
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
                "roc_auc_ci_low": np.nan,
                "roc_auc_ci_high": np.nan,
                "pr_auc": float(group["pr_auc"].mean()),
                "balanced_accuracy": float(group["balanced_accuracy"].mean()),
                "sensitivity": float(group["sensitivity"].mean()),
                "specificity": float(group["specificity"].mean()),
                "f1": float(group["f1"].mean()),
                "brier_score": float(group["brier_score"].mean()),
                "permutation_p_value": np.nan,
                "comparison_to_best_ml_auc": float(group["comparison_to_best_ml_auc"].mean()),
                "comparison_to_fma_only_auc": float(group["comparison_to_fma_only_auc"].mean()),
                "comparison_to_clinical_only_auc": float(group["comparison_to_clinical_only_auc"].mean()),
            }
        )
    return pd.DataFrame(rows)


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
    model = MatrixNet(model_config)
    criterion = nn.BCEWithLogitsLoss(pos_weight=_pos_weight(inputs.labels[train_indices]))
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
        val_loss, _val_scores, _val_true = _eval_epoch(model, val_loader, criterion)
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
    val_loss, val_scores, val_true = _eval_epoch(model, val_loader, criterion)
    threshold, threshold_source = _select_threshold(val_true, val_scores)
    test_loader = DataLoader(MatrixDataset(arrays, inputs.labels, [test_index]), batch_size=1, shuffle=False)
    _test_loss, test_scores, _test_true = _eval_epoch(model, test_loader, criterion)
    score = float(test_scores[0])
    pred_int = int(score >= threshold)
    prediction = {
        "model_name": model_name,
        "outer_fold": int(fold["outer_fold"]),
        "patient_id": test_subject,
        "true_label": INT_TO_LABEL[int(inputs.labels[test_index])],
        "predicted_score": score,
        "predicted_label": INT_TO_LABEL[pred_int],
        "threshold": float(threshold),
        "threshold_source": threshold_source,
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
    for batch in loader:
        labels = batch.pop("label")
        optimizer.zero_grad()
        logits = model(**batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else np.nan


def _eval_epoch(model: MatrixNet, loader: DataLoader, criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    labels_all: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("label")
            logits = model(**batch)
            loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()))
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
    return float(np.mean(losses)) if losses else np.nan, np.asarray(scores, dtype=float), np.asarray(labels_all, dtype=int)


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


def _pos_weight(labels: np.ndarray) -> torch.Tensor | None:
    pos = float(np.sum(labels == 1))
    neg = float(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)


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
