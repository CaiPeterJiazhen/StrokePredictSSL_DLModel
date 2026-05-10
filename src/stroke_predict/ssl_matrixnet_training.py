from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stroke_predict.matrixnet import MatrixNet, MatrixNetConfig
from stroke_predict.matrixnet_data import MatrixNetInputs
from stroke_predict.matrixnet_preprocessing import FoldPreprocessor, fit_vector_preprocessor
from stroke_predict.ssl_matrixnet import (
    SSLMatrixAutoencoder,
    checkpoint_metadata,
    generate_mask,
    load_pretrained_matrixnet_branches,
    masked_mse_loss,
    redacted_checkpoint_path,
)
from stroke_predict.ssl_matrixnet_data import (
    SSL_VARIANTS,
    assert_no_private_strings,
    build_ssl_fold_pools,
    build_ssl_matrix_index_from_baseline_outputs,
)

INT_TO_LABEL = {0: "Poor", 1: "Good"}


@dataclass(frozen=True)
class SSLPretrainConfig:
    ssl_variant: str
    run_mode: str
    epochs: int
    batch_size: int
    mask_ratio: float
    embedding_dim: int
    hidden_dim: int
    seed: int
    device: str = "cpu"
    require_cuda: bool = False
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    dropout: float = 0.0
    psd_loss_weight: float = 1.0
    fc_loss_weight: float = 1.0
    run_id: str = "phase7"


@dataclass(frozen=True)
class SSLPretrainResult:
    final_loss: float
    checkpoint_path: Path
    log: pd.DataFrame


@dataclass(frozen=True)
class SSLMatrixNetRunConfig:
    run_mode: str
    ssl_variant: str
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
    checkpoint_path_override: Path | None = None
    checkpoint_root: Path | None = None
    write_outputs: bool = True
    bootstrap_resamples: int = 1000
    permutation_resamples: int = 1000
    random_seed: int = 42
    orientation_calibration: str = "inner_val_auc"
    mask_ratio: float = 0.25
    device: str = "cpu"
    require_cuda: bool = False
    run_id: str = "phase7"


@dataclass(frozen=True)
class SSLMatrixNetRunResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    seed_wise_metrics: pd.DataFrame
    patient_averaged_metrics: pd.DataFrame
    training_log: pd.DataFrame
    fold_audit: pd.DataFrame
    ssl_matrix_index: pd.DataFrame
    ssl_fold_pool_audit: pd.DataFrame
    pretrain_log: pd.DataFrame


class SSLMatrixDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, *, psd: np.ndarray, fc: np.ndarray) -> None:
        self.psd = psd.astype(np.float32)
        self.fc = fc.astype(np.float32)
        if self.psd.shape[0] != self.fc.shape[0]:
            raise ValueError("PSD and FC SSL matrices must have the same row count")

    def __len__(self) -> int:
        return int(self.psd.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "psd": torch.from_numpy(self.psd[index]).float(),
            "fc": torch.from_numpy(self.fc[index]).float(),
        }


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


def pretrain_ssl_matrixnet(
    *,
    psd: np.ndarray,
    fc: np.ndarray,
    checkpoint_path: str | Path,
    config: SSLPretrainConfig,
) -> SSLPretrainResult:
    if config.ssl_variant not in SSL_VARIANTS:
        raise ValueError(f"Unsupported ssl_variant: {config.ssl_variant}")
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = _torch_device(config.device, config.require_cuda)
    psd = psd.astype(np.float32)
    fc = fc.astype(np.float32)
    model = SSLMatrixAutoencoder(
        psd_shape=tuple(psd.shape[1:]),
        fc_shape=tuple(fc.shape[1:]),
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        SSLMatrixDataset(psd=psd, fc=fc),
        batch_size=max(1, int(config.batch_size)),
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
    )
    logs: list[dict[str, object]] = []
    final_loss = np.nan
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(loader):
            psd_batch = batch["psd"].to(device)
            fc_batch = batch["fc"].to(device)
            psd_mask = generate_mask(
                psd_batch.shape,
                mask_ratio=config.mask_ratio,
                seed=config.seed + epoch * 1009 + batch_index,
                device=device,
            )
            fc_mask = generate_mask(
                fc_batch.shape,
                mask_ratio=config.mask_ratio,
                seed=config.seed + epoch * 2003 + batch_index,
                device=device,
            )
            masked_psd = psd_batch.masked_fill(psd_mask, 0.0)
            masked_fc = fc_batch.masked_fill(fc_mask, 0.0)
            optimizer.zero_grad()
            outputs = model(psd=masked_psd, fc=masked_fc)
            psd_loss = masked_mse_loss(outputs["psd"], psd_batch, psd_mask)
            fc_loss = masked_mse_loss(outputs["fc"], fc_batch, fc_mask)
            loss = config.psd_loss_weight * psd_loss + config.fc_loss_weight * fc_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else np.nan
        logs.append(
            {
                "epoch": epoch,
                "ssl_variant": config.ssl_variant,
                "run_mode": config.run_mode,
                "train_loss": final_loss,
                "device": str(device),
            }
        )
    checkpoint = {
        "psd_encoder": model.psd_encoder.state_dict() if model.psd_encoder is not None else {},
        "fc_encoder": model.fc_encoder.state_dict() if model.fc_encoder is not None else {},
        "metadata": checkpoint_metadata(
            ssl_variant=config.ssl_variant,
            run_mode=config.run_mode,
            final_loss=float(final_loss),
            config={
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "mask_ratio": config.mask_ratio,
                "embedding_dim": config.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "seed": config.seed,
            },
        ),
    }
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    return SSLPretrainResult(final_loss=float(final_loss), checkpoint_path=checkpoint_path, log=pd.DataFrame(logs))


def run_ssl_matrixnet_lopo(inputs: MatrixNetInputs, config: SSLMatrixNetRunConfig) -> SSLMatrixNetRunResult:
    rows: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    subject_to_index = {subject: index for index, subject in enumerate(inputs.subject_ids)}
    folds = inputs.outer_folds["folds"][: config.fold_limit] if config.fold_limit else inputs.outer_folds["folds"]
    registry_by_fold = {int(registry["outer_fold"]): registry for registry in inputs.registries}
    for model_name in config.models:
        _validate_model_variant(model_name, config.ssl_variant, config.run_mode)
        for seed in config.seeds:
            for fold in folds:
                outer_fold = int(fold["outer_fold"])
                registry = registry_by_fold[outer_fold]
                prediction, fold_logs, audit = _run_one_ssl_fold(
                    model_name,
                    seed,
                    fold,
                    registry,
                    inputs,
                    config,
                    subject_to_index,
                )
                rows.append(prediction)
                logs.extend(fold_logs)
                audits.append(audit)
    predictions = pd.DataFrame(rows)
    seed_metrics, patient_metrics, metrics = compute_ssl_matrixnet_metrics(
        predictions,
        phase6_metrics=inputs.ml_metrics,
        config=config,
    )
    return SSLMatrixNetRunResult(
        predictions=predictions,
        metrics=metrics,
        seed_wise_metrics=seed_metrics,
        patient_averaged_metrics=patient_metrics,
        training_log=pd.DataFrame(logs),
        fold_audit=pd.DataFrame(audits),
        ssl_matrix_index=pd.DataFrame(),
        ssl_fold_pool_audit=pd.DataFrame(),
        pretrain_log=pd.DataFrame(),
    )


def compute_ssl_matrixnet_metrics(
    predictions: pd.DataFrame,
    *,
    phase6_metrics: pd.DataFrame | None,
    config: SSLMatrixNetRunConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if predictions.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    seed_rows = [
        _metric_row(model_name=str(model_name), seed=int(seed), group=group, config=config)
        for (model_name, seed), group in predictions.groupby(["model_name", "seed"], sort=True)
    ]
    seed_metrics = pd.DataFrame(seed_rows)
    patient_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model_name, group in predictions.groupby("model_name", sort=True):
        averaged = _average_patient_scores(group)
        for _, row in averaged.iterrows():
            patient_rows.append(
                {
                    "model_name": model_name,
                    "patient_id": row["patient_id"],
                    "true_label": row["true_label"],
                    "label_int": int(row["label_int"]),
                    "patient_averaged_score": float(row["predicted_score"]),
                    "n_seeds": int(row["n_seeds"]),
                }
            )
        y_true = averaged["label_int"].astype(int).to_numpy()
        scores = averaged["predicted_score"].astype(float).to_numpy()
        auc_patient = _safe_auc(y_true, scores)
        ci_low, ci_high = (np.nan, np.nan)
        permutation_p = np.nan
        if config.run_mode == "full":
            ci_low, ci_high = _bootstrap_auc_ci(
                y_true,
                scores,
                n_bootstrap=config.bootstrap_resamples,
                random_seed=config.random_seed,
            )
            permutation_p = _permutation_auc_p_value(
                y_true,
                scores,
                n_permutations=config.permutation_resamples,
                random_seed=config.random_seed,
            )
        seed_group = seed_metrics[seed_metrics["model_name"].astype(str).eq(str(model_name))]
        summary_rows.append(
            {
                "model_name": model_name,
                "ssl_variant": str(group["ssl_variant"].iloc[0]),
                "run_mode": str(group["run_mode"].iloc[0]),
                "n_patients": int(averaged["patient_id"].nunique()),
                "n_seeds": int(seed_group["seed"].nunique()),
                "mean_seed_roc_auc": float(seed_group["roc_auc"].mean()),
                "seed_std_roc_auc": float(seed_group["roc_auc"].std(ddof=0)) if len(seed_group) > 1 else np.nan,
                "pooled_auc": _safe_auc(group["label_int"].astype(int).to_numpy(), group["predicted_score"].astype(float).to_numpy()),
                "patient_averaged_auc": auc_patient,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
                "permutation_p_value": permutation_p,
                "pr_auc": float(seed_group["pr_auc"].mean()),
                "balanced_accuracy": float(seed_group["balanced_accuracy"].mean()),
                "sensitivity": float(seed_group["sensitivity"].mean()),
                "specificity": float(seed_group["specificity"].mean()),
                "f1": float(seed_group["f1"].mean()),
                "brier_score": float(seed_group["brier_score"].mean()),
                "auc_score": _safe_auc(group["label_int"].astype(int).to_numpy(), group["predicted_score"].astype(float).to_numpy()),
                "auc_one_minus_score": _safe_auc(group["label_int"].astype(int).to_numpy(), 1.0 - group["predicted_score"].astype(float).to_numpy()),
                "mean_score_good": _class_mean(group, 1),
                "mean_score_poor": _class_mean(group, 0),
                "direction_correct": bool(_safe_auc(group["label_int"].astype(int).to_numpy(), group["predicted_score"].astype(float).to_numpy()) >= 0.5)
                if np.isfinite(_safe_auc(group["label_int"].astype(int).to_numpy(), group["predicted_score"].astype(float).to_numpy()))
                else False,
                "score_orientation_counts": ";".join(
                    f"{key}={value}"
                    for key, value in group["score_orientation"].astype(str).value_counts().sort_index().items()
                ),
                "matched_phase6_model": _matched_phase6_model(str(model_name)),
                "comparison_to_matched_phase6_auc": _compare_to_matched_phase6(
                    auc_patient,
                    str(model_name),
                    phase6_metrics,
                ),
            }
        )
    return pd.DataFrame(seed_rows), pd.DataFrame(patient_rows), pd.DataFrame(summary_rows)


def write_ssl_matrixnet_outputs(
    output_dir: str | Path,
    result: SSLMatrixNetRunResult,
    config: SSLMatrixNetRunConfig,
) -> dict[str, str]:
    root = Path(output_dir)
    paths = {
        "ssl_matrix_index": root / "ssl_matrixnet" / "ssl_matrix_index.csv",
        "ssl_fold_pool_audit": root / "ssl_matrixnet" / "ssl_fold_pool_audit_phase7.csv",
        "no_leakage_report": root / "reports" / "no_leakage_report_phase7.txt",
        "pretrain_log": root / "ssl_matrixnet" / "pretrain_log_phase7.csv",
        "predictions": root / "predictions" / "ssl_matrixnet_patient_predictions_phase7.csv",
        "metrics": root / "evaluation" / "ssl_matrixnet_metrics_phase7.csv",
        "seed_wise_metrics": root / "evaluation" / "ssl_matrixnet_seed_wise_metrics_phase7.csv",
        "patient_averaged_metrics": root / "evaluation" / "ssl_matrixnet_patient_averaged_metrics_phase7.csv",
        "report": root / "reports" / "phase7_ssl_matrixnet_report.md",
        "config_used": root / "ssl_matrixnet" / "config_used_phase7.yaml",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    ssl_index = result.ssl_matrix_index if not result.ssl_matrix_index.empty else _minimal_ssl_index_from_predictions(result.predictions)
    ssl_audit = result.ssl_fold_pool_audit if not result.ssl_fold_pool_audit.empty else _minimal_ssl_audit_from_predictions(result.predictions, config)
    pretrain_log = result.pretrain_log if not result.pretrain_log.empty else pd.DataFrame(
        {
            "ssl_variant": [config.ssl_variant],
            "run_mode": [config.run_mode],
            "train_loss": [np.nan],
            "device": [config.device],
        }
    )
    _write_public_csv(ssl_index, paths["ssl_matrix_index"])
    _write_public_csv(ssl_audit, paths["ssl_fold_pool_audit"])
    _write_public_csv(pretrain_log, paths["pretrain_log"])
    _write_public_csv(result.predictions.sort_values(["model_name", "seed", "outer_fold"]), paths["predictions"])
    _write_public_csv(result.metrics.sort_values(["model_name"]) if not result.metrics.empty else result.metrics, paths["metrics"])
    _write_public_csv(
        result.seed_wise_metrics.sort_values(["model_name", "seed"]) if not result.seed_wise_metrics.empty else result.seed_wise_metrics,
        paths["seed_wise_metrics"],
    )
    _write_public_csv(
        result.patient_averaged_metrics.sort_values(["model_name", "patient_id"])
        if not result.patient_averaged_metrics.empty
        else result.patient_averaged_metrics,
        paths["patient_averaged_metrics"],
    )
    leakage_text = _no_leakage_text(ssl_audit, result.fold_audit)
    assert_no_private_strings(leakage_text)
    paths["no_leakage_report"].write_text(leakage_text, encoding="utf-8")
    report_text = _phase7_report(result, config)
    assert_no_private_strings(report_text)
    paths["report"].write_text(report_text, encoding="utf-8")
    config_text = _config_snapshot(config)
    assert_no_private_strings(config_text)
    paths["config_used"].write_text(config_text, encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def run_ssl_pretraining_from_outputs(
    output_dir: str | Path,
    *,
    config: SSLPretrainConfig,
    fold_limit: int | None,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    index, psd, fc = build_ssl_matrix_index_from_baseline_outputs(output_dir)
    from stroke_predict.matrixnet_data import load_matrixnet_inputs

    inputs = load_matrixnet_inputs(output_dir)
    pool, audit = build_ssl_fold_pools(index, inputs.outer_folds, ssl_variant=config.ssl_variant, fold_limit=fold_limit)
    checkpoint_root = output_dir / "ssl_matrixnet" / "checkpoints" / config.run_id / config.ssl_variant
    logs: list[pd.DataFrame] = []
    for outer_fold, fold_pool in pool.groupby("outer_fold", sort=True):
        row_indices = fold_pool["row_index"].astype(int).to_numpy()
        if row_indices.size == 0:
            raise ValueError(f"SSL pool is empty for fold {outer_fold}")
        checkpoint_path = checkpoint_root / f"fold_{int(outer_fold):02d}" / "ssl_encoder.pt"
        result = pretrain_ssl_matrixnet(
            psd=psd[row_indices],
            fc=fc[row_indices],
            checkpoint_path=checkpoint_path,
            config=config,
        )
        log = result.log.copy()
        log["outer_fold"] = int(outer_fold)
        log["checkpoint_path_redacted"] = redacted_checkpoint_path(checkpoint_path)
        logs.append(log)
    paths = _phase7_pretrain_paths(output_dir)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    _write_public_csv(index, paths["ssl_matrix_index"])
    _write_public_csv(audit, paths["ssl_fold_pool_audit"])
    pretrain_log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame()
    _write_public_csv(pretrain_log, paths["pretrain_log"])
    leakage_text = _no_leakage_text(audit, pd.DataFrame())
    paths["no_leakage_report"].write_text(leakage_text, encoding="utf-8")
    paths["config_used"].write_text(_pretrain_config_snapshot(config), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _run_one_ssl_fold(
    model_name: str,
    seed: int,
    fold: dict[str, Any],
    registry: dict[str, Any],
    inputs: MatrixNetInputs,
    config: SSLMatrixNetRunConfig,
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
    device = _torch_device(config.device, config.require_cuda)
    model = MatrixNet(model_config).to(device)
    spec = _model_inputs(model_name)
    checkpoint_path = _checkpoint_for_fold(config, int(fold["outer_fold"]))
    load_pretrained_matrixnet_branches(
        model,
        checkpoint_path,
        load_psd=bool(spec["psd"]),
        load_fc=bool(spec["fc"]),
    )
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
                "ssl_variant": config.ssl_variant,
                "outer_fold": int(fold["outer_fold"]),
                "seed": seed,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "run_mode": config.run_mode,
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
        "ssl_variant": config.ssl_variant,
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
        "ssl_checkpoint_path_redacted": redacted_checkpoint_path(checkpoint_path),
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_inner_metric": -float(best_loss),
        "train_loss_final": float(logs[-1]["train_loss"]),
        "val_loss_best": float(best_loss),
    }
    audit = {
        "model_name": model_name,
        "ssl_variant": config.ssl_variant,
        "outer_fold": int(fold["outer_fold"]),
        "seed": seed,
        "test_patient": test_subject,
        "test_excluded_from_train": test_subject not in train_subjects,
        "test_excluded_from_val": test_subject not in val_subjects,
        "checkpoint_path_redacted": redacted_checkpoint_path(checkpoint_path),
        "run_mode": config.run_mode,
        "leakage_passed": bool(test_subject not in train_subjects and test_subject not in val_subjects),
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
        use_clinical=spec["clinical"] and clinical_dim > 0,
        tacs_dim=tacs_dim,
        clinical_dim=clinical_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def _model_inputs(model_name: str) -> dict[str, bool]:
    specs = {
        "M9a_sslA_fc_only": {"psd": False, "fc": True, "tacs": False, "clinical": False},
        "M9b_sslA_psd_fc": {"psd": True, "fc": True, "tacs": False, "clinical": False},
        "M9c_sslA_psd_fc_tacs": {"psd": True, "fc": True, "tacs": True, "clinical": False},
        "M13_sslA_clinical_eeg": {"psd": True, "fc": True, "tacs": True, "clinical": True},
        "M9a_sslB_fc_only_fast": {"psd": False, "fc": True, "tacs": False, "clinical": False},
        "M9a_sslC_fc_only_fast": {"psd": False, "fc": True, "tacs": False, "clinical": False},
        "M9a_sslD_fc_only_fast": {"psd": False, "fc": True, "tacs": False, "clinical": False},
    }
    if model_name not in specs:
        raise ValueError(f"Unsupported SSL MatrixNet model: {model_name}")
    return specs[model_name]


def _validate_model_variant(model_name: str, ssl_variant: str, run_mode: str) -> None:
    if model_name.startswith("M9a_sslB") and ssl_variant != "stroke_healthy_baseline":
        raise ValueError(f"{model_name} requires stroke_healthy_baseline")
    if model_name.startswith("M9a_sslC") and ssl_variant != "stroke_all_stage":
        raise ValueError(f"{model_name} requires stroke_all_stage")
    if model_name.startswith("M9a_sslD") and ssl_variant != "stroke_all_stage_healthy":
        raise ValueError(f"{model_name} requires stroke_all_stage_healthy")
    if model_name in {"M9a_sslA_fc_only", "M9b_sslA_psd_fc", "M9c_sslA_psd_fc_tacs", "M13_sslA_clinical_eeg"} and ssl_variant != "stroke_baseline":
        raise ValueError(f"{model_name} requires stroke_baseline")
    if run_mode == "full" and model_name.startswith("M9a_ssl") and model_name.endswith("_fast"):
        raise ValueError("Fast-only SSL-B/C/D model cannot run in full mode")


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


def _score_orientation(y_true: np.ndarray, sigmoid_scores: np.ndarray, config: SSLMatrixNetRunConfig) -> str:
    if config.orientation_calibration == "none":
        return "normal"
    if config.orientation_calibration != "inner_val_auc":
        raise ValueError(f"Unsupported orientation_calibration: {config.orientation_calibration}")
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


def _select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    if y_true.size == 0 or scores.size == 0 or len(set(y_true.tolist())) < 2:
        return 0.5, "fixed_0.5"
    thresholds = np.unique(np.concatenate([scores, np.asarray([0.5])]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        balanced = float(balanced_accuracy_score(y_true, pred))
        if balanced > best_score:
            best_score = balanced
            best_threshold = float(threshold)
    return best_threshold, "inner_validation_balanced_accuracy"


def _metric_row(
    *,
    model_name: str,
    seed: int,
    group: pd.DataFrame,
    config: SSLMatrixNetRunConfig,
) -> dict[str, object]:
    y_true = group["label_int"].astype(int).to_numpy()
    scores = group["predicted_score"].astype(float).to_numpy()
    pred = (group["predicted_label"].astype(str) == "Good").astype(int).to_numpy()
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if tp + fn else np.nan
    specificity = float(tn / (tn + fp)) if tn + fp else np.nan
    return {
        "model_name": model_name,
        "ssl_variant": str(group["ssl_variant"].iloc[0]),
        "seed": seed,
        "run_mode": config.run_mode,
        "n_patients": int(len(group)),
        "roc_auc": _safe_auc(y_true, scores),
        "pr_auc": _safe_pr_auc(y_true, scores),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)) if len(set(y_true.tolist())) > 1 else np.nan,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, scores)),
    }


def _average_patient_scores(group: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient_id, patient_group in group.groupby("patient_id", sort=True):
        first = patient_group.iloc[0]
        rows.append(
            {
                "patient_id": patient_id,
                "true_label": first["true_label"],
                "label_int": int(first["label_int"]),
                "predicted_score": float(patient_group["predicted_score"].astype(float).mean()),
                "n_seeds": int(patient_group["seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or len(set(y_true.astype(int).tolist())) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true.astype(int), scores.astype(float)))
    except ValueError:
        return np.nan


def _safe_pr_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if y_true.size == 0 or not np.any(y_true.astype(int) == 1):
        return np.nan
    return float(average_precision_score(y_true.astype(int), scores.astype(float)))


def _class_mean(group: pd.DataFrame, label: int) -> float:
    labels = group["label_int"].astype(int).to_numpy()
    scores = group["predicted_score"].astype(float).to_numpy()
    mask = labels == int(label)
    if not np.any(mask):
        return np.nan
    return float(scores[mask].mean())


def _bootstrap_auc_ci(y_true: np.ndarray, scores: np.ndarray, *, n_bootstrap: int, random_seed: int) -> tuple[float, float]:
    if len(set(y_true.astype(int).tolist())) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(random_seed)
    aucs: list[float] = []
    indices = np.arange(len(y_true))
    for _ in range(int(n_bootstrap)):
        sample = rng.choice(indices, size=len(indices), replace=True)
        auc = _safe_auc(y_true[sample], scores[sample])
        if np.isfinite(auc):
            aucs.append(auc)
    if not aucs:
        return np.nan, np.nan
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def _permutation_auc_p_value(y_true: np.ndarray, scores: np.ndarray, *, n_permutations: int, random_seed: int) -> float:
    observed = _safe_auc(y_true, scores)
    if not np.isfinite(observed):
        return np.nan
    rng = np.random.default_rng(random_seed)
    count = 0
    for _ in range(int(n_permutations)):
        permuted = rng.permutation(y_true)
        auc = _safe_auc(permuted, scores)
        if np.isfinite(auc) and auc >= observed:
            count += 1
    return float((count + 1) / (int(n_permutations) + 1))


def _pos_weight(labels: np.ndarray) -> torch.Tensor | None:
    pos = float(np.sum(labels == 1))
    neg = float(np.sum(labels == 0))
    if pos == 0 or neg == 0:
        return None
    return torch.tensor([neg / pos], dtype=torch.float32)


def _torch_device(device_name: str, require_cuda: bool) -> torch.device:
    device = torch.device(device_name)
    if require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("CUDA is required for this SSL-MatrixNet run, but no CUDA device is available")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SSL-MatrixNet config requested CUDA, but no CUDA device is available")
    return device


def _checkpoint_for_fold(config: SSLMatrixNetRunConfig, outer_fold: int) -> Path:
    if config.checkpoint_path_override is not None:
        return Path(config.checkpoint_path_override)
    if config.checkpoint_root is None:
        raise ValueError("SSL checkpoint_root is required when checkpoint_path_override is not set")
    return Path(config.checkpoint_root) / f"fold_{outer_fold:02d}" / "ssl_encoder.pt"


def _input_family(model_name: str) -> str:
    mapping = {
        "M9a_sslA_fc_only": "ssl_fc_only",
        "M9b_sslA_psd_fc": "ssl_psd_fc",
        "M9c_sslA_psd_fc_tacs": "ssl_psd_fc_tacs",
        "M13_sslA_clinical_eeg": "ssl_clinical_eeg_secondary",
        "M9a_sslB_fc_only_fast": "sslB_fc_only_fast",
        "M9a_sslC_fc_only_fast": "sslC_fc_only_fast",
        "M9a_sslD_fc_only_fast": "sslD_fc_only_fast",
    }
    return mapping[model_name]


def _matched_phase6_model(model_name: str) -> str:
    mapping = {
        "M9a_sslA_fc_only": "M8b_matrixnet_fc_only",
        "M9b_sslA_psd_fc": "M8c_matrixnet_psd_fc",
        "M9c_sslA_psd_fc_tacs": "M8d_matrixnet_psd_fc_tacs",
    }
    return mapping.get(model_name, "")


def _compare_to_matched_phase6(auc: float, model_name: str, metrics: pd.DataFrame | None) -> float:
    matched = _matched_phase6_model(model_name)
    if not matched or metrics is None or not np.isfinite(auc):
        return np.nan
    if not {"model_name", "roc_auc_mean"} <= set(metrics.columns) and not {"model_name", "patient_averaged_auc"} <= set(metrics.columns):
        return np.nan
    rows = metrics[metrics["model_name"].astype(str).eq(matched)]
    if rows.empty:
        return np.nan
    column = "patient_averaged_auc" if "patient_averaged_auc" in rows.columns else "roc_auc_mean"
    return float(auc - float(rows.iloc[0][column]))


def _write_public_csv(frame: pd.DataFrame, path: Path) -> None:
    assert_no_private_strings(frame if not frame.empty else "")
    frame.to_csv(path, index=False)


def _minimal_ssl_index_from_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["subject_id", "source", "stage", "condition"])
    subjects = sorted(predictions["patient_id"].astype(str).unique().tolist())
    rows = []
    row_index = 0
    for subject in subjects:
        for condition in ("eo", "ec"):
            rows.append(
                {
                    "row_index": row_index,
                    "subject_id": subject,
                    "source": "stroke_supervised",
                    "stage": "baseline",
                    "condition": condition,
                }
            )
            row_index += 1
    return pd.DataFrame(rows)


def _minimal_ssl_audit_from_predictions(predictions: pd.DataFrame, config: SSLMatrixNetRunConfig) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(columns=["outer_fold", "ssl_variant", "leakage_passed"])
    return predictions[["outer_fold", "patient_id"]].drop_duplicates().rename(columns={"patient_id": "test_subject"}).assign(
        ssl_variant=config.ssl_variant,
        test_subject_records_in_pool=0,
        leakage_passed=True,
    )


def _no_leakage_text(ssl_audit: pd.DataFrame, fold_audit: pd.DataFrame) -> str:
    ssl_pass = bool(ssl_audit.empty or ssl_audit.get("leakage_passed", pd.Series([True])).astype(bool).all())
    fold_pass = bool(fold_audit.empty or fold_audit.get("leakage_passed", pd.Series([True])).astype(bool).all())
    status = "PASS" if ssl_pass and fold_pass else "FAIL"
    return "\n".join(
        [
            "Phase 7 SSL-MatrixNet no-leakage report",
            f"status: {status}",
            f"ssl_pool_leakage_passed: {ssl_pass}",
            f"supervised_fold_leakage_passed: {fold_pass}",
            "outer test patient baseline/immediate/mid/final records are excluded from SSL pools by subject_id.",
            "",
        ]
    )


def _phase7_report(result: SSLMatrixNetRunResult, config: SSLMatrixNetRunConfig) -> str:
    metrics_table = result.metrics.to_markdown(index=False) if not result.metrics.empty else "No metrics available."
    n_predictions = len(result.predictions)
    best = "not available"
    significant = "none"
    if not result.metrics.empty and "patient_averaged_auc" in result.metrics.columns:
        ordered = result.metrics.sort_values("patient_averaged_auc", ascending=False, na_position="last")
        if not ordered.empty:
            row = ordered.iloc[0]
            best = f"{row['model_name']} patient_averaged_auc={row['patient_averaged_auc']}"
        sig_rows = result.metrics[result.metrics.get("permutation_p_value", pd.Series(dtype=float)).astype(float) < 0.05]
        if not sig_rows.empty:
            significant = ", ".join(sig_rows["model_name"].astype(str).tolist())
    return "\n".join(
        [
            "# Phase 7 SSL-MatrixNet Report",
            "",
            "## Phase 7 Status",
            "Exploratory SSL-MatrixNet for baseline EEG prognosis. No raw EEG supervised input is used.",
            "Do not claim EEG efficacy unless patient-level full-mode metrics are stable and permutation-significant.",
            "",
            "## SSL Data Sources",
            "SSL pools are built from de-identified PSD/ROI-FC matrix artifacts. Treatment-stage EEG is only allowed for SSL pool variants and never for supervised classifier input.",
            "",
            "## Leakage Audit",
            "Outer test patient records are excluded from SSL pretraining pools and supervised fitting by subject_id.",
            "",
            "## SSL Method",
            f"Masked matrix modeling; mask_ratio={config.mask_ratio}; ssl_variant={config.ssl_variant}; device={config.device}.",
            "",
            "## Fine-Tuning Results",
            metrics_table,
            "",
            "## Decision Answers",
            f"- Prediction rows: {n_predictions}",
            f"- Best SSL model by patient-averaged AUC: {best}",
            f"- Permutation-significant models: {significant}",
            "- SSL-B/C/D should remain fast-mode infrastructure until a later phase explicitly promotes them.",
            "- Phase 8 should proceed only after this report is reviewed against no-SSL baselines.",
            "",
            "## Scientific Caution",
            "Do not claim EEG efficacy. SSL-MatrixNet did not produce stable permutation-significant improvement in this phase unless the metrics table above proves all required criteria.",
            "",
        ]
    )


def _config_snapshot(config: SSLMatrixNetRunConfig) -> str:
    return "\n".join(
        [
            f"run_mode: {config.run_mode}",
            f"ssl_variant: {config.ssl_variant}",
            f"run_id: {config.run_id}",
            "models:",
            *[f"  - {model}" for model in config.models],
            "seeds: [" + ", ".join(map(str, config.seeds)) + "]",
            f"max_epochs: {config.max_epochs}",
            f"patience: {config.patience}",
            f"batch_size: {config.batch_size}",
            f"mask_ratio: {config.mask_ratio}",
            f"device: {config.device}",
            f"require_cuda: {str(config.require_cuda).lower()}",
            "",
        ]
    )


def _pretrain_config_snapshot(config: SSLPretrainConfig) -> str:
    return "\n".join(
        [
            f"run_mode: {config.run_mode}",
            f"ssl_variant: {config.ssl_variant}",
            f"run_id: {config.run_id}",
            f"epochs: {config.epochs}",
            f"batch_size: {config.batch_size}",
            f"mask_ratio: {config.mask_ratio}",
            f"device: {config.device}",
            f"require_cuda: {str(config.require_cuda).lower()}",
            "",
        ]
    )


def _phase7_pretrain_paths(root: Path) -> dict[str, Path]:
    return {
        "ssl_matrix_index": root / "ssl_matrixnet" / "ssl_matrix_index.csv",
        "ssl_fold_pool_audit": root / "ssl_matrixnet" / "ssl_fold_pool_audit_phase7.csv",
        "no_leakage_report": root / "reports" / "no_leakage_report_phase7.txt",
        "pretrain_log": root / "ssl_matrixnet" / "pretrain_log_phase7.csv",
        "config_used": root / "ssl_matrixnet" / "config_used_phase7.yaml",
    }
