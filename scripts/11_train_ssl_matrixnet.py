from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_yaml_mapping
from stroke_predict.matrixnet_data import load_matrixnet_inputs
from stroke_predict.ssl_matrixnet_data import build_ssl_fold_pools, build_ssl_matrix_index_from_baseline_outputs
from stroke_predict.ssl_matrixnet_training import (
    SSLMatrixNetRunConfig,
    SSLMatrixNetRunResult,
    run_ssl_matrixnet_lopo,
    write_ssl_matrixnet_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--ssl-variant", required=True)
    parser.add_argument("--fold-limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw = load_yaml_mapping(config_path)
    output_dir = _resolve(config_path, str(raw.get("output_dir", "outputs")))
    mode = raw["run_modes"][args.run_mode]
    device = str(args.device or mode.get("device", "cpu"))
    require_cuda = bool(args.require_cuda or mode.get("require_cuda", False))
    _guard_cuda(device, require_cuda)
    models = _models_for_variant(raw["models"][args.run_mode], args.ssl_variant)
    run_id = str(raw.get("run_id", "phase7"))
    run_config = SSLMatrixNetRunConfig(
        run_mode=args.run_mode,
        ssl_variant=args.ssl_variant,
        models=models,
        seeds=[int(value) for value in mode["seeds"]],
        max_epochs=int(mode["finetune_epochs"]),
        patience=int(mode["patience"]),
        batch_size=int(mode["batch_size"]),
        learning_rates=[float(value) for value in mode["learning_rates"]],
        weight_decays=[float(value) for value in mode["weight_decays"]],
        dropouts=[float(value) for value in mode["dropouts"]],
        embedding_dims=[int(value) for value in mode["embedding_dims"]],
        hidden_dims=[int(value) for value in mode["hidden_dims"]],
        fold_limit=args.fold_limit,
        checkpoint_root=output_dir / "ssl_matrixnet" / "checkpoints" / run_id / args.ssl_variant,
        bootstrap_resamples=int(mode.get("bootstrap_resamples", 1000)),
        permutation_resamples=int(mode.get("permutation_resamples", 1000)),
        random_seed=int(mode.get("random_seed", 42)),
        orientation_calibration=str(mode.get("orientation_calibration", "inner_val_auc")),
        mask_ratio=float(mode.get("mask_ratio", 0.25)),
        device=device,
        require_cuda=require_cuda,
        run_id=run_id,
    )
    inputs = load_matrixnet_inputs(output_dir)
    result = run_ssl_matrixnet_lopo(inputs, run_config)
    ssl_index, _psd, _fc = build_ssl_matrix_index_from_baseline_outputs(output_dir)
    _pool, ssl_audit = build_ssl_fold_pools(ssl_index, inputs.outer_folds, ssl_variant=args.ssl_variant, fold_limit=args.fold_limit)
    pretrain_log_path = output_dir / "ssl_matrixnet" / "pretrain_log_phase7.csv"
    pretrain_log = pd.read_csv(pretrain_log_path) if pretrain_log_path.exists() else pd.DataFrame()
    enriched = SSLMatrixNetRunResult(
        predictions=result.predictions,
        metrics=result.metrics,
        seed_wise_metrics=result.seed_wise_metrics,
        patient_averaged_metrics=result.patient_averaged_metrics,
        training_log=result.training_log,
        fold_audit=result.fold_audit,
        ssl_matrix_index=ssl_index,
        ssl_fold_pool_audit=ssl_audit,
        pretrain_log=pretrain_log,
    )
    paths = write_ssl_matrixnet_outputs(output_dir, enriched, run_config)
    print("SSL_MATRIXNET_TRAIN_OK")
    print(f"run_mode={args.run_mode}")
    print(f"ssl_variant={args.ssl_variant}")
    print(f"device={device}")
    if torch.cuda.is_available() and device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"n_predictions={len(result.predictions)}")
    print(f"predictions={paths['predictions']}")
    print(f"metrics={paths['metrics']}")
    print(f"report={paths['report']}")
    return 0


def _models_for_variant(raw_models: object, ssl_variant: str) -> list[str]:
    if isinstance(raw_models, dict):
        values = raw_models[ssl_variant]
    else:
        values = raw_models
    return [str(value) for value in values]


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if config_path.parent.name == "configs":
        return (config_path.parent.parent / path).resolve()
    return (config_path.parent / path).resolve()


def _guard_cuda(device: str, require_cuda: bool) -> None:
    if require_cuda and (device != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("CUDA is required for this SSL-MatrixNet run, but no CUDA device is available")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SSL-MatrixNet config requested CUDA, but no CUDA device is available")


if __name__ == "__main__":
    raise SystemExit(main())
