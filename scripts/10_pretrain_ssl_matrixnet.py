from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_yaml_mapping
from stroke_predict.ssl_matrixnet_training import SSLPretrainConfig, run_ssl_pretraining_from_outputs


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
    config = SSLPretrainConfig(
        ssl_variant=args.ssl_variant,
        run_mode=args.run_mode,
        epochs=int(mode["ssl_epochs"]),
        batch_size=int(mode["batch_size"]),
        mask_ratio=float(mode.get("mask_ratio", 0.25)),
        embedding_dim=int(mode["embedding_dims"][0]),
        hidden_dim=int(mode["hidden_dims"][0]),
        seed=int(mode.get("seeds", [0])[0]),
        device=device,
        require_cuda=require_cuda,
        learning_rate=float(mode.get("learning_rates", [0.001])[0]),
        weight_decay=float(mode.get("weight_decays", [0.01])[0]),
        dropout=float(mode.get("dropouts", [0.0])[0]),
        run_id=str(raw.get("run_id", "phase7")),
    )
    paths = run_ssl_pretraining_from_outputs(output_dir, config=config, fold_limit=args.fold_limit)
    print("SSL_MATRIXNET_PRETRAIN_OK")
    print(f"run_mode={args.run_mode}")
    print(f"ssl_variant={args.ssl_variant}")
    print(f"device={device}")
    if torch.cuda.is_available() and device == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"pretrain_log={paths['pretrain_log']}")
    print(f"fold_pool_audit={paths['ssl_fold_pool_audit']}")
    return 0


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
