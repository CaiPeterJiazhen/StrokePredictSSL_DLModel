from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_yaml_mapping
from stroke_predict.matrixnet_data import load_matrixnet_inputs
from stroke_predict.matrixnet_training import MatrixNetRunConfig, run_matrixnet_lopo, write_matrixnet_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--fold-limit", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw = load_yaml_mapping(config_path)
    output_dir = _resolve(config_path, str(raw.get("output_dir", "outputs")))
    mode = raw["run_modes"][args.run_mode]
    run_config = MatrixNetRunConfig(
        run_mode=args.run_mode,
        models=[str(value) for value in raw["models"][args.run_mode]],
        seeds=[int(value) for value in mode["seeds"]],
        max_epochs=int(mode["max_epochs"]),
        patience=int(mode["patience"]),
        batch_size=int(mode["batch_size"]),
        learning_rates=[float(value) for value in mode["learning_rates"]],
        weight_decays=[float(value) for value in mode["weight_decays"]],
        dropouts=[float(value) for value in mode["dropouts"]],
        embedding_dims=[int(value) for value in mode["embedding_dims"]],
        hidden_dims=[int(value) for value in mode["hidden_dims"]],
        fold_limit=args.fold_limit,
        write_outputs=True,
    )
    inputs = load_matrixnet_inputs(output_dir)
    result = run_matrixnet_lopo(inputs, run_config)
    paths = write_matrixnet_outputs(output_dir, result, run_config)
    print("MATRIXNET_OK")
    print(f"run_mode={args.run_mode}")
    print(f"n_predictions={len(result.predictions)}")
    print(f"predictions={paths['predictions']}")
    print(f"metrics={paths['metrics']}")
    print(f"report={paths['report']}")
    return 0


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if config_path.parent.name == "configs":
        return (config_path.parent.parent / path).resolve()
    return (config_path.parent / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
