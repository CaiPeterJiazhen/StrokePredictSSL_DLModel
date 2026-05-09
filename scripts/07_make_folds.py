from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_project_config, load_yaml_mapping
from stroke_predict.splits import build_outer_folds, write_fold_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cv_config_path = Path(args.config).resolve()
    cv_config = load_yaml_mapping(cv_config_path)
    project_config_path = _resolve_project_config(cv_config_path, str(cv_config.get("project_config", "project.yaml")))
    project = load_project_config(project_config_path)
    inner_k = int(cv_config.get("inner_k", 3))

    cohort = pd.read_csv(project.output_dir / "cohort" / "cohort_master.csv")
    qc = pd.read_csv(project.output_dir / "qc" / "eeg_qc_summary.csv")
    features = pd.read_csv(project.output_dir / "features" / "handcrafted_features.csv", usecols=["subject_id"])

    result = build_outer_folds(cohort, qc, features, inner_k=inner_k)
    output_dir = project.output_dir / "folds"
    write_fold_outputs(result, output_dir)

    print("FOLDS_OK")
    print(f"n_outer_folds={result['n_supervised_main']}")
    print(f"output_dir={output_dir}")
    return 0


def _resolve_project_config(cv_config_path: Path, project_config: str) -> Path:
    path = Path(project_config)
    if path.is_absolute():
        return path
    return (cv_config_path.parent / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
