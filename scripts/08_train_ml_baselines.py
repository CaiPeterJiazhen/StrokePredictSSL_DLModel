from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_project_config, load_yaml_mapping
from stroke_predict.ml_models import (
    flatten_fc_matrices,
    flatten_psd_matrices,
    load_fold_registries,
    run_classical_ml_baselines,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_mapping(config_path)
    project_config_path = _resolve(config_path, str(config.get("project_config", "project.yaml")))
    project = load_project_config(project_config_path)
    _resolve_output_paths(config, project.project_root)

    cohort = pd.read_csv(project.output_dir / "cohort" / "cohort_master.csv")
    supervised = cohort.loc[cohort["role"].eq("supervised_main")].sort_values("subject_id").reset_index(drop=True)
    subjects = supervised["subject_id"].astype(str).tolist()
    psd_summary = pd.read_csv(project.output_dir / "features" / "features_psd_summary.csv")
    fc_summary = pd.read_csv(project.output_dir / "features" / "features_fc_summary.csv")
    tacs_summary = pd.read_csv(project.output_dir / "features" / "features_tacs_target_summary.csv")
    reactivity = pd.read_csv(project.output_dir / "features" / "features_eo_ec_reactivity.csv")
    all_summary = pd.read_csv(project.output_dir / "features" / "features_all_summary.csv")
    dictionary = pd.read_csv(project.output_dir / "features" / "feature_dictionary.csv")
    matrix_dir = project.output_dir / "features" / "matrices"
    psd_matrix_flat = flatten_psd_matrices(
        subjects,
        np.load(matrix_dir / "psd_eo.npy"),
        np.load(matrix_dir / "psd_ec.npy"),
        dictionary,
    )
    fc_matrix_flat = flatten_fc_matrices(
        subjects,
        np.load(matrix_dir / "fc_roi_eo.npy"),
        np.load(matrix_dir / "fc_roi_ec.npy"),
        dictionary,
    )

    fold_dir = project.output_dir / "folds"
    outer_folds = json.loads((fold_dir / "outer_folds.json").read_text(encoding="utf-8"))
    registries = load_fold_registries(fold_dir, outer_folds)
    outputs = run_classical_ml_baselines(
        config,
        cohort=cohort,
        folds=outer_folds,
        registries=registries,
        psd_summary=psd_summary,
        fc_summary=fc_summary,
        tacs_summary=tacs_summary,
        reactivity=reactivity,
        all_summary=all_summary,
        psd_matrix_flat=psd_matrix_flat,
        fc_matrix_flat=fc_matrix_flat,
    )
    predictions = pd.read_csv(outputs["predictions"])
    print("CLASSICAL_ML_OK")
    print(f"n_models={predictions['model_id'].nunique()}")
    print(f"n_predictions={len(predictions)}")
    return 0


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


def _resolve_output_paths(config: dict[str, object], project_root: Path) -> None:
    output_paths = config.get("output_paths")
    if not isinstance(output_paths, dict):
        raise ValueError("models_ml config must define output_paths")
    for key, value in list(output_paths.items()):
        path = Path(str(value))
        if not path.is_absolute():
            path = project_root / path
        output_paths[key] = str(path)


if __name__ == "__main__":
    raise SystemExit(main())
