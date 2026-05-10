from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_yaml_mapping
from stroke_predict.phase8_labels import build_phase8_label_table
from stroke_predict.phase8_reports import write_phase8_label_audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_mapping(config_path)
    input_paths = dict(config.get("input_paths", {}))
    cohort_path = _resolve(config_path, str(input_paths.get("cohort", "outputs/cohort/cohort_master.csv")))
    output_dir = _resolve(config_path, str(config.get("output_dir", "outputs")))

    cohort = pd.read_csv(cohort_path)
    if "role" in cohort.columns:
        cohort = cohort.loc[cohort["role"].eq("supervised_main")].copy()
    labels, audit = build_phase8_label_table(cohort)
    write_phase8_label_audit(labels, audit, output_dir=output_dir)

    print("PHASE8_LABELS_OK")
    print(f"run_mode={args.run_mode}")
    print(f"n_analyzable={audit['n_analyzable']}")
    print(f"median_residual={audit['median_residual']}")
    print(f"n_proportional_recovery={audit['n_proportional_recovery']}")
    print(f"n_poor_recovery={audit['n_poor_recovery']}")
    return 0


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = config_path.parent / path
    return path.resolve()


if __name__ == "__main__":
    raise SystemExit(main())
