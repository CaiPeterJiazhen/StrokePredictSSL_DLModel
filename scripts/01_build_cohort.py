from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.cohort.build import build_cohort_tables
from stroke_predict.cohort.outputs import write_cohort_outputs
from stroke_predict.config import load_project_config
from stroke_predict.io.excel_status import read_status_workbook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_project_config(args.config)
    status = read_status_workbook(
        config.workbook_path,
        sheets={
            "summary": config.sheet("summary"),
            "clinical_overview": config.sheet("clinical_overview"),
            "clinical_raw": config.sheet("clinical_raw"),
            "preprocessed_summary": config.sheet("preprocessed_summary"),
            "preprocessed_files": config.sheet("preprocessed_files"),
        },
    )
    label_settings = {
        "fma_full_score": config.label_setting("fma_full_score"),
        "low_fma_threshold": config.label_setting("low_fma_threshold"),
        "low_fma_delta_good": config.label_setting("low_fma_delta_good"),
        "near_ceiling_delta_good": config.label_setting("near_ceiling_delta_good"),
        "proportional_good_threshold": config.label_setting("proportional_good_threshold"),
    }
    tables = build_cohort_tables(
        status,
        pii_columns=config.pii_columns,
        label_settings=label_settings,
    )
    paths = write_cohort_outputs(
        tables,
        cohort_dir=config.output_subdir("cohort_dir"),
        figures_dir=config.output_subdir("figures_dir"),
    )
    print("COHORT_BUILD_OK")
    for key, path in paths.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
