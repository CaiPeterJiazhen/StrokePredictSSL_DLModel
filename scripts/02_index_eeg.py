from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_project_config
from stroke_predict.eeg.config import load_eeg_config
from stroke_predict.eeg.index import build_eeg_record_index
from stroke_predict.eeg.outputs import write_record_index_output
from stroke_predict.io.excel_status import read_status_workbook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    eeg_config = load_eeg_config(args.config)
    project = load_project_config(eeg_config.project_config_path)
    status = read_status_workbook(
        project.workbook_path,
        sheets={
            "summary": project.sheet("summary"),
            "clinical_overview": project.sheet("clinical_overview"),
            "clinical_raw": project.sheet("clinical_raw"),
            "preprocessed_summary": project.sheet("preprocessed_summary"),
            "preprocessed_files": project.sheet("preprocessed_files"),
        },
    )
    record_index = build_eeg_record_index(status)
    path = write_record_index_output(record_index, project.output_dir / "qc")
    print("EEG_INDEX_OK")
    print(f"record_index={path}")
    print(f"n_records={len(record_index)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
