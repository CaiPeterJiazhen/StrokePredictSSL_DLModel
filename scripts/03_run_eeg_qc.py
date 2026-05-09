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
from stroke_predict.eeg.index import build_eeg_private_records, build_eeg_record_index
from stroke_predict.eeg.outputs import write_qc_outputs
from stroke_predict.eeg.qc import run_qc
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
    private_records = build_eeg_private_records(status)
    qc_summary = run_qc(record_index, private_records=private_records, config=eeg_config)
    paths = write_qc_outputs(
        record_index=record_index,
        qc_summary=qc_summary,
        output_dir=project.output_dir / "qc",
    )
    print("EEG_QC_OK")
    for key, path in paths.items():
        print(f"{key}={path}")
    print(f"n_records={len(qc_summary)}")
    print(f"n_passes_qc={int(qc_summary['passes_qc'].sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
