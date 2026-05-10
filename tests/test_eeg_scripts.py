from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stroke_predict.eeg.outputs import (
    assert_public_eeg_output,
    build_channel_order_report,
    write_qc_outputs,
    write_record_index_output,
)


def test_public_output_rejects_path_columns() -> None:
    private_path = "F" + ":/private/name" + ".set"
    frame = pd.DataFrame({"subject_id": ["STK-001"], "set_path": [private_path]})

    with pytest.raises(ValueError, match="set_path"):
        assert_public_eeg_output(frame)


def test_public_output_rejects_path_like_values() -> None:
    private_path = "F" + ":/private/name" + ".set"
    frame = pd.DataFrame({"subject_id": ["STK-001"], "note": [private_path]})

    with pytest.raises(ValueError, match="path-like"):
        assert_public_eeg_output(frame)


def test_write_outputs_and_channel_report(tmp_path: Path) -> None:
    record_index = pd.DataFrame(
        {
            "record_id": ["STK-001_baseline_eyes_open_01"],
            "subject_id": ["STK-001"],
            "source": ["stroke"],
            "stage": ["baseline"],
            "condition": ["eyes_open"],
            "record_index": [1],
            "set_exists": [True],
            "fdt_exists": [True],
        }
    )
    qc = pd.DataFrame(
        {
            "record_id": ["STK-001_baseline_eyes_open_01"],
            "subject_id": ["STK-001"],
            "source": ["stroke"],
            "stage": ["baseline"],
            "condition": ["eyes_open"],
            "exists": [True],
            "readable": [True],
            "n_channels": [62],
            "sfreq": [250],
            "channel_order_hash": ["abc"],
            "duration_sec": [60.0],
            "n_valid_samples": [15000],
            "n_valid_windows_2s": [59],
            "n_valid_windows_4s": [29],
            "n_valid_windows_8s": [14],
            "bad_channel_count": [0],
            "artifact_ratio_if_available": [None],
            "passes_qc": [True],
            "qc_reason": ["pass"],
        }
    )

    index_path = write_record_index_output(record_index, tmp_path)
    paths = write_qc_outputs(record_index=record_index, qc_summary=qc, output_dir=tmp_path)
    report = build_channel_order_report(qc)

    assert index_path.name == "eeg_record_index.csv"
    assert paths["qc_summary"].exists()
    assert paths["channel_order_report"].exists()
    assert report.loc[0, "n_records"] == 1


def test_index_and_qc_scripts_exist() -> None:
    root = Path(__file__).resolve().parents[1]

    assert (root / "scripts" / "02_index_eeg.py").exists()
    assert (root / "scripts" / "03_run_eeg_qc.py").exists()
