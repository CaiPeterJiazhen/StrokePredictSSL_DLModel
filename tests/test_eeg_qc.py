from __future__ import annotations

from pathlib import Path

import pandas as pd
import scipy.io

from stroke_predict.eeg.config import EEGConfig
from stroke_predict.eeg.header import EEGHeader, read_eeglab_set_header
from stroke_predict.eeg.qc import channel_order_hash, count_windows, evaluate_qc, run_qc


def _config() -> EEGConfig:
    return EEGConfig(path=Path(__file__), project_config_path=Path(__file__), raw={})


def test_count_windows_uses_overlap() -> None:
    assert count_windows(duration_sec=60, length_sec=4, overlap=0.5) == 29
    assert count_windows(duration_sec=3.9, length_sec=4, overlap=0.5) == 0


def test_channel_order_hash_is_stable() -> None:
    assert channel_order_hash([" FP1 ", "C3"]) == channel_order_hash(["fp1", " c3"])
    assert channel_order_hash(["FP1", "C3"]) != channel_order_hash(["C3", "FP1"])


def test_reads_eeglab_set_header_without_raw_array(tmp_path: Path) -> None:
    set_path = tmp_path / "synthetic.set"
    scipy.io.savemat(
        set_path,
        {
            "nbchan": 2,
            "srate": 250,
            "pnts": 500,
            "trials": 1,
            "datfile": "synthetic.fdt",
            "chanlocs": [{"labels": "C3"}, {"labels": "C4"}],
        },
    )

    header = read_eeglab_set_header(set_path)

    assert header.n_channels == 2
    assert header.sfreq == 250
    assert header.pnts == 500
    assert header.trials == 1
    assert header.duration_sec == 2
    assert header.channel_labels == ["C3", "C4"]
    assert header.datfile == "synthetic.fdt"


def test_evaluate_qc_rejects_bad_sampling_rate() -> None:
    header = EEGHeader(
        n_channels=62,
        sfreq=500,
        pnts=15000,
        trials=1,
        channel_labels=["C3", "C4"],
        datfile="x.fdt",
    )

    result = evaluate_qc(
        header,
        source="stroke",
        stage="baseline",
        condition="eyes_open",
        set_exists=True,
        fdt_exists=True,
        readable=True,
        config=_config(),
    )

    assert result["passes_qc"] is False
    assert "bad_sampling_rate" in result["qc_reason"]


def test_run_qc_preserves_public_schema() -> None:
    public_index = pd.DataFrame(
        {
            "record_id": ["STK-001_baseline_eyes_open_01"],
            "subject_id": ["STK-001"],
            "source": ["stroke"],
            "stage": ["baseline"],
            "condition": ["eyes_open"],
            "record_index": [1],
            "set_exists": [False],
            "fdt_exists": [False],
        }
    )

    qc = run_qc(
        public_index,
        private_records=[{"record_id": "STK-001_baseline_eyes_open_01", "set_path": "private.set", "fdt_path": "private.fdt"}],
        config=_config(),
    )

    assert {"set_path", "fdt_path", "file_path", "subject_name"}.isdisjoint(qc.columns)
    assert bool(qc.loc[0, "passes_qc"]) is False
    assert "missing_set" in qc.loc[0, "qc_reason"]
