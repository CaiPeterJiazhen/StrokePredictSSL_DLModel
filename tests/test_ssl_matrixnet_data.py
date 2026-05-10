from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stroke_predict.ssl_matrixnet_data import (
    SSL_VARIANTS,
    build_ssl_matrix_index_from_baseline_outputs,
    validate_ssl_matrix_arrays,
    validate_ssl_matrix_index,
)

from tests.test_matrixnet_no_leakage import _write_minimal_inputs


def test_ssl_matrix_index_has_required_public_columns() -> None:
    index = _synthetic_ssl_index()

    validated = validate_ssl_matrix_index(index)

    assert {"subject_id", "source", "stage", "condition"} <= set(validated.columns)
    for variant in SSL_VARIANTS:
        assert f"eligible_{variant}" in validated.columns
    assert validated["subject_id"].tolist() == index["subject_id"].astype(str).tolist()


def test_ssl_matrix_index_rejects_raw_local_paths() -> None:
    index = _synthetic_ssl_index()
    index.loc[0, "source_file"] = r"C:\Users\person\raw_file.set"

    with pytest.raises(ValueError, match="private or raw path"):
        validate_ssl_matrix_index(index)


def test_ssl_matrix_array_row_count_matches_metadata() -> None:
    index = validate_ssl_matrix_index(_synthetic_ssl_index())
    psd = np.zeros((len(index), 2, 3, 4), dtype=np.float32)
    fc = np.zeros((len(index), 4, 3, 2), dtype=np.float32)

    validate_ssl_matrix_arrays(index, psd=psd, fc=fc)


def test_ssl_matrix_array_row_count_mismatch_fails() -> None:
    index = validate_ssl_matrix_index(_synthetic_ssl_index())
    psd = np.zeros((len(index) - 1, 2, 3, 4), dtype=np.float32)
    fc = np.zeros((len(index), 4, 3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="PSD row count"):
        validate_ssl_matrix_arrays(index, psd=psd, fc=fc)


def test_build_ssl_matrix_index_from_baseline_outputs_aligns_patients(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)

    index, psd, fc = build_ssl_matrix_index_from_baseline_outputs(tmp_path)

    assert index["subject_id"].tolist() == ["S01", "S01", "S02", "S02", "S03", "S03"]
    assert index["condition"].tolist() == ["eo", "ec", "eo", "ec", "eo", "ec"]
    assert psd.shape[0] == len(index)
    assert fc.shape[0] == len(index)
    validate_ssl_matrix_arrays(index, psd=psd, fc=fc)


def _synthetic_ssl_index() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["STK-001", "STK-002", "STK-003", "HC-001"],
            "source": ["stroke_supervised", "stroke_supervised", "stroke_ssl_only", "healthy"],
            "stage": ["baseline", "baseline", "mid", "baseline"],
            "condition": ["eo", "ec", "eo", "ec"],
            "row_index": [0, 1, 2, 3],
        }
    )
