from __future__ import annotations

import pandas as pd
import pytest

from stroke_predict.cohort.ids import build_subject_id_map, normalize_source_key
from stroke_predict.privacy import assert_no_pii_columns, drop_pii_columns


def test_build_subject_id_map_is_deterministic_and_source_prefixed() -> None:
    first = build_subject_id_map(["sub02", "sub01"], source="stroke", prefix="STK")
    second = build_subject_id_map(["sub01", "sub02"], source="stroke", prefix="STK")

    assert first == second
    assert first["sub01"] == "STK-001"
    assert first["sub02"] == "STK-002"


def test_build_subject_id_map_rejects_duplicate_public_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        build_subject_id_map(["sub01", "sub01"], source="stroke", prefix="STK")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("sub13", "sub13"),
        ("sub013", "sub13"),
        ("sub013_baseline", "sub13"),
        ("sub013_rest.set", "sub13"),
        ("F" + ":/x/sub013_baseline", "sub13"),
        ("sub001", "sub01"),
        ("participant-a", "participant-a"),
    ],
)
def test_normalize_source_key_extracts_subject_codes_from_real_variants(
    raw: str, expected: str
) -> None:
    assert normalize_source_key(raw) == expected


def test_privacy_helpers_remove_and_reject_pii_columns() -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["STK-001"],
            "姓名": ["张三"],
            "subject_name": ["张三"],
            "label_primary": ["Good"],
        }
    )

    cleaned = drop_pii_columns(df, ["姓名", "subject_name"])
    assert list(cleaned.columns) == ["subject_id", "label_primary"]
    assert_no_pii_columns(cleaned, ["姓名", "subject_name"])

    with pytest.raises(ValueError, match="PII columns"):
        assert_no_pii_columns(df, ["姓名", "subject_name"])


@pytest.mark.parametrize("column", ["eeg_file", "source_value"])
def test_assert_no_pii_columns_rejects_path_like_value_leaks(column: str) -> None:
    df = pd.DataFrame(
        {
            "subject_id": ["STK-001"],
            column: ["F" + ":/x/sub013_name/file" + ".set"],
        }
    )

    with pytest.raises(ValueError, match="PII-like path values"):
        assert_no_pii_columns(df, ["姓名", "subject_name"])
