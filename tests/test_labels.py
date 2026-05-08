from __future__ import annotations

import math

import pandas as pd
import pytest

from stroke_predict.cohort.labels import build_label_record, parse_optional_float


@pytest.mark.parametrize(
    ("baseline", "post", "expected"),
    [
        (None, 40, "missing"),
        (40, None, "missing"),
        ("", 40, "missing"),
        (66, 66, "ceiling_exclude"),
        (40, 44, "Poor"),
        (40, 45, "Good"),
        (61, 65, "Poor"),
        (61, 66, "Good"),
        (64, 64, "Poor"),
        (64, 65, "Poor"),
        (64, 66, "Good"),
        (65, 65, "Poor"),
        (65, 66, "Good"),
    ],
)
def test_primary_label_rules(baseline, post, expected) -> None:
    record = build_label_record(baseline, post)
    assert record["label_primary"] == expected


def test_label_record_contains_numeric_audit_fields() -> None:
    record = build_label_record(40, 45)
    assert record["baseline_fma"] == 40.0
    assert record["post_fma"] == 45.0
    assert record["delta_fma"] == 5.0
    assert record["possible_recovery"] == 26.0
    assert math.isclose(record["recovery_ratio"], 5.0 / 26.0)
    assert record["outcome_delta_fma"] == 5.0
    assert record["outcome_post_fma"] == 45.0


def test_delta5_all_and_prop70_labels() -> None:
    record = build_label_record(60, 65)
    assert record["label_delta5_all"] == "Good"
    assert record["label_prop70"] == "Poor"

    strong_recovery = build_label_record(60, 65, proportional_good_threshold=0.19)
    assert strong_recovery["label_prop70"] == "Good"


def test_low_baseline_only_excludes_near_ceiling() -> None:
    low = build_label_record(61, 66)
    near_ceiling = build_label_record(64, 65)

    assert low["label_low_baseline_only"] == "Good"
    assert near_ceiling["label_low_baseline_only"] == "missing"


def test_parse_optional_float_accepts_numbers_and_blanks() -> None:
    assert parse_optional_float(" 42.5 ") == 42.5
    assert parse_optional_float("") is None
    assert parse_optional_float(None) is None
    assert parse_optional_float("not numeric") is None


@pytest.mark.parametrize(
    ("baseline", "post"),
    [
        (float("nan"), 45),
        (40, float("nan")),
        (pd.NA, 45),
    ],
)
def test_nan_and_pandas_na_are_missing_fma(baseline, post) -> None:
    record = build_label_record(baseline, post)
    assert record["label_primary"] == "missing"
    assert record["label_reason"] == "missing_fma"


@pytest.mark.parametrize(
    ("baseline", "post"),
    [
        (70, 70),
        (-1, 45),
        (40, 70),
    ],
)
def test_fma_values_outside_valid_range_are_missing(baseline, post) -> None:
    record = build_label_record(baseline, post)
    assert record["label_primary"] == "missing"
    assert record["label_reason"] == "invalid_fma_range"


def test_label_primary_values_stay_in_allowed_set() -> None:
    records = [
        build_label_record(40, 45),
        build_label_record(40, 44),
        build_label_record(66, 66),
        build_label_record(float("nan"), 45),
        build_label_record(70, 70),
    ]
    assert {record["label_primary"] for record in records} <= {
        "Good",
        "Poor",
        "ceiling_exclude",
        "missing",
    }
