from __future__ import annotations

import re

import pandas as pd


DEFAULT_PII_COLUMNS = {
    "姓名",
    "姓名写法",
    "EEG文件夹",
    "subject_name",
    "set_path",
    "fdt_path",
}

PATH_VALUE_RE = re.compile(
    r"(\.set\b|\.fdt\b|Patient_tACS|Health_tACS|RestingStateEEG_afterProcess|[A-Za-z]:[\\/])",
    re.IGNORECASE,
)


def drop_pii_columns(df: pd.DataFrame, pii_columns: list[str] | set[str]) -> pd.DataFrame:
    return df.drop(columns=[column for column in pii_columns if column in df.columns])


def assert_no_pii_columns(df: pd.DataFrame, pii_columns: list[str] | set[str]) -> None:
    blocked = sorted(set(df.columns).intersection(set(pii_columns)))
    if blocked:
        raise ValueError(f"PII columns present in public output: {blocked}")

    leaks = _find_path_like_value_leaks(df)
    if leaks:
        raise ValueError(f"PII-like path values present in public output: {leaks}")


def _find_path_like_value_leaks(df: pd.DataFrame) -> list[str]:
    leaks: list[str] = []
    for column in df.columns:
        values = df[column].dropna()
        for value in values:
            if isinstance(value, str) and PATH_VALUE_RE.search(value):
                leaks.append(str(column))
                break
    return sorted(leaks)
