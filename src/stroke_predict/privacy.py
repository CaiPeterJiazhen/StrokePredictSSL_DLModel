from __future__ import annotations

import pandas as pd


DEFAULT_PII_COLUMNS = {
    "姓名",
    "姓名写法",
    "EEG文件夹",
    "subject_name",
    "set_path",
    "fdt_path",
}


def drop_pii_columns(df: pd.DataFrame, pii_columns: list[str] | set[str]) -> pd.DataFrame:
    return df.drop(columns=[column for column in pii_columns if column in df.columns])


def assert_no_pii_columns(df: pd.DataFrame, pii_columns: list[str] | set[str]) -> None:
    blocked = sorted(set(df.columns).intersection(set(pii_columns)))
    if blocked:
        raise ValueError(f"PII columns present in public output: {blocked}")
