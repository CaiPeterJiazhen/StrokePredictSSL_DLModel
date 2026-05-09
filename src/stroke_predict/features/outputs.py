from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


FORBIDDEN_PUBLIC_COLUMNS = {"subject_name", "set_path", "fdt_path", "file_path", "_source_key", "姓名", "姓名写法", "EEG文件夹"}
FEATURE_DICTIONARY_COLUMNS = [
    "feature_name",
    "feature_group",
    "condition",
    "band",
    "channel",
    "roi",
    "metric",
    "hemisphere_space",
    "matrix_file",
    "axis0_subject_index",
    "axis1_view_index",
    "axis2_feature_index",
    "axis3_feature_index",
]


def assert_public_feature_output(frame: pd.DataFrame) -> None:
    leaked = sorted(FORBIDDEN_PUBLIC_COLUMNS.intersection(frame.columns))
    if leaked:
        raise ValueError(f"Feature output contains forbidden columns: {leaked}")
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        values = frame[column].dropna().astype(str)
        if values.str.contains(r"[A-Za-z]:[\\/]|\.set$|\.fdt$", regex=True).any():
            raise ValueError(f"Feature output contains path-like values in column {column}")


def validate_feature_dictionary(dictionary: pd.DataFrame) -> None:
    missing = [column for column in FEATURE_DICTIONARY_COLUMNS if column not in dictionary.columns]
    if missing:
        raise ValueError(f"feature_dictionary missing required columns: {missing}")
    assert_public_feature_output(dictionary)


def save_matrix(path: str | Path, matrix: np.ndarray) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, matrix)
    return output


def write_public_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    assert_public_feature_output(frame)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return output


def dictionary_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in FEATURE_DICTIONARY_COLUMNS:
        if column not in frame.columns:
            frame[column] = None
    frame = frame[FEATURE_DICTIONARY_COLUMNS]
    validate_feature_dictionary(frame)
    return frame

