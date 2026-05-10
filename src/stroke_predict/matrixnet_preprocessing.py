from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MatrixStats:
    mean: float
    std: float
    n_fit_subjects: int
    fit_subjects: tuple[str, ...]


@dataclass(frozen=True)
class FoldPreprocessor:
    subject_ids: tuple[str, ...]
    train_subjects: tuple[str, ...]
    matrix_stats: dict[str, MatrixStats]

    @classmethod
    def fit(cls, subject_ids: list[str], train_subjects: list[str], matrices: dict[str, np.ndarray]) -> "FoldPreprocessor":
        subject_to_index = {subject: index for index, subject in enumerate(subject_ids)}
        missing = [subject for subject in train_subjects if subject not in subject_to_index]
        if missing:
            raise ValueError(f"Unknown training subjects for matrix preprocessing: {missing}")
        indices = [subject_to_index[subject] for subject in train_subjects]
        if not indices:
            raise ValueError("No training subjects for matrix preprocessing")
        stats: dict[str, MatrixStats] = {}
        for name, matrix in matrices.items():
            train_values = matrix[indices].astype(float)
            if not np.isfinite(train_values).all():
                raise ValueError(f"{name} training matrix contains NaN or Inf")
            mean = float(np.mean(train_values))
            std = float(np.std(train_values))
            if std == 0.0:
                std = 1.0
            stats[name] = MatrixStats(mean=mean, std=std, n_fit_subjects=len(indices), fit_subjects=tuple(train_subjects))
        return cls(subject_ids=tuple(subject_ids), train_subjects=tuple(train_subjects), matrix_stats=stats)

    def transform_matrix(self, name: str, matrix: np.ndarray) -> np.ndarray:
        stats = self.matrix_stats[name]
        values = matrix.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} matrix contains NaN or Inf")
        return ((values - stats.mean) / stats.std).astype(np.float32)


@dataclass(frozen=True)
class VectorPreprocessor:
    subject_ids: tuple[str, ...]
    feature_columns: tuple[str, ...]
    medians: dict[str, float]
    means: dict[str, float]
    stds: dict[str, float]
    categories: dict[str, tuple[str, ...]]

    @property
    def output_dim(self) -> int:
        total = 0
        for column in self.feature_columns:
            total += len(self.categories[column]) if column in self.categories else 1
        return total

    def transform(self, frame: pd.DataFrame, subject_ids: list[str]) -> np.ndarray:
        aligned = _align(frame, subject_ids)
        columns: list[np.ndarray] = []
        for column in self.feature_columns:
            if column in self.categories:
                values = aligned[column].fillna("__missing__").astype(str)
                for category in self.categories[column]:
                    columns.append((values == category).astype(float).to_numpy()[:, None])
            else:
                values = pd.to_numeric(aligned[column], errors="coerce").fillna(self.medians[column]).astype(float)
                scaled = (values.to_numpy() - self.means[column]) / self.stds[column]
                columns.append(scaled[:, None])
        if not columns:
            return np.zeros((len(subject_ids), 0), dtype=np.float32)
        return np.concatenate(columns, axis=1).astype(np.float32)


def fit_vector_preprocessor(frame: pd.DataFrame, *, subject_ids: list[str], train_subjects: list[str]) -> VectorPreprocessor:
    aligned = _align(frame, subject_ids)
    train_subject_set = set(train_subjects)
    train = aligned[aligned["subject_id"].astype(str).isin(train_subject_set)].copy()
    if train.empty:
        raise ValueError("No training subjects for vector preprocessing")
    feature_columns = [column for column in aligned.columns if column != "subject_id"]
    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    categories: dict[str, tuple[str, ...]] = {}
    for column in feature_columns:
        numeric = pd.to_numeric(train[column], errors="coerce")
        if numeric.notna().any():
            median = float(numeric.median())
            filled = numeric.fillna(median).astype(float)
            mean = float(filled.mean())
            std = float(filled.std(ddof=0))
            medians[column] = median
            means[column] = mean
            stds[column] = std if std > 0 else 1.0
        else:
            values = train[column].fillna("__missing__").astype(str)
            categories[column] = tuple(sorted(values.unique().tolist()))
    return VectorPreprocessor(
        subject_ids=tuple(subject_ids),
        feature_columns=tuple(feature_columns),
        medians=medians,
        means=means,
        stds=stds,
        categories=categories,
    )


def _align(frame: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    if "subject_id" not in frame.columns:
        raise ValueError("Vector frame missing subject_id")
    indexed = frame.assign(subject_id=lambda value: value["subject_id"].astype(str))
    indexed = indexed.drop_duplicates("subject_id").set_index("subject_id")
    missing = [subject for subject in subject_ids if subject not in indexed.index]
    if missing:
        raise ValueError(f"Vector frame missing subjects: {missing}")
    return indexed.loc[subject_ids].reset_index()
