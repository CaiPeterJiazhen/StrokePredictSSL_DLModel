# Phase 6 MatrixNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现并以 fast mode 跑通 supervised no-SSL Lin-style EEG MatrixNet，生成 19-fold patient-level predictions、metrics、audit 和 Phase 6 report。

**Architecture:** 新增 MatrixNet 专用数据加载、fold-safe preprocessing、PyTorch 多分支模型、训练评估输出模块和入口脚本。所有 scaler/imputer/threshold/early stopping 都绑定 outer train 或 inner validation，不使用 outer test patient。

**Tech Stack:** Python, PyTorch, NumPy, pandas, scikit-learn metrics, PyYAML, pytest.

---

## File Structure

- Create `src/stroke_predict/matrixnet.py`
  - PyTorch `MatrixBranch`、`VectorBranch`、`MatrixNetConfig`、`MatrixNet`。
- Create `src/stroke_predict/matrixnet_data.py`
  - 加载 cohort/folds/matrices/vector features/comparison metrics，并验证 subject alignment。
- Create `src/stroke_predict/matrixnet_preprocessing.py`
  - Fold-safe matrix z-score、vector median impute + scaling、audit stats。
- Create `src/stroke_predict/matrixnet_training.py`
  - Dataset、training loop、early stopping、threshold selection、LOPO runner、metrics/report writers。
- Create `scripts/09_train_matrixnet.py`
  - CLI：`--config configs/matrixnet.yaml --run-mode fast|full`，可选 `--fold-limit` 用于测试。
- Create `configs/matrixnet.yaml`
  - fast/full run settings、model families、paths、hyperparameter grid。
- Create tests:
  - `tests/test_matrixnet_shapes.py`
  - `tests/test_matrixnet_no_leakage.py`
  - `tests/test_matrixnet_training_smoke.py`
  - `tests/test_matrixnet_outputs.py`
- Modify `requirements.txt`
  - Add `torch` only if dependency declaration is required by project policy. Do not vendor dependencies.

## Task 1: RED tests for MatrixNet shapes

**Files:**
- Create: `tests/test_matrixnet_shapes.py`
- Create later: `src/stroke_predict/matrixnet.py`

- [ ] **Step 1: Write the failing test**

Add this test file:

```python
from __future__ import annotations

import torch

from stroke_predict.matrixnet import MatrixNet, MatrixNetConfig


def test_matrixnet_accepts_psd_only_inputs() -> None:
    model = MatrixNet(MatrixNetConfig(use_psd=True, use_fc=False, use_tacs=False, use_clinical=False))
    logits = model(
        psd_eo=torch.randn(4, 2, 62, 90),
        psd_ec=torch.randn(4, 2, 62, 90),
    )
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_matrixnet_accepts_fc_only_inputs() -> None:
    model = MatrixNet(MatrixNetConfig(use_psd=False, use_fc=True, use_tacs=False, use_clinical=False))
    logits = model(
        fc_eo=torch.randn(3, 2, 36, 6, 2),
        fc_ec=torch.randn(3, 2, 36, 6, 2),
    )
    assert logits.shape == (3,)
    assert torch.isfinite(logits).all()


def test_matrixnet_accepts_psd_fc_tacs_clinical_inputs() -> None:
    model = MatrixNet(
        MatrixNetConfig(
            use_psd=True,
            use_fc=True,
            use_tacs=True,
            use_clinical=True,
            tacs_dim=7,
            clinical_dim=5,
            embedding_dim=16,
            hidden_dim=32,
            dropout=0.2,
        )
    )
    logits = model(
        psd_eo=torch.randn(2, 2, 62, 90),
        psd_ec=torch.randn(2, 2, 62, 90),
        fc_eo=torch.randn(2, 2, 36, 6, 2),
        fc_ec=torch.randn(2, 2, 36, 6, 2),
        tacs=torch.randn(2, 7),
        clinical=torch.randn(2, 5),
    )
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_shapes.py -q
Pop-Location
```

Expected: FAIL with `ModuleNotFoundError: No module named 'stroke_predict.matrixnet'`.

- [ ] **Step 3: Implement minimal MatrixNet**

Create `src/stroke_predict/matrixnet.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MatrixNetConfig:
    use_psd: bool = True
    use_fc: bool = True
    use_tacs: bool = False
    use_clinical: bool = False
    tacs_dim: int = 0
    clinical_dim: int = 0
    embedding_dim: int = 32
    hidden_dim: int = 64
    dropout: float = 0.5


class MatrixBranch(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projection = nn.Linear(32, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _as_single_channel_image(x)
        return self.projection(self.features(x))


class VectorBranch(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("VectorBranch input_dim must be positive")
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())


class MatrixNet(nn.Module):
    def __init__(self, config: MatrixNetConfig) -> None:
        super().__init__()
        if not any([config.use_psd, config.use_fc, config.use_tacs, config.use_clinical]):
            raise ValueError("At least one input family must be enabled")
        self.config = config
        self.psd_branch = MatrixBranch(config.embedding_dim, config.dropout) if config.use_psd else None
        self.fc_branch = MatrixBranch(config.embedding_dim, config.dropout) if config.use_fc else None
        self.tacs_branch = (
            VectorBranch(config.tacs_dim, config.embedding_dim, config.hidden_dim, config.dropout)
            if config.use_tacs
            else None
        )
        self.clinical_branch = (
            VectorBranch(config.clinical_dim, config.embedding_dim, config.hidden_dim, config.dropout)
            if config.use_clinical
            else None
        )
        n_embeddings = int(config.use_psd) * 2 + int(config.use_fc) * 2 + int(config.use_tacs) + int(config.use_clinical)
        fusion_dim = n_embeddings * config.embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        *,
        psd_eo: torch.Tensor | None = None,
        psd_ec: torch.Tensor | None = None,
        fc_eo: torch.Tensor | None = None,
        fc_ec: torch.Tensor | None = None,
        tacs: torch.Tensor | None = None,
        clinical: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings: list[torch.Tensor] = []
        if self.psd_branch is not None:
            if psd_eo is None or psd_ec is None:
                raise ValueError("PSD inputs are required for this MatrixNet")
            embeddings.extend([self.psd_branch(psd_eo), self.psd_branch(psd_ec)])
        if self.fc_branch is not None:
            if fc_eo is None or fc_ec is None:
                raise ValueError("FC inputs are required for this MatrixNet")
            embeddings.extend([self.fc_branch(fc_eo), self.fc_branch(fc_ec)])
        if self.tacs_branch is not None:
            if tacs is None:
                raise ValueError("tACS input is required for this MatrixNet")
            embeddings.append(self.tacs_branch(tacs))
        if self.clinical_branch is not None:
            if clinical is None:
                raise ValueError("Clinical input is required for this MatrixNet")
            embeddings.append(self.clinical_branch(clinical))
        fused = torch.cat(embeddings, dim=1)
        return self.classifier(fused).squeeze(-1)


def _as_single_channel_image(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    if x.ndim == 3:
        return x.unsqueeze(1)
    if x.ndim == 4 and x.shape[1] == 1:
        return x
    if x.ndim >= 4:
        batch = x.shape[0]
        height = x.shape[-2]
        width = x.shape[-1]
        return x.reshape(batch, 1, -1, width) if x.ndim > 4 else x.reshape(batch, 1, x.shape[1] * height, width)
    raise ValueError(f"Matrix input must have at least 3 dimensions, found {tuple(x.shape)}")
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_shapes.py -q
Pop-Location
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git -C .worktrees/phase6-matrixnet add tests/test_matrixnet_shapes.py src/stroke_predict/matrixnet.py
git -C .worktrees/phase6-matrixnet commit -m "feat: add MatrixNet model skeleton"
```

## Task 2: RED tests for data loading and subject alignment

**Files:**
- Create: `tests/test_matrixnet_no_leakage.py`
- Create later: `src/stroke_predict/matrixnet_data.py`

- [ ] **Step 1: Write the failing tests**

Add the first half of `tests/test_matrixnet_no_leakage.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stroke_predict.matrixnet_data import load_matrixnet_inputs, validate_fold_registry


def _write_minimal_inputs(root: Path) -> None:
    (root / "cohort").mkdir(parents=True)
    (root / "folds").mkdir(parents=True)
    (root / "matrices").mkdir(parents=True)
    (root / "features").mkdir(parents=True)
    cohort = pd.DataFrame(
        {
            "subject_id": ["S01", "S02", "S03"],
            "role": ["supervised_main", "supervised_main", "supervised_main"],
            "label_primary": ["Good", "Poor", "Good"],
            "baseline_fma": [50, 40, 55],
            "age": [60, 61, 62],
            "sex": ["F", "M", "F"],
            "baseline_mbi": [80, 70, 90],
            "mmse": [28, 27, 29],
            "affected_hand": ["left", "right", "left"],
            "treated_hand": ["left", "right", "left"],
            "has_baseline_eo": [True, True, True],
            "has_baseline_ec": [True, True, True],
        }
    )
    cohort.to_csv(root / "cohort" / "cohort_master.csv", index=False)
    folds = {
        "n_supervised_main": 3,
        "supervised_subjects": ["S01", "S02", "S03"],
        "folds": [
            {"outer_fold": 1, "test_subject": "S01", "supervised_train_subjects": ["S02", "S03"], "registry_path": "fold_01_registry.json"},
            {"outer_fold": 2, "test_subject": "S02", "supervised_train_subjects": ["S01", "S03"], "registry_path": "fold_02_registry.json"},
            {"outer_fold": 3, "test_subject": "S03", "supervised_train_subjects": ["S01", "S02"], "registry_path": "fold_03_registry.json"},
        ],
    }
    (root / "folds" / "outer_folds.json").write_text(json.dumps(folds), encoding="utf-8")
    for fold in folds["folds"]:
        test_subject = fold["test_subject"]
        train_subjects = fold["supervised_train_subjects"]
        registry = {
            "outer_fold": fold["outer_fold"],
            "test_subject": test_subject,
            "supervised_train_subjects": train_subjects,
            "inner_splits": [{"inner_fold": 1, "train_subjects": train_subjects[:1], "val_subjects": train_subjects[1:]}],
            "normalization_fit_subjects": train_subjects,
            "threshold_selection_subjects": train_subjects,
        }
        (root / "folds" / fold["registry_path"]).write_text(json.dumps(registry), encoding="utf-8")
    for name, shape in {
        "psd_eo.npy": (3, 2, 4, 5),
        "psd_ec.npy": (3, 2, 4, 5),
        "fc_roi_eo.npy": (3, 2, 3, 2, 2),
        "fc_roi_ec.npy": (3, 2, 3, 2, 2),
    }.items():
        np.save(root / "matrices" / name, np.ones(shape, dtype=np.float32))
    pd.DataFrame({"subject_id": ["S01", "S02", "S03"], "tacs_a": [1.0, None, 3.0]}).to_csv(
        root / "features" / "features_tacs_target_summary.csv",
        index=False,
    )


def test_load_matrixnet_inputs_aligns_rows_to_sorted_supervised_subjects(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    inputs = load_matrixnet_inputs(tmp_path)
    assert inputs.subject_ids == ["S01", "S02", "S03"]
    assert inputs.labels.tolist() == [1, 0, 1]
    assert inputs.psd_eo.shape == (3, 2, 4, 5)
    assert inputs.fc_ec.shape == (3, 2, 3, 2, 2)


def test_load_matrixnet_inputs_fails_when_matrix_rows_do_not_match_subjects(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    np.save(tmp_path / "matrices" / "psd_eo.npy", np.ones((2, 2, 4, 5), dtype=np.float32))
    with pytest.raises(ValueError, match="psd_eo.npy first dimension"):
        load_matrixnet_inputs(tmp_path)


def test_validate_fold_registry_rejects_test_subject_in_fit_sets() -> None:
    registry = {
        "outer_fold": 1,
        "test_subject": "S01",
        "supervised_train_subjects": ["S02", "S03"],
        "inner_splits": [{"inner_fold": 1, "train_subjects": ["S01"], "val_subjects": ["S02"]}],
        "normalization_fit_subjects": ["S02", "S03"],
        "threshold_selection_subjects": ["S02", "S03"],
    }
    with pytest.raises(ValueError, match="outer test subject"):
        validate_fold_registry(registry)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_no_leakage.py -q
Pop-Location
```

Expected: FAIL with `ModuleNotFoundError: No module named 'stroke_predict.matrixnet_data'`.

- [ ] **Step 3: Implement data loader**

Create `src/stroke_predict/matrixnet_data.py` with:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LABEL_TO_INT = {"Poor": 0, "Good": 1}
MATRIX_FILES = ("psd_eo.npy", "psd_ec.npy", "fc_roi_eo.npy", "fc_roi_ec.npy")


@dataclass(frozen=True)
class MatrixNetInputs:
    output_dir: Path
    subject_ids: list[str]
    labels: np.ndarray
    label_names: list[str]
    cohort: pd.DataFrame
    outer_folds: dict[str, Any]
    registries: list[dict[str, Any]]
    psd_eo: np.ndarray
    psd_ec: np.ndarray
    fc_eo: np.ndarray
    fc_ec: np.ndarray
    tacs: pd.DataFrame | None
    clinical: pd.DataFrame
    ml_metrics: pd.DataFrame | None


def load_matrixnet_inputs(output_dir: str | Path) -> MatrixNetInputs:
    root = Path(output_dir)
    cohort = pd.read_csv(root / "cohort" / "cohort_master.csv")
    supervised = cohort.loc[cohort["role"].eq("supervised_main")].copy()
    supervised = supervised.sort_values("subject_id").reset_index(drop=True)
    if supervised.empty:
        raise ValueError("No supervised_main patients found")
    label_names = supervised["label_primary"].astype(str).tolist()
    unknown = sorted(set(label_names) - set(LABEL_TO_INT))
    if unknown:
        raise ValueError(f"Labels must be Good/Poor, found: {unknown}")
    subject_ids = supervised["subject_id"].astype(str).tolist()
    labels = np.asarray([LABEL_TO_INT[label] for label in label_names], dtype=np.int64)

    matrix_dir = _matrix_dir(root)
    matrices = {name: np.load(matrix_dir / name) for name in MATRIX_FILES}
    for name, array in matrices.items():
        if array.shape[0] != len(subject_ids):
            raise ValueError(f"{name} first dimension {array.shape[0]} does not match supervised_main {len(subject_ids)}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or Inf")

    outer_folds = json.loads((root / "folds" / "outer_folds.json").read_text(encoding="utf-8"))
    if sorted(map(str, outer_folds.get("supervised_subjects", []))) != sorted(subject_ids):
        raise ValueError("outer_folds supervised_subjects do not match matrix subject IDs")
    registries = []
    for fold in outer_folds["folds"]:
        registry = json.loads((root / "folds" / str(fold["registry_path"])).read_text(encoding="utf-8"))
        validate_fold_registry(registry)
        registries.append(registry)

    tacs_path = root / "features" / "features_tacs_target_summary.csv"
    tacs = pd.read_csv(tacs_path) if tacs_path.exists() else None
    clinical = _clinical_frame(supervised)
    ml_metrics_path = root / "evaluation" / "ml_metrics_all.csv"
    ml_metrics = pd.read_csv(ml_metrics_path) if ml_metrics_path.exists() else None
    return MatrixNetInputs(
        output_dir=root,
        subject_ids=subject_ids,
        labels=labels,
        label_names=label_names,
        cohort=cohort,
        outer_folds=outer_folds,
        registries=registries,
        psd_eo=matrices["psd_eo.npy"],
        psd_ec=matrices["psd_ec.npy"],
        fc_eo=matrices["fc_roi_eo.npy"],
        fc_ec=matrices["fc_roi_ec.npy"],
        tacs=tacs,
        clinical=clinical,
        ml_metrics=ml_metrics,
    )


def validate_fold_registry(registry: dict[str, Any]) -> None:
    test_subject = str(registry["test_subject"])
    train = set(map(str, registry.get("supervised_train_subjects", [])))
    if test_subject in train:
        raise ValueError("outer test subject appears in supervised_train_subjects")
    checked_keys = ("normalization_fit_subjects", "feature_selection_fit_subjects", "threshold_selection_subjects")
    for key in checked_keys:
        values = set(map(str, registry.get(key, [])))
        if test_subject in values:
            raise ValueError(f"outer test subject appears in {key}")
    for split in registry.get("inner_splits", []):
        for key in ("train_subjects", "val_subjects"):
            if test_subject in set(map(str, split.get(key, []))):
                raise ValueError(f"outer test subject appears in inner {key}")


def _matrix_dir(root: Path) -> Path:
    canonical = root / "matrices"
    legacy = root / "features" / "matrices"
    if all((canonical / name).exists() for name in MATRIX_FILES):
        return canonical
    if all((legacy / name).exists() for name in MATRIX_FILES):
        return legacy
    missing = [name for name in MATRIX_FILES if not (canonical / name).exists() and not (legacy / name).exists()]
    raise FileNotFoundError(f"Missing matrix inputs: {missing}")


def _clinical_frame(supervised: pd.DataFrame) -> pd.DataFrame:
    candidate_columns = [
        "subject_id",
        "baseline_fma",
        "baseline_mbi",
        "mmse",
        "age",
        "sex",
        "affected_hand",
        "treated_hand",
        "affected_side",
        "disease_duration",
        "disease_duration_days",
        "time_since_stroke",
    ]
    columns = [column for column in candidate_columns if column in supervised.columns]
    return supervised[columns].copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_no_leakage.py -q
Pop-Location
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```powershell
git -C .worktrees/phase6-matrixnet add tests/test_matrixnet_no_leakage.py src/stroke_predict/matrixnet_data.py
git -C .worktrees/phase6-matrixnet commit -m "feat: add MatrixNet input loader"
```

## Task 3: RED tests for fold-safe preprocessing

**Files:**
- Modify: `tests/test_matrixnet_no_leakage.py`
- Create later: `src/stroke_predict/matrixnet_preprocessing.py`

- [ ] **Step 1: Add failing preprocessing tests**

Append:

```python
from stroke_predict.matrixnet_preprocessing import FoldPreprocessor, fit_vector_preprocessor


def test_matrix_scaler_uses_only_outer_training_subjects() -> None:
    subject_ids = ["S01", "S02", "S03"]
    matrix = np.asarray([[[1.0, 1.0]], [[3.0, 3.0]], [[100.0, 100.0]]], dtype=np.float32)
    preprocessor = FoldPreprocessor.fit(subject_ids, train_subjects=["S01", "S02"], matrices={"psd": matrix})
    transformed = preprocessor.transform_matrix("psd", matrix)
    assert np.allclose(preprocessor.matrix_stats["psd"].mean, 2.0)
    assert np.allclose(preprocessor.matrix_stats["psd"].std, 1.0)
    assert np.allclose(transformed[0], -1.0)
    assert np.allclose(transformed[1], 1.0)
    assert np.allclose(transformed[2], 98.0)


def test_vector_preprocessor_imputes_and_scales_from_training_rows_only() -> None:
    frame = pd.DataFrame({"subject_id": ["S01", "S02", "S03"], "a": [1.0, 3.0, None], "b": [10.0, 14.0, 100.0]})
    processed = fit_vector_preprocessor(frame, subject_ids=["S01", "S02", "S03"], train_subjects=["S01", "S02"])
    values = processed.transform(frame, ["S01", "S02", "S03"])
    assert values.shape == (3, 2)
    assert np.isfinite(values).all()
    assert np.allclose(processed.medians["a"], 2.0)
    assert np.allclose(processed.means["b"], 12.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_no_leakage.py -q
Pop-Location
```

Expected: FAIL with `ModuleNotFoundError: No module named 'stroke_predict.matrixnet_preprocessing'`.

- [ ] **Step 3: Implement preprocessing**

Create `src/stroke_predict/matrixnet_preprocessing.py`:

```python
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

    def transform(self, frame: pd.DataFrame, subject_ids: list[str]) -> np.ndarray:
        aligned = _align(frame, subject_ids)
        columns: list[np.ndarray] = []
        for column in self.feature_columns:
            if column in self.categories:
                values = aligned[column].astype(str).fillna("__missing__")
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
    train = aligned[aligned["subject_id"].astype(str).isin(set(train_subjects))].copy()
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
            cats = tuple(sorted(train[column].astype(str).fillna("__missing__").unique().tolist()))
            categories[column] = cats
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
    indexed = frame.assign(subject_id=lambda value: value["subject_id"].astype(str)).drop_duplicates("subject_id").set_index("subject_id")
    missing = [subject for subject in subject_ids if subject not in indexed.index]
    if missing:
        raise ValueError(f"Vector frame missing subjects: {missing}")
    return indexed.loc[subject_ids].reset_index()
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_no_leakage.py -q
Pop-Location
```

Expected: all tests in file pass.

- [ ] **Step 5: Commit**

```powershell
git -C .worktrees/phase6-matrixnet add tests/test_matrixnet_no_leakage.py src/stroke_predict/matrixnet_preprocessing.py
git -C .worktrees/phase6-matrixnet commit -m "feat: add fold-safe MatrixNet preprocessing"
```

## Task 4: RED training smoke test

**Files:**
- Create: `tests/test_matrixnet_training_smoke.py`
- Create later: `src/stroke_predict/matrixnet_training.py`

- [ ] **Step 1: Write the failing smoke test**

Add:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stroke_predict.matrixnet_data import load_matrixnet_inputs
from stroke_predict.matrixnet_training import MatrixNetRunConfig, run_matrixnet_lopo

from tests.test_matrixnet_no_leakage import _write_minimal_inputs


def test_matrixnet_fast_smoke_produces_finite_predictions(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    inputs = load_matrixnet_inputs(tmp_path)
    config = MatrixNetRunConfig(
        run_mode="fast",
        models=["M8a_matrixnet_psd_only"],
        seeds=[0],
        max_epochs=3,
        patience=2,
        batch_size=2,
        learning_rates=[1e-3],
        weight_decays=[1e-2],
        dropouts=[0.3],
        embedding_dims=[8],
        hidden_dims=[16],
        fold_limit=2,
        write_outputs=False,
    )
    result = run_matrixnet_lopo(inputs, config)
    predictions = result.predictions
    assert len(predictions) == 2
    assert predictions["model_name"].eq("M8a_matrixnet_psd_only").all()
    assert predictions["predicted_score"].between(0, 1).all()
    assert np.isfinite(predictions["train_loss_final"]).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_training_smoke.py -q
Pop-Location
```

Expected: FAIL with `ModuleNotFoundError: No module named 'stroke_predict.matrixnet_training'`.

- [ ] **Step 3: Implement minimal training loop**

Create `src/stroke_predict/matrixnet_training.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from stroke_predict.matrixnet import MatrixNet, MatrixNetConfig
from stroke_predict.matrixnet_data import MatrixNetInputs
from stroke_predict.matrixnet_preprocessing import FoldPreprocessor, fit_vector_preprocessor

INT_TO_LABEL = {0: "Poor", 1: "Good"}


@dataclass(frozen=True)
class MatrixNetRunConfig:
    run_mode: str
    models: list[str]
    seeds: list[int]
    max_epochs: int
    patience: int
    batch_size: int
    learning_rates: list[float]
    weight_decays: list[float]
    dropouts: list[float]
    embedding_dims: list[int]
    hidden_dims: list[int]
    fold_limit: int | None = None
    write_outputs: bool = True


@dataclass(frozen=True)
class MatrixNetRunResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    training_log: pd.DataFrame
    fold_audit: pd.DataFrame


class MatrixDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, arrays: dict[str, np.ndarray], labels: np.ndarray, indices: list[int]) -> None:
        self.arrays = arrays
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = self.indices[item]
        sample = {name: torch.from_numpy(array[index]).float() for name, array in self.arrays.items()}
        sample["label"] = torch.tensor(float(self.labels[index]), dtype=torch.float32)
        return sample


def run_matrixnet_lopo(inputs: MatrixNetInputs, config: MatrixNetRunConfig) -> MatrixNetRunResult:
    rows: list[dict[str, object]] = []
    logs: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []
    subject_to_index = {subject: index for index, subject in enumerate(inputs.subject_ids)}
    folds = inputs.outer_folds["folds"][: config.fold_limit] if config.fold_limit else inputs.outer_folds["folds"]
    registry_by_fold = {int(registry["outer_fold"]): registry for registry in inputs.registries}
    for model_name in config.models:
        for seed in config.seeds:
            for fold in folds:
                outer_fold = int(fold["outer_fold"])
                registry = registry_by_fold[outer_fold]
                prediction, fold_logs, audit = _run_one_fold(model_name, seed, fold, registry, inputs, config, subject_to_index)
                rows.append(prediction)
                logs.extend(fold_logs)
                audits.append(audit)
    predictions = pd.DataFrame(rows)
    metrics = compute_matrixnet_metrics(predictions, inputs.ml_metrics)
    return MatrixNetRunResult(predictions=predictions, metrics=metrics, training_log=pd.DataFrame(logs), fold_audit=pd.DataFrame(audits))


def _run_one_fold(
    model_name: str,
    seed: int,
    fold: dict[str, Any],
    registry: dict[str, Any],
    inputs: MatrixNetInputs,
    config: MatrixNetRunConfig,
    subject_to_index: dict[str, int],
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    split = registry["inner_splits"][0]
    train_subjects = list(map(str, split["train_subjects"]))
    val_subjects = list(map(str, split["val_subjects"]))
    test_subject = str(fold["test_subject"])
    train_indices = [subject_to_index[subject] for subject in train_subjects]
    val_indices = [subject_to_index[subject] for subject in val_subjects]
    test_index = subject_to_index[test_subject]
    arrays, tacs_dim, clinical_dim = _prepared_arrays(model_name, inputs, registry["supervised_train_subjects"])
    model_config = _model_config(model_name, tacs_dim=tacs_dim, clinical_dim=clinical_dim, embedding_dim=config.embedding_dims[0], hidden_dim=config.hidden_dims[0], dropout=config.dropouts[0])
    model = MatrixNet(model_config)
    train_loader = DataLoader(MatrixDataset(arrays, inputs.labels, train_indices), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(MatrixDataset(arrays, inputs.labels, val_indices), batch_size=max(1, len(val_indices)), shuffle=False)
    pos = float(np.sum(inputs.labels[train_indices] == 1))
    neg = float(np.sum(inputs.labels[train_indices] == 0))
    pos_weight = torch.tensor([neg / pos], dtype=torch.float32) if pos > 0 else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rates[0], weight_decay=config.weight_decays[0])
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    wait = 0
    logs: list[dict[str, object]] = []
    for epoch in range(1, config.max_epochs + 1):
        train_loss = _train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_scores, val_true = _eval_epoch(model, val_loader, criterion)
        logs.append({"model_name": model_name, "outer_fold": fold["outer_fold"], "seed": seed, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "run_mode": config.run_mode})
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= config.patience:
                break
    model.load_state_dict(best_state)
    val_loss, val_scores, val_true = _eval_epoch(model, val_loader, criterion)
    threshold, threshold_source = _select_threshold(val_true, val_scores)
    test_loader = DataLoader(MatrixDataset(arrays, inputs.labels, [test_index]), batch_size=1, shuffle=False)
    _test_loss, test_scores, _test_true = _eval_epoch(model, test_loader, criterion)
    score = float(test_scores[0])
    pred_int = int(score >= threshold)
    prediction = {
        "model_name": model_name,
        "outer_fold": int(fold["outer_fold"]),
        "patient_id": test_subject,
        "true_label": INT_TO_LABEL[int(inputs.labels[test_index])],
        "predicted_score": score,
        "predicted_label": INT_TO_LABEL[pred_int],
        "threshold": float(threshold),
        "threshold_source": threshold_source,
        "seed": seed,
        "run_mode": config.run_mode,
        "input_family": _input_family(model_name),
        "best_epoch": best_epoch,
        "best_inner_metric": -float(best_loss),
        "train_loss_final": float(logs[-1]["train_loss"]),
        "val_loss_best": float(best_loss),
    }
    audit = {
        "model_name": model_name,
        "outer_fold": int(fold["outer_fold"]),
        "seed": seed,
        "test_patient": test_subject,
        "test_excluded_from_train": test_subject not in train_subjects,
        "test_excluded_from_val": test_subject not in val_subjects,
        "scaler_fit_subjects": ";".join(map(str, registry["supervised_train_subjects"])),
        "threshold_fit_subjects": ";".join(val_subjects),
        "run_mode": config.run_mode,
    }
    return prediction, logs, audit


def _prepared_arrays(model_name: str, inputs: MatrixNetInputs, train_subjects: list[str]) -> tuple[dict[str, np.ndarray], int, int]:
    pre = FoldPreprocessor.fit(
        inputs.subject_ids,
        list(map(str, train_subjects)),
        {"psd_eo": inputs.psd_eo, "psd_ec": inputs.psd_ec, "fc_eo": inputs.fc_eo, "fc_ec": inputs.fc_ec},
    )
    arrays: dict[str, np.ndarray] = {}
    if "psd" in model_name:
        arrays["psd_eo"] = pre.transform_matrix("psd_eo", inputs.psd_eo)
        arrays["psd_ec"] = pre.transform_matrix("psd_ec", inputs.psd_ec)
    if "fc" in model_name:
        arrays["fc_eo"] = pre.transform_matrix("fc_eo", inputs.fc_eo)
        arrays["fc_ec"] = pre.transform_matrix("fc_ec", inputs.fc_ec)
    tacs_dim = 0
    if "tacs" in model_name and inputs.tacs is not None:
        tacs_pre = fit_vector_preprocessor(inputs.tacs, subject_ids=inputs.subject_ids, train_subjects=list(map(str, train_subjects)))
        arrays["tacs"] = tacs_pre.transform(inputs.tacs, inputs.subject_ids)
        tacs_dim = arrays["tacs"].shape[1]
    clinical_dim = 0
    if "clinical" in model_name:
        clinical_pre = fit_vector_preprocessor(inputs.clinical, subject_ids=inputs.subject_ids, train_subjects=list(map(str, train_subjects)))
        arrays["clinical"] = clinical_pre.transform(inputs.clinical, inputs.subject_ids)
        clinical_dim = arrays["clinical"].shape[1]
    return arrays, tacs_dim, clinical_dim


def _model_config(model_name: str, *, tacs_dim: int, clinical_dim: int, embedding_dim: int, hidden_dim: int, dropout: float) -> MatrixNetConfig:
    return MatrixNetConfig(
        use_psd="psd" in model_name,
        use_fc="fc" in model_name,
        use_tacs="tacs" in model_name,
        use_clinical="clinical" in model_name,
        tacs_dim=tacs_dim,
        clinical_dim=clinical_dim,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )


def _train_epoch(model: MatrixNet, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    losses = []
    for batch in loader:
        labels = batch.pop("label")
        optimizer.zero_grad()
        logits = model(**batch)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _eval_epoch(model: MatrixNet, loader: DataLoader, criterion: nn.Module) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    losses: list[float] = []
    scores: list[float] = []
    labels_all: list[int] = []
    with torch.no_grad():
        for batch in loader:
            labels = batch.pop("label")
            logits = model(**batch)
            loss = criterion(logits, labels)
            losses.append(float(loss.detach().cpu()))
            scores.extend(torch.sigmoid(logits).detach().cpu().numpy().astype(float).tolist())
            labels_all.extend(labels.detach().cpu().numpy().astype(int).tolist())
    return float(np.mean(losses)), np.asarray(scores, dtype=float), np.asarray(labels_all, dtype=int)


def _select_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, str]:
    if len(set(y_true.tolist())) < 2:
        return 0.5, "fixed_0.5"
    thresholds = np.unique(np.concatenate([scores, np.asarray([0.5])]))
    best_threshold = 0.5
    best_score = -1.0
    for threshold in thresholds:
        pred = (scores >= threshold).astype(int)
        sensitivity = np.mean(pred[y_true == 1] == 1) if np.any(y_true == 1) else np.nan
        specificity = np.mean(pred[y_true == 0] == 0) if np.any(y_true == 0) else np.nan
        balanced = float((sensitivity + specificity) / 2)
        if balanced > best_score:
            best_score = balanced
            best_threshold = float(threshold)
    return best_threshold, "inner_validation_balanced_accuracy"


def compute_matrixnet_metrics(predictions: pd.DataFrame, ml_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (model_name, seed), group in predictions.groupby(["model_name", "seed"], sort=True):
        y_true = (group["true_label"].astype(str) == "Good").astype(int).to_numpy()
        scores = group["predicted_score"].astype(float).to_numpy()
        pred = (group["predicted_label"].astype(str) == "Good").astype(int).to_numpy()
        roc_auc = float(roc_auc_score(y_true, scores)) if len(set(y_true.tolist())) == 2 else np.nan
        rows.append({
            "model_name": model_name,
            "input_family": _input_family(str(model_name)),
            "run_mode": str(group["run_mode"].iloc[0]),
            "n_patients": int(len(group)),
            "n_good": int(np.sum(y_true == 1)),
            "n_poor": int(np.sum(y_true == 0)),
            "n_seeds": 1,
            "roc_auc_mean": roc_auc,
            "roc_auc_std_across_seeds": np.nan,
            "roc_auc_ci_low": np.nan,
            "roc_auc_ci_high": np.nan,
            "pr_auc": np.nan,
            "balanced_accuracy": float((np.mean(pred[y_true == 1] == 1) + np.mean(pred[y_true == 0] == 0)) / 2) if len(set(y_true.tolist())) == 2 else np.nan,
            "sensitivity": float(np.mean(pred[y_true == 1] == 1)) if np.any(y_true == 1) else np.nan,
            "specificity": float(np.mean(pred[y_true == 0] == 0)) if np.any(y_true == 0) else np.nan,
            "f1": np.nan,
            "brier_score": float(np.mean((scores - y_true) ** 2)),
            "permutation_p_value": np.nan,
            "comparison_to_best_ml_auc": _compare_ml(roc_auc, ml_metrics, None),
            "comparison_to_fma_only_auc": _compare_ml(roc_auc, ml_metrics, "M1_fma_only"),
            "comparison_to_clinical_only_auc": _compare_ml(roc_auc, ml_metrics, "M2_clinical_only"),
        })
    return pd.DataFrame(rows)


def _compare_ml(auc: float, ml_metrics: pd.DataFrame | None, model_name: str | None) -> float:
    if ml_metrics is None or np.isnan(auc):
        return np.nan
    if model_name is None:
        baseline = float(ml_metrics["roc_auc"].max())
    else:
        rows = ml_metrics[ml_metrics["model_name"].astype(str).eq(model_name)]
        if rows.empty:
            return np.nan
        baseline = float(rows.iloc[0]["roc_auc"])
    return float(auc - baseline)


def _input_family(model_name: str) -> str:
    if model_name == "M8a_matrixnet_psd_only":
        return "psd_only"
    if model_name == "M8b_matrixnet_fc_only":
        return "fc_only"
    if model_name == "M8c_matrixnet_psd_fc":
        return "psd_fc"
    if model_name == "M8d_matrixnet_psd_fc_tacs":
        return "psd_fc_tacs"
    if model_name == "M12_matrixnet_clinical_eeg":
        return "clinical_eeg"
    return "unknown"
```

- [ ] **Step 4: Run smoke test**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_training_smoke.py -q
Pop-Location
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git -C .worktrees/phase6-matrixnet add tests/test_matrixnet_training_smoke.py src/stroke_predict/matrixnet_training.py
git -C .worktrees/phase6-matrixnet commit -m "feat: add MatrixNet training smoke path"
```

## Task 5: RED tests for output files and no-duplicate predictions

**Files:**
- Create: `tests/test_matrixnet_outputs.py`
- Modify later: `src/stroke_predict/matrixnet_training.py`

- [ ] **Step 1: Write failing output tests**

Add:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd

from stroke_predict.matrixnet_data import load_matrixnet_inputs
from stroke_predict.matrixnet_training import MatrixNetRunConfig, run_matrixnet_lopo, write_matrixnet_outputs

from tests.test_matrixnet_no_leakage import _write_minimal_inputs


def test_write_matrixnet_outputs_creates_required_files_and_columns(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    inputs = load_matrixnet_inputs(tmp_path)
    config = MatrixNetRunConfig(
        run_mode="fast",
        models=["M8a_matrixnet_psd_only"],
        seeds=[0],
        max_epochs=2,
        patience=1,
        batch_size=2,
        learning_rates=[1e-3],
        weight_decays=[1e-2],
        dropouts=[0.3],
        embedding_dims=[8],
        hidden_dims=[16],
        write_outputs=True,
    )
    result = run_matrixnet_lopo(inputs, config)
    write_matrixnet_outputs(tmp_path, result, config)

    predictions_path = tmp_path / "predictions" / "matrixnet_patient_predictions.csv"
    metrics_path = tmp_path / "evaluation" / "matrixnet_metrics.csv"
    report_path = tmp_path / "reports" / "phase6_matrixnet_report.md"
    audit_path = tmp_path / "reports" / "matrixnet_fold_audit.csv"
    assert predictions_path.exists()
    assert metrics_path.exists()
    assert report_path.exists()
    assert audit_path.exists()

    predictions = pd.read_csv(predictions_path)
    assert {
        "model_name",
        "outer_fold",
        "patient_id",
        "true_label",
        "predicted_score",
        "predicted_label",
        "threshold",
        "threshold_source",
        "seed",
        "run_mode",
        "input_family",
        "best_epoch",
        "best_inner_metric",
        "train_loss_final",
        "val_loss_best",
    } <= set(predictions.columns)
    assert not predictions.duplicated(["model_name", "patient_id", "seed"]).any()
    assert predictions.groupby(["model_name", "seed"])["patient_id"].nunique().eq(3).all()

    metrics = pd.read_csv(metrics_path)
    assert {
        "model_name",
        "input_family",
        "run_mode",
        "n_patients",
        "n_good",
        "n_poor",
        "n_seeds",
        "roc_auc_mean",
        "roc_auc_std_across_seeds",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "pr_auc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "brier_score",
        "permutation_p_value",
        "comparison_to_best_ml_auc",
        "comparison_to_fma_only_auc",
        "comparison_to_clinical_only_auc",
    } <= set(metrics.columns)
    assert "supervised no-SSL" in report_path.read_text(encoding="utf-8")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_outputs.py -q
Pop-Location
```

Expected: FAIL with `ImportError` for `write_matrixnet_outputs`.

- [ ] **Step 3: Implement output writers**

Add to `src/stroke_predict/matrixnet_training.py`:

```python
def write_matrixnet_outputs(output_dir: str | Path, result: MatrixNetRunResult, config: MatrixNetRunConfig) -> dict[str, str]:
    root = Path(output_dir)
    paths = {
        "predictions": root / "predictions" / "matrixnet_patient_predictions.csv",
        "metrics": root / "evaluation" / "matrixnet_metrics.csv",
        "training_log": root / "matrixnet" / "training_log.csv",
        "config_used": root / "matrixnet" / "config_used.yaml",
        "fold_audit": root / "reports" / "matrixnet_fold_audit.csv",
        "no_leakage_report": root / "reports" / "matrixnet_no_leakage_report.txt",
        "report": root / "reports" / "phase6_matrixnet_report.md",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    result.predictions.sort_values(["model_name", "seed", "outer_fold"]).to_csv(paths["predictions"], index=False)
    result.metrics.sort_values(["model_name"]).to_csv(paths["metrics"], index=False)
    result.training_log.to_csv(paths["training_log"], index=False)
    result.fold_audit.to_csv(paths["fold_audit"], index=False)
    paths["config_used"].write_text(_config_snapshot(config), encoding="utf-8")
    paths["no_leakage_report"].write_text(_no_leakage_text(result), encoding="utf-8")
    paths["report"].write_text(_phase6_report(result, config), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def _config_snapshot(config: MatrixNetRunConfig) -> str:
    lines = ["run_mode: " + config.run_mode, "models:"]
    lines.extend([f"  - {model}" for model in config.models])
    lines.append("seeds: [" + ", ".join(map(str, config.seeds)) + "]")
    lines.append(f"max_epochs: {config.max_epochs}")
    lines.append(f"patience: {config.patience}")
    return "\n".join(lines) + "\n"


def _no_leakage_text(result: MatrixNetRunResult) -> str:
    audit = result.fold_audit
    checks = [
        ("test patient excluded from train", bool(audit["test_excluded_from_train"].all()) if not audit.empty else False),
        ("test patient excluded from val", bool(audit["test_excluded_from_val"].all()) if not audit.empty else False),
        ("no duplicated model-patient-seed predictions", not result.predictions.duplicated(["model_name", "patient_id", "seed"]).any()),
    ]
    return "\n".join([f"{'PASS' if ok else 'FAIL'}: {name}" for name, ok in checks]) + "\n"


def _phase6_report(result: MatrixNetRunResult, config: MatrixNetRunConfig) -> str:
    metrics_table = result.metrics.to_markdown(index=False) if not result.metrics.empty else "_No metrics._"
    return "\n".join(
        [
            "# Phase 6 MatrixNet Report",
            "",
            f"Run mode: **{config.run_mode}**",
            "",
            "Phase 6 is supervised no-SSL MatrixNet. No self-supervised pretraining was started.",
            "",
            "Fast mode is an engineering and smoke-test setting. Do not claim EEG efficacy from fast-mode results.",
            "",
            "## MatrixNet performance",
            "",
            metrics_table,
            "",
            "## Recommendation for Phase 7",
            "",
            "After full multi-seed supervised MatrixNet is stable, Phase 7 can evaluate SSL-MatrixNet under the existing leakage checks.",
        ]
    ) + "\n"
```

- [ ] **Step 4: Run output tests**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_outputs.py -q
Pop-Location
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git -C .worktrees/phase6-matrixnet add tests/test_matrixnet_outputs.py src/stroke_predict/matrixnet_training.py
git -C .worktrees/phase6-matrixnet commit -m "feat: add MatrixNet output writers"
```

## Task 6: Add config and CLI script

**Files:**
- Create: `configs/matrixnet.yaml`
- Create: `scripts/09_train_matrixnet.py`
- Modify: `tests/test_matrixnet_outputs.py`

- [ ] **Step 1: Add failing script smoke test**

Append to `tests/test_matrixnet_outputs.py`:

```python
import subprocess
import sys


def test_matrixnet_script_fast_mode_with_fold_limit(tmp_path: Path) -> None:
    _write_minimal_inputs(tmp_path)
    config_path = tmp_path / "matrixnet.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"output_dir: {tmp_path.as_posix()}",
                "run_modes:",
                "  fast:",
                "    seeds: [0]",
                "    max_epochs: 2",
                "    patience: 1",
                "    batch_size: 2",
                "    learning_rates: [0.001]",
                "    weight_decays: [0.01]",
                "    dropouts: [0.3]",
                "    embedding_dims: [8]",
                "    hidden_dims: [16]",
                "models:",
                "  fast:",
                "    - M8a_matrixnet_psd_only",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/09_train_matrixnet.py",
            "--config",
            str(config_path),
            "--run-mode",
            "fast",
            "--fold-limit",
            "2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "MATRIXNET_OK" in completed.stdout
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_outputs.py::test_matrixnet_script_fast_mode_with_fold_limit -q
Pop-Location
```

Expected: FAIL because `scripts/09_train_matrixnet.py` does not exist.

- [ ] **Step 3: Add config and script**

Create `configs/matrixnet.yaml`:

```yaml
output_dir: outputs
primary_label: label_primary
run_modes:
  fast:
    seeds: [0]
    max_epochs: 30
    patience: 5
    batch_size: 4
    learning_rates: [0.001]
    weight_decays: [0.01]
    dropouts: [0.5]
    embedding_dims: [32]
    hidden_dims: [64]
  full:
    seeds: [0, 1, 2, 3, 4]
    max_epochs: 200
    patience: 25
    batch_size: 4
    learning_rates: [0.001, 0.0003, 0.0001]
    weight_decays: [0.01, 0.001, 0.0001]
    dropouts: [0.3, 0.5]
    embedding_dims: [32, 64]
    hidden_dims: [64, 128]
models:
  fast:
    - M8a_matrixnet_psd_only
    - M8b_matrixnet_fc_only
    - M8c_matrixnet_psd_fc
    - M8d_matrixnet_psd_fc_tacs
    - M12_matrixnet_clinical_eeg
  full:
    - M8a_matrixnet_psd_only
    - M8b_matrixnet_fc_only
    - M8c_matrixnet_psd_fc
    - M8d_matrixnet_psd_fc_tacs
    - M12_matrixnet_clinical_eeg
preprocessing:
  psd_transform: none
  matrix_normalization: zscore
  vector_imputation: train_median
  vector_scaling: train_standard
outputs:
  predictions: outputs/predictions/matrixnet_patient_predictions.csv
  metrics: outputs/evaluation/matrixnet_metrics.csv
  report: outputs/reports/phase6_matrixnet_report.md
```

Create `scripts/09_train_matrixnet.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_yaml_mapping
from stroke_predict.matrixnet_data import load_matrixnet_inputs
from stroke_predict.matrixnet_training import MatrixNetRunConfig, run_matrixnet_lopo, write_matrixnet_outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-mode", choices=["fast", "full"], default="fast")
    parser.add_argument("--fold-limit", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    raw = load_yaml_mapping(config_path)
    output_dir = _resolve(config_path, str(raw.get("output_dir", "outputs")))
    mode = raw["run_modes"][args.run_mode]
    run_config = MatrixNetRunConfig(
        run_mode=args.run_mode,
        models=[str(value) for value in raw["models"][args.run_mode]],
        seeds=[int(value) for value in mode["seeds"]],
        max_epochs=int(mode["max_epochs"]),
        patience=int(mode["patience"]),
        batch_size=int(mode["batch_size"]),
        learning_rates=[float(value) for value in mode["learning_rates"]],
        weight_decays=[float(value) for value in mode["weight_decays"]],
        dropouts=[float(value) for value in mode["dropouts"]],
        embedding_dims=[int(value) for value in mode["embedding_dims"]],
        hidden_dims=[int(value) for value in mode["hidden_dims"]],
        fold_limit=args.fold_limit,
        write_outputs=True,
    )
    inputs = load_matrixnet_inputs(output_dir)
    result = run_matrixnet_lopo(inputs, run_config)
    paths = write_matrixnet_outputs(output_dir, result, run_config)
    print("MATRIXNET_OK")
    print(f"run_mode={args.run_mode}")
    print(f"n_predictions={len(result.predictions)}")
    print(f"predictions={paths['predictions']}")
    print(f"metrics={paths['metrics']}")
    print(f"report={paths['report']}")
    return 0


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent.parent / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run script test**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_outputs.py::test_matrixnet_script_fast_mode_with_fold_limit -q
Pop-Location
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```powershell
git -C .worktrees/phase6-matrixnet add configs/matrixnet.yaml scripts/09_train_matrixnet.py tests/test_matrixnet_outputs.py
git -C .worktrees/phase6-matrixnet commit -m "feat: add MatrixNet training CLI"
```

## Task 7: Metrics, report quality, and Phase 5.2 comparison

**Files:**
- Modify: `src/stroke_predict/matrixnet_training.py`
- Modify: `tests/test_matrixnet_outputs.py`

- [ ] **Step 1: Add failing metrics assertions**

Extend `test_write_matrixnet_outputs_creates_required_files_and_columns`:

```python
    assert metrics["comparison_to_best_ml_auc"].notna().all() or inputs.ml_metrics is None
    report_text = report_path.read_text(encoding="utf-8")
    assert "Phase 5.2" in report_text
    assert "flattened" in report_text.lower()
    assert "Do not claim EEG efficacy" in report_text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_outputs.py -q
Pop-Location
```

Expected: FAIL because report lacks detailed comparison text.

- [ ] **Step 3: Improve metrics/report**

Update `compute_matrixnet_metrics` to aggregate across seeds by model:

```python
def compute_matrixnet_metrics(predictions: pd.DataFrame, ml_metrics: pd.DataFrame | None = None) -> pd.DataFrame:
    seed_rows = []
    for (model_name, seed), group in predictions.groupby(["model_name", "seed"], sort=True):
        seed_rows.append(_seed_metric_row(str(model_name), int(seed), group, ml_metrics))
    seed_frame = pd.DataFrame(seed_rows)
    rows = []
    for model_name, group in seed_frame.groupby("model_name", sort=True):
        first = group.iloc[0]
        rows.append({
            "model_name": model_name,
            "input_family": first["input_family"],
            "run_mode": first["run_mode"],
            "n_patients": int(first["n_patients"]),
            "n_good": int(first["n_good"]),
            "n_poor": int(first["n_poor"]),
            "n_seeds": int(group["seed"].nunique()),
            "roc_auc_mean": float(group["roc_auc"].mean()),
            "roc_auc_std_across_seeds": float(group["roc_auc"].std(ddof=0)) if len(group) > 1 else np.nan,
            "roc_auc_ci_low": np.nan,
            "roc_auc_ci_high": np.nan,
            "pr_auc": float(group["pr_auc"].mean()),
            "balanced_accuracy": float(group["balanced_accuracy"].mean()),
            "sensitivity": float(group["sensitivity"].mean()),
            "specificity": float(group["specificity"].mean()),
            "f1": float(group["f1"].mean()),
            "brier_score": float(group["brier_score"].mean()),
            "permutation_p_value": np.nan,
            "comparison_to_best_ml_auc": float(group["comparison_to_best_ml_auc"].mean()),
            "comparison_to_fma_only_auc": float(group["comparison_to_fma_only_auc"].mean()),
            "comparison_to_clinical_only_auc": float(group["comparison_to_clinical_only_auc"].mean()),
        })
    return pd.DataFrame(rows)
```

Add helper `_seed_metric_row` using `average_precision_score`, `f1_score`, `confusion_matrix`, and update `_phase6_report` to include:

- Phase 6 objective
- input artifact audit summary
- model architecture summary
- parameter counts if available
- training settings
- LOPO/no-leakage summary
- MatrixNet performance table
- Phase 5.2 comparison text
- supervised no-SSL statement
- caution about efficacy
- Phase 7 recommendation

- [ ] **Step 4: Run tests**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_outputs.py -q
Pop-Location
```

Expected: all output tests pass.

- [ ] **Step 5: Commit**

```powershell
git -C .worktrees/phase6-matrixnet add src/stroke_predict/matrixnet_training.py tests/test_matrixnet_outputs.py
git -C .worktrees/phase6-matrixnet commit -m "feat: compare MatrixNet with phase 5 baselines"
```

## Task 8: Full test suite and real-data fast acceptance

**Files:**
- No production code unless tests expose a bug.

- [ ] **Step 1: Run all tests**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider
Pop-Location
```

Expected: all tests pass.

- [ ] **Step 2: Prepare ignored real-data artifacts inside worktree if needed**

If `.worktrees/phase6-matrixnet/outputs` is absent, copy only required ignored artifacts from the main checkout:

```powershell
New-Item -ItemType Directory -Force -Path .worktrees/phase6-matrixnet/outputs | Out-Null
Copy-Item -Recurse -Force outputs/cohort .worktrees/phase6-matrixnet/outputs/
Copy-Item -Recurse -Force outputs/folds .worktrees/phase6-matrixnet/outputs/
Copy-Item -Recurse -Force outputs/matrices .worktrees/phase6-matrixnet/outputs/
Copy-Item -Recurse -Force outputs/features .worktrees/phase6-matrixnet/outputs/
Copy-Item -Recurse -Force outputs/evaluation .worktrees/phase6-matrixnet/outputs/
Copy-Item -Recurse -Force outputs/reports .worktrees/phase6-matrixnet/outputs/
```

Do not stage copied outputs.

- [ ] **Step 3: Run real-data fast mode**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python scripts/09_train_matrixnet.py --config configs/matrixnet.yaml --run-mode fast
Pop-Location
```

Expected stdout contains:

```text
MATRIXNET_OK
run_mode=fast
```

Expected predictions:

- `M8a_matrixnet_psd_only`: 19 rows for seed 0
- `M8b_matrixnet_fc_only`: 19 rows for seed 0
- `M8c_matrixnet_psd_fc`: 19 rows for seed 0
- `M8d_matrixnet_psd_fc_tacs`: 19 rows for seed 0
- `M12_matrixnet_clinical_eeg`: 19 rows for seed 0 if clinical vector is available

- [ ] **Step 4: Validate real-data outputs**

Run:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests/test_matrixnet_outputs.py tests/test_matrixnet_no_leakage.py tests/test_matrixnet_training_smoke.py tests/test_matrixnet_shapes.py -q
Pop-Location
```

Then inspect:

```powershell
Import-Csv .worktrees/phase6-matrixnet/outputs/predictions/matrixnet_patient_predictions.csv |
  Group-Object model_name,seed |
  Select-Object Name,Count
```

Expected: each completed model/seed count is 19.

- [ ] **Step 5: Verify no forbidden files are staged**

Run:

```powershell
git -C .worktrees/phase6-matrixnet status --short
```

Expected: no `outputs/`, `.xlsx`, `.set`, `.fdt` staged or tracked.

- [ ] **Step 6: Commit final fixes if any**

If verification found fixes:

```powershell
git -C .worktrees/phase6-matrixnet add <code-and-test-files-only>
git -C .worktrees/phase6-matrixnet commit -m "fix: harden MatrixNet phase 6 acceptance"
```

## Task 9: Finish branch

**Files:**
- No file edits expected.

- [ ] **Step 1: Use verification-before-completion**

Run fresh:

```powershell
Push-Location .worktrees/phase6-matrixnet
python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider
python scripts/09_train_matrixnet.py --config configs/matrixnet.yaml --run-mode fast
Pop-Location
```

Expected: tests pass and real-data fast mode writes required Phase 6 outputs.

- [ ] **Step 2: Confirm branch diff excludes forbidden artifacts**

Run:

```powershell
git -C .worktrees/phase6-matrixnet diff --stat main...HEAD
git -C .worktrees/phase6-matrixnet status --short
```

Expected: only code/config/docs/tests are committed or unstaged; no outputs/raw files.

- [ ] **Step 3: Merge and push only after approval**

After user approves implementation completion:

```powershell
git checkout main
git pull --ff-only
git merge codex/phase-6-matrixnet
python -m pytest tests --basetemp=.codex_pytest_tmp -p no:cacheprovider
git push origin main
```

Do not merge or push if tests fail, if forbidden artifacts appear, or if user has not approved proceeding past the Superpowers spec/plan gate.

## Self-Review

- Spec coverage: covers data loading, fold-safe normalization, MatrixNet architecture, fast/full modes, required outputs, Phase 5.2 comparison, tests, real-data acceptance, no SSL.
- Placeholder scan: no TBD/TODO placeholders remain.
- Type consistency: planned class/function names are consistent across tests and implementation snippets.
- Scope check: implementation remains Phase 6 supervised no-SSL; Phase 7 is only mentioned as recommendation.
