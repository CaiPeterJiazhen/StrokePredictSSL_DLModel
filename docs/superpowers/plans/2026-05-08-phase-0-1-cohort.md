# Phase 0+1 Cohort And Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 Phase 0+1 的可复现队列与标签流水线，生成去标识化 `cohort_master.csv`、`label_audit.csv`、标签分布 JSON 和标签分布图。

**Architecture:** 采用小型 Python 包 `stroke_predict`，把配置读取、Excel 读取、标签规则、匿名 ID、隐私检查、队列构建和输出写入拆成独立模块。脚本层只负责参数解析和调用模块，测试使用 synthetic fixture 先覆盖行为，再跑真实数据验收。

**Tech Stack:** Python 3.12, pandas, openpyxl, Pillow, pytest, Git, Windows PowerShell.

---

## 执行前要求

开始实现前必须使用 `superpowers:using-git-worktrees`。当前仓库已经初始化 Git，执行实现计划时先检测是否已在隔离 worktree；如果不是，创建 Phase 0+1 专用 worktree 分支，例如 `phase-0-1-cohort`。

推荐执行方式是 `superpowers:subagent-driven-development`，每个 Task 由一个实现 subagent 完成，并经过 spec compliance review 和 code quality review。

## 文件结构

本计划会创建或修改以下文件：

- Create: `pyproject.toml`，定义 pytest 配置和本地包元数据。
- Create: `requirements.txt`，记录运行依赖，不在计划中联网安装。
- Create: `README_dev.md`，写明 Phase 0+1 运行方式和隐私边界。
- Create: `configs/paths.yaml`，保存本机输入路径和输出目录。
- Create: `configs/project.yaml`，保存 sheet 名、标签参数、隐私字段和输出文件名。
- Create: `src/stroke_predict/__init__.py`。
- Create: `src/stroke_predict/config.py`，读取项目配置，支持 PyYAML 和本项目需要的简单 YAML fallback。
- Create: `src/stroke_predict/privacy.py`，检查和移除公开输出中的 PII 字段。
- Create: `src/stroke_predict/io/__init__.py`。
- Create: `src/stroke_predict/io/excel_status.py`，读取 Excel 并验证 sheet/column。
- Create: `src/stroke_predict/cohort/__init__.py`。
- Create: `src/stroke_predict/cohort/labels.py`，实现主标签和敏感性标签。
- Create: `src/stroke_predict/cohort/ids.py`，生成确定性匿名 ID。
- Create: `src/stroke_predict/cohort/build.py`，合并临床和 EEG 元数据并分配 role。
- Create: `src/stroke_predict/cohort/outputs.py`，写出 CSV/JSON/PNG。
- Create: `scripts/00_validate_environment.py`。
- Create: `scripts/01_build_cohort.py`。
- Create: `tests/conftest.py`。
- Create: `tests/test_config.py`。
- Create: `tests/test_labels.py`。
- Create: `tests/test_privacy_ids.py`。
- Create: `tests/test_excel_status.py`。
- Create: `tests/test_cohort_build.py`。
- Create: `tests/test_scripts.py`。

## Task 1: 项目骨架、配置文件和配置读取

**Files:**
- Create: `pyproject.toml`
- Create: `requirements.txt`
- Create: `README_dev.md`
- Create: `configs/paths.yaml`
- Create: `configs/project.yaml`
- Create: `src/stroke_predict/__init__.py`
- Create: `src/stroke_predict/config.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: 写失败测试**

Create `tests/conftest.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

Create `tests/test_config.py`:

```python
from __future__ import annotations

from pathlib import Path

from stroke_predict.config import load_project_config


def test_loads_project_and_paths_config(tmp_path: Path) -> None:
    workbook = tmp_path / "status.xlsx"
    stroke_root = tmp_path / "stroke_eeg"
    healthy_root = tmp_path / "healthy_eeg"
    workbook.write_bytes(b"xlsx placeholder")
    stroke_root.mkdir()
    healthy_root.mkdir()

    paths_yaml = tmp_path / "paths.yaml"
    paths_yaml.write_text(
        "\n".join(
            [
                "paths:",
                f"  workbook: \"{workbook.as_posix()}\"",
                f"  stroke_eeg_root: \"{stroke_root.as_posix()}\"",
                f"  healthy_eeg_root: \"{healthy_root.as_posix()}\"",
                "  output_dir: outputs",
            ]
        ),
        encoding="utf-8",
    )
    project_yaml = tmp_path / "project.yaml"
    project_yaml.write_text(
        "\n".join(
            [
                "paths_config: paths.yaml",
                "sheets:",
                "  clinical_overview: 01_患者数据总览",
                "  clinical_raw: 03_临床量表原始",
                "  preprocessed_summary: 06_预处理静息态阶段汇总",
                "  preprocessed_files: 07_预处理静息态文件明细",
                "labels:",
                "  fma_full_score: 66",
                "  low_fma_threshold: 61",
                "  low_fma_delta_good: 5",
                "  near_ceiling_delta_good: 3",
                "privacy:",
                "  pii_columns:",
                "    - 姓名",
                "    - subject_name",
                "outputs:",
                "  cohort_dir: cohort",
                "  figures_dir: figures",
            ]
        ),
        encoding="utf-8",
    )

    config = load_project_config(project_yaml)

    assert config.project_path == project_yaml
    assert config.workbook_path == workbook
    assert config.stroke_eeg_root == stroke_root
    assert config.healthy_eeg_root == healthy_root
    assert config.output_dir == tmp_path / "outputs"
    assert config.sheet("clinical_overview") == "01_患者数据总览"
    assert config.label_setting("fma_full_score") == 66
    assert "姓名" in config.pii_columns
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: FAIL，原因是 `ModuleNotFoundError: No module named 'stroke_predict.config'`。

- [ ] **Step 3: 写最小实现和项目文件**

Create `pyproject.toml`:

```toml
[project]
name = "stroke-predict-ssl-dlmodel"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-q"
```

Create `requirements.txt`:

```text
pandas
openpyxl
Pillow
pytest
```

Create `README_dev.md`:

```markdown
# StrokePredictSSL-DLModel 开发说明

本仓库只提交代码、配置、文档和去标识化输出。原始 Excel、`.set`、`.fdt` 和运行输出均不提交。

Phase 0+1 常用命令：

```bash
python scripts/00_validate_environment.py
python scripts/01_build_cohort.py --config configs/project.yaml
python -m pytest tests -q
```
```

Create `configs/paths.yaml`:

```yaml
paths:
  workbook: "F:/CJZProjectFile/StrokePredictSSL-DLModel/current_data_status_overview_data_only.xlsx"
  stroke_eeg_root: "F:/CJZFile/EEG_M1/Patient_tACS_M1_RestingStateEEG_afterProcess"
  healthy_eeg_root: "F:/CJZFile/EEG_M1/Health_tACS_M1_RestingStateEEG_afterProcess"
  output_dir: "outputs"
```

Create `configs/project.yaml`:

```yaml
paths_config: "paths.yaml"
sheets:
  summary: "02_统计汇总"
  clinical_overview: "01_患者数据总览"
  clinical_raw: "03_临床量表原始"
  preprocessed_summary: "06_预处理静息态阶段汇总"
  preprocessed_files: "07_预处理静息态文件明细"
labels:
  fma_full_score: 66
  low_fma_threshold: 61
  low_fma_delta_good: 5
  near_ceiling_delta_good: 3
  proportional_good_threshold: 0.70
privacy:
  pii_columns:
    - "姓名"
    - "姓名写法"
    - "EEG文件夹"
    - "subject_name"
    - "set_path"
    - "fdt_path"
outputs:
  cohort_dir: "cohort"
  figures_dir: "figures"
```

Create `src/stroke_predict/__init__.py`:

```python
"""Utilities for leakage-controlled stroke EEG recovery prediction."""
```

Create `src/stroke_predict/config.py` with these public objects:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    project_path: Path
    project_root: Path
    paths_config_path: Path
    workbook_path: Path
    stroke_eeg_root: Path
    healthy_eeg_root: Path
    output_dir: Path
    raw: dict[str, Any]
    paths_raw: dict[str, Any]

    def sheet(self, key: str) -> str:
        return str(self.raw["sheets"][key])

    def label_setting(self, key: str) -> Any:
        return self.raw["labels"][key]

    @property
    def pii_columns(self) -> list[str]:
        return [str(value) for value in self.raw["privacy"]["pii_columns"]]

    def output_subdir(self, key: str) -> Path:
        return self.output_dir / str(self.raw["outputs"][key])


def load_project_config(config_path: str | Path) -> ProjectConfig:
    project_path = Path(config_path).resolve()
    project_root = project_path.parent.parent if project_path.parent.name == "configs" else project_path.parent
    raw = load_yaml_mapping(project_path)
    paths_config_name = str(raw.get("paths_config", "paths.yaml"))
    paths_config_path = (project_path.parent / paths_config_name).resolve()
    paths_raw = load_yaml_mapping(paths_config_path)
    paths = paths_raw["paths"]

    def resolve_path(value: str) -> Path:
        path = Path(str(value))
        if path.is_absolute():
            return path
        return (project_root / path).resolve()

    return ProjectConfig(
        project_path=project_path,
        project_root=project_root,
        paths_config_path=paths_config_path,
        workbook_path=resolve_path(paths["workbook"]),
        stroke_eeg_root=resolve_path(paths["stroke_eeg_root"]),
        healthy_eeg_root=resolve_path(paths["healthy_eeg_root"]),
        output_dir=resolve_path(paths["output_dir"]),
        raw=raw,
        paths_raw=paths_raw,
    )


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _load_simple_yaml(path)

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def _load_simple_yaml(path: Path) -> dict[str, Any]:
    return _load_simple_yaml_by_structure(path)


def _load_simple_yaml_by_structure(path: Path) -> dict[str, Any]:
    lines = [
        line.rstrip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    index = 0

    def parse_block(indent: int) -> dict[str, Any] | list[Any]:
        nonlocal index
        result_dict: dict[str, Any] = {}
        result_list: list[Any] = []
        mode: str | None = None
        while index < len(lines):
            line = lines[index]
            current_indent = len(line) - len(line.lstrip(" "))
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation in {path}: {line}")
            stripped = line.strip()
            if stripped.startswith("- "):
                mode = mode or "list"
                if mode != "list":
                    raise ValueError(f"Mixed list and mapping in {path}: {line}")
                result_list.append(_parse_scalar(stripped[2:].strip()))
                index += 1
                continue
            mode = mode or "dict"
            if mode != "dict":
                raise ValueError(f"Mixed list and mapping in {path}: {line}")
            key, sep, value = stripped.partition(":")
            if not sep:
                raise ValueError(f"Invalid YAML line in {path}: {line}")
            index += 1
            if value.strip():
                result_dict[key.strip()] = _parse_scalar(value.strip())
            else:
                result_dict[key.strip()] = parse_block(indent + 2)
        return result_list if mode == "list" else result_dict

    parsed = parse_block(0)
    if not isinstance(parsed, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return parsed


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
python -m pytest tests/test_config.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add pyproject.toml requirements.txt README_dev.md configs src tests
git commit -m "feat: add project config scaffold"
```

## Task 2: FMA 主标签和敏感性标签

**Files:**
- Create: `src/stroke_predict/cohort/__init__.py`
- Create: `src/stroke_predict/cohort/labels.py`
- Create: `tests/test_labels.py`

- [ ] **Step 1: 写失败测试**

Create `src/stroke_predict/cohort/__init__.py`:

```python
"""Cohort assembly and label generation."""
```

Create `tests/test_labels.py`:

```python
from __future__ import annotations

import math

import pytest

from stroke_predict.cohort.labels import build_label_record


@pytest.mark.parametrize(
    ("baseline", "post", "expected"),
    [
        (None, 40, "missing"),
        (40, None, "missing"),
        (66, 66, "ceiling_exclude"),
        (40, 44, "Poor"),
        (40, 45, "Good"),
        (64, 65, "Good"),
        (64, 64, "Poor"),
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
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_labels.py -q
```

Expected: FAIL，原因是 `stroke_predict.cohort.labels` 不存在。

- [ ] **Step 3: 写最小实现**

Create `src/stroke_predict/cohort/labels.py`:

```python
from __future__ import annotations

from typing import Any

MISSING_LABEL = "missing"
CEILING_LABEL = "ceiling_exclude"
GOOD_LABEL = "Good"
POOR_LABEL = "Poor"


def build_label_record(
    baseline_fma: Any,
    post_fma: Any,
    *,
    fma_full_score: float = 66.0,
    low_fma_threshold: float = 61.0,
    low_fma_delta_good: float = 5.0,
    near_ceiling_delta_good: float = 3.0,
    proportional_good_threshold: float = 0.70,
) -> dict[str, Any]:
    baseline = parse_optional_float(baseline_fma)
    post = parse_optional_float(post_fma)
    if baseline is None or post is None:
        return _record(
            baseline,
            post,
            None,
            None,
            None,
            MISSING_LABEL,
            MISSING_LABEL,
            MISSING_LABEL,
            MISSING_LABEL,
            "missing_fma",
        )

    delta = post - baseline
    possible_recovery = fma_full_score - baseline
    recovery_ratio = delta / possible_recovery if possible_recovery > 0 else None

    if baseline == fma_full_score:
        return _record(
            baseline,
            post,
            delta,
            possible_recovery,
            recovery_ratio,
            CEILING_LABEL,
            CEILING_LABEL,
            CEILING_LABEL,
            MISSING_LABEL,
            "baseline_full_score",
        )

    if baseline <= low_fma_threshold:
        primary = GOOD_LABEL if delta >= low_fma_delta_good else POOR_LABEL
        reason = "low_baseline_delta_met" if primary == GOOD_LABEL else "low_baseline_delta_not_met"
    else:
        required_delta = min(near_ceiling_delta_good, fma_full_score - baseline)
        primary = GOOD_LABEL if delta >= required_delta else POOR_LABEL
        reason = (
            "near_ceiling_adjusted_delta_met"
            if primary == GOOD_LABEL
            else "near_ceiling_adjusted_delta_not_met"
        )

    delta5_all = GOOD_LABEL if delta >= low_fma_delta_good else POOR_LABEL
    prop70 = (
        GOOD_LABEL
        if recovery_ratio is not None and recovery_ratio >= proportional_good_threshold
        else POOR_LABEL
    )
    low_baseline_only = (
        GOOD_LABEL if delta >= low_fma_delta_good else POOR_LABEL
    ) if baseline <= low_fma_threshold else MISSING_LABEL

    return _record(
        baseline,
        post,
        delta,
        possible_recovery,
        recovery_ratio,
        primary,
        delta5_all,
        prop70,
        low_baseline_only,
        reason,
    )


def parse_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _record(
    baseline: float | None,
    post: float | None,
    delta: float | None,
    possible_recovery: float | None,
    recovery_ratio: float | None,
    label_primary: str,
    label_delta5_all: str,
    label_prop70: str,
    label_low_baseline_only: str,
    label_reason: str,
) -> dict[str, Any]:
    return {
        "baseline_fma": baseline,
        "post_fma": post,
        "delta_fma": delta,
        "possible_recovery": possible_recovery,
        "recovery_ratio": recovery_ratio,
        "label_primary": label_primary,
        "label_delta5_all": label_delta5_all,
        "label_prop70": label_prop70,
        "label_low_baseline_only": label_low_baseline_only,
        "label_reason": label_reason,
        "outcome_delta_fma": delta,
        "outcome_post_fma": post,
    }
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
python -m pytest tests/test_labels.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/stroke_predict/cohort tests/test_labels.py
git commit -m "feat: add FMA response labels"
```

## Task 3: 隐私检查和确定性匿名 ID

**Files:**
- Create: `src/stroke_predict/privacy.py`
- Create: `src/stroke_predict/cohort/ids.py`
- Create: `tests/test_privacy_ids.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_privacy_ids.py`:

```python
from __future__ import annotations

import pandas as pd
import pytest

from stroke_predict.cohort.ids import build_subject_id_map
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
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_privacy_ids.py -q
```

Expected: FAIL，原因是 `stroke_predict.privacy` 或 `stroke_predict.cohort.ids` 不存在。

- [ ] **Step 3: 写最小实现**

Create `src/stroke_predict/privacy.py`:

```python
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
    return df.drop(columns=[col for col in pii_columns if col in df.columns])


def assert_no_pii_columns(df: pd.DataFrame, pii_columns: list[str] | set[str]) -> None:
    blocked = sorted(set(df.columns).intersection(set(pii_columns)))
    if blocked:
        raise ValueError(f"PII columns present in public output: {blocked}")
```

Create `src/stroke_predict/cohort/ids.py`:

```python
from __future__ import annotations


def build_subject_id_map(raw_keys: list[str], *, source: str, prefix: str) -> dict[str, str]:
    normalized = [normalize_source_key(key) for key in raw_keys if normalize_source_key(key)]
    if len(normalized) != len(set(normalized)):
        raise ValueError(f"Duplicate source keys for {source}")
    return {
        key: f"{prefix}-{index:03d}"
        for index, key in enumerate(sorted(normalized), start=1)
    }


def normalize_source_key(value: object) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        return ""
    if text.lower().startswith("sub"):
        suffix = text[3:]
        if suffix.isdigit():
            return f"sub{int(suffix):02d}"
    return text
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
python -m pytest tests/test_privacy_ids.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/stroke_predict/privacy.py src/stroke_predict/cohort/ids.py tests/test_privacy_ids.py
git commit -m "feat: add privacy checks and anonymized ids"
```

## Task 4: Excel 工作簿读取和 schema 验证

**Files:**
- Create: `src/stroke_predict/io/__init__.py`
- Create: `src/stroke_predict/io/excel_status.py`
- Create: `tests/test_excel_status.py`

- [ ] **Step 1: 写失败测试**

Create `src/stroke_predict/io/__init__.py`:

```python
"""Input readers."""
```

Create `tests/test_excel_status.py`:

```python
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stroke_predict.io.excel_status import read_status_workbook


def test_reads_required_workbook_sheets(tmp_path: Path) -> None:
    workbook = tmp_path / "status.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"统计项": ["临床表患者数"], "数值": [1]}).to_excel(
            writer, sheet_name="02_统计汇总", index=False
        )
        pd.DataFrame(
            {
                "患者编号": ["sub01"],
                "姓名": ["张三"],
                "治疗前FMA": [40],
                "治疗后FMA": [45],
                "FMA前后完整": [True],
            }
        ).to_excel(writer, sheet_name="01_患者数据总览", index=False)
        pd.DataFrame(
            {
                "患者编号": ["sub01"],
                "姓名": ["张三"],
                "治疗前FMA": [40],
                "治疗后FMA": [45],
                "FMA前后完整": [True],
            }
        ).to_excel(writer, sheet_name="03_临床量表原始", index=False)
        pd.DataFrame({"来源": ["stroke"], "受试者编号": ["sub01"]}).to_excel(
            writer, sheet_name="06_预处理静息态阶段汇总", index=False
        )
        pd.DataFrame(
            {
                "source": ["stroke"],
                "subject_id": ["sub01"],
                "subject_name": ["张三"],
                "stage": ["baseline"],
                "condition": ["eyes_open"],
                "set_path": ["private.set"],
                "fdt_path": ["private.fdt"],
            }
        ).to_excel(writer, sheet_name="07_预处理静息态文件明细", index=False)

    status = read_status_workbook(
        workbook,
        sheets={
            "summary": "02_统计汇总",
            "clinical_overview": "01_患者数据总览",
            "clinical_raw": "03_临床量表原始",
            "preprocessed_summary": "06_预处理静息态阶段汇总",
            "preprocessed_files": "07_预处理静息态文件明细",
        },
    )

    assert status.clinical_overview.shape[0] == 1
    assert status.preprocessed_files.loc[0, "condition"] == "eyes_open"


def test_missing_required_column_fails(tmp_path: Path) -> None:
    workbook = tmp_path / "bad.xlsx"
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"患者编号": ["sub01"]}).to_excel(
            writer, sheet_name="01_患者数据总览", index=False
        )
        pd.DataFrame({"患者编号": ["sub01"]}).to_excel(
            writer, sheet_name="03_临床量表原始", index=False
        )
        pd.DataFrame({"统计项": []}).to_excel(writer, sheet_name="02_统计汇总", index=False)
        pd.DataFrame({"来源": []}).to_excel(writer, sheet_name="06_预处理静息态阶段汇总", index=False)
        pd.DataFrame({"source": []}).to_excel(writer, sheet_name="07_预处理静息态文件明细", index=False)

    with pytest.raises(ValueError, match="missing columns"):
        read_status_workbook(
            workbook,
            sheets={
                "summary": "02_统计汇总",
                "clinical_overview": "01_患者数据总览",
                "clinical_raw": "03_临床量表原始",
                "preprocessed_summary": "06_预处理静息态阶段汇总",
                "preprocessed_files": "07_预处理静息态文件明细",
            },
        )
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_excel_status.py -q
```

Expected: FAIL，原因是 `stroke_predict.io.excel_status` 不存在。

- [ ] **Step 3: 写最小实现**

Create `src/stroke_predict/io/excel_status.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class StatusWorkbook:
    summary: pd.DataFrame
    clinical_overview: pd.DataFrame
    clinical_raw: pd.DataFrame
    preprocessed_summary: pd.DataFrame
    preprocessed_files: pd.DataFrame


REQUIRED_COLUMNS = {
    "clinical_overview": ["患者编号", "治疗前FMA", "治疗后FMA", "FMA前后完整"],
    "clinical_raw": ["患者编号", "治疗前FMA", "治疗后FMA", "FMA前后完整"],
    "preprocessed_files": ["source", "subject_id", "stage", "condition"],
}


def read_status_workbook(path: str | Path, *, sheets: dict[str, str]) -> StatusWorkbook:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Workbook does not exist: {path}")

    excel = pd.ExcelFile(path)
    missing_sheets = [sheet for sheet in sheets.values() if sheet not in excel.sheet_names]
    if missing_sheets:
        raise ValueError(f"Workbook missing sheets: {missing_sheets}")

    frames = {key: pd.read_excel(path, sheet_name=sheet) for key, sheet in sheets.items()}
    for key, required in REQUIRED_COLUMNS.items():
        require_columns(frames[key], required, sheet_name=sheets[key])

    return StatusWorkbook(
        summary=frames["summary"],
        clinical_overview=frames["clinical_overview"],
        clinical_raw=frames["clinical_raw"],
        preprocessed_summary=frames["preprocessed_summary"],
        preprocessed_files=frames["preprocessed_files"],
    )


def require_columns(df: pd.DataFrame, required: list[str], *, sheet_name: str) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Sheet {sheet_name} missing columns: {missing}")
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
python -m pytest tests/test_excel_status.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/stroke_predict/io tests/test_excel_status.py
git commit -m "feat: read status workbook"
```

## Task 5: 队列构建和角色分配

**Files:**
- Create: `src/stroke_predict/cohort/build.py`
- Create: `tests/test_cohort_build.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_cohort_build.py`:

```python
from __future__ import annotations

import pandas as pd

from stroke_predict.cohort.build import build_cohort_tables
from stroke_predict.io.excel_status import StatusWorkbook


def _status_workbook() -> StatusWorkbook:
    clinical = pd.DataFrame(
        {
            "患者编号": ["sub01", "sub02", "sub03", "sub04"],
            "姓名": ["甲", "乙", "丙", "丁"],
            "年龄": ["60岁", "61岁", "62岁", "63岁"],
            "性别": ["男", "女", "男", "女"],
            "患侧": ["右手", "左手", "右手", "左手"],
            "治疗前FMA": [40, 66, 50, 40],
            "治疗后FMA": [45, 66, None, 44],
            "FMA前后完整": [True, True, False, True],
            "治疗前MBI": [80, 90, None, 70],
            "治疗后MBI": [90, 90, None, 75],
            "MMSE": [28, 29, 27, 26],
        }
    )
    preprocessed = pd.DataFrame(
        {
            "source": ["stroke", "stroke", "stroke", "stroke", "healthy", "healthy"],
            "subject_id": ["sub01", "sub01", "sub02", "sub03", "sub001", "sub001"],
            "subject_name": ["甲", "甲", "乙", "丙", "健康甲", "健康甲"],
            "stage": ["baseline", "baseline", "baseline", "baseline", "baseline", "baseline"],
            "condition": ["eyes_open", "eyes_closed", "eyes_open", "eyes_open", "eyes_open", "eyes_closed"],
            "set_path": ["private"] * 6,
            "fdt_path": ["private"] * 6,
        }
    )
    return StatusWorkbook(
        summary=pd.DataFrame(),
        clinical_overview=clinical,
        clinical_raw=clinical.copy(),
        preprocessed_summary=pd.DataFrame(),
        preprocessed_files=preprocessed,
    )


def test_builds_deidentified_cohort_and_roles() -> None:
    tables = build_cohort_tables(
        _status_workbook(),
        pii_columns=["姓名", "subject_name", "set_path", "fdt_path"],
    )

    cohort = tables.cohort_master.sort_values("subject_id").reset_index(drop=True)
    roles = dict(zip(cohort["subject_id"], cohort["role"]))

    assert roles["HC-001"] == "healthy_ssl"
    assert roles["STK-001"] == "supervised_main"
    assert roles["STK-002"] == "ceiling_exclude"
    assert roles["STK-003"] == "ssl_only_stroke"
    assert roles["STK-004"] == "excluded_no_eeg"
    assert "姓名" not in cohort.columns
    assert "subject_name" not in cohort.columns
    assert "set_path" not in cohort.columns


def test_label_audit_contains_required_fields() -> None:
    tables = build_cohort_tables(
        _status_workbook(),
        pii_columns=["姓名", "subject_name", "set_path", "fdt_path"],
    )

    audit = tables.label_audit
    expected = {
        "subject_id",
        "baseline_fma",
        "post_fma",
        "delta_fma",
        "possible_recovery",
        "recovery_ratio",
        "label_primary",
        "label_delta5_all",
        "label_prop70",
        "label_low_baseline_only",
        "label_reason",
    }
    assert expected.issubset(set(audit.columns))
    assert set(audit["label_primary"]).issubset({"Good", "Poor", "ceiling_exclude", "missing"})
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_cohort_build.py -q
```

Expected: FAIL，原因是 `stroke_predict.cohort.build` 不存在。

- [ ] **Step 3: 写最小实现**

Create `src/stroke_predict/cohort/build.py` with these functions and dataclass:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from stroke_predict.cohort.ids import build_subject_id_map, normalize_source_key
from stroke_predict.cohort.labels import build_label_record
from stroke_predict.io.excel_status import StatusWorkbook
from stroke_predict.privacy import assert_no_pii_columns, drop_pii_columns


@dataclass(frozen=True)
class CohortTables:
    cohort_master: pd.DataFrame
    label_audit: pd.DataFrame
    label_distribution: dict[str, Any]
    cohort_summary: dict[str, Any]


def build_cohort_tables(
    status: StatusWorkbook,
    *,
    pii_columns: list[str],
    label_settings: dict[str, Any] | None = None,
) -> CohortTables:
    label_settings = label_settings or {}
    clinical = status.clinical_overview.copy()
    eeg = status.preprocessed_files.copy()
    clinical["_source_key"] = clinical["患者编号"].map(normalize_source_key)
    eeg["_source_key"] = eeg["subject_id"].map(normalize_source_key)

    stroke_keys = sorted(
        set(clinical["_source_key"].dropna()).union(
            set(eeg.loc[eeg["source"].eq("stroke"), "_source_key"].dropna())
        )
    )
    healthy_keys = sorted(set(eeg.loc[eeg["source"].eq("healthy"), "_source_key"].dropna()))
    stroke_id_map = build_subject_id_map(stroke_keys, source="stroke", prefix="STK")
    healthy_id_map = build_subject_id_map(healthy_keys, source="healthy", prefix="HC")

    eeg_flags = _build_eeg_flags(eeg)
    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    clinical_by_key = clinical.drop_duplicates("_source_key").set_index("_source_key", drop=False)
    for key in stroke_keys:
        clinical_row = clinical_by_key.loc[key].to_dict() if key in clinical_by_key.index else {}
        label = build_label_record(
            clinical_row.get("治疗前FMA"),
            clinical_row.get("治疗后FMA"),
            **label_settings,
        )
        flags = eeg_flags.get(("stroke", key), {})
        subject_id = stroke_id_map[key]
        role = _assign_stroke_role(label["label_primary"], flags)
        rows.append(
            {
                "subject_id": subject_id,
                "source": "stroke",
                "role": role,
                "age": _parse_age(clinical_row.get("年龄")),
                "sex": _normalize_sex(clinical_row.get("性别")),
                "affected_hand": _normalize_hand(clinical_row.get("患侧")),
                "treated_hand": _normalize_hand(clinical_row.get("患侧")),
                "baseline_mbi": label_number(clinical_row.get("治疗前MBI")),
                "post_mbi": label_number(clinical_row.get("治疗后MBI")),
                "mmse": label_number(clinical_row.get("MMSE")),
                "has_baseline_eo": bool(flags.get("baseline_eyes_open", False)),
                "has_baseline_ec": bool(flags.get("baseline_eyes_closed", False)),
                "has_any_resting_eeg": bool(flags.get("any_resting_eeg", False)),
                **label,
            }
        )
        audit_rows.append({"subject_id": subject_id, **label})

    for key in healthy_keys:
        flags = eeg_flags.get(("healthy", key), {})
        rows.append(
            {
                "subject_id": healthy_id_map[key],
                "source": "healthy",
                "role": "healthy_ssl",
                "age": None,
                "sex": None,
                "affected_hand": None,
                "treated_hand": None,
                "baseline_mbi": None,
                "post_mbi": None,
                "mmse": None,
                "has_baseline_eo": bool(flags.get("baseline_eyes_open", False)),
                "has_baseline_ec": bool(flags.get("baseline_eyes_closed", False)),
                "has_any_resting_eeg": bool(flags.get("any_resting_eeg", False)),
                **build_label_record(None, None),
            }
        )

    cohort = pd.DataFrame(rows).sort_values(["source", "subject_id"]).reset_index(drop=True)
    audit = pd.DataFrame(audit_rows).sort_values("subject_id").reset_index(drop=True)
    cohort = drop_pii_columns(cohort, pii_columns)
    audit = drop_pii_columns(audit, pii_columns)
    assert_no_pii_columns(cohort, pii_columns)
    assert_no_pii_columns(audit, pii_columns)
    distribution = cohort["label_primary"].value_counts(dropna=False).to_dict()
    summary = {
        "n_total": int(len(cohort)),
        "n_stroke": int((cohort["source"] == "stroke").sum()),
        "n_healthy": int((cohort["source"] == "healthy").sum()),
        "n_supervised_main": int((cohort["role"] == "supervised_main").sum()),
        "role_counts": cohort["role"].value_counts(dropna=False).to_dict(),
        "label_primary_counts": distribution,
    }
    return CohortTables(cohort, audit, distribution, summary)


def _build_eeg_flags(eeg: pd.DataFrame) -> dict[tuple[str, str], dict[str, bool]]:
    flags: dict[tuple[str, str], dict[str, bool]] = {}
    for row in eeg.to_dict("records"):
        source = str(row.get("source", "")).strip()
        key = normalize_source_key(row.get("_source_key", row.get("subject_id")))
        stage = str(row.get("stage", "")).strip()
        condition = str(row.get("condition", "")).strip()
        item = flags.setdefault((source, key), {"any_resting_eeg": False})
        item["any_resting_eeg"] = True
        if stage == "baseline" and condition == "eyes_open":
            item["baseline_eyes_open"] = True
        if stage == "baseline" and condition == "eyes_closed":
            item["baseline_eyes_closed"] = True
    return flags


def _assign_stroke_role(label_primary: str, flags: dict[str, bool]) -> str:
    has_eo = bool(flags.get("baseline_eyes_open", False))
    has_ec = bool(flags.get("baseline_eyes_closed", False))
    has_any = bool(flags.get("any_resting_eeg", False))
    if label_primary in {"Good", "Poor"} and has_eo and has_ec:
        return "supervised_main"
    if label_primary == "ceiling_exclude":
        return "ceiling_exclude"
    if has_any:
        return "ssl_only_stroke"
    return "excluded_no_eeg"


def label_number(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, str) and not value.strip()) or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_age(value: Any) -> float | None:
    text = "" if value is None else str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return float(digits) if digits else label_number(value)


def _normalize_sex(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    if text == "男":
        return "male"
    if text == "女":
        return "female"
    return text or None


def _normalize_hand(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    if "左" in text:
        return "left"
    if "右" in text:
        return "right"
    return None
```

- [ ] **Step 4: 运行测试确认通过**

Run:

```bash
python -m pytest tests/test_cohort_build.py -q
```

Expected: PASS。

- [ ] **Step 5: 运行已完成单元测试**

Run:

```bash
python -m pytest tests/test_config.py tests/test_labels.py tests/test_privacy_ids.py tests/test_excel_status.py tests/test_cohort_build.py -q
```

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add src/stroke_predict/cohort/build.py tests/test_cohort_build.py
git commit -m "feat: build deidentified cohort tables"
```

## Task 6: 输出写入、环境验证脚本和队列构建脚本

**Files:**
- Create: `src/stroke_predict/cohort/outputs.py`
- Create: `scripts/00_validate_environment.py`
- Create: `scripts/01_build_cohort.py`
- Create: `tests/test_scripts.py`

- [ ] **Step 1: 写失败测试**

Create `tests/test_scripts.py`:

```python
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_fixture_workbook(path: Path) -> None:
    clinical = pd.DataFrame(
        {
            "患者编号": ["sub01"],
            "姓名": ["甲"],
            "年龄": ["60岁"],
            "性别": ["男"],
            "患侧": ["右手"],
            "治疗前FMA": [40],
            "治疗后FMA": [45],
            "FMA前后完整": [True],
        }
    )
    files = pd.DataFrame(
        {
            "source": ["stroke", "stroke"],
            "subject_id": ["sub01", "sub01"],
            "subject_name": ["甲", "甲"],
            "stage": ["baseline", "baseline"],
            "condition": ["eyes_open", "eyes_closed"],
            "set_path": ["private.set", "private.set"],
            "fdt_path": ["private.fdt", "private.fdt"],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame({"统计项": ["临床表患者数"], "数值": [1]}).to_excel(
            writer, sheet_name="02_统计汇总", index=False
        )
        clinical.to_excel(writer, sheet_name="01_患者数据总览", index=False)
        clinical.to_excel(writer, sheet_name="03_临床量表原始", index=False)
        pd.DataFrame({"来源": ["stroke"], "受试者编号": ["sub01"]}).to_excel(
            writer, sheet_name="06_预处理静息态阶段汇总", index=False
        )
        files.to_excel(writer, sheet_name="07_预处理静息态文件明细", index=False)


def test_build_cohort_script_writes_outputs(tmp_path: Path) -> None:
    workbook = tmp_path / "status.xlsx"
    stroke_root = tmp_path / "stroke"
    healthy_root = tmp_path / "healthy"
    output_dir = tmp_path / "outputs"
    stroke_root.mkdir()
    healthy_root.mkdir()
    _write_fixture_workbook(workbook)

    (tmp_path / "paths.yaml").write_text(
        "\n".join(
            [
                "paths:",
                f"  workbook: \"{workbook.as_posix()}\"",
                f"  stroke_eeg_root: \"{stroke_root.as_posix()}\"",
                f"  healthy_eeg_root: \"{healthy_root.as_posix()}\"",
                f"  output_dir: \"{output_dir.as_posix()}\"",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "project.yaml"
    config.write_text(
        "\n".join(
            [
                "paths_config: paths.yaml",
                "sheets:",
                "  summary: 02_统计汇总",
                "  clinical_overview: 01_患者数据总览",
                "  clinical_raw: 03_临床量表原始",
                "  preprocessed_summary: 06_预处理静息态阶段汇总",
                "  preprocessed_files: 07_预处理静息态文件明细",
                "labels:",
                "  fma_full_score: 66",
                "  low_fma_threshold: 61",
                "  low_fma_delta_good: 5",
                "  near_ceiling_delta_good: 3",
                "  proportional_good_threshold: 0.70",
                "privacy:",
                "  pii_columns:",
                "    - 姓名",
                "    - subject_name",
                "    - set_path",
                "    - fdt_path",
                "outputs:",
                "  cohort_dir: cohort",
                "  figures_dir: figures",
            ]
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "scripts/01_build_cohort.py", "--config", str(config)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=True,
    )

    assert "COHORT_BUILD_OK" in result.stdout
    assert (output_dir / "cohort" / "cohort_master.csv").exists()
    assert (output_dir / "cohort" / "label_audit.csv").exists()
    summary = json.loads((output_dir / "cohort" / "cohort_summary.json").read_text(encoding="utf-8"))
    assert summary["n_supervised_main"] == 1
    assert (output_dir / "figures" / "fig_label_distribution.png").exists()
```

- [ ] **Step 2: 运行测试并确认失败**

Run:

```bash
python -m pytest tests/test_scripts.py -q
```

Expected: FAIL，原因是 `scripts/01_build_cohort.py` 不存在。

- [ ] **Step 3: 写最小实现**

Create `src/stroke_predict/cohort/outputs.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from stroke_predict.cohort.build import CohortTables


def write_cohort_outputs(tables: CohortTables, *, output_dir: Path) -> dict[str, Path]:
    cohort_dir = output_dir / "cohort"
    figures_dir = output_dir / "figures"
    cohort_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "cohort_master": cohort_dir / "cohort_master.csv",
        "label_audit": cohort_dir / "label_audit.csv",
        "label_distribution": cohort_dir / "label_distribution.json",
        "cohort_summary": cohort_dir / "cohort_summary.json",
        "label_figure": figures_dir / "fig_label_distribution.png",
    }
    tables.cohort_master.to_csv(paths["cohort_master"], index=False, encoding="utf-8-sig")
    tables.label_audit.to_csv(paths["label_audit"], index=False, encoding="utf-8-sig")
    paths["label_distribution"].write_text(
        json.dumps(tables.label_distribution, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["cohort_summary"].write_text(
        json.dumps(tables.cohort_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_label_distribution_png(tables.label_distribution, paths["label_figure"])
    return paths


def write_label_distribution_png(distribution: dict[str, int], path: Path) -> None:
    width, height = 900, 520
    margin = 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 30), "Primary label distribution", fill="black")
    labels = list(distribution.keys())
    values = [int(distribution[label]) for label in labels]
    max_value = max(values) if values else 1
    bar_area_width = width - 2 * margin
    bar_width = max(35, bar_area_width // max(len(labels) * 2, 1))
    baseline = height - margin
    for index, (label, value) in enumerate(zip(labels, values)):
        x0 = margin + index * (bar_width * 2)
        x1 = x0 + bar_width
        bar_height = int((height - 2 * margin) * value / max_value)
        y0 = baseline - bar_height
        draw.rectangle([x0, y0, x1, baseline], fill="#4C78A8")
        draw.text((x0, baseline + 10), str(label), fill="black")
        draw.text((x0, y0 - 20), str(value), fill="black")
    image.save(path)
```

Create `scripts/00_validate_environment.py`:

```python
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_project_config


def main() -> int:
    config = load_project_config(PROJECT_ROOT / "configs" / "project.yaml")
    missing = [
        str(path)
        for path in [config.workbook_path, config.stroke_eeg_root, config.healthy_eeg_root]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Required paths do not exist: {missing}")
    import pandas  # noqa: F401
    import openpyxl  # noqa: F401
    import PIL  # noqa: F401
    print("ENVIRONMENT_OK")
    print(f"workbook={config.workbook_path}")
    print(f"stroke_eeg_root={config.stroke_eeg_root}")
    print(f"healthy_eeg_root={config.healthy_eeg_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Create `scripts/01_build_cohort.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.cohort.build import build_cohort_tables
from stroke_predict.cohort.outputs import write_cohort_outputs
from stroke_predict.config import load_project_config
from stroke_predict.io.excel_status import read_status_workbook


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_project_config(args.config)
    status = read_status_workbook(
        config.workbook_path,
        sheets={
            "summary": config.sheet("summary"),
            "clinical_overview": config.sheet("clinical_overview"),
            "clinical_raw": config.sheet("clinical_raw"),
            "preprocessed_summary": config.sheet("preprocessed_summary"),
            "preprocessed_files": config.sheet("preprocessed_files"),
        },
    )
    label_settings = {
        "fma_full_score": config.label_setting("fma_full_score"),
        "low_fma_threshold": config.label_setting("low_fma_threshold"),
        "low_fma_delta_good": config.label_setting("low_fma_delta_good"),
        "near_ceiling_delta_good": config.label_setting("near_ceiling_delta_good"),
        "proportional_good_threshold": config.label_setting("proportional_good_threshold"),
    }
    tables = build_cohort_tables(
        status,
        pii_columns=config.pii_columns,
        label_settings=label_settings,
    )
    paths = write_cohort_outputs(tables, output_dir=config.output_dir)
    print("COHORT_BUILD_OK")
    for key, path in paths.items():
        print(f"{key}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: 运行脚本测试确认通过**

Run:

```bash
python -m pytest tests/test_scripts.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add src/stroke_predict/cohort/outputs.py scripts tests/test_scripts.py
git commit -m "feat: add cohort output scripts"
```

## Task 7: 全量测试、真实数据验收和最终提交

**Files:**
- Modify if needed: files created in previous tasks

- [ ] **Step 1: 运行全部自动测试**

Run:

```bash
python -m pytest tests -q
```

Expected: all tests PASS。

- [ ] **Step 2: 运行环境验证**

Run:

```bash
python scripts/00_validate_environment.py
```

Expected output contains:

```text
ENVIRONMENT_OK
workbook=F:\CJZProjectFile\StrokePredictSSL-DLModel\current_data_status_overview_data_only.xlsx
```

- [ ] **Step 3: 用真实 Excel 构建 Phase 0+1 输出**

Run:

```bash
python scripts/01_build_cohort.py --config configs/project.yaml
```

Expected output contains:

```text
COHORT_BUILD_OK
cohort_master=
label_audit=
label_distribution=
cohort_summary=
label_figure=
```

- [ ] **Step 4: 检查公开输出没有 PII 列**

Run:

```bash
python -c "import pandas as pd; from pathlib import Path; blocked={'姓名','姓名写法','EEG文件夹','subject_name','set_path','fdt_path'}; root=Path('outputs/cohort'); files=['cohort_master.csv','label_audit.csv']; bad={f: sorted(blocked.intersection(pd.read_csv(root/f).columns)) for f in files}; print(bad); assert all(not cols for cols in bad.values())"
```

Expected output:

```text
{'cohort_master.csv': [], 'label_audit.csv': []}
```

- [ ] **Step 5: 检查主标签取值和监督队列数量**

Run:

```bash
python -c "import json, pandas as pd; c=pd.read_csv('outputs/cohort/cohort_master.csv'); print(sorted(c['label_primary'].dropna().unique())); print(json.load(open('outputs/cohort/cohort_summary.json', encoding='utf-8'))['n_supervised_main']); assert set(c['label_primary'].dropna()).issubset({'Good','Poor','ceiling_exclude','missing'})"
```

Expected: 第一行只包含 `Good`、`Poor`、`ceiling_exclude`、`missing` 的子集；第二行是实际 `supervised_main` 数量。

- [ ] **Step 6: 检查 Git 没有跟踪原始数据和 outputs**

Run:

```bash
git status --short
git check-ignore -q current_data_status_overview_data_only.xlsx
git check-ignore -q outputs/cohort/cohort_master.csv
```

Expected: `git status --short` 不显示 `current_data_status_overview_data_only.xlsx` 或 `outputs/` 下文件；两个 `git check-ignore` 命令退出码为 0。

- [ ] **Step 7: 如有修复，提交最终调整**

If files changed during acceptance fixes:

```bash
git add pyproject.toml requirements.txt README_dev.md configs src scripts tests
git commit -m "test: verify phase 0-1 cohort pipeline"
```

If no files changed, do not create an empty commit.

## 自检清单

- Spec 覆盖：本计划覆盖配置、环境验证、Excel 读取、匿名 ID、PII 过滤、标签规则、角色分配、输出文件和真实数据验收。
- TDD 顺序：每个生产模块都有先失败测试，再实现，再通过测试的步骤。
- 类型一致性：`ProjectConfig`、`StatusWorkbook`、`CohortTables` 在后续任务中使用的字段名与定义一致。
- 隐私约束：公开输出前调用 `drop_pii_columns` 和 `assert_no_pii_columns`。
- 范围控制：没有加入 EEG 信号读取、PSD/FC、LOPO、模型训练或 manuscript 生成。
