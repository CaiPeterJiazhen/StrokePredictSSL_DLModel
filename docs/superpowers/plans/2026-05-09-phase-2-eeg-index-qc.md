# Phase 2 EEG Index and QC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Phase 2 EEG record indexing and header-level QC outputs without leaking names, raw file names, or local paths.

**Architecture:** Add a focused `stroke_predict.eeg` package with configuration loading, deidentified record indexing, EEGLAB `.set` header parsing, QC evaluation, and output writers. Scripts stay thin and call package APIs; tests use synthetic DataFrames and temporary header fixtures before real-data acceptance.

**Tech Stack:** Python 3.12, pandas, PyYAML, scipy.io for MATLAB `.set` headers, pytest.

---

## File Structure

- Create `configs/eeg.yaml`: non-private Phase 2 thresholds and links to `project.yaml`.
- Modify `requirements.txt`: add `scipy` because Phase 2 reads EEGLAB MATLAB `.set` headers.
- Create `src/stroke_predict/eeg/__init__.py`: package export marker.
- Create `src/stroke_predict/eeg/config.py`: load `eeg.yaml`, resolve `project_config`, expose QC/window settings.
- Create `src/stroke_predict/eeg/index.py`: build deidentified EEG record index from `StatusWorkbook.preprocessed_files`.
- Create `src/stroke_predict/eeg/header.py`: read `.set` header metadata and channel labels without loading raw EEG arrays.
- Create `src/stroke_predict/eeg/qc.py`: compute window counts, channel hash, QC pass/fail and reasons.
- Create `src/stroke_predict/eeg/outputs.py`: write CSV outputs and enforce public schema privacy.
- Create `scripts/02_index_eeg.py`: write `outputs/qc/eeg_record_index.csv`.
- Create `scripts/03_run_eeg_qc.py`: write `outputs/qc/eeg_record_index.csv`, `eeg_qc_summary.csv`, and `channel_order_report.csv`.
- Create `tests/test_eeg_config.py`: config defaults and path resolution.
- Create `tests/test_eeg_index.py`: deidentified record index and privacy schema.
- Create `tests/test_eeg_qc.py`: header parsing helpers, QC rules, window counts, channel hash.
- Create `tests/test_eeg_scripts.py`: smoke tests for both scripts using synthetic workbook/config fixtures.

## Task 1: EEG Config

**Files:**
- Create: `configs/eeg.yaml`
- Create: `src/stroke_predict/eeg/__init__.py`
- Create: `src/stroke_predict/eeg/config.py`
- Test: `tests/test_eeg_config.py`

- [ ] **Step 1: Write the failing config test**

```python
from pathlib import Path

from stroke_predict.eeg.config import load_eeg_config


def test_loads_eeg_config_defaults(tmp_path: Path) -> None:
    project = tmp_path / "project.yaml"
    project.write_text(
        "paths_config: paths.yaml\n"
        "sheets:\n"
        "  summary: S\n"
        "privacy:\n"
        "  pii_columns: []\n",
        encoding="utf-8",
    )
    eeg = tmp_path / "eeg.yaml"
    eeg.write_text("project_config: project.yaml\n", encoding="utf-8")

    config = load_eeg_config(eeg)

    assert config.path == eeg.resolve()
    assert config.project_config_path == project.resolve()
    assert config.required_channels == 62
    assert config.allowed_sampling_rate_hz == 250
    assert config.window_length_sec == 4.0
    assert config.window_overlap == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eeg_config.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'stroke_predict.eeg'`.

- [ ] **Step 3: Write minimal implementation**

Create `src/stroke_predict/eeg/config.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stroke_predict.config import load_yaml_mapping


@dataclass(frozen=True)
class EEGConfig:
    path: Path
    project_config_path: Path
    raw: dict[str, Any]

    @property
    def required_channels(self) -> int:
        return int(self.raw.get("qc", {}).get("required_channels", 62))

    @property
    def allowed_sampling_rate_hz(self) -> float:
        return float(self.raw.get("qc", {}).get("allowed_sampling_rate_hz", 250))

    @property
    def min_duration_sec_main(self) -> float:
        return float(self.raw.get("qc", {}).get("min_duration_sec_main", 60))

    @property
    def min_duration_sec_ssl(self) -> float:
        return float(self.raw.get("qc", {}).get("min_duration_sec_ssl", 30))

    @property
    def window_length_sec(self) -> float:
        return float(self.raw.get("window", {}).get("length_sec", 4))

    @property
    def window_overlap(self) -> float:
        return float(self.raw.get("window", {}).get("overlap", 0.5))

    @property
    def min_valid_windows_per_condition(self) -> int:
        return int(self.raw.get("window", {}).get("min_valid_windows_per_condition", 10))


def load_eeg_config(path: str | Path) -> EEGConfig:
    config_path = Path(path).resolve()
    raw = load_yaml_mapping(config_path)
    project_name = str(raw.get("project_config", "project.yaml"))
    project_path = Path(project_name)
    if not project_path.is_absolute():
        project_path = (config_path.parent / project_path).resolve()
    return EEGConfig(path=config_path, project_config_path=project_path, raw=raw)
```

Create `src/stroke_predict/eeg/__init__.py`:

```python
"""EEG indexing and QC helpers."""
```

Create `configs/eeg.yaml`:

```yaml
project_config: "project.yaml"
qc:
  min_duration_sec_main: 60
  min_duration_sec_ssl: 30
  allowed_sampling_rate_hz: 250
  required_channels: 62
window:
  length_sec: 4
  overlap: 0.5
  min_valid_windows_per_condition: 10
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eeg_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/eeg.yaml src/stroke_predict/eeg/__init__.py src/stroke_predict/eeg/config.py tests/test_eeg_config.py
git commit -m "feat: add eeg phase config"
```

## Task 2: Deidentified EEG Record Index

**Files:**
- Create: `src/stroke_predict/eeg/index.py`
- Test: `tests/test_eeg_index.py`

- [ ] **Step 1: Write the failing index tests**

```python
import pandas as pd

from stroke_predict.eeg.index import build_eeg_record_index
from stroke_predict.io.excel_status import StatusWorkbook


PRIVATE_COLUMNS = {"subject_name", "set_path", "fdt_path", "file_path", "_source_key"}


def _status() -> StatusWorkbook:
    clinical = pd.DataFrame({"患者编号": ["p01"], "治疗前FMA": [40], "治疗后FMA": [45], "FMA前后完整": [True]})
    files = pd.DataFrame(
        {
            "source": ["stroke", "stroke", "healthy"],
            "subject_id": ["p01", "p01", "h01"],
            "subject_name": ["Name A", "Name A", "Name H"],
            "stage": ["基线", "baseline", "baseline"],
            "condition": ["任务 1", "eyes_closed", "eyes_open"],
            "set_path": ["private-a.set", "private-b.set", "private-h.set"],
            "fdt_path": ["private-a.fdt", "private-b.fdt", "private-h.fdt"],
        }
    )
    return StatusWorkbook(
        summary=pd.DataFrame(),
        clinical_overview=clinical,
        clinical_raw=clinical.copy(),
        preprocessed_summary=pd.DataFrame(),
        preprocessed_files=files,
    )


def test_builds_deidentified_record_index() -> None:
    index = build_eeg_record_index(_status())

    assert list(index["subject_id"]) == ["HC-001", "STK-001", "STK-001"]
    assert set(index["stage"]) == {"baseline"}
    assert set(index["condition"]) == {"eyes_open", "eyes_closed"}
    assert index["record_id"].is_unique
    assert PRIVATE_COLUMNS.isdisjoint(index.columns)


def test_record_index_keeps_file_existence_flags(tmp_path) -> None:
    status = _status()
    set_file = tmp_path / "x.set"
    fdt_file = tmp_path / "x.fdt"
    set_file.write_text("placeholder", encoding="utf-8")
    fdt_file.write_bytes(b"")
    status.preprocessed_files.loc[0, "set_path"] = str(set_file)
    status.preprocessed_files.loc[0, "fdt_path"] = str(fdt_file)

    index = build_eeg_record_index(status)

    row = index.loc[index["record_id"].str.contains("eyes_open")].iloc[0]
    assert bool(row["set_exists"]) is True
    assert bool(row["fdt_exists"]) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_eeg_index.py -q`

Expected: FAIL with `ModuleNotFoundError` or missing `build_eeg_record_index`.

- [ ] **Step 3: Write minimal implementation**

Create `src/stroke_predict/eeg/index.py` with:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from stroke_predict.cohort.ids import build_subject_id_map, normalize_source_key
from stroke_predict.io.excel_status import StatusWorkbook


def build_eeg_record_index(status: StatusWorkbook) -> pd.DataFrame:
    clinical = status.clinical_overview.copy()
    files = status.preprocessed_files.copy()
    clinical["_source_key"] = clinical["患者编号"].map(normalize_source_key)
    files["_source_key"] = files["subject_id"].map(normalize_source_key)
    stroke_keys = sorted(
        set(clinical["_source_key"].dropna()).union(
            set(files.loc[files["source"].eq("stroke"), "_source_key"].dropna())
        )
    )
    healthy_keys = sorted(set(files.loc[files["source"].eq("healthy"), "_source_key"].dropna()))
    stroke_ids = build_subject_id_map(stroke_keys, source="stroke", prefix="STK")
    healthy_ids = build_subject_id_map(healthy_keys, source="healthy", prefix="HC")

    rows: list[dict[str, Any]] = []
    sorted_files = files.sort_values(["source", "_source_key", "stage", "condition"]).reset_index(drop=True)
    counters: dict[tuple[str, str, str, str], int] = {}
    for row in sorted_files.to_dict("records"):
        source = str(row.get("source", "")).strip()
        source_key = normalize_source_key(row.get("_source_key"))
        subject_id = stroke_ids[source_key] if source == "stroke" else healthy_ids[source_key]
        stage = normalize_stage(row.get("stage"))
        condition = normalize_condition(row.get("condition"))
        key = (subject_id, source, stage, condition)
        counters[key] = counters.get(key, 0) + 1
        record_index = counters[key]
        rows.append(
            {
                "record_id": f"{subject_id}_{stage}_{condition}_{record_index:02d}",
                "subject_id": subject_id,
                "source": source,
                "stage": stage,
                "condition": condition,
                "record_index": record_index,
                "set_exists": _path_exists(row.get("set_path")),
                "fdt_exists": _path_exists(row.get("fdt_path")),
            }
        )
    return pd.DataFrame(rows).sort_values(["subject_id", "stage", "condition", "record_id"]).reset_index(drop=True)


def normalize_stage(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower()
    mapping = {"基线": "baseline", "即时": "immediate", "阶段": "mid", "最终": "final"}
    return mapping.get(text, text)


def normalize_condition(value: Any) -> str:
    text = "" if value is None else str(value).strip().lower().replace(" ", "_")
    mapping = {"任务_1": "eyes_open", "任务_2": "eyes_closed"}
    return mapping.get(text, text)


def _path_exists(value: Any) -> bool:
    if value is None or pd.isna(value):
        return False
    return Path(str(value)).exists()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eeg_index.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stroke_predict/eeg/index.py tests/test_eeg_index.py
git commit -m "feat: build deidentified eeg index"
```

## Task 3: Header Reader and QC Rules

**Files:**
- Create: `src/stroke_predict/eeg/header.py`
- Create: `src/stroke_predict/eeg/qc.py`
- Test: `tests/test_eeg_qc.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Write failing QC tests**

```python
import pandas as pd

from stroke_predict.eeg.config import EEGConfig
from stroke_predict.eeg.header import EEGHeader
from stroke_predict.eeg.qc import channel_order_hash, count_windows, evaluate_qc, run_qc


def _config() -> EEGConfig:
    return EEGConfig(path=__file__, project_config_path=__file__, raw={})


def test_count_windows_uses_overlap() -> None:
    assert count_windows(duration_sec=60, length_sec=4, overlap=0.5) == 29
    assert count_windows(duration_sec=3.9, length_sec=4, overlap=0.5) == 0


def test_channel_order_hash_is_stable() -> None:
    assert channel_order_hash([" FP1 ", "C3"]) == channel_order_hash(["fp1", " c3"])


def test_evaluate_qc_rejects_bad_sampling_rate() -> None:
    header = EEGHeader(n_channels=62, sfreq=500, pnts=15000, trials=1, channel_labels=["C3", "C4"], datfile="x.fdt")
    result = evaluate_qc(header, source="stroke", stage="baseline", condition="eyes_open", set_exists=True, fdt_exists=True, readable=True, config=_config())

    assert result["passes_qc"] is False
    assert "bad_sampling_rate" in result["qc_reason"]


def test_run_qc_preserves_public_schema() -> None:
    private = pd.DataFrame(
        {
            "record_id": ["STK-001_baseline_eyes_open_01"],
            "subject_id": ["STK-001"],
            "source": ["stroke"],
            "stage": ["baseline"],
            "condition": ["eyes_open"],
            "set_exists": [False],
            "fdt_exists": [False],
        }
    )

    qc = run_qc(private, private_records=[{"set_path": "private.set", "fdt_path": "private.fdt"}], config=_config())

    assert {"set_path", "fdt_path", "file_path", "subject_name"}.isdisjoint(qc.columns)
    assert bool(qc.loc[0, "passes_qc"]) is False
    assert "missing_set" in qc.loc[0, "qc_reason"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_eeg_qc.py -q`

Expected: FAIL with missing modules/functions.

- [ ] **Step 3: Write minimal implementation**

Create `src/stroke_predict/eeg/header.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EEGHeader:
    n_channels: int | None
    sfreq: float | None
    pnts: int | None
    trials: int | None
    channel_labels: list[str]
    datfile: str | None = None

    @property
    def duration_sec(self) -> float | None:
        if self.sfreq in (None, 0) or self.pnts is None or self.trials is None:
            return None
        return float(self.pnts * self.trials) / float(self.sfreq)


def read_eeglab_set_header(path: str | Path) -> EEGHeader:
    import scipy.io

    mat = scipy.io.loadmat(path, simplify_cells=True, verify_compressed_data_integrity=False)
    return EEGHeader(
        n_channels=_optional_int(mat.get("nbchan")),
        sfreq=_optional_float(mat.get("srate")),
        pnts=_optional_int(mat.get("pnts")),
        trials=_optional_int(mat.get("trials")),
        channel_labels=_extract_channel_labels(mat.get("chanlocs")),
        datfile=_optional_str(mat.get("datfile") or mat.get("data")),
    )


def _extract_channel_labels(chanlocs: Any) -> list[str]:
    if chanlocs is None:
        return []
    if isinstance(chanlocs, dict):
        return [_optional_str(chanlocs.get("labels")) or ""]
    labels: list[str] = []
    for item in chanlocs:
        if isinstance(item, dict):
            labels.append(_optional_str(item.get("labels")) or "")
    return labels


def _optional_int(value: Any) -> int | None:
    try:
        return None if value is None else int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _optional_str(value: Any) -> str | None:
    text = "" if value is None else str(value).strip()
    return text or None
```

Create `src/stroke_predict/eeg/qc.py` with:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from stroke_predict.eeg.config import EEGConfig
from stroke_predict.eeg.header import EEGHeader, read_eeglab_set_header


QC_COLUMNS = [
    "record_id", "subject_id", "source", "stage", "condition", "exists", "readable",
    "n_channels", "sfreq", "channel_order_hash", "duration_sec", "n_valid_samples",
    "n_valid_windows_2s", "n_valid_windows_4s", "n_valid_windows_8s",
    "bad_channel_count", "artifact_ratio_if_available", "passes_qc", "qc_reason",
]


def count_windows(duration_sec: float | None, length_sec: float, overlap: float) -> int:
    if duration_sec is None or duration_sec < length_sec:
        return 0
    step = length_sec * (1.0 - overlap)
    if step <= 0:
        raise ValueError("window overlap must be less than 1.0")
    return int((duration_sec - length_sec) // step) + 1


def channel_order_hash(labels: Iterable[str]) -> str:
    normalized = "|".join(str(label).strip().upper() for label in labels)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def evaluate_qc(header: EEGHeader | None, *, source: str, stage: str, condition: str, set_exists: bool, fdt_exists: bool, readable: bool, config: EEGConfig) -> dict[str, Any]:
    reasons: list[str] = []
    if not set_exists:
        reasons.append("missing_set")
    if not fdt_exists:
        reasons.append("missing_fdt")
    if not readable or header is None:
        reasons.append("unreadable_set")
    n_channels = header.n_channels if header else None
    sfreq = header.sfreq if header else None
    duration = header.duration_sec if header else None
    if n_channels != config.required_channels:
        reasons.append("bad_channel_count")
    if sfreq != config.allowed_sampling_rate_hz:
        reasons.append("bad_sampling_rate")
    min_duration = config.min_duration_sec_main if source == "stroke" and stage == "baseline" and condition in {"eyes_open", "eyes_closed"} else config.min_duration_sec_ssl
    if duration is None or duration < min_duration:
        reasons.append("short_duration")
    windows_4s = count_windows(duration, config.window_length_sec, config.window_overlap)
    if windows_4s < config.min_valid_windows_per_condition:
        reasons.append("too_few_4s_windows")
    labels = header.channel_labels if header else []
    return {
        "exists": bool(set_exists and fdt_exists),
        "readable": bool(readable),
        "n_channels": n_channels,
        "sfreq": sfreq,
        "channel_order_hash": channel_order_hash(labels) if labels else None,
        "duration_sec": duration,
        "n_valid_samples": (header.pnts or 0) * (header.trials or 0) if header else None,
        "n_valid_windows_2s": count_windows(duration, 2, config.window_overlap),
        "n_valid_windows_4s": windows_4s,
        "n_valid_windows_8s": count_windows(duration, 8, config.window_overlap),
        "bad_channel_count": 0,
        "artifact_ratio_if_available": None,
        "passes_qc": not reasons,
        "qc_reason": "pass" if not reasons else ";".join(dict.fromkeys(reasons)),
    }


def run_qc(public_index: pd.DataFrame, *, private_records: list[dict[str, Any]], config: EEGConfig) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for public_row, private in zip(public_index.to_dict("records"), private_records):
        set_path = private.get("set_path")
        fdt_path = private.get("fdt_path")
        set_exists = bool(public_row.get("set_exists", False))
        fdt_exists = bool(public_row.get("fdt_exists", False))
        header = None
        readable = False
        if set_exists and set_path:
            try:
                header = read_eeglab_set_header(Path(str(set_path)))
                readable = True
            except Exception:
                readable = False
        result = evaluate_qc(
            header,
            source=str(public_row["source"]),
            stage=str(public_row["stage"]),
            condition=str(public_row["condition"]),
            set_exists=set_exists,
            fdt_exists=fdt_exists,
            readable=readable,
            config=config,
        )
        rows.append({**{key: public_row[key] for key in ["record_id", "subject_id", "source", "stage", "condition"]}, **result})
    return pd.DataFrame(rows, columns=QC_COLUMNS)
```

Add `scipy` to `requirements.txt`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_eeg_qc.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt src/stroke_predict/eeg/header.py src/stroke_predict/eeg/qc.py tests/test_eeg_qc.py
git commit -m "feat: add eeg header qc"
```

## Task 4: Output Writers and Scripts

**Files:**
- Create: `src/stroke_predict/eeg/outputs.py`
- Create: `scripts/02_index_eeg.py`
- Create: `scripts/03_run_eeg_qc.py`
- Test: `tests/test_eeg_scripts.py`

- [ ] **Step 1: Write failing script smoke tests**

```python
from pathlib import Path

import pandas as pd

from stroke_predict.eeg.outputs import assert_public_eeg_output


def test_public_output_rejects_path_columns() -> None:
    private_path = "F" + ":/private/name" + ".set"
    frame = pd.DataFrame({"subject_id": ["STK-001"], "set_path": [private_path]})

    try:
        assert_public_eeg_output(frame)
    except ValueError as exc:
        assert "set_path" in str(exc)
    else:
        raise AssertionError("Expected path column rejection")


def test_index_and_qc_scripts_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "02_index_eeg.py").exists()
    assert (root / "scripts" / "03_run_eeg_qc.py").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_eeg_scripts.py -q`

Expected: FAIL because writer and scripts do not exist.

- [ ] **Step 3: Write minimal implementation**

Create `src/stroke_predict/eeg/outputs.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


FORBIDDEN_PUBLIC_COLUMNS = {"subject_name", "set_path", "fdt_path", "file_path", "_source_key", "姓名", "姓名写法", "EEG文件夹"}


def assert_public_eeg_output(frame: pd.DataFrame) -> None:
    leaked = sorted(FORBIDDEN_PUBLIC_COLUMNS.intersection(frame.columns))
    if leaked:
        raise ValueError(f"EEG public output contains forbidden columns: {leaked}")
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        values = frame[column].dropna().astype(str)
        if values.str.contains(r"[A-Za-z]:[\\/]|\.set$|\.fdt$", regex=True).any():
            raise ValueError(f"EEG public output contains path-like values in column {column}")


def write_qc_outputs(*, record_index: pd.DataFrame, qc_summary: pd.DataFrame, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    channel_report = build_channel_order_report(qc_summary)
    for frame in (record_index, qc_summary, channel_report):
        assert_public_eeg_output(frame)
    paths = {
        "record_index": output / "eeg_record_index.csv",
        "qc_summary": output / "eeg_qc_summary.csv",
        "channel_order_report": output / "channel_order_report.csv",
    }
    record_index.to_csv(paths["record_index"], index=False)
    qc_summary.to_csv(paths["qc_summary"], index=False)
    channel_report.to_csv(paths["channel_order_report"], index=False)
    return paths


def build_channel_order_report(qc_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = qc_summary.dropna(subset=["channel_order_hash"])
    for hash_value, group in valid.groupby("channel_order_hash", sort=True):
        first = group.iloc[0]
        rows.append(
            {
                "channel_order_hash": hash_value,
                "n_records": int(len(group)),
                "n_channels": int(first["n_channels"]) if pd.notna(first["n_channels"]) else None,
                "example_subject_id": first["subject_id"],
                "example_record_id": first["record_id"],
            }
        )
    return pd.DataFrame(rows)
```

Create both scripts with the existing project pattern:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
```

`02_index_eeg.py` then loads config/workbook, calls `build_eeg_record_index`, checks privacy, writes only `eeg_record_index.csv`, and prints `EEG_INDEX_OK`.

`03_run_eeg_qc.py` loads config/workbook, builds the public index plus private records, calls `run_qc`, writes all QC outputs, and prints `EEG_QC_OK`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_eeg_scripts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/stroke_predict/eeg/outputs.py scripts/02_index_eeg.py scripts/03_run_eeg_qc.py tests/test_eeg_scripts.py
git commit -m "feat: add eeg qc scripts"
```

## Task 5: Integration Tests and Real Data Acceptance

**Files:**
- Modify: `tests/test_eeg_index.py`
- Modify: `tests/test_eeg_qc.py`
- Modify: `tests/test_eeg_scripts.py`

- [ ] **Step 1: Add integration assertions**

Add tests that verify:

```python
assert {"record_id", "subject_id", "source", "stage", "condition"}.issubset(index.columns)
assert {"duration_sec", "n_valid_windows_4s", "passes_qc", "qc_reason"}.issubset(qc.columns)
assert not any(".set" in str(value) or ".fdt" in str(value) for value in frame.astype(str).to_numpy().ravel())
```

- [ ] **Step 2: Run targeted tests to verify failures if any schema gaps remain**

Run: `python -m pytest tests/test_eeg_config.py tests/test_eeg_index.py tests/test_eeg_qc.py tests/test_eeg_scripts.py -q`

Expected: PASS after fixing only schema issues revealed by the tests.

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests -q`

Expected: PASS.

- [ ] **Step 4: Run real data scripts**

Run:

```bash
python scripts/00_validate_environment.py
python scripts/01_build_cohort.py --config configs/project.yaml
python scripts/02_index_eeg.py --config configs/eeg.yaml
python scripts/03_run_eeg_qc.py --config configs/eeg.yaml
```

Expected:

```text
ENVIRONMENT_OK
COHORT_BUILD_OK
EEG_INDEX_OK
EEG_QC_OK
```

- [ ] **Step 5: Verify real Phase 2 acceptance**

Run a read-only acceptance check that loads `outputs/cohort/cohort_master.csv`, `outputs/qc/eeg_record_index.csv`, and `outputs/qc/eeg_qc_summary.csv` and asserts:

```python
supervised = cohort.loc[cohort["role"].eq("supervised_main"), "subject_id"]
assert len(supervised) == 19
assert set(supervised).issubset(set(qc.loc[qc["stage"].eq("baseline"), "subject_id"]))
for condition in ["eyes_open", "eyes_closed"]:
    rows = qc[(qc["subject_id"].isin(supervised)) & (qc["stage"].eq("baseline")) & (qc["condition"].eq(condition))]
    assert len(rows) == 19
    assert rows["passes_qc"].all()
assert not any(".set" in text or ".fdt" in text or ":/" in text or ":\\" in text for text in public_outputs)
```

- [ ] **Step 6: Commit verification-facing adjustments**

```bash
git add configs/eeg.yaml requirements.txt src/stroke_predict/eeg scripts/02_index_eeg.py scripts/03_run_eeg_qc.py tests
git commit -m "test: cover eeg qc integration"
```

Skip this commit if no files changed after the previous commits.

## Self-Review Checklist

- Spec coverage: Phase 2 outputs, privacy boundary, QC thresholds, scripts, and real-data acceptance are covered.
- Placeholder scan: no unresolved placeholder markers or "implement later" steps.
- Type consistency: `EEGConfig`, `EEGHeader`, `build_eeg_record_index`, `run_qc`, and `write_qc_outputs` are used consistently across tasks.
- Scope guard: no PSD/FC extraction, no fold registry, no model training.
