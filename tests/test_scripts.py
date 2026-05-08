from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


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
                "  cohort_dir: cohort_custom",
                "  figures_dir: figures_custom",
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

    cohort_dir = output_dir / "cohort_custom"
    figures_dir = output_dir / "figures_custom"
    assert "COHORT_BUILD_OK" in result.stdout
    cohort_master = cohort_dir / "cohort_master.csv"
    label_audit = cohort_dir / "label_audit.csv"
    label_distribution = cohort_dir / "label_distribution.json"
    cohort_summary = cohort_dir / "cohort_summary.json"
    label_figure = figures_dir / "fig_label_distribution.png"
    assert cohort_master.exists()
    assert label_audit.exists()
    assert label_distribution.exists()
    assert cohort_summary.exists()
    assert label_figure.exists()

    private_columns = {"姓名", "subject_name", "set_path", "fdt_path", "_source_key"}
    assert private_columns.isdisjoint(pd.read_csv(cohort_master).columns)
    assert private_columns.isdisjoint(pd.read_csv(label_audit).columns)
    assert isinstance(json.loads(label_distribution.read_text(encoding="utf-8")), dict)

    summary = json.loads(cohort_summary.read_text(encoding="utf-8"))
    assert summary["n_supervised_main"] == 1
    assert summary["supervised_label_primary_counts"]["Good"] == 1
    with Image.open(label_figure) as image:
        assert image.format == "PNG"
