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
    assert config.sheet("summary") == "02_统计汇总"
    assert config.sheet("clinical_overview") == "01_患者数据总览"
    assert config.sheet("clinical_raw") == "03_临床量表原始"
    assert config.sheet("preprocessed_summary") == "06_预处理静息态阶段汇总"
    assert config.sheet("preprocessed_files") == "07_预处理静息态文件明细"
    assert config.label_setting("fma_full_score") == 66
    assert config.label_setting("low_fma_threshold") == 61
    assert config.label_setting("low_fma_delta_good") == 5
    assert config.label_setting("near_ceiling_delta_good") == 3
    assert config.label_setting("proportional_good_threshold") == 0.70
    assert config.pii_columns == ["姓名", "subject_name"]
