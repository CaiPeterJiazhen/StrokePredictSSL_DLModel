from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
        json.dumps(_json_ready(tables.label_distribution), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths["cohort_summary"].write_text(
        json.dumps(_json_ready(tables.cohort_summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_label_distribution_png(tables.label_distribution, paths["label_figure"])
    return paths


def write_label_distribution_png(distribution: dict[str, Any], path: Path) -> None:
    width, height = 900, 520
    margin = 70
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((margin, 30), "Primary label distribution", fill="black")
    labels = [str(label) for label in distribution.keys()]
    values = [int(value) for value in distribution.values()]
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
        draw.text((x0, baseline + 10), label, fill="black")
        draw.text((x0, y0 - 20), str(value), fill="black")
    image.save(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
