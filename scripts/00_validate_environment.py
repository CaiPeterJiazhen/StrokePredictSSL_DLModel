from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stroke_predict.config import load_project_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "project.yaml"))
    args = parser.parse_args()

    config = load_project_config(args.config)
    missing = [
        str(path)
        for path in [config.workbook_path, config.stroke_eeg_root, config.healthy_eeg_root]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(f"Required paths do not exist: {missing}")
    import openpyxl  # noqa: F401
    import pandas  # noqa: F401
    import PIL  # noqa: F401

    print("ENVIRONMENT_OK")
    print(f"workbook={config.workbook_path}")
    print(f"stroke_eeg_root={config.stroke_eeg_root}")
    print(f"healthy_eeg_root={config.healthy_eeg_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
