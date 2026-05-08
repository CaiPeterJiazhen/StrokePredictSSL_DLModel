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
    project_root = (
        project_path.parent.parent
        if project_path.parent.name == "configs"
        else project_path.parent
    )
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
