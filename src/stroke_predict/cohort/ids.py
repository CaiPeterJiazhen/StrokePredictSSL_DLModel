from __future__ import annotations

import re

SUBJECT_CODE_RE = re.compile(r"sub0*(\d+)", re.IGNORECASE)


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
    match = SUBJECT_CODE_RE.search(text)
    if match:
        return f"sub{int(match.group(1)):02d}"
    return text
