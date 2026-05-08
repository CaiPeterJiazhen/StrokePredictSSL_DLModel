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
    private_columns = set(pii_columns).union({"_source_key", "set_path", "fdt_path"})
    cohort = drop_pii_columns(cohort, private_columns)
    audit = drop_pii_columns(audit, private_columns)
    assert_no_pii_columns(cohort, private_columns)
    assert_no_pii_columns(audit, private_columns)

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
