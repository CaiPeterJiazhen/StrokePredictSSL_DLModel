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
    proportional_denominator = max(possible_recovery, fma_full_score - 40.0)
    proportional_delta = delta / proportional_denominator if proportional_denominator > 0 else None
    prop70 = (
        GOOD_LABEL
        if proportional_delta is not None and proportional_delta >= proportional_good_threshold
        else POOR_LABEL
    )
    low_baseline_only = (
        (GOOD_LABEL if delta >= low_fma_delta_good else POOR_LABEL)
        if baseline <= low_fma_threshold
        else MISSING_LABEL
    )

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
