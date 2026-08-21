"""Shared presentation rules for SRM factor-of-safety results."""

from __future__ import annotations

import math
from typing import Any, Mapping


CONFIRMED_BRACKET = "confirmed_bracket"
BRACKET_TOLERANCE_NOT_MET = "bracket_tolerance_not_met"
LOWER_BOUND_INDETERMINATE = "lower_bound_indeterminate"
UNRESOLVED_INDETERMINATE_INTERVAL = "unresolved_indeterminate_interval"
UNBOUNDED_SEARCH_LIMIT = "unbounded_search_limit"
NONMONOTONIC_EVIDENCE = "nonmonotonic_evidence"
MATERIAL_FALLBACK_EVIDENCE = "material_fallback_evidence"


def srm_result_status(srm: Mapping[str, Any] | None) -> str:
    """Return a normalized status, including compatibility with older summaries."""

    if not isinstance(srm, Mapping):
        return ""
    raw = str(srm.get("factor_of_safety_status", "") or "").strip().lower()
    if raw:
        return raw
    stable = _finite_number(srm.get("stable_factor"))
    failed = _finite_number(srm.get("failed_factor"))
    if stable is not None and failed is not None and stable <= failed:
        return CONFIRMED_BRACKET
    if bool(srm.get("nonmonotonic_evidence", False)):
        return NONMONOTONIC_EVIDENCE
    if int(_finite_number(srm.get("indeterminate_trial_count")) or 0) > 0:
        return LOWER_BOUND_INDETERMINATE
    return UNBOUNDED_SEARCH_LIMIT


def srm_result_confidence(srm: Mapping[str, Any] | None) -> str:
    if not isinstance(srm, Mapping):
        return ""
    raw = str(srm.get("factor_of_safety_confidence", "") or "").strip().lower()
    if raw:
        return raw
    return "high" if srm_result_status(srm) == CONFIRMED_BRACKET else "limited"


def srm_fos_is_confirmed(srm: Mapping[str, Any] | None) -> bool:
    if not isinstance(srm, Mapping):
        return False
    explicitly_certified = srm.get("factor_of_safety_certified")
    if explicitly_certified is not None:
        return bool(explicitly_certified) and srm_result_status(srm) == CONFIRMED_BRACKET
    return srm_result_status(srm) == CONFIRMED_BRACKET and srm_result_confidence(srm) == "high"


def srm_fos_display(srm: Mapping[str, Any] | None, *, locale: str = "en") -> str:
    """Format FOS without presenting an unresolved search result as a final value."""

    if not isinstance(srm, Mapping):
        return ""
    fos = _finite_number(srm.get("factor_of_safety"))
    if fos is None:
        return ""
    value = f"{fos:g}"
    status = srm_result_status(srm)
    japanese = str(locale).lower().startswith("ja")
    if status == CONFIRMED_BRACKET:
        if not srm_fos_is_confirmed(srm):
            return f"FOS>={value} (confirmed interval; tolerance not met)"
        suffix = "確定bracket" if japanese else "confirmed bracket"
        return f"FOS={value} ({suffix})"
    if status == LOWER_BOUND_INDETERMINATE:
        suffix = "下限値・未確定trialあり" if japanese else "lower bound; indeterminate trial"
        return f"FOS>={value} ({suffix})"
    if status == UNRESOLVED_INDETERMINATE_INTERVAL:
        return f"FOS>={value} (unresolved interval; indeterminate trial inside bracket)"
    if status == BRACKET_TOLERANCE_NOT_MET:
        return f"FOS>={value} (confirmed interval; tolerance not met)"
    if status == UNBOUNDED_SEARCH_LIMIT:
        suffix = "探索上限まで安定・上限未確定" if japanese else "stable to search limit; upper bound unknown"
        return f"FOS>={value} ({suffix})"
    if status == NONMONOTONIC_EVIDENCE:
        suffix = "非単調な判定証拠・未確定" if japanese else "nonmonotonic evidence; unconfirmed"
        return f"FOS~{value} ({suffix})"
    if status == MATERIAL_FALLBACK_EVIDENCE:
        suffix = "材料更新fallback使用・要検証" if japanese else "material fallback used; verification required"
        return f"FOS~{value} ({suffix})"
    suffix = "未確定" if japanese else "unconfirmed"
    return f"FOS~{value} ({suffix})"


def srm_safety_verdict(srm: Mapping[str, Any] | None) -> str:
    """Only a confirmed bracket may produce an OK safety verdict."""

    if not srm_fos_is_confirmed(srm):
        return "WARN"
    fos = _finite_number(srm.get("factor_of_safety")) if isinstance(srm, Mapping) else None
    return "OK" if fos is not None and fos >= 1.0 else "WARN"


def _finite_number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


__all__ = [
    "BRACKET_TOLERANCE_NOT_MET",
    "CONFIRMED_BRACKET",
    "LOWER_BOUND_INDETERMINATE",
    "MATERIAL_FALLBACK_EVIDENCE",
    "NONMONOTONIC_EVIDENCE",
    "UNBOUNDED_SEARCH_LIMIT",
    "UNRESOLVED_INDETERMINATE_INTERVAL",
    "srm_fos_display",
    "srm_fos_is_confirmed",
    "srm_result_confidence",
    "srm_result_status",
    "srm_safety_verdict",
]
