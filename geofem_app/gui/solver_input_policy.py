"""GUI-only solver input policy helpers."""

from __future__ import annotations

import copy
from typing import Any, Mapping

import yaml


_OUTPUT_DEFER_KEYS = {
    "lazy_reports",
    "defer_reports",
    "lazy_result_view",
    "defer_result_view",
    "lazy_standard_report",
    "defer_standard_report",
}
_REPORT_DEFER_KEYS = {"lazy", "defer", "defer_generation"}


def apply_gui_solver_output_defaults(input_text: str) -> str:
    """Return solver YAML with GUI-only lazy report defaults applied.

    The editor text itself is left untouched by callers; this policy only affects
    the run-specific input copy passed to the background solver process.
    """

    text = "" if input_text is None else str(input_text)
    try:
        loaded = yaml.safe_load(text) if text.strip() else {}
    except Exception:
        return text
    if not isinstance(loaded, Mapping):
        return text
    cfg: dict[str, Any] = copy.deepcopy(dict(loaded))
    output_key = "output" if "output" in cfg or "outputs" not in cfg else "outputs"
    output = cfg.get(output_key, {})
    if output is None:
        output = {}
    if not isinstance(output, Mapping):
        return text
    output_map = dict(output)
    report = cfg.get("report", cfg.get("calculation_report", {}))
    report_map = report if isinstance(report, Mapping) else {}
    has_explicit_defer_policy = any(key in output_map for key in _OUTPUT_DEFER_KEYS) or any(key in report_map for key in _REPORT_DEFER_KEYS)
    if not has_explicit_defer_policy:
        output_map["lazy_reports"] = True
        cfg[output_key] = output_map
    solver = cfg.get("solver", {})
    solver_map = dict(solver) if isinstance(solver, Mapping) else {}
    execution = solver_map.get("execution", {})
    execution_map = dict(execution) if isinstance(execution, Mapping) else {}
    execution_map.setdefault("context", "gui")
    execution_map.setdefault("profile", "interactive")
    solver_map["execution"] = execution_map
    srm = solver_map.get("srm", {})
    srm_map = dict(srm) if isinstance(srm, Mapping) else {}
    srm_map.setdefault("progress_stdout", True)
    solver_map["srm"] = srm_map
    cfg["solver"] = solver_map
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


__all__ = ["apply_gui_solver_output_defaults"]
