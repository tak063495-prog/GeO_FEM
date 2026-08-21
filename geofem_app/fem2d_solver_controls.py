"""Solver control parsing and small stage-control helpers."""

from __future__ import annotations

import math
from typing import Any, Mapping

from .fem2d_types import FEM2DError, Interface2D, StageResult2D
from .fem2d_utils import _ensure_list, _merge_solver_config, _require_sequence


def newton_settings(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    raw: Mapping[str, Any] = solver if isinstance(solver, Mapping) else {}
    newton = raw.get("newton", {})
    if not isinstance(newton, Mapping):
        newton = {}
    return {
        "max_iter": int(newton.get("max_iter", newton.get("maxiter", 25))),
        "tol_rel": float(newton.get("tol_rel", newton.get("rtol", 1.0e-7))),
        "tol_abs": float(newton.get("tol_abs", newton.get("atol", 1.0e-9))),
        "tol_pressure_rel": float(newton.get("tol_pressure_rel", newton.get("pressure_rtol", newton.get("tol_rel", 1.0e-7)))),
        "tol_pressure_abs": float(newton.get("tol_pressure_abs", newton.get("pressure_atol", newton.get("tol_abs", 1.0e-9)))),
        "mixed_convergence": bool(newton.get("mixed_convergence", newton.get("multi_criteria_convergence", True))),
        "tol_displacement_rel": float(newton.get("tol_displacement_rel", newton.get("displacement_rtol", newton.get("tol_rel", 1.0e-7)))),
        "tol_displacement_abs": float(newton.get("tol_displacement_abs", newton.get("displacement_atol", 1.0e-12))),
        "tol_energy_rel": float(newton.get("tol_energy_rel", newton.get("energy_rtol", newton.get("tol_rel", 1.0e-7)))),
        "tol_energy_abs": float(newton.get("tol_energy_abs", newton.get("energy_atol", 1.0e-12))),
        "strict_force_bypass_ratio": float(newton.get("strict_force_bypass_ratio", 0.5)),
        "allow_nonconvergence": bool(newton.get("allow_nonconvergence", False)),
        "line_search": bool(newton.get("line_search", newton.get("linesearch", False))),
        "max_line_search": int(newton.get("max_line_search", newton.get("line_search_max", 8))),
        "line_search_factor": float(newton.get("line_search_factor", newton.get("line_search_shrink", 0.5))),
        "line_search_armijo": float(newton.get("line_search_armijo", newton.get("armijo", 1.0e-4))),
        "min_line_search_alpha": float(newton.get("min_line_search_alpha", 1.0e-4)),
    }


def tangent_method(solver: Mapping[str, Any] | None) -> str:
    raw: Mapping[str, Any] = solver if isinstance(solver, Mapping) else {}
    tangent = raw.get("tangent", raw.get("material_tangent", "analytic"))
    if isinstance(tangent, Mapping):
        tangent = tangent.get("method", "analytic")
    method = str(tangent or "analytic").lower().strip()
    aliases = {"analytical": "analytic", "consistent": "analytic", "finite_difference": "numerical", "finite-difference": "numerical", "fd": "numerical"}
    method = aliases.get(method, method)
    if method not in {"analytic", "numerical"}:
        raise FEM2DError(f"unsupported 2D tangent method '{tangent}'")
    return method


def increment_settings(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    raw: Mapping[str, Any] = solver if isinstance(solver, Mapping) else {}
    inc = raw.get("increments", raw.get("increment", {}))
    explicit_enabled = "increments" in raw or "increment" in raw
    if isinstance(inc, bool):
        enabled = inc
        inc_map: Mapping[str, Any] = {}
    elif isinstance(inc, (int, float)):
        enabled = True
        inc_map = {"steps": int(inc)}
    elif isinstance(inc, Mapping):
        enabled = bool(inc.get("enabled", True))
        inc_map = inc
    else:
        enabled = False
        inc_map = {}
    steps = int(inc_map.get("steps", inc_map.get("n", inc_map.get("count", 1))))
    max_cutbacks = int(inc_map.get("max_cutbacks", inc_map.get("cutbacks", 0)))
    if steps <= 0:
        raise FEM2DError("increment steps must be positive")
    cutback_factor = float(inc_map.get("cutback_factor", inc_map.get("factor", 0.5)))
    growth = float(inc_map.get("growth", 1.0))
    min_step = float(inc_map.get("min_step", 1.0e-6))
    if not (0.0 < cutback_factor < 1.0):
        raise FEM2DError("increment cutback_factor must satisfy 0 < factor < 1")
    if growth <= 0.0:
        raise FEM2DError("increment growth must be positive")
    if min_step <= 0.0:
        raise FEM2DError("increment min_step must be positive")
    enabled = enabled and (explicit_enabled or steps > 1 or max_cutbacks > 0)
    return {
        "enabled": enabled,
        "steps": steps,
        "max_cutbacks": max_cutbacks,
        "cutback_factor": cutback_factor,
        "growth": growth,
        "min_step": min_step,
    }


def solver_without_increments(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    clean = _merge_solver_config(solver or {}, {})
    clean.pop("increments", None)
    clean.pop("increment", None)
    return clean


def scale_loads(loads: Any, factor: float) -> list[Any]:
    scaled: list[Any] = []
    for load in _ensure_list(loads):
        if not isinstance(load, Mapping):
            scaled.append(load)
            continue
        scaled.append(scale_load_item(load, factor))
    return scaled


def scale_load_item(load: Mapping[str, Any], factor: float) -> dict[str, Any]:
    item = dict(load)
    body_keys = ("bx", "by", "body_fx", "body_fy", "body_x", "body_y")
    for key in (
        "fx",
        "fy",
        "px",
        "py",
        "tx",
        "ty",
        "qx",
        "qy",
        "tx1",
        "ty1",
        "tx2",
        "ty2",
        "qx1",
        "qy1",
        "qx2",
        "qy2",
        "tx_start",
        "ty_start",
        "tx_end",
        "ty_end",
        "qx_local",
        "qy_local",
        "qa",
        "qn",
        "q",
        "mz",
        "moment",
        *body_keys,
    ):
        if key in item and item[key] is not None:
            item[key] = float(item[key]) * factor
    if str(item.get("type", "")).lower().strip() in {"gravity", "self_weight", "body"} or bool(item.get("self_weight", False)):
        if not any(key in item for key in body_keys):
            item["scale"] = float(item.get("scale", 1.0)) * factor
    item["load_case_factor"] = factor
    return item


def scale_boundary_conditions(boundary_conditions: Any, factor: float) -> list[Any]:
    scaled: list[Any] = []
    for bc in _ensure_list(boundary_conditions):
        if not isinstance(bc, Mapping):
            scaled.append(bc)
            continue
        item = dict(bc)
        for key in ("ux", "uy", "value"):
            if key in item and item[key] is not None:
                item[key] = float(item[key]) * factor
        scaled.append(item)
    return scaled


def has_nonlinear_interfaces(interfaces: list[Interface2D] | None) -> bool:
    return any(interface.active and (interface.friction > 0.0 or interface.cohesion > 0.0 or interface.no_tension) for interface in interfaces or [])


def srm_factors(cfg: Mapping[str, Any]) -> list[float]:
    raw = cfg.get("factors")
    if raw is not None:
        factors = [float(v) for v in _require_sequence(raw, "srm.factors")]
    else:
        start = float(cfg.get("factor_start", cfg.get("start_factor", cfg.get("start", cfg.get("fs_start", 1.0)))))
        stop = float(cfg.get("factor_max", cfg.get("end_factor", cfg.get("max_factor", cfg.get("end", cfg.get("fs_max", 2.0))))))
        step = float(cfg.get("factor_step", cfg.get("step", cfg.get("fs_step", 0.1))))
        if step <= 0.0:
            raise FEM2DError("srm factor_step must be positive")
        count = int(math.floor((stop - start) / step + 0.5)) + 1
        factors = [start + i * step for i in range(max(count, 1))]
        if factors[-1] < stop:
            factors.append(stop)
    if not factors or any(v <= 0.0 for v in factors):
        raise FEM2DError("srm factors must be positive")
    return factors


def plastic_ratio(result: StageResult2D) -> float:
    active_rows = [row for row in result.element_results if row.get("active", 0.0)]
    if not active_rows:
        return 0.0
    plastic = sum(1 for row in active_rows if float(row.get("plastic", 0.0)) > 0.0)
    return plastic / len(active_rows)


__all__ = [
    "has_nonlinear_interfaces",
    "increment_settings",
    "newton_settings",
    "plastic_ratio",
    "scale_boundary_conditions",
    "scale_load_item",
    "scale_loads",
    "solver_without_increments",
    "srm_factors",
    "tangent_method",
]
