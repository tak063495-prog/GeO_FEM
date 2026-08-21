"""Stage progression helpers for the 2D solver orchestration."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .fem2d_performance_contract import DEFORMATION_MODE_LARGE, deformation_mode_from_config, normalize_deformation_mode
from .fem2d_types import GEOSTATIC_2D_STAGE_TYPES
from .fem2d_utils import _merge_lists, _merge_solver_config


SOLVER_PROGRESS_FUNCTIONS = (
    "solver_progress_contract",
    "stage_sequence_from_config",
    "stage_display_name",
    "stage_type",
    "stage_time",
    "stage_boundary_conditions",
    "stage_loads",
    "stage_mpc_constraints",
    "stage_solver_config",
    "stage_srm_solver_config",
    "stage_riks_solver_config",
    "stage_state_after_result",
)


def solver_progress_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.solver_progress.v1",
        "module": "geofem_app.fem2d_solver_progress",
        "function_count": len(SOLVER_PROGRESS_FUNCTIONS),
        "functions": list(SOLVER_PROGRESS_FUNCTIONS),
        "covered_surfaces": [
            "stage_sequence_normalization",
            "stage_type_and_time",
            "stage_input_merge",
            "stage_solver_config_merge",
            "stage_geostatic_default_load",
            "stage_state_carryover",
        ],
    }


def stage_sequence_from_config(cfg: Mapping[str, Any]) -> list[Any]:
    stages = cfg.get("stages", cfg.get("steps"))
    if isinstance(stages, list) and stages:
        mode = deformation_mode_from_config(cfg)
        return [_stage_with_default_deformation_mode(stage, mode) if isinstance(stage, Mapping) else stage for stage in stages]
    mode = deformation_mode_from_config(cfg)
    stage_type_value = "large_deformation" if mode == DEFORMATION_MODE_LARGE else "static"
    return [
        {
            "name": "Stage-1",
            "type": stage_type_value,
            "deformation_mode": mode,
            "boundary_conditions": cfg.get("boundary_conditions", cfg.get("bc", [])),
            "loads": cfg.get("loads", []),
        }
    ]


def stage_display_name(stage_cfg: Mapping[str, Any], index: int) -> str:
    return str(stage_cfg.get("name", f"Stage-{index}"))


def stage_type(stage_cfg: Mapping[str, Any]) -> str:
    raw_type = str(stage_cfg.get("type", "static")).lower().strip()
    mode = normalize_deformation_mode(stage_cfg.get("deformation_mode", stage_cfg.get("geometry_mode", "")), default="")
    if mode == DEFORMATION_MODE_LARGE and raw_type in {"", "static", "linear_static"}:
        return "large_deformation"
    return raw_type


def stage_time(stage_cfg: Mapping[str, Any], index: int) -> float:
    value = stage_cfg.get("time", stage_cfg.get("t", stage_cfg.get("elapsed_time", index - 1)))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(index - 1)


def stage_boundary_conditions(global_cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> list[Any]:
    return _merge_lists(
        global_cfg.get("boundary_conditions", global_cfg.get("bc", [])),
        stage_cfg.get("boundary_conditions", stage_cfg.get("bc", [])),
    )


def stage_loads(global_cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any], stage_kind: str) -> list[Any]:
    merged = _merge_lists(global_cfg.get("loads", []), stage_cfg.get("loads", []))
    if stage_kind in GEOSTATIC_2D_STAGE_TYPES and not merged and bool(stage_cfg.get("apply_gravity", True)):
        return [
            {
                "type": "gravity",
                "gx": float(stage_cfg.get("gx", 0.0)),
                "gy": float(stage_cfg.get("gy", -1.0)),
                "scale": float(stage_cfg.get("scale", 1.0)),
            }
        ]
    return merged


def stage_mpc_constraints(global_cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> list[Any]:
    return _merge_lists(
        global_cfg.get("mpc_constraints", global_cfg.get("mpc", [])),
        stage_cfg.get("mpc_constraints", stage_cfg.get("mpc", [])),
    )


def stage_solver_config(base_solver_cfg: Any, stage_cfg: Mapping[str, Any]) -> dict[str, Any]:
    solver_cfg = _merge_solver_config(base_solver_cfg, stage_cfg.get("solver", {}))
    mode = normalize_deformation_mode(stage_cfg.get("deformation_mode", stage_cfg.get("geometry_mode", "")), default="")
    if mode == DEFORMATION_MODE_LARGE:
        large = solver_cfg.get("large_deformation", {})
        if not isinstance(large, Mapping) or "enabled" not in large:
            solver_cfg = _merge_solver_config({"large_deformation": {"enabled": True, "backend": "auto"}}, solver_cfg)
    if "increments" in stage_cfg:
        solver_cfg = _merge_solver_config(solver_cfg, {"increments": stage_cfg["increments"]})
    if "increment" in stage_cfg:
        solver_cfg = _merge_solver_config(solver_cfg, {"increments": stage_cfg["increment"]})
    return solver_cfg


def _stage_with_default_deformation_mode(stage: Mapping[str, Any], mode: str) -> Mapping[str, Any]:
    if any(key in stage for key in ("deformation_mode", "geometry_mode", "kinematics")):
        return stage
    copied = dict(stage)
    copied["deformation_mode"] = mode
    return copied


def stage_srm_solver_config(solver_cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return _merge_solver_config(solver_cfg, {"srm": stage_cfg.get("srm", stage_cfg)})


def stage_riks_solver_config(solver_cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> dict[str, Any]:
    return _merge_solver_config(solver_cfg, {"riks": stage_cfg.get("riks", stage_cfg.get("arc_length", stage_cfg))})


def stage_state_after_result(
    result: Any,
    previous_pressure: np.ndarray | None,
    *,
    copy_pressure: bool = False,
    missing_pressure: str = "preserve",
) -> tuple[np.ndarray | None, dict[str, Any]]:
    if result.pore_pressure is None:
        next_pressure = None if missing_pressure == "clear" else previous_pressure
    elif copy_pressure:
        next_pressure = result.pore_pressure.copy()
    else:
        next_pressure = result.pore_pressure
    return next_pressure, dict(result.plastic_state)


__all__ = [
    "SOLVER_PROGRESS_FUNCTIONS",
    "solver_progress_contract",
    "stage_sequence_from_config",
    "stage_display_name",
    "stage_type",
    "stage_time",
    "stage_boundary_conditions",
    "stage_loads",
    "stage_mpc_constraints",
    "stage_solver_config",
    "stage_srm_solver_config",
    "stage_riks_solver_config",
    "stage_state_after_result",
]
