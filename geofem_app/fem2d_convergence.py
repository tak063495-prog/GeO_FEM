"""Unit-consistent convergence criteria shared by nonlinear solvers."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


def riks_convergence_metrics(
    *,
    force_norm: float,
    force_reference: float,
    pressure_norm: float,
    pressure_reference: float,
    pressure_enabled: bool,
    arc_residual: float,
    arc_reference: float,
    mpc_norm: float,
    mpc_reference: float,
    mpc_enabled: bool,
    riks_cfg: Mapping[str, Any],
    legacy_tol: float,
) -> dict[str, Any]:
    common_rel = max(float(riks_cfg.get("tol_rel", legacy_tol)), 0.0)
    common_abs = max(float(riks_cfg.get("tol_abs", 0.0)), 0.0)

    def limit(prefix: str, reference: float) -> float:
        rel = max(float(riks_cfg.get(f"{prefix}_tol_rel", riks_cfg.get(f"tol_{prefix}_rel", common_rel))), 0.0)
        abs_value = max(float(riks_cfg.get(f"{prefix}_tol_abs", riks_cfg.get(f"tol_{prefix}_abs", common_abs))), 0.0)
        return abs_value + rel * max(float(reference), 0.0)

    force_limit = limit("force", force_reference)
    pressure_limit = limit("pressure", pressure_reference)
    arc_limit = limit("arc", arc_reference)
    mpc_limit = limit("mpc", mpc_reference)
    force_converged = bool(float(force_norm) <= force_limit)
    pressure_converged = bool((not pressure_enabled) or float(pressure_norm) <= pressure_limit)
    arc_converged = bool(abs(float(arc_residual)) <= arc_limit)
    mpc_converged = bool((not mpc_enabled) or float(mpc_norm) <= mpc_limit)
    return {
        "converged": bool(force_converged and pressure_converged and arc_converged and mpc_converged),
        "force_converged": force_converged,
        "force_reference_norm": float(force_reference),
        "force_tolerance": force_limit,
        "pressure_converged": pressure_converged,
        "pressure_reference_norm": float(pressure_reference),
        "pressure_tolerance": pressure_limit,
        "arc_converged": arc_converged,
        "arc_reference": float(arc_reference),
        "arc_tolerance": arc_limit,
        "mpc_converged": mpc_converged,
        "mpc_constraint_norm": float(mpc_norm),
        "mpc_reference_norm": float(mpc_reference),
        "mpc_tolerance": mpc_limit,
    }


def dynamic_residual_metrics(
    *,
    force_norm: float,
    force_reference: float,
    pressure_norm: float,
    pressure_reference: float,
    pressure_enabled: bool,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    force_limit = max(float(settings.get("tol_abs", 0.0)), 0.0) + max(
        float(settings.get("tol_rel", 0.0)), 0.0
    ) * max(float(force_reference), 0.0)
    pressure_limit = max(float(settings.get("tol_pressure_abs", settings.get("tol_abs", 0.0))), 0.0) + max(
        float(settings.get("tol_pressure_rel", settings.get("tol_rel", 0.0))), 0.0
    ) * max(float(pressure_reference), 0.0)
    force_converged = bool(float(force_norm) <= force_limit)
    pressure_converged = bool((not pressure_enabled) or float(pressure_norm) <= pressure_limit)

    def normalized(value: float, limit: float) -> float:
        if limit > 0.0:
            return float(value) / limit
        return 0.0 if float(value) == 0.0 else math.inf

    force_ratio = normalized(force_norm, force_limit)
    pressure_ratio = normalized(pressure_norm, pressure_limit) if pressure_enabled else 0.0
    return {
        "converged": bool(force_converged and pressure_converged),
        "force_converged": force_converged,
        "force_residual_norm": float(force_norm),
        "force_reference_norm": float(force_reference),
        "force_tolerance": force_limit,
        "pressure_converged": pressure_converged,
        "pressure_residual_norm": float(pressure_norm),
        "pressure_reference_norm": float(pressure_reference),
        "pressure_tolerance": pressure_limit,
        "normalized_residual_merit": float(math.hypot(force_ratio, pressure_ratio)),
    }


def newton_convergence_metrics(
    *,
    residual_free: np.ndarray,
    external_free: np.ndarray,
    displacement_free: np.ndarray,
    previous_update_free: np.ndarray,
    has_previous_update: bool,
    constraint_norm: float,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    residual_vec = np.asarray(residual_free, dtype=float)
    external_vec = np.asarray(external_free, dtype=float)
    displacement_vec = np.asarray(displacement_free, dtype=float)
    update_vec = np.asarray(previous_update_free, dtype=float)
    force_norm = float(np.linalg.norm(residual_vec))
    force_reference = float(np.linalg.norm(external_vec))
    force_limit = max(float(settings.get("tol_abs", 0.0)), 0.0) + max(float(settings.get("tol_rel", 0.0)), 0.0) * force_reference
    constraint_limit = max(float(settings.get("tol_abs", 0.0)), 0.0) + max(float(settings.get("tol_rel", 0.0)), 0.0)
    force_converged = bool(force_norm <= force_limit)
    constraint_converged = bool(float(constraint_norm) <= constraint_limit)

    update_norm = float(np.linalg.norm(update_vec)) if has_previous_update else 0.0
    displacement_reference = float(np.linalg.norm(displacement_vec))
    displacement_limit = max(float(settings.get("tol_displacement_abs", 0.0)), 0.0) + max(
        float(settings.get("tol_displacement_rel", 0.0)), 0.0
    ) * displacement_reference
    displacement_converged = bool((not has_previous_update) or update_norm <= displacement_limit)

    energy_error = abs(float(update_vec @ residual_vec)) if has_previous_update and update_vec.size else 0.0
    external_work = abs(float(displacement_vec @ external_vec)) if displacement_vec.size else 0.0
    energy_reference = float(max(external_work, np.finfo(float).eps))
    energy_limit = max(float(settings.get("tol_energy_abs", 0.0)), 0.0) + max(float(settings.get("tol_energy_rel", 0.0)), 0.0) * energy_reference
    energy_converged = bool((not has_previous_update) or energy_error <= energy_limit)

    mixed_enabled = bool(settings.get("mixed_convergence", True))
    bypass_ratio = max(float(settings.get("strict_force_bypass_ratio", 0.5)), 0.0)
    strict_force_converged = bool(force_converged and force_norm <= bypass_ratio * force_limit)
    mixed_converged = bool((not mixed_enabled) or strict_force_converged or (displacement_converged and energy_converged))
    return {
        "converged": bool(force_converged and constraint_converged and mixed_converged),
        "mixed_convergence_enabled": mixed_enabled,
        "force_converged": force_converged,
        "force_residual_norm": force_norm,
        "force_reference_norm": force_reference,
        "force_tolerance": force_limit,
        "constraint_converged": constraint_converged,
        "constraint_norm": float(constraint_norm),
        "constraint_tolerance": constraint_limit,
        "displacement_converged": displacement_converged,
        "displacement_increment_norm": update_norm,
        "displacement_reference_norm": displacement_reference,
        "displacement_tolerance": displacement_limit,
        "energy_converged": energy_converged,
        "energy_error": energy_error,
        "energy_reference": energy_reference,
        "energy_tolerance": energy_limit,
        "strict_force_converged": strict_force_converged,
    }


def newton_convergence_with_force_norm(
    convergence: Mapping[str, Any],
    force_norm: float,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(convergence)
    result["force_residual_norm"] = float(force_norm)
    result["force_converged"] = bool(float(force_norm) <= float(result["force_tolerance"]))
    result["strict_force_converged"] = bool(
        result["force_converged"]
        and float(force_norm)
        <= max(float(settings.get("strict_force_bypass_ratio", 0.5)), 0.0) * float(result["force_tolerance"])
    )
    result["converged"] = bool(
        result["force_converged"]
        and result["constraint_converged"]
        and (
            not result["mixed_convergence_enabled"]
            or result["strict_force_converged"]
            or (result["displacement_converged"] and result["energy_converged"])
        )
    )
    return result


__all__ = [
    "dynamic_residual_metrics",
    "newton_convergence_metrics",
    "newton_convergence_with_force_norm",
    "riks_convergence_metrics",
]
