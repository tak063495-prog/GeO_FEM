"""Dynamic-analysis helper functions split from the 2D solver.

The solver orchestrator keeps stage progression, element assembly, constraints,
and output writing. This module owns dynamic profile parsing, time-history
interpolation/integration, Rayleigh damping helpers, mass regularization, and
Newmark history row construction.
"""

from __future__ import annotations

import math
import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.sparse import csr_matrix, diags

from .fem2d_types import FEM2DError
from .fem2d_utils import _ensure_list


DYNAMIC_HELPER_FUNCTIONS = (
    '_dynamic_stage_settings',
    '_apply_dynamic_profile',
    '_dynamic_solver_config',
    '_rayleigh_coefficients_from_damping_spec',
    '_float_sequence',
    '_dynamic_up_enabled',
    '_dynamic_boundary_unit_scale',
    '_dynamic_boundary_value_at',
    '_integrated_history_rows',
    '_integrated_history_value',
    '_dynamic_time_vector',
    '_dynamic_initial_vector',
    '_regularized_dynamic_mass',
    '_rayleigh_damping_matrix',
    '_dynamic_load_scale',
    '_dynamic_history_row',
    '_time_history_rows',
    '_history_row_k',
    '_history_axis_k_at',
    '_history_value_at',
    '_row_first_float',
    '_acceleration_to_g',
)


def dynamic_helper_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.dynamic_helpers.v1",
        "function_count": len(DYNAMIC_HELPER_FUNCTIONS),
        "functions": list(DYNAMIC_HELPER_FUNCTIONS),
        "owner_boundary": "dynamic helper module parses dynamic settings, interpolates time histories, builds damping/mass helpers, and summarizes Newmark history; fem2d_solver keeps coupled nonlinear stage orchestration",
        "covered_surfaces": [
            "dynamic_profile",
            "time_history_interpolation",
            "rayleigh_damping",
            "mass_regularization",
            "newmark_history",
            "seismic_history_helpers",
        ],
    }


def _dynamic_stage_settings(stage_cfg: Mapping[str, Any], solver_cfg: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for source in (solver_cfg.get("dynamic", {}), solver_cfg.get("newmark", {}), stage_cfg.get("dynamic", {}), stage_cfg.get("newmark", {})):
        if isinstance(source, Mapping):
            out.update(dict(source))
    for key in (
        "dt",
        "time_step",
        "steps",
        "n_steps",
        "duration",
        "end_time",
        "time_history",
        "load_time_history",
        "initial_displacement",
        "initial_velocity",
        "u0",
        "v0",
        "beta",
        "gamma",
        "newmark_beta",
        "newmark_gamma",
        "rayleigh_alpha",
        "rayleigh_beta",
        "alpha_m",
        "beta_k",
        "lumped_mass",
        "mass_floor",
        "profile",
        "dynamic_profile",
        "commercial_profile",
        "convergence",
        "newton",
        "damping",
        "damping_profile",
        "boundary_input",
        "boundary_units",
        "time_unit",
    ):
        if key in stage_cfg:
            out[key] = stage_cfg[key]
    return _apply_dynamic_profile(out)
def _apply_dynamic_profile(dynamic_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(dynamic_cfg)
    profile = str(cfg.get("profile", cfg.get("dynamic_profile", cfg.get("commercial_profile", ""))) or "").lower().strip().replace("-", "_")
    profile_info: dict[str, Any] = {}
    if profile in {"geofeas", "geofeas_like", "commercial", "commercial_robust"}:
        defaults: dict[str, Any] = {
            "newmark_beta": 0.25,
            "newmark_gamma": 0.5,
            "time_history_mode": "linear",
            "max_cutbacks": 6 if profile == "commercial_robust" else 4,
            "cutback_factor": 0.5,
            "min_dt": 1.0e-7,
            "mass_floor": 1.0e-12,
        }
        convergence = dict(cfg.get("convergence", cfg.get("newton", {})) if isinstance(cfg.get("convergence", cfg.get("newton", {})), Mapping) else {})
        convergence_defaults = {"max_iter": 20, "tol_rel": 1.0e-7, "tol_abs": 1.0e-9, "line_search": True}
        for key, value in convergence_defaults.items():
            convergence.setdefault(key, value)
        cfg["convergence"] = convergence
        for key, value in defaults.items():
            cfg.setdefault(key, value)
        profile_info = {
            "name": profile,
            "source": "built_in_geo_feas_like_profile",
            "note": "Open profile for repeatable verification; not a proprietary GeoFEAS internal setting.",
        }
    damping = cfg.get("damping", cfg.get("damping_profile"))
    if isinstance(damping, Mapping):
        alpha, beta = _rayleigh_coefficients_from_damping_spec(damping)
        if alpha is not None:
            cfg["rayleigh_alpha"] = alpha
        if beta is not None:
            cfg["rayleigh_beta"] = beta
        cfg["damping_spec"] = {str(key): value for key, value in damping.items()}
    if profile_info:
        cfg["profile_info"] = profile_info
    return cfg
def _dynamic_solver_config(solver_cfg: Mapping[str, Any], dynamic_cfg: Mapping[str, Any]) -> dict[str, Any]:
    merged = dict(solver_cfg)
    convergence = dynamic_cfg.get("convergence", dynamic_cfg.get("newton", {}))
    if isinstance(convergence, Mapping):
        base_newton = dict(merged.get("newton", {})) if isinstance(merged.get("newton", {}), Mapping) else {}
        base_newton.update({str(key): value for key, value in convergence.items()})
        merged["newton"] = base_newton
    if "tangent" in dynamic_cfg or "material_tangent" in dynamic_cfg:
        merged["tangent"] = dynamic_cfg.get("tangent", dynamic_cfg.get("material_tangent"))
    return merged
def _rayleigh_coefficients_from_damping_spec(spec: Mapping[str, Any]) -> tuple[float | None, float | None]:
    if spec.get("rayleigh_alpha", spec.get("alpha_m", spec.get("alpha"))) not in (None, "") or spec.get("rayleigh_beta", spec.get("beta_k", spec.get("beta"))) not in (None, ""):
        alpha = spec.get("rayleigh_alpha", spec.get("alpha_m", spec.get("alpha")))
        beta = spec.get("rayleigh_beta", spec.get("beta_k", spec.get("beta")))
        return (None if alpha in (None, "") else float(alpha), None if beta in (None, "") else float(beta))
    zetas = _float_sequence(spec.get("damping_ratios", spec.get("zeta", spec.get("damping_ratio"))))
    freqs = _float_sequence(spec.get("frequencies_hz", spec.get("frequency_hz", spec.get("target_frequencies_hz"))))
    if len(zetas) >= 2 and len(freqs) >= 2:
        w1 = 2.0 * math.pi * freqs[0]
        w2 = 2.0 * math.pi * freqs[1]
        if w1 <= 0.0 or w2 <= 0.0 or abs(w1 - w2) <= 1.0e-15:
            raise FEM2DError("dynamic damping frequencies must be positive and distinct")
        matrix = np.array([[1.0 / (2.0 * w1), w1 / 2.0], [1.0 / (2.0 * w2), w2 / 2.0]], dtype=float)
        alpha, beta = np.linalg.solve(matrix, np.array([zetas[0], zetas[1]], dtype=float))
        return float(alpha), float(beta)
    return None, None
def _float_sequence(raw: Any) -> list[float]:
    if raw in (None, ""):
        return []
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, str):
        return [float(part) for part in raw.replace(";", ",").split(",") if part.strip()]
    return [float(value) for value in _ensure_list(raw) if value not in (None, "")]
def _dynamic_up_enabled(stage_cfg: Mapping[str, Any], dynamic_cfg: Mapping[str, Any], hydro: Mapping[str, Any]) -> bool:
    if bool(dynamic_cfg.get("up_coupled", dynamic_cfg.get("u_p", dynamic_cfg.get("coupled_up", False)))):
        return True
    if bool(stage_cfg.get("up_coupled", stage_cfg.get("u_p", False))):
        return True
    fields = stage_cfg.get("fields", stage_cfg.get("unknowns", []))
    if isinstance(fields, str):
        fields = [fields]
    if any(str(field).lower().replace("_", "-") in {"u-p", "p", "pore-pressure", "pressure"} for field in _ensure_list(fields)):
        return True
    return bool(hydro and bool(hydro.get("dynamic_up", hydro.get("up_coupled", False))))
def _dynamic_boundary_unit_scale(spec: Mapping[str, Any]) -> float:
    unit = str(spec.get("unit", spec.get("displacement_unit", "")) or "").lower().strip()
    if unit in {"", "m", "meter", "metre"}:
        return 1.0
    if unit in {"mm", "millimeter", "millimetre"}:
        return 1.0e-3
    if unit in {"cm", "centimeter", "centimetre"}:
        return 1.0e-2
    if unit in {"in", "inch"}:
        return 0.0254
    return 1.0
def _dynamic_boundary_value_at(rows: list[dict[str, Any]], time: float, dof: str, *, default: float) -> float:
    fields = (dof, "value", "displacement", "disp", "u")
    if any(any(row.get(field) not in (None, "") for field in fields) for row in rows):
        return _history_value_at(rows, time, fields, default=default)
    if any(any(row.get(field) not in (None, "") for field in ("velocity", "vel", f"v{dof[-1:]}", "v")) for row in rows):
        return default + _integrated_history_value(rows, time, ("velocity", "vel", f"v{dof[-1:]}", "v"), initial=0.0)
    if any(any(row.get(field) not in (None, "") for field in ("acceleration", "accel", f"a{dof[-1:]}", "a")) for row in rows):
        velocity_history = _integrated_history_rows(rows, ("acceleration", "accel", f"a{dof[-1:]}", "a"), "velocity")
        return default + _integrated_history_value(velocity_history, time, ("velocity",), initial=0.0)
    return default
def _integrated_history_rows(rows: list[dict[str, Any]], fields: tuple[str, ...], out_field: str) -> list[dict[str, Any]]:
    timed = sorted((float(row.get("time", row.get("t", 0.0)) or 0.0), row) for row in rows if row.get("time", row.get("t")) not in (None, ""))
    if not timed:
        return []
    value = 0.0
    out = [{"time": timed[0][0], out_field: value}]
    previous_t, previous_row = timed[0]
    previous_value = _row_first_float(previous_row, fields, 0.0)
    for current_t, current_row in timed[1:]:
        current_value = _row_first_float(current_row, fields, previous_value)
        value += 0.5 * (previous_value + current_value) * (current_t - previous_t)
        out.append({"time": current_t, out_field: value})
        previous_t, previous_value = current_t, current_value
    return out
def _integrated_history_value(rows: list[dict[str, Any]], time: float, fields: tuple[str, ...], *, initial: float) -> float:
    timed = sorted((float(row.get("time", row.get("t", 0.0)) or 0.0), row) for row in rows if row.get("time", row.get("t")) not in (None, ""))
    if not timed:
        return initial
    value = initial
    previous_t, previous_row = timed[0]
    previous_value = _row_first_float(previous_row, fields, 0.0)
    if time <= previous_t:
        return value
    for current_t, current_row in timed[1:]:
        current_value = _row_first_float(current_row, fields, previous_value)
        if time <= current_t:
            a = (time - previous_t) / max(current_t - previous_t, np.finfo(float).eps)
            mid_value = previous_value + a * (current_value - previous_value)
            return value + 0.5 * (previous_value + mid_value) * (time - previous_t)
        value += 0.5 * (previous_value + current_value) * (current_t - previous_t)
        previous_t, previous_value = current_t, current_value
    return value + previous_value * (time - previous_t)
def _dynamic_time_vector(dynamic_cfg: Mapping[str, Any], seismic: Mapping[str, Any] | None, *, start_time: float) -> np.ndarray:
    rows = _time_history_rows(dynamic_cfg.get("time_history", dynamic_cfg.get("history")))
    if not rows and seismic:
        rows = _time_history_rows(seismic.get("time_history", seismic.get("acceleration_history", seismic.get("history"))))
    times = sorted({float(row.get("time", row.get("t", 0.0)) or 0.0) for row in rows if row.get("time", row.get("t")) not in (None, "")})
    if len(times) >= 2:
        return np.asarray(times, dtype=float)
    dt_raw = dynamic_cfg.get("dt", dynamic_cfg.get("time_step"))
    duration_raw = dynamic_cfg.get("duration", dynamic_cfg.get("end_time"))
    steps_raw = dynamic_cfg.get("steps", dynamic_cfg.get("n_steps"))
    if dt_raw in (None, ""):
        dt = 1.0
    else:
        dt = float(dt_raw)
    if dt <= 0.0:
        raise FEM2DError("dynamic time step must be positive")
    if steps_raw not in (None, ""):
        steps = max(int(steps_raw), 1)
    elif duration_raw not in (None, ""):
        steps = max(int(math.ceil(float(duration_raw) / dt)), 1)
    else:
        steps = 1
    start = float(dynamic_cfg.get("start_time", start_time) or 0.0)
    return start + np.arange(steps + 1, dtype=float) * dt
def _dynamic_initial_vector(dynamic_cfg: Mapping[str, Any], ndof: int, *keys: str) -> np.ndarray:
    raw: Any = None
    for key in keys:
        if key in dynamic_cfg:
            raw = dynamic_cfg[key]
            break
    vector = np.zeros(ndof, dtype=float)
    if raw is None:
        return vector
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            idx = int(key)
            if 0 <= idx < ndof:
                vector[idx] = float(value)
        return vector
    values = np.asarray(_ensure_list(raw), dtype=float)
    if values.size != ndof:
        raise FEM2DError(f"dynamic initial vector size must be {ndof}")
    return values.copy()
def _regularized_dynamic_mass(mass: csr_matrix, dynamic_cfg: Mapping[str, Any], stage_name: str) -> tuple[csr_matrix, dict[str, Any]]:
    diag = np.asarray(mass.diagonal(), dtype=float)
    max_mass = float(np.max(np.abs(diag))) if diag.size else 0.0
    if not np.isfinite(max_mass) or max_mass <= 0.0:
        raise FEM2DError(f"{stage_name}: dynamic analysis requires positive mass; set material gamma or density")
    default_floor = max_mass * 1.0e-12
    floor = float(dynamic_cfg.get("mass_floor", default_floor) or 0.0)
    if floor <= 0.0:
        return mass, {"total_mass": float(mass.sum()), "regularized_dofs": 0, "mass_floor": 0.0}
    add = np.where(np.abs(diag) < floor, floor - np.abs(diag), 0.0)
    regularized = int(np.count_nonzero(add > 0.0))
    if regularized:
        mass = (mass + diags(add, format="csr")).tocsr()
    return mass, {"total_mass": float(mass.sum()), "regularized_dofs": regularized, "mass_floor": floor}
def _rayleigh_damping_matrix(mass: csr_matrix, stiffness: csr_matrix, dynamic_cfg: Mapping[str, Any]) -> csr_matrix:
    alpha = float(dynamic_cfg.get("rayleigh_alpha", dynamic_cfg.get("alpha_m", dynamic_cfg.get("mass_damping", 0.0))) or 0.0)
    beta = float(dynamic_cfg.get("rayleigh_beta", dynamic_cfg.get("beta_k", dynamic_cfg.get("stiffness_damping", 0.0))) or 0.0)
    damping_spec = dynamic_cfg.get("damping", dynamic_cfg.get("damping_spec"))
    if isinstance(damping_spec, Mapping):
        computed_alpha, computed_beta = _rayleigh_coefficients_from_damping_spec(damping_spec)
        if computed_alpha is not None:
            alpha = computed_alpha
        if computed_beta is not None:
            beta = computed_beta
    damping = mass * alpha + stiffness * beta
    zeta = dynamic_cfg.get("damping_ratio")
    freq = dynamic_cfg.get("target_frequency_hz", dynamic_cfg.get("frequency_hz"))
    if zeta not in (None, "") and freq not in (None, ""):
        omega = 2.0 * math.pi * float(freq)
        if omega > 0.0:
            damping = damping + mass * (2.0 * float(zeta) * omega)
    return damping.tocsr()
def _dynamic_load_scale(dynamic_cfg: Mapping[str, Any], time: float) -> float:
    rows = _time_history_rows(dynamic_cfg.get("load_time_history", dynamic_cfg.get("load_history")))
    if not rows:
        rows = _time_history_rows(dynamic_cfg.get("time_history", dynamic_cfg.get("history")))
    if not rows:
        return 1.0
    return _history_value_at(rows, time, ("load_scale", "scale", "factor"), default=1.0)
def _dynamic_history_row(
    step: int,
    time: float,
    dt: float,
    displacement: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    mass: csr_matrix,
    damping: csr_matrix,
    stiffness: csr_matrix,
    load: np.ndarray,
    seismic_info: Mapping[str, Any],
    pressure: np.ndarray | None = None,
) -> dict[str, Any]:
    residual = stiffness @ displacement + damping @ velocity + mass @ acceleration - load
    row = {
        "step": int(step),
        "time": float(time),
        "dt": float(dt),
        "kh": float(seismic_info.get("kh", 0.0) or 0.0),
        "kv": float(seismic_info.get("kv", 0.0) or 0.0),
        "load_scale": float(seismic_info.get("load_scale", 1.0) or 1.0),
        "load_norm": float(np.linalg.norm(load)),
        "max_displacement": float(np.max(np.abs(displacement))) if displacement.size else 0.0,
        "max_velocity": float(np.max(np.abs(velocity))) if velocity.size else 0.0,
        "max_acceleration": float(np.max(np.abs(acceleration))) if acceleration.size else 0.0,
        "kinetic_energy": float(0.5 * velocity @ (mass @ velocity)),
        "strain_energy": float(0.5 * displacement @ (stiffness @ displacement)),
        "residual_norm": float(np.linalg.norm(residual)),
    }
    if pressure is not None:
        row["max_pore_pressure"] = float(np.max(pressure)) if pressure.size else 0.0
        row["min_pore_pressure"] = float(np.min(pressure)) if pressure.size else 0.0
    return row
def _time_history_rows(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if isinstance(raw, (str, Path)):
        with Path(raw).open(newline="", encoding="utf-8-sig") as f:
            return [dict(row) for row in csv.DictReader(f)]
    if isinstance(raw, Mapping):
        path = raw.get("file", raw.get("csv", raw.get("path")))
        if path:
            return _time_history_rows(path)
        raw = raw.get("rows", raw.get("data", []))
    rows: list[dict[str, Any]] = []
    for row in _ensure_list(raw):
        if isinstance(row, Mapping):
            rows.append(dict(row))
    return rows
def _history_row_k(row: Mapping[str, Any], axis: str, unit: str) -> float:
    if axis == "x":
        direct = row.get("kh", row.get("kx"))
        accel = row.get("ax", row.get("accel_x", row.get("horizontal_response_acceleration")))
    else:
        direct = row.get("kv", row.get("ky"))
        accel = row.get("ay", row.get("accel_y", row.get("vertical_response_acceleration")))
    if direct not in (None, ""):
        return float(direct)
    return _acceleration_to_g(float(accel or 0.0), unit)
def _history_axis_k_at(rows: list[dict[str, Any]], time: float, axis: str, unit: str) -> float:
    timed = sorted((float(row.get("time", row.get("t", 0.0)) or 0.0), row) for row in rows if row.get("time", row.get("t")) not in (None, ""))
    if not timed:
        return _history_row_k(rows[0], axis, unit) if rows else 0.0
    if time <= timed[0][0]:
        return _history_row_k(timed[0][1], axis, unit)
    if time >= timed[-1][0]:
        return _history_row_k(timed[-1][1], axis, unit)
    for (t0, r0), (t1, r1) in zip(timed, timed[1:]):
        if t0 <= time <= t1:
            if abs(t1 - t0) <= 1.0e-15:
                return _history_row_k(r1, axis, unit)
            a = (time - t0) / (t1 - t0)
            return (1.0 - a) * _history_row_k(r0, axis, unit) + a * _history_row_k(r1, axis, unit)
    return _history_row_k(timed[-1][1], axis, unit)
def _history_value_at(rows: list[dict[str, Any]], time: float, fields: tuple[str, ...], *, default: float) -> float:
    timed = sorted((float(row.get("time", row.get("t", 0.0)) or 0.0), row) for row in rows if row.get("time", row.get("t")) not in (None, ""))
    if not timed:
        return _row_first_float(rows[0], fields, default) if rows else default
    if time <= timed[0][0]:
        return _row_first_float(timed[0][1], fields, default)
    if time >= timed[-1][0]:
        return _row_first_float(timed[-1][1], fields, default)
    for (t0, r0), (t1, r1) in zip(timed, timed[1:]):
        if t0 <= time <= t1:
            v0 = _row_first_float(r0, fields, default)
            v1 = _row_first_float(r1, fields, default)
            if abs(t1 - t0) <= 1.0e-15:
                return v1
            a = (time - t0) / (t1 - t0)
            return (1.0 - a) * v0 + a * v1
    return _row_first_float(timed[-1][1], fields, default)
def _row_first_float(row: Mapping[str, Any], fields: tuple[str, ...], default: float) -> float:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return float(value)
    return default
def _acceleration_to_g(value: float, unit: str) -> float:
    if unit in {"g", "gravity", "gee"}:
        return value
    if unit in {"m/s2", "m/s^2", "mps2", "m_s2"}:
        return value / 9.80665
    if unit in {"gal", "cm/s2", "cm/s^2"}:
        return value / 980.665
    return value

