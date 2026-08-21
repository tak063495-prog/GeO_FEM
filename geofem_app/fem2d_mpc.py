"""Multi-point constraint linear solves and Lagrange corrections."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix

from .fem2d_linear_solver import solve_linear_system
from .reduced_matrix_cache import build_reduced_matrix_cache_from_csr
from .fem2d_types import FEM2DError
from .fem2d_utils import _ensure_list

MPC_SOLVER_FUNCTIONS = (
    "solve_linear_system_with_mpc_elimination",
    "solve_linear_system_with_mpc_lagrange",
    "mpc_constraint_matrix",
    "solve_lagrange_augmented_system",
    "solve_lagrange_mpc_correction",
    "lagrange_mpc_projected_residual",
    "solve_arc_length_lagrange_correction",
    "solve_axisymmetric_up_arc_length_correction",
    "mpc_elimination_requested",
    "mpc_lagrange_requested",
    "mpc_arc_length_stage_plan",
    "mpc_stage_plan",
    "MPCStagePlan",
)


def mpc_solver_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.mpc_solver.v1",
        "module": "geofem_app.fem2d_mpc",
        "function_count": len(MPC_SOLVER_FUNCTIONS),
        "functions": list(MPC_SOLVER_FUNCTIONS),
        "covered_surfaces": [
            "exact_elimination_linear_solve",
            "lagrange_multiplier_linear_solve",
            "lagrange_augmented_direct_fill",
            "lagrange_correction",
            "arc_length_mpc_correction",
            "axisymmetric_up_arc_length_mpc_correction",
            "projected_residual",
            "mpc_method_selection",
            "stage_application_plan",
            "arc_length_stage_application_plan",
        ],
    }


@dataclass(frozen=True)
class MPCStagePlan:
    active: bool
    count: int
    exact_requested: bool
    lagrange_requested: bool
    use_elimination_linear: bool
    use_lagrange_linear: bool
    add_penalty_to_stage_matrix: bool
    applied_method: str


def _free_dof_indices(
    size: int,
    constrained: Mapping[int, float] | set[int] | list[int] | np.ndarray,
    stage_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    fixed_values = list(constrained.keys()) if isinstance(constrained, Mapping) else list(constrained)
    if not fixed_values:
        return np.arange(size, dtype=int), np.zeros(0, dtype=int)
    fixed = np.asarray(sorted({int(value) for value in fixed_values}), dtype=int)
    if fixed.size and (int(fixed[0]) < 0 or int(fixed[-1]) >= size):
        raise FEM2DError(f"{stage_name}: constrained dof index is outside the solution vector")
    mask = np.ones(size, dtype=bool)
    mask[fixed] = False
    return np.flatnonzero(mask), fixed


def solve_linear_system_with_mpc_elimination(
    matrix: csr_matrix,
    rhs: np.ndarray,
    constrained: Mapping[int, float],
    mpc_info: Mapping[str, Any],
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = rhs.size
    fixed = {int(k): float(v) for k, v in constrained.items()}
    equations = mpc_info.get("equations", [])
    if not isinstance(equations, list):
        equations = []

    dependent: dict[int, tuple[int, float, float]] = {}
    for eq in equations:
        if not isinstance(eq, Mapping):
            continue
        slave_dof = int(eq.get("slave_dof", -1))
        master_dof = int(eq.get("master_dof", -1))
        if slave_dof < 0 or master_dof < 0:
            continue
        if slave_dof in fixed:
            raise FEM2DError(f"{stage_name}: MPC slave dof is also fixed")
        if slave_dof in dependent:
            raise FEM2DError(f"{stage_name}: duplicate MPC slave dof")
        dependent[slave_dof] = (master_dof, float(eq.get("coefficient", 1.0)), float(eq.get("value", 0.0)))
    if any(master in dependent for master, _coef, _value in dependent.values()):
        raise FEM2DError(f"{stage_name}: chained MPC elimination is not supported")

    independent = [i for i in range(n) if i not in fixed and i not in dependent]
    col_by_dof = {dof: col for col, dof in enumerate(independent)}
    c = np.zeros(n, dtype=float)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for dof, value in fixed.items():
        c[dof] = value
    for dof in independent:
        rows.append(dof)
        cols.append(col_by_dof[dof])
        data.append(1.0)
    for slave_dof, (master_dof, coefficient, value) in dependent.items():
        c[slave_dof] = value
        if master_dof in fixed:
            c[slave_dof] += coefficient * fixed[master_dof]
        else:
            if master_dof not in col_by_dof:
                raise FEM2DError(f"{stage_name}: MPC master dof is not an independent dof")
            rows.append(slave_dof)
            cols.append(col_by_dof[master_dof])
            data.append(coefficient)

    if not independent:
        residual = matrix @ c - rhs
        return c, {
            "method": "mpc_elimination",
            "linear_method": "none",
            "iterations": 0,
            "residual_norm": float(np.linalg.norm(residual)),
            "equilibrated": False,
        }

    transform = coo_matrix((data, (rows, cols)), shape=(n, len(independent))).tocsr()
    reduced_matrix = transform.T @ matrix @ transform
    reduced_rhs = transform.T @ (rhs - matrix @ c)
    reduced_solution, info = solve_linear_system(reduced_matrix.tocsr(), np.asarray(reduced_rhs).ravel(), stage_name=stage_name, solver=solver)
    u = np.asarray(transform @ reduced_solution).ravel() + c
    residual = float(np.linalg.norm(matrix @ u - rhs))
    return u, {
        "method": "mpc_elimination",
        "linear_method": info.get("method", "direct"),
        "iterations": info.get("iterations", 1),
        "residual_norm": residual,
        "equilibrated": bool(info.get("equilibrated", False)),
    }


def solve_linear_system_with_mpc_lagrange(
    matrix: csr_matrix,
    rhs: np.ndarray,
    constrained: Mapping[int, float],
    mpc_info: Mapping[str, Any],
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    return solve_lagrange_augmented_system(
        matrix,
        rhs,
        constrained,
        mpc_info,
        stage_name=stage_name,
        solver=solver,
        method="mpc_lagrange",
    )


def mpc_constraint_matrix(mpc_info: Mapping[str, Any], n: int, stage_name: str) -> tuple[csr_matrix, np.ndarray]:
    equations = mpc_info.get("equations", [])
    if not isinstance(equations, list):
        equations = []
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    values: list[float] = []
    for eq in equations:
        if not isinstance(eq, Mapping):
            continue
        row = len(values)
        slave_dof = int(eq.get("slave_dof", -1))
        master_dof = int(eq.get("master_dof", -1))
        if not (0 <= slave_dof < n and 0 <= master_dof < n):
            raise FEM2DError(f"{stage_name}: MPC dof index is outside the displacement vector")
        rows.extend([row, row])
        cols.extend([slave_dof, master_dof])
        data.extend([1.0, -float(eq.get("coefficient", 1.0))])
        values.append(float(eq.get("value", 0.0)))
    return coo_matrix((data, (rows, cols)), shape=(len(values), n)).tocsr(), np.asarray(values, dtype=float)


def _assemble_lagrange_saddle_direct(kff: csr_matrix, c_free: csr_matrix) -> tuple[csr_matrix, dict[str, Any]]:
    k = kff.tocsr()
    c = c_free.tocsr()
    n = int(k.shape[0])
    m = int(c.shape[0])
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    k_coo = k.tocoo()
    if k_coo.nnz:
        rows.append(np.asarray(k_coo.row, dtype=np.int64))
        cols.append(np.asarray(k_coo.col, dtype=np.int64))
        data.append(np.asarray(k_coo.data, dtype=float))
    c_coo = c.tocoo()
    if c_coo.nnz:
        c_rows = np.asarray(c_coo.row, dtype=np.int64)
        c_cols = np.asarray(c_coo.col, dtype=np.int64)
        c_data = np.asarray(c_coo.data, dtype=float)
        rows.append(c_cols)
        cols.append(n + c_rows)
        data.append(c_data)
        rows.append(n + c_rows)
        cols.append(c_cols)
        data.append(c_data)
    if data:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        values = np.concatenate(data)
    else:
        row = np.zeros(0, dtype=np.int64)
        col = np.zeros(0, dtype=np.int64)
        values = np.zeros(0, dtype=float)
    matrix = coo_matrix((values, (row, col)), shape=(n + m, n + m)).tocsr()
    return matrix, {
        "enabled": True,
        "mode": "lagrange_saddle_direct_fill",
        "free_dofs": n,
        "active_constraints": m,
        "nnz": int(matrix.nnz),
    }


def _assemble_arc_length_lagrange_direct(
    kff: csr_matrix,
    reference_load_free: np.ndarray,
    du_step: np.ndarray,
    dl_step: float,
    psi: float,
    c_active: csr_matrix,
) -> tuple[csr_matrix, dict[str, Any]]:
    k = kff.tocsr()
    c = c_active.tocsr()
    n = int(k.shape[0])
    m = int(c.shape[0])
    lambda_col = n
    constraint_offset = n + 1
    size = n + 1 + m
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    k_coo = k.tocoo()
    if k_coo.nnz:
        rows.append(np.asarray(k_coo.row, dtype=np.int64))
        cols.append(np.asarray(k_coo.col, dtype=np.int64))
        data.append(np.asarray(k_coo.data, dtype=float))
    ref = np.asarray(reference_load_free, dtype=float).reshape(-1)
    if ref.size != n:
        raise FEM2DError("arc-length direct fill reference load size mismatch")
    du = np.asarray(du_step, dtype=float).reshape(-1)
    if du.size != n:
        raise FEM2DError("arc-length direct fill displacement step size mismatch")
    if n:
        rows.append(np.arange(n, dtype=np.int64))
        cols.append(np.full(n, lambda_col, dtype=np.int64))
        data.append(-ref)
        rows.append(np.full(n, lambda_col, dtype=np.int64))
        cols.append(np.arange(n, dtype=np.int64))
        data.append(2.0 * du)
    rows.append(np.asarray([lambda_col], dtype=np.int64))
    cols.append(np.asarray([lambda_col], dtype=np.int64))
    data.append(np.asarray([2.0 * float(psi) * float(psi) * float(dl_step)], dtype=float))
    c_coo = c.tocoo()
    if c_coo.nnz:
        c_rows = np.asarray(c_coo.row, dtype=np.int64)
        c_cols = np.asarray(c_coo.col, dtype=np.int64)
        c_data = np.asarray(c_coo.data, dtype=float)
        rows.append(c_cols)
        cols.append(constraint_offset + c_rows)
        data.append(c_data)
        rows.append(constraint_offset + c_rows)
        cols.append(c_cols)
        data.append(c_data)
    row = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
    col = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
    values = np.concatenate(data) if data else np.zeros(0, dtype=float)
    matrix = coo_matrix((values, (row, col)), shape=(size, size)).tocsr()
    return matrix, {
        "enabled": True,
        "mode": "arc_length_lagrange_direct_fill",
        "free_dofs": n,
        "active_constraints": m,
        "nnz": int(matrix.nnz),
    }


def solve_lagrange_augmented_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    constrained: Mapping[int, float],
    mpc_info: Mapping[str, Any],
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    method: str,
    constraint_values_override: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    n = rhs.size
    fixed = {int(k): float(v) for k, v in constrained.items()}
    profile: dict[str, float] = {
        "constraint_matrix_elapsed_seconds": 0.0,
        "constraint_filter_elapsed_seconds": 0.0,
        "reduced_matrix_elapsed_seconds": 0.0,
        "bmat_elapsed_seconds": 0.0,
        "linear_solve_elapsed_seconds": 0.0,
    }
    constraint_start = time.perf_counter()
    constraint_matrix, default_values = mpc_constraint_matrix(mpc_info, n, stage_name)
    profile["constraint_matrix_elapsed_seconds"] += max(time.perf_counter() - constraint_start, 0.0)
    constraint_values = default_values if constraint_values_override is None else np.asarray(constraint_values_override, dtype=float)
    if constraint_values.shape != default_values.shape:
        raise FEM2DError(f"{stage_name}: MPC constraint override size mismatch")
    if constraint_values.size == 0:
        linear_start = time.perf_counter()
        u, info = solve_linear_system(matrix, rhs, stage_name=stage_name, solver=solver)
        profile["linear_solve_elapsed_seconds"] += max(time.perf_counter() - linear_start, 0.0)
        return u, {
            "method": method,
            "linear_method": info.get("method", "direct"),
            "iterations": info.get("iterations", 1),
            "residual_norm": float(np.linalg.norm(matrix @ u - rhs)),
            "equilibrated": bool(info.get("equilibrated", False)),
            "multipliers": [],
            "constraint_norm": 0.0,
            "profile": profile,
            **profile,
        }

    free, fixed_dofs = _free_dof_indices(n, fixed, stage_name)
    u = np.zeros(n, dtype=float)
    for dof, value in fixed.items():
        u[dof] = value

    filter_start = time.perf_counter()
    c_free = constraint_matrix[:, free]
    constraint_rhs = constraint_values.copy()
    if fixed_dofs.size:
        constraint_rhs = constraint_rhs - np.asarray(constraint_matrix[:, fixed_dofs] @ u[fixed_dofs]).ravel()
    active_rows = np.asarray(np.abs(c_free).sum(axis=1)).ravel() > 0.0
    inconsistent = np.logical_not(active_rows) & (np.abs(constraint_rhs) > 1.0e-10)
    if np.any(inconsistent):
        raise FEM2DError(f"{stage_name}: MPC constraints conflict with fixed displacement constraints")
    c_free = c_free[active_rows]
    constraint_rhs = constraint_rhs[active_rows]
    profile["constraint_filter_elapsed_seconds"] += max(time.perf_counter() - filter_start, 0.0)
    if free.size == 0:
        residual = matrix @ u - rhs
        return u, {
            "method": method,
            "linear_method": "none",
            "iterations": 0,
            "residual_norm": float(np.linalg.norm(residual)),
            "equilibrated": False,
            "multipliers": [],
            "profile": profile,
            **profile,
        }

    matrix_csr = matrix.tocsr()
    reduced_start = time.perf_counter()
    reduction_cache = build_reduced_matrix_cache_from_csr(matrix_csr, free, fixed_dofs, source="mpc_lagrange_free_dofs")
    reduced_rhs = reduction_cache.reduced_rhs(matrix_csr, rhs, u[fixed_dofs])
    kff = reduction_cache.extract_free_free(matrix_csr)
    profile["reduced_matrix_elapsed_seconds"] += max(time.perf_counter() - reduced_start, 0.0)
    bmat_start = time.perf_counter()
    saddle, saddle_info = _assemble_lagrange_saddle_direct(kff, c_free)
    profile["bmat_elapsed_seconds"] += max(time.perf_counter() - bmat_start, 0.0)
    augmented_rhs = np.concatenate([np.asarray(reduced_rhs).ravel(), constraint_rhs])
    linear_start = time.perf_counter()
    solution, info = solve_linear_system(saddle, augmented_rhs, stage_name=stage_name, solver=solver)
    profile["linear_solve_elapsed_seconds"] += max(time.perf_counter() - linear_start, 0.0)
    u[free] = solution[: free.size]
    multipliers = solution[free.size :]
    full_multipliers = np.zeros(default_values.size, dtype=float)
    full_multipliers[active_rows] = multipliers
    residual = float(np.linalg.norm(matrix @ u + constraint_matrix.T @ full_multipliers - rhs))
    constraint_norm = float(np.linalg.norm(constraint_matrix @ u - constraint_values, ord=np.inf))
    return u, {
        "method": method,
        "linear_method": info.get("method", "direct"),
        "iterations": info.get("iterations", 1),
        "residual_norm": residual,
        "constraint_norm": constraint_norm,
        "equilibrated": bool(info.get("equilibrated", False)),
        "multipliers": [float(v) for v in full_multipliers],
        "reduced_matrix_cache": reduction_cache.info(),
        "lagrange_augmented_assembly": saddle_info,
        "profile": profile,
        **profile,
    }


def solve_lagrange_mpc_correction(
    matrix: csr_matrix,
    rhs: np.ndarray,
    constrained: Mapping[int, float],
    mpc_info: Mapping[str, Any],
    current: np.ndarray,
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    constraint_start = time.perf_counter()
    constraint_matrix, values = mpc_constraint_matrix(mpc_info, rhs.size, stage_name)
    correction_constraint_elapsed = max(time.perf_counter() - constraint_start, 0.0)
    correction_constraints = {int(dof): float(value) - float(current[int(dof)]) for dof, value in constrained.items()}
    correction_values = values - np.asarray(constraint_matrix @ current).ravel()
    solution, info = solve_lagrange_augmented_system(
        matrix,
        rhs,
        correction_constraints,
        mpc_info,
        stage_name=stage_name,
        solver=solver,
        method="mpc_lagrange_correction",
        constraint_values_override=correction_values,
    )
    profile = dict(info.get("profile", {})) if isinstance(info.get("profile", {}), Mapping) else {}
    profile["correction_constraint_matrix_elapsed_seconds"] = correction_constraint_elapsed
    profile["constraint_matrix_elapsed_seconds"] = float(profile.get("constraint_matrix_elapsed_seconds", 0.0) or 0.0) + correction_constraint_elapsed
    info["profile"] = profile
    info.update(profile)
    return solution, info


def lagrange_mpc_projected_residual(
    residual: np.ndarray,
    constrained: Mapping[int, float],
    mpc_info: Mapping[str, Any],
    u: np.ndarray,
    stage_name: str,
) -> tuple[float, float, list[float]]:
    constraint_matrix, values = mpc_constraint_matrix(mpc_info, residual.size, stage_name)
    free, _fixed = _free_dof_indices(residual.size, constrained, stage_name)
    if values.size == 0:
        return (float(np.linalg.norm(residual[free])) if free.size else 0.0), 0.0, []
    if free.size == 0:
        constraint_norm = float(np.linalg.norm(constraint_matrix @ u - values, ord=np.inf))
        return 0.0, constraint_norm, [0.0 for _ in range(values.size)]
    c_free = constraint_matrix[:, free]
    active_rows = np.asarray(np.abs(c_free).sum(axis=1)).ravel() > 0.0
    if not np.any(active_rows):
        return float(np.linalg.norm(residual[free])), float(np.linalg.norm(constraint_matrix @ u - values, ord=np.inf)), [0.0 for _ in range(values.size)]
    c_active = c_free[active_rows]
    rhs = residual[free]
    multipliers, *_ = np.linalg.lstsq(c_active.T.toarray(), rhs, rcond=None)
    projected = rhs - np.asarray(c_active.T @ multipliers).ravel()
    full_multipliers = np.zeros(values.size, dtype=float)
    full_multipliers[active_rows] = multipliers
    constraint_norm = float(np.linalg.norm(constraint_matrix @ u - values, ord=np.inf))
    return float(np.linalg.norm(projected)), constraint_norm, [float(v) for v in full_multipliers]


def solve_arc_length_lagrange_correction(
    tangent: csr_matrix,
    reference_load: np.ndarray,
    residual: np.ndarray,
    constrained: Mapping[int, float],
    mpc_info: Mapping[str, Any],
    u_trial: np.ndarray,
    free: np.ndarray,
    du_step: np.ndarray,
    dl_step: float,
    constraint_value: float,
    psi: float,
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
) -> tuple[np.ndarray, float, list[float]]:
    constraint_matrix, values = mpc_constraint_matrix(mpc_info, tangent.shape[0], stage_name)
    c_free = constraint_matrix[:, free]
    rhs_mpc = values - np.asarray(constraint_matrix @ u_trial).ravel()
    active_rows = np.asarray(np.abs(c_free).sum(axis=1)).ravel() > 0.0
    inconsistent = np.logical_not(active_rows) & (np.abs(rhs_mpc) > 1.0e-10)
    if np.any(inconsistent):
        raise FEM2DError(f"{stage_name}: MPC constraints conflict with fixed displacement constraints")
    c_active = c_free[active_rows]
    rhs_mpc = rhs_mpc[active_rows]

    tangent_csr = tangent.tocsr()
    empty_fixed = np.zeros(0, dtype=np.int64)
    reduction_cache = build_reduced_matrix_cache_from_csr(tangent_csr, free, empty_fixed, source="arc_length_mpc_lagrange_free_dofs")
    kff = reduction_cache.extract_free_free(tangent_csr)
    direct_system, _direct_info = _assemble_arc_length_lagrange_direct(
        kff,
        reference_load[free],
        du_step,
        dl_step,
        psi,
        c_active,
    )
    if c_active.shape[0] == 0:
        rhs = -np.concatenate([residual[free], np.array([constraint_value], dtype=float)])
        correction, _info = solve_linear_system(direct_system, rhs, stage_name=stage_name, solver=solver)
        return correction[:-1], float(correction[-1]), [0.0 for _ in range(values.size)]

    rhs = np.concatenate([-residual[free], np.array([-constraint_value], dtype=float), rhs_mpc])
    correction, _info = solve_linear_system(direct_system, rhs, stage_name=stage_name, solver=solver)
    full_multipliers = np.zeros(values.size, dtype=float)
    full_multipliers[active_rows] = correction[free.size + 1 :]
    return correction[: free.size], float(correction[free.size]), [float(v) for v in full_multipliers]


def solve_axisymmetric_up_arc_length_correction(
    tangent: csr_matrix,
    biot: csr_matrix,
    pressure_lhs: csr_matrix,
    reference_load: np.ndarray,
    residual: np.ndarray,
    pressure_residual: np.ndarray,
    constrained: Mapping[int, float],
    fixed_pressure: Mapping[int, float],
    mpc_info: Mapping[str, Any] | None,
    u_trial: np.ndarray,
    p_trial: np.ndarray,
    free_u: np.ndarray,
    free_p: np.ndarray,
    du_step: np.ndarray,
    dl_step: float,
    constraint_value: float,
    psi: float,
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, float, list[float]]:
    _ = constrained, fixed_pressure, p_trial
    n_u = int(free_u.size)
    n_p = int(free_p.size)
    if n_u == 0:
        raise FEM2DError(f"{stage_name}: coupled axisymmetric Riks requires at least one free displacement dof")
    k_uu = tangent[free_u][:, free_u]
    k_up = -biot[free_u][:, free_p] if n_p else csr_matrix((n_u, 0), dtype=float)
    k_ul = csr_matrix((-reference_load[free_u]).reshape(-1, 1))
    p_u = (biot.T[free_p][:, free_u] / dt) if n_p else csr_matrix((0, n_u), dtype=float)
    p_p = pressure_lhs[free_p][:, free_p] if n_p else csr_matrix((0, 0), dtype=float)
    p_l = csr_matrix((n_p, 1), dtype=float)
    arc_u = csr_matrix((2.0 * du_step).reshape(1, -1))
    arc_p = csr_matrix((1, n_p), dtype=float)
    arc_l = csr_matrix([[2.0 * psi * psi * dl_step]])
    system = bmat([[k_uu, k_up, k_ul], [p_u, p_p, p_l], [arc_u, arc_p, arc_l]], format="csr")
    rhs = -np.concatenate(
        [
            residual[free_u],
            pressure_residual[free_p] if n_p else np.zeros(0, dtype=float),
            np.array([constraint_value], dtype=float),
        ]
    )
    full_multipliers: list[float] = []
    if mpc_info and int(mpc_info.get("count", 0)) > 0:
        constraint_matrix, values = mpc_constraint_matrix(mpc_info, tangent.shape[0], stage_name)
        c_free = constraint_matrix[:, free_u]
        rhs_mpc = values - np.asarray(constraint_matrix @ u_trial).ravel()
        active_rows = np.asarray(np.abs(c_free).sum(axis=1)).ravel() > 0.0
        inconsistent = np.logical_not(active_rows) & (np.abs(rhs_mpc) > 1.0e-10)
        if np.any(inconsistent):
            raise FEM2DError(f"{stage_name}: MPC constraints conflict with fixed displacement constraints")
        c_active = c_free[active_rows]
        rhs_mpc = rhs_mpc[active_rows]
        if c_active.shape[0]:
            c_aug = bmat([[c_active, csr_matrix((c_active.shape[0], n_p), dtype=float), csr_matrix((c_active.shape[0], 1), dtype=float)]], format="csr")
            system = bmat([[system, c_aug.T], [c_aug, csr_matrix((c_active.shape[0], c_active.shape[0]), dtype=float)]], format="csr")
            rhs = np.concatenate([rhs, rhs_mpc])
        full_multipliers = [0.0 for _ in range(values.size)]
    correction, _info = solve_linear_system(system, rhs, stage_name=stage_name, solver=solver)
    if mpc_info and int(mpc_info.get("count", 0)) > 0:
        constraint_matrix, values = mpc_constraint_matrix(mpc_info, tangent.shape[0], stage_name)
        c_free = constraint_matrix[:, free_u]
        active_rows = np.asarray(np.abs(c_free).sum(axis=1)).ravel() > 0.0
        if np.any(active_rows):
            raw_multipliers = correction[n_u + n_p + 1 :]
            full = np.zeros(values.size, dtype=float)
            full[active_rows] = raw_multipliers
            full_multipliers = [float(value) for value in full]
    return correction[:n_u], correction[n_u : n_u + n_p], float(correction[n_u + n_p]), full_multipliers


def mpc_elimination_requested(mpc_constraints: Any, solver: Mapping[str, Any] | None) -> bool:
    solver_map = solver if isinstance(solver, Mapping) else {}
    solver_mpc = solver_map.get("mpc", {})
    if isinstance(solver_mpc, Mapping):
        method = str(solver_mpc.get("method", solver_mpc.get("type", ""))).lower().strip()
        if method in {"elimination", "exact", "transform"}:
            return True
    for spec in _ensure_list(mpc_constraints):
        if isinstance(spec, Mapping):
            method = str(spec.get("method", spec.get("type", ""))).lower().strip()
            if method in {"elimination", "exact", "transform"}:
                return True
    return False


def mpc_lagrange_requested(mpc_constraints: Any, solver: Mapping[str, Any] | None) -> bool:
    solver_map = solver if isinstance(solver, Mapping) else {}
    solver_mpc = solver_map.get("mpc", {})
    if isinstance(solver_mpc, Mapping):
        method = str(solver_mpc.get("method", solver_mpc.get("type", ""))).lower().strip()
        if method in {"lagrange", "lagrange_multiplier", "multiplier", "lm"}:
            return True
    for spec in _ensure_list(mpc_constraints):
        if isinstance(spec, Mapping):
            method = str(spec.get("method", spec.get("type", ""))).lower().strip()
            if method in {"lagrange", "lagrange_multiplier", "multiplier", "lm"}:
                return True
    return False


def mpc_stage_plan(
    mpc_constraints: Any,
    solver: Mapping[str, Any] | None,
    mpc_info: Mapping[str, Any],
    *,
    nonlinear: bool,
    allow_elimination_linear: bool = True,
    allow_lagrange_linear: bool = True,
    add_plain_penalty_to_stage_matrix: bool = True,
    add_penalty_when_exact_linear_blocked: bool = False,
    add_penalty_when_lagrange_linear_blocked: bool = False,
) -> MPCStagePlan:
    count = int(mpc_info.get("count", 0) or 0)
    active = count > 0
    exact_requested = mpc_elimination_requested(mpc_constraints, solver)
    lagrange_requested = mpc_lagrange_requested(mpc_constraints, solver)
    use_elimination_linear = bool(active and exact_requested and allow_elimination_linear and not nonlinear)
    use_lagrange_linear = bool(active and lagrange_requested and allow_lagrange_linear and not nonlinear)
    plain_penalty = bool(
        active
        and not exact_requested
        and not lagrange_requested
        and add_plain_penalty_to_stage_matrix
        and not (use_elimination_linear or use_lagrange_linear)
    )
    exact_penalty = bool(
        active
        and exact_requested
        and not use_elimination_linear
        and add_penalty_when_exact_linear_blocked
    )
    lagrange_penalty = bool(
        active
        and lagrange_requested
        and not use_lagrange_linear
        and add_penalty_when_lagrange_linear_blocked
    )
    add_penalty = bool(plain_penalty or exact_penalty or lagrange_penalty)
    if not active:
        applied_method = "none"
    elif use_lagrange_linear or (lagrange_requested and not lagrange_penalty):
        applied_method = "lagrange"
    elif use_elimination_linear:
        applied_method = "elimination"
    elif exact_requested or lagrange_penalty:
        applied_method = "penalty_fallback"
    else:
        applied_method = "penalty"
    return MPCStagePlan(
        active=active,
        count=count,
        exact_requested=exact_requested,
        lagrange_requested=lagrange_requested,
        use_elimination_linear=use_elimination_linear,
        use_lagrange_linear=use_lagrange_linear,
        add_penalty_to_stage_matrix=add_penalty,
        applied_method=applied_method,
    )


def mpc_arc_length_stage_plan(
    mpc_constraints: Any,
    solver: Mapping[str, Any] | None,
    mpc_info: Mapping[str, Any],
) -> MPCStagePlan:
    return mpc_stage_plan(
        mpc_constraints,
        solver,
        mpc_info,
        nonlinear=True,
        allow_elimination_linear=False,
        allow_lagrange_linear=False,
        add_plain_penalty_to_stage_matrix=True,
        add_penalty_when_exact_linear_blocked=True,
        add_penalty_when_lagrange_linear_blocked=False,
    )


__all__ = [
    "lagrange_mpc_projected_residual",
    "mpc_arc_length_stage_plan",
    "mpc_constraint_matrix",
    "mpc_elimination_requested",
    "mpc_lagrange_requested",
    "mpc_solver_contract",
    "mpc_stage_plan",
    "MPCStagePlan",
    "solve_arc_length_lagrange_correction",
    "solve_axisymmetric_up_arc_length_correction",
    "solve_lagrange_augmented_system",
    "solve_lagrange_mpc_correction",
    "solve_linear_system_with_mpc_elimination",
    "solve_linear_system_with_mpc_lagrange",
]
