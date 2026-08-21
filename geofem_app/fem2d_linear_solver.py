"""Linear algebra helpers for the 2D FEM solver.

The stage solver owns model orchestration and residual assembly; this module
owns sparse linear-system controls, constrained reduction, and direct LU reuse.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import threading
from time import perf_counter
from typing import Any, Mapping

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import LinearOperator, bicgstab, cg, gmres, spilu, splu, spsolve

from .fem2d_types import FEM2DError
from .reduced_matrix_cache import ReducedMatrixCache, build_reduced_matrix_cache_from_csr


_FACTOR_CACHE_MAX_DEFAULT = 16


@dataclass
class _LUCacheEntry:
    lu: Any
    estimated_bytes: int
    solve_lock: threading.Lock = field(default_factory=threading.Lock)


_FACTOR_CACHE: "OrderedDict[tuple[Any, ...], _LUCacheEntry]" = OrderedDict()
_SYMBOLIC_ORDERING_CACHE: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_FACTOR_PATTERN_STATS: dict[tuple[Any, ...], dict[str, int | bool]] = {}
_FACTOR_CACHE_LOCK = threading.Lock()
_FACTOR_CACHE_CONDITION = threading.Condition(_FACTOR_CACHE_LOCK)
_FACTOR_CACHE_BUILDING: set[tuple[Any, ...]] = set()
_FACTOR_CACHE_HITS = 0
_FACTOR_CACHE_MISSES = 0
_FACTOR_CACHE_BYTES = 0
_FACTOR_CACHE_EVICTIONS = 0
_FACTOR_CACHE_AUTO_SKIPS = 0
_SYMBOLIC_CACHE_HITS = 0
_SYMBOLIC_CACHE_MISSES = 0

_METHOD_ALIASES = {
    "": "direct",
    "spsolve": "direct",
    "superlu": "direct",
    "lu": "direct",
    "automatic": "auto",
    "conjugate_gradient": "cg",
    "bicg": "bicgstab",
    "bi_cgstab": "bicgstab",
    "bi_cg_stab": "bicgstab",
}
_LINEAR_METHODS = {"auto", "direct", "cg", "gmres", "bicgstab"}
_ITERATIVE_METHODS = {"cg", "gmres", "bicgstab"}
_EXTERNAL_DIRECT_METHODS = {"pypardiso", "pardiso", "umfpack", "scikit_umfpack"}
_PRECONDITIONER_ALIASES = {
    "": "none",
    "off": "none",
    "false": "none",
    "no": "none",
    "diag": "jacobi",
    "diagonal": "jacobi",
    "spilu": "ilu",
    "incomplete_lu": "ilu",
    "block": "block_ilu",
    "block_jacobi_ilu": "block_ilu",
    "up_block": "block_ilu",
    "up_block_jacobi": "block_jacobi",
    "up_block_ilu": "block_ilu",
    "pyamg": "amg",
    "algebraic_multigrid": "amg",
    "smoothed_aggregation": "amg",
    "ruge_stuben": "amg",
}
_PRECONDITIONERS = {"none", "jacobi", "ilu", "block_jacobi", "block_ilu", "amg"}
_LARGE_SCALE_PROFILES = {"large", "large_scale", "large_srm", "srm_large", "srm_large_scale"}
_DIRECT_BACKEND_ALIASES = {
    "": "scipy",
    "default": "scipy",
    "scipy": "scipy",
    "superlu": "scipy",
    "spsolve": "scipy",
    "pypardiso": "pypardiso",
    "pardiso": "pypardiso",
    "intel_pardiso": "pypardiso",
    "umfpack": "umfpack",
    "scikit_umfpack": "umfpack",
    "scikits_umfpack": "umfpack",
}


def clear_linear_factor_cache() -> None:
    """Clear cached sparse LU factorizations and symbolic orderings."""

    global _FACTOR_CACHE_HITS, _FACTOR_CACHE_MISSES, _FACTOR_CACHE_BYTES, _FACTOR_CACHE_EVICTIONS, _FACTOR_CACHE_AUTO_SKIPS
    global _SYMBOLIC_CACHE_HITS, _SYMBOLIC_CACHE_MISSES
    with _FACTOR_CACHE_CONDITION:
        _FACTOR_CACHE.clear()
        _FACTOR_PATTERN_STATS.clear()
        _SYMBOLIC_ORDERING_CACHE.clear()
        _FACTOR_CACHE_HITS = 0
        _FACTOR_CACHE_MISSES = 0
        _FACTOR_CACHE_BYTES = 0
        _FACTOR_CACHE_EVICTIONS = 0
        _FACTOR_CACHE_AUTO_SKIPS = 0
        _SYMBOLIC_CACHE_HITS = 0
        _SYMBOLIC_CACHE_MISSES = 0
        _FACTOR_CACHE_CONDITION.notify_all()


def linear_factor_cache_info() -> dict[str, Any]:
    """Return small diagnostics for direct-solver factor and ordering reuse."""

    with _FACTOR_CACHE_LOCK:
        return {
            "entries": len(_FACTOR_CACHE),
            "hits": _FACTOR_CACHE_HITS,
            "misses": _FACTOR_CACHE_MISSES,
            "estimated_bytes": _FACTOR_CACHE_BYTES,
            "estimated_megabytes": float(_FACTOR_CACHE_BYTES) / (1024.0 * 1024.0),
            "evictions": _FACTOR_CACHE_EVICTIONS,
            "auto_skips": _FACTOR_CACHE_AUTO_SKIPS,
            "auto_disabled_patterns": sum(1 for stats in _FACTOR_PATTERN_STATS.values() if bool(stats.get("disabled", False))),
            "inflight_builds": len(_FACTOR_CACHE_BUILDING),
            "solve_lock_scope": "per_lu_entry",
            "symbolic_entries": len(_SYMBOLIC_ORDERING_CACHE),
            "symbolic_hits": _SYMBOLIC_CACHE_HITS,
            "symbolic_misses": _SYMBOLIC_CACHE_MISSES,
        }


def solve_linear_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve a sparse 2D core linear system using direct or iterative controls."""

    settings = linear_solver_settings(solver)
    requested_method = settings["method"]
    if rhs.size == 0:
        return np.zeros(0, dtype=float), {"method": "direct" if requested_method == "auto" else requested_method, "method_requested": requested_method, "iterations": 0, "residual_norm": 0.0}
    matrix_solve = matrix.tocsr()
    rhs_solve = np.asarray(rhs, dtype=float)
    method, auto_info = _select_linear_method(matrix_solve, settings)
    if requested_method == "auto":
        settings = dict(settings)
        settings["method"] = method
        if method != "direct" and str(settings.get("preconditioner", "none")) == "none":
            settings["preconditioner"] = str(settings.get("auto_preconditioner", "jacobi"))
    scale_vec: np.ndarray | None = None
    if settings["equilibrate"]:
        diag_abs = np.abs(matrix_solve.diagonal())
        scale_vec = 1.0 / np.sqrt(np.maximum(diag_abs, settings["equilibration_floor"]))
        Dscale = diags(scale_vec, format="csr")
        matrix_solve = (Dscale @ matrix_solve @ Dscale).tocsr()
        rhs_solve = scale_vec * rhs_solve

    if method == "direct":
        try:
            y, cache_state, symbolic_state = _solve_direct(matrix_solve, rhs_solve, stage_name, settings)
        except FEM2DError as exc:
            if not _direct_fallback_to_iterative(settings, requested_method):
                raise
            fallback_method = str(settings.get("auto_iterative_method", "gmres"))
            fallback_settings = dict(settings)
            fallback_settings["method"] = fallback_method
            if str(fallback_settings.get("preconditioner", "none")) == "none":
                fallback_settings["preconditioner"] = str(fallback_settings.get("auto_preconditioner", "jacobi"))
            x, info = _solve_iterative_system(
                matrix_solve,
                rhs_solve,
                original_matrix=matrix,
                original_rhs=rhs,
                scale_vec=scale_vec,
                method=fallback_method,
                requested_method=requested_method,
                settings=fallback_settings,
                auto_info=auto_info,
                stage_name=stage_name,
            )
            info["direct_fallback"] = {
                "enabled": True,
                "from_method": "direct",
                "selected": fallback_method,
                "reason": str(exc),
            }
            return x, info
        x = scale_vec * y if scale_vec is not None else y
        check_linear_solution(stage_name, method, x)
        residual = float(np.linalg.norm(matrix @ x - rhs))
        return x, {
            "method": method,
            "method_requested": requested_method,
            "iterations": 1,
            "residual_norm": residual,
            "equilibrated": bool(settings["equilibrate"]),
            "factor_cache": cache_state,
            "symbolic_cache": symbolic_state,
            "auto_selection": auto_info,
            "linear_profile": str(settings.get("profile", "")),
        }

    return _solve_iterative_system(
        matrix_solve,
        rhs_solve,
        original_matrix=matrix,
        original_rhs=rhs,
        scale_vec=scale_vec,
        method=method,
        requested_method=requested_method,
        settings=settings,
        auto_info=auto_info,
        stage_name=stage_name,
    )


def _solve_iterative_system(
    matrix_solve: csr_matrix,
    rhs_solve: np.ndarray,
    *,
    original_matrix: csr_matrix,
    original_rhs: np.ndarray,
    scale_vec: np.ndarray | None,
    method: str,
    requested_method: str,
    settings: Mapping[str, Any],
    auto_info: Mapping[str, Any],
    stage_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    iterations = 0

    def count_iteration(_value: Any) -> None:
        nonlocal iterations
        iterations += 1

    kwargs: dict[str, Any] = {
        "rtol": settings["tol_rel"],
        "atol": settings["tol_abs"],
        "maxiter": settings["max_iter"],
        "callback": count_iteration,
    }
    preconditioner, preconditioner_info = _iterative_preconditioner(matrix_solve, settings, stage_name=stage_name)
    if preconditioner is not None:
        kwargs["M"] = preconditioner
    if method == "cg":
        y, info_code = cg(matrix_solve, rhs_solve, **kwargs)
    elif method == "gmres":
        restart = settings["restart"]
        if restart is not None:
            kwargs["restart"] = restart
        kwargs["callback_type"] = "pr_norm"
        y, info_code = gmres(matrix_solve, rhs_solve, **kwargs)
    elif method == "bicgstab":
        y, info_code = bicgstab(matrix_solve, rhs_solve, **kwargs)
    else:
        raise FEM2DError(f"{stage_name}: unsupported 2D linear solver method '{method}'")

    if info_code != 0:
        raise FEM2DError(f"{stage_name}: {method} did not converge, scipy info={info_code}")
    x = scale_vec * y if scale_vec is not None else y
    check_linear_solution(stage_name, method, x)
    residual = float(np.linalg.norm(original_matrix @ x - original_rhs))
    return x, {
        "method": method,
        "method_requested": requested_method,
        "iterations": iterations,
        "residual_norm": residual,
        "equilibrated": bool(settings["equilibrate"]),
        "preconditioner": settings["preconditioner"],
        "preconditioner_info": preconditioner_info,
        "auto_selection": dict(auto_info),
        "linear_profile": str(settings.get("profile", "")),
    }


def _direct_fallback_to_iterative(settings: Mapping[str, Any], requested_method: str) -> bool:
    return bool(settings.get("direct_fallback_to_iterative", requested_method == "auto"))


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled", "none", ""}:
        return False
    return bool(default)


def solve_sparse_with_constraints(
    matrix: csr_matrix,
    rhs: np.ndarray,
    fixed_values: Mapping[int, float],
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
) -> np.ndarray:
    """Solve ``matrix x = rhs`` while enforcing prescribed dof values."""

    n = rhs.size
    if not fixed_values:
        x, _info = solve_linear_system(matrix, rhs, stage_name=stage_name, solver=solver)
        return x
    fixed = np.fromiter((int(key) for key in sorted(fixed_values)), dtype=int)
    if fixed.size and (int(fixed[0]) < 0 or int(fixed[-1]) >= n):
        raise FEM2DError(f"{stage_name}: constrained dof index is outside the solution vector")
    free_mask = np.ones(n, dtype=bool)
    free_mask[fixed] = False
    free = np.flatnonzero(free_mask)
    x = np.zeros(n, dtype=float)
    for idx, value in fixed_values.items():
        x[int(idx)] = float(value)
    if free.size:
        x[free], _info, _cache = solve_reduced_linear_system(
            matrix,
            rhs,
            free,
            fixed,
            fixed_values=x[fixed],
            stage_name=stage_name,
            solver=solver,
            validate_cache=True,
        )
    if not np.all(np.isfinite(x)):
        raise FEM2DError(f"{stage_name}: constrained sparse solve produced non-finite values")
    return x


def solve_reduced_linear_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    free_dofs: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    fixed_values: np.ndarray | None = None,
    reduction_cache: ReducedMatrixCache | None = None,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    validate_cache: bool = True,
) -> tuple[np.ndarray, dict[str, Any], ReducedMatrixCache]:
    """Solve the free-DOF block while reusing CSR submatrix extraction maps."""

    matrix_csr = matrix.tocsr()
    free = np.asarray(free_dofs, dtype=np.int64).ravel()
    fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel()
    cache_reused = bool(
        reduction_cache is not None
        and reduction_cache.matches(matrix_csr, free, fixed, validate_structure=validate_cache)
    )
    reduction_start = perf_counter()
    cache = reduction_cache if cache_reused and reduction_cache is not None else build_reduced_matrix_cache_from_csr(matrix_csr, free, fixed)
    reduced_rhs = cache.reduced_rhs(matrix_csr, rhs, fixed_values)
    reduced_matrix = cache.extract_free_free(matrix_csr)
    reduction_elapsed = max(perf_counter() - reduction_start, 0.0)
    reduced_solver = _solver_with_reduced_block_sizes(solver, free)
    solution, info = solve_linear_system(reduced_matrix, reduced_rhs, stage_name=stage_name, solver=reduced_solver)
    info["reduced_matrix_elapsed_seconds"] = reduction_elapsed
    info["reduced_matrix_cache"] = {
        **cache.info(),
        "reused": cache_reused,
        "built": not cache_reused,
        "validated": bool(validate_cache and reduction_cache is not None),
    }
    return solution, info, cache


def _solver_with_reduced_block_sizes(solver: Mapping[str, Any] | None, free_dofs: np.ndarray) -> Mapping[str, Any] | None:
    if not isinstance(solver, Mapping):
        return solver
    raw_linear = solver.get("linear", solver)
    if not isinstance(raw_linear, Mapping):
        return solver
    if "block_sizes" in raw_linear:
        return solver
    ranges = raw_linear.get("block_dof_ranges")
    if not isinstance(ranges, (list, tuple)):
        return solver
    free = np.asarray(free_dofs, dtype=np.int64).ravel()
    sizes: list[int] = []
    for raw in ranges:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        start, end = int(raw[0]), int(raw[1])
        if end <= start:
            continue
        count = int(np.count_nonzero((free >= start) & (free < end)))
        if count > 0:
            sizes.append(count)
    if not sizes or sum(sizes) != int(free.size):
        return solver
    if "linear" in solver and isinstance(solver.get("linear"), Mapping):
        updated = dict(solver)
        linear = dict(raw_linear)
        linear["block_sizes"] = sizes
        updated["linear"] = linear
        return updated
    linear = dict(raw_linear)
    linear["block_sizes"] = sizes
    return linear


def linear_solver_settings(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    raw: Mapping[str, Any] = solver if isinstance(solver, Mapping) else {}
    linear = raw.get("linear", raw)
    if not isinstance(linear, Mapping):
        linear = {}
    profile = str(linear.get("profile", linear.get("preset", ""))).lower().strip().replace("-", "_")
    large_scale_profile = profile in _LARGE_SCALE_PROFILES
    method_default = "auto" if large_scale_profile else "direct"
    method_raw = str(linear.get("method", linear.get("type", method_default))).lower().strip().replace("-", "_")
    direct_backend_default = method_raw if method_raw in _EXTERNAL_DIRECT_METHODS else "scipy"
    method = "direct" if method_raw in _EXTERNAL_DIRECT_METHODS else _METHOD_ALIASES.get(method_raw, method_raw)
    if method not in _LINEAR_METHODS:
        raise FEM2DError(f"unsupported 2D linear solver method '{method_raw}'")
    preconditioner_raw = str(linear.get("preconditioner", linear.get("precond", "none"))).lower().strip().replace("-", "_")
    preconditioner = _PRECONDITIONER_ALIASES.get(preconditioner_raw, preconditioner_raw)
    if preconditioner not in _PRECONDITIONERS:
        raise FEM2DError(f"unsupported 2D linear solver preconditioner '{preconditioner_raw}'")
    fallback_preconditioner_raw = str(linear.get("preconditioner_fallback", linear.get("amg_fallback", "jacobi"))).lower().strip().replace("-", "_")
    fallback_preconditioner = _PRECONDITIONER_ALIASES.get(fallback_preconditioner_raw, fallback_preconditioner_raw)
    if fallback_preconditioner == "amg":
        fallback_preconditioner = "jacobi"
    if fallback_preconditioner not in _PRECONDITIONERS:
        raise FEM2DError(f"unsupported 2D fallback preconditioner '{fallback_preconditioner_raw}'")
    max_iter_raw = linear.get("max_iter", linear.get("maxiter"))
    restart_raw = linear.get("restart")
    cache_default = _as_bool(linear.get("cache_factorization", linear.get("factor_cache", True)), True)
    symbolic_default = _as_bool(linear.get("cache_symbolic", linear.get("symbolic_cache", True)), True)
    ordering_raw = str(linear.get("symbolic_ordering", linear.get("ordering", "rcm"))).lower().strip().replace("-", "_")
    ordering_aliases = {
        "": "rcm",
        "auto": "rcm",
        "reverse_cuthill_mckee": "rcm",
        "cuthill_mckee": "rcm",
        "natural": "identity",
        "none": "identity",
        "off": "identity",
        "false": "identity",
    }
    symbolic_ordering = ordering_aliases.get(ordering_raw, ordering_raw)
    if symbolic_ordering not in {"rcm", "identity"}:
        raise FEM2DError(f"unsupported 2D symbolic ordering '{ordering_raw}'")
    auto_iterative_raw = str(linear.get("auto_iterative_method", "gmres")).lower().strip().replace("-", "_")
    auto_iterative_method = _METHOD_ALIASES.get(auto_iterative_raw, auto_iterative_raw)
    if auto_iterative_method not in _ITERATIVE_METHODS:
        raise FEM2DError(f"unsupported 2D auto iterative method '{auto_iterative_raw}'")
    auto_preconditioner_default = "ilu" if large_scale_profile else "jacobi"
    auto_preconditioner_raw = str(linear.get("auto_preconditioner", auto_preconditioner_default)).lower().strip().replace("-", "_")
    auto_preconditioner = _PRECONDITIONER_ALIASES.get(auto_preconditioner_raw, auto_preconditioner_raw)
    if auto_preconditioner not in _PRECONDITIONERS:
        raise FEM2DError(f"unsupported 2D auto preconditioner '{auto_preconditioner_raw}'")
    direct_fallback_default = large_scale_profile or method == "auto"
    amg_solver = str(linear.get("amg_solver", "smoothed_aggregation")).lower().strip().replace("-", "_")
    amg_solver_aliases = {
        "": "smoothed_aggregation",
        "sa": "smoothed_aggregation",
        "smoothed": "smoothed_aggregation",
        "aggregation": "smoothed_aggregation",
        "rs": "ruge_stuben",
        "ruge": "ruge_stuben",
    }
    amg_solver = amg_solver_aliases.get(amg_solver, amg_solver)
    if amg_solver not in {"smoothed_aggregation", "ruge_stuben"}:
        raise FEM2DError(f"unsupported 2D AMG solver '{amg_solver}'")
    amg_max_levels_raw = linear.get("amg_max_levels")
    amg_max_coarse_raw = linear.get("amg_max_coarse")
    direct_backend_raw = str(linear.get("direct_backend", linear.get("backend", direct_backend_default))).lower().strip().replace("-", "_")
    direct_backend = _DIRECT_BACKEND_ALIASES.get(direct_backend_raw, direct_backend_raw)
    if direct_backend not in {"scipy", "pypardiso", "umfpack"}:
        raise FEM2DError(f"unsupported 2D direct solver backend '{direct_backend_raw}'")
    return {
        "method": method,
        "profile": profile,
        "tol_rel": float(linear.get("tol_rel", linear.get("rtol", linear.get("tol", 1.0e-8)))),
        "tol_abs": float(linear.get("tol_abs", linear.get("atol", 0.0))),
        "max_iter": None if max_iter_raw is None else int(max_iter_raw),
        "restart": None if restart_raw is None else int(restart_raw),
        "equilibrate": _as_bool(linear.get("equilibrate", linear.get("equilibration", large_scale_profile)), large_scale_profile),
        "equilibration_floor": float(linear.get("equilibration_floor", 1.0e-30)),
        "preconditioner": preconditioner,
        "preconditioner_fallback": fallback_preconditioner,
        "preconditioner_floor": float(linear.get("preconditioner_floor", linear.get("jacobi_floor", 1.0e-30))),
        "ilu_drop_tol": float(linear.get("ilu_drop_tol", linear.get("drop_tol", 1.0e-4))),
        "ilu_fill_factor": float(linear.get("ilu_fill_factor", linear.get("fill_factor", 10.0))),
        "amg_solver": amg_solver,
        "amg_max_levels": None if amg_max_levels_raw is None else int(amg_max_levels_raw),
        "amg_max_coarse": None if amg_max_coarse_raw is None else int(amg_max_coarse_raw),
        "block_sizes": [int(value) for value in linear.get("block_sizes", [])] if isinstance(linear.get("block_sizes", []), (list, tuple)) else [],
        "block_dof_ranges": list(linear.get("block_dof_ranges", [])) if isinstance(linear.get("block_dof_ranges", []), (list, tuple)) else [],
        "auto_iterative_size": int(linear.get("auto_iterative_size", linear.get("auto_size_threshold", 5000))),
        "auto_iterative_nnz": int(linear.get("auto_iterative_nnz", linear.get("auto_nnz_threshold", 200000))),
        "auto_iterative_method": auto_iterative_method,
        "auto_preconditioner": auto_preconditioner,
        "direct_fallback_to_iterative": _as_bool(
            linear.get("fallback_to_iterative", linear.get("direct_fallback_to_iterative", direct_fallback_default)),
            direct_fallback_default,
        ),
        "direct_backend": direct_backend,
        "direct_backend_fallback": _as_bool(linear.get("direct_backend_fallback", linear.get("external_backend_fallback", True)), True),
        "cache_factorization": cache_default,
        "cache_min_size": int(linear.get("cache_min_size", 32)),
        "cache_max_entries": int(linear.get("cache_max_entries", _FACTOR_CACHE_MAX_DEFAULT)),
        "cache_max_memory_mb": float(linear.get("cache_max_memory_mb", linear.get("factor_cache_memory_mb", 512.0))),
        "cache_auto_disable_miss_streak": int(linear.get("cache_auto_disable_miss_streak", 3)),
        "cache_auto_reprobe_interval": int(linear.get("cache_auto_reprobe_interval", 16)),
        "cache_symbolic": symbolic_default,
        "symbolic_cache_min_size": int(linear.get("symbolic_cache_min_size", linear.get("cache_min_size", 32))),
        "symbolic_cache_max_entries": int(linear.get("symbolic_cache_max_entries", linear.get("cache_max_entries", _FACTOR_CACHE_MAX_DEFAULT))),
        "symbolic_ordering": symbolic_ordering,
        "permc_spec": str(linear.get("permc_spec", "COLAMD")).upper(),
    }


def check_linear_solution(stage_name: str, method: str, x: np.ndarray) -> None:
    if not np.all(np.isfinite(x)):
        raise FEM2DError(f"{stage_name}: {method} produced non-finite displacements")


def _select_linear_method(matrix: csr_matrix, settings: Mapping[str, Any]) -> tuple[str, dict[str, Any]]:
    requested = str(settings.get("method", "direct"))
    if requested != "auto":
        return requested, {"enabled": False, "requested": requested, "selected": requested, "profile": str(settings.get("profile", ""))}
    size = int(matrix.shape[0])
    nnz = int(matrix.nnz)
    threshold_size = int(settings.get("auto_iterative_size", 5000))
    threshold_nnz = int(settings.get("auto_iterative_nnz", 200000))
    use_iterative = size >= threshold_size or nnz >= threshold_nnz
    selected = str(settings.get("auto_iterative_method", "gmres")) if use_iterative else "direct"
    return selected, {
        "enabled": True,
        "requested": "auto",
        "selected": selected,
        "profile": str(settings.get("profile", "")),
        "size": size,
        "nnz": nnz,
        "iterative_size_threshold": threshold_size,
        "iterative_nnz_threshold": threshold_nnz,
        "reason": "threshold_exceeded" if use_iterative else "below_threshold",
    }


def _iterative_preconditioner(matrix: csr_matrix, settings: Mapping[str, Any], *, stage_name: str) -> tuple[Any | None, dict[str, Any]]:
    preconditioner = str(settings.get("preconditioner", "none"))
    if preconditioner == "none":
        return None, {"enabled": False, "type": "none"}
    floor = max(float(settings.get("preconditioner_floor", 1.0e-30)), 1.0e-300)
    if preconditioner == "ilu":
        return _ilu_preconditioner(matrix, settings, stage_name=stage_name, label="ilu")
    if preconditioner == "amg":
        return _amg_preconditioner(matrix, settings, stage_name=stage_name)
    if preconditioner in {"block_jacobi", "block_ilu"}:
        return _block_preconditioner(matrix, settings, stage_name=stage_name, block_kind="jacobi" if preconditioner == "block_jacobi" else "ilu")
    if preconditioner != "jacobi":
        return None, {"enabled": False, "type": preconditioner, "reason": "unsupported"}
    diag = np.asarray(matrix.diagonal(), dtype=float)
    safe = diag.copy()
    small = np.abs(safe) < floor
    if np.any(small):
        signs = np.where(safe[small] < 0.0, -1.0, 1.0)
        safe[small] = signs * floor
    return diags(1.0 / safe, format="csr"), {
        "enabled": True,
        "type": "jacobi",
        "size": int(matrix.shape[0]),
        "floor": floor,
        "regularized_diagonal_entries": int(np.count_nonzero(small)),
    }


def _ilu_preconditioner(matrix: csr_matrix, settings: Mapping[str, Any], *, stage_name: str, label: str) -> tuple[LinearOperator, dict[str, Any]]:
    drop_tol = float(settings.get("ilu_drop_tol", 1.0e-4))
    fill_factor = float(settings.get("ilu_fill_factor", 10.0))
    try:
        ilu = spilu(matrix.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    except RuntimeError as exc:
        raise FEM2DError(f"{stage_name}: ILU preconditioner failed: {exc}") from exc
    operator = LinearOperator(matrix.shape, matvec=lambda x: ilu.solve(np.asarray(x, dtype=float)), dtype=float)
    return operator, {
        "enabled": True,
        "type": label,
        "size": int(matrix.shape[0]),
        "drop_tol": drop_tol,
        "fill_factor": fill_factor,
        "nnz_l": int(getattr(ilu, "L").nnz),
        "nnz_u": int(getattr(ilu, "U").nnz),
    }


def _amg_preconditioner(matrix: csr_matrix, settings: Mapping[str, Any], *, stage_name: str) -> tuple[Any | None, dict[str, Any]]:
    pyamg, load_error = _load_pyamg()
    if pyamg is None:
        return _fallback_preconditioner(
            matrix,
            settings,
            stage_name=stage_name,
            requested_type="amg",
            reason=f"pyamg unavailable: {load_error}",
        )

    solver_name = str(settings.get("amg_solver", "smoothed_aggregation"))
    kwargs: dict[str, Any] = {}
    max_levels = settings.get("amg_max_levels")
    max_coarse = settings.get("amg_max_coarse")
    if max_levels is not None:
        kwargs["max_levels"] = int(max_levels)
    if max_coarse is not None:
        kwargs["max_coarse"] = int(max_coarse)
    try:
        if solver_name == "ruge_stuben":
            multilevel = pyamg.ruge_stuben_solver(matrix.tocsr(), **kwargs)
        else:
            multilevel = pyamg.smoothed_aggregation_solver(matrix.tocsr(), **kwargs)
        operator = multilevel.aspreconditioner()
    except Exception as exc:  # pragma: no cover - depends on optional pyamg internals
        return _fallback_preconditioner(
            matrix,
            settings,
            stage_name=stage_name,
            requested_type="amg",
            reason=f"pyamg {solver_name} setup failed: {exc}",
        )
    return operator, {
        "enabled": True,
        "type": "amg",
        "backend": "pyamg",
        "available": True,
        "solver": solver_name,
        "size": int(matrix.shape[0]),
        "levels": int(len(getattr(multilevel, "levels", []))),
    }


def _fallback_preconditioner(
    matrix: csr_matrix,
    settings: Mapping[str, Any],
    *,
    stage_name: str,
    requested_type: str,
    reason: str,
) -> tuple[Any | None, dict[str, Any]]:
    fallback = str(settings.get("preconditioner_fallback", "jacobi"))
    if fallback in {"", "none", requested_type}:
        return None, {
            "enabled": False,
            "type": requested_type,
            "available": False,
            "fallback": "none",
            "reason": reason,
        }
    fallback_settings = dict(settings)
    fallback_settings["preconditioner"] = fallback
    operator, fallback_info = _iterative_preconditioner(matrix, fallback_settings, stage_name=stage_name)
    return operator, {
        "enabled": bool(fallback_info.get("enabled", False)),
        "type": requested_type,
        "available": False,
        "effective_type": fallback_info.get("type", fallback),
        "fallback": fallback_info,
        "reason": reason,
    }


def _load_pyamg() -> tuple[Any | None, Exception | None]:
    try:
        import pyamg  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - optional dependency varies by environment
        return None, exc
    return pyamg, None


def _block_preconditioner(matrix: csr_matrix, settings: Mapping[str, Any], *, stage_name: str, block_kind: str) -> tuple[LinearOperator | csr_matrix, dict[str, Any]]:
    sizes = [int(value) for value in settings.get("block_sizes", []) if int(value) > 0]
    n = int(matrix.shape[0])
    if not sizes or sum(sizes) != n:
        return _ilu_preconditioner(matrix, settings, stage_name=stage_name, label=f"block_{block_kind}_fallback")
    starts = np.cumsum([0, *sizes[:-1]])
    if block_kind == "jacobi":
        pieces = []
        regularized = 0
        floor = max(float(settings.get("preconditioner_floor", 1.0e-30)), 1.0e-300)
        for start, size in zip(starts, sizes, strict=True):
            block_diag = np.asarray(matrix[start : start + size, start : start + size].diagonal(), dtype=float)
            safe = block_diag.copy()
            small = np.abs(safe) < floor
            if np.any(small):
                signs = np.where(safe[small] < 0.0, -1.0, 1.0)
                safe[small] = signs * floor
                regularized += int(np.count_nonzero(small))
            pieces.append(1.0 / safe)
        return diags(np.concatenate(pieces), format="csr"), {
            "enabled": True,
            "type": "block_jacobi",
            "block_count": len(sizes),
            "block_sizes": sizes,
            "regularized_diagonal_entries": regularized,
        }

    solvers = []
    block_infos: list[dict[str, Any]] = []
    drop_tol = float(settings.get("ilu_drop_tol", 1.0e-4))
    fill_factor = float(settings.get("ilu_fill_factor", 10.0))
    for index, (start, size) in enumerate(zip(starts, sizes, strict=True)):
        block = matrix[start : start + size, start : start + size].tocsc()
        try:
            ilu = spilu(block, drop_tol=drop_tol, fill_factor=fill_factor)
        except RuntimeError as exc:
            raise FEM2DError(f"{stage_name}: block ILU preconditioner failed for block {index}: {exc}") from exc
        solvers.append((int(start), int(size), ilu))
        block_infos.append({"size": int(size), "nnz_l": int(ilu.L.nnz), "nnz_u": int(ilu.U.nnz)})

    def solve_blocks(vector: np.ndarray) -> np.ndarray:
        raw = np.asarray(vector, dtype=float)
        out = np.zeros_like(raw)
        for start, size, ilu in solvers:
            out[start : start + size] = ilu.solve(raw[start : start + size])
        return out

    return LinearOperator(matrix.shape, matvec=solve_blocks, dtype=float), {
        "enabled": True,
        "type": "block_ilu",
        "block_count": len(sizes),
        "block_sizes": sizes,
        "drop_tol": drop_tol,
        "fill_factor": fill_factor,
        "blocks": block_infos,
    }


def _solve_direct(matrix: csr_matrix, rhs: np.ndarray, stage_name: str, settings: Mapping[str, Any]) -> tuple[np.ndarray, str, dict[str, Any]]:
    backend = str(settings.get("direct_backend", "scipy"))
    if backend != "scipy":
        return _solve_external_direct(matrix, rhs, stage_name, settings, backend=backend)
    return _solve_direct_scipy(matrix, rhs, stage_name, settings)


def _solve_direct_scipy(matrix: csr_matrix, rhs: np.ndarray, stage_name: str, settings: Mapping[str, Any]) -> tuple[np.ndarray, str, dict[str, Any]]:
    use_cache = bool(settings.get("cache_factorization", True)) and matrix.shape[0] >= int(settings.get("cache_min_size", 32))
    solve_matrix, solve_rhs, restore_permutation, symbolic_state, permc_spec = _apply_symbolic_ordering_cache(matrix, rhs, settings)
    if not use_cache:
        try:
            if symbolic_state.get("enabled", False):
                lu = splu(solve_matrix.tocsc(), permc_spec=permc_spec)
                solved = np.asarray(lu.solve(solve_rhs), dtype=float)
                return _restore_symbolic_solution(solved, restore_permutation), "disabled", symbolic_state
            return np.asarray(spsolve(matrix, rhs, permc_spec=permc_spec), dtype=float), "disabled", symbolic_state
        except RuntimeError as exc:
            raise FEM2DError(f"{stage_name}: direct solver failed: {exc}") from exc
    cache_entry, cache_state = _cached_lu(
        solve_matrix,
        int(settings.get("cache_max_entries", _FACTOR_CACHE_MAX_DEFAULT)),
        permc_spec=permc_spec,
        max_memory_mb=float(settings.get("cache_max_memory_mb", 512.0)),
        auto_disable_miss_streak=int(settings.get("cache_auto_disable_miss_streak", 3)),
        auto_reprobe_interval=int(settings.get("cache_auto_reprobe_interval", 16)),
    )
    try:
        with cache_entry.solve_lock:
            solved = np.asarray(cache_entry.lu.solve(solve_rhs), dtype=float)
            return _restore_symbolic_solution(solved, restore_permutation), cache_state, symbolic_state
    except RuntimeError as exc:
        raise FEM2DError(f"{stage_name}: cached direct solver failed: {exc}") from exc


def _solve_external_direct(
    matrix: csr_matrix,
    rhs: np.ndarray,
    stage_name: str,
    settings: Mapping[str, Any],
    *,
    backend: str,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    solver_fn, load_error = _load_external_direct_solver(backend)
    if solver_fn is None:
        if _as_bool(settings.get("direct_backend_fallback", True), True):
            return _solve_direct_backend_fallback(matrix, rhs, stage_name, settings, backend, f"{load_error}")
        raise FEM2DError(f"{stage_name}: direct solver backend '{backend}' is unavailable: {load_error}")
    try:
        solved = np.asarray(solver_fn(matrix.tocsr(), rhs), dtype=float)
    except Exception as exc:
        if _as_bool(settings.get("direct_backend_fallback", True), True):
            return _solve_direct_backend_fallback(matrix, rhs, stage_name, settings, backend, f"{exc}")
        raise FEM2DError(f"{stage_name}: direct solver backend '{backend}' failed: {exc}") from exc
    return solved, backend, {
        "enabled": False,
        "state": "external",
        "direct_backend": {
            "requested": backend,
            "used": backend,
            "fallback": False,
        },
    }


def _solve_direct_backend_fallback(
    matrix: csr_matrix,
    rhs: np.ndarray,
    stage_name: str,
    settings: Mapping[str, Any],
    backend: str,
    reason: str,
) -> tuple[np.ndarray, str, dict[str, Any]]:
    fallback_settings = dict(settings)
    fallback_settings["direct_backend"] = "scipy"
    solved, cache_state, symbolic_state = _solve_direct_scipy(matrix, rhs, stage_name, fallback_settings)
    symbolic_info = dict(symbolic_state)
    symbolic_info["direct_backend"] = {
        "requested": backend,
        "used": "scipy",
        "fallback": True,
        "reason": reason,
    }
    return solved, f"{backend}_fallback_{cache_state}", symbolic_info


def _load_external_direct_solver(backend: str) -> tuple[Any | None, Exception | None]:
    if backend == "pypardiso":
        try:
            from pypardiso import spsolve as pardiso_spsolve  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency varies by environment
            return None, exc
        return pardiso_spsolve, None
    if backend == "umfpack":
        try:
            from scikits.umfpack import spsolve as umfpack_spsolve  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - optional dependency varies by environment
            return None, exc
        return umfpack_spsolve, None
    return None, FEM2DError(f"unsupported backend '{backend}'")


def _cached_lu(
    matrix: csr_matrix,
    max_entries: int,
    *,
    permc_spec: str = "COLAMD",
    max_memory_mb: float = 512.0,
    auto_disable_miss_streak: int = 3,
    auto_reprobe_interval: int = 16,
) -> tuple[_LUCacheEntry, str]:
    global _FACTOR_CACHE_HITS, _FACTOR_CACHE_MISSES, _FACTOR_CACHE_BYTES, _FACTOR_CACHE_EVICTIONS, _FACTOR_CACHE_AUTO_SKIPS
    pattern_key = (*_sparse_pattern_key(matrix), str(permc_spec))
    miss_limit = max(int(auto_disable_miss_streak), 0)
    reprobe_interval = max(int(auto_reprobe_interval), 1)
    with _FACTOR_CACHE_LOCK:
        stats = _FACTOR_PATTERN_STATS.setdefault(pattern_key, {"miss_streak": 0, "skips": 0, "disabled": False})
        auto_disabled = bool(stats.get("disabled", False))
        if auto_disabled:
            stats["skips"] = int(stats.get("skips", 0)) + 1
            should_probe = int(stats["skips"]) % reprobe_interval == 0
            if not should_probe:
                _FACTOR_CACHE_AUTO_SKIPS += 1
    if auto_disabled and not should_probe:
        lu = splu(matrix.tocsc(), permc_spec=permc_spec)
        return _LUCacheEntry(lu=lu, estimated_bytes=_estimate_lu_bytes(lu)), "auto_disabled"

    key = (*_sparse_matrix_key(matrix), str(permc_spec))
    with _FACTOR_CACHE_CONDITION:
        cached = _FACTOR_CACHE.get(key)
        if cached is not None:
            _FACTOR_CACHE.move_to_end(key)
            _FACTOR_CACHE_HITS += 1
            stats = _FACTOR_PATTERN_STATS.setdefault(pattern_key, {"miss_streak": 0, "skips": 0, "disabled": False})
            stats.update({"miss_streak": 0, "skips": 0, "disabled": False})
            return cached, "hit"
        while key in _FACTOR_CACHE_BUILDING:
            _FACTOR_CACHE_CONDITION.wait()
            cached = _FACTOR_CACHE.get(key)
            if cached is not None:
                _FACTOR_CACHE.move_to_end(key)
                _FACTOR_CACHE_HITS += 1
                stats = _FACTOR_PATTERN_STATS.setdefault(pattern_key, {"miss_streak": 0, "skips": 0, "disabled": False})
                stats.update({"miss_streak": 0, "skips": 0, "disabled": False})
                return cached, "hit"
        _FACTOR_CACHE_BUILDING.add(key)
    try:
        lu = splu(matrix.tocsc(), permc_spec=permc_spec)
    except BaseException:
        with _FACTOR_CACHE_CONDITION:
            _FACTOR_CACHE_BUILDING.discard(key)
            _FACTOR_CACHE_CONDITION.notify_all()
        raise
    entry = _LUCacheEntry(lu=lu, estimated_bytes=_estimate_lu_bytes(lu))
    max_bytes = max(int(float(max_memory_mb) * 1024.0 * 1024.0), 0)
    with _FACTOR_CACHE_CONDITION:
        cached = _FACTOR_CACHE.get(key)
        if cached is not None:
            _FACTOR_CACHE.move_to_end(key)
            _FACTOR_CACHE_HITS += 1
            stats = _FACTOR_PATTERN_STATS.setdefault(pattern_key, {"miss_streak": 0, "skips": 0, "disabled": False})
            stats.update({"miss_streak": 0, "skips": 0, "disabled": False})
            _FACTOR_CACHE_BUILDING.discard(key)
            _FACTOR_CACHE_CONDITION.notify_all()
            return cached, "hit"
        _FACTOR_CACHE_MISSES += 1
        stats = _FACTOR_PATTERN_STATS.setdefault(pattern_key, {"miss_streak": 0, "skips": 0, "disabled": False})
        stats["miss_streak"] = int(stats.get("miss_streak", 0)) + 1
        if miss_limit > 0 and int(stats["miss_streak"]) >= miss_limit:
            stats["disabled"] = True
            stats["skips"] = 0
        if max_bytes > 0 and entry.estimated_bytes > max_bytes:
            _FACTOR_CACHE_BUILDING.discard(key)
            _FACTOR_CACHE_CONDITION.notify_all()
            return entry, "oversize"
        _FACTOR_CACHE[key] = entry
        _FACTOR_CACHE.move_to_end(key)
        _FACTOR_CACHE_BYTES += entry.estimated_bytes
        limit = max(1, max_entries)
        while len(_FACTOR_CACHE) > limit or (max_bytes > 0 and _FACTOR_CACHE_BYTES > max_bytes):
            _old_key, old_entry = _FACTOR_CACHE.popitem(last=False)
            _FACTOR_CACHE_BYTES = max(0, _FACTOR_CACHE_BYTES - int(old_entry.estimated_bytes))
            _FACTOR_CACHE_EVICTIONS += 1
        _FACTOR_CACHE_BUILDING.discard(key)
        _FACTOR_CACHE_CONDITION.notify_all()
    return entry, "miss"


def _estimate_lu_bytes(lu: Any) -> int:
    total = 0
    for sparse_part_name in ("L", "U"):
        sparse_part = getattr(lu, sparse_part_name, None)
        for array_name in ("data", "indices", "indptr"):
            array = getattr(sparse_part, array_name, None)
            if isinstance(array, np.ndarray):
                total += int(array.nbytes)
    for permutation_name in ("perm_r", "perm_c"):
        permutation = getattr(lu, permutation_name, None)
        if isinstance(permutation, np.ndarray):
            total += int(permutation.nbytes)
    return total


def _apply_symbolic_ordering_cache(
    matrix: csr_matrix,
    rhs: np.ndarray,
    settings: Mapping[str, Any],
) -> tuple[csr_matrix, np.ndarray, np.ndarray | None, dict[str, Any], str]:
    default_permc = _normalized_permc_spec(settings.get("permc_spec", "COLAMD"))
    enabled = bool(settings.get("cache_symbolic", True)) and matrix.shape[0] >= int(settings.get("symbolic_cache_min_size", 32))
    if not enabled:
        return matrix, rhs, None, {"enabled": False, "state": "disabled"}, default_permc
    ordering = str(settings.get("symbolic_ordering", "rcm"))
    entry, state = _cached_symbolic_ordering(matrix, ordering, int(settings.get("symbolic_cache_max_entries", _FACTOR_CACHE_MAX_DEFAULT)))
    permutation = np.asarray(entry["permutation"], dtype=np.int64)
    info = {
        "enabled": True,
        "state": state,
        "ordering": str(entry.get("ordering", ordering)),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "source_nnz": int(matrix.nnz),
        "permuted": bool(entry.get("permuted", False)),
        "permc_spec": "NATURAL" if bool(entry.get("permuted", False)) else default_permc,
        "direct_fill_permutation": bool(entry.get("permuted_source_positions") is not None),
    }
    if not bool(entry.get("permuted", False)):
        return matrix, rhs, None, info, default_permc
    source_positions = entry.get("permuted_source_positions")
    if isinstance(source_positions, np.ndarray):
        permuted_matrix = csr_matrix(
            (
                np.asarray(matrix.data)[source_positions].copy(),
                np.asarray(entry["permuted_indices"], dtype=matrix.indices.dtype),
                np.asarray(entry["permuted_indptr"], dtype=matrix.indptr.dtype),
            ),
            shape=matrix.shape,
        )
    else:
        permuted_matrix = matrix[permutation, :][:, permutation].tocsr()
    return permuted_matrix, np.asarray(rhs, dtype=float)[permutation], permutation, info, "NATURAL"


def _restore_symbolic_solution(solution: np.ndarray, permutation: np.ndarray | None) -> np.ndarray:
    if permutation is None:
        return solution
    restored = np.empty_like(solution)
    restored[permutation] = solution
    return restored


def _cached_symbolic_ordering(matrix: csr_matrix, ordering: str, max_entries: int) -> tuple[dict[str, Any], str]:
    global _SYMBOLIC_CACHE_HITS, _SYMBOLIC_CACHE_MISSES
    key = (*_sparse_pattern_key(matrix), ordering)
    with _FACTOR_CACHE_LOCK:
        cached = _SYMBOLIC_ORDERING_CACHE.get(key)
        if cached is not None:
            _SYMBOLIC_ORDERING_CACHE.move_to_end(key)
            _SYMBOLIC_CACHE_HITS += 1
            return cached, "hit"
    permutation = _build_symbolic_permutation(matrix, ordering)
    identity = np.arange(matrix.shape[0], dtype=np.int64)
    entry = {
        "ordering": ordering,
        "permutation": permutation,
        "permuted": bool(permutation.size and not np.array_equal(permutation, identity)),
    }
    if bool(entry["permuted"]):
        template = _build_permuted_csr_template(matrix, permutation)
        if template is not None:
            entry.update(template)
    with _FACTOR_CACHE_LOCK:
        cached = _SYMBOLIC_ORDERING_CACHE.get(key)
        if cached is not None:
            _SYMBOLIC_ORDERING_CACHE.move_to_end(key)
            _SYMBOLIC_CACHE_HITS += 1
            return cached, "hit"
        _SYMBOLIC_ORDERING_CACHE[key] = entry
        _SYMBOLIC_ORDERING_CACHE.move_to_end(key)
        _SYMBOLIC_CACHE_MISSES += 1
        limit = max(1, max_entries)
        while len(_SYMBOLIC_ORDERING_CACHE) > limit:
            _SYMBOLIC_ORDERING_CACHE.popitem(last=False)
    return entry, "miss"


def _build_permuted_csr_template(matrix: csr_matrix, permutation: np.ndarray) -> dict[str, np.ndarray] | None:
    csr = matrix.tocsr()
    if not bool(csr.has_canonical_format) or csr.nnz == 0:
        return None
    source_markers = np.arange(1, csr.nnz + 1, dtype=np.float64)
    marker_matrix = csr_matrix((source_markers, csr.indices.copy(), csr.indptr.copy()), shape=csr.shape)
    permuted_markers = marker_matrix[permutation, :][:, permutation].tocsr()
    source_positions = np.rint(np.asarray(permuted_markers.data, dtype=float)).astype(np.int64) - 1
    if (
        source_positions.size != csr.nnz
        or np.any(source_positions < 0)
        or np.any(source_positions >= csr.nnz)
        or np.unique(source_positions).size != csr.nnz
    ):
        return None
    return {
        "permuted_source_positions": source_positions,
        "permuted_indices": permuted_markers.indices.copy(),
        "permuted_indptr": permuted_markers.indptr.copy(),
    }


def _build_symbolic_permutation(matrix: csr_matrix, ordering: str) -> np.ndarray:
    n = int(matrix.shape[0])
    if ordering == "identity" or n <= 1:
        return np.arange(n, dtype=np.int64)
    if ordering == "rcm":
        return np.asarray(reverse_cuthill_mckee(matrix, symmetric_mode=False), dtype=np.int64)
    return np.arange(n, dtype=np.int64)


def _normalized_permc_spec(raw: Any) -> str:
    value = str(raw or "COLAMD").upper().replace("-", "_")
    aliases = {
        "COL_AMD": "COLAMD",
        "MMD_ATA": "MMD_ATA",
        "MMD_AT_PLUS_A": "MMD_AT_PLUS_A",
        "NATURAL": "NATURAL",
    }
    return aliases.get(value, "COLAMD")


def _sparse_matrix_key(matrix: csr_matrix) -> tuple[Any, ...]:
    csr = matrix.tocsr()
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.asarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.indices, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.data, dtype=np.float64).tobytes())
    return (csr.shape, csr.nnz, str(csr.dtype), digest.hexdigest())


def _sparse_pattern_key(matrix: csr_matrix) -> tuple[Any, ...]:
    csr = matrix.tocsr()
    digest = hashlib.blake2b(digest_size=20)
    digest.update(np.asarray(csr.indptr, dtype=np.int64).tobytes())
    digest.update(np.asarray(csr.indices, dtype=np.int64).tobytes())
    return (csr.shape, csr.nnz, str(csr.dtype), digest.hexdigest())


__all__ = [
    "check_linear_solution",
    "clear_linear_factor_cache",
    "linear_factor_cache_info",
    "linear_solver_settings",
    "solve_linear_system",
    "solve_reduced_linear_system",
    "solve_sparse_with_constraints",
]
