"""Representative Numba warmup helpers for benchmarks and GUI startup."""

from __future__ import annotations

from dataclasses import dataclass
import os
from threading import Lock
import time
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .fem2d_elements import (
    _quad4_axisymmetric_biot_matrix_fast,
    _quad4_axisymmetric_element_stiffness_fast,
    _quad4_axisymmetric_j2dp_tangent_force_fast,
    _quad4_axisymmetric_pressure_matrices_fast,
    _quad4_biot_matrix_fast,
    _quad4_consistent_mass_matrix_fast,
    _quad4_element_stiffness_fast,
    _quad4_j2dp_tangent_force_fast,
    _quad4_j2dp_post_fast,
    _quad4_mc_post_fast,
    _quad4_pressure_matrices_fast,
    _quad8_axisymmetric_biot_matrix_fast,
    _quad8_axisymmetric_element_stiffness_fast,
    _quad8_axisymmetric_j2dp_tangent_force_fast,
    _quad8_axisymmetric_pressure_matrices_fast,
    _quad8_biot_matrix_fast,
    _quad8_consistent_mass_matrix_fast,
    _quad8_element_stiffness_fast,
    _quad8_j2dp_tangent_force_fast,
    _quad8_j2dp_post_fast,
    _quad8_mc_post_fast,
    _quad8_pressure_matrices_fast,
    _quad8_elastic_post_fast,
)
from .fem2d_large_deformation import _updated_coords_numba
from .fem2d_element_mohr_coulomb_kernels import (
    _quad4_mc_stress_tangent_state_regularized_numba,
)
from .fem2d_materials import _yield_surface_parameters
from .fem2d_plastic_batch import _build_mc_material_arrays
from .fem2d_pressure import _tri_biot_matrix_fast, _tri_pressure_matrices_fast
from .fem2d_structural_assembly import _tri_consistent_mass_matrix_fast
from .fem2d_types import ElasticPlaneStrainMaterial


@dataclass(frozen=True)
class NumbaWarmupKernel:
    name: str
    fn: Callable[[], Any]
    profiles: tuple[str, ...] = ("benchmark", "gui")


_WARMUP_LOCK = Lock()
_WARMUP_CACHE: dict[tuple[str, tuple[str, ...]], dict[str, Any]] = {}


def gui_numba_warmup_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    if values.get("GEOFEM_GUI_NUMBA_WARMUP") is not None:
        return _env_flag(values, "GEOFEM_GUI_NUMBA_WARMUP", default=True)
    if str(values.get("QT_QPA_PLATFORM", "")).strip().lower() == "offscreen":
        return False
    return _env_flag(values, "GEOFEM_NUMBA_WARMUP", default=True)


def benchmark_numba_warmup_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = os.environ if env is None else env
    return _env_flag(values, "GEOFEM_BENCHMARK_NUMBA_WARMUP", default=_env_flag(values, "GEOFEM_NUMBA_WARMUP", default=True))


def warmup_numba_kernels(
    *,
    profile: str = "benchmark",
    kernels: Iterable[NumbaWarmupKernel] | None = None,
    force: bool = False,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    selected = _select_kernels(profile, tuple(kernels) if kernels is not None else _representative_numba_kernels())
    names = tuple(kernel.name for kernel in selected)
    cache_key = (str(profile), names)
    if not force:
        with _WARMUP_LOCK:
            cached = _WARMUP_CACHE.get(cache_key)
        if cached is not None:
            return {
                **cached,
                "elapsed_seconds": 0.0,
                "cached": True,
                "already_warmed": True,
                "previous_elapsed_seconds": cached.get("elapsed_seconds", 0.0),
            }

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for kernel in selected:
        k0 = time.perf_counter()
        ok = True
        error = ""
        try:
            kernel.fn()
        except Exception as exc:
            ok = False
            error = str(exc)
            if raise_on_error:
                raise
        rows.append(
            {
                "name": kernel.name,
                "ok": ok,
                "elapsed_seconds": time.perf_counter() - k0,
                "error": error,
            }
        )
    elapsed = time.perf_counter() - started
    summary = {
        "schema": "geofem.numba_warmup.v1",
        "enabled": True,
        "profile": str(profile),
        "scope": "representative_numba_kernels",
        "cached": False,
        "already_warmed": False,
        "elapsed_seconds": elapsed,
        "kernel_count": len(rows),
        "warmed_count": sum(1 for row in rows if row["ok"]),
        "failed_count": sum(1 for row in rows if not row["ok"]),
        "kernels": rows,
    }
    if summary["failed_count"] == 0:
        with _WARMUP_LOCK:
            _WARMUP_CACHE[cache_key] = summary
    return summary


def skipped_numba_warmup_summary(*, profile: str, reason: str) -> dict[str, Any]:
    return {
        "schema": "geofem.numba_warmup.v1",
        "enabled": False,
        "profile": str(profile),
        "scope": "representative_numba_kernels",
        "cached": False,
        "already_warmed": False,
        "elapsed_seconds": 0.0,
        "kernel_count": 0,
        "warmed_count": 0,
        "failed_count": 0,
        "skip_reason": str(reason),
        "kernels": [],
    }


def clear_numba_warmup_state() -> None:
    with _WARMUP_LOCK:
        _WARMUP_CACHE.clear()


def _select_kernels(profile: str, kernels: tuple[NumbaWarmupKernel, ...]) -> tuple[NumbaWarmupKernel, ...]:
    profile_norm = str(profile or "benchmark").lower().strip()
    if profile_norm in {"all", "*"}:
        return kernels
    return tuple(kernel for kernel in kernels if profile_norm in {str(item).lower().strip() for item in kernel.profiles})


def _env_flag(values: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = values.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _representative_numba_kernels() -> tuple[NumbaWarmupKernel, ...]:
    coords4 = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=float)
    coords4_axis = coords4.copy()
    coords4_axis[:, 0] += 1.0
    coords8 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [0.0, 0.5],
        ],
        dtype=float,
    )
    coords8_axis = coords8.copy()
    coords8_axis[:, 0] += 1.0
    coords3 = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=float)
    coords3_axis = coords3.copy()
    coords3_axis[:, 0] += 1.0
    coords6 = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.0],
            [0.5, 0.5],
            [0.0, 0.5],
        ],
        dtype=float,
    )
    coords6_axis = coords6.copy()
    coords6_axis[:, 0] += 1.0
    ue4 = _benchmark_displacements(4)
    ue8 = _benchmark_displacements(8)
    initial = np.zeros(4, dtype=float)
    elastic = ElasticPlaneStrainMaterial("numba_warmup_elastic", E=50000.0, nu=0.30, gamma=18.0)
    plastic = ElasticPlaneStrainMaterial(
        "numba_warmup_plastic",
        E=50000.0,
        nu=0.30,
        gamma=18.0,
        model="drucker_prager",
        cohesion=80.0,
        friction_angle=28.0,
    )
    mohr_coulomb = ElasticPlaneStrainMaterial(
        "numba_warmup_mohr_coulomb",
        E=50000.0,
        nu=0.30,
        gamma=18.0,
        model="mohr_coulomb",
        cohesion=80.0,
        friction_angle=28.0,
    )
    mohr_coulomb_regularized = ElasticPlaneStrainMaterial(
        "numba_warmup_mohr_coulomb_regularized",
        E=14000.0,
        nu=0.30,
        model="mohr_coulomb",
        cohesion=10.0,
        friction_angle=25.0,
        dilation_angle=0.0,
    )
    mc_regularized_factor = 1.340625
    (
        mc_d4,
        mc_s4,
        mc_yield,
        mc_flow,
        mc_cohesion,
        mc_hardening,
        _mc_thickness,
        _mc_operator_indices,
        mc_operator1,
        mc_operator2,
        mc_operator3,
        mc_candidate_h,
    ) = _build_mc_material_arrays([mohr_coulomb_regularized], mc_regularized_factor)
    mc_regularized_strain = np.linalg.solve(
        mohr_coulomb_regularized.D4,
        np.array([-30.322152808175474, 199.43875977836774, 258.087431004466, 0.0]),
    )
    alpha, cohesion_term = _yield_surface_parameters(plastic, 1.0)
    p4 = np.zeros((4, 4), dtype=float)
    k4 = np.zeros(4, dtype=float)
    p8 = np.zeros((9, 4), dtype=float)
    k8 = np.zeros(9, dtype=float)
    p8_sri = np.zeros((13, 4), dtype=float)
    k8_sri = np.zeros(13, dtype=float)
    displacement = np.zeros(coords8.shape[0] * 2, dtype=float)
    return (
        NumbaWarmupKernel("quad4_elastic_stiffness_full", lambda: _quad4_element_stiffness_fast(coords4, elastic, "FULL")),
        NumbaWarmupKernel("quad4_elastic_stiffness_bbar", lambda: _quad4_element_stiffness_fast(coords4, elastic, "B-BAR"), ("benchmark",)),
        NumbaWarmupKernel("quad8_elastic_stiffness_full", lambda: _quad8_element_stiffness_fast(coords8, elastic, "FULL")),
        NumbaWarmupKernel("quad8_elastic_stiffness_sri", lambda: _quad8_element_stiffness_fast(coords8, elastic, "SRI"), ("benchmark",)),
        NumbaWarmupKernel("quad8_elastic_stiffness_bbar", lambda: _quad8_element_stiffness_fast(coords8, elastic, "B-BAR"), ("benchmark",)),
        NumbaWarmupKernel(
            "quad4_j2dp_tangent_force",
            lambda: _quad4_j2dp_tangent_force_fast(coords4, ue4, plastic, initial_stress=initial, plastic_strains=p4, kappas=k4, alpha=alpha, cohesion_term=cohesion_term),
        ),
        NumbaWarmupKernel(
            "quad8_j2dp_tangent_force_full",
            lambda: _quad8_j2dp_tangent_force_fast(coords8, ue8, plastic, "FULL", initial_stress=initial, plastic_strains=p8, kappas=k8, alpha=alpha, cohesion_term=cohesion_term),
        ),
        NumbaWarmupKernel(
            "quad8_j2dp_tangent_force_sri",
            lambda: _quad8_j2dp_tangent_force_fast(coords8, ue8, plastic, "SRI", initial_stress=initial, plastic_strains=p8_sri, kappas=k8_sri, alpha=alpha, cohesion_term=cohesion_term),
            ("benchmark",),
        ),
        NumbaWarmupKernel(
            "quad4_j2dp_post",
            lambda: _quad4_j2dp_post_fast(coords4, ue4, plastic, initial_stress=initial, plastic_strains=p4, kappas=k4, alpha=alpha, cohesion_term=cohesion_term),
        ),
        NumbaWarmupKernel(
            "quad8_j2dp_post_full",
            lambda: _quad8_j2dp_post_fast(coords8, ue8, plastic, initial_stress=initial, plastic_strains=p8, kappas=k8, alpha=alpha, cohesion_term=cohesion_term),
            ("benchmark",),
        ),
        NumbaWarmupKernel("quad4_mohr_coulomb_post", lambda: _quad4_mc_post_fast(coords4, ue4, mohr_coulomb, initial_stress=initial, plastic_strains=p4, kappas=k4, strength_factor=1.0), ("benchmark",)),
        NumbaWarmupKernel(
            "quad4_mohr_coulomb_regularized_projection",
            lambda: _quad4_mc_stress_tangent_state_regularized_numba(
                mc_regularized_strain,
                np.zeros(4, dtype=float),
                0.0,
                mc_d4[0],
                mc_s4[0],
                initial,
                mc_yield[0],
                mc_flow[0],
                mc_cohesion[0],
                mc_hardening[0],
                mc_operator1[0],
                mc_operator2[0],
                mc_operator3[0],
                mc_candidate_h[0],
            ),
        ),
        NumbaWarmupKernel("quad8_mohr_coulomb_post", lambda: _quad8_mc_post_fast(coords8, ue8, mohr_coulomb, initial_stress=initial, plastic_strains=p8, kappas=k8, strength_factor=1.0), ("benchmark",)),
        NumbaWarmupKernel("quad8_elastic_post_full", lambda: _quad8_elastic_post_fast(coords8, ue8, elastic, initial_stress=initial), ("benchmark",)),
        NumbaWarmupKernel("quad4_pressure_matrices", lambda: _quad4_pressure_matrices_fast(coords4, elastic, storage=1.0e-5, permeability=1.0e-6)),
        NumbaWarmupKernel("quad8_pressure_matrices", lambda: _quad8_pressure_matrices_fast(coords8, elastic, storage=1.0e-5, permeability=1.0e-6), ("benchmark",)),
        NumbaWarmupKernel("tri3_pressure_matrices", lambda: _tri_pressure_matrices_fast(coords3, elastic, storage=1.0e-5, permeability=1.0e-6, axisymmetric=False)),
        NumbaWarmupKernel("tri6_pressure_matrices", lambda: _tri_pressure_matrices_fast(coords6, elastic, storage=1.0e-5, permeability=1.0e-6, axisymmetric=False), ("benchmark",)),
        NumbaWarmupKernel("quad4_biot_matrix", lambda: _quad4_biot_matrix_fast(coords4, elastic, 1.0)),
        NumbaWarmupKernel("quad8_biot_matrix", lambda: _quad8_biot_matrix_fast(coords8, elastic, 1.0, "FULL"), ("benchmark",)),
        NumbaWarmupKernel("tri3_biot_matrix", lambda: _tri_biot_matrix_fast(coords3, elastic, 1.0, axisymmetric=False)),
        NumbaWarmupKernel("tri6_biot_matrix", lambda: _tri_biot_matrix_fast(coords6, elastic, 1.0, axisymmetric=False), ("benchmark",)),
        NumbaWarmupKernel("quad4_consistent_mass", lambda: _quad4_consistent_mass_matrix_fast(coords4, elastic, 1800.0)),
        NumbaWarmupKernel("quad8_consistent_mass", lambda: _quad8_consistent_mass_matrix_fast(coords8, elastic, 1800.0), ("benchmark",)),
        NumbaWarmupKernel("tri3_consistent_mass", lambda: _tri_consistent_mass_matrix_fast(coords3, elastic, 1800.0)),
        NumbaWarmupKernel("tri6_consistent_mass", lambda: _tri_consistent_mass_matrix_fast(coords6, elastic, 1800.0), ("benchmark",)),
        NumbaWarmupKernel("axisymmetric_quad4_elastic_stiffness", lambda: _quad4_axisymmetric_element_stiffness_fast(coords4_axis, elastic)),
        NumbaWarmupKernel("axisymmetric_quad8_elastic_stiffness", lambda: _quad8_axisymmetric_element_stiffness_fast(coords8_axis, elastic, "FULL"), ("benchmark",)),
        NumbaWarmupKernel(
            "axisymmetric_quad4_j2dp_tangent_force",
            lambda: _quad4_axisymmetric_j2dp_tangent_force_fast(coords4_axis, ue4, plastic, initial_stress=initial, plastic_strains=p4, kappas=k4, alpha=alpha, cohesion_term=cohesion_term),
        ),
        NumbaWarmupKernel(
            "axisymmetric_quad8_j2dp_tangent_force",
            lambda: _quad8_axisymmetric_j2dp_tangent_force_fast(coords8_axis, ue8, plastic, "FULL", initial_stress=initial, plastic_strains=p8, kappas=k8, alpha=alpha, cohesion_term=cohesion_term),
            ("benchmark",),
        ),
        NumbaWarmupKernel("axisymmetric_quad4_pressure_matrices", lambda: _quad4_axisymmetric_pressure_matrices_fast(coords4_axis, elastic, storage=1.0e-5, permeability=1.0e-6)),
        NumbaWarmupKernel("axisymmetric_quad8_pressure_matrices", lambda: _quad8_axisymmetric_pressure_matrices_fast(coords8_axis, elastic, storage=1.0e-5, permeability=1.0e-6), ("benchmark",)),
        NumbaWarmupKernel("axisymmetric_tri3_pressure_matrices", lambda: _tri_pressure_matrices_fast(coords3_axis, elastic, storage=1.0e-5, permeability=1.0e-6, axisymmetric=True), ("benchmark",)),
        NumbaWarmupKernel("axisymmetric_tri6_pressure_matrices", lambda: _tri_pressure_matrices_fast(coords6_axis, elastic, storage=1.0e-5, permeability=1.0e-6, axisymmetric=True), ("benchmark",)),
        NumbaWarmupKernel("axisymmetric_quad4_biot_matrix", lambda: _quad4_axisymmetric_biot_matrix_fast(coords4_axis, elastic, 1.0), ("benchmark",)),
        NumbaWarmupKernel("axisymmetric_quad8_biot_matrix", lambda: _quad8_axisymmetric_biot_matrix_fast(coords8_axis, elastic, 1.0), ("benchmark",)),
        NumbaWarmupKernel("axisymmetric_tri3_biot_matrix", lambda: _tri_biot_matrix_fast(coords3_axis, elastic, 1.0, axisymmetric=True), ("benchmark",)),
        NumbaWarmupKernel("axisymmetric_tri6_biot_matrix", lambda: _tri_biot_matrix_fast(coords6_axis, elastic, 1.0, axisymmetric=True), ("benchmark",)),
        NumbaWarmupKernel("large_deformation_updated_coords", lambda: _updated_coords_numba(coords8, displacement, 1.0)),
    )


def _benchmark_displacements(node_count: int) -> np.ndarray:
    values = np.zeros(node_count * 2, dtype=float)
    for node in range(node_count):
        values[2 * node] = 1.0e-4 * (1.0 + 0.1 * node)
        values[2 * node + 1] = -7.0e-5 * (1.0 - 0.05 * node)
    return values


__all__ = [
    "NumbaWarmupKernel",
    "benchmark_numba_warmup_enabled",
    "clear_numba_warmup_state",
    "gui_numba_warmup_enabled",
    "skipped_numba_warmup_summary",
    "warmup_numba_kernels",
]
