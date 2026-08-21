"""Tension-cutoff nonlinear element kernels for QUAD8 plane-strain analyses."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fem2d_element_numba_primitives import (
    _quad8_add_btcb_numba,
    _quad8_add_btlcbr_numba,
    _quad8_add_btstress_numba,
    _quad8_b_det_numba,
    _quad8_gp_full,
    _quad8_gp_reduced,
    _quad8_project_b_numba,
)
from .fem2d_materials import (
    _elastic_tension_cutoff_numerical_tangent_numba,
    _tension_cutoff_plane_strain_numba,
)
from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, njit

_QUAD4_MODE_FULL = 0
_QUAD4_MODE_SRI = 1
_QUAD4_MODE_BBAR = 2

TENSION_CUTOFF_ELEMENT_KERNEL_FUNCTIONS = (
    "tension_cutoff_element_kernel_contract",
    "_quad8_elastic_tension_tangent_force_numba",
    "_quad8_elastic_tension_tangent_force_fast",
)


def _quad4_mode_code(mode: str) -> int:
    if mode == "FULL":
        return _QUAD4_MODE_FULL
    if mode == "SRI":
        return _QUAD4_MODE_SRI
    if mode == "B-BAR":
        return _QUAD4_MODE_BBAR
    raise FEM2DError(f"unsupported integration '{mode}'")


def tension_cutoff_element_kernel_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_tension_cutoff_kernels.v1",
        "module": "geofem_app.fem2d_element_tension_cutoff_kernels",
        "function_count": len(TENSION_CUTOFF_ELEMENT_KERNEL_FUNCTIONS),
        "functions": list(TENSION_CUTOFF_ELEMENT_KERNEL_FUNCTIONS),
        "covered_surfaces": [
            "quad8_elastic_tension_cutoff_tangent",
            "quad8_elastic_tension_cutoff_internal_force",
            "full_sri_bbar_dispatch",
        ],
    }


@njit(cache=True)
def _quad8_elastic_tension_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    tensile_strength: float,
    thickness: float,
    mode_code: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    ke = np.zeros((16, 16), dtype=np.float64)
    fe = np.zeros(16, dtype=np.float64)
    min_det = 1.0e300
    volume = 0.0

    if mode_code == _QUAD4_MODE_FULL:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, fe, min_det, volume
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            trial = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = initial_stress[i]
                for j in range(4):
                    value += D4[i, j] * strain[j]
                trial[i] = value
            stress, _clipped, _excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
            tangent = _elastic_tension_cutoff_numerical_tangent_numba(strain, D4, initial_stress, tensile_strength)
            dV = det * weight * thickness
            _quad8_add_btcb_numba(ke, B, tangent, dV)
            _quad8_add_btstress_numba(fe, B, stress, dV)
        return ke, fe, min_det, volume

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, fe, min_det, volume
            Bdev = _quad8_project_b_numba(Idev, B)
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            trial = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = initial_stress[i]
                for j in range(4):
                    value += D4[i, j] * strain[j]
                trial[i] = value
            stress, _clipped, _excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
            tangent = _elastic_tension_cutoff_numerical_tangent_numba(strain, D4, initial_stress, tensile_strength)
            dV = det * weight * thickness
            _quad8_add_btlcbr_numba(ke, Bdev, tangent, B, dV)
            _quad8_add_btstress_numba(fe, Bdev, stress, dV)
        for gp in range(4):
            xi, eta, weight = _quad8_gp_reduced(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, fe, min_det, volume
            Bv = _quad8_project_b_numba(Pvol, B)
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            trial = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = initial_stress[i]
                for j in range(4):
                    value += D4[i, j] * strain[j]
                trial[i] = value
            stress, _clipped, _excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
            tangent = _elastic_tension_cutoff_numerical_tangent_numba(strain, D4, initial_stress, tensile_strength)
            dV = det * weight * thickness
            _quad8_add_btlcbr_numba(ke, Bv, tangent, B, dV)
            _quad8_add_btstress_numba(fe, Bv, stress, dV)
        return ke, fe, min_det, volume

    B_cache = np.zeros((9, 4, 16), dtype=np.float64)
    dV_cache = np.zeros(9, dtype=np.float64)
    Bv_acc = np.zeros((4, 16), dtype=np.float64)
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return ke, fe, min_det, volume
        dV = det * weight * thickness
        dV_cache[gp] = dV
        volume += dV
        Bv = _quad8_project_b_numba(Pvol, B)
        for r in range(4):
            for c in range(16):
                B_cache[gp, r, c] = B[r, c]
                Bv_acc[r, c] += Bv[r, c] * dV
    if volume <= 0.0:
        return ke, fe, min_det, volume
    Bv_bar = Bv_acc / volume
    for gp in range(9):
        B = B_cache[gp]
        Bdev = _quad8_project_b_numba(Idev, B)
        B_eff = Bdev + Bv_bar
        strain = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = 0.0
            for j in range(16):
                value += B_eff[i, j] * ue[j]
            strain[i] = value
        trial = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = initial_stress[i]
            for j in range(4):
                value += D4[i, j] * strain[j]
            trial[i] = value
        stress, _clipped, _excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
        tangent = _elastic_tension_cutoff_numerical_tangent_numba(strain, D4, initial_stress, tensile_strength)
        dV = dV_cache[gp]
        _quad8_add_btcb_numba(ke, B_eff, tangent, dV)
        _quad8_add_btstress_numba(fe, B_eff, stress, dV)
    return ke, fe, min_det, volume


def _quad8_elastic_tension_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    mode: str,
    initial_stress: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    if not math.isfinite(float(material.tensile_strength)):
        raise FEM2DError("QUAD8 tension cutoff requires finite tensile strength")
    mode_code = _quad4_mode_code(mode)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    ke, fe, min_det, volume = _quad8_elastic_tension_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        Pvol,
        Idev,
        float(material.tensile_strength),
        float(material.thickness),
        mode_code,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return ke, fe


__all__ = list(TENSION_CUTOFF_ELEMENT_KERNEL_FUNCTIONS)
