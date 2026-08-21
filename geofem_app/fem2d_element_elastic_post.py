"""Elastic integration-point post-stress kernels for QUAD4/QUAD8 elements."""

from __future__ import annotations

from typing import Any
import math

import numpy as np

from .fem2d_element_numba_primitives import (
    _quad4_b_det_numba,
    _quad8_b_det_numba,
    _quad8_gp_full,
    _quad8_shape_grad_numba,
)
from .fem2d_materials import _tension_cutoff_plane_strain_numba
from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, njit


ELEMENT_ELASTIC_POST_FUNCTIONS = (
    "element_elastic_post_contract",
    "_quad4_post_principal_values_numba",
    "_quad4_elastic_post_numba",
    "_quad4_elastic_bbar_post_numba",
    "_quad8_elastic_post_numba",
    "_quad8_elastic_bbar_post_numba",
    "_quad8_elastic_tension_post_numba",
    "_quad8_elastic_tension_bbar_post_numba",
    "_quad4_elastic_post_fast",
    "_quad4_elastic_bbar_post_fast",
    "_quad8_elastic_post_fast",
    "_quad8_elastic_bbar_post_fast",
    "_quad8_elastic_tension_post_fast",
    "_quad8_elastic_tension_bbar_post_fast",
)


def element_elastic_post_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_elastic_post.v1",
        "module": "geofem_app.fem2d_element_elastic_post",
        "function_count": len(ELEMENT_ELASTIC_POST_FUNCTIONS),
        "functions": list(ELEMENT_ELASTIC_POST_FUNCTIONS),
        "covered_surfaces": [
            "quad4_elastic_post",
            "quad4_bbar_elastic_post",
            "quad8_elastic_post",
            "quad8_bbar_elastic_post",
            "quad8_elastic_tension_cutoff_post",
            "principal_stress_recovery",
        ],
    }


@njit(cache=True)
def _quad4_post_principal_values_numba(stress: np.ndarray) -> tuple[float, float, float]:
    sx = stress[0]
    sy = stress[1]
    sz = stress[2]
    txy = stress[3]
    mean = 0.5 * (sx + sy)
    radius = math.sqrt(0.25 * (sx - sy) * (sx - sy) + txy * txy)
    values = np.empty(3, dtype=np.float64)
    values[0] = mean + radius
    values[1] = mean - radius
    values[2] = sz
    for i in range(2):
        best = i
        for j in range(i + 1, 3):
            if values[j] > values[best]:
                best = j
        if best != i:
            tmp = values[i]
            values[i] = values[best]
            values[best] = tmp
    return values[0], values[1], values[2]

@njit(cache=True)
def _quad4_elastic_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 20), dtype=np.float64)
    min_det = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        B, det = _quad4_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return data, min_det
        n0 = 0.25 * (1.0 - xi) * (1.0 - eta)
        n1 = 0.25 * (1.0 + xi) * (1.0 - eta)
        n2 = 0.25 * (1.0 + xi) * (1.0 + eta)
        n3 = 0.25 * (1.0 - xi) * (1.0 + eta)
        x = n0 * coords[0, 0] + n1 * coords[1, 0] + n2 * coords[2, 0] + n3 * coords[3, 0]
        y = n0 * coords[0, 1] + n1 * coords[1, 1] + n2 * coords[2, 1] + n3 * coords[3, 1]
        strain = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = 0.0
            for j in range(8):
                value += B[i, j] * ue[j]
            strain[i] = value
        stress = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = initial_stress[i]
            for j in range(4):
                value += D4[i, j] * strain[j]
            stress[i] = value
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
        mean_stress = (stress[0] + stress[1] + stress[2]) / 3.0
        dev0 = stress[0] - mean_stress
        dev1 = stress[1] - mean_stress
        dev2 = stress[2] - mean_stress
        q = math.sqrt(max(1.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + 3.0 * stress[3] * stress[3], 0.0))
        data[gp, 0] = xi
        data[gp, 1] = eta
        data[gp, 2] = 1.0
        data[gp, 3] = x
        data[gp, 4] = y
        data[gp, 5] = det * thickness
        for i in range(4):
            data[gp, 6 + i] = strain[i]
            data[gp, 10 + i] = stress[i]
        data[gp, 14] = s1
        data[gp, 15] = s2
        data[gp, 16] = s3
        data[gp, 17] = 0.5 * (s1 - s3)
        data[gp, 18] = -mean_stress
        data[gp, 19] = q
    return data, min_det

@njit(cache=True)
def _quad4_elastic_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    Pvol: np.ndarray,
    initial_stress: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 20), dtype=np.float64)
    strains = np.zeros((4, 4), dtype=np.float64)
    xs = np.zeros(4, dtype=np.float64)
    ys = np.zeros(4, dtype=np.float64)
    dvols = np.zeros(4, dtype=np.float64)
    epsv_acc = np.zeros(4, dtype=np.float64)
    min_det = 1.0e300
    volume = 0.0
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        B, det = _quad4_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return data, min_det
        n0 = 0.25 * (1.0 - xi) * (1.0 - eta)
        n1 = 0.25 * (1.0 + xi) * (1.0 - eta)
        n2 = 0.25 * (1.0 + xi) * (1.0 + eta)
        n3 = 0.25 * (1.0 - xi) * (1.0 + eta)
        xs[gp] = n0 * coords[0, 0] + n1 * coords[1, 0] + n2 * coords[2, 0] + n3 * coords[3, 0]
        ys[gp] = n0 * coords[0, 1] + n1 * coords[1, 1] + n2 * coords[2, 1] + n3 * coords[3, 1]
        dV = det * thickness
        dvols[gp] = dV
        volume += dV
        for i in range(4):
            value = 0.0
            for j in range(8):
                value += B[i, j] * ue[j]
            strains[gp, i] = value
        for i in range(4):
            value = 0.0
            for j in range(4):
                value += Pvol[i, j] * strains[gp, j]
            epsv_acc[i] += value * dV
    if volume <= np.finfo(np.float64).eps:
        return data, min_det

    epsv_bar = np.empty(4, dtype=np.float64)
    for i in range(4):
        epsv_bar[i] = epsv_acc[i] / volume
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        strain_eff = np.empty(4, dtype=np.float64)
        for i in range(4):
            volumetric = 0.0
            for j in range(4):
                volumetric += Pvol[i, j] * strains[gp, j]
            strain_eff[i] = strains[gp, i] - volumetric + epsv_bar[i]
        stress = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = initial_stress[i]
            for j in range(4):
                value += D4[i, j] * strain_eff[j]
            stress[i] = value
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
        mean_stress = (stress[0] + stress[1] + stress[2]) / 3.0
        dev0 = stress[0] - mean_stress
        dev1 = stress[1] - mean_stress
        dev2 = stress[2] - mean_stress
        q = math.sqrt(max(1.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + 3.0 * stress[3] * stress[3], 0.0))
        data[gp, 0] = xi
        data[gp, 1] = eta
        data[gp, 2] = 1.0
        data[gp, 3] = xs[gp]
        data[gp, 4] = ys[gp]
        data[gp, 5] = dvols[gp]
        for i in range(4):
            data[gp, 6 + i] = strain_eff[i]
            data[gp, 10 + i] = stress[i]
        data[gp, 14] = s1
        data[gp, 15] = s2
        data[gp, 16] = s3
        data[gp, 17] = 0.5 * (s1 - s3)
        data[gp, 18] = -mean_stress
        data[gp, 19] = q
    return data, min_det

@njit(cache=True)
def _quad8_elastic_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((9, 20), dtype=np.float64)
    min_det = 1.0e300
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        N, _grad, det_shape = _quad8_shape_grad_numba(coords, xi, eta)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det_shape < min_det:
            min_det = det_shape
        if det <= 0.0 or det_shape <= 0.0:
            return data, min_det
        x = 0.0
        y = 0.0
        for i in range(8):
            x += N[i] * coords[i, 0]
            y += N[i] * coords[i, 1]
        strain = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = 0.0
            for j in range(16):
                value += B[i, j] * ue[j]
            strain[i] = value
        stress = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = initial_stress[i]
            for j in range(4):
                value += D4[i, j] * strain[j]
            stress[i] = value
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
        mean_stress = (stress[0] + stress[1] + stress[2]) / 3.0
        dev0 = stress[0] - mean_stress
        dev1 = stress[1] - mean_stress
        dev2 = stress[2] - mean_stress
        q = math.sqrt(max(1.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + 3.0 * stress[3] * stress[3], 0.0))
        data[gp, 0] = xi
        data[gp, 1] = eta
        data[gp, 2] = weight
        data[gp, 3] = x
        data[gp, 4] = y
        data[gp, 5] = det * weight * thickness
        for i in range(4):
            data[gp, 6 + i] = strain[i]
            data[gp, 10 + i] = stress[i]
        data[gp, 14] = s1
        data[gp, 15] = s2
        data[gp, 16] = s3
        data[gp, 17] = 0.5 * (s1 - s3)
        data[gp, 18] = -mean_stress
        data[gp, 19] = q
    return data, min_det

@njit(cache=True)
def _quad8_elastic_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    Pvol: np.ndarray,
    initial_stress: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, float, float]:
    data = np.zeros((9, 20), dtype=np.float64)
    strains = np.zeros((9, 4), dtype=np.float64)
    xs = np.zeros(9, dtype=np.float64)
    ys = np.zeros(9, dtype=np.float64)
    weights = np.zeros(9, dtype=np.float64)
    dvols = np.zeros(9, dtype=np.float64)
    epsv_acc = np.zeros(4, dtype=np.float64)
    min_det = 1.0e300
    volume = 0.0
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        N, _grad, det_shape = _quad8_shape_grad_numba(coords, xi, eta)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det_shape < min_det:
            min_det = det_shape
        if det <= 0.0 or det_shape <= 0.0:
            return data, min_det, volume
        x = 0.0
        y = 0.0
        for i in range(8):
            x += N[i] * coords[i, 0]
            y += N[i] * coords[i, 1]
        xs[gp] = x
        ys[gp] = y
        weights[gp] = weight
        dV = det * weight * thickness
        dvols[gp] = dV
        volume += dV
        for i in range(4):
            value = 0.0
            for j in range(16):
                value += B[i, j] * ue[j]
            strains[gp, i] = value
        for i in range(4):
            value = 0.0
            for j in range(4):
                value += Pvol[i, j] * strains[gp, j]
            epsv_acc[i] += value * dV
    if volume <= 0.0:
        return data, min_det, volume

    epsv_bar = np.empty(4, dtype=np.float64)
    for i in range(4):
        epsv_bar[i] = epsv_acc[i] / volume
    for gp in range(9):
        xi, eta, _weight = _quad8_gp_full(gp)
        strain_eff = np.empty(4, dtype=np.float64)
        for i in range(4):
            volumetric = 0.0
            for j in range(4):
                volumetric += Pvol[i, j] * strains[gp, j]
            strain_eff[i] = strains[gp, i] - volumetric + epsv_bar[i]
        stress = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = initial_stress[i]
            for j in range(4):
                value += D4[i, j] * strain_eff[j]
            stress[i] = value
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
        mean_stress = (stress[0] + stress[1] + stress[2]) / 3.0
        dev0 = stress[0] - mean_stress
        dev1 = stress[1] - mean_stress
        dev2 = stress[2] - mean_stress
        q = math.sqrt(max(1.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + 3.0 * stress[3] * stress[3], 0.0))
        data[gp, 0] = xi
        data[gp, 1] = eta
        data[gp, 2] = weights[gp]
        data[gp, 3] = xs[gp]
        data[gp, 4] = ys[gp]
        data[gp, 5] = dvols[gp]
        for i in range(4):
            data[gp, 6 + i] = strain_eff[i]
            data[gp, 10 + i] = stress[i]
        data[gp, 14] = s1
        data[gp, 15] = s2
        data[gp, 16] = s3
        data[gp, 17] = 0.5 * (s1 - s3)
        data[gp, 18] = -mean_stress
        data[gp, 19] = q
    return data, min_det, volume

@njit(cache=True)
def _quad8_elastic_tension_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    tensile_strength: float,
    thickness: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((9, 22), dtype=np.float64)
    min_det = 1.0e300
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        N, _grad, det_shape = _quad8_shape_grad_numba(coords, xi, eta)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det_shape < min_det:
            min_det = det_shape
        if det <= 0.0 or det_shape <= 0.0:
            return data, min_det
        x = 0.0
        y = 0.0
        for i in range(8):
            x += N[i] * coords[i, 0]
            y += N[i] * coords[i, 1]
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
        stress, clipped, excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
        mean_stress = (stress[0] + stress[1] + stress[2]) / 3.0
        dev0 = stress[0] - mean_stress
        dev1 = stress[1] - mean_stress
        dev2 = stress[2] - mean_stress
        q = math.sqrt(max(1.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + 3.0 * stress[3] * stress[3], 0.0))
        data[gp, 0] = xi
        data[gp, 1] = eta
        data[gp, 2] = weight
        data[gp, 3] = x
        data[gp, 4] = y
        data[gp, 5] = det * weight * thickness
        for i in range(4):
            data[gp, 6 + i] = strain[i]
            data[gp, 10 + i] = stress[i]
        data[gp, 14] = s1
        data[gp, 15] = s2
        data[gp, 16] = s3
        data[gp, 17] = 0.5 * (s1 - s3)
        data[gp, 18] = -mean_stress
        data[gp, 19] = q
        data[gp, 20] = 1.0 if clipped else 0.0
        data[gp, 21] = max(excess, 0.0)
    return data, min_det

@njit(cache=True)
def _quad8_elastic_tension_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    Pvol: np.ndarray,
    initial_stress: np.ndarray,
    tensile_strength: float,
    thickness: float,
) -> tuple[np.ndarray, float, float]:
    data = np.zeros((9, 22), dtype=np.float64)
    strains = np.zeros((9, 4), dtype=np.float64)
    xs = np.zeros(9, dtype=np.float64)
    ys = np.zeros(9, dtype=np.float64)
    weights = np.zeros(9, dtype=np.float64)
    dvols = np.zeros(9, dtype=np.float64)
    epsv_acc = np.zeros(4, dtype=np.float64)
    min_det = 1.0e300
    volume = 0.0
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        N, _grad, det_shape = _quad8_shape_grad_numba(coords, xi, eta)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det_shape < min_det:
            min_det = det_shape
        if det <= 0.0 or det_shape <= 0.0:
            return data, min_det, volume
        x = 0.0
        y = 0.0
        for i in range(8):
            x += N[i] * coords[i, 0]
            y += N[i] * coords[i, 1]
        xs[gp] = x
        ys[gp] = y
        weights[gp] = weight
        dV = det * weight * thickness
        dvols[gp] = dV
        volume += dV
        for i in range(4):
            value = 0.0
            for j in range(16):
                value += B[i, j] * ue[j]
            strains[gp, i] = value
        for i in range(4):
            value = 0.0
            for j in range(4):
                value += Pvol[i, j] * strains[gp, j]
            epsv_acc[i] += value * dV
    if volume <= 0.0:
        return data, min_det, volume

    epsv_bar = np.empty(4, dtype=np.float64)
    for i in range(4):
        epsv_bar[i] = epsv_acc[i] / volume
    for gp in range(9):
        xi, eta, _weight = _quad8_gp_full(gp)
        strain_eff = np.empty(4, dtype=np.float64)
        for i in range(4):
            volumetric = 0.0
            for j in range(4):
                volumetric += Pvol[i, j] * strains[gp, j]
            strain_eff[i] = strains[gp, i] - volumetric + epsv_bar[i]
        trial = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = initial_stress[i]
            for j in range(4):
                value += D4[i, j] * strain_eff[j]
            trial[i] = value
        stress, clipped, excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
        mean_stress = (stress[0] + stress[1] + stress[2]) / 3.0
        dev0 = stress[0] - mean_stress
        dev1 = stress[1] - mean_stress
        dev2 = stress[2] - mean_stress
        q = math.sqrt(max(1.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + 3.0 * stress[3] * stress[3], 0.0))
        data[gp, 0] = xi
        data[gp, 1] = eta
        data[gp, 2] = weights[gp]
        data[gp, 3] = xs[gp]
        data[gp, 4] = ys[gp]
        data[gp, 5] = dvols[gp]
        for i in range(4):
            data[gp, 6 + i] = strain_eff[i]
            data[gp, 10 + i] = stress[i]
        data[gp, 14] = s1
        data[gp, 15] = s2
        data[gp, 16] = s3
        data[gp, 17] = 0.5 * (s1 - s3)
        data[gp, 18] = -mean_stress
        data[gp, 19] = q
        data[gp, 20] = 1.0 if clipped else 0.0
        data[gp, 21] = max(excess, 0.0)
    return data, min_det, volume

def _quad4_elastic_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    data, min_det = _quad4_elastic_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data

def _quad4_elastic_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    data, min_det = _quad4_elastic_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data

def _quad8_elastic_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    data, min_det = _quad8_elastic_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return data

def _quad8_elastic_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    data, min_det, volume = _quad8_elastic_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return data

def _quad8_elastic_tension_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    if not math.isfinite(float(material.tensile_strength)):
        raise FEM2DError("QUAD8 tension cutoff requires finite tensile strength")
    data, min_det = _quad8_elastic_tension_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        float(material.tensile_strength),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return data

def _quad8_elastic_tension_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    if not math.isfinite(float(material.tensile_strength)):
        raise FEM2DError("QUAD8 tension cutoff requires finite tensile strength")
    data, min_det, volume = _quad8_elastic_tension_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        float(material.tensile_strength),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return data


__all__ = list(ELEMENT_ELASTIC_POST_FUNCTIONS)
