"""Linear elastic element kernels for QUAD4/QUAD8 plane-strain and axisymmetric analyses."""

from __future__ import annotations

from typing import Any
import math

import numpy as np

from .fem2d_element_numba_primitives import (
    _quad4_add_btcb_numba,
    _quad4_b_det_numba,
    _quad4_project_b_numba,
    _quad4_shape_grad_numba,
    _quad4_symmetrize_numba,
    _quad8_add_btcb_numba,
    _quad8_b_det_numba,
    _quad8_gp_full,
    _quad8_gp_reduced,
    _quad8_project_b_numba,
    _quad8_shape_grad_numba,
    _quad8_symmetrize_numba,
)
from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, njit, normalize_integration

_QUAD4_MODE_FULL = 0
_QUAD4_MODE_SRI = 1
_QUAD4_MODE_BBAR = 2


ELASTIC_ELEMENT_KERNEL_FUNCTIONS = (
    "elastic_element_kernel_contract",
    "_quad4_element_stiffness_numba",
    "_quad4_pressure_matrices_numba",
    "_quad4_biot_matrix_numba",
    "_quad4_consistent_mass_matrix_numba",
    "_quad4_axisymmetric_pressure_matrices_numba",
    "_quad4_axisymmetric_biot_matrix_numba",
    "_quad4_axisymmetric_b_matrix_numba",
    "_quad4_axisymmetric_element_stiffness_numba",
    "_quad4_axisymmetric_internal_force_elastic_numba",
    "_quad4_internal_force_elastic_numba",
    "_quad8_element_stiffness_numba",
    "_quad8_consistent_mass_matrix_numba",
    "_quad8_pressure_matrices_numba",
    "_quad8_biot_matrix_numba",
    "_quad8_internal_force_elastic_numba",
    "_quad8_axisymmetric_b_matrix_numba",
    "_quad8_axisymmetric_element_stiffness_numba",
    "_quad8_axisymmetric_internal_force_elastic_numba",
    "_quad8_axisymmetric_pressure_matrices_numba",
    "_quad8_axisymmetric_biot_matrix_numba",
    "_quad8_axisymmetric_edge_traction_numba",
    "_quad4_element_stiffness_fast",
    "_quad4_pressure_matrices_fast",
    "_quad4_biot_matrix_fast",
    "_quad4_consistent_mass_matrix_fast",
    "_quad8_element_stiffness_fast",
    "_quad8_pressure_matrices_fast",
    "_quad8_biot_matrix_fast",
    "_quad8_consistent_mass_matrix_fast",
    "_quad8_internal_force_elastic_fast",
    "_quad4_axisymmetric_pressure_matrices_fast",
    "_quad4_axisymmetric_biot_matrix_fast",
    "_quad4_axisymmetric_element_stiffness_fast",
    "_quad8_axisymmetric_element_stiffness_fast",
    "_quad8_axisymmetric_pressure_matrices_fast",
    "_quad8_axisymmetric_biot_matrix_fast",
    "_quad4_axisymmetric_internal_force_elastic_fast",
    "_quad8_axisymmetric_internal_force_elastic_fast",
    "_quad8_axisymmetric_edge_traction_fast",
    "_quad4_internal_force_elastic_fast",
)


def _quad4_mode_code(mode: str) -> int:
    if mode == "FULL":
        return _QUAD4_MODE_FULL
    if mode == "SRI":
        return _QUAD4_MODE_SRI
    if mode == "B-BAR":
        return _QUAD4_MODE_BBAR
    raise FEM2DError(f"unsupported integration '{mode}'")


def elastic_element_kernel_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_elastic_kernels.v1",
        "module": "geofem_app.fem2d_element_elastic_kernels",
        "function_count": len(ELASTIC_ELEMENT_KERNEL_FUNCTIONS),
        "functions": list(ELASTIC_ELEMENT_KERNEL_FUNCTIONS),
        "covered_surfaces": [
            "quad4_plane_strain_stiffness_internal_force",
            "quad8_plane_strain_stiffness_internal_force",
            "quad4_pressure_biot_mass",
            "quad8_pressure_biot_mass",
            "quad4_axisymmetric_stiffness_internal_force",
            "quad8_axisymmetric_stiffness_internal_force",
            "axisymmetric_edge_traction",
        ],
    }


@njit(cache=True)
def _quad4_element_stiffness_numba(
    coords: np.ndarray,
    D4: np.ndarray,
    Cdev: np.ndarray,
    Cvol: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    thickness: float,
    mode_code: int,
) -> tuple[np.ndarray, float, float]:
    ke = np.zeros((8, 8), dtype=np.float64)
    min_det = 1.0e300
    volume = 0.0
    a = 1.0 / math.sqrt(3.0)

    if mode_code == _QUAD4_MODE_FULL:
        for gp in range(4):
            xi = -a if gp == 0 or gp == 3 else a
            eta = -a if gp == 0 or gp == 1 else a
            B, det = _quad4_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, min_det, volume
            _quad4_add_btcb_numba(ke, B, D4, det * thickness)
        _quad4_symmetrize_numba(ke)
        return ke, min_det, volume

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(4):
            xi = -a if gp == 0 or gp == 3 else a
            eta = -a if gp == 0 or gp == 1 else a
            B, det = _quad4_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, min_det, volume
            _quad4_add_btcb_numba(ke, B, Cdev, det * thickness)
        Bc, detc = _quad4_b_det_numba(coords, 0.0, 0.0)
        if detc < min_det:
            min_det = detc
        if detc <= 0.0:
            return ke, min_det, volume
        Bv = _quad4_project_b_numba(Pvol, Bc)
        _quad4_add_btcb_numba(ke, Bv, Cvol, detc * 4.0 * thickness)
        _quad4_symmetrize_numba(ke)
        return ke, min_det, volume

    B_cache = np.zeros((4, 4, 8), dtype=np.float64)
    dV_cache = np.zeros(4, dtype=np.float64)
    Bv_acc = np.zeros((4, 8), dtype=np.float64)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        B, det = _quad4_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return ke, min_det, volume
        dV = det * thickness
        dV_cache[gp] = dV
        volume += dV
        Bv = _quad4_project_b_numba(Pvol, B)
        for r in range(4):
            for c in range(8):
                B_cache[gp, r, c] = B[r, c]
                Bv_acc[r, c] += Bv[r, c] * dV
    if volume <= 0.0:
        return ke, min_det, volume
    Bv_bar = Bv_acc / volume
    for gp in range(4):
        B = B_cache[gp]
        Bdev = _quad4_project_b_numba(Idev, B)
        dV = dV_cache[gp]
        _quad4_add_btcb_numba(ke, Bdev, Cdev, dV)
        _quad4_add_btcb_numba(ke, Bv_bar, Cvol, dV)
    _quad4_symmetrize_numba(ke)
    return ke, min_det, volume

@njit(cache=True)
def _quad4_pressure_matrices_numba(
    coords: np.ndarray,
    storage: float,
    permeability: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    me = np.zeros((4, 4), dtype=np.float64)
    ke = np.zeros((4, 4), dtype=np.float64)
    min_det = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        N, grad, det = _quad4_shape_grad_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return me, ke, min_det
        dV = det * thickness
        for i in range(4):
            for j in range(4):
                me[i, j] += storage * N[i] * N[j] * dV
                ke[i, j] += permeability * (grad[0, i] * grad[0, j] + grad[1, i] * grad[1, j]) * dV
    return me, ke, min_det

@njit(cache=True)
def _quad4_biot_matrix_numba(coords: np.ndarray, alpha: float, thickness: float) -> tuple[np.ndarray, float]:
    block = np.zeros((8, 4), dtype=np.float64)
    min_det = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        B, det = _quad4_b_det_numba(coords, xi, eta)
        N, _grad, det_shape = _quad4_shape_grad_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det_shape < min_det:
            min_det = det_shape
        if det <= 0.0 or det_shape <= 0.0:
            return block, min_det
        dV = det * thickness
        for i in range(8):
            volumetric_row = B[0, i] + B[1, i] + B[2, i]
            for j in range(4):
                block[i, j] += alpha * volumetric_row * N[j] * dV
    return block, min_det


def _quad4_biot_matrix_python(coords: np.ndarray, alpha: float, thickness: float) -> tuple[np.ndarray, float]:
    points = np.asarray(coords, dtype=float)
    block = np.zeros((8, 4), dtype=float)
    min_det = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for xi, eta in ((-a, -a), (a, -a), (a, a), (-a, a)):
        N = np.array(
            [
                0.25 * (1.0 - xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 - eta),
                0.25 * (1.0 + xi) * (1.0 + eta),
                0.25 * (1.0 - xi) * (1.0 + eta),
            ],
            dtype=float,
        )
        dxi = np.array(
            [
                -0.25 * (1.0 - eta),
                0.25 * (1.0 - eta),
                0.25 * (1.0 + eta),
                -0.25 * (1.0 + eta),
            ],
            dtype=float,
        )
        deta = np.array(
            [
                -0.25 * (1.0 - xi),
                -0.25 * (1.0 + xi),
                0.25 * (1.0 + xi),
                0.25 * (1.0 - xi),
            ],
            dtype=float,
        )
        j00 = float(dxi @ points[:, 0])
        j01 = float(dxi @ points[:, 1])
        j10 = float(deta @ points[:, 0])
        j11 = float(deta @ points[:, 1])
        det = j00 * j11 - j01 * j10
        min_det = min(min_det, det)
        if det <= 0.0:
            return block, min_det
        inv00 = j11 / det
        inv01 = -j01 / det
        inv10 = -j10 / det
        inv11 = j00 / det
        B = np.zeros((4, 8), dtype=float)
        for i in range(4):
            dndx = inv00 * dxi[i] + inv01 * deta[i]
            dndy = inv10 * dxi[i] + inv11 * deta[i]
            c = 2 * i
            B[0, c] = dndx
            B[1, c + 1] = dndy
            B[3, c] = dndy
            B[3, c + 1] = dndx
        dV = det * thickness
        for i in range(8):
            volumetric_row = B[0, i] + B[1, i] + B[2, i]
            for j in range(4):
                block[i, j] += alpha * volumetric_row * N[j] * dV
    return block, min_det


@njit(cache=True)
def _quad4_consistent_mass_matrix_numba(coords: np.ndarray, density: float, thickness: float) -> tuple[np.ndarray, float]:
    me = np.zeros((8, 8), dtype=np.float64)
    min_det = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        N, _grad, det = _quad4_shape_grad_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return me, min_det
        dV = det * thickness
        for i in range(4):
            row_u = 2 * i
            row_v = row_u + 1
            Ni = N[i]
            for j in range(4):
                value = density * Ni * N[j] * dV
                col_u = 2 * j
                me[row_u, col_u] += value
                me[row_v, col_u + 1] += value
    return me, min_det

@njit(cache=True)
def _quad4_axisymmetric_pressure_matrices_numba(
    coords: np.ndarray,
    storage: float,
    permeability: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    me = np.zeros((4, 4), dtype=np.float64)
    ke = np.zeros((4, 4), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        N, grad, det = _quad4_shape_grad_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return me, ke, min_det, min_radius
        radius = 0.0
        for i in range(4):
            radius += N[i] * coords[i, 0]
        if radius < min_radius:
            min_radius = radius
        if radius <= 0.0:
            return me, ke, min_det, min_radius
        dV = det * thickness * 2.0 * math.pi * radius
        for i in range(4):
            for j in range(4):
                me[i, j] += storage * N[i] * N[j] * dV
                ke[i, j] += permeability * (grad[0, i] * grad[0, j] + grad[1, i] * grad[1, j]) * dV
    return me, ke, min_det, min_radius

@njit(cache=True)
def _quad4_axisymmetric_biot_matrix_numba(coords: np.ndarray, alpha: float, thickness: float) -> tuple[np.ndarray, float, float]:
    block = np.zeros((8, 4), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        N, grad, det = _quad4_shape_grad_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return block, min_det, min_radius
        radius = 0.0
        for i in range(4):
            radius += N[i] * coords[i, 0]
        if radius < min_radius:
            min_radius = radius
        if radius <= 0.0:
            return block, min_det, min_radius
        dV = det * thickness * 2.0 * math.pi * radius
        for i in range(4):
            ur = 2 * i
            uz = ur + 1
            volumetric_ur = grad[0, i] + N[i] / radius
            volumetric_uz = grad[1, i]
            for j in range(4):
                weighted_n = alpha * N[j] * dV
                block[ur, j] += volumetric_ur * weighted_n
                block[uz, j] += volumetric_uz * weighted_n
    return block, min_det, min_radius

@njit(cache=True)
def _quad4_axisymmetric_b_matrix_numba(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    B = np.zeros((4, 8), dtype=np.float64)
    N, grad, det = _quad4_shape_grad_numba(coords, xi, eta)
    radius = 0.0
    if det <= 0.0:
        return B, N, det, radius
    for i in range(4):
        radius += N[i] * coords[i, 0]
    if radius <= 0.0:
        return B, N, det, radius
    for i in range(4):
        c = 2 * i
        dndr = grad[0, i]
        dndz = grad[1, i]
        B[0, c] = dndr
        B[1, c + 1] = dndz
        B[2, c] = N[i] / radius
        B[3, c] = dndz
        B[3, c + 1] = dndr
    return B, N, det, radius

@njit(cache=True)
def _quad4_axisymmetric_element_stiffness_numba(coords: np.ndarray, D4: np.ndarray, thickness: float) -> tuple[np.ndarray, float, float]:
    ke = np.zeros((8, 8), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        B, _N, det, radius = _quad4_axisymmetric_b_matrix_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if radius < min_radius:
            min_radius = radius
        if det <= 0.0 or radius <= 0.0:
            return ke, min_det, min_radius
        _quad4_add_btcb_numba(ke, B, D4, det * thickness * 2.0 * math.pi * radius)
    _quad4_symmetrize_numba(ke)
    return ke, min_det, min_radius

@njit(cache=True)
def _quad4_axisymmetric_internal_force_elastic_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, float, float]:
    fe = np.zeros(8, dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        B, _N, det, radius = _quad4_axisymmetric_b_matrix_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if radius < min_radius:
            min_radius = radius
        if det <= 0.0 or radius <= 0.0:
            return fe, min_det, min_radius
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
        dV = det * thickness * 2.0 * math.pi * radius
        for i in range(8):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    return fe, min_det, min_radius

@njit(cache=True)
def _quad4_internal_force_elastic_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    thickness: float,
) -> tuple[np.ndarray, float]:
    fe = np.zeros(8, dtype=np.float64)
    min_det = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        B, det = _quad4_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return fe, min_det
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
        dV = det * thickness
        for i in range(8):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    return fe, min_det

@njit(cache=True)
def _quad8_element_stiffness_numba(
    coords: np.ndarray,
    D4: np.ndarray,
    Cdev: np.ndarray,
    Cvol: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    thickness: float,
    mode_code: int,
) -> tuple[np.ndarray, float, float]:
    ke = np.zeros((16, 16), dtype=np.float64)
    min_det = 1.0e300
    volume = 0.0

    if mode_code == _QUAD4_MODE_FULL:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, min_det, volume
            _quad8_add_btcb_numba(ke, B, D4, det * weight * thickness)
        _quad8_symmetrize_numba(ke)
        return ke, min_det, volume

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, min_det, volume
            _quad8_add_btcb_numba(ke, B, Cdev, det * weight * thickness)
        for gp in range(4):
            xi, eta, weight = _quad8_gp_reduced(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, min_det, volume
            Bv = _quad8_project_b_numba(Pvol, B)
            _quad8_add_btcb_numba(ke, Bv, Cvol, det * weight * thickness)
        _quad8_symmetrize_numba(ke)
        return ke, min_det, volume

    B_cache = np.zeros((9, 4, 16), dtype=np.float64)
    dV_cache = np.zeros(9, dtype=np.float64)
    Bv_acc = np.zeros((4, 16), dtype=np.float64)
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return ke, min_det, volume
        dV = det * weight * thickness
        dV_cache[gp] = dV
        volume += dV
        Bv = _quad8_project_b_numba(Pvol, B)
        for r in range(4):
            for c in range(16):
                B_cache[gp, r, c] = B[r, c]
                Bv_acc[r, c] += Bv[r, c] * dV
    if volume <= 0.0:
        return ke, min_det, volume
    Bv_bar = Bv_acc / volume
    for gp in range(9):
        B = B_cache[gp]
        Bdev = _quad8_project_b_numba(Idev, B)
        dV = dV_cache[gp]
        _quad8_add_btcb_numba(ke, Bdev, Cdev, dV)
        _quad8_add_btcb_numba(ke, Bv_bar, Cvol, dV)
    _quad8_symmetrize_numba(ke)
    return ke, min_det, volume

@njit(cache=True)
def _quad8_consistent_mass_matrix_numba(coords: np.ndarray, density: float, thickness: float) -> tuple[np.ndarray, float]:
    me = np.zeros((16, 16), dtype=np.float64)
    min_det = 1.0e300
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        N, _grad, det = _quad8_shape_grad_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return me, min_det
        dV = det * weight * thickness
        for i in range(8):
            row_u = 2 * i
            row_v = row_u + 1
            Ni = N[i]
            for j in range(8):
                value = density * Ni * N[j] * dV
                col_u = 2 * j
                me[row_u, col_u] += value
                me[row_v, col_u + 1] += value
    return me, min_det

@njit(cache=True)
def _quad8_pressure_matrices_numba(
    coords: np.ndarray,
    storage: float,
    permeability: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    me = np.zeros((8, 8), dtype=np.float64)
    ke = np.zeros((8, 8), dtype=np.float64)
    min_det = 1.0e300
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        N, grad, det = _quad8_shape_grad_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return me, ke, min_det
        dV = det * weight * thickness
        for i in range(8):
            for j in range(8):
                me[i, j] += storage * N[i] * N[j] * dV
                ke[i, j] += permeability * (grad[0, i] * grad[0, j] + grad[1, i] * grad[1, j]) * dV
    return me, ke, min_det

@njit(cache=True)
def _quad8_biot_matrix_numba(
    coords: np.ndarray,
    alpha: float,
    thickness: float,
    Pvol: np.ndarray,
    mode_code: int,
) -> tuple[np.ndarray, float, float]:
    block = np.zeros((16, 8), dtype=np.float64)
    min_det = 1.0e300
    volume = 0.0

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(4):
            xi, eta, weight = _quad8_gp_reduced(gp)
            N, _grad, det_shape = _quad8_shape_grad_numba(coords, xi, eta)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det_shape < min_det:
                min_det = det_shape
            if det <= 0.0 or det_shape <= 0.0:
                return block, min_det, volume
            Bv = _quad8_project_b_numba(Pvol, B)
            dV = det * weight * thickness
            for i in range(16):
                volumetric_row = Bv[0, i] + Bv[1, i] + Bv[2, i]
                for j in range(8):
                    block[i, j] += alpha * volumetric_row * N[j] * dV
        return block, min_det, volume

    if mode_code == _QUAD4_MODE_BBAR:
        Bv_acc = np.zeros((4, 16), dtype=np.float64)
        dV_cache = np.zeros(9, dtype=np.float64)
        N_cache = np.zeros((9, 8), dtype=np.float64)
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            N, _grad, det_shape = _quad8_shape_grad_numba(coords, xi, eta)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det_shape < min_det:
                min_det = det_shape
            if det <= 0.0 or det_shape <= 0.0:
                return block, min_det, volume
            dV = det * weight * thickness
            dV_cache[gp] = dV
            volume += dV
            Bv = _quad8_project_b_numba(Pvol, B)
            for i in range(4):
                for j in range(16):
                    Bv_acc[i, j] += Bv[i, j] * dV
            for i in range(8):
                N_cache[gp, i] = N[i]
        if volume <= 0.0:
            return block, min_det, volume
        Bv_bar = Bv_acc / volume
        for gp in range(9):
            dV = dV_cache[gp]
            for i in range(16):
                volumetric_row = Bv_bar[0, i] + Bv_bar[1, i] + Bv_bar[2, i]
                for j in range(8):
                    block[i, j] += alpha * volumetric_row * N_cache[gp, j] * dV
        return block, min_det, volume

    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        N, _grad, det_shape = _quad8_shape_grad_numba(coords, xi, eta)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det_shape < min_det:
            min_det = det_shape
        if det <= 0.0 or det_shape <= 0.0:
            return block, min_det, volume
        dV = det * weight * thickness
        for i in range(16):
            volumetric_row = B[0, i] + B[1, i] + B[2, i]
            for j in range(8):
                block[i, j] += alpha * volumetric_row * N[j] * dV
    return block, min_det, volume

@njit(cache=True)
def _quad8_internal_force_elastic_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    thickness: float,
    mode_code: int,
) -> tuple[np.ndarray, float, float]:
    fe = np.zeros(16, dtype=np.float64)
    min_det = 1.0e300
    volume = 0.0

    if mode_code == _QUAD4_MODE_BBAR:
        B_cache = np.zeros((9, 4, 16), dtype=np.float64)
        dV_cache = np.zeros(9, dtype=np.float64)
        Bv_acc = np.zeros((4, 16), dtype=np.float64)
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return fe, min_det, volume
            dV = det * weight * thickness
            dV_cache[gp] = dV
            volume += dV
            Bv = _quad8_project_b_numba(Pvol, B)
            for r in range(4):
                for c in range(16):
                    B_cache[gp, r, c] = B[r, c]
                    Bv_acc[r, c] += Bv[r, c] * dV
        if volume <= 0.0:
            return fe, min_det, volume
        Bv_bar = Bv_acc / volume
        for gp in range(9):
            B = B_cache[gp]
            Bdev = _quad8_project_b_numba(Idev, B)
            B_eff = Bdev + Bv_bar
            strain = np.zeros(4, dtype=np.float64)
            stress = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B_eff[i, j] * ue[j]
                strain[i] = value
            for i in range(4):
                value = initial_stress[i]
                for j in range(4):
                    value += D4[i, j] * strain[j]
                stress[i] = value
            dV = dV_cache[gp]
            for i in range(16):
                value = 0.0
                for j in range(4):
                    value += B_eff[j, i] * stress[j]
                fe[i] += value * dV
        return fe, min_det, volume

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return fe, min_det, volume
            Bdev = _quad8_project_b_numba(Idev, B)
            strain = np.zeros(4, dtype=np.float64)
            stress = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            for i in range(4):
                value = initial_stress[i]
                for j in range(4):
                    value += D4[i, j] * strain[j]
                stress[i] = value
            dV = det * weight * thickness
            for i in range(16):
                value = 0.0
                for j in range(4):
                    value += Bdev[j, i] * stress[j]
                fe[i] += value * dV
        for gp in range(4):
            xi, eta, weight = _quad8_gp_reduced(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return fe, min_det, volume
            Bv = _quad8_project_b_numba(Pvol, B)
            strain = np.zeros(4, dtype=np.float64)
            stress = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            for i in range(4):
                value = initial_stress[i]
                for j in range(4):
                    value += D4[i, j] * strain[j]
                stress[i] = value
            dV = det * weight * thickness
            for i in range(16):
                value = 0.0
                for j in range(4):
                    value += Bv[j, i] * stress[j]
                fe[i] += value * dV
        return fe, min_det, volume

    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return fe, min_det, volume
        strain = np.zeros(4, dtype=np.float64)
        stress = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = 0.0
            for j in range(16):
                value += B[i, j] * ue[j]
            strain[i] = value
        for i in range(4):
            value = initial_stress[i]
            for j in range(4):
                value += D4[i, j] * strain[j]
            stress[i] = value
        dV = det * weight * thickness
        for i in range(16):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    return fe, min_det, volume

@njit(cache=True)
def _quad8_axisymmetric_b_matrix_numba(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    B = np.zeros((4, 16), dtype=np.float64)
    N, grad, det = _quad8_shape_grad_numba(coords, xi, eta)
    radius = 0.0
    if det <= 0.0:
        return B, N, det, radius
    for i in range(8):
        radius += N[i] * coords[i, 0]
    if radius <= 0.0:
        return B, N, det, radius
    for i in range(8):
        c = 2 * i
        dndr = grad[0, i]
        dndz = grad[1, i]
        B[0, c] = dndr
        B[1, c + 1] = dndz
        B[2, c] = N[i] / radius
        B[3, c] = dndz
        B[3, c + 1] = dndr
    return B, N, det, radius

@njit(cache=True)
def _quad8_axisymmetric_element_stiffness_numba(
    coords: np.ndarray,
    D4: np.ndarray,
    Cdev: np.ndarray,
    Cvol: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    thickness: float,
    mode_code: int,
) -> tuple[np.ndarray, float, float, float]:
    ke = np.zeros((16, 16), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    volume = 0.0

    if mode_code == _QUAD4_MODE_FULL:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if radius < min_radius:
                min_radius = radius
            if det <= 0.0 or radius <= 0.0:
                return ke, min_det, min_radius, volume
            _quad8_add_btcb_numba(ke, B, D4, det * weight * thickness * 2.0 * math.pi * radius)
        _quad8_symmetrize_numba(ke)
        return ke, min_det, min_radius, volume

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if radius < min_radius:
                min_radius = radius
            if det <= 0.0 or radius <= 0.0:
                return ke, min_det, min_radius, volume
            _quad8_add_btcb_numba(ke, B, Cdev, det * weight * thickness * 2.0 * math.pi * radius)
        for gp in range(4):
            xi, eta, weight = _quad8_gp_reduced(gp)
            B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if radius < min_radius:
                min_radius = radius
            if det <= 0.0 or radius <= 0.0:
                return ke, min_det, min_radius, volume
            Bv = _quad8_project_b_numba(Pvol, B)
            _quad8_add_btcb_numba(ke, Bv, Cvol, det * weight * thickness * 2.0 * math.pi * radius)
        _quad8_symmetrize_numba(ke)
        return ke, min_det, min_radius, volume

    B_cache = np.zeros((9, 4, 16), dtype=np.float64)
    dV_cache = np.zeros(9, dtype=np.float64)
    Bv_acc = np.zeros((4, 16), dtype=np.float64)
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if radius < min_radius:
            min_radius = radius
        if det <= 0.0 or radius <= 0.0:
            return ke, min_det, min_radius, volume
        dV = det * weight * thickness * 2.0 * math.pi * radius
        dV_cache[gp] = dV
        volume += dV
        Bv = _quad8_project_b_numba(Pvol, B)
        for r in range(4):
            for c in range(16):
                B_cache[gp, r, c] = B[r, c]
                Bv_acc[r, c] += Bv[r, c] * dV
    if volume <= 0.0:
        return ke, min_det, min_radius, volume
    Bv_bar = Bv_acc / volume
    for gp in range(9):
        B = B_cache[gp]
        Bdev = _quad8_project_b_numba(Idev, B)
        dV = dV_cache[gp]
        _quad8_add_btcb_numba(ke, Bdev, Cdev, dV)
        _quad8_add_btcb_numba(ke, Bv_bar, Cvol, dV)
    _quad8_symmetrize_numba(ke)
    return ke, min_det, min_radius, volume

@njit(cache=True)
def _quad8_axisymmetric_internal_force_elastic_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    thickness: float,
    mode_code: int,
) -> tuple[np.ndarray, float, float, float]:
    fe = np.zeros(16, dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    volume = 0.0

    if mode_code == _QUAD4_MODE_BBAR:
        B_cache = np.zeros((9, 4, 16), dtype=np.float64)
        dV_cache = np.zeros(9, dtype=np.float64)
        Bv_acc = np.zeros((4, 16), dtype=np.float64)
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if radius < min_radius:
                min_radius = radius
            if det <= 0.0 or radius <= 0.0:
                return fe, min_det, min_radius, volume
            dV = det * weight * thickness * 2.0 * math.pi * radius
            dV_cache[gp] = dV
            volume += dV
            Bv = _quad8_project_b_numba(Pvol, B)
            for r in range(4):
                for c in range(16):
                    B_cache[gp, r, c] = B[r, c]
                    Bv_acc[r, c] += Bv[r, c] * dV
        if volume <= 0.0:
            return fe, min_det, min_radius, volume
        Bv_bar = Bv_acc / volume
        for gp in range(9):
            B = B_cache[gp]
            Bdev = _quad8_project_b_numba(Idev, B)
            B_eff = Bdev + Bv_bar
            strain = np.zeros(4, dtype=np.float64)
            stress = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B_eff[i, j] * ue[j]
                strain[i] = value
            for i in range(4):
                value = initial_stress[i]
                for j in range(4):
                    value += D4[i, j] * strain[j]
                stress[i] = value
            dV = dV_cache[gp]
            for i in range(16):
                value = 0.0
                for j in range(4):
                    value += B_eff[j, i] * stress[j]
                fe[i] += value * dV
        return fe, min_det, min_radius, volume

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if radius < min_radius:
                min_radius = radius
            if det <= 0.0 or radius <= 0.0:
                return fe, min_det, min_radius, volume
            Bdev = _quad8_project_b_numba(Idev, B)
            strain = np.zeros(4, dtype=np.float64)
            stress = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            for i in range(4):
                value = initial_stress[i]
                for j in range(4):
                    value += D4[i, j] * strain[j]
                stress[i] = value
            dV = det * weight * thickness * 2.0 * math.pi * radius
            for i in range(16):
                value = 0.0
                for j in range(4):
                    value += Bdev[j, i] * stress[j]
                fe[i] += value * dV
        for gp in range(4):
            xi, eta, weight = _quad8_gp_reduced(gp)
            B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if radius < min_radius:
                min_radius = radius
            if det <= 0.0 or radius <= 0.0:
                return fe, min_det, min_radius, volume
            Bv = _quad8_project_b_numba(Pvol, B)
            strain = np.zeros(4, dtype=np.float64)
            stress = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            for i in range(4):
                value = initial_stress[i]
                for j in range(4):
                    value += D4[i, j] * strain[j]
                stress[i] = value
            dV = det * weight * thickness * 2.0 * math.pi * radius
            for i in range(16):
                value = 0.0
                for j in range(4):
                    value += Bv[j, i] * stress[j]
                fe[i] += value * dV
        return fe, min_det, min_radius, volume

    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if radius < min_radius:
            min_radius = radius
        if det <= 0.0 or radius <= 0.0:
            return fe, min_det, min_radius, volume
        strain = np.zeros(4, dtype=np.float64)
        stress = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = 0.0
            for j in range(16):
                value += B[i, j] * ue[j]
            strain[i] = value
        for i in range(4):
            value = initial_stress[i]
            for j in range(4):
                value += D4[i, j] * strain[j]
            stress[i] = value
        dV = det * weight * thickness * 2.0 * math.pi * radius
        for i in range(16):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    return fe, min_det, min_radius, volume

@njit(cache=True)
def _quad8_axisymmetric_pressure_matrices_numba(
    coords: np.ndarray,
    storage: float,
    permeability: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    me = np.zeros((8, 8), dtype=np.float64)
    ke = np.zeros((8, 8), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        N, grad, det = _quad8_shape_grad_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return me, ke, min_det, min_radius
        radius = 0.0
        for i in range(8):
            radius += N[i] * coords[i, 0]
        if radius < min_radius:
            min_radius = radius
        if radius <= 0.0:
            return me, ke, min_det, min_radius
        dV = det * weight * thickness * 2.0 * math.pi * radius
        for i in range(8):
            for j in range(8):
                me[i, j] += storage * N[i] * N[j] * dV
                ke[i, j] += permeability * (grad[0, i] * grad[0, j] + grad[1, i] * grad[1, j]) * dV
    return me, ke, min_det, min_radius

@njit(cache=True)
def _quad8_axisymmetric_biot_matrix_numba(coords: np.ndarray, alpha: float, thickness: float) -> tuple[np.ndarray, float, float]:
    block = np.zeros((16, 8), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        B, N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if radius < min_radius:
            min_radius = radius
        if det <= 0.0 or radius <= 0.0:
            return block, min_det, min_radius
        dV = det * weight * thickness * 2.0 * math.pi * radius
        for i in range(16):
            volumetric_row = B[0, i] + B[1, i] + B[2, i]
            for j in range(8):
                block[i, j] += alpha * volumetric_row * N[j] * dV
    return block, min_det, min_radius

@njit(cache=True)
def _quad8_axisymmetric_edge_traction_numba(points: np.ndarray, tx: float, ty: float) -> tuple[np.ndarray, float, float]:
    nnode = points.shape[0]
    fe = np.zeros(2 * nnode, dtype=np.float64)
    min_radius = 1.0e300
    min_jac = 1.0e300
    if nnode == 2:
        dx = points[1, 0] - points[0, 0]
        dy = points[1, 1] - points[0, 1]
        jac = math.sqrt(dx * dx + dy * dy)
        radius = 0.5 * (points[0, 0] + points[1, 0])
        min_radius = radius
        min_jac = jac
        if radius <= 0.0 or jac <= 0.0:
            return fe, min_radius, min_jac
        surface = 2.0 * math.pi * radius * jac
        fe[0] = tx * surface * 0.5
        fe[1] = ty * surface * 0.5
        fe[2] = tx * surface * 0.5
        fe[3] = ty * surface * 0.5
        return fe, min_radius, min_jac

    for gp in range(3):
        if gp == 0:
            s = -math.sqrt(3.0 / 5.0)
            w = 5.0 / 9.0
        elif gp == 1:
            s = 0.0
            w = 8.0 / 9.0
        else:
            s = math.sqrt(3.0 / 5.0)
            w = 5.0 / 9.0
        n0 = 0.5 * s * (s - 1.0)
        n1 = 1.0 - s * s
        n2 = 0.5 * s * (s + 1.0)
        d0 = s - 0.5
        d1 = -2.0 * s
        d2 = s + 0.5
        txg = d0 * points[0, 0] + d1 * points[1, 0] + d2 * points[2, 0]
        tyg = d0 * points[0, 1] + d1 * points[1, 1] + d2 * points[2, 1]
        jac = math.sqrt(txg * txg + tyg * tyg)
        radius = n0 * points[0, 0] + n1 * points[1, 0] + n2 * points[2, 0]
        if radius < min_radius:
            min_radius = radius
        if jac < min_jac:
            min_jac = jac
        if radius <= 0.0 or jac <= 0.0:
            return fe, min_radius, min_jac
        dS = 2.0 * math.pi * radius * jac * w
        fe[0] += n0 * tx * dS
        fe[1] += n0 * ty * dS
        fe[2] += n1 * tx * dS
        fe[3] += n1 * ty * dS
        fe[4] += n2 * tx * dS
        fe[5] += n2 * ty * dS
    return fe, min_radius, min_jac

def _quad4_element_stiffness_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial, mode: str) -> np.ndarray:
    mode_code = _quad4_mode_code(mode)
    coords64 = np.ascontiguousarray(coords, dtype=np.float64)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    ke, min_det, volume = _quad4_element_stiffness_numba(
        coords64,
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(material.C_dev, dtype=np.float64),
        np.ascontiguousarray(material.C_vol, dtype=np.float64),
        Pvol,
        Idev,
        float(material.thickness),
        mode_code,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD4: non-positive element measure")
    return ke

def _quad4_pressure_matrices_fast(
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    storage: float,
    permeability: float,
) -> tuple[np.ndarray, np.ndarray]:
    me, ke, min_det = _quad4_pressure_matrices_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(storage),
        float(permeability),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return me, ke

def _quad4_biot_matrix_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial, alpha: float) -> np.ndarray:
    coords64 = np.ascontiguousarray(coords, dtype=np.float64)
    try:
        block, min_det = _quad4_biot_matrix_numba(coords64, float(alpha), float(material.thickness))
    except (RuntimeError, SystemError):
        block, min_det = _quad4_biot_matrix_python(coords64, float(alpha), float(material.thickness))
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return block

def _quad4_consistent_mass_matrix_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial, density: float) -> np.ndarray:
    matrix, min_det = _quad4_consistent_mass_matrix_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(density),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return matrix

def _quad8_element_stiffness_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial, mode: str) -> np.ndarray:
    mode_code = _quad4_mode_code(mode)
    coords64 = np.ascontiguousarray(coords, dtype=np.float64)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    ke, min_det, volume = _quad8_element_stiffness_numba(
        coords64,
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(material.C_dev, dtype=np.float64),
        np.ascontiguousarray(material.C_vol, dtype=np.float64),
        Pvol,
        Idev,
        float(material.thickness),
        mode_code,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return ke

def _quad8_pressure_matrices_fast(
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    storage: float,
    permeability: float,
) -> tuple[np.ndarray, np.ndarray]:
    me, ke, min_det = _quad8_pressure_matrices_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(storage),
        float(permeability),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return me, ke

def _quad8_biot_matrix_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial, alpha: float, mode: str = "FULL") -> np.ndarray:
    mode_code = _quad4_mode_code(normalize_integration(mode))
    block, min_det, volume = _quad8_biot_matrix_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(alpha),
        float(material.thickness),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        mode_code,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return block

def _quad8_consistent_mass_matrix_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial, density: float) -> np.ndarray:
    matrix, min_det = _quad8_consistent_mass_matrix_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(density),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return matrix

def _quad8_internal_force_elastic_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    mode: str,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    mode_code = _quad4_mode_code(mode)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    fe, min_det, volume = _quad8_internal_force_elastic_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        Pvol,
        Idev,
        float(material.thickness),
        mode_code,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return fe

def _quad4_axisymmetric_pressure_matrices_fast(
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    storage: float,
    permeability: float,
) -> tuple[np.ndarray, np.ndarray]:
    me, ke, min_det, min_radius = _quad4_axisymmetric_pressure_matrices_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(storage),
        float(permeability),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD4: axisymmetric radius must be positive, got {min_radius:.6e}")
    return me, ke

def _quad4_axisymmetric_biot_matrix_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial, alpha: float) -> np.ndarray:
    block, min_det, min_radius = _quad4_axisymmetric_biot_matrix_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(alpha),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD4: axisymmetric radius must be positive, got {min_radius:.6e}")
    return block

def _quad4_axisymmetric_element_stiffness_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial) -> np.ndarray:
    ke, min_det, min_radius = _quad4_axisymmetric_element_stiffness_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD4: axisymmetric radius must be positive, got {min_radius:.6e}")
    return ke

def _quad8_axisymmetric_element_stiffness_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial, mode: str) -> np.ndarray:
    mode_code = _quad4_mode_code(mode)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    ke, min_det, min_radius, volume = _quad8_axisymmetric_element_stiffness_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(material.C_dev, dtype=np.float64),
        np.ascontiguousarray(material.C_vol, dtype=np.float64),
        Pvol,
        Idev,
        float(material.thickness),
        mode_code,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD8: axisymmetric radius must be positive, got {min_radius:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive axisymmetric element measure")
    return ke

def _quad8_axisymmetric_pressure_matrices_fast(
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    storage: float,
    permeability: float,
) -> tuple[np.ndarray, np.ndarray]:
    me, ke, min_det, min_radius = _quad8_axisymmetric_pressure_matrices_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(storage),
        float(permeability),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD8: axisymmetric radius must be positive, got {min_radius:.6e}")
    return me, ke

def _quad8_axisymmetric_biot_matrix_fast(coords: np.ndarray, material: ElasticPlaneStrainMaterial, alpha: float) -> np.ndarray:
    block, min_det, min_radius = _quad8_axisymmetric_biot_matrix_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(alpha),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD8: axisymmetric radius must be positive, got {min_radius:.6e}")
    return block

def _quad4_axisymmetric_internal_force_elastic_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    fe, min_det, min_radius = _quad4_axisymmetric_internal_force_elastic_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD4: axisymmetric radius must be positive, got {min_radius:.6e}")
    return fe

def _quad8_axisymmetric_internal_force_elastic_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    mode: str,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    mode_code = _quad4_mode_code(mode)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    fe, min_det, min_radius, volume = _quad8_axisymmetric_internal_force_elastic_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        Pvol,
        Idev,
        float(material.thickness),
        mode_code,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD8: axisymmetric radius must be positive, got {min_radius:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive axisymmetric element measure")
    return fe

def _quad8_axisymmetric_edge_traction_fast(points: np.ndarray, tx: float, ty: float) -> np.ndarray:
    pts = np.ascontiguousarray(points, dtype=np.float64)
    if pts.shape not in {(2, 2), (3, 2)}:
        raise FEM2DError("axisymmetric edge traction supports 2-node or 3-node edges")
    fe, min_radius, min_jac = _quad8_axisymmetric_edge_traction_numba(pts, float(tx), float(ty))
    if min_jac <= 0.0:
        raise FEM2DError(f"axisymmetric edge traction length must be positive, got {min_jac:.6e}")
    if min_radius < -np.finfo(float).eps:
        raise FEM2DError(f"axisymmetric edge traction radius must be positive, got {min_radius:.6e}")
    return fe

def _quad4_internal_force_elastic_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial_stress: np.ndarray | None = None,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    fe, min_det = _quad4_internal_force_elastic_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return fe


__all__ = list(ELASTIC_ELEMENT_KERNEL_FUNCTIONS) + ["_QUAD4_MODE_FULL", "_QUAD4_MODE_SRI", "_QUAD4_MODE_BBAR"]
