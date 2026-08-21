"""Shared Numba geometry and algebra primitives for QUAD4/QUAD8 elements."""

from __future__ import annotations

from typing import Any
import math

import numpy as np

from .fem2d_types import njit


ELEMENT_NUMBA_PRIMITIVE_FUNCTIONS = (
    "element_numba_primitives_contract",
    "_quad4_b_det_numba",
    "_quad4_shape_grad_numba",
    "_quad4_project_b_numba",
    "_quad4_add_btcb_numba",
    "_quad4_symmetrize_numba",
    "_quad8_shape_grad_numba",
    "_quad8_b_det_numba",
    "_quad8_project_b_numba",
    "_quad8_add_btcb_numba",
    "_quad8_add_btlcbr_numba",
    "_quad8_add_btstress_numba",
    "_quad8_symmetrize_numba",
    "_quad8_gp_full",
    "_quad8_gp_reduced",
)


def element_numba_primitives_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_numba_primitives.v1",
        "module": "geofem_app.fem2d_element_numba_primitives",
        "function_count": len(ELEMENT_NUMBA_PRIMITIVE_FUNCTIONS),
        "functions": list(ELEMENT_NUMBA_PRIMITIVE_FUNCTIONS),
        "covered_surfaces": [
            "quad4_b_matrix_and_jacobian",
            "quad4_matrix_accumulation",
            "quad8_shape_gradient_and_b_matrix",
            "quad8_matrix_accumulation",
            "quad8_gauss_rules",
        ],
    }


@njit(cache=True)
def _quad4_b_det_numba(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    dxi = np.empty(4, dtype=np.float64)
    deta = np.empty(4, dtype=np.float64)
    dxi[0] = -0.25 * (1.0 - eta)
    dxi[1] = 0.25 * (1.0 - eta)
    dxi[2] = 0.25 * (1.0 + eta)
    dxi[3] = -0.25 * (1.0 + eta)
    deta[0] = -0.25 * (1.0 - xi)
    deta[1] = -0.25 * (1.0 + xi)
    deta[2] = 0.25 * (1.0 + xi)
    deta[3] = 0.25 * (1.0 - xi)

    j00 = 0.0
    j01 = 0.0
    j10 = 0.0
    j11 = 0.0
    for i in range(4):
        x = coords[i, 0]
        y = coords[i, 1]
        j00 += dxi[i] * x
        j01 += dxi[i] * y
        j10 += deta[i] * x
        j11 += deta[i] * y
    det = j00 * j11 - j01 * j10
    B = np.zeros((4, 8), dtype=np.float64)
    if det <= 0.0:
        return B, det
    inv00 = j11 / det
    inv01 = -j01 / det
    inv10 = -j10 / det
    inv11 = j00 / det
    for i in range(4):
        dndx = inv00 * dxi[i] + inv01 * deta[i]
        dndy = inv10 * dxi[i] + inv11 * deta[i]
        c = 2 * i
        B[0, c] = dndx
        B[1, c + 1] = dndy
        B[3, c] = dndy
        B[3, c + 1] = dndx
    return B, det

@njit(cache=True)
def _quad4_shape_grad_numba(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float]:
    N = np.empty(4, dtype=np.float64)
    N[0] = 0.25 * (1.0 - xi) * (1.0 - eta)
    N[1] = 0.25 * (1.0 + xi) * (1.0 - eta)
    N[2] = 0.25 * (1.0 + xi) * (1.0 + eta)
    N[3] = 0.25 * (1.0 - xi) * (1.0 + eta)

    dxi = np.empty(4, dtype=np.float64)
    deta = np.empty(4, dtype=np.float64)
    dxi[0] = -0.25 * (1.0 - eta)
    dxi[1] = 0.25 * (1.0 - eta)
    dxi[2] = 0.25 * (1.0 + eta)
    dxi[3] = -0.25 * (1.0 + eta)
    deta[0] = -0.25 * (1.0 - xi)
    deta[1] = -0.25 * (1.0 + xi)
    deta[2] = 0.25 * (1.0 + xi)
    deta[3] = 0.25 * (1.0 - xi)

    j00 = 0.0
    j01 = 0.0
    j10 = 0.0
    j11 = 0.0
    for i in range(4):
        x = coords[i, 0]
        y = coords[i, 1]
        j00 += dxi[i] * x
        j01 += dxi[i] * y
        j10 += deta[i] * x
        j11 += deta[i] * y
    det = j00 * j11 - j01 * j10
    grad = np.zeros((2, 4), dtype=np.float64)
    if det <= 0.0:
        return N, grad, det
    inv00 = j11 / det
    inv01 = -j01 / det
    inv10 = -j10 / det
    inv11 = j00 / det
    for i in range(4):
        grad[0, i] = inv00 * dxi[i] + inv01 * deta[i]
        grad[1, i] = inv10 * dxi[i] + inv11 * deta[i]
    return N, grad, det

@njit(cache=True)
def _quad4_project_b_numba(projector: np.ndarray, B: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 8), dtype=np.float64)
    for i in range(4):
        for j in range(8):
            value = 0.0
            for k in range(4):
                value += projector[i, k] * B[k, j]
            out[i, j] = value
    return out

@njit(cache=True)
def _quad4_add_btcb_numba(ke: np.ndarray, B: np.ndarray, C: np.ndarray, scale: float) -> None:
    for i in range(8):
        for j in range(8):
            value = 0.0
            for a in range(4):
                bia = B[a, i]
                if bia == 0.0:
                    continue
                for b in range(4):
                    value += bia * C[a, b] * B[b, j]
            ke[i, j] += value * scale

@njit(cache=True)
def _quad4_symmetrize_numba(matrix: np.ndarray) -> None:
    for i in range(8):
        for j in range(i + 1, 8):
            value = 0.5 * (matrix[i, j] + matrix[j, i])
            matrix[i, j] = value
            matrix[j, i] = value

@njit(cache=True)
def _quad8_shape_grad_numba(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float]:
    x = xi
    y = eta
    N = np.empty(8, dtype=np.float64)
    N[0] = -0.25 * (1.0 - x) * (1.0 - y) * (1.0 + x + y)
    N[1] = -0.25 * (1.0 + x) * (1.0 - y) * (1.0 - x + y)
    N[2] = -0.25 * (1.0 + x) * (1.0 + y) * (1.0 - x - y)
    N[3] = -0.25 * (1.0 - x) * (1.0 + y) * (1.0 + x - y)
    N[4] = 0.5 * (1.0 - x * x) * (1.0 - y)
    N[5] = 0.5 * (1.0 + x) * (1.0 - y * y)
    N[6] = 0.5 * (1.0 - x * x) * (1.0 + y)
    N[7] = 0.5 * (1.0 - x) * (1.0 - y * y)

    dxi = np.empty(8, dtype=np.float64)
    dxi[0] = 0.25 * (1.0 - y) * (2.0 * x + y)
    dxi[1] = 0.25 * (1.0 - y) * (2.0 * x - y)
    dxi[2] = 0.25 * (1.0 + y) * (2.0 * x + y)
    dxi[3] = 0.25 * (1.0 + y) * (2.0 * x - y)
    dxi[4] = -x * (1.0 - y)
    dxi[5] = 0.5 * (1.0 - y * y)
    dxi[6] = -x * (1.0 + y)
    dxi[7] = -0.5 * (1.0 - y * y)

    deta = np.empty(8, dtype=np.float64)
    deta[0] = 0.25 * (1.0 - x) * (x + 2.0 * y)
    deta[1] = 0.25 * (1.0 + x) * (2.0 * y - x)
    deta[2] = 0.25 * (1.0 + x) * (x + 2.0 * y)
    deta[3] = 0.25 * (1.0 - x) * (2.0 * y - x)
    deta[4] = -0.5 * (1.0 - x * x)
    deta[5] = -(1.0 + x) * y
    deta[6] = 0.5 * (1.0 - x * x)
    deta[7] = -(1.0 - x) * y

    j00 = 0.0
    j01 = 0.0
    j10 = 0.0
    j11 = 0.0
    for i in range(8):
        px = coords[i, 0]
        py = coords[i, 1]
        j00 += dxi[i] * px
        j01 += dxi[i] * py
        j10 += deta[i] * px
        j11 += deta[i] * py
    det = j00 * j11 - j01 * j10
    grad = np.zeros((2, 8), dtype=np.float64)
    if det <= 0.0:
        return N, grad, det
    inv00 = j11 / det
    inv01 = -j01 / det
    inv10 = -j10 / det
    inv11 = j00 / det
    for i in range(8):
        grad[0, i] = inv00 * dxi[i] + inv01 * deta[i]
        grad[1, i] = inv10 * dxi[i] + inv11 * deta[i]
    return N, grad, det

@njit(cache=True)
def _quad8_b_det_numba(coords: np.ndarray, xi: float, eta: float) -> tuple[np.ndarray, float]:
    _N, grad, det = _quad8_shape_grad_numba(coords, xi, eta)
    B = np.zeros((4, 16), dtype=np.float64)
    if det <= 0.0:
        return B, det
    for i in range(8):
        c = 2 * i
        dndx = grad[0, i]
        dndy = grad[1, i]
        B[0, c] = dndx
        B[1, c + 1] = dndy
        B[3, c] = dndy
        B[3, c + 1] = dndx
    return B, det

@njit(cache=True)
def _quad8_project_b_numba(projector: np.ndarray, B: np.ndarray) -> np.ndarray:
    out = np.zeros((4, 16), dtype=np.float64)
    for i in range(4):
        for j in range(16):
            value = 0.0
            for k in range(4):
                value += projector[i, k] * B[k, j]
            out[i, j] = value
    return out

@njit(cache=True)
def _quad8_add_btcb_numba(ke: np.ndarray, B: np.ndarray, C: np.ndarray, scale: float) -> None:
    for i in range(16):
        for j in range(16):
            value = 0.0
            for a in range(4):
                bia = B[a, i]
                if bia == 0.0:
                    continue
                for b in range(4):
                    value += bia * C[a, b] * B[b, j]
            ke[i, j] += value * scale

@njit(cache=True)
def _quad8_add_btlcbr_numba(ke: np.ndarray, B_left: np.ndarray, C: np.ndarray, B_right: np.ndarray, scale: float) -> None:
    for i in range(16):
        for j in range(16):
            value = 0.0
            for a in range(4):
                bia = B_left[a, i]
                if bia == 0.0:
                    continue
                for b in range(4):
                    value += bia * C[a, b] * B_right[b, j]
            ke[i, j] += value * scale

@njit(cache=True)
def _quad8_add_btstress_numba(fe: np.ndarray, B: np.ndarray, stress: np.ndarray, scale: float) -> None:
    for i in range(16):
        value = 0.0
        for j in range(4):
            value += B[j, i] * stress[j]
        fe[i] += value * scale

@njit(cache=True)
def _quad8_symmetrize_numba(matrix: np.ndarray) -> None:
    for i in range(16):
        for j in range(i + 1, 16):
            value = 0.5 * (matrix[i, j] + matrix[j, i])
            matrix[i, j] = value
            matrix[j, i] = value

@njit(cache=True)
def _quad8_gp_full(gp: int) -> tuple[float, float, float]:
    a = math.sqrt(3.0 / 5.0)
    ix = gp // 3
    iy = gp - ix * 3
    if ix == 0:
        xi = -a
        wx = 5.0 / 9.0
    elif ix == 1:
        xi = 0.0
        wx = 8.0 / 9.0
    else:
        xi = a
        wx = 5.0 / 9.0
    if iy == 0:
        eta = -a
        wy = 5.0 / 9.0
    elif iy == 1:
        eta = 0.0
        wy = 8.0 / 9.0
    else:
        eta = a
        wy = 5.0 / 9.0
    return xi, eta, wx * wy

@njit(cache=True)
def _quad8_gp_reduced(gp: int) -> tuple[float, float, float]:
    a = 1.0 / math.sqrt(3.0)
    if gp == 0:
        return -a, -a, 1.0
    if gp == 1:
        return a, -a, 1.0
    if gp == 2:
        return a, a, 1.0
    return -a, a, 1.0


__all__ = list(ELEMENT_NUMBA_PRIMITIVE_FUNCTIONS)
