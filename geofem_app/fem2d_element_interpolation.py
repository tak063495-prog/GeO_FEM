"""Element interpolation, integration-point, and B-matrix helpers."""

from __future__ import annotations

from typing import Any
import math

import numpy as np

from .fem2d_types import FEM2DError


ELEMENT_INTERPOLATION_FUNCTIONS = (
    "element_interpolation_contract",
    "strain_displacement_matrix",
    "axisymmetric_strain_displacement_matrix",
    "shape_functions",
    "integration_points",
)


def element_interpolation_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_interpolation.v1",
        "module": "geofem_app.fem2d_element_interpolation",
        "function_count": len(ELEMENT_INTERPOLATION_FUNCTIONS),
        "functions": list(ELEMENT_INTERPOLATION_FUNCTIONS),
        "covered_surfaces": [
            "shape_functions",
            "gauss_integration_points",
            "plane_strain_b_matrix",
            "axisymmetric_b_matrix",
            "jacobian_validation",
        ],
    }


def strain_displacement_matrix(element_type: str, coords: np.ndarray, gp: tuple[float, float, float]) -> tuple[np.ndarray, float, np.ndarray]:
    xi, eta, _weight = gp
    N, dN_dnatural = shape_functions(element_type, xi, eta)
    jac = dN_dnatural @ coords
    detJ = float(np.linalg.det(jac))
    if detJ <= 0.0:
        raise FEM2DError(f"{element_type}: detJ must be positive, got {detJ:.6e}")
    invJ = np.linalg.inv(jac)
    dN_dxdy = invJ @ dN_dnatural
    nnode = coords.shape[0]
    B4 = np.zeros((4, 2 * nnode), dtype=float)
    for i in range(nnode):
        dNdx = dN_dxdy[0, i]
        dNdy = dN_dxdy[1, i]
        B4[0, 2 * i] = dNdx
        B4[1, 2 * i + 1] = dNdy
        B4[3, 2 * i] = dNdy
        B4[3, 2 * i + 1] = dNdx
    return B4, detJ, N


def axisymmetric_strain_displacement_matrix(element_type: str, coords: np.ndarray, gp: tuple[float, float, float]) -> tuple[np.ndarray, float, np.ndarray, float]:
    xi, eta, _weight = gp
    N, dN_dnatural = shape_functions(element_type, xi, eta)
    jac = dN_dnatural @ coords
    detJ = float(np.linalg.det(jac))
    if detJ <= 0.0:
        raise FEM2DError(f"{element_type}: detJ must be positive, got {detJ:.6e}")
    invJ = np.linalg.inv(jac)
    dN_dxdy = invJ @ dN_dnatural
    radius = float(N @ coords[:, 0])
    if radius <= 0.0:
        raise FEM2DError(f"{element_type}: axisymmetric radius must be positive, got {radius:.6e}")
    nnode = coords.shape[0]
    B4 = np.zeros((4, 2 * nnode), dtype=float)
    for i in range(nnode):
        dNdr = dN_dxdy[0, i]
        dNdz = dN_dxdy[1, i]
        B4[0, 2 * i] = dNdr
        B4[1, 2 * i + 1] = dNdz
        B4[2, 2 * i] = N[i] / radius
        B4[3, 2 * i] = dNdz
        B4[3, 2 * i + 1] = dNdr
    return B4, detJ, N, radius


def shape_functions(element_type: str, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray]:
    etype = element_type.upper()
    if etype == "TRI3":
        N = np.array([1.0 - xi - eta, xi, eta], dtype=float)
        dN = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]], dtype=float)
        return N, dN
    if etype == "TRI6":
        L1 = 1.0 - xi - eta
        L2 = xi
        L3 = eta
        N = np.array(
            [
                L1 * (2.0 * L1 - 1.0),
                L2 * (2.0 * L2 - 1.0),
                L3 * (2.0 * L3 - 1.0),
                4.0 * L1 * L2,
                4.0 * L2 * L3,
                4.0 * L3 * L1,
            ],
            dtype=float,
        )
        dN_dxi = np.array(
            [
                -(4.0 * L1 - 1.0),
                4.0 * L2 - 1.0,
                0.0,
                4.0 * (L1 - L2),
                4.0 * L3,
                -4.0 * L3,
            ],
            dtype=float,
        )
        dN_deta = np.array(
            [
                -(4.0 * L1 - 1.0),
                0.0,
                4.0 * L3 - 1.0,
                -4.0 * L2,
                4.0 * L2,
                4.0 * (L1 - L3),
            ],
            dtype=float,
        )
        return N, np.vstack([dN_dxi, dN_deta])
    if etype == "QUAD4":
        x = xi
        y = eta
        N = 0.25 * np.array(
            [
                (1.0 - x) * (1.0 - y),
                (1.0 + x) * (1.0 - y),
                (1.0 + x) * (1.0 + y),
                (1.0 - x) * (1.0 + y),
            ],
            dtype=float,
        )
        dN_dxi = 0.25 * np.array([-(1.0 - y), 1.0 - y, 1.0 + y, -(1.0 + y)], dtype=float)
        dN_deta = 0.25 * np.array([-(1.0 - x), -(1.0 + x), 1.0 + x, 1.0 - x], dtype=float)
        return N, np.vstack([dN_dxi, dN_deta])
    if etype == "QUAD8":
        x = xi
        y = eta
        N = np.array(
            [
                -0.25 * (1.0 - x) * (1.0 - y) * (1.0 + x + y),
                -0.25 * (1.0 + x) * (1.0 - y) * (1.0 - x + y),
                -0.25 * (1.0 + x) * (1.0 + y) * (1.0 - x - y),
                -0.25 * (1.0 - x) * (1.0 + y) * (1.0 + x - y),
                0.5 * (1.0 - x * x) * (1.0 - y),
                0.5 * (1.0 + x) * (1.0 - y * y),
                0.5 * (1.0 - x * x) * (1.0 + y),
                0.5 * (1.0 - x) * (1.0 - y * y),
            ],
            dtype=float,
        )
        dN_dxi = np.array(
            [
                0.25 * (1.0 - y) * (2.0 * x + y),
                0.25 * (1.0 - y) * (2.0 * x - y),
                0.25 * (1.0 + y) * (2.0 * x + y),
                0.25 * (1.0 + y) * (2.0 * x - y),
                -x * (1.0 - y),
                0.5 * (1.0 - y * y),
                -x * (1.0 + y),
                -0.5 * (1.0 - y * y),
            ],
            dtype=float,
        )
        dN_deta = np.array(
            [
                0.25 * (1.0 - x) * (x + 2.0 * y),
                0.25 * (1.0 + x) * (2.0 * y - x),
                0.25 * (1.0 + x) * (x + 2.0 * y),
                0.25 * (1.0 - x) * (2.0 * y - x),
                -0.5 * (1.0 - x * x),
                -(1.0 + x) * y,
                0.5 * (1.0 - x * x),
                -(1.0 - x) * y,
            ],
            dtype=float,
        )
        return N, np.vstack([dN_dxi, dN_deta])
    raise FEM2DError(f"unsupported element type '{element_type}'")


def integration_points(element_type: str, mode: str) -> list[tuple[float, float, float]]:
    etype = element_type.upper()
    imode = mode.upper()
    if etype.startswith("TRI"):
        if imode == "REDUCED":
            return [(1.0 / 3.0, 1.0 / 3.0, 0.5)]
        if etype == "TRI3":
            return [(1.0 / 3.0, 1.0 / 3.0, 0.5)]
        return [(1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0), (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0), (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)]
    if etype == "QUAD4":
        if imode == "REDUCED":
            return [(0.0, 0.0, 4.0)]
        a = 1.0 / math.sqrt(3.0)
        return [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]
    if etype == "QUAD8":
        if imode == "REDUCED":
            a = 1.0 / math.sqrt(3.0)
            return [(-a, -a, 1.0), (a, -a, 1.0), (a, a, 1.0), (-a, a, 1.0)]
        a = math.sqrt(3.0 / 5.0)
        one_d = [(-a, 5.0 / 9.0), (0.0, 8.0 / 9.0), (a, 5.0 / 9.0)]
        return [(x, y, wx * wy) for x, wx in one_d for y, wy in one_d]
    raise FEM2DError(f"unsupported element type '{element_type}'")


__all__ = [
    "ELEMENT_INTERPOLATION_FUNCTIONS",
    "element_interpolation_contract",
    "strain_displacement_matrix",
    "axisymmetric_strain_displacement_matrix",
    "shape_functions",
    "integration_points",
]
