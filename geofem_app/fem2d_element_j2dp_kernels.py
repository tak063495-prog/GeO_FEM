"""J2/Drucker-Prager nonlinear element kernels for plane-strain and axisymmetric analyses."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fem2d_element_elastic_kernels import (
    _quad4_axisymmetric_b_matrix_numba,
    _quad8_axisymmetric_b_matrix_numba,
)
from .fem2d_element_elastic_post import _quad4_post_principal_values_numba
from .fem2d_element_numba_primitives import (
    _quad4_add_btcb_numba,
    _quad4_b_det_numba,
    _quad4_project_b_numba,
    _quad4_symmetrize_numba,
    _quad8_add_btcb_numba,
    _quad8_add_btlcbr_numba,
    _quad8_add_btstress_numba,
    _quad8_b_det_numba,
    _quad8_gp_full,
    _quad8_gp_reduced,
    _quad8_project_b_numba,
    _quad8_shape_grad_numba,
    _quad8_symmetrize_numba,
)
from .fem2d_materials import (
    _j2dp_tension_cutoff_numerical_tangent_numba,
    _j2dp_tension_cutoff_stress_numba,
    _tension_cutoff_plane_strain_numba,
)
from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, njit

_QUAD4_MODE_FULL = 0
_QUAD4_MODE_SRI = 1
_QUAD4_MODE_BBAR = 2

J2DP_ELEMENT_KERNEL_FUNCTIONS = (
    "j2dp_element_kernel_contract",
    "_j2dp_post_update_numba",
    "_j2dp_tension_cutoff_post_update_numba",
    "_j2dp_stress_tangent_numba",
    "_j2dp_maybe_tension_stress_tangent_numba",
    "_quad4_j2dp_post_numba",
    "_quad4_j2dp_bbar_post_numba",
    "_quad8_j2dp_post_numba",
    "_quad8_j2dp_bbar_post_numba",
    "_quad4_j2dp_tangent_force_numba",
    "_quad8_j2dp_tangent_force_numba",
    "_quad4_axisymmetric_j2dp_tangent_force_numba",
    "_quad8_axisymmetric_j2dp_tangent_force_numba",
    "_quad8_axisymmetric_j2dp_post_numba",
    "_quad8_axisymmetric_j2dp_bbar_post_numba",
    "_quad8_j2dp_tangent_force_fast",
    "_quad4_j2dp_post_fast",
    "_quad4_j2dp_bbar_post_fast",
    "_quad8_j2dp_post_fast",
    "_quad8_j2dp_bbar_post_fast",
    "_quad4_j2dp_tangent_force_fast",
    "_quad4_axisymmetric_j2dp_tangent_force_fast",
    "_quad8_axisymmetric_j2dp_tangent_force_fast",
    "_quad8_axisymmetric_j2dp_post_fast",
    "_quad8_axisymmetric_j2dp_bbar_post_fast",
)


def _quad4_mode_code(mode: str) -> int:
    if mode == "FULL":
        return _QUAD4_MODE_FULL
    if mode == "SRI":
        return _QUAD4_MODE_SRI
    if mode == "B-BAR":
        return _QUAD4_MODE_BBAR
    raise FEM2DError(f"unsupported integration '{mode}'")


def j2dp_element_kernel_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_j2dp_kernels.v1",
        "module": "geofem_app.fem2d_element_j2dp_kernels",
        "function_count": len(J2DP_ELEMENT_KERNEL_FUNCTIONS),
        "functions": list(J2DP_ELEMENT_KERNEL_FUNCTIONS),
        "covered_surfaces": [
            "quad4_j2dp_tangent_internal_force",
            "quad8_j2dp_tangent_internal_force",
            "quad4_quad8_j2dp_post",
            "quad8_axisymmetric_j2dp_tangent_internal_force",
            "quad8_axisymmetric_j2dp_post",
            "full_sri_bbar_dispatch",
        ],
    }


@njit(cache=True)
def _j2dp_pq_numba(stress: np.ndarray) -> tuple[float, float]:
    mean = (stress[0] + stress[1] + stress[2]) / 3.0
    d0 = stress[0] - mean
    d1 = stress[1] - mean
    d2 = stress[2] - mean
    d3 = stress[3]
    q = math.sqrt(max(1.5 * (d0 * d0 + d1 * d1 + d2 * d2) + 3.0 * d3 * d3, 0.0))
    return -mean, q


@njit(cache=True)
def _j2dp_post_update_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
) -> tuple[np.ndarray, float, float, float, float, np.ndarray, float]:
    trial = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = initial_stress[i]
        for j in range(4):
            value += D4[i, j] * (strain[j] - plastic_strain[j])
        trial[i] = value

    mean = (trial[0] + trial[1] + trial[2]) / 3.0
    dev0 = trial[0] - mean
    dev1 = trial[1] - mean
    dev2 = trial[2] - mean
    dev3 = trial[3]
    q = math.sqrt(max(1.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + 3.0 * dev3 * dev3, 0.0))
    p = -mean
    yield_trial = q - alpha * p - cohesion_term - hardening * kappa
    tol_ref = max(q, abs(alpha * p) + cohesion_term, 1.0)
    if yield_trial <= max(1.0e-10, 1.0e-10 * tol_ref):
        return trial, 0.0, max(yield_trial, 0.0), p, q, plastic_strain.copy(), kappa

    eps = np.finfo(np.float64).eps
    denom = max(3.0 * shear_mu + hardening, eps)
    dgamma = max(yield_trial / denom, 0.0)
    kappa_new = kappa + dgamma
    q_limit = max(alpha * p + cohesion_term + hardening * kappa_new, 0.0)
    scale = 0.0 if q <= eps else min(q_limit / q, 1.0)
    corrected = np.empty(4, dtype=np.float64)
    corrected[0] = mean + scale * dev0
    corrected[1] = mean + scale * dev1
    corrected[2] = mean + scale * dev2
    corrected[3] = scale * dev3
    elastic_strain_new = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = 0.0
        for j in range(4):
            value += S4[i, j] * (corrected[j] - initial_stress[j])
        elastic_strain_new[i] = value
    plastic_strain_new = np.empty(4, dtype=np.float64)
    for i in range(4):
        plastic_strain_new[i] = strain[i] - elastic_strain_new[i]
    mean_new = (corrected[0] + corrected[1] + corrected[2]) / 3.0
    nd0 = corrected[0] - mean_new
    nd1 = corrected[1] - mean_new
    nd2 = corrected[2] - mean_new
    nd3 = corrected[3]
    q_new = math.sqrt(max(1.5 * (nd0 * nd0 + nd1 * nd1 + nd2 * nd2) + 3.0 * nd3 * nd3, 0.0))
    return corrected, 1.0, max(yield_trial, 0.0), -mean_new, q_new, plastic_strain_new, kappa_new


@njit(cache=True)
def _j2dp_tension_cutoff_post_update_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    tensile_strength: float,
) -> tuple[np.ndarray, float, float, float, float, np.ndarray, float]:
    trial = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = initial_stress[i]
        for j in range(4):
            value += D4[i, j] * (strain[j] - plastic_strain[j])
        trial[i] = value

    mean = (trial[0] + trial[1] + trial[2]) / 3.0
    dev0 = trial[0] - mean
    dev1 = trial[1] - mean
    dev2 = trial[2] - mean
    dev3 = trial[3]
    q = math.sqrt(max(1.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + 3.0 * dev3 * dev3, 0.0))
    p = -mean
    yield_trial = q - alpha * p - cohesion_term - hardening * kappa
    tol_ref = max(q, abs(alpha * p) + cohesion_term, 1.0)
    tol = max(1.0e-10, 1.0e-10 * tol_ref)
    if yield_trial <= tol:
        corrected, clipped, excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
        p_new, q_new = _j2dp_pq_numba(corrected)
        plastic_flag = 1.0 if clipped else 0.0
        return corrected, plastic_flag, max(yield_trial, excess), p_new, q_new, plastic_strain.copy(), kappa

    eps = np.finfo(np.float64).eps
    denom = max(3.0 * shear_mu + hardening, eps)
    dgamma = max(yield_trial / denom, 0.0)
    kappa_new = kappa + dgamma
    q_limit = max(alpha * p + cohesion_term + hardening * kappa_new, 0.0)
    scale = 0.0 if q <= eps else min(q_limit / q, 1.0)
    corrected_pre = np.empty(4, dtype=np.float64)
    corrected_pre[0] = mean + scale * dev0
    corrected_pre[1] = mean + scale * dev1
    corrected_pre[2] = mean + scale * dev2
    corrected_pre[3] = scale * dev3
    corrected, _clipped, excess = _tension_cutoff_plane_strain_numba(corrected_pre, tensile_strength)
    elastic_strain_new = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = 0.0
        for j in range(4):
            value += S4[i, j] * (corrected[j] - initial_stress[j])
        elastic_strain_new[i] = value
    plastic_strain_new = np.empty(4, dtype=np.float64)
    for i in range(4):
        plastic_strain_new[i] = strain[i] - elastic_strain_new[i]
    p_new, q_new = _j2dp_pq_numba(corrected)
    return corrected, 1.0, max(yield_trial, excess), p_new, q_new, plastic_strain_new, kappa_new


@njit(cache=True)
def _j2dp_stress_tangent_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    trial = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = initial_stress[i]
        for j in range(4):
            value += D4[i, j] * (strain[j] - plastic_strain[j])
        trial[i] = value

    mean = (trial[0] + trial[1] + trial[2]) / 3.0
    dev = np.empty(4, dtype=np.float64)
    dev[0] = trial[0] - mean
    dev[1] = trial[1] - mean
    dev[2] = trial[2] - mean
    dev[3] = trial[3]
    j2 = 0.5 * (dev[0] * dev[0] + dev[1] * dev[1] + dev[2] * dev[2]) + dev[3] * dev[3]
    q = math.sqrt(max(3.0 * j2, 0.0))
    p = -mean
    yield_value = q - alpha * p - cohesion_term - hardening * kappa
    tol_ref = max(q, abs(alpha * p) + cohesion_term, 1.0)
    if yield_value <= max(1.0e-10, 1.0e-10 * tol_ref):
        return trial, D4.copy()

    mean_projector = np.zeros((4, 4), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            mean_projector[i, j] = 1.0 / 3.0
    deviatoric_projector = np.eye(4, dtype=np.float64) - mean_projector

    eps = np.finfo(np.float64).eps
    if q <= eps:
        return trial, mean_projector @ D4

    denom = max(3.0 * shear_mu + hardening, eps)
    eta = hardening / denom
    beta = 1.0 - eta
    kappa_new = kappa + max(yield_value / denom, 0.0)
    q_limit = max(alpha * p + cohesion_term + hardening * kappa_new, 0.0)
    scale = min(q_limit / q, 1.0)
    stress = np.empty(4, dtype=np.float64)
    stress[0] = mean + scale * dev[0]
    stress[1] = mean + scale * dev[1]
    stress[2] = mean + scale * dev[2]
    stress[3] = scale * dev[3]

    dscale = np.zeros(4, dtype=np.float64)
    if q_limit <= 0.0:
        scale = 0.0
        stress[0] = mean
        stress[1] = mean
        stress[2] = mean
        stress[3] = 0.0
    else:
        mean_grad = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=np.float64)
        dq = np.array([1.5 * dev[0] / q, 1.5 * dev[1] / q, 1.5 * dev[2] / q, 3.0 * dev[3] / q], dtype=np.float64)
        for i in range(4):
            dscale[i] = (-alpha * beta * mean_grad[i] + (eta - scale) * dq[i]) / q

    jac_trial = mean_projector + scale * deviatoric_projector + np.outer(dev, dscale)
    tangent = jac_trial @ D4
    return stress, tangent


@njit(cache=True)
def _j2dp_maybe_tension_stress_tangent_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    if use_tension_cutoff > 0.5:
        stress = _j2dp_tension_cutoff_stress_numba(
            strain,
            plastic_strain,
            D4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
            kappa,
            tensile_strength,
        )
        tangent = _j2dp_tension_cutoff_numerical_tangent_numba(
            strain,
            plastic_strain,
            D4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
            kappa,
            tensile_strength,
        )
        return stress, tangent
    return _j2dp_stress_tangent_numba(
        strain,
        plastic_strain,
        kappa,
        D4,
        initial_stress,
        alpha,
        cohesion_term,
        hardening,
        shear_mu,
    )


@njit(cache=True)
def _quad4_j2dp_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 27), dtype=np.float64)
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
        stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new = _j2dp_post_update_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
        )
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
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
        data[gp, 18] = p
        data[gp, 19] = q
        data[gp, 20] = plastic
        data[gp, 21] = yield_value
        data[gp, 22] = kappa_new
        for i in range(4):
            data[gp, 23 + i] = plastic_strain_new[i]
    return data, min_det


@njit(cache=True)
def _quad4_j2dp_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    Pvol: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 27), dtype=np.float64)
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
        stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new = _j2dp_post_update_numba(
            strain_eff,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
        )
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
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
        data[gp, 18] = p
        data[gp, 19] = q
        data[gp, 20] = plastic
        data[gp, 21] = yield_value
        data[gp, 22] = kappa_new
        for i in range(4):
            data[gp, 23 + i] = plastic_strain_new[i]
    return data, min_det


@njit(cache=True)
def _quad8_j2dp_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((9, 27), dtype=np.float64)
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
        if use_tension_cutoff > 0.5:
            stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new = _j2dp_tension_cutoff_post_update_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                S4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
                tensile_strength,
            )
        else:
            stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new = _j2dp_post_update_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                S4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
            )
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
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
        data[gp, 18] = p
        data[gp, 19] = q
        data[gp, 20] = plastic
        data[gp, 21] = yield_value
        data[gp, 22] = kappa_new
        for i in range(4):
            data[gp, 23 + i] = plastic_strain_new[i]
    return data, min_det


@njit(cache=True)
def _quad8_j2dp_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    Pvol: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((9, 27), dtype=np.float64)
    strains = np.zeros((9, 4), dtype=np.float64)
    xs = np.zeros(9, dtype=np.float64)
    ys = np.zeros(9, dtype=np.float64)
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
            return data, min_det
        x = 0.0
        y = 0.0
        for i in range(8):
            x += N[i] * coords[i, 0]
            y += N[i] * coords[i, 1]
        xs[gp] = x
        ys[gp] = y
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
    if volume <= np.finfo(np.float64).eps:
        return data, min_det

    epsv_bar = np.empty(4, dtype=np.float64)
    for i in range(4):
        epsv_bar[i] = epsv_acc[i] / volume
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        strain_eff = np.empty(4, dtype=np.float64)
        for i in range(4):
            volumetric = 0.0
            for j in range(4):
                volumetric += Pvol[i, j] * strains[gp, j]
            strain_eff[i] = strains[gp, i] - volumetric + epsv_bar[i]
        if use_tension_cutoff > 0.5:
            stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new = _j2dp_tension_cutoff_post_update_numba(
                strain_eff,
                plastic_strains[gp],
                kappas[gp],
                D4,
                S4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
                tensile_strength,
            )
        else:
            stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new = _j2dp_post_update_numba(
                strain_eff,
                plastic_strains[gp],
                kappas[gp],
                D4,
                S4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
            )
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
        data[gp, 0] = xi
        data[gp, 1] = eta
        data[gp, 2] = weight
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
        data[gp, 18] = p
        data[gp, 19] = q
        data[gp, 20] = plastic
        data[gp, 21] = yield_value
        data[gp, 22] = kappa_new
        for i in range(4):
            data[gp, 23 + i] = plastic_strain_new[i]
    return data, min_det


@njit(cache=True)
def _quad4_j2dp_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    ke = np.zeros((8, 8), dtype=np.float64)
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
            return ke, fe, min_det
        strain = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = 0.0
            for j in range(8):
                value += B[i, j] * ue[j]
            strain[i] = value
        stress, tangent = _j2dp_stress_tangent_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
        )
        dV = det * thickness
        _quad4_add_btcb_numba(ke, B, tangent, dV)
        for i in range(8):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    if abs(alpha) <= 1.0e-14:
        _quad4_symmetrize_numba(ke)
    return ke, fe, min_det


@njit(cache=True)
def _quad8_j2dp_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
    mode_code: int,
    tensile_strength: float,
    use_tension_cutoff: float,
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
            stress, tangent = _j2dp_maybe_tension_stress_tangent_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
                tensile_strength,
                use_tension_cutoff,
            )
            dV = det * weight * thickness
            _quad8_add_btcb_numba(ke, B, tangent, dV)
            _quad8_add_btstress_numba(fe, B, stress, dV)
        if abs(alpha) <= 1.0e-14:
            _quad8_symmetrize_numba(ke)
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
            stress, tangent = _j2dp_maybe_tension_stress_tangent_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
                tensile_strength,
                use_tension_cutoff,
            )
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
            state_index = 9 + gp
            stress, tangent = _j2dp_maybe_tension_stress_tangent_numba(
                strain,
                plastic_strains[state_index],
                kappas[state_index],
                D4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
                tensile_strength,
                use_tension_cutoff,
            )
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
        stress, tangent = _j2dp_maybe_tension_stress_tangent_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
            tensile_strength,
            use_tension_cutoff,
        )
        dV = dV_cache[gp]
        _quad8_add_btcb_numba(ke, B_eff, tangent, dV)
        _quad8_add_btstress_numba(fe, B_eff, stress, dV)
    if abs(alpha) <= 1.0e-14:
        _quad8_symmetrize_numba(ke)
    return ke, fe, min_det, volume


@njit(cache=True)
def _quad4_axisymmetric_j2dp_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    ke = np.zeros((8, 8), dtype=np.float64)
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
            return ke, fe, min_det, min_radius
        strain = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = 0.0
            for j in range(8):
                value += B[i, j] * ue[j]
            strain[i] = value
        stress, tangent = _j2dp_stress_tangent_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
        )
        dV = det * thickness * 2.0 * math.pi * radius
        _quad4_add_btcb_numba(ke, B, tangent, dV)
        for i in range(8):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    if abs(alpha) <= 1.0e-14:
        _quad4_symmetrize_numba(ke)
    return ke, fe, min_det, min_radius


@njit(cache=True)
def _quad8_axisymmetric_j2dp_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
    mode_code: int,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    ke = np.zeros((16, 16), dtype=np.float64)
    fe = np.zeros(16, dtype=np.float64)
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
                return ke, fe, min_det, min_radius, volume
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            stress, tangent = _j2dp_stress_tangent_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
            )
            dV = det * weight * thickness * 2.0 * math.pi * radius
            _quad8_add_btcb_numba(ke, B, tangent, dV)
            _quad8_add_btstress_numba(fe, B, stress, dV)
        if abs(alpha) <= 1.0e-14:
            _quad8_symmetrize_numba(ke)
        return ke, fe, min_det, min_radius, volume

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if radius < min_radius:
                min_radius = radius
            if det <= 0.0 or radius <= 0.0:
                return ke, fe, min_det, min_radius, volume
            Bdev = _quad8_project_b_numba(Idev, B)
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            stress, tangent = _j2dp_stress_tangent_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
            )
            dV = det * weight * thickness * 2.0 * math.pi * radius
            _quad8_add_btlcbr_numba(ke, Bdev, tangent, B, dV)
            _quad8_add_btstress_numba(fe, Bdev, stress, dV)
        for gp in range(4):
            xi, eta, weight = _quad8_gp_reduced(gp)
            B, _N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if radius < min_radius:
                min_radius = radius
            if det <= 0.0 or radius <= 0.0:
                return ke, fe, min_det, min_radius, volume
            Bv = _quad8_project_b_numba(Pvol, B)
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            state_index = 9 + gp
            stress, tangent = _j2dp_stress_tangent_numba(
                strain,
                plastic_strains[state_index],
                kappas[state_index],
                D4,
                initial_stress,
                alpha,
                cohesion_term,
                hardening,
                shear_mu,
            )
            dV = det * weight * thickness * 2.0 * math.pi * radius
            _quad8_add_btlcbr_numba(ke, Bv, tangent, B, dV)
            _quad8_add_btstress_numba(fe, Bv, stress, dV)
        return ke, fe, min_det, min_radius, volume

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
            return ke, fe, min_det, min_radius, volume
        dV = det * weight * thickness * 2.0 * math.pi * radius
        dV_cache[gp] = dV
        volume += dV
        Bv = _quad8_project_b_numba(Pvol, B)
        for r in range(4):
            for c in range(16):
                B_cache[gp, r, c] = B[r, c]
                Bv_acc[r, c] += Bv[r, c] * dV
    if volume <= 0.0:
        return ke, fe, min_det, min_radius, volume
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
        stress, tangent = _j2dp_stress_tangent_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
        )
        dV = dV_cache[gp]
        _quad8_add_btcb_numba(ke, B_eff, tangent, dV)
        _quad8_add_btstress_numba(fe, B_eff, stress, dV)
    if abs(alpha) <= 1.0e-14:
        _quad8_symmetrize_numba(ke)
    return ke, fe, min_det, min_radius, volume


@njit(cache=True)
def _quad8_axisymmetric_j2dp_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
) -> tuple[np.ndarray, float, float]:
    data = np.zeros((9, 28), dtype=np.float64)
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
            return data, min_det, min_radius
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
        stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new = _j2dp_post_update_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
        )
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
        data[gp, 0] = xi
        data[gp, 1] = eta
        data[gp, 2] = weight
        data[gp, 3] = x
        data[gp, 4] = y
        data[gp, 5] = det * weight * thickness * 2.0 * math.pi * radius
        for i in range(4):
            data[gp, 6 + i] = strain[i]
            data[gp, 10 + i] = stress[i]
        data[gp, 14] = s1
        data[gp, 15] = s2
        data[gp, 16] = s3
        data[gp, 17] = 0.5 * (s1 - s3)
        data[gp, 18] = p
        data[gp, 19] = q
        data[gp, 20] = plastic
        data[gp, 21] = yield_value
        data[gp, 22] = kappa_new
        for i in range(4):
            data[gp, 23 + i] = plastic_strain_new[i]
        data[gp, 27] = radius
    return data, min_det, min_radius


@njit(cache=True)
def _quad8_axisymmetric_j2dp_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    Pvol: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    thickness: float,
) -> tuple[np.ndarray, float, float, float]:
    data = np.zeros((9, 28), dtype=np.float64)
    strains = np.zeros((9, 4), dtype=np.float64)
    xs = np.zeros(9, dtype=np.float64)
    ys = np.zeros(9, dtype=np.float64)
    radii = np.zeros(9, dtype=np.float64)
    dvols = np.zeros(9, dtype=np.float64)
    epsv_acc = np.zeros(4, dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    volume = 0.0
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        B, N, det, radius = _quad8_axisymmetric_b_matrix_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if radius < min_radius:
            min_radius = radius
        if det <= 0.0 or radius <= 0.0:
            return data, min_det, min_radius, volume
        x = 0.0
        y = 0.0
        for i in range(8):
            x += N[i] * coords[i, 0]
            y += N[i] * coords[i, 1]
        xs[gp] = x
        ys[gp] = y
        radii[gp] = radius
        dV = det * weight * thickness * 2.0 * math.pi * radius
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
    if volume <= np.finfo(np.float64).eps:
        return data, min_det, min_radius, volume

    epsv_bar = np.empty(4, dtype=np.float64)
    for i in range(4):
        epsv_bar[i] = epsv_acc[i] / volume
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        strain_eff = np.empty(4, dtype=np.float64)
        for i in range(4):
            volumetric = 0.0
            for j in range(4):
                volumetric += Pvol[i, j] * strains[gp, j]
            strain_eff[i] = strains[gp, i] - volumetric + epsv_bar[i]
        stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new = _j2dp_post_update_numba(
            strain_eff,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            alpha,
            cohesion_term,
            hardening,
            shear_mu,
        )
        s1, s2, s3 = _quad4_post_principal_values_numba(stress)
        data[gp, 0] = xi
        data[gp, 1] = eta
        data[gp, 2] = weight
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
        data[gp, 18] = p
        data[gp, 19] = q
        data[gp, 20] = plastic
        data[gp, 21] = yield_value
        data[gp, 22] = kappa_new
        for i in range(4):
            data[gp, 23 + i] = plastic_strain_new[i]
        data[gp, 27] = radii[gp]
    return data, min_det, min_radius, volume


def _quad8_j2dp_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    mode: str,
    *,
    initial_stress: np.ndarray | None = None,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    mode_code = _quad4_mode_code(mode)
    expected_points = 13 if mode_code == _QUAD4_MODE_SRI else 9
    plastic_arr = np.ascontiguousarray(plastic_strains, dtype=np.float64)
    kappa_arr = np.ascontiguousarray(kappas, dtype=np.float64)
    if plastic_arr.shape != (expected_points, 4) or kappa_arr.shape != (expected_points,):
        raise FEM2DError(f"QUAD8 {mode}: plastic state arrays must have {expected_points} integration points")
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    ke, fe, min_det, volume = _quad8_j2dp_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        plastic_arr,
        kappa_arr,
        Pvol,
        Idev,
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
        mode_code,
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return ke, fe


def _quad4_j2dp_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (4, 4) or state_kappas.shape != (4,):
        raise FEM2DError("QUAD4 plastic state arrays must have shapes (4, 4) and (4,)")
    data, min_det = _quad4_j2dp_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data


def _quad4_j2dp_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (4, 4) or state_kappas.shape != (4,):
        raise FEM2DError("QUAD4 plastic state arrays must have shapes (4, 4) and (4,)")
    data, min_det = _quad4_j2dp_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data


def _quad8_j2dp_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (9, 4) or state_kappas.shape != (9,):
        raise FEM2DError("QUAD8 plastic state arrays must have shapes (9, 4) and (9,)")
    data, min_det = _quad8_j2dp_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return data


def _quad8_j2dp_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (9, 4) or state_kappas.shape != (9,):
        raise FEM2DError("QUAD8 plastic state arrays must have shapes (9, 4) and (9,)")
    data, min_det = _quad8_j2dp_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return data


def _quad4_j2dp_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (4, 4) or state_kappas.shape != (4,):
        raise FEM2DError("QUAD4 plastic state arrays must have shapes (4, 4) and (4,)")
    ke, fe, min_det = _quad4_j2dp_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return ke, fe


def _quad4_axisymmetric_j2dp_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (4, 4) or state_kappas.shape != (4,):
        raise FEM2DError("axisymmetric QUAD4 plastic state arrays must have shapes (4, 4) and (4,)")
    ke, fe, min_det, min_radius = _quad4_axisymmetric_j2dp_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD4: axisymmetric radius must be positive, got {min_radius:.6e}")
    return ke, fe


def _quad8_axisymmetric_j2dp_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    mode: str,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    mode_code = _quad4_mode_code(mode)
    expected_points = 13 if mode_code == _QUAD4_MODE_SRI else 9
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (expected_points, 4) or state_kappas.shape != (expected_points,):
        raise FEM2DError(f"axisymmetric QUAD8 {mode}: plastic state arrays must have {expected_points} integration points")
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    ke, fe, min_det, min_radius, volume = _quad8_axisymmetric_j2dp_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        Pvol,
        Idev,
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
        mode_code,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD8: axisymmetric radius must be positive, got {min_radius:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive axisymmetric element measure")
    return ke, fe


def _quad8_axisymmetric_j2dp_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (9, 4) or state_kappas.shape != (9,):
        raise FEM2DError("axisymmetric QUAD8 plastic state arrays must have shapes (9, 4) and (9,)")
    data, min_det, min_radius = _quad8_axisymmetric_j2dp_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD8: axisymmetric radius must be positive, got {min_radius:.6e}")
    return data


def _quad8_axisymmetric_j2dp_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (9, 4) or state_kappas.shape != (9,):
        raise FEM2DError("axisymmetric QUAD8 plastic state arrays must have shapes (9, 4) and (9,)")
    data, min_det, min_radius, volume = _quad8_axisymmetric_j2dp_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        float(alpha),
        float(cohesion_term),
        float(material.hardening),
        float(material.shear_mu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD8: axisymmetric radius must be positive, got {min_radius:.6e}")
    if volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive axisymmetric element measure")
    return data


__all__ = list(J2DP_ELEMENT_KERNEL_FUNCTIONS)
