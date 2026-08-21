"""Advanced elastic post-processing kernels for plane-strain QUAD elements."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fem2d_element_elastic_post import _quad4_post_principal_values_numba
from .fem2d_element_fast_paths import (
    _quad4_advanced_state_arrays,
    _quad8_advanced_state_arrays,
)
from .fem2d_element_numba_primitives import (
    _quad4_b_det_numba,
    _quad8_b_det_numba,
    _quad8_gp_full,
    _quad8_shape_grad_numba,
)
from .fem2d_materials import (
    _advanced_history_state_numba,
    _advanced_model_id,
    _advanced_params_array,
    _tension_cutoff_plane_strain_numba,
)
from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, njit

ADVANCED_ELASTIC_POST_FUNCTIONS = (
    "advanced_elastic_post_contract",
    "_quad4_advanced_elastic_stress_numba",
    "_quad4_fill_advanced_post_row_numba",
    "_quad4_advanced_elastic_post_numba",
    "_quad4_advanced_elastic_bbar_post_numba",
    "_quad4_fill_advanced_tension_post_row_numba",
    "_quad4_advanced_elastic_tension_post_numba",
    "_quad4_advanced_elastic_tension_bbar_post_numba",
    "_quad8_fill_advanced_post_row_numba",
    "_quad8_fill_advanced_tension_post_row_numba",
    "_quad8_advanced_elastic_post_numba",
    "_quad8_advanced_elastic_bbar_post_numba",
    "_quad8_advanced_elastic_tension_post_numba",
    "_quad8_advanced_elastic_tension_bbar_post_numba",
    "_quad4_advanced_elastic_post_fast",
    "_quad4_advanced_elastic_bbar_post_fast",
    "_quad4_advanced_elastic_tension_post_fast",
    "_quad4_advanced_elastic_tension_bbar_post_fast",
    "_quad8_advanced_elastic_post_fast",
    "_quad8_advanced_elastic_bbar_post_fast",
    "_quad8_advanced_elastic_tension_post_fast",
    "_quad8_advanced_elastic_tension_bbar_post_fast",
)


def advanced_elastic_post_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_advanced_elastic_post.v1",
        "module": "geofem_app.fem2d_element_advanced_elastic_post",
        "function_count": len(ADVANCED_ELASTIC_POST_FUNCTIONS),
        "functions": list(ADVANCED_ELASTIC_POST_FUNCTIONS),
        "covered_surfaces": [
            "quad4_advanced_elastic_post",
            "quad8_advanced_elastic_post",
            "quad4_advanced_elastic_tension_post",
            "quad8_advanced_elastic_tension_post",
            "bbar_post_processing",
        ],
    }


@njit(cache=True)
def _quad4_advanced_elastic_stress_numba(strain: np.ndarray, initial_stress: np.ndarray, effective_e: float, nu: float) -> np.ndarray:
    lam = effective_e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = effective_e / (2.0 * (1.0 + nu))
    stress = np.empty(4, dtype=np.float64)
    normal_sum = strain[0] + strain[1] + strain[2]
    stress[0] = initial_stress[0] + lam * normal_sum + 2.0 * mu * strain[0]
    stress[1] = initial_stress[1] + lam * normal_sum + 2.0 * mu * strain[1]
    stress[2] = initial_stress[2] + lam * normal_sum + 2.0 * mu * strain[2]
    stress[3] = initial_stress[3] + mu * strain[3]
    return stress

@njit(cache=True)
def _quad4_fill_advanced_post_row_numba(
    data: np.ndarray,
    gp: int,
    xi: float,
    eta: float,
    x: float,
    y: float,
    dV: float,
    strain: np.ndarray,
    stress: np.ndarray,
    history: np.ndarray,
    kappa: float,
) -> None:
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
    data[gp, 5] = dV
    for i in range(4):
        data[gp, 6 + i] = strain[i]
        data[gp, 10 + i] = stress[i]
    data[gp, 14] = s1
    data[gp, 15] = s2
    data[gp, 16] = s3
    data[gp, 17] = 0.5 * (s1 - s3)
    data[gp, 18] = -mean_stress
    data[gp, 19] = q
    for i in range(history.shape[0]):
        data[gp, 20 + i] = history[i]
    data[gp, 35] = kappa

@njit(cache=True)
def _quad4_advanced_elastic_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    previous_kappas: np.ndarray,
    params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 36), dtype=np.float64)
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
        history = _advanced_history_state_numba(strain, previous_states[gp], params, model_id)
        stress = _quad4_advanced_elastic_stress_numba(strain, initial_stress, history[12], nu)
        kappa = max(previous_kappas[gp], history[0])
        _quad4_fill_advanced_post_row_numba(data, gp, xi, eta, x, y, det * thickness, strain, stress, history, kappa)
    return data, min_det

@njit(cache=True)
def _quad4_advanced_elastic_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    previous_states: np.ndarray,
    previous_kappas: np.ndarray,
    params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 36), dtype=np.float64)
    strains = np.zeros((4, 4), dtype=np.float64)
    xs = np.zeros(4, dtype=np.float64)
    ys = np.zeros(4, dtype=np.float64)
    dvols = np.zeros(4, dtype=np.float64)
    epsv_acc = np.zeros(4, dtype=np.float64)
    volume = 0.0
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
        history = _advanced_history_state_numba(strain_eff, previous_states[gp], params, model_id)
        stress = _quad4_advanced_elastic_stress_numba(strain_eff, initial_stress, history[12], nu)
        kappa = max(previous_kappas[gp], history[0])
        _quad4_fill_advanced_post_row_numba(data, gp, xi, eta, xs[gp], ys[gp], dvols[gp], strain_eff, stress, history, kappa)
    return data, min_det

@njit(cache=True)
def _quad4_fill_advanced_tension_post_row_numba(
    data: np.ndarray,
    gp: int,
    xi: float,
    eta: float,
    x: float,
    y: float,
    dV: float,
    strain: np.ndarray,
    stress: np.ndarray,
    history: np.ndarray,
    kappa: float,
    plastic: float,
    yield_value: float,
) -> None:
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
    data[gp, 5] = dV
    for i in range(4):
        data[gp, 6 + i] = strain[i]
        data[gp, 10 + i] = stress[i]
    data[gp, 14] = s1
    data[gp, 15] = s2
    data[gp, 16] = s3
    data[gp, 17] = 0.5 * (s1 - s3)
    data[gp, 18] = -mean_stress
    data[gp, 19] = q
    for i in range(history.shape[0]):
        data[gp, 20 + i] = history[i]
    data[gp, 35] = kappa
    data[gp, 36] = plastic
    data[gp, 37] = yield_value

@njit(cache=True)
def _quad4_advanced_elastic_tension_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    previous_kappas: np.ndarray,
    params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    tensile_strength: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 38), dtype=np.float64)
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
        history = _advanced_history_state_numba(strain, previous_states[gp], params, model_id)
        stress = _quad4_advanced_elastic_stress_numba(strain, initial_stress, history[12], nu)
        stress, clipped, excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
        kappa = max(previous_kappas[gp], history[0])
        _quad4_fill_advanced_tension_post_row_numba(
            data,
            gp,
            xi,
            eta,
            x,
            y,
            det * thickness,
            strain,
            stress,
            history,
            kappa,
            1.0 if clipped else 0.0,
            max(excess, 0.0),
        )
    return data, min_det

@njit(cache=True)
def _quad4_advanced_elastic_tension_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    previous_states: np.ndarray,
    previous_kappas: np.ndarray,
    params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    tensile_strength: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 38), dtype=np.float64)
    strains = np.zeros((4, 4), dtype=np.float64)
    xs = np.zeros(4, dtype=np.float64)
    ys = np.zeros(4, dtype=np.float64)
    dvols = np.zeros(4, dtype=np.float64)
    epsv_acc = np.zeros(4, dtype=np.float64)
    volume = 0.0
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
        history = _advanced_history_state_numba(strain_eff, previous_states[gp], params, model_id)
        stress = _quad4_advanced_elastic_stress_numba(strain_eff, initial_stress, history[12], nu)
        stress, clipped, excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
        kappa = max(previous_kappas[gp], history[0])
        _quad4_fill_advanced_tension_post_row_numba(
            data,
            gp,
            xi,
            eta,
            xs[gp],
            ys[gp],
            dvols[gp],
            strain_eff,
            stress,
            history,
            kappa,
            1.0 if clipped else 0.0,
            max(excess, 0.0),
        )
    return data, min_det

@njit(cache=True)
def _quad8_fill_advanced_post_row_numba(
    data: np.ndarray,
    gp: int,
    xi: float,
    eta: float,
    weight: float,
    x: float,
    y: float,
    dV: float,
    strain: np.ndarray,
    stress: np.ndarray,
    history: np.ndarray,
    kappa: float,
) -> None:
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
    data[gp, 5] = dV
    for i in range(4):
        data[gp, 6 + i] = strain[i]
        data[gp, 10 + i] = stress[i]
    data[gp, 14] = s1
    data[gp, 15] = s2
    data[gp, 16] = s3
    data[gp, 17] = 0.5 * (s1 - s3)
    data[gp, 18] = -mean_stress
    data[gp, 19] = q
    for i in range(history.shape[0]):
        data[gp, 20 + i] = history[i]
    data[gp, 35] = kappa

@njit(cache=True)
def _quad8_fill_advanced_tension_post_row_numba(
    data: np.ndarray,
    gp: int,
    xi: float,
    eta: float,
    weight: float,
    x: float,
    y: float,
    dV: float,
    strain: np.ndarray,
    stress: np.ndarray,
    history: np.ndarray,
    kappa: float,
    plastic: float,
    yield_value: float,
) -> None:
    _quad8_fill_advanced_post_row_numba(data, gp, xi, eta, weight, x, y, dV, strain, stress, history, kappa)
    data[gp, 36] = plastic
    data[gp, 37] = yield_value

@njit(cache=True)
def _quad8_advanced_elastic_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    previous_kappas: np.ndarray,
    params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((9, 36), dtype=np.float64)
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
        history = _advanced_history_state_numba(strain, previous_states[gp], params, model_id)
        stress = _quad4_advanced_elastic_stress_numba(strain, initial_stress, history[12], nu)
        kappa = max(previous_kappas[gp], history[0])
        _quad8_fill_advanced_post_row_numba(data, gp, xi, eta, weight, x, y, det * weight * thickness, strain, stress, history, kappa)
    return data, min_det

@njit(cache=True)
def _quad8_advanced_elastic_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    previous_states: np.ndarray,
    previous_kappas: np.ndarray,
    params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
) -> tuple[np.ndarray, float, float]:
    data = np.zeros((9, 36), dtype=np.float64)
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
    if volume <= np.finfo(np.float64).eps:
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
        history = _advanced_history_state_numba(strain_eff, previous_states[gp], params, model_id)
        stress = _quad4_advanced_elastic_stress_numba(strain_eff, initial_stress, history[12], nu)
        kappa = max(previous_kappas[gp], history[0])
        _quad8_fill_advanced_post_row_numba(data, gp, xi, eta, weights[gp], xs[gp], ys[gp], dvols[gp], strain_eff, stress, history, kappa)
    return data, min_det, volume

@njit(cache=True)
def _quad8_advanced_elastic_tension_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    previous_kappas: np.ndarray,
    params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    tensile_strength: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((9, 38), dtype=np.float64)
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
        history = _advanced_history_state_numba(strain, previous_states[gp], params, model_id)
        stress = _quad4_advanced_elastic_stress_numba(strain, initial_stress, history[12], nu)
        stress, clipped, excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
        kappa = max(previous_kappas[gp], history[0])
        _quad8_fill_advanced_tension_post_row_numba(
            data,
            gp,
            xi,
            eta,
            weight,
            x,
            y,
            det * weight * thickness,
            strain,
            stress,
            history,
            kappa,
            1.0 if clipped else 0.0,
            max(excess, 0.0),
        )
    return data, min_det

@njit(cache=True)
def _quad8_advanced_elastic_tension_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    previous_states: np.ndarray,
    previous_kappas: np.ndarray,
    params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    tensile_strength: float,
) -> tuple[np.ndarray, float, float]:
    data = np.zeros((9, 38), dtype=np.float64)
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
    if volume <= np.finfo(np.float64).eps:
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
        history = _advanced_history_state_numba(strain_eff, previous_states[gp], params, model_id)
        stress = _quad4_advanced_elastic_stress_numba(strain_eff, initial_stress, history[12], nu)
        stress, clipped, excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
        kappa = max(previous_kappas[gp], history[0])
        _quad8_fill_advanced_tension_post_row_numba(
            data,
            gp,
            xi,
            eta,
            weights[gp],
            xs[gp],
            ys[gp],
            dvols[gp],
            strain_eff,
            stress,
            history,
            kappa,
            1.0 if clipped else 0.0,
            max(excess, 0.0),
        )
    return data, min_det, volume

def _quad4_advanced_elastic_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, previous_kappas = _quad4_advanced_state_arrays(element_id, plastic_state, params)
    data, min_det = _quad4_advanced_elastic_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(previous_kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data

def _quad4_advanced_elastic_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, previous_kappas = _quad4_advanced_state_arrays(element_id, plastic_state, params)
    data, min_det = _quad4_advanced_elastic_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(previous_kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data

def _quad4_advanced_elastic_tension_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, previous_kappas = _quad4_advanced_state_arrays(element_id, plastic_state, params)
    data, min_det = _quad4_advanced_elastic_tension_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(previous_kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(material.tensile_strength),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data

def _quad4_advanced_elastic_tension_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, previous_kappas = _quad4_advanced_state_arrays(element_id, plastic_state, params)
    data, min_det = _quad4_advanced_elastic_tension_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(previous_kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(material.tensile_strength),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data

def _quad8_advanced_elastic_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params)
    data, min_det = _quad8_advanced_elastic_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(previous_kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return data

def _quad8_advanced_elastic_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params)
    data, min_det, volume = _quad8_advanced_elastic_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(previous_kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return data

def _quad8_advanced_elastic_tension_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    if not math.isfinite(float(material.tensile_strength)):
        raise FEM2DError("QUAD8 advanced tension cutoff requires finite tensile strength")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params)
    data, min_det = _quad8_advanced_elastic_tension_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(previous_kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(material.tensile_strength),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return data

def _quad8_advanced_elastic_tension_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    if not math.isfinite(float(material.tensile_strength)):
        raise FEM2DError("QUAD8 advanced tension cutoff requires finite tensile strength")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params)
    data, min_det, volume = _quad8_advanced_elastic_tension_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(previous_kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(material.tensile_strength),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return data

__all__ = list(ADVANCED_ELASTIC_POST_FUNCTIONS)
