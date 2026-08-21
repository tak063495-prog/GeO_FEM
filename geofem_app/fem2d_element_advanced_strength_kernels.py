"""Advanced material strength kernels for plane-strain QUAD elements."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .fem2d_element_elastic_post import _quad4_post_principal_values_numba
from .fem2d_element_fast_paths import (
    _quad4_advanced_state_arrays,
    _quad4_post_state_arrays,
    _quad8_advanced_state_arrays,
    _quad8_post_state_arrays,
)
from .fem2d_element_j2dp_kernels import (
    _j2dp_maybe_tension_stress_tangent_numba,
    _j2dp_post_update_numba,
)
from .fem2d_element_mohr_coulomb_kernels import (
    _quad4_mc_maybe_tension_post_update_numba,
    _quad4_mc_maybe_tension_stress_tangent_numba,
    _quad4_mc_plane_coeffs_numba,
    _quad4_mc_post_update_numba,
)
from .fem2d_element_numba_primitives import (
    _quad4_add_btcb_numba,
    _quad4_b_det_numba,
    _quad8_add_btcb_numba,
    _quad8_add_btlcbr_numba,
    _quad8_add_btstress_numba,
    _quad8_b_det_numba,
    _quad8_gp_full,
    _quad8_gp_reduced,
    _quad8_project_b_numba,
    _quad8_shape_grad_numba,
)
from .fem2d_materials import (
    _ADV_MODEL_BILINEAR_LIQUEFACTION,
    _ADV_MODEL_LIQUEFACTION,
    _ADV_MODEL_PZ_CLAY,
    _ADV_MODEL_PZ_SAND,
    _ADV_MODEL_UW_CLAY,
    _ADV_PARAM_GAMMA_REF,
    _ADV_STATE_DILATANCY,
    _ADV_STATE_EFFECTIVE_E,
    _ADV_STATE_HARDENING_VARIABLE,
    _ADV_STATE_RU,
    _advanced_history_state_numba,
    _advanced_model_id,
    _advanced_params_array,
    _advanced_strength_model_name,
    _angle_radians,
    _param_float,
    _tension_cutoff_plane_strain_numba,
)
from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, njit

_QUAD4_MODE_FULL = 0
_QUAD4_MODE_SRI = 1
_QUAD4_MODE_BBAR = 2

_ADV_STRENGTH_COHESION = 0
_ADV_STRENGTH_YIELD_STRESS = 1
_ADV_STRENGTH_FRICTION = 2
_ADV_STRENGTH_HARDENING = 3
_ADV_STRENGTH_HARDENING_RATE = 4
_ADV_STRENGTH_UW_SU = 5
_ADV_STRENGTH_PZ_CLAY_SU = 6
_ADV_STRENGTH_PZ_CLAY_PHI = 7
_ADV_STRENGTH_PZ_SAND_PHI = 8
_ADV_STRENGTH_PZ_SAND_COHESION = 9
_ADV_STRENGTH_RESIDUAL_RATIO = 10
_ADV_STRENGTH_DILATANCY_BASE = 11
_ADV_STRENGTH_DILATANCY_PEAK = 12
_ADV_STRENGTH_DILATANCY_RESIDUAL = 13
_ADV_STRENGTH_MODEL = 14
_ADV_STRENGTH_COUNT = 15
_ADV_STRENGTH_MODEL_AUTO = 0.0
_ADV_STRENGTH_MODEL_MC = 1.0

ADVANCED_STRENGTH_ELEMENT_KERNEL_FUNCTIONS = (
    "advanced_strength_element_kernel_contract",
    "_angle_radians_numba",
    "_quad4_effective_d4_s4_numba",
    "_advanced_strength_dilatancy_numba",
    "_advanced_strength_yield_params_numba",
    "_quad4_fill_advanced_strength_post_row_numba",
    "_quad4_advanced_strength_j2dp_post_numba",
    "_quad4_advanced_strength_j2dp_bbar_post_numba",
    "_quad8_advanced_strength_j2dp_post_numba",
    "_quad8_advanced_strength_j2dp_bbar_post_numba",
    "_advanced_strength_j2dp_stress_numba",
    "_advanced_strength_j2dp_stress_tangent_numba",
    "_quad4_advanced_strength_j2dp_tangent_force_numba",
    "_quad8_advanced_strength_j2dp_tangent_force_numba",
    "_advanced_strength_mc_params_numba",
    "_advanced_strength_mc_stress_numba",
    "_advanced_strength_mc_post_update_numba",
    "_advanced_strength_mc_stress_tangent_numba",
    "_quad4_advanced_strength_mc_tangent_force_numba",
    "_quad8_advanced_strength_mc_post_numba",
    "_quad8_advanced_strength_mc_bbar_post_numba",
    "_quad8_advanced_strength_mc_tangent_force_numba",
    "_advanced_strength_params_array",
    "_quad4_advanced_strength_j2dp_post_fast",
    "_quad4_advanced_strength_j2dp_bbar_post_fast",
    "_quad8_advanced_strength_j2dp_post_fast",
    "_quad8_advanced_strength_j2dp_bbar_post_fast",
    "_quad8_advanced_strength_mc_post_fast",
    "_quad8_advanced_strength_mc_bbar_post_fast",
    "_quad4_advanced_strength_j2dp_tangent_force_fast",
    "_quad8_advanced_strength_j2dp_tangent_force_fast",
    "_quad4_advanced_strength_mc_tangent_force_fast",
    "_quad8_advanced_strength_mc_tangent_force_fast",
)


def _quad4_mode_code(mode: str) -> int:
    if mode == "FULL":
        return _QUAD4_MODE_FULL
    if mode == "SRI":
        return _QUAD4_MODE_SRI
    if mode == "B-BAR":
        return _QUAD4_MODE_BBAR
    raise FEM2DError(f"unsupported integration '{mode}'")


def advanced_strength_element_kernel_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_advanced_strength_kernels.v1",
        "module": "geofem_app.fem2d_element_advanced_strength_kernels",
        "function_count": len(ADVANCED_STRENGTH_ELEMENT_KERNEL_FUNCTIONS),
        "functions": list(ADVANCED_STRENGTH_ELEMENT_KERNEL_FUNCTIONS),
        "covered_surfaces": [
            "quad4_advanced_strength_j2dp_post",
            "quad8_advanced_strength_j2dp_post",
            "quad4_advanced_strength_j2dp_tangent_internal_force",
            "quad8_advanced_strength_j2dp_tangent_internal_force",
            "quad8_advanced_strength_mohr_coulomb_post",
            "quad4_quad8_advanced_strength_mohr_coulomb_tangent_internal_force",
            "full_sri_bbar_dispatch",
        ],
    }


@njit(cache=True)
def _angle_radians_numba(value: float) -> float:
    return math.radians(value) if abs(value) > math.pi / 2.0 else value

@njit(cache=True)
def _quad4_effective_d4_s4_numba(effective_e: float, nu: float) -> tuple[np.ndarray, np.ndarray, float]:
    lam = effective_e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = effective_e / (2.0 * (1.0 + nu))
    D4 = np.zeros((4, 4), dtype=np.float64)
    D4[0, 0] = lam + 2.0 * mu
    D4[0, 1] = lam
    D4[0, 2] = lam
    D4[1, 0] = lam
    D4[1, 1] = lam + 2.0 * mu
    D4[1, 2] = lam
    D4[2, 0] = lam
    D4[2, 1] = lam
    D4[2, 2] = lam + 2.0 * mu
    D4[3, 3] = mu
    S4 = np.zeros((4, 4), dtype=np.float64)
    inv_e = 1.0 / max(effective_e, np.finfo(np.float64).eps)
    S4[0, 0] = inv_e
    S4[0, 1] = -nu * inv_e
    S4[0, 2] = -nu * inv_e
    S4[1, 0] = -nu * inv_e
    S4[1, 1] = inv_e
    S4[1, 2] = -nu * inv_e
    S4[2, 0] = -nu * inv_e
    S4[2, 1] = -nu * inv_e
    S4[2, 2] = inv_e
    S4[3, 3] = 1.0 / max(mu, np.finfo(np.float64).eps)
    return D4, S4, mu

@njit(cache=True)
def _advanced_strength_dilatancy_numba(model_id: int, gamma_eq: float, gamma_ref: float, strength_params: np.ndarray) -> float:
    base = strength_params[_ADV_STRENGTH_DILATANCY_BASE]
    if model_id != _ADV_MODEL_PZ_SAND and model_id != _ADV_MODEL_PZ_CLAY:
        return base
    peak = strength_params[_ADV_STRENGTH_DILATANCY_PEAK]
    residual = strength_params[_ADV_STRENGTH_DILATANCY_RESIDUAL]
    weight = math.exp(-max(gamma_eq / max(gamma_ref, 1.0e-12), 0.0))
    return residual + (peak - residual) * weight

@njit(cache=True)
def _advanced_strength_yield_params_numba(
    model_id: int,
    history: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    strength_factor: float,
) -> tuple[float, float, float, float]:
    hardening_variable = history[_ADV_STATE_HARDENING_VARIABLE]
    cohesion = strength_params[_ADV_STRENGTH_COHESION]
    yield_stress = strength_params[_ADV_STRENGTH_YIELD_STRESS]
    phi = strength_params[_ADV_STRENGTH_FRICTION]
    if model_id == _ADV_MODEL_UW_CLAY:
        cohesion = max(strength_params[_ADV_STRENGTH_UW_SU] + hardening_variable, 0.0)
        phi = 0.0
        yield_stress = 0.0
    elif model_id == _ADV_MODEL_PZ_CLAY:
        cohesion = max(strength_params[_ADV_STRENGTH_PZ_CLAY_SU] + hardening_variable, strength_params[_ADV_STRENGTH_COHESION], 0.0)
        phi = strength_params[_ADV_STRENGTH_PZ_CLAY_PHI]
        yield_stress = 0.0
    elif model_id == _ADV_MODEL_PZ_SAND:
        cohesion = max(strength_params[_ADV_STRENGTH_COHESION] + hardening_variable, strength_params[_ADV_STRENGTH_PZ_SAND_COHESION])
        phi = strength_params[_ADV_STRENGTH_PZ_SAND_PHI]
        yield_stress = 0.0
    elif model_id == _ADV_MODEL_LIQUEFACTION or model_id == _ADV_MODEL_BILINEAR_LIQUEFACTION:
        strength_ratio = max(strength_params[_ADV_STRENGTH_RESIDUAL_RATIO], 1.0 - history[_ADV_STATE_RU])
        if strength_params[_ADV_STRENGTH_YIELD_STRESS] > 0.0:
            yield_stress = max(strength_params[_ADV_STRENGTH_YIELD_STRESS] * strength_ratio, np.finfo(np.float64).eps)
            cohesion = 0.0
        else:
            yield_stress = 0.0
            cohesion = max(strength_params[_ADV_STRENGTH_COHESION] * strength_ratio, 0.0)
            phi = max(strength_params[_ADV_STRENGTH_FRICTION] * strength_ratio, 0.0)

    if yield_stress > 0.0:
        return 0.0, yield_stress / max(strength_factor, np.finfo(np.float64).eps), strength_params[_ADV_STRENGTH_HARDENING], 1.0

    c = cohesion / max(strength_factor, np.finfo(np.float64).eps)
    tan_phi_reduced = math.tan(phi) / max(strength_factor, np.finfo(np.float64).eps)
    phi_reduced = math.atan(tan_phi_reduced)
    sin_phi = math.sin(phi_reduced)
    denom = max(math.sqrt(3.0) * (3.0 - sin_phi), np.finfo(np.float64).eps)
    alpha = 2.0 * sin_phi / denom
    cohesion_term = 6.0 * c * math.cos(phi_reduced) / denom
    if cohesion_term <= 0.0 and strength_params[_ADV_STRENGTH_YIELD_STRESS] > 0.0:
        cohesion_term = strength_params[_ADV_STRENGTH_YIELD_STRESS] / max(strength_factor, np.finfo(np.float64).eps)
    return alpha, cohesion_term, strength_params[_ADV_STRENGTH_HARDENING], 0.0

@njit(cache=True)
def _quad4_fill_advanced_strength_post_row_numba(
    data: np.ndarray,
    gp: int,
    xi: float,
    eta: float,
    x: float,
    y: float,
    dV: float,
    strain: np.ndarray,
    stress: np.ndarray,
    p: float,
    q: float,
    plastic: float,
    yield_value: float,
    kappa: float,
    plastic_strain: np.ndarray,
    history: np.ndarray,
    plastic_multiplier: float,
) -> None:
    s1, s2, s3 = _quad4_post_principal_values_numba(stress)
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
    data[gp, 18] = p
    data[gp, 19] = q
    data[gp, 20] = plastic
    data[gp, 21] = yield_value
    data[gp, 22] = kappa
    for i in range(4):
        data[gp, 23 + i] = plastic_strain[i]
    for i in range(history.shape[0]):
        data[gp, 27 + i] = history[i]
    data[gp, 42] = plastic_multiplier

@njit(cache=True)
def _quad4_advanced_strength_j2dp_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 43), dtype=np.float64)
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
        alpha, cohesion_term, hardening, _strength_model = _advanced_strength_yield_params_numba(model_id, history, params, strength_params, strength_factor)
        D4, S4, shear_mu = _quad4_effective_d4_s4_numba(history[_ADV_STATE_EFFECTIVE_E], nu)
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
        if use_tension_cutoff > 0.5:
            stress, clipped, excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
            if clipped:
                plastic = 1.0
            yield_value = max(yield_value, excess)
            elastic_strain_new = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(4):
                    value += S4[i, j] * (stress[j] - initial_stress[j])
                elastic_strain_new[i] = value
            for i in range(4):
                plastic_strain_new[i] = strain[i] - elastic_strain_new[i]
            mean_new = (stress[0] + stress[1] + stress[2]) / 3.0
            nd0 = stress[0] - mean_new
            nd1 = stress[1] - mean_new
            nd2 = stress[2] - mean_new
            nd3 = stress[3]
            q = math.sqrt(max(1.5 * (nd0 * nd0 + nd1 * nd1 + nd2 * nd2) + 3.0 * nd3 * nd3, 0.0))
            p = -mean_new
        plastic_multiplier = max(kappa_new - kappas[gp], 0.0)
        history[_ADV_STATE_HARDENING_VARIABLE] = history[_ADV_STATE_HARDENING_VARIABLE] + plastic_multiplier * strength_params[_ADV_STRENGTH_HARDENING_RATE]
        history[_ADV_STATE_DILATANCY] = _advanced_strength_dilatancy_numba(model_id, history[0], params[_ADV_PARAM_GAMMA_REF], strength_params)
        _quad4_fill_advanced_strength_post_row_numba(
            data,
            gp,
            xi,
            eta,
            x,
            y,
            det * thickness,
            strain,
            stress,
            p,
            q,
            plastic,
            yield_value,
            kappa_new,
            plastic_strain_new,
            history,
            plastic_multiplier,
        )
    return data, min_det

@njit(cache=True)
def _quad4_advanced_strength_j2dp_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((4, 43), dtype=np.float64)
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
        history = _advanced_history_state_numba(strain_eff, previous_states[gp], params, model_id)
        alpha, cohesion_term, hardening, _strength_model = _advanced_strength_yield_params_numba(model_id, history, params, strength_params, strength_factor)
        D4, S4, shear_mu = _quad4_effective_d4_s4_numba(history[_ADV_STATE_EFFECTIVE_E], nu)
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
        if use_tension_cutoff > 0.5:
            stress, clipped, excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
            if clipped:
                plastic = 1.0
            yield_value = max(yield_value, excess)
            elastic_strain_new = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(4):
                    value += S4[i, j] * (stress[j] - initial_stress[j])
                elastic_strain_new[i] = value
            for i in range(4):
                plastic_strain_new[i] = strain_eff[i] - elastic_strain_new[i]
            mean_new = (stress[0] + stress[1] + stress[2]) / 3.0
            nd0 = stress[0] - mean_new
            nd1 = stress[1] - mean_new
            nd2 = stress[2] - mean_new
            nd3 = stress[3]
            q = math.sqrt(max(1.5 * (nd0 * nd0 + nd1 * nd1 + nd2 * nd2) + 3.0 * nd3 * nd3, 0.0))
            p = -mean_new
        plastic_multiplier = max(kappa_new - kappas[gp], 0.0)
        history[_ADV_STATE_HARDENING_VARIABLE] = history[_ADV_STATE_HARDENING_VARIABLE] + plastic_multiplier * strength_params[_ADV_STRENGTH_HARDENING_RATE]
        history[_ADV_STATE_DILATANCY] = _advanced_strength_dilatancy_numba(model_id, history[0], params[_ADV_PARAM_GAMMA_REF], strength_params)
        _quad4_fill_advanced_strength_post_row_numba(
            data,
            gp,
            xi,
            eta,
            xs[gp],
            ys[gp],
            dvols[gp],
            strain_eff,
            stress,
            p,
            q,
            plastic,
            yield_value,
            kappa_new,
            plastic_strain_new,
            history,
            plastic_multiplier,
        )
    return data, min_det

@njit(cache=True)
def _quad8_advanced_strength_j2dp_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float]:
    data = np.zeros((9, 43), dtype=np.float64)
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
        alpha, cohesion_term, hardening, _strength_model = _advanced_strength_yield_params_numba(model_id, history, params, strength_params, strength_factor)
        D4, S4, shear_mu = _quad4_effective_d4_s4_numba(history[_ADV_STATE_EFFECTIVE_E], nu)
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
        if use_tension_cutoff > 0.5:
            stress, clipped, excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
            if clipped:
                plastic = 1.0
            yield_value = max(yield_value, excess)
            elastic_strain_new = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(4):
                    value += S4[i, j] * (stress[j] - initial_stress[j])
                elastic_strain_new[i] = value
            for i in range(4):
                plastic_strain_new[i] = strain[i] - elastic_strain_new[i]
            mean_new = (stress[0] + stress[1] + stress[2]) / 3.0
            nd0 = stress[0] - mean_new
            nd1 = stress[1] - mean_new
            nd2 = stress[2] - mean_new
            nd3 = stress[3]
            q = math.sqrt(max(1.5 * (nd0 * nd0 + nd1 * nd1 + nd2 * nd2) + 3.0 * nd3 * nd3, 0.0))
            p = -mean_new
        plastic_multiplier = max(kappa_new - kappas[gp], 0.0)
        history[_ADV_STATE_HARDENING_VARIABLE] = history[_ADV_STATE_HARDENING_VARIABLE] + plastic_multiplier * strength_params[_ADV_STRENGTH_HARDENING_RATE]
        history[_ADV_STATE_DILATANCY] = _advanced_strength_dilatancy_numba(model_id, history[0], params[_ADV_PARAM_GAMMA_REF], strength_params)
        _quad4_fill_advanced_strength_post_row_numba(
            data,
            gp,
            xi,
            eta,
            x,
            y,
            det * weight * thickness,
            strain,
            stress,
            p,
            q,
            plastic,
            yield_value,
            kappa_new,
            plastic_strain_new,
            history,
            plastic_multiplier,
        )
        data[gp, 2] = weight
    return data, min_det

@njit(cache=True)
def _quad8_advanced_strength_j2dp_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float, float]:
    data = np.zeros((9, 43), dtype=np.float64)
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
        alpha, cohesion_term, hardening, _strength_model = _advanced_strength_yield_params_numba(model_id, history, params, strength_params, strength_factor)
        D4, S4, shear_mu = _quad4_effective_d4_s4_numba(history[_ADV_STATE_EFFECTIVE_E], nu)
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
        if use_tension_cutoff > 0.5:
            stress, clipped, excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
            if clipped:
                plastic = 1.0
            yield_value = max(yield_value, excess)
            elastic_strain_new = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(4):
                    value += S4[i, j] * (stress[j] - initial_stress[j])
                elastic_strain_new[i] = value
            for i in range(4):
                plastic_strain_new[i] = strain_eff[i] - elastic_strain_new[i]
            mean_new = (stress[0] + stress[1] + stress[2]) / 3.0
            nd0 = stress[0] - mean_new
            nd1 = stress[1] - mean_new
            nd2 = stress[2] - mean_new
            nd3 = stress[3]
            q = math.sqrt(max(1.5 * (nd0 * nd0 + nd1 * nd1 + nd2 * nd2) + 3.0 * nd3 * nd3, 0.0))
            p = -mean_new
        plastic_multiplier = max(kappa_new - kappas[gp], 0.0)
        history[_ADV_STATE_HARDENING_VARIABLE] = history[_ADV_STATE_HARDENING_VARIABLE] + plastic_multiplier * strength_params[_ADV_STRENGTH_HARDENING_RATE]
        history[_ADV_STATE_DILATANCY] = _advanced_strength_dilatancy_numba(model_id, history[0], params[_ADV_PARAM_GAMMA_REF], strength_params)
        _quad4_fill_advanced_strength_post_row_numba(
            data,
            gp,
            xi,
            eta,
            xs[gp],
            ys[gp],
            dvols[gp],
            strain_eff,
            stress,
            p,
            q,
            plastic,
            yield_value,
            kappa_new,
            plastic_strain_new,
            history,
            plastic_multiplier,
        )
        data[gp, 2] = weights[gp]
    return data, min_det, volume

@njit(cache=True)
def _advanced_strength_j2dp_stress_numba(
    strain: np.ndarray,
    previous_state: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    initial_stress: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> np.ndarray:
    history = _advanced_history_state_numba(strain, previous_state, params, model_id)
    alpha, cohesion_term, hardening, _strength_model = _advanced_strength_yield_params_numba(model_id, history, params, strength_params, strength_factor)
    D4, S4, shear_mu = _quad4_effective_d4_s4_numba(history[_ADV_STATE_EFFECTIVE_E], nu)
    stress, _plastic, _yield_value, _p, _q, _plastic_strain_new, _kappa_new = _j2dp_post_update_numba(
        strain,
        plastic_strain,
        kappa,
        D4,
        S4,
        initial_stress,
        alpha,
        cohesion_term,
        hardening,
        shear_mu,
    )
    if use_tension_cutoff > 0.5:
        stress, _clipped, _excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
    return stress

@njit(cache=True)
def _advanced_strength_j2dp_stress_tangent_numba(
    strain: np.ndarray,
    previous_state: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    initial_stress: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, np.ndarray]:
    norm = 0.0
    for i in range(4):
        norm += strain[i] * strain[i]
    stress = _advanced_strength_j2dp_stress_numba(
        strain,
        previous_state,
        plastic_strain,
        kappa,
        initial_stress,
        params,
        strength_params,
        model_id,
        nu,
        strength_factor,
        tensile_strength,
        use_tension_cutoff,
    )
    delta = 1.0e-8 * max(1.0, math.sqrt(norm))
    tangent = np.zeros((4, 4), dtype=np.float64)
    for col in range(4):
        perturbed = strain.copy()
        perturbed[col] += delta
        plus = _advanced_strength_j2dp_stress_numba(
            perturbed,
            previous_state,
            plastic_strain,
            kappa,
            initial_stress,
            params,
            strength_params,
            model_id,
            nu,
            strength_factor,
            tensile_strength,
            use_tension_cutoff,
        )
        for row in range(4):
            tangent[row, col] = (plus[row] - stress[row]) / delta
    return stress, tangent

@njit(cache=True)
def _quad4_advanced_strength_j2dp_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
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
        norm = 0.0
        for i in range(4):
            value = 0.0
            for j in range(8):
                value += B[i, j] * ue[j]
            strain[i] = value
            norm += value * value
        stress = _advanced_strength_j2dp_stress_numba(
            strain,
            previous_states[gp],
            plastic_strains[gp],
            kappas[gp],
            initial_stress,
            params,
            strength_params,
            model_id,
            nu,
            strength_factor,
            tensile_strength,
            use_tension_cutoff,
        )
        delta = 1.0e-8 * max(1.0, math.sqrt(norm))
        tangent = np.zeros((4, 4), dtype=np.float64)
        for col in range(4):
            perturbed = strain.copy()
            perturbed[col] += delta
            plus = _advanced_strength_j2dp_stress_numba(
                perturbed,
                previous_states[gp],
                plastic_strains[gp],
                kappas[gp],
                initial_stress,
                params,
                strength_params,
                model_id,
                nu,
                strength_factor,
                tensile_strength,
                use_tension_cutoff,
            )
            for row in range(4):
                tangent[row, col] = (plus[row] - stress[row]) / delta
        dV = det * thickness
        _quad4_add_btcb_numba(ke, B, tangent, dV)
        for i in range(8):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    return ke, fe, min_det

@njit(cache=True)
def _quad8_advanced_strength_j2dp_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    mode_code: int,
    strength_factor: float,
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
            stress, tangent = _advanced_strength_j2dp_stress_tangent_numba(
                strain,
                previous_states[gp],
                plastic_strains[gp],
                kappas[gp],
                initial_stress,
                params,
                strength_params,
                model_id,
                nu,
                strength_factor,
                tensile_strength,
                use_tension_cutoff,
            )
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
            stress, tangent = _advanced_strength_j2dp_stress_tangent_numba(
                strain,
                previous_states[gp],
                plastic_strains[gp],
                kappas[gp],
                initial_stress,
                params,
                strength_params,
                model_id,
                nu,
                strength_factor,
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
            stress, tangent = _advanced_strength_j2dp_stress_tangent_numba(
                strain,
                previous_states[state_index],
                plastic_strains[state_index],
                kappas[state_index],
                initial_stress,
                params,
                strength_params,
                model_id,
                nu,
                strength_factor,
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
        stress, tangent = _advanced_strength_j2dp_stress_tangent_numba(
            strain,
            previous_states[gp],
            plastic_strains[gp],
            kappas[gp],
            initial_stress,
            params,
            strength_params,
            model_id,
            nu,
            strength_factor,
            tensile_strength,
            use_tension_cutoff,
        )
        dV = dV_cache[gp]
        _quad8_add_btcb_numba(ke, B_eff, tangent, dV)
        _quad8_add_btstress_numba(fe, B_eff, stress, dV)
    return ke, fe, min_det, volume

@njit(cache=True)
def _advanced_strength_mc_params_numba(
    model_id: int,
    history: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    strength_factor: float,
) -> tuple[float, float, float, float]:
    hardening_variable = history[_ADV_STATE_HARDENING_VARIABLE]
    cohesion = strength_params[_ADV_STRENGTH_COHESION]
    phi = strength_params[_ADV_STRENGTH_FRICTION]
    psi = _angle_radians_numba(strength_params[_ADV_STRENGTH_DILATANCY_BASE])
    if model_id == _ADV_MODEL_UW_CLAY:
        cohesion = max(strength_params[_ADV_STRENGTH_UW_SU] + hardening_variable, 0.0)
        phi = 0.0
        psi = 0.0
    elif model_id == _ADV_MODEL_PZ_CLAY:
        cohesion = max(strength_params[_ADV_STRENGTH_PZ_CLAY_SU] + hardening_variable, strength_params[_ADV_STRENGTH_COHESION], 0.0)
        phi = strength_params[_ADV_STRENGTH_PZ_CLAY_PHI]
        psi = min(_angle_radians_numba(_advanced_strength_dilatancy_numba(model_id, history[0], params[_ADV_PARAM_GAMMA_REF], strength_params)), phi)
    elif model_id == _ADV_MODEL_PZ_SAND:
        cohesion = max(strength_params[_ADV_STRENGTH_COHESION] + hardening_variable, strength_params[_ADV_STRENGTH_PZ_SAND_COHESION])
        phi = strength_params[_ADV_STRENGTH_PZ_SAND_PHI]
        psi = min(_angle_radians_numba(_advanced_strength_dilatancy_numba(model_id, history[0], params[_ADV_PARAM_GAMMA_REF], strength_params)), phi)
    elif model_id == _ADV_MODEL_LIQUEFACTION or model_id == _ADV_MODEL_BILINEAR_LIQUEFACTION:
        strength_ratio = max(strength_params[_ADV_STRENGTH_RESIDUAL_RATIO], 1.0 - history[_ADV_STATE_RU])
        cohesion = max(strength_params[_ADV_STRENGTH_COHESION] * strength_ratio, 0.0)
        if cohesion <= 0.0 and strength_params[_ADV_STRENGTH_YIELD_STRESS] > 0.0:
            cohesion = strength_params[_ADV_STRENGTH_YIELD_STRESS] * strength_ratio / math.sqrt(3.0)
        phi = max(strength_params[_ADV_STRENGTH_FRICTION] * strength_ratio, 0.0)
        psi = min(_angle_radians_numba(strength_params[_ADV_STRENGTH_DILATANCY_BASE]), phi)

    factor = max(strength_factor, np.finfo(np.float64).eps)
    c = max(cohesion, 0.0) / factor
    phi_reduced = math.atan(math.tan(phi) / factor)
    psi_reduced = math.atan(math.tan(psi) / factor)
    cohesion_term = 2.0 * c * math.cos(phi_reduced)
    return phi_reduced, psi_reduced, cohesion_term, strength_params[_ADV_STRENGTH_HARDENING]

@njit(cache=True)
def _advanced_strength_mc_stress_numba(
    strain: np.ndarray,
    previous_state: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    initial_stress: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[bool, np.ndarray]:
    history = _advanced_history_state_numba(strain, previous_state, params, model_id)
    D4, S4, _shear_mu = _quad4_effective_d4_s4_numba(history[_ADV_STATE_EFFECTIVE_E], nu)
    phi, psi, cohesion_term, hardening = _advanced_strength_mc_params_numba(model_id, history, params, strength_params, strength_factor)
    ok, stress, _plastic, _yield_value, _p, _q, _plastic_strain_new, _kappa_new, _active_ids, _active_count = _quad4_mc_post_update_numba(
        strain,
        plastic_strain,
        kappa,
        D4,
        S4,
        initial_stress,
        _quad4_mc_plane_coeffs_numba(phi),
        _quad4_mc_plane_coeffs_numba(psi),
        cohesion_term,
        hardening,
    )
    if not ok:
        return False, stress
    if use_tension_cutoff > 0.5:
        stress, _clipped, _excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
    return True, stress

@njit(cache=True)
def _advanced_strength_mc_post_update_numba(
    strain: np.ndarray,
    previous_state: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    initial_stress: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[bool, np.ndarray, np.ndarray, float, float, float, float, np.ndarray, float, float]:
    history = _advanced_history_state_numba(strain, previous_state, params, model_id)
    D4, S4, _shear_mu = _quad4_effective_d4_s4_numba(history[_ADV_STATE_EFFECTIVE_E], nu)
    phi, psi, cohesion_term, hardening = _advanced_strength_mc_params_numba(
        model_id,
        history,
        params,
        strength_params,
        strength_factor,
    )
    ok, stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new, _active_ids, _active_count = _quad4_mc_maybe_tension_post_update_numba(
        strain,
        plastic_strain,
        kappa,
        D4,
        S4,
        initial_stress,
        _quad4_mc_plane_coeffs_numba(phi),
        _quad4_mc_plane_coeffs_numba(psi),
        cohesion_term,
        hardening,
        tensile_strength,
        use_tension_cutoff,
    )
    if not ok:
        return False, history, stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new, 0.0
    plastic_multiplier = max(kappa_new - kappa, 0.0)
    history[_ADV_STATE_HARDENING_VARIABLE] = history[_ADV_STATE_HARDENING_VARIABLE] + plastic_multiplier * strength_params[_ADV_STRENGTH_HARDENING_RATE]
    history[_ADV_STATE_DILATANCY] = _advanced_strength_dilatancy_numba(model_id, history[0], params[_ADV_PARAM_GAMMA_REF], strength_params)
    return True, history, stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new, plastic_multiplier

@njit(cache=True)
def _advanced_strength_mc_stress_tangent_numba(
    strain: np.ndarray,
    previous_state: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    initial_stress: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[bool, np.ndarray, np.ndarray]:
    norm = 0.0
    for i in range(4):
        norm += strain[i] * strain[i]
    ok, stress = _advanced_strength_mc_stress_numba(
        strain,
        previous_state,
        plastic_strain,
        kappa,
        initial_stress,
        params,
        strength_params,
        model_id,
        nu,
        strength_factor,
        tensile_strength,
        use_tension_cutoff,
    )
    tangent = np.zeros((4, 4), dtype=np.float64)
    if not ok:
        return False, stress, tangent
    delta = 1.0e-8 * max(1.0, math.sqrt(norm))
    for col in range(4):
        perturbed = strain.copy()
        perturbed[col] += delta
        ok_plus, plus = _advanced_strength_mc_stress_numba(
            perturbed,
            previous_state,
            plastic_strain,
            kappa,
            initial_stress,
            params,
            strength_params,
            model_id,
            nu,
            strength_factor,
            tensile_strength,
            use_tension_cutoff,
        )
        if not ok_plus:
            return False, stress, tangent
        for row in range(4):
            tangent[row, col] = (plus[row] - stress[row]) / delta
    return True, stress, tangent

@njit(cache=True)
def _quad4_advanced_strength_mc_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, float, bool]:
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
            return ke, fe, min_det, False
        strain = np.zeros(4, dtype=np.float64)
        norm = 0.0
        for i in range(4):
            value = 0.0
            for j in range(8):
                value += B[i, j] * ue[j]
            strain[i] = value
            norm += value * value
        ok, stress = _advanced_strength_mc_stress_numba(
            strain,
            previous_states[gp],
            plastic_strains[gp],
            kappas[gp],
            initial_stress,
            params,
            strength_params,
            model_id,
            nu,
            strength_factor,
            tensile_strength,
            use_tension_cutoff,
        )
        if not ok:
            return ke, fe, min_det, False
        delta = 1.0e-8 * max(1.0, math.sqrt(norm))
        tangent = np.zeros((4, 4), dtype=np.float64)
        for col in range(4):
            perturbed = strain.copy()
            perturbed[col] += delta
            ok_plus, plus = _advanced_strength_mc_stress_numba(
                perturbed,
                previous_states[gp],
                plastic_strains[gp],
                kappas[gp],
                initial_stress,
                params,
                strength_params,
                model_id,
                nu,
                strength_factor,
                tensile_strength,
                use_tension_cutoff,
            )
            if not ok_plus:
                return ke, fe, min_det, False
            for row in range(4):
                tangent[row, col] = (plus[row] - stress[row]) / delta
        dV = det * thickness
        _quad4_add_btcb_numba(ke, B, tangent, dV)
        for i in range(8):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    return ke, fe, min_det, True

@njit(cache=True)
def _quad8_advanced_strength_mc_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float, bool]:
    data = np.zeros((9, 43), dtype=np.float64)
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
            return data, min_det, False
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
        ok, history, stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new, plastic_multiplier = _advanced_strength_mc_post_update_numba(
            strain,
            previous_states[gp],
            plastic_strains[gp],
            kappas[gp],
            initial_stress,
            params,
            strength_params,
            model_id,
            nu,
            strength_factor,
            tensile_strength,
            use_tension_cutoff,
        )
        if not ok:
            return data, min_det, False
        _quad4_fill_advanced_strength_post_row_numba(
            data,
            gp,
            xi,
            eta,
            x,
            y,
            det * weight * thickness,
            strain,
            stress,
            p,
            q,
            plastic,
            yield_value,
            kappa_new,
            plastic_strain_new,
            history,
            plastic_multiplier,
        )
        data[gp, 2] = weight
    return data, min_det, True

@njit(cache=True)
def _quad8_advanced_strength_mc_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    Pvol: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float, float, bool]:
    data = np.zeros((9, 43), dtype=np.float64)
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
            return data, min_det, volume, False
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
        return data, min_det, volume, False

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
        ok, history, stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new, plastic_multiplier = _advanced_strength_mc_post_update_numba(
            strain_eff,
            previous_states[gp],
            plastic_strains[gp],
            kappas[gp],
            initial_stress,
            params,
            strength_params,
            model_id,
            nu,
            strength_factor,
            tensile_strength,
            use_tension_cutoff,
        )
        if not ok:
            return data, min_det, volume, False
        _quad4_fill_advanced_strength_post_row_numba(
            data,
            gp,
            xi,
            eta,
            xs[gp],
            ys[gp],
            dvols[gp],
            strain_eff,
            stress,
            p,
            q,
            plastic,
            yield_value,
            kappa_new,
            plastic_strain_new,
            history,
            plastic_multiplier,
        )
        data[gp, 2] = weights[gp]
    return data, min_det, volume, True

@njit(cache=True)
def _quad8_advanced_strength_mc_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    initial_stress: np.ndarray,
    previous_states: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    params: np.ndarray,
    strength_params: np.ndarray,
    model_id: int,
    nu: float,
    thickness: float,
    mode_code: int,
    strength_factor: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, np.ndarray, float, float, bool]:
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
                return ke, fe, min_det, volume, False
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            ok, stress, tangent = _advanced_strength_mc_stress_tangent_numba(
                strain,
                previous_states[gp],
                plastic_strains[gp],
                kappas[gp],
                initial_stress,
                params,
                strength_params,
                model_id,
                nu,
                strength_factor,
                tensile_strength,
                use_tension_cutoff,
            )
            if not ok:
                return ke, fe, min_det, volume, False
            dV = det * weight * thickness
            _quad8_add_btcb_numba(ke, B, tangent, dV)
            _quad8_add_btstress_numba(fe, B, stress, dV)
        return ke, fe, min_det, volume, True

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, fe, min_det, volume, False
            Bdev = _quad8_project_b_numba(Idev, B)
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            ok, stress, tangent = _advanced_strength_mc_stress_tangent_numba(
                strain,
                previous_states[gp],
                plastic_strains[gp],
                kappas[gp],
                initial_stress,
                params,
                strength_params,
                model_id,
                nu,
                strength_factor,
                tensile_strength,
                use_tension_cutoff,
            )
            if not ok:
                return ke, fe, min_det, volume, False
            dV = det * weight * thickness
            _quad8_add_btlcbr_numba(ke, Bdev, tangent, B, dV)
            _quad8_add_btstress_numba(fe, Bdev, stress, dV)
        for gp in range(4):
            xi, eta, weight = _quad8_gp_reduced(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return ke, fe, min_det, volume, False
            Bv = _quad8_project_b_numba(Pvol, B)
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            state_index = 9 + gp
            ok, stress, tangent = _advanced_strength_mc_stress_tangent_numba(
                strain,
                previous_states[state_index],
                plastic_strains[state_index],
                kappas[state_index],
                initial_stress,
                params,
                strength_params,
                model_id,
                nu,
                strength_factor,
                tensile_strength,
                use_tension_cutoff,
            )
            if not ok:
                return ke, fe, min_det, volume, False
            dV = det * weight * thickness
            _quad8_add_btlcbr_numba(ke, Bv, tangent, B, dV)
            _quad8_add_btstress_numba(fe, Bv, stress, dV)
        return ke, fe, min_det, volume, True

    B_cache = np.zeros((9, 4, 16), dtype=np.float64)
    dV_cache = np.zeros(9, dtype=np.float64)
    Bv_acc = np.zeros((4, 16), dtype=np.float64)
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return ke, fe, min_det, volume, False
        dV = det * weight * thickness
        dV_cache[gp] = dV
        volume += dV
        Bv = _quad8_project_b_numba(Pvol, B)
        for r in range(4):
            for c in range(16):
                B_cache[gp, r, c] = B[r, c]
                Bv_acc[r, c] += Bv[r, c] * dV
    if volume <= 0.0:
        return ke, fe, min_det, volume, False
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
        ok, stress, tangent = _advanced_strength_mc_stress_tangent_numba(
            strain,
            previous_states[gp],
            plastic_strains[gp],
            kappas[gp],
            initial_stress,
            params,
            strength_params,
            model_id,
            nu,
            strength_factor,
            tensile_strength,
            use_tension_cutoff,
        )
        if not ok:
            return ke, fe, min_det, volume, False
        dV = dV_cache[gp]
        _quad8_add_btcb_numba(ke, B_eff, tangent, dV)
        _quad8_add_btstress_numba(fe, B_eff, stress, dV)
    return ke, fe, min_det, volume, True

def _advanced_strength_params_array(material: ElasticPlaneStrainMaterial, source_model: str) -> np.ndarray:
    params = material.advanced_params or {}
    liq = params.get("liquefaction")
    liq_map = liq if isinstance(liq, Mapping) else params
    out = np.zeros(_ADV_STRENGTH_COUNT, dtype=float)
    out[_ADV_STRENGTH_COHESION] = float(material.cohesion)
    out[_ADV_STRENGTH_YIELD_STRESS] = float(material.yield_stress)
    out[_ADV_STRENGTH_FRICTION] = _angle_radians(float(material.friction_angle))
    out[_ADV_STRENGTH_HARDENING] = float(material.hardening)
    out[_ADV_STRENGTH_HARDENING_RATE] = _param_float(params, ("advanced_hardening", "pz_hardening", "hardening_rate"), max(material.hardening, 0.0))
    out[_ADV_STRENGTH_UW_SU] = _param_float(
        params,
        ("su", "cu", "undrained_shear_strength"),
        material.cohesion if material.cohesion > 0.0 else max(material.yield_stress / math.sqrt(3.0), 1.0),
    )
    out[_ADV_STRENGTH_PZ_CLAY_SU] = _param_float(
        params,
        ("su", "cu", "undrained_shear_strength"),
        material.cohesion if material.cohesion > 0.0 else 5.0,
    )
    out[_ADV_STRENGTH_PZ_CLAY_PHI] = _angle_radians(_param_float(params, ("phi_cs", "critical_state_phi", "friction_angle"), material.friction_angle))
    out[_ADV_STRENGTH_PZ_SAND_PHI] = _angle_radians(_param_float(params, ("phi_cs", "critical_state_phi", "friction_angle"), material.friction_angle if material.friction_angle > 0.0 else 32.0))
    out[_ADV_STRENGTH_PZ_SAND_COHESION] = _param_float(params, ("c", "cohesion"), 0.0)
    out[_ADV_STRENGTH_RESIDUAL_RATIO] = max(
        _param_float(liq_map, ("residual_strength_ratio", "post_liquefaction_strength_ratio"), 0.05) if isinstance(liq_map, Mapping) else 0.05,
        0.0,
    )
    base_dilatancy = _param_float(params, ("dilation_angle", "psi"), material.dilation_angle)
    out[_ADV_STRENGTH_DILATANCY_BASE] = base_dilatancy
    out[_ADV_STRENGTH_DILATANCY_PEAK] = _param_float(params, ("peak_dilation_angle", "psi_peak"), base_dilatancy)
    out[_ADV_STRENGTH_DILATANCY_RESIDUAL] = _param_float(params, ("residual_dilation_angle", "psi_residual"), min(base_dilatancy, 0.0))
    if _advanced_strength_model_name(params) in {"mohr_coulomb", "mc"}:
        out[_ADV_STRENGTH_MODEL] = _ADV_STRENGTH_MODEL_MC
    else:
        out[_ADV_STRENGTH_MODEL] = _ADV_STRENGTH_MODEL_AUTO
    return out

def _quad4_advanced_strength_j2dp_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad4_advanced_state_arrays(element_id, plastic_state, params)
    plastic_strains, kappas = _quad4_post_state_arrays(element_id, plastic_state)
    data, min_det = _quad4_advanced_strength_j2dp_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(plastic_strains, dtype=np.float64),
        np.ascontiguousarray(kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(strength_factor),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data

def _quad4_advanced_strength_j2dp_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad4_advanced_state_arrays(element_id, plastic_state, params)
    plastic_strains, kappas = _quad4_post_state_arrays(element_id, plastic_state)
    data, min_det = _quad4_advanced_strength_j2dp_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(plastic_strains, dtype=np.float64),
        np.ascontiguousarray(kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(strength_factor),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data

def _quad8_advanced_strength_j2dp_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params)
    plastic_strains, kappas = _quad8_post_state_arrays(element_id, plastic_state)
    data, min_det = _quad8_advanced_strength_j2dp_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(plastic_strains, dtype=np.float64),
        np.ascontiguousarray(kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(strength_factor),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return data

def _quad8_advanced_strength_j2dp_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> np.ndarray:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params)
    plastic_strains, kappas = _quad8_post_state_arrays(element_id, plastic_state)
    data, min_det, volume = _quad8_advanced_strength_j2dp_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(plastic_strains, dtype=np.float64),
        np.ascontiguousarray(kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(strength_factor),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return data

def _quad8_advanced_strength_mc_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> np.ndarray | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params)
    plastic_strains, kappas = _quad8_post_state_arrays(element_id, plastic_state)
    try:
        data, min_det, ok = _quad8_advanced_strength_mc_post_numba(
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(ue, dtype=np.float64),
            np.ascontiguousarray(initial, dtype=np.float64),
            np.ascontiguousarray(previous_states, dtype=np.float64),
            np.ascontiguousarray(plastic_strains, dtype=np.float64),
            np.ascontiguousarray(kappas, dtype=np.float64),
            np.ascontiguousarray(params, dtype=np.float64),
            np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
            _advanced_model_id(source_model),
            float(material.nu),
            float(material.thickness),
            float(strength_factor),
            float(material.tensile_strength),
            1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
        )
    except (ZeroDivisionError, FloatingPointError):
        return None
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if not ok:
        return None
    return data

def _quad8_advanced_strength_mc_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> np.ndarray | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params)
    plastic_strains, kappas = _quad8_post_state_arrays(element_id, plastic_state)
    try:
        data, min_det, volume, ok = _quad8_advanced_strength_mc_bbar_post_numba(
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(ue, dtype=np.float64),
            np.ascontiguousarray(initial, dtype=np.float64),
            np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
            np.ascontiguousarray(previous_states, dtype=np.float64),
            np.ascontiguousarray(plastic_strains, dtype=np.float64),
            np.ascontiguousarray(kappas, dtype=np.float64),
            np.ascontiguousarray(params, dtype=np.float64),
            np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
            _advanced_model_id(source_model),
            float(material.nu),
            float(material.thickness),
            float(strength_factor),
            float(material.tensile_strength),
            1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
        )
    except (ZeroDivisionError, FloatingPointError):
        return None
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    if not ok:
        return None
    return data

def _quad4_advanced_strength_j2dp_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad4_advanced_state_arrays(element_id, plastic_state, params)
    plastic_strains, kappas = _quad4_post_state_arrays(element_id, plastic_state)
    ke, fe, min_det = _quad4_advanced_strength_j2dp_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(plastic_strains, dtype=np.float64),
        np.ascontiguousarray(kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(strength_factor),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return ke, fe

def _quad8_advanced_strength_j2dp_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    mode: str,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    mode_code = _quad4_mode_code(mode)
    expected_points = 13 if mode_code == _QUAD4_MODE_SRI else 9
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params, expected_points)
    plastic_strains, kappas = _quad8_post_state_arrays(element_id, plastic_state, expected_points)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    ke, fe, min_det, volume = _quad8_advanced_strength_j2dp_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(plastic_strains, dtype=np.float64),
        np.ascontiguousarray(kappas, dtype=np.float64),
        Pvol,
        Idev,
        np.ascontiguousarray(params, dtype=np.float64),
        np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        mode_code,
        float(strength_factor),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return ke, fe

def _quad4_advanced_strength_mc_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad4_advanced_state_arrays(element_id, plastic_state, params)
    plastic_strains, kappas = _quad4_post_state_arrays(element_id, plastic_state)
    ke, fe, min_det, ok = _quad4_advanced_strength_mc_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(previous_states, dtype=np.float64),
        np.ascontiguousarray(plastic_strains, dtype=np.float64),
        np.ascontiguousarray(kappas, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
        _advanced_model_id(source_model),
        float(material.nu),
        float(material.thickness),
        float(strength_factor),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    if not ok or not np.all(np.isfinite(ke)) or not np.all(np.isfinite(fe)):
        return None
    return ke, fe

def _quad8_advanced_strength_mc_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    mode: str,
    *,
    initial_stress: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    strength_factor: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    mode_code = _quad4_mode_code(mode)
    expected_points = 13 if mode_code == _QUAD4_MODE_SRI else 9
    source_model = material.advanced_model or material.model
    params = _advanced_params_array(material, source_model)
    previous_states, _previous_kappas = _quad8_advanced_state_arrays(element_id, plastic_state, params, expected_points)
    plastic_strains, kappas = _quad8_post_state_arrays(element_id, plastic_state, expected_points)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    try:
        ke, fe, min_det, volume, ok = _quad8_advanced_strength_mc_tangent_force_numba(
            np.ascontiguousarray(coords, dtype=np.float64),
            np.ascontiguousarray(ue, dtype=np.float64),
            np.ascontiguousarray(initial, dtype=np.float64),
            np.ascontiguousarray(previous_states, dtype=np.float64),
            np.ascontiguousarray(plastic_strains, dtype=np.float64),
            np.ascontiguousarray(kappas, dtype=np.float64),
            Pvol,
            Idev,
            np.ascontiguousarray(params, dtype=np.float64),
            np.ascontiguousarray(_advanced_strength_params_array(material, source_model), dtype=np.float64),
            _advanced_model_id(source_model),
            float(material.nu),
            float(material.thickness),
            mode_code,
            float(strength_factor),
            float(material.tensile_strength),
            1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
        )
    except (ZeroDivisionError, FloatingPointError):
        return None
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    if not ok or not np.all(np.isfinite(ke)) or not np.all(np.isfinite(fe)):
        return None
    return ke, fe

__all__ = list(ADVANCED_STRENGTH_ELEMENT_KERNEL_FUNCTIONS)
