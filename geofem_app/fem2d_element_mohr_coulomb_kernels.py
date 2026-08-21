"""Mohr-Coulomb nonlinear element kernels for plane-strain analyses."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fem2d_element_elastic_post import _quad4_post_principal_values_numba
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
    _mc_consistent_tangent_spectral_numba,
    _mc_consistent_tangent_spectral_precomputed_numba,
    _mc_eval_active_candidate_numba,
    _mc_legacy_bounded_projection_return_mapping_numba,
    _mc_plane_coeffs,
    _mc_refine_active_candidate_lstsq_numba,
    _mc_reduced_parameters,
    _mc_regularized_projection_return_mapping_numba,
    _mc_return_mapping_principal_numba,
    _mc_return_mapping_principal_precomputed_numba,
    _mc_tension_cutoff_consistent_tangent_numba,
    _mc_tension_cutoff_stress_numba,
    _tension_cutoff_plane_strain_numba,
)
from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, njit

_QUAD4_MODE_FULL = 0
_QUAD4_MODE_SRI = 1
_QUAD4_MODE_BBAR = 2

MOHR_COULOMB_ELEMENT_KERNEL_FUNCTIONS = (
    "mohr_coulomb_element_kernel_contract",
    "_quad4_mc_plane_coeffs_numba",
    "_quad4_mc_principal_frame_numba",
    "_quad4_mc_pq_numba",
    "_quad4_mc_post_update_numba",
    "_quad4_mc_tension_cutoff_post_update_numba",
    "_quad4_mc_maybe_tension_post_update_numba",
    "_quad4_mc_stress_tangent_numba",
    "_quad4_mc_stress_tangent_state_regularized_numba",
    "_quad4_mc_stress_tangent_state_active_set_numba",
    "_quad4_mc_maybe_tension_stress_tangent_numba",
    "_quad4_mc_tangent_force_numba",
    "_quad4_mc_internal_force_numba",
    "_quad8_mc_post_numba",
    "_quad8_mc_bbar_post_numba",
    "_quad4_mc_post_numba",
    "_quad4_mc_bbar_post_numba",
    "_quad8_mc_tangent_force_numba",
    "_quad8_mc_internal_force_numba",
    "_quad4_mc_post_fast",
    "_quad4_mc_bbar_post_fast",
    "_quad8_mc_post_fast",
    "_quad8_mc_bbar_post_fast",
    "_quad4_mc_tangent_force_fast",
    "_quad4_mc_internal_force_fast",
    "_quad8_mc_tangent_force_fast",
    "_quad8_mc_internal_force_fast",
)


def _quad4_mode_code(mode: str) -> int:
    if mode == "FULL":
        return _QUAD4_MODE_FULL
    if mode == "SRI":
        return _QUAD4_MODE_SRI
    if mode == "B-BAR":
        return _QUAD4_MODE_BBAR
    raise FEM2DError(f"unsupported integration '{mode}'")


def mohr_coulomb_element_kernel_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_mohr_coulomb_kernels.v1",
        "module": "geofem_app.fem2d_element_mohr_coulomb_kernels",
        "function_count": len(MOHR_COULOMB_ELEMENT_KERNEL_FUNCTIONS),
        "functions": list(MOHR_COULOMB_ELEMENT_KERNEL_FUNCTIONS),
        "covered_surfaces": [
            "quad4_mohr_coulomb_tangent_internal_force",
            "quad8_mohr_coulomb_tangent_internal_force",
            "quad4_quad8_mohr_coulomb_post",
            "mohr_coulomb_tension_cutoff",
            "advanced_material_shared_mc_update",
            "full_sri_bbar_dispatch",
        ],
    }


@njit(cache=True)
def _quad4_mc_plane_coeffs_numba(angle: float) -> np.ndarray:
    s = math.sin(angle)
    out = np.zeros((6, 3), dtype=np.float64)
    out[0, 0] = 1.0 + s
    out[0, 1] = -(1.0 - s)
    out[1, 0] = 1.0 + s
    out[1, 2] = -(1.0 - s)
    out[2, 1] = 1.0 + s
    out[2, 0] = -(1.0 - s)
    out[3, 1] = 1.0 + s
    out[3, 2] = -(1.0 - s)
    out[4, 2] = 1.0 + s
    out[4, 0] = -(1.0 - s)
    out[5, 2] = 1.0 + s
    out[5, 1] = -(1.0 - s)
    return out

@njit(cache=True)
def _quad4_mc_principal_frame_numba(stress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sx = stress[0]
    sy = stress[1]
    sz = stress[2]
    txy = stress[3]
    mean = 0.5 * (sx + sy)
    diff = 0.5 * (sx - sy)
    radius = math.sqrt(diff * diff + txy * txy)
    low = mean - radius
    high = mean + radius
    values = np.empty(3, dtype=np.float64)
    vectors = np.zeros((3, 3), dtype=np.float64)
    values[0] = low
    values[1] = high
    values[2] = sz
    if radius <= 1.0e-30:
        vectors[0, 0] = 0.0
        vectors[1, 0] = 1.0
        vectors[0, 1] = 1.0
        vectors[1, 1] = 0.0
    else:
        theta = 0.5 * math.atan2(2.0 * txy, sx - sy)
        c = math.cos(theta)
        s = math.sin(theta)
        vectors[0, 0] = -s
        vectors[1, 0] = c
        vectors[0, 1] = c
        vectors[1, 1] = s
    vectors[2, 2] = 1.0
    for i in range(2):
        best = i
        for j in range(i + 1, 3):
            if values[j] < values[best]:
                best = j
        if best != i:
            tmp = values[i]
            values[i] = values[best]
            values[best] = tmp
            for row in range(3):
                vtmp = vectors[row, i]
                vectors[row, i] = vectors[row, best]
                vectors[row, best] = vtmp
    return values, vectors


@njit(cache=True)
def _quad4_mc_principal_frame_lapack_numba(stress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Match NumPy's fallback eigensystem only when the fast active set misses."""

    tensor = np.zeros((3, 3), dtype=np.float64)
    tensor[0, 0] = stress[0]
    tensor[1, 1] = stress[1]
    tensor[2, 2] = stress[2]
    tensor[0, 1] = stress[3]
    tensor[1, 0] = stress[3]
    return np.linalg.eigh(tensor)


@njit(cache=True)
def _quad4_mc_pq_numba(stress: np.ndarray) -> tuple[float, float]:
    mean = (stress[0] + stress[1] + stress[2]) / 3.0
    d0 = stress[0] - mean
    d1 = stress[1] - mean
    d2 = stress[2] - mean
    d3 = stress[3]
    q = math.sqrt(max(1.5 * (d0 * d0 + d1 * d1 + d2 * d2) + 3.0 * d3 * d3, 0.0))
    return -mean, q

@njit(cache=True)
def _quad4_mc_post_update_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
) -> tuple[bool, np.ndarray, float, float, float, float, np.ndarray, float, np.ndarray, int]:
    trial = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = initial_stress[i]
        for j in range(4):
            value += D4[i, j] * (strain[j] - plastic_strain[j])
        trial[i] = value

    sig_p, vecs = _quad4_mc_principal_frame_numba(trial)
    vals_tr = np.zeros(6, dtype=np.float64)
    f_trial = -1.0e300
    for row in range(6):
        value = -cohesion_term - hardening * kappa
        for j in range(3):
            value += yield_coeffs[row, j] * sig_p[j]
        vals_tr[row] = value
        if value > f_trial:
            f_trial = value
    norm_sig = math.sqrt(sig_p[0] * sig_p[0] + sig_p[1] * sig_p[1] + sig_p[2] * sig_p[2])
    tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
    p, q = _quad4_mc_pq_numba(trial)
    empty_ids = np.full(3, -1, dtype=np.int64)
    if f_trial <= tol:
        return True, trial, 0.0, max(f_trial, 0.0), p, q, plastic_strain.copy(), kappa, empty_ids, 0

    ok, sig_corr_p, active_ids, active_count, gamma, vals_corr = _mc_return_mapping_principal_numba(
        sig_p,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        D4[:3, :3],
        hardening,
        kappa,
        tol,
    )
    if not ok:
        return False, trial, 0.0, f_trial, p, q, plastic_strain.copy(), kappa, empty_ids, 0

    corrected = np.zeros(4, dtype=np.float64)
    for a in range(3):
        va0 = vecs[0, a]
        va1 = vecs[1, a]
        va2 = vecs[2, a]
        s = sig_corr_p[a]
        corrected[0] += s * va0 * va0
        corrected[1] += s * va1 * va1
        corrected[2] += s * va2 * va2
        corrected[3] += s * va0 * va1

    elastic_strain_new = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = 0.0
        for j in range(4):
            value += S4[i, j] * (corrected[j] - initial_stress[j])
        elastic_strain_new[i] = value
    plastic_strain_new = np.empty(4, dtype=np.float64)
    for i in range(4):
        plastic_strain_new[i] = strain[i] - elastic_strain_new[i]
    dgamma = 0.0
    for i in range(active_count):
        dgamma += gamma[i]
    kappa_new = kappa + max(dgamma, 0.0)
    residual = vals_corr[0]
    for i in range(1, 6):
        if vals_corr[i] > residual:
            residual = vals_corr[i]
    p_new, q_new = _quad4_mc_pq_numba(corrected)
    return True, corrected, 1.0, residual, p_new, q_new, plastic_strain_new, kappa_new, active_ids, active_count

@njit(cache=True)
def _quad4_mc_tension_cutoff_post_update_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    tensile_strength: float,
) -> tuple[bool, np.ndarray, float, float, float, float, np.ndarray, float, np.ndarray, int]:
    trial = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = initial_stress[i]
        for j in range(4):
            value += D4[i, j] * (strain[j] - plastic_strain[j])
        trial[i] = value

    sig_p, vecs = _quad4_mc_principal_frame_numba(trial)
    vals_tr = np.zeros(6, dtype=np.float64)
    f_trial = -1.0e300
    for row in range(6):
        value = -cohesion_term - hardening * kappa
        for j in range(3):
            value += yield_coeffs[row, j] * sig_p[j]
        vals_tr[row] = value
        if value > f_trial:
            f_trial = value
    norm_sig = math.sqrt(sig_p[0] * sig_p[0] + sig_p[1] * sig_p[1] + sig_p[2] * sig_p[2])
    tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
    empty_ids = np.full(3, -1, dtype=np.int64)
    if f_trial <= tol:
        corrected, clipped, excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
        p_new, q_new = _quad4_mc_pq_numba(corrected)
        plastic_flag = 1.0 if clipped else 0.0
        return True, corrected, plastic_flag, max(f_trial, excess), p_new, q_new, plastic_strain.copy(), kappa, empty_ids, 0

    ok, sig_corr_p, active_ids, active_count, gamma, vals_corr = _mc_return_mapping_principal_numba(
        sig_p,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        D4[:3, :3],
        hardening,
        kappa,
        tol,
    )
    if not ok:
        p, q = _quad4_mc_pq_numba(trial)
        return False, trial, 0.0, f_trial, p, q, plastic_strain.copy(), kappa, empty_ids, 0

    corrected_pre = np.zeros(4, dtype=np.float64)
    for a in range(3):
        va0 = vecs[0, a]
        va1 = vecs[1, a]
        va2 = vecs[2, a]
        s = sig_corr_p[a]
        corrected_pre[0] += s * va0 * va0
        corrected_pre[1] += s * va1 * va1
        corrected_pre[2] += s * va2 * va2
        corrected_pre[3] += s * va0 * va1
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
    dgamma = 0.0
    for i in range(active_count):
        dgamma += gamma[i]
    kappa_new = kappa + max(dgamma, 0.0)
    residual = vals_corr[0]
    for i in range(1, 6):
        if vals_corr[i] > residual:
            residual = vals_corr[i]
    p_new, q_new = _quad4_mc_pq_numba(corrected)
    return True, corrected, 1.0, max(residual, excess), p_new, q_new, plastic_strain_new, kappa_new, active_ids, active_count

@njit(cache=True)
def _quad4_mc_maybe_tension_post_update_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[bool, np.ndarray, float, float, float, float, np.ndarray, float, np.ndarray, int]:
    if use_tension_cutoff > 0.5:
        return _quad4_mc_tension_cutoff_post_update_numba(
            strain,
            plastic_strain,
            kappa,
            D4,
            S4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
            tensile_strength,
        )
    return _quad4_mc_post_update_numba(
        strain,
        plastic_strain,
        kappa,
        D4,
        S4,
        initial_stress,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        hardening,
    )

@njit(cache=True)
def _quad4_mc_stress_tangent_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
) -> tuple[bool, np.ndarray, np.ndarray]:
    trial = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = initial_stress[i]
        for j in range(4):
            value += D4[i, j] * (strain[j] - plastic_strain[j])
        trial[i] = value

    sig_p, vecs = _quad4_mc_principal_frame_numba(trial)
    f_trial = -1.0e300
    for row in range(6):
        value = -cohesion_term - hardening * kappa
        for j in range(3):
            value += yield_coeffs[row, j] * sig_p[j]
        if value > f_trial:
            f_trial = value
    norm_sig = math.sqrt(sig_p[0] * sig_p[0] + sig_p[1] * sig_p[1] + sig_p[2] * sig_p[2])
    tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
    if f_trial <= tol:
        return True, trial, D4.copy()

    ok, sig_corr_p, active_ids, active_count, _gamma, _vals_corr = _mc_return_mapping_principal_numba(
        sig_p,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        D4[:3, :3],
        hardening,
        kappa,
        tol,
    )
    if not ok or active_count <= 0:
        return False, trial, D4.copy()

    corrected = np.zeros(4, dtype=np.float64)
    for a in range(3):
        va0 = vecs[0, a]
        va1 = vecs[1, a]
        va2 = vecs[2, a]
        s = sig_corr_p[a]
        corrected[0] += s * va0 * va0
        corrected[1] += s * va1 * va1
        corrected[2] += s * va2 * va2
        corrected[3] += s * va0 * va1

    tangent_ok, tangent = _mc_consistent_tangent_spectral_numba(
        sig_p,
        sig_corr_p,
        vecs,
        active_ids,
        active_count,
        yield_coeffs,
        flow_coeffs,
        D4[:3, :3],
        D4,
        hardening,
    )
    if not tangent_ok:
        return False, corrected, D4.copy()
    for i in range(4):
        for j in range(4):
            if not math.isfinite(tangent[i, j]):
                return False, corrected, D4.copy()
    return True, corrected, tangent


@njit(cache=True)
def _mc_active_set_seed_numba(
    sig_tr_p: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
    active_ids_hint: np.ndarray,
    active_count_hint: int,
    candidate_h: np.ndarray,
) -> tuple[bool, bool, np.ndarray, np.ndarray, np.ndarray]:
    """Re-evaluate one cached MC active set and classify its KKT residual."""

    empty_sig = np.zeros(3, dtype=np.float64)
    empty_gamma = np.zeros(3, dtype=np.float64)
    empty_vals = np.zeros(6, dtype=np.float64)
    if active_count_hint <= 0 or active_count_hint > 3:
        return False, False, empty_sig, empty_gamma, empty_vals
    subset_ids = np.full(3, -1, dtype=np.int64)
    used = np.zeros(6, dtype=np.bool_)
    for slot in range(active_count_hint):
        active_id = int(active_ids_hint[slot])
        if active_id < 0 or active_id >= 6 or used[active_id]:
            return False, False, empty_sig, empty_gamma, empty_vals
        used[active_id] = True
        subset_ids[slot] = active_id

    vals_tr = (
        yield_coeffs @ sig_tr_p
        - cohesion_term
        - hardening * kappa
    )
    ok, sig_corr, gamma, vals_corr, metric = (
        _mc_refine_active_candidate_lstsq_numba(
            sig_tr_p,
            vals_tr,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            Cn,
            hardening,
            kappa,
            subset_ids,
            active_count_hint,
            candidate_h,
        )
    )
    if not ok:
        return False, False, empty_sig, gamma, empty_vals
    for value in sig_corr:
        if not math.isfinite(value):
            return False, False, empty_sig, gamma, empty_vals
    norm_sig = math.sqrt(
        sig_tr_p[0] * sig_tr_p[0]
        + sig_tr_p[1] * sig_tr_p[1]
        + sig_tr_p[2] * sig_tr_p[2]
    )
    tol_active = 10.0 * tol
    tol_gamma = 1.0e-12 * max(1.0, norm_sig)
    exact = (
        metric[0] <= tol_active
        and metric[1] <= tol_active
        and metric[2] <= tol_gamma
    )
    return True, exact, sig_corr, gamma, vals_corr


@njit(cache=True)
def _quad4_mc_stress_precomputed_regularized_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
    active_ids_hint: np.ndarray,
    active_count_hint: int,
    allow_regularized_active_set_update: bool,
    apex_policy_code: int = 0,
) -> tuple[bool, np.ndarray, int, float, float, float, int, int, int]:
    """Stress-only MC update with bounded cone-tip fallback regularization."""

    trial = D4 @ (strain - plastic_strain) + initial_stress
    sig_p, vecs = _quad4_mc_principal_frame_numba(trial)
    f_trial = -1.0e300
    for row in range(6):
        value = -cohesion_term - hardening * kappa
        for j in range(3):
            value += yield_coeffs[row, j] * sig_p[j]
        if value > f_trial:
            f_trial = value
    norm_sig = math.sqrt(sig_p[0] * sig_p[0] + sig_p[1] * sig_p[1] + sig_p[2] * sig_p[2])
    tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
    if f_trial <= tol:
        return True, trial, 0, 0.0, 0.0, 0.0, 0, 0, 0

    active_set_attempts = 0
    active_set_hits = 0
    regularized_active_set_hits = 0
    regularized_count = 0
    yield_violation = 0.0
    relative_yield_violation = 0.0
    relaxed_tolerance = 0.0
    ok = False
    sig_corr_p = np.zeros(3, dtype=np.float64)
    active_count = 0
    gamma = np.zeros(3, dtype=np.float64)
    if active_count_hint > 0:
        active_set_attempts = 1
        seed_ok, exact_seed, seed_sig, seed_gamma, _seed_vals = (
            _mc_active_set_seed_numba(
                sig_p,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                D4[:3, :3],
                hardening,
                kappa,
                tol,
                active_ids_hint,
                active_count_hint,
                candidate_h,
            )
        )
        if seed_ok and exact_seed:
            ok = True
            sig_corr_p = seed_sig
            gamma = seed_gamma
            active_count = active_count_hint
            active_set_hits = 1
        elif seed_ok and allow_regularized_active_set_update:
            (
                update_ok,
                update_sig,
                _update_ids,
                _update_active_count,
                _update_gamma,
                _update_gamma_count,
                _update_vals,
                update_yield_violation,
                update_relative_yield_violation,
                update_relaxed_tolerance,
            ) = (
                _mc_legacy_bounded_projection_return_mapping_numba(
                    sig_p,
                    yield_coeffs,
                    flow_coeffs,
                    cohesion_term,
                    D4[:3, :3],
                    hardening,
                    kappa,
                    tol,
                    seed_sig,
                    active_ids_hint,
                    seed_gamma,
                    active_count_hint,
                )
                if apex_policy_code == 1
                else _mc_regularized_projection_return_mapping_numba(
                    sig_p,
                    yield_coeffs,
                    flow_coeffs,
                    cohesion_term,
                    D4[:3, :3],
                    hardening,
                    kappa,
                    tol,
                    seed_sig,
                    active_ids_hint,
                    seed_gamma,
                    active_count_hint,
                )
            )
            if update_ok:
                ok = True
                sig_corr_p = update_sig
                regularized_count = 1
                yield_violation = update_yield_violation
                relative_yield_violation = update_relative_yield_violation
                relaxed_tolerance = update_relaxed_tolerance
                active_set_hits = 1
                regularized_active_set_hits = 1

    if not ok:
        ok, sig_corr_p, _active_ids, active_count, gamma, _vals_corr = _mc_return_mapping_principal_numba(
            sig_p,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            D4[:3, :3],
            hardening,
            kappa,
            tol,
        )
    if not ok:
        sig_p, vecs = _quad4_mc_principal_frame_lapack_numba(trial)
        norm_sig = math.sqrt(
            sig_p[0] * sig_p[0]
            + sig_p[1] * sig_p[1]
            + sig_p[2] * sig_p[2]
        )
        tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
        (
            ok,
            sig_corr_p,
            _active_ids,
            active_count,
            gamma,
            _vals_corr,
        ) = _mc_return_mapping_principal_numba(
            sig_p,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            D4[:3, :3],
            hardening,
            kappa,
            tol,
        )
        gamma_count = active_count
    if not ok:
        # Refine the Python-equivalent shortlist before regularized projection.
        (
            ok,
            sig_corr_p,
            _active_ids,
            active_count,
            gamma,
            _vals_corr,
        ) = _mc_return_mapping_principal_precomputed_numba(
            sig_p,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            D4[:3, :3],
            hardening,
            kappa,
            tol,
            operator1,
            operator2,
            operator3,
            candidate_h,
        )
    if not ok:
        (
            ok,
            sig_corr_p,
            _active_ids,
            _active_count,
            _gamma,
            _gamma_count,
            _vals_corr,
            yield_violation,
            relative_yield_violation,
            relaxed_tolerance,
        ) = (
            _mc_legacy_bounded_projection_return_mapping_numba(
                sig_p,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                D4[:3, :3],
                hardening,
                kappa,
                tol,
                sig_corr_p,
                _active_ids,
                gamma,
                active_count,
            )
            if apex_policy_code == 1
            else _mc_regularized_projection_return_mapping_numba(
                sig_p,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                D4[:3, :3],
                hardening,
                kappa,
                tol,
                sig_corr_p,
                _active_ids,
                gamma,
                active_count,
            )
        )
        if not ok:
            return (
                False,
                trial,
                0,
                yield_violation,
                relative_yield_violation,
                relaxed_tolerance,
                active_set_attempts,
                active_set_hits,
                regularized_active_set_hits,
            )
        regularized_count = 1
    corrected = np.zeros(4, dtype=np.float64)
    for a in range(3):
        va0 = vecs[0, a]
        va1 = vecs[1, a]
        va2 = vecs[2, a]
        value = sig_corr_p[a]
        corrected[0] += value * va0 * va0
        corrected[1] += value * va1 * va1
        corrected[2] += value * va2 * va2
        corrected[3] += value * va0 * va1
    return (
        True,
        corrected,
        regularized_count,
        yield_violation,
        relative_yield_violation,
        relaxed_tolerance,
        active_set_attempts,
        active_set_hits,
        regularized_active_set_hits,
    )


@njit(cache=True)
def _quad4_mc_numerical_tangent_precomputed_regularized_numba(
    strain: np.ndarray,
    base_stress: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
    active_ids_hint: np.ndarray,
    active_count_hint: int,
    allow_regularized_active_set_update: bool,
    apex_policy_code: int = 0,
) -> tuple[bool, np.ndarray, int, float, float, float, int, int, int]:
    tangent = np.zeros((4, 4), dtype=np.float64)
    strain_norm_sq = 0.0
    for i in range(4):
        strain_norm_sq += strain[i] * strain[i]
    delta = 1.0e-8 * max(1.0, math.sqrt(strain_norm_sq))
    regularized_count = 0
    yield_violation = 0.0
    relative_yield_violation = 0.0
    relaxed_tolerance = 0.0
    active_set_attempts = 0
    active_set_hits = 0
    regularized_active_set_hits = 0
    for column in range(4):
        perturbed = strain.copy()
        perturbed[column] += delta
        (
            ok,
            plus_stress,
            plus_regularized_count,
            plus_yield_violation,
            plus_relative_yield_violation,
            plus_relaxed_tolerance,
            plus_active_set_attempts,
            plus_active_set_hits,
            plus_regularized_active_set_hits,
        ) = _quad4_mc_stress_precomputed_regularized_numba(
            perturbed,
            plastic_strain,
            kappa,
            D4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
            operator1,
            operator2,
            operator3,
            candidate_h,
            active_ids_hint,
            active_count_hint,
            allow_regularized_active_set_update,
            apex_policy_code,
        )
        if not ok:
            return (
                False,
                tangent,
                regularized_count,
                yield_violation,
                relative_yield_violation,
                relaxed_tolerance,
                active_set_attempts,
                active_set_hits,
                regularized_active_set_hits,
            )
        regularized_count += plus_regularized_count
        yield_violation = max(yield_violation, plus_yield_violation)
        relative_yield_violation = max(relative_yield_violation, plus_relative_yield_violation)
        relaxed_tolerance = max(relaxed_tolerance, plus_relaxed_tolerance)
        active_set_attempts += plus_active_set_attempts
        active_set_hits += plus_active_set_hits
        regularized_active_set_hits += plus_regularized_active_set_hits
        for row in range(4):
            tangent[row, column] = (plus_stress[row] - base_stress[row]) / delta
    return (
        True,
        tangent,
        regularized_count,
        yield_violation,
        relative_yield_violation,
        relaxed_tolerance,
        active_set_attempts,
        active_set_hits,
        regularized_active_set_hits,
    )


@njit(cache=True)
def _mc_active_ids_match_numba(
    left: np.ndarray,
    left_count: int,
    right: np.ndarray,
    right_count: int,
) -> bool:
    if left_count <= 0 or left_count != right_count:
        return False
    for left_index in range(left_count):
        found = False
        for right_index in range(right_count):
            if left[left_index] == right[right_index]:
                found = True
                break
        if not found:
            return False
    return True


@njit(cache=True)
def _mc_secant_tangent_update_numba(
    strain: np.ndarray,
    stress: np.ndarray,
    active_ids: np.ndarray,
    active_count: int,
    D4: np.ndarray,
    tangent_hint: np.ndarray,
    tangent_strain_hint: np.ndarray,
    tangent_stress_hint: np.ndarray,
    tangent_active_ids_hint: np.ndarray,
    tangent_active_count_hint: int,
    tangent_hint_valid: bool,
    tangent_hint_reuse_count: int,
) -> tuple[bool, np.ndarray]:
    tangent = np.zeros((4, 4), dtype=np.float64)
    if (
        not tangent_hint_valid
        or tangent_hint_reuse_count >= 8
        or not _mc_active_ids_match_numba(
            active_ids,
            active_count,
            tangent_active_ids_hint,
            tangent_active_count_hint,
        )
    ):
        return False, tangent

    delta = strain - tangent_strain_hint
    delta_norm_sq = float(delta @ delta)
    strain_norm = math.sqrt(float(strain @ strain))
    previous_norm = math.sqrt(float(tangent_strain_hint @ tangent_strain_hint))
    max_delta = 0.15 * max(strain_norm, previous_norm, 1.0e-6)
    if delta_norm_sq > max_delta * max_delta:
        return False, tangent

    tangent[:, :] = tangent_hint
    if delta_norm_sq > 1.0e-28:
        stress_delta = stress - tangent_stress_hint
        secant_error = stress_delta - tangent @ delta
        tangent += np.outer(secant_error, delta) / delta_norm_sq

    tangent_norm_sq = 0.0
    hint_norm_sq = 0.0
    elastic_norm_sq = 0.0
    for row in range(4):
        for column in range(4):
            value = tangent[row, column]
            if not math.isfinite(value):
                return False, tangent
            tangent_norm_sq += value * value
            hint_norm_sq += tangent_hint[row, column] * tangent_hint[row, column]
            elastic_norm_sq += D4[row, column] * D4[row, column]
    norm_limit = 4.0 * max(math.sqrt(hint_norm_sq), 1.0e-6 * math.sqrt(elastic_norm_sq), 1.0)
    if math.sqrt(tangent_norm_sq) > norm_limit:
        return False, tangent
    return True, tangent


@njit(cache=True)
def _quad4_mc_stress_tangent_state_active_set_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
    active_ids_hint: np.ndarray,
    active_count_hint: int,
    tangent_hint: np.ndarray,
    tangent_strain_hint: np.ndarray,
    tangent_stress_hint: np.ndarray,
    tangent_hint_valid: bool,
    tangent_hint_reuse_count: int,
    active_set_update_enabled: bool,
    tangent_reuse_enabled: bool,
    direct_consistent_tangent_enabled: bool,
    apex_policy_code: int = 0,
) -> tuple:
    """Combined MC update with validated active-set and cone-tip regularization."""

    trial = D4 @ (strain - plastic_strain) + initial_stress

    sig_p, vecs = _quad4_mc_principal_frame_numba(trial)
    f_trial = -1.0e300
    for row in range(6):
        value = -cohesion_term - hardening * kappa
        for j in range(3):
            value += yield_coeffs[row, j] * sig_p[j]
        if value > f_trial:
            f_trial = value
    norm_sig = math.sqrt(sig_p[0] * sig_p[0] + sig_p[1] * sig_p[1] + sig_p[2] * sig_p[2])
    tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
    if f_trial <= tol:
        return (
            True,
            trial,
            D4.copy(),
            plastic_strain.copy(),
            kappa,
            0,
            0.0,
            0.0,
            0.0,
            np.full(3, -1, dtype=np.int64),
            0,
            0,
            0,
            0,
            False,
            False,
            0,
        )

    active_set_attempts = 0
    active_set_hits = 0
    regularized_active_set_hits = 0
    regularized_count = 0
    base_regularized = False
    tangent_cache_reused = False
    tangent_cache_reuse_count_next = 0
    yield_violation = 0.0
    relative_yield_violation = 0.0
    relaxed_tolerance = 0.0
    ok = False
    sig_corr_p = np.zeros(3, dtype=np.float64)
    active_ids = np.full(3, -1, dtype=np.int64)
    active_count = 0
    cache_active_ids = np.full(3, -1, dtype=np.int64)
    cache_active_count = 0
    cache_hint_locked = False
    gamma = np.zeros(3, dtype=np.float64)
    gamma_count = 0
    if active_set_update_enabled and active_count_hint > 0:
        # Cached plane IDs belong to the deterministic LAPACK principal frame
        # used by the apex fallback, so replay them in that same frame.
        sig_p, vecs = _quad4_mc_principal_frame_lapack_numba(trial)
        norm_sig = math.sqrt(
            sig_p[0] * sig_p[0]
            + sig_p[1] * sig_p[1]
            + sig_p[2] * sig_p[2]
        )
        tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
        active_set_attempts = 1
        seed_ok, exact_seed, seed_sig, seed_gamma, _seed_vals = (
            _mc_active_set_seed_numba(
                sig_p,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                D4[:3, :3],
                hardening,
                kappa,
                tol,
                active_ids_hint,
                active_count_hint,
                candidate_h,
            )
        )
        if seed_ok and exact_seed:
            ok = True
            sig_corr_p = seed_sig
            gamma = seed_gamma
            active_count = active_count_hint
            gamma_count = active_count_hint
            for slot in range(active_count_hint):
                active_ids[slot] = active_ids_hint[slot]
                cache_active_ids[slot] = active_ids_hint[slot]
            cache_active_count = active_count_hint
            cache_hint_locked = True
            active_set_hits = 1
        elif apex_policy_code == 0:
            vals_tr = (
                yield_coeffs @ sig_p
                - cohesion_term
                - hardening * kappa
            )
            (
                associated_ok,
                associated_sig,
                associated_gamma,
                associated_vals,
                associated_metric,
            ) = _mc_eval_active_candidate_numba(
                sig_p,
                vals_tr,
                yield_coeffs,
                yield_coeffs,
                cohesion_term,
                D4[:3, :3],
                hardening,
                kappa,
                active_ids_hint,
                active_count_hint,
            )
            norm_sig = math.sqrt(
                sig_p[0] * sig_p[0]
                + sig_p[1] * sig_p[1]
                + sig_p[2] * sig_p[2]
            )
            verified_tol = max(
                10.0 * tol,
                1.0e-12 * max(1.0, abs(cohesion_term), norm_sig),
            )
            if (
                associated_ok
                and associated_metric[0] <= verified_tol
                and associated_metric[1] <= verified_tol
                and associated_metric[2] <= 1.0e-12 * max(1.0, norm_sig)
            ):
                ok = True
                sig_corr_p = associated_sig
                gamma = associated_gamma
                active_count = active_count_hint
                gamma_count = active_count_hint
                for slot in range(active_count_hint):
                    active_ids[slot] = active_ids_hint[slot]
                    cache_active_ids[slot] = active_ids_hint[slot]
                cache_active_count = active_count_hint
                cache_hint_locked = True
                active_set_hits = 1
                regularized_active_set_hits = 1
                regularized_count = 1
                base_regularized = True
                yield_violation = max(0.0, associated_metric[0])
                relative_yield_violation = yield_violation / max(
                    1.0, abs(cohesion_term), norm_sig
                )
                relaxed_tolerance = verified_tol
    if not ok:
        ok, sig_corr_p, active_ids, active_count, gamma, _vals_corr = _mc_return_mapping_principal_numba(
            sig_p,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            D4[:3, :3],
            hardening,
            kappa,
            tol,
        )
        gamma_count = active_count
    if not ok:
        sig_p, vecs = _quad4_mc_principal_frame_lapack_numba(trial)
        norm_sig = math.sqrt(
            sig_p[0] * sig_p[0]
            + sig_p[1] * sig_p[1]
            + sig_p[2] * sig_p[2]
        )
        tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
        (
            ok,
            sig_corr_p,
            active_ids,
            active_count,
            gamma,
            _vals_corr,
        ) = _mc_return_mapping_principal_numba(
            sig_p,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            D4[:3, :3],
            hardening,
            kappa,
            tol,
        )
        gamma_count = active_count
    if not ok:
        (
            ok,
            sig_corr_p,
            active_ids,
            active_count,
            gamma,
            _vals_corr,
        ) = _mc_return_mapping_principal_precomputed_numba(
            sig_p,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            D4[:3, :3],
            hardening,
            kappa,
            tol,
            operator1,
            operator2,
            operator3,
            candidate_h,
        )
        gamma_count = active_count
    if not cache_hint_locked and active_count > 0:
        for slot in range(active_count):
            cache_active_ids[slot] = active_ids[slot]
        cache_active_count = active_count
    if not ok:
        (
            ok,
            sig_corr_p,
            active_ids,
            active_count,
            gamma,
            gamma_count,
            _vals_corr,
            yield_violation,
            relative_yield_violation,
            relaxed_tolerance,
        ) = (
            _mc_legacy_bounded_projection_return_mapping_numba(
                sig_p,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                D4[:3, :3],
                hardening,
                kappa,
                tol,
                sig_corr_p,
                active_ids,
                gamma,
                active_count,
            )
            if apex_policy_code == 1
            else _mc_regularized_projection_return_mapping_numba(
                sig_p,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                D4[:3, :3],
                hardening,
                kappa,
                tol,
                sig_corr_p,
                active_ids,
                gamma,
                active_count,
            )
        )
        if not ok:
            return (
                False,
                trial,
                D4.copy(),
                plastic_strain.copy(),
                kappa,
                0,
                yield_violation,
                relative_yield_violation,
                relaxed_tolerance,
                cache_active_ids,
                cache_active_count,
                active_set_attempts,
                active_set_hits,
                regularized_active_set_hits,
                base_regularized,
                False,
                0,
            )
        regularized_count = 1
        base_regularized = True
        cache_active_ids[:] = -1
        for slot in range(active_count):
            cache_active_ids[slot] = active_ids[slot]
        cache_active_count = active_count

    corrected = np.zeros(4, dtype=np.float64)
    for a in range(3):
        va0 = vecs[0, a]
        va1 = vecs[1, a]
        va2 = vecs[2, a]
        value = sig_corr_p[a]
        corrected[0] += value * va0 * va0
        corrected[1] += value * va1 * va1
        corrected[2] += value * va2 * va2
        corrected[3] += value * va0 * va1

    elastic_strain_new = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = 0.0
        for j in range(4):
            value += S4[i, j] * (corrected[j] - initial_stress[j])
        elastic_strain_new[i] = value
    plastic_strain_new = np.empty(4, dtype=np.float64)
    for i in range(4):
        plastic_strain_new[i] = strain[i] - elastic_strain_new[i]
    dgamma = 0.0
    for i in range(gamma_count):
        dgamma += gamma[i]
    kappa_new = kappa + max(dgamma, 0.0)

    if active_count <= 0:
        tangent = D4.copy()
    else:
        tangent_ok = False
        tangent = np.zeros((4, 4), dtype=np.float64)
        if (
            direct_consistent_tangent_enabled
            and base_regularized
            and apex_policy_code == 0
        ):
            tangent_ok, tangent = _mc_consistent_tangent_spectral_numba(
                sig_p,
                sig_corr_p,
                vecs,
                active_ids,
                active_count,
                yield_coeffs,
                yield_coeffs,
                D4[:3, :3],
                D4,
                hardening,
            )
        if (
            not tangent_ok
            and
            active_set_update_enabled
            and tangent_reuse_enabled
            and base_regularized
        ):
            active_set_attempts += 1
            tangent_ok, tangent = _mc_secant_tangent_update_numba(
                    strain,
                    corrected,
                    active_ids,
                    active_count,
                    D4,
                    tangent_hint,
                    tangent_strain_hint,
                    tangent_stress_hint,
                    active_ids_hint,
                    active_count_hint,
                    tangent_hint_valid,
                    tangent_hint_reuse_count,
            )
            if tangent_ok:
                tangent_cache_reused = True
                tangent_cache_reuse_count_next = tangent_hint_reuse_count + 1
                active_set_hits += 1
                regularized_active_set_hits += 1
        # Ordinary points retain their configured non-associated tangent.  The
        # apex submodel above uses its associated multisurface tangent.
        if not tangent_ok and direct_consistent_tangent_enabled and not base_regularized:
            tangent_ok, tangent = (
                _mc_consistent_tangent_spectral_precomputed_numba(
                    sig_p,
                    sig_corr_p,
                    vecs,
                    active_ids,
                    active_count,
                    yield_coeffs,
                    flow_coeffs,
                    D4[:3, :3],
                    D4,
                    operator1,
                    operator2,
                    operator3,
                    candidate_h,
                )
            )
        if not tangent_ok and direct_consistent_tangent_enabled and not base_regularized:
            tangent_ok, tangent = _mc_consistent_tangent_spectral_numba(
                sig_p,
                sig_corr_p,
                vecs,
                active_ids,
                active_count,
                yield_coeffs,
                flow_coeffs,
                D4[:3, :3],
                D4,
                hardening,
            )
        tangent_finite = tangent_ok
        if tangent_finite:
            for i in range(4):
                for j in range(4):
                    if not math.isfinite(tangent[i, j]):
                        tangent_finite = False
        if not tangent_finite and tangent_cache_reused:
            tangent_cache_reused = False
            tangent_cache_reuse_count_next = 0
            active_set_hits -= 1
            regularized_active_set_hits -= 1
        if not tangent_finite:
            (
                tangent_ok,
                tangent,
                extra_regularized_count,
                extra_yield_violation,
                extra_relative_yield_violation,
                extra_relaxed_tolerance,
                extra_active_set_attempts,
                extra_active_set_hits,
                extra_regularized_active_set_hits,
            ) = _quad4_mc_numerical_tangent_precomputed_regularized_numba(
                strain,
                corrected,
                plastic_strain,
                kappa,
                D4,
                initial_stress,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                hardening,
                operator1,
                operator2,
                operator3,
                candidate_h,
                cache_active_ids,
                cache_active_count,
                False,
                apex_policy_code,
            )
            if not tangent_ok:
                return (
                    False,
                    corrected,
                    D4.copy(),
                    plastic_strain_new,
                    kappa_new,
                    regularized_count,
                    yield_violation,
                    relative_yield_violation,
                    relaxed_tolerance,
                    active_ids,
                    active_count,
                    active_set_attempts + extra_active_set_attempts,
                    active_set_hits + extra_active_set_hits,
                    regularized_active_set_hits + extra_regularized_active_set_hits,
                    base_regularized,
                    False,
                    0,
                )
            regularized_count += extra_regularized_count
            yield_violation = max(yield_violation, extra_yield_violation)
            relative_yield_violation = max(
                relative_yield_violation,
                extra_relative_yield_violation,
            )
            relaxed_tolerance = max(relaxed_tolerance, extra_relaxed_tolerance)
            active_set_attempts += extra_active_set_attempts
            active_set_hits += extra_active_set_hits
            regularized_active_set_hits += extra_regularized_active_set_hits
    return (
        True,
        corrected,
        tangent,
        plastic_strain_new,
        kappa_new,
        regularized_count,
        yield_violation,
        relative_yield_violation,
        relaxed_tolerance,
        active_ids,
        active_count,
        active_set_attempts,
        active_set_hits,
        regularized_active_set_hits,
        base_regularized,
        tangent_cache_reused,
        tangent_cache_reuse_count_next,
    )


@njit(cache=True)
def _quad4_mc_stress_tangent_state_regularized_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, float, int, float, float, float]:
    """Compatibility entry point using the verified direct apex tangent."""

    result = _quad4_mc_stress_tangent_state_active_set_numba(
        strain,
        plastic_strain,
        kappa,
        D4,
        S4,
        initial_stress,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        hardening,
        operator1,
        operator2,
        operator3,
        candidate_h,
        np.full(3, -1, dtype=np.int64),
        0,
        np.zeros((4, 4), dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        np.zeros(4, dtype=np.float64),
        False,
        0,
        False,
        False,
        True,
    )
    return (
        result[0],
        result[1],
        result[2],
        result[3],
        result[4],
        result[5],
        result[6],
        result[7],
        result[8],
    )


@njit(cache=True)
def _quad4_mc_maybe_tension_stress_tangent_numba(
    strain: np.ndarray,
    plastic_strain: np.ndarray,
    kappa: float,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    sin_phi: float,
    Cn: np.ndarray,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[bool, np.ndarray, np.ndarray]:
    if use_tension_cutoff > 0.5:
        ok, stress = _mc_tension_cutoff_stress_numba(
            strain,
            plastic_strain,
            D4,
            initial_stress,
            sin_phi,
            cohesion_term,
            hardening,
            kappa,
            yield_coeffs,
            flow_coeffs,
            Cn,
            tensile_strength,
        )
        if not ok:
            return False, stress, D4.copy()
        tangent_ok, tangent = _mc_tension_cutoff_consistent_tangent_numba(
            strain,
            plastic_strain,
            D4,
            initial_stress,
            sin_phi,
            cohesion_term,
            hardening,
            kappa,
            yield_coeffs,
            flow_coeffs,
            Cn,
            tensile_strength,
        )
        if not tangent_ok:
            return False, stress, tangent
        return True, stress, tangent
    return _quad4_mc_stress_tangent_numba(
        strain,
        plastic_strain,
        kappa,
        D4,
        initial_stress,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        hardening,
    )

@njit(cache=True)
def _quad4_mc_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    thickness: float,
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
        for i in range(4):
            value = 0.0
            for j in range(8):
                value += B[i, j] * ue[j]
            strain[i] = value
        ok, stress, tangent = _quad4_mc_stress_tangent_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
        )
        if not ok:
            return ke, fe, min_det, False
        dV = det * thickness
        _quad4_add_btcb_numba(ke, B, tangent, dV)
        for i in range(8):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    return ke, fe, min_det, True

@njit(cache=True)
def _quad4_mc_internal_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    thickness: float,
) -> tuple[np.ndarray, float, bool]:
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
            return fe, min_det, False
        strain = np.zeros(4, dtype=np.float64)
        for i in range(4):
            value = 0.0
            for j in range(8):
                value += B[i, j] * ue[j]
            strain[i] = value
        ok, stress, _plastic, _yield_value, _p, _q, _plastic_strain_new, _kappa_new, _active_ids, _active_count = _quad4_mc_post_update_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
        )
        if not ok:
            return fe, min_det, False
        dV = det * thickness
        for i in range(8):
            value = 0.0
            for j in range(4):
                value += B[j, i] * stress[j]
            fe[i] += value * dV
    return fe, min_det, True

@njit(cache=True)
def _quad8_mc_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    thickness: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float, bool]:
    data = np.zeros((9, 31), dtype=np.float64)
    min_det = 1.0e300
    ok_all = True
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        N, _grad, det_shape = _quad8_shape_grad_numba(coords, xi, eta)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det_shape < min_det:
            min_det = det_shape
        if det <= 0.0 or det_shape <= 0.0:
            return data, min_det, ok_all
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
        ok, stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new, active_ids, active_count = _quad4_mc_maybe_tension_post_update_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
            tensile_strength,
            use_tension_cutoff,
        )
        if not ok:
            ok_all = False
            return data, min_det, ok_all
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
        data[gp, 27] = float(active_count)
        for i in range(3):
            data[gp, 28 + i] = float(active_ids[i])
    return data, min_det, ok_all

@njit(cache=True)
def _quad8_mc_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    Pvol: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    thickness: float,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float, bool]:
    data = np.zeros((9, 31), dtype=np.float64)
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
            return data, min_det, True
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
        return data, min_det, False

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
        ok, stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new, active_ids, active_count = _quad4_mc_maybe_tension_post_update_numba(
            strain_eff,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
            tensile_strength,
            use_tension_cutoff,
        )
        if not ok:
            return data, min_det, False
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
        data[gp, 27] = float(active_count)
        for i in range(3):
            data[gp, 28 + i] = float(active_ids[i])
    return data, min_det, True

@njit(cache=True)
def _quad4_mc_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    thickness: float,
) -> tuple[np.ndarray, float, bool]:
    data = np.zeros((4, 31), dtype=np.float64)
    min_det = 1.0e300
    ok_all = True
    a = 1.0 / math.sqrt(3.0)
    for gp in range(4):
        xi = -a if gp == 0 or gp == 3 else a
        eta = -a if gp == 0 or gp == 1 else a
        B, det = _quad4_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return data, min_det, ok_all
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
        ok, stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new, active_ids, active_count = _quad4_mc_post_update_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
        )
        if not ok:
            ok_all = False
            return data, min_det, ok_all
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
        data[gp, 27] = float(active_count)
        for i in range(3):
            data[gp, 28 + i] = float(active_ids[i])
    return data, min_det, ok_all

@njit(cache=True)
def _quad4_mc_bbar_post_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    Pvol: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    thickness: float,
) -> tuple[np.ndarray, float, bool]:
    data = np.zeros((4, 31), dtype=np.float64)
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
            return data, min_det, True
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
        return data, min_det, False

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
        ok, stress, plastic, yield_value, p, q, plastic_strain_new, kappa_new, active_ids, active_count = _quad4_mc_post_update_numba(
            strain_eff,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
        )
        if not ok:
            return data, min_det, False
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
        data[gp, 27] = float(active_count)
        for i in range(3):
            data[gp, 28 + i] = float(active_ids[i])
    return data, min_det, True

@njit(cache=True)
def _quad8_mc_tangent_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    thickness: float,
    mode_code: int,
    sin_phi: float,
    Cn: np.ndarray,
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
            ok, stress, tangent = _quad4_mc_maybe_tension_stress_tangent_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                initial_stress,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                hardening,
                sin_phi,
                Cn,
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
            ok, stress, tangent = _quad4_mc_maybe_tension_stress_tangent_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                initial_stress,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                hardening,
                sin_phi,
                Cn,
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
            ok, stress, tangent = _quad4_mc_maybe_tension_stress_tangent_numba(
                strain,
                plastic_strains[state_index],
                kappas[state_index],
                D4,
                initial_stress,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                hardening,
                sin_phi,
                Cn,
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
        ok, stress, tangent = _quad4_mc_maybe_tension_stress_tangent_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
            sin_phi,
            Cn,
            tensile_strength,
            use_tension_cutoff,
        )
        if not ok:
            return ke, fe, min_det, volume, False
        dV = dV_cache[gp]
        _quad8_add_btcb_numba(ke, B_eff, tangent, dV)
        _quad8_add_btstress_numba(fe, B_eff, stress, dV)
    return ke, fe, min_det, volume, True

@njit(cache=True)
def _quad8_mc_internal_force_numba(
    coords: np.ndarray,
    ue: np.ndarray,
    D4: np.ndarray,
    S4: np.ndarray,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    Pvol: np.ndarray,
    Idev: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    hardening: float,
    thickness: float,
    mode_code: int,
    tensile_strength: float,
    use_tension_cutoff: float,
) -> tuple[np.ndarray, float, float, bool]:
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
                return fe, min_det, volume, False
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            ok, stress, _plastic, _yield_value, _p, _q, _plastic_strain_new, _kappa_new, _active_ids, _active_count = _quad4_mc_maybe_tension_post_update_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                S4,
                initial_stress,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                hardening,
                tensile_strength,
                use_tension_cutoff,
            )
            if not ok:
                return fe, min_det, volume, False
            _quad8_add_btstress_numba(fe, B, stress, det * weight * thickness)
        return fe, min_det, volume, True

    if mode_code == _QUAD4_MODE_SRI:
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return fe, min_det, volume, False
            Bdev = _quad8_project_b_numba(Idev, B)
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            ok, stress, _plastic, _yield_value, _p, _q, _plastic_strain_new, _kappa_new, _active_ids, _active_count = _quad4_mc_maybe_tension_post_update_numba(
                strain,
                plastic_strains[gp],
                kappas[gp],
                D4,
                S4,
                initial_stress,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                hardening,
                tensile_strength,
                use_tension_cutoff,
            )
            if not ok:
                return fe, min_det, volume, False
            _quad8_add_btstress_numba(fe, Bdev, stress, det * weight * thickness)
        for gp in range(4):
            xi, eta, weight = _quad8_gp_reduced(gp)
            B, det = _quad8_b_det_numba(coords, xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                return fe, min_det, volume, False
            Bv = _quad8_project_b_numba(Pvol, B)
            strain = np.zeros(4, dtype=np.float64)
            for i in range(4):
                value = 0.0
                for j in range(16):
                    value += B[i, j] * ue[j]
                strain[i] = value
            state_index = 9 + gp
            ok, stress, _plastic, _yield_value, _p, _q, _plastic_strain_new, _kappa_new, _active_ids, _active_count = _quad4_mc_maybe_tension_post_update_numba(
                strain,
                plastic_strains[state_index],
                kappas[state_index],
                D4,
                S4,
                initial_stress,
                yield_coeffs,
                flow_coeffs,
                cohesion_term,
                hardening,
                tensile_strength,
                use_tension_cutoff,
            )
            if not ok:
                return fe, min_det, volume, False
            _quad8_add_btstress_numba(fe, Bv, stress, det * weight * thickness)
        return fe, min_det, volume, True

    B_cache = np.zeros((9, 4, 16), dtype=np.float64)
    dV_cache = np.zeros(9, dtype=np.float64)
    Bv_acc = np.zeros((4, 16), dtype=np.float64)
    for gp in range(9):
        xi, eta, weight = _quad8_gp_full(gp)
        B, det = _quad8_b_det_numba(coords, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            return fe, min_det, volume, False
        dV = det * weight * thickness
        dV_cache[gp] = dV
        volume += dV
        Bv = _quad8_project_b_numba(Pvol, B)
        for r in range(4):
            for c in range(16):
                B_cache[gp, r, c] = B[r, c]
                Bv_acc[r, c] += Bv[r, c] * dV
    if volume <= 0.0:
        return fe, min_det, volume, False
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
        ok, stress, _plastic, _yield_value, _p, _q, _plastic_strain_new, _kappa_new, _active_ids, _active_count = _quad4_mc_maybe_tension_post_update_numba(
            strain,
            plastic_strains[gp],
            kappas[gp],
            D4,
            S4,
            initial_stress,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
            tensile_strength,
            use_tension_cutoff,
        )
        if not ok:
            return fe, min_det, volume, False
        _quad8_add_btstress_numba(fe, B_eff, stress, dV_cache[gp])
    return fe, min_det, volume, True

def _quad4_mc_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    strength_factor: float,
) -> np.ndarray | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (4, 4) or state_kappas.shape != (4,):
        raise FEM2DError("QUAD4 plastic state arrays must have shapes (4, 4) and (4,)")
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    data, min_det, ok = _quad4_mc_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
        float(2.0 * c * math.cos(phi)),
        float(material.hardening),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data if bool(ok) else None

def _quad4_mc_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    strength_factor: float,
) -> np.ndarray | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (4, 4) or state_kappas.shape != (4,):
        raise FEM2DError("QUAD4 plastic state arrays must have shapes (4, 4) and (4,)")
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    data, min_det, ok = _quad4_mc_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
        float(2.0 * c * math.cos(phi)),
        float(material.hardening),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return data if bool(ok) else None

def _quad8_mc_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    strength_factor: float,
) -> np.ndarray | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (9, 4) or state_kappas.shape != (9,):
        raise FEM2DError("QUAD8 plastic state arrays must have shapes (9, 4) and (9,)")
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    data, min_det, ok = _quad8_mc_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
        float(2.0 * c * math.cos(phi)),
        float(material.hardening),
        float(material.thickness),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return data if bool(ok) else None

def _quad8_mc_bbar_post_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    strength_factor: float,
) -> np.ndarray | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (9, 4) or state_kappas.shape != (9,):
        raise FEM2DError("QUAD8 plastic state arrays must have shapes (9, 4) and (9,)")
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    data, min_det, ok = _quad8_mc_bbar_post_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(material.volumetric_projector, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
        float(2.0 * c * math.cos(phi)),
        float(material.hardening),
        float(material.thickness),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    return data if bool(ok) else None

def _quad4_mc_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    strength_factor: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (4, 4) or state_kappas.shape != (4,):
        raise FEM2DError("QUAD4 plastic state arrays must have shapes (4, 4) and (4,)")
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    ke, fe, min_det, ok = _quad4_mc_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
        float(2.0 * c * math.cos(phi)),
        float(material.hardening),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return (ke, fe) if bool(ok) else None

def _quad4_mc_internal_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    strength_factor: float,
) -> np.ndarray | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (4, 4) or state_kappas.shape != (4,):
        raise FEM2DError("QUAD4 plastic state arrays must have shapes (4, 4) and (4,)")
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    fe, min_det, ok = _quad4_mc_internal_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
        float(2.0 * c * math.cos(phi)),
        float(material.hardening),
        float(material.thickness),
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4: detJ must be positive, got {min_det:.6e}")
    return fe if bool(ok) else None

def _quad8_mc_tangent_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    mode: str,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    strength_factor: float,
) -> tuple[np.ndarray, np.ndarray] | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    mode_code = _quad4_mode_code(mode)
    expected_points = 13 if mode_code == _QUAD4_MODE_SRI else 9
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (expected_points, 4) or state_kappas.shape != (expected_points,):
        raise FEM2DError(f"QUAD8 {mode}: plastic state arrays must have {expected_points} integration points")
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    ke, fe, min_det, volume, ok = _quad8_mc_tangent_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        Pvol,
        Idev,
        np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
        float(2.0 * c * math.cos(phi)),
        float(material.hardening),
        float(material.thickness),
        mode_code,
        float(math.sin(phi)),
        np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return (ke, fe) if bool(ok) else None

def _quad8_mc_internal_force_fast(
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    mode: str,
    *,
    initial_stress: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    strength_factor: float,
) -> np.ndarray | None:
    initial = np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    mode_code = _quad4_mode_code(mode)
    expected_points = 13 if mode_code == _QUAD4_MODE_SRI else 9
    state_strains = np.asarray(plastic_strains, dtype=float)
    state_kappas = np.asarray(kappas, dtype=float)
    if state_strains.shape != (expected_points, 4) or state_kappas.shape != (expected_points,):
        raise FEM2DError(f"QUAD8 {mode}: plastic state arrays must have {expected_points} integration points")
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    Pvol = np.ascontiguousarray(material.volumetric_projector, dtype=np.float64)
    Idev = np.ascontiguousarray(np.eye(4, dtype=float) - Pvol, dtype=np.float64)
    fe, min_det, volume, ok = _quad8_mc_internal_force_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(ue, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(np.linalg.inv(material.D4), dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        np.ascontiguousarray(state_strains, dtype=np.float64),
        np.ascontiguousarray(state_kappas, dtype=np.float64),
        Pvol,
        Idev,
        np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
        float(2.0 * c * math.cos(phi)),
        float(material.hardening),
        float(material.thickness),
        mode_code,
        float(material.tensile_strength),
        1.0 if bool(material.tension_cutoff) and math.isfinite(float(material.tensile_strength)) else 0.0,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8: detJ must be positive, got {min_det:.6e}")
    if mode_code == _QUAD4_MODE_BBAR and volume <= 0.0:
        raise FEM2DError("QUAD8: non-positive element measure")
    return fe if bool(ok) else None

__all__ = list(MOHR_COULOMB_ELEMENT_KERNEL_FUNCTIONS)
