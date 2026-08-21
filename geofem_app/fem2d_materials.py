"""Plane-strain material updates and consistent tangents."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from itertools import permutations
import math
import threading
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, PlasticState2D, PlasticStateView2D, StressUpdate2D, njit, _symmetrize


PlasticStateLike = PlasticState2D | PlasticStateView2D


_MC_FALLBACK_LOCAL = threading.local()
_MC_FALLBACK_SAMPLE_LIMIT = 32
_MC_REGULARIZATION_METHOD = "associated_multisurface_apex"
_MC_LEGACY_REGULARIZATION_METHOD = "bounded_sequential_cone_tip"


def reset_mohr_coulomb_fallback_telemetry() -> None:
    """Reset fallback counters for the current SRM trial thread."""

    _MC_FALLBACK_LOCAL.data = {
        "numba_to_python_count": 0,
        "numba_regularized_projection_count": 0,
        "regularized_projection_count": 0,
        "apex_regularization_count": 0,
        "associated_apex_projection_count": 0,
        "legacy_bounded_projection_count": 0,
        "regularized_projection_above_relaxed_tolerance_count": 0,
        "active_set_update_attempt_count": 0,
        "active_set_update_hit_count": 0,
        "active_set_regularized_update_hit_count": 0,
        "active_set_full_scan_avoided_count": 0,
        "max_yield_violation": 0.0,
        "max_relative_yield_violation": 0.0,
        "samples": [],
    }


def mohr_coulomb_fallback_telemetry() -> dict[str, Any]:
    raw = getattr(_MC_FALLBACK_LOCAL, "data", None)
    if not isinstance(raw, dict):
        reset_mohr_coulomb_fallback_telemetry()
        raw = _MC_FALLBACK_LOCAL.data
    associated_count = int(raw.get("associated_apex_projection_count", 0) or 0)
    legacy_count = int(raw.get("legacy_bounded_projection_count", 0) or 0)
    above_tolerance_count = int(
        raw.get("regularized_projection_above_relaxed_tolerance_count", 0) or 0
    )
    configured_policy_verified = legacy_count == 0 and above_tolerance_count == 0
    if associated_count and legacy_count:
        regularization_method = "mixed_apex_projection"
    elif legacy_count:
        regularization_method = _MC_LEGACY_REGULARIZATION_METHOD
    else:
        regularization_method = _MC_REGULARIZATION_METHOD
    return {
        "numba_to_python_count": int(raw.get("numba_to_python_count", 0) or 0),
        "numba_regularized_projection_count": int(raw.get("numba_regularized_projection_count", 0) or 0),
        "regularized_projection_count": int(raw.get("regularized_projection_count", 0) or 0),
        "apex_regularization_count": int(raw.get("apex_regularization_count", 0) or 0),
        "associated_apex_projection_count": associated_count,
        "legacy_bounded_projection_count": legacy_count,
        "regularization_method": regularization_method,
        "configured_apex_policy_verified": configured_policy_verified,
        "constitutive_model_fidelity": configured_policy_verified,
        "flow_rule_verified": configured_policy_verified,
        "base_nonassociated_flow_rule_verified": associated_count == 0 and legacy_count == 0,
        "regularized_projection_above_relaxed_tolerance_count": above_tolerance_count,
        "active_set_update_attempt_count": int(raw.get("active_set_update_attempt_count", 0) or 0),
        "active_set_update_hit_count": int(raw.get("active_set_update_hit_count", 0) or 0),
        "active_set_regularized_update_hit_count": int(
            raw.get("active_set_regularized_update_hit_count", 0) or 0
        ),
        "active_set_full_scan_avoided_count": int(
            raw.get("active_set_full_scan_avoided_count", 0) or 0
        ),
        "max_yield_violation": float(raw.get("max_yield_violation", 0.0) or 0.0),
        "max_relative_yield_violation": float(raw.get("max_relative_yield_violation", 0.0) or 0.0),
        "samples": [dict(item) for item in raw.get("samples", []) if isinstance(item, Mapping)],
    }


def mohr_coulomb_adaptive_tangent_counters() -> tuple[int, int, int, int]:
    """Return lightweight counters used by the Newton tangent controller."""

    raw = getattr(_MC_FALLBACK_LOCAL, "data", None)
    if not isinstance(raw, dict):
        reset_mohr_coulomb_fallback_telemetry()
        raw = _MC_FALLBACK_LOCAL.data
    return (
        int(raw.get("regularized_projection_count", 0) or 0),
        int(raw.get("active_set_update_attempt_count", 0) or 0),
        int(raw.get("active_set_update_hit_count", 0) or 0),
        int(raw.get("active_set_regularized_update_hit_count", 0) or 0),
    )


def _record_mohr_coulomb_fallback(
    event: str,
    *,
    diagnostic_context: tuple[Any, int] | None = None,
    yield_violation: float = 0.0,
    relative_yield_violation: float = 0.0,
    relaxed_tolerance: float = 0.0,
    regularization_method: str = _MC_REGULARIZATION_METHOD,
    configured_apex_policy_verified: bool = True,
) -> None:
    raw = getattr(_MC_FALLBACK_LOCAL, "data", None)
    if not isinstance(raw, dict):
        reset_mohr_coulomb_fallback_telemetry()
        raw = _MC_FALLBACK_LOCAL.data
    if event == "numba_to_python":
        raw["numba_to_python_count"] += 1
        return
    raw["regularized_projection_count"] += 1
    raw["apex_regularization_count"] += 1
    if regularization_method == _MC_LEGACY_REGULARIZATION_METHOD:
        raw["legacy_bounded_projection_count"] += 1
    else:
        raw["associated_apex_projection_count"] += 1
    if yield_violation > relaxed_tolerance:
        raw["regularized_projection_above_relaxed_tolerance_count"] += 1
    raw["max_yield_violation"] = max(float(raw["max_yield_violation"]), float(yield_violation))
    raw["max_relative_yield_violation"] = max(
        float(raw["max_relative_yield_violation"]), float(relative_yield_violation)
    )
    element_id = ""
    integration_point: int | str = ""
    if diagnostic_context is not None:
        element_id = str(diagnostic_context[0])
        integration_point = int(diagnostic_context[1])
    sample = {
        "element_id": element_id,
        "integration_point": integration_point,
        "yield_violation": float(yield_violation),
        "relative_yield_violation": float(relative_yield_violation),
        "relaxed_tolerance": float(relaxed_tolerance),
        "regularization_method": regularization_method,
        "configured_apex_policy_verified": bool(configured_apex_policy_verified),
        "base_nonassociated_flow_rule_verified": False,
    }
    samples = raw["samples"]
    if len(samples) < _MC_FALLBACK_SAMPLE_LIMIT:
        samples.append(sample)
        return
    weakest = min(
        range(len(samples)),
        key=lambda index: float(samples[index].get("relative_yield_violation", 0.0) or 0.0),
    )
    if relative_yield_violation > float(
        samples[weakest].get("relative_yield_violation", 0.0) or 0.0
    ):
        samples[weakest] = sample


def record_mohr_coulomb_numba_regularized_batch(
    element_ids: tuple[str, ...],
    regularized_counts: np.ndarray,
    yield_violations: np.ndarray,
    relative_yield_violations: np.ndarray,
    relaxed_tolerances: np.ndarray,
    status_flags: np.ndarray,
    *,
    apex_policy: str = "associated_multisurface",
) -> int:
    """Merge Numba batch projection telemetry without per-event Python calls."""

    counts = np.asarray(regularized_counts, dtype=np.int64)
    if counts.ndim != 2 or counts.size == 0:
        return 0
    statuses = np.asarray(status_flags, dtype=np.int64).reshape(-1)
    if statuses.size != counts.shape[0]:
        return 0
    accepted = statuses == 0
    mask = (counts > 0) & accepted[:, None]
    if not np.any(mask):
        return 0

    weighted_counts = np.where(mask, counts, 0)
    total = int(np.sum(weighted_counts, dtype=np.int64))
    violations = np.asarray(yield_violations, dtype=float)
    relative = np.asarray(relative_yield_violations, dtype=float)
    relaxed = np.asarray(relaxed_tolerances, dtype=float)
    if violations.shape != counts.shape or relative.shape != counts.shape or relaxed.shape != counts.shape:
        return 0

    raw = getattr(_MC_FALLBACK_LOCAL, "data", None)
    if not isinstance(raw, dict):
        reset_mohr_coulomb_fallback_telemetry()
        raw = _MC_FALLBACK_LOCAL.data
    raw["regularized_projection_count"] += total
    raw["numba_regularized_projection_count"] += total
    raw["apex_regularization_count"] += total
    legacy_bounded = apex_policy == "legacy_bounded"
    regularization_method = (
        _MC_LEGACY_REGULARIZATION_METHOD
        if legacy_bounded
        else _MC_REGULARIZATION_METHOD
    )
    if legacy_bounded:
        raw["legacy_bounded_projection_count"] += total
    else:
        raw["associated_apex_projection_count"] += total
    above_mask = mask & (violations > relaxed)
    raw["regularized_projection_above_relaxed_tolerance_count"] += int(
        np.sum(np.where(above_mask, counts, 0), dtype=np.int64)
    )
    raw["max_yield_violation"] = max(
        float(raw["max_yield_violation"]),
        float(np.max(violations[mask])),
    )
    raw["max_relative_yield_violation"] = max(
        float(raw["max_relative_yield_violation"]),
        float(np.max(relative[mask])),
    )

    samples = raw["samples"]
    remaining = max(_MC_FALLBACK_SAMPLE_LIMIT - len(samples), 0)
    if remaining:
        for element_index, point_index in np.argwhere(mask)[:remaining]:
            samples.append(
                {
                    "element_id": str(element_ids[int(element_index)]),
                    "integration_point": int(point_index),
                    "yield_violation": float(violations[element_index, point_index]),
                    "relative_yield_violation": float(relative[element_index, point_index]),
                    "relaxed_tolerance": float(relaxed[element_index, point_index]),
                    "batch_occurrences": int(counts[element_index, point_index]),
                    "projection_backend": "numba_batch",
                    "regularization_method": regularization_method,
                    "configured_apex_policy_verified": not legacy_bounded,
                    "base_nonassociated_flow_rule_verified": False,
                }
            )
    elif samples:
        ranked_relative = np.where(mask, relative, -np.inf)
        flat_index = int(np.argmax(ranked_relative))
        element_index, point_index = np.unravel_index(flat_index, counts.shape)
        candidate_relative = float(relative[element_index, point_index])
        weakest = min(
            range(len(samples)),
            key=lambda index: float(samples[index].get("relative_yield_violation", 0.0) or 0.0),
        )
        if candidate_relative > float(
            samples[weakest].get("relative_yield_violation", 0.0) or 0.0
        ):
            samples[weakest] = {
                "element_id": str(element_ids[int(element_index)]),
                "integration_point": int(point_index),
                "yield_violation": float(violations[element_index, point_index]),
                "relative_yield_violation": candidate_relative,
                "relaxed_tolerance": float(relaxed[element_index, point_index]),
                "batch_occurrences": int(counts[element_index, point_index]),
                "projection_backend": "numba_batch",
                "regularization_method": regularization_method,
                "configured_apex_policy_verified": not legacy_bounded,
                "base_nonassociated_flow_rule_verified": False,
            }
    return total


def record_mohr_coulomb_active_set_batch(
    attempt_counts: np.ndarray,
    hit_counts: np.ndarray,
    regularized_hit_counts: np.ndarray,
    status_flags: np.ndarray,
) -> dict[str, int]:
    """Merge active-set update counters once per Numba element batch."""

    attempts = np.asarray(attempt_counts, dtype=np.int64)
    hits = np.asarray(hit_counts, dtype=np.int64)
    regularized_hits = np.asarray(regularized_hit_counts, dtype=np.int64)
    statuses = np.asarray(status_flags, dtype=np.int64).reshape(-1)
    if (
        attempts.ndim != 2
        or attempts.shape != hits.shape
        or attempts.shape != regularized_hits.shape
        or attempts.shape[0] != statuses.size
    ):
        return {
            "attempt_count": 0,
            "hit_count": 0,
            "regularized_hit_count": 0,
            "full_scan_avoided_count": 0,
        }
    accepted = statuses == 0
    attempt_total = int(np.sum(np.where(accepted[:, None], attempts, 0), dtype=np.int64))
    hit_total = int(np.sum(np.where(accepted[:, None], hits, 0), dtype=np.int64))
    regularized_total = int(
        np.sum(np.where(accepted[:, None], regularized_hits, 0), dtype=np.int64)
    )
    full_scan_avoided_total = hit_total + 3 * regularized_total
    raw = getattr(_MC_FALLBACK_LOCAL, "data", None)
    if not isinstance(raw, dict):
        reset_mohr_coulomb_fallback_telemetry()
        raw = _MC_FALLBACK_LOCAL.data
    raw["active_set_update_attempt_count"] += attempt_total
    raw["active_set_update_hit_count"] += hit_total
    raw["active_set_regularized_update_hit_count"] += regularized_total
    raw["active_set_full_scan_avoided_count"] += full_scan_avoided_total
    return {
        "attempt_count": attempt_total,
        "hit_count": hit_total,
        "regularized_hit_count": regularized_total,
        "full_scan_avoided_count": full_scan_avoided_total,
    }


@njit(cache=True)
def _principal_stresses_plane_strain_numba(stress4: np.ndarray) -> np.ndarray:
    sx = stress4[0]
    sy = stress4[1]
    sz = stress4[2]
    txy = stress4[3]
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
    return values


@njit(cache=True)
def _tension_cutoff_plane_strain_numba(stress4: np.ndarray, tensile_strength: float) -> tuple[np.ndarray, bool, float]:
    sx = stress4[0]
    sy = stress4[1]
    sz = stress4[2]
    txy = stress4[3]
    mean = 0.5 * (sx + sy)
    radius = math.sqrt(0.25 * (sx - sy) * (sx - sy) + txy * txy)
    lmax = mean + radius
    lmin = mean - radius
    max_principal = max(lmax, lmin, sz)
    excess = max_principal - tensile_strength
    tol = max(1.0e-10, 1.0e-10 * max(abs(tensile_strength), 1.0))
    updated = np.empty(4, dtype=np.float64)
    updated[0] = sx
    updated[1] = sy
    updated[2] = sz
    updated[3] = txy
    if excess <= tol:
        return updated, False, excess

    cmax = min(lmax, tensile_strength)
    cmin = min(lmin, tensile_strength)
    if radius <= 1.0e-30:
        updated[0] = cmax
        updated[1] = cmax
        updated[3] = 0.0
    else:
        denom = lmax - lmin
        pxx = (sx - lmin) / denom
        pyy = (sy - lmin) / denom
        pxy = txy / denom
        updated[0] = cmax * pxx + cmin * (1.0 - pxx)
        updated[1] = cmax * pyy + cmin * (1.0 - pyy)
        updated[3] = (cmax - cmin) * pxy
    updated[2] = min(sz, tensile_strength)
    return updated, True, excess


@njit(cache=True)
def _elastic_tension_cutoff_numerical_tangent_numba(
    strain4: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    tensile_strength: float,
) -> np.ndarray:
    norm = 0.0
    for i in range(4):
        norm += strain4[i] * strain4[i]
    delta = 1.0e-8 * max(1.0, math.sqrt(norm))
    trial = np.zeros(4, dtype=np.float64)
    for i in range(4):
        value = initial_stress[i]
        for j in range(4):
            value += D4[i, j] * strain4[j]
        trial[i] = value
    base, _clipped, _excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
    tangent = np.zeros((4, 4), dtype=np.float64)
    for col in range(4):
        stress = trial.copy()
        for row in range(4):
            stress[row] += D4[row, col] * delta
        plus, _plus_clipped, _plus_excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
        for row in range(4):
            tangent[row, col] = (plus[row] - base[row]) / delta
    return tangent


@njit(cache=True)
def _j2dp_tension_cutoff_stress_numba(
    strain4: np.ndarray,
    plastic_strain: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    kappa: float,
    tensile_strength: float,
) -> np.ndarray:
    trial = np.empty(4, dtype=np.float64)
    for row in range(4):
        value = initial_stress[row]
        for col in range(4):
            value += D4[row, col] * (strain4[col] - plastic_strain[col])
        trial[row] = value

    mean = (trial[0] + trial[1] + trial[2]) / 3.0
    dev0 = trial[0] - mean
    dev1 = trial[1] - mean
    dev2 = trial[2] - mean
    dev3 = trial[3]
    j2 = 0.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + dev3 * dev3
    q = math.sqrt(max(3.0 * j2, 0.0))
    p = -mean
    yield_value = q - alpha * p - cohesion_term - hardening * kappa
    tol_scale = max(q, abs(alpha * p) + cohesion_term)
    tol_scale = max(tol_scale, 1.0)
    if yield_value <= max(1.0e-10, 1.0e-10 * tol_scale):
        corrected, _clipped, _excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
        return corrected

    eps = 2.220446049250313e-16
    denom = max(3.0 * shear_mu + hardening, eps)
    dgamma = max(yield_value / denom, 0.0)
    kappa_new = kappa + dgamma
    q_limit = max(alpha * p + cohesion_term + hardening * kappa_new, 0.0)
    scale = 0.0
    if q > eps:
        scale = min(q_limit / q, 1.0)
    corrected = np.empty(4, dtype=np.float64)
    corrected[0] = mean + scale * dev0
    corrected[1] = mean + scale * dev1
    corrected[2] = mean + scale * dev2
    corrected[3] = scale * dev3
    corrected, _clipped, _excess = _tension_cutoff_plane_strain_numba(corrected, tensile_strength)
    return corrected


@njit(cache=True)
def _j2dp_tension_cutoff_numerical_tangent_numba(
    strain4: np.ndarray,
    plastic_strain: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    alpha: float,
    cohesion_term: float,
    hardening: float,
    shear_mu: float,
    kappa: float,
    tensile_strength: float,
) -> np.ndarray:
    norm = 0.0
    for i in range(4):
        norm += strain4[i] * strain4[i]
    delta = 1.0e-8 * max(1.0, math.sqrt(norm))
    base = _j2dp_tension_cutoff_stress_numba(
        strain4,
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
    tangent = np.zeros((4, 4), dtype=np.float64)
    for col in range(4):
        perturbed = strain4.copy()
        perturbed[col] += delta
        plus = _j2dp_tension_cutoff_stress_numba(
            perturbed,
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
        for row in range(4):
            tangent[row, col] = (plus[row] - base[row]) / delta
    return tangent


@njit(cache=True)
def _stress4_eigh_plane_strain_numba(stress4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sx = stress4[0]
    sy = stress4[1]
    sz = stress4[2]
    txy = stress4[3]
    mean = 0.5 * (sx + sy)
    radius = math.sqrt(0.25 * (sx - sy) * (sx - sy) + txy * txy)
    lp = mean + radius
    lm = mean - radius
    theta = 0.5 * math.atan2(2.0 * txy, sx - sy)
    c = math.cos(theta)
    s = math.sin(theta)
    values = np.empty(3, dtype=np.float64)
    vecs = np.empty((3, 3), dtype=np.float64)
    raw_values = np.empty(3, dtype=np.float64)
    raw_vecs = np.empty((3, 3), dtype=np.float64)
    raw_values[0] = lm
    raw_vecs[0, 0] = -s
    raw_vecs[1, 0] = c
    raw_vecs[2, 0] = 0.0
    raw_values[1] = lp
    raw_vecs[0, 1] = c
    raw_vecs[1, 1] = s
    raw_vecs[2, 1] = 0.0
    raw_values[2] = sz
    raw_vecs[0, 2] = 0.0
    raw_vecs[1, 2] = 0.0
    raw_vecs[2, 2] = 1.0
    used = np.zeros(3, dtype=np.bool_)
    for out_col in range(3):
        best = -1
        best_value = 1.0e300
        for candidate in range(3):
            if not used[candidate] and raw_values[candidate] < best_value:
                best = candidate
                best_value = raw_values[candidate]
        used[best] = True
        values[out_col] = raw_values[best]
        for row in range(3):
            vecs[row, out_col] = raw_vecs[row, best]
    return values, vecs


@njit(cache=True)
def _mc_tension_cutoff_stress_numba(
    strain4: np.ndarray,
    plastic_strain: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    sin_phi: float,
    cohesion_term: float,
    hardening: float,
    kappa: float,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    Cn: np.ndarray,
    tensile_strength: float,
) -> tuple[bool, np.ndarray]:
    trial = np.empty(4, dtype=np.float64)
    for row in range(4):
        value = initial_stress[row]
        for col in range(4):
            value += D4[row, col] * (strain4[col] - plastic_strain[col])
        trial[row] = value
    f_trial, tol_fast = _mc_trial_yield_numba(trial, sin_phi, cohesion_term, hardening, kappa)
    if f_trial <= tol_fast:
        corrected, _clipped, _excess = _tension_cutoff_plane_strain_numba(trial, tensile_strength)
        return True, corrected

    sig_tr_p, vecs = _stress4_eigh_plane_strain_numba(trial)
    norm_sig = math.sqrt(sig_tr_p[0] * sig_tr_p[0] + sig_tr_p[1] * sig_tr_p[1] + sig_tr_p[2] * sig_tr_p[2])
    tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
    ok, sig_corr_p, _active_ids, _active_count, _gamma, _vals_corr = _mc_return_mapping_principal_numba(
        sig_tr_p,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        Cn,
        hardening,
        kappa,
        tol,
    )
    stress = np.zeros(4, dtype=np.float64)
    if not ok:
        return False, stress
    corrected_tensor = np.zeros((3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            value = 0.0
            for i in range(3):
                value += vecs[a, i] * sig_corr_p[i] * vecs[b, i]
            corrected_tensor[a, b] = value
    stress[0] = corrected_tensor[0, 0]
    stress[1] = corrected_tensor[1, 1]
    stress[2] = corrected_tensor[2, 2]
    stress[3] = corrected_tensor[0, 1]
    stress, _clipped, _excess = _tension_cutoff_plane_strain_numba(stress, tensile_strength)
    return True, stress


@njit(cache=True)
def _mc_tension_cutoff_numerical_tangent_numba(
    strain4: np.ndarray,
    plastic_strain: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    sin_phi: float,
    cohesion_term: float,
    hardening: float,
    kappa: float,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    Cn: np.ndarray,
    tensile_strength: float,
) -> tuple[bool, np.ndarray]:
    tangent = np.zeros((4, 4), dtype=np.float64)
    norm = 0.0
    for i in range(4):
        norm += strain4[i] * strain4[i]
    delta = 1.0e-8 * max(1.0, math.sqrt(norm))
    ok, base = _mc_tension_cutoff_stress_numba(
        strain4,
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
        return False, tangent
    for col in range(4):
        perturbed = strain4.copy()
        perturbed[col] += delta
        ok_plus, plus = _mc_tension_cutoff_stress_numba(
            perturbed,
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
        if not ok_plus:
            return False, tangent
        for row in range(4):
            tangent[row, col] = (plus[row] - base[row]) / delta
    return True, tangent


@njit(cache=True)
def _tension_cutoff_projection_jacobian_numba(stress4: np.ndarray, tensile_strength: float) -> tuple[bool, np.ndarray, bool]:
    jac = np.zeros((4, 4), dtype=np.float64)
    values, vecs = _stress4_eigh_plane_strain_numba(stress4)
    tol = max(1.0e-10, 1.0e-10 * max(abs(tensile_strength), 1.0))
    clipped_values = np.empty(3, dtype=np.float64)
    slopes = np.empty(3, dtype=np.float64)
    clipped = False
    for i in range(3):
        distance = values[i] - tensile_strength
        if distance > tol:
            clipped_values[i] = tensile_strength
            slopes[i] = 0.0
            clipped = True
        elif distance < -tol:
            clipped_values[i] = values[i]
            slopes[i] = 1.0
        else:
            return False, jac, True
    if not clipped:
        for i in range(4):
            jac[i, i] = 1.0
        return True, jac, False

    divided = np.empty((3, 3), dtype=np.float64)
    sep_tol = max(tol, 1.0e-12)
    for i in range(3):
        for j in range(3):
            if i == j:
                divided[i, j] = slopes[i]
            else:
                denom = values[i] - values[j]
                if abs(denom) <= sep_tol:
                    if abs(clipped_values[i] - clipped_values[j]) <= sep_tol:
                        divided[i, j] = 0.5 * (slopes[i] + slopes[j])
                    else:
                        return False, jac, True
                else:
                    divided[i, j] = (clipped_values[i] - clipped_values[j]) / denom

    for col in range(4):
        modal = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                value = 0.0
                if col == 0:
                    value = vecs[0, i] * vecs[0, j]
                elif col == 1:
                    value = vecs[1, i] * vecs[1, j]
                elif col == 2:
                    value = vecs[2, i] * vecs[2, j]
                else:
                    value = vecs[0, i] * vecs[1, j] + vecs[1, i] * vecs[0, j]
                modal[i, j] = divided[i, j] * value
        tensor = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                value = 0.0
                for i in range(3):
                    for j in range(3):
                        value += vecs[a, i] * modal[i, j] * vecs[b, j]
                tensor[a, b] = value
        jac[0, col] = tensor[0, 0]
        jac[1, col] = tensor[1, 1]
        jac[2, col] = tensor[2, 2]
        jac[3, col] = tensor[0, 1]
    return True, jac, True


@njit(cache=True)
def _mc_tension_cutoff_consistent_tangent_numba(
    strain4: np.ndarray,
    plastic_strain: np.ndarray,
    D4: np.ndarray,
    initial_stress: np.ndarray,
    sin_phi: float,
    cohesion_term: float,
    hardening: float,
    kappa: float,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    Cn: np.ndarray,
    tensile_strength: float,
) -> tuple[bool, np.ndarray]:
    tangent = np.zeros((4, 4), dtype=np.float64)
    trial = np.empty(4, dtype=np.float64)
    for row in range(4):
        value = initial_stress[row]
        for col in range(4):
            value += D4[row, col] * (strain4[col] - plastic_strain[col])
        trial[row] = value

    f_trial, tol_fast = _mc_trial_yield_numba(trial, sin_phi, cohesion_term, hardening, kappa)
    if f_trial <= tol_fast:
        ok_cut, cutoff_jac, _clipped = _tension_cutoff_projection_jacobian_numba(trial, tensile_strength)
        if not ok_cut:
            return False, tangent
        for i in range(4):
            for j in range(4):
                value = 0.0
                for k in range(4):
                    value += cutoff_jac[i, k] * D4[k, j]
                tangent[i, j] = value
        return True, tangent

    sig_tr_p, vecs = _stress4_eigh_plane_strain_numba(trial)
    norm_sig = math.sqrt(sig_tr_p[0] * sig_tr_p[0] + sig_tr_p[1] * sig_tr_p[1] + sig_tr_p[2] * sig_tr_p[2])
    tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm_sig)
    ok, sig_corr_p, active_ids, active_count, _gamma, _vals_corr = _mc_return_mapping_principal_numba(
        sig_tr_p,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        Cn,
        hardening,
        kappa,
        tol,
    )
    if not ok or active_count <= 0:
        return False, tangent

    corrected_tensor = np.zeros((3, 3), dtype=np.float64)
    for a in range(3):
        for b in range(3):
            value = 0.0
            for i in range(3):
                value += vecs[a, i] * sig_corr_p[i] * vecs[b, i]
            corrected_tensor[a, b] = value
    corrected = np.empty(4, dtype=np.float64)
    corrected[0] = corrected_tensor[0, 0]
    corrected[1] = corrected_tensor[1, 1]
    corrected[2] = corrected_tensor[2, 2]
    corrected[3] = corrected_tensor[0, 1]

    tangent_ok, mc_tangent = _mc_consistent_tangent_spectral_numba(
        sig_tr_p,
        sig_corr_p,
        vecs,
        active_ids,
        active_count,
        yield_coeffs,
        flow_coeffs,
        Cn,
        D4,
        hardening,
    )
    if not tangent_ok:
        return False, tangent
    ok_cut, cutoff_jac, _clipped = _tension_cutoff_projection_jacobian_numba(corrected, tensile_strength)
    if not ok_cut:
        return False, tangent
    for i in range(4):
        for j in range(4):
            value = 0.0
            for k in range(4):
                value += cutoff_jac[i, k] * mc_tangent[k, j]
            tangent[i, j] = value
    return True, tangent


@njit(cache=True)
def _mc_trial_yield_numba(
    stress4: np.ndarray,
    sin_phi: float,
    cohesion_term: float,
    hardening: float,
    kappa: float,
) -> tuple[float, float]:
    principal = _principal_stresses_plane_strain_numba(stress4)
    f_trial = -1.0e300
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            value = (1.0 + sin_phi) * principal[i] - (1.0 - sin_phi) * principal[j] - cohesion_term - hardening * kappa
            if value > f_trial:
                f_trial = value
    norm = math.sqrt(principal[0] * principal[0] + principal[1] * principal[1] + principal[2] * principal[2])
    tol = 1.0e-10 * max(1.0, abs(cohesion_term), norm)
    return f_trial, tol


@njit(cache=True)
def _advanced_equivalent_shear_strain_numba(strain4: np.ndarray) -> float:
    mean = (strain4[0] + strain4[1] + strain4[2]) / 3.0
    dev0 = strain4[0] - mean
    dev1 = strain4[1] - mean
    dev2 = strain4[2] - mean
    dev3 = 0.5 * strain4[3]
    j2e = 0.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + dev3 * dev3
    return math.sqrt(max(4.0 * j2e / 3.0, 0.0))


@njit(cache=True)
def _mc_solve_active_system_numba(H: np.ndarray, rhs: np.ndarray, n: int) -> tuple[bool, np.ndarray]:
    gamma = np.zeros(3, dtype=np.float64)
    scale = 1.0
    for i in range(n):
        scale = max(scale, abs(rhs[i]))
        for j in range(n):
            scale = max(scale, abs(H[i, j]))
    tol = 1.0e-13 * scale
    if n == 1:
        det = H[0, 0]
        if abs(det) <= tol:
            return False, gamma
        gamma[0] = rhs[0] / det
        return math.isfinite(gamma[0]), gamma

    if n == 2:
        a = H[0, 0]
        b = H[0, 1]
        c = H[1, 0]
        d = H[1, 1]
        det = a * d - b * c
        if abs(det) <= tol:
            return False, gamma
        gamma[0] = (rhs[0] * d - b * rhs[1]) / det
        gamma[1] = (a * rhs[1] - rhs[0] * c) / det
        return math.isfinite(gamma[0]) and math.isfinite(gamma[1]), gamma

    a = H[0, 0]
    b = H[0, 1]
    c = H[0, 2]
    d = H[1, 0]
    e = H[1, 1]
    f = H[1, 2]
    g = H[2, 0]
    h = H[2, 1]
    i = H[2, 2]
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if abs(det) <= tol:
        return False, gamma
    r0 = rhs[0]
    r1 = rhs[1]
    r2 = rhs[2]
    gamma[0] = (r0 * (e * i - f * h) - b * (r1 * i - f * r2) + c * (r1 * h - e * r2)) / det
    gamma[1] = (a * (r1 * i - f * r2) - r0 * (d * i - f * g) + c * (d * r2 - r1 * g)) / det
    gamma[2] = (a * (e * r2 - r1 * h) - b * (d * r2 - r1 * g) + r0 * (d * h - e * g)) / det
    ok = math.isfinite(gamma[0]) and math.isfinite(gamma[1]) and math.isfinite(gamma[2])
    return ok, gamma


@njit(cache=True)
def _mc_eval_active_candidate_numba(
    sig_tr_p: np.ndarray,
    vals_tr: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    subset_ids: np.ndarray,
    nact: int,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    H = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for aidx in range(nact):
        active_a = subset_ids[aidx]
        rhs[aidx] = vals_tr[active_a]
        for bidx in range(nact):
            active_b = subset_ids[bidx]
            value = 0.0
            for p in range(3):
                tmp = 0.0
                for q in range(3):
                    tmp += Cn[p, q] * flow_coeffs[active_b, q]
                value += yield_coeffs[active_a, p] * tmp
            if aidx == bidx:
                value += hardening
            H[aidx, bidx] = value
    ok, gamma = _mc_solve_active_system_numba(H, rhs, nact)
    sig_corr = np.zeros(3, dtype=np.float64)
    vals_corr = np.zeros(6, dtype=np.float64)
    metric = np.empty(5, dtype=np.float64)
    if not ok:
        for i in range(5):
            metric[i] = 1.0e300
        return False, sig_corr, gamma, vals_corr, metric

    flow_sum = np.zeros(3, dtype=np.float64)
    sum_gamma = 0.0
    min_gamma = 1.0e300
    for aidx in range(nact):
        active = subset_ids[aidx]
        sum_gamma += gamma[aidx]
        min_gamma = min(min_gamma, gamma[aidx])
        for p in range(3):
            flow_sum[p] += flow_coeffs[active, p] * gamma[aidx]

    correction = np.zeros(3, dtype=np.float64)
    corr_norm_sq = 0.0
    for p in range(3):
        value = 0.0
        for q in range(3):
            value += Cn[p, q] * flow_sum[q]
        correction[p] = value
        corr_norm_sq += value * value
        sig_corr[p] = sig_tr_p[p] - value

    max_val = -1.0e300
    for row in range(6):
        value = -cohesion_term - hardening * (kappa + sum_gamma)
        for p in range(3):
            value += yield_coeffs[row, p] * sig_corr[p]
        vals_corr[row] = value
        if value > max_val:
            max_val = value

    active_res = 0.0
    for aidx in range(nact):
        value = abs(vals_corr[subset_ids[aidx]])
        if value > active_res:
            active_res = value
    metric[0] = max(0.0, max_val)
    metric[1] = active_res
    metric[2] = max(0.0, -min_gamma)
    metric[3] = math.sqrt(corr_norm_sq)
    metric[4] = float(nact)
    return True, sig_corr, gamma, vals_corr, metric


@njit(cache=True)
def _mc_metric_less_numba(metric: np.ndarray, best_metric: np.ndarray) -> bool:
    for i in range(5):
        if metric[i] < best_metric[i]:
            return True
        if metric[i] > best_metric[i]:
            return False
    return False


@njit(cache=True)
def _mc_eval_active_candidate_precomputed_numba(
    sig_tr_p: np.ndarray,
    vals_tr: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    subset_ids: np.ndarray,
    nact: int,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rhs = np.zeros(nact, dtype=np.float64)
    gamma_exact = np.zeros(nact, dtype=np.float64)
    for i in range(nact):
        rhs[i] = vals_tr[subset_ids[i]]
    if nact == 1:
        gamma_exact[0] = operator1[subset_ids[0]] * rhs[0]
    elif nact == 2:
        first = subset_ids[0]
        second = subset_ids[1]
        gamma_exact = operator2[first, second, :, :] @ rhs
    else:
        first = subset_ids[0]
        second = subset_ids[1]
        third = subset_ids[2]
        gamma_exact = operator3[first, second, third, :, :] @ rhs
    gamma = np.zeros(3, dtype=np.float64)
    for i in range(nact):
        gamma[i] = gamma_exact[i]
        if not math.isfinite(gamma_exact[i]):
            return (
                False,
                np.zeros(3, dtype=np.float64),
                gamma,
                np.zeros(6, dtype=np.float64),
                np.full(5, 1.0e300, dtype=np.float64),
            )

    M = np.zeros((nact, 3), dtype=np.float64)
    min_gamma = 1.0e300
    for active_index in range(nact):
        active = subset_ids[active_index]
        min_gamma = min(min_gamma, gamma[active_index])
        for component in range(3):
            M[active_index, component] = flow_coeffs[active, component]

    correction = Cn @ (M.T @ gamma_exact)
    sig_corr = sig_tr_p - correction
    vals_corr = (
        yield_coeffs @ sig_corr
        - cohesion_term
        - hardening * (kappa + float(np.sum(gamma_exact)))
    )
    active_res = 0.0
    for active_index in range(nact):
        active_res = max(active_res, abs(vals_corr[subset_ids[active_index]]))
    metric = np.empty(5, dtype=np.float64)
    metric[0] = max(0.0, float(np.max(vals_corr)))
    metric[1] = active_res
    metric[2] = max(0.0, -min_gamma)
    metric[3] = float(np.linalg.norm(correction))
    metric[4] = float(nact)
    return True, sig_corr, gamma, vals_corr, metric


@njit(cache=True)
def _mc_refine_active_candidate_lstsq_numba(
    sig_tr_p: np.ndarray,
    vals_tr: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    subset_ids: np.ndarray,
    active_count: int,
    candidate_h: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Recompute one shortlisted candidate with the Python fallback's least squares."""

    if active_count <= 0 or active_count > 3:
        return (
            False,
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            np.full(5, 1.0e300, dtype=np.float64),
        )
    M = np.zeros((active_count, 3), dtype=np.float64)
    rhs = np.zeros(active_count, dtype=np.float64)
    for row in range(active_count):
        active = subset_ids[row]
        rhs[row] = vals_tr[active]
        for component in range(3):
            M[row, component] = flow_coeffs[active, component]
    if active_count == 1:
        candidate_index = subset_ids[0]
    elif active_count == 2:
        candidate_index = 6 + subset_ids[0] * 6 + subset_ids[1]
    else:
        candidate_index = 42 + subset_ids[0] * 36 + subset_ids[1] * 6 + subset_ids[2]
    H = np.zeros((active_count, active_count), dtype=np.float64)
    for row in range(active_count):
        for column in range(active_count):
            H[row, column] = candidate_h[candidate_index, row, column]
    gamma_exact = np.linalg.lstsq(H, rhs, rcond=-1.0)[0]
    gamma = np.zeros(3, dtype=np.float64)
    for row in range(active_count):
        gamma[row] = gamma_exact[row]
        if not math.isfinite(gamma[row]):
            return (
                False,
                np.zeros(3, dtype=np.float64),
                gamma,
                np.zeros(6, dtype=np.float64),
                np.full(5, 1.0e300, dtype=np.float64),
            )

    correction = Cn @ (M.T @ gamma_exact)
    sig_corr = sig_tr_p - correction
    vals_corr = (
        yield_coeffs @ sig_corr
        - cohesion_term
        - hardening * (kappa + float(np.sum(gamma_exact)))
    )
    metric = np.empty(5, dtype=np.float64)
    metric[0] = max(0.0, float(np.max(vals_corr)))
    active_res = 0.0
    min_gamma = gamma[0]
    for row in range(active_count):
        active_res = max(active_res, abs(float(vals_corr[subset_ids[row]])))
        min_gamma = min(min_gamma, gamma[row])
    metric[1] = active_res
    metric[2] = max(0.0, -min_gamma)
    metric[3] = float(np.linalg.norm(correction))
    metric[4] = float(active_count)
    return True, sig_corr, gamma, vals_corr, metric


@njit(cache=True)
def _mc_return_mapping_principal_precomputed_numba(
    sig_tr_p: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
) -> tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Active-set scan using cached solve/pseudoinverse operators."""

    vals_tr = (
        yield_coeffs @ sig_tr_p
        - cohesion_term
        - hardening * kappa
    )

    order = np.arange(6, dtype=np.int64)
    for i in range(5):
        best = i
        for j in range(i + 1, 6):
            if vals_tr[order[j]] > vals_tr[order[best]]:
                best = j
        if best != i:
            tmp = order[i]
            order[i] = order[best]
            order[best] = tmp

    norm_sig = math.sqrt(sig_tr_p[0] * sig_tr_p[0] + sig_tr_p[1] * sig_tr_p[1] + sig_tr_p[2] * sig_tr_p[2])
    tol_active = 10.0 * tol
    tol_gamma = 1.0e-12 * max(1.0, norm_sig)
    candidate_ids = np.full((41, 3), -1, dtype=np.int64)
    candidate_counts = np.zeros(41, dtype=np.int64)
    candidate_count = 0
    for ia in range(6):
        candidate_ids[candidate_count, 0] = order[ia]
        candidate_counts[candidate_count] = 1
        candidate_count += 1
        for ib in range(ia + 1, 6):
            candidate_ids[candidate_count, 0] = order[ia]
            candidate_ids[candidate_count, 1] = order[ib]
            candidate_counts[candidate_count] = 2
            candidate_count += 1
            for ic in range(ib + 1, 6):
                candidate_ids[candidate_count, 0] = order[ia]
                candidate_ids[candidate_count, 1] = order[ib]
                candidate_ids[candidate_count, 2] = order[ic]
                candidate_counts[candidate_count] = 3
                candidate_count += 1

    approximate_ok = np.zeros(candidate_count, dtype=np.bool_)
    approximate_valid = np.zeros(candidate_count, dtype=np.bool_)
    approximate_metrics = np.full((candidate_count, 5), 1.0e300, dtype=np.float64)
    subset_ids = np.full(3, -1, dtype=np.int64)
    for candidate_index in range(candidate_count):
        nact = candidate_counts[candidate_index]
        for component in range(3):
            subset_ids[component] = candidate_ids[candidate_index, component]
        ok, sig_corr, gamma, vals_corr, metric = _mc_eval_active_candidate_precomputed_numba(
            sig_tr_p,
            vals_tr,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            Cn,
            hardening,
            kappa,
            subset_ids,
            nact,
            operator1,
            operator2,
            operator3,
        )
        if not ok:
            continue
        approximate_ok[candidate_index] = True
        approximate_valid[candidate_index] = (
            metric[0] <= tol_active
            and metric[1] <= tol_active
            and metric[2] <= tol_gamma
        )
        for component in range(5):
            approximate_metrics[candidate_index, component] = metric[component]

    shortlist_size = 8
    selected = np.zeros(candidate_count, dtype=np.bool_)
    approximate_ranks = np.full(candidate_count, shortlist_size, dtype=np.int64)
    for valid_group in range(2):
        want_valid = valid_group == 1
        for rank in range(shortlist_size):
            best_index = -1
            best_metric = np.full(5, 1.0e300, dtype=np.float64)
            for candidate_index in range(candidate_count):
                if (
                    not approximate_ok[candidate_index]
                    or selected[candidate_index]
                    or approximate_valid[candidate_index] != want_valid
                ):
                    continue
                metric = approximate_metrics[candidate_index]
                if best_index < 0 or _mc_metric_less_numba(metric, best_metric):
                    best_index = candidate_index
                    best_metric[:] = metric
            if best_index < 0:
                break
            selected[best_index] = True
            approximate_ranks[best_index] = rank

    best_valid_metric = np.full(5, 1.0e300, dtype=np.float64)
    best_near_metric = np.full(5, 1.0e300, dtype=np.float64)
    best_valid_sig = np.zeros(3, dtype=np.float64)
    best_near_sig = np.zeros(3, dtype=np.float64)
    best_valid_ids = np.full(3, -1, dtype=np.int64)
    best_near_ids = np.full(3, -1, dtype=np.int64)
    best_valid_gamma = np.zeros(3, dtype=np.float64)
    best_near_gamma = np.zeros(3, dtype=np.float64)
    best_valid_vals = np.zeros(6, dtype=np.float64)
    best_near_vals = np.zeros(6, dtype=np.float64)
    best_valid_count = 0
    best_near_count = 0
    best_valid_position = -1
    best_near_position = -1
    have_valid = False
    have_near = False
    shortlist_consistent = True

    for candidate_index in range(candidate_count):
        if not selected[candidate_index]:
            continue
        nact = candidate_counts[candidate_index]
        subset_ids[:] = candidate_ids[candidate_index, :]
        refined_ok, refined_sig, refined_gamma, refined_vals, refined_metric = (
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
                nact,
                candidate_h,
            )
        )
        if not refined_ok:
            shortlist_consistent = False
            break
        refined_valid = (
            refined_metric[0] <= tol_active
            and refined_metric[1] <= tol_active
            and refined_metric[2] <= tol_gamma
        )
        if refined_valid != approximate_valid[candidate_index]:
            shortlist_consistent = False
            break
        if refined_valid:
            if not have_valid or _mc_metric_less_numba(refined_metric, best_valid_metric):
                have_valid = True
                best_valid_position = candidate_index
                best_valid_count = nact
                best_valid_metric[:] = refined_metric
                best_valid_sig[:] = refined_sig
                best_valid_gamma[:] = refined_gamma
                best_valid_vals[:] = refined_vals
                best_valid_ids[:] = candidate_ids[candidate_index, :]
        elif not have_near or _mc_metric_less_numba(refined_metric, best_near_metric):
            have_near = True
            best_near_position = candidate_index
            best_near_count = nact
            best_near_metric[:] = refined_metric
            best_near_sig[:] = refined_sig
            best_near_gamma[:] = refined_gamma
            best_near_vals[:] = refined_vals
            best_near_ids[:] = candidate_ids[candidate_index, :]

    chosen_position = best_valid_position if have_valid else best_near_position
    if (
        not shortlist_consistent
        or chosen_position < 0
        or approximate_ranks[chosen_position] >= shortlist_size - 1
    ):
        have_valid = False
        have_near = False
        best_valid_metric[:] = 1.0e300
        best_near_metric[:] = 1.0e300
        for candidate_index in range(candidate_count):
            nact = candidate_counts[candidate_index]
            subset_ids[:] = candidate_ids[candidate_index, :]
            refined_ok, refined_sig, refined_gamma, refined_vals, refined_metric = (
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
                    nact,
                    candidate_h,
                )
            )
            if not refined_ok:
                continue
            refined_valid = (
                refined_metric[0] <= tol_active
                and refined_metric[1] <= tol_active
                and refined_metric[2] <= tol_gamma
            )
            if refined_valid:
                if not have_valid or _mc_metric_less_numba(refined_metric, best_valid_metric):
                    have_valid = True
                    best_valid_count = nact
                    best_valid_metric[:] = refined_metric
                    best_valid_sig[:] = refined_sig
                    best_valid_gamma[:] = refined_gamma
                    best_valid_vals[:] = refined_vals
                    best_valid_ids[:] = candidate_ids[candidate_index, :]
            elif not have_near or _mc_metric_less_numba(refined_metric, best_near_metric):
                have_near = True
                best_near_count = nact
                best_near_metric[:] = refined_metric
                best_near_sig[:] = refined_sig
                best_near_gamma[:] = refined_gamma
                best_near_vals[:] = refined_vals
                best_near_ids[:] = candidate_ids[candidate_index, :]

    if have_valid:
        return True, best_valid_sig, best_valid_ids, best_valid_count, best_valid_gamma, best_valid_vals
    if (
        have_near
        and best_near_metric[0] <= 100.0 * tol_active
        and best_near_metric[1] <= 100.0 * tol_active
        and best_near_metric[2] <= 100.0 * tol_gamma
    ):
        return True, best_near_sig, best_near_ids, best_near_count, best_near_gamma, best_near_vals
    return False, best_near_sig, best_near_ids, best_near_count, best_near_gamma, best_near_vals


@njit(cache=True)
def _mc_return_mapping_principal_numba(
    sig_tr_p: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
) -> tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    vals_tr = np.zeros(6, dtype=np.float64)
    for i in range(6):
        value = -cohesion_term - hardening * kappa
        for j in range(3):
            value += yield_coeffs[i, j] * sig_tr_p[j]
        vals_tr[i] = value

    order = np.empty(6, dtype=np.int64)
    for i in range(6):
        order[i] = i
    for i in range(5):
        best = i
        for j in range(i + 1, 6):
            if vals_tr[order[j]] > vals_tr[order[best]]:
                best = j
        if best != i:
            tmp = order[i]
            order[i] = order[best]
            order[best] = tmp

    norm_sig = math.sqrt(sig_tr_p[0] * sig_tr_p[0] + sig_tr_p[1] * sig_tr_p[1] + sig_tr_p[2] * sig_tr_p[2])
    tol_active = 10.0 * tol
    tol_gamma = 1.0e-12 * max(1.0, norm_sig)
    best_valid_metric = np.empty(5, dtype=np.float64)
    best_near_metric = np.empty(5, dtype=np.float64)
    for i in range(5):
        best_valid_metric[i] = 1.0e300
        best_near_metric[i] = 1.0e300
    have_valid = False
    have_near = False
    best_valid_sig = np.zeros(3, dtype=np.float64)
    best_near_sig = np.zeros(3, dtype=np.float64)
    best_valid_ids = np.full(3, -1, dtype=np.int64)
    best_near_ids = np.full(3, -1, dtype=np.int64)
    best_valid_gamma = np.zeros(3, dtype=np.float64)
    best_near_gamma = np.zeros(3, dtype=np.float64)
    best_valid_vals = np.zeros(6, dtype=np.float64)
    best_near_vals = np.zeros(6, dtype=np.float64)
    best_valid_count = 0
    best_near_count = 0

    candidate_ids = np.full((41, 3), -1, dtype=np.int64)
    candidate_counts = np.zeros(41, dtype=np.int64)
    candidate_count = 0
    for ia in range(6):
        candidate_ids[candidate_count, 0] = order[ia]
        candidate_counts[candidate_count] = 1
        candidate_count += 1
        for ib in range(ia + 1, 6):
            candidate_ids[candidate_count, 0] = order[ia]
            candidate_ids[candidate_count, 1] = order[ib]
            candidate_counts[candidate_count] = 2
            candidate_count += 1
            for ic in range(ib + 1, 6):
                candidate_ids[candidate_count, 0] = order[ia]
                candidate_ids[candidate_count, 1] = order[ib]
                candidate_ids[candidate_count, 2] = order[ic]
                candidate_counts[candidate_count] = 3
                candidate_count += 1

    subset_ids = np.full(3, -1, dtype=np.int64)
    for candidate_index in range(candidate_count):
        nact = candidate_counts[candidate_index]
        for p in range(3):
            subset_ids[p] = candidate_ids[candidate_index, p]
        ok, sig_corr, gamma, vals_corr, metric = _mc_eval_active_candidate_numba(
            sig_tr_p,
            vals_tr,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            Cn,
            hardening,
            kappa,
            subset_ids,
            nact,
        )
        if not ok:
            continue
        if metric[0] <= tol_active and metric[1] <= tol_active and metric[2] <= tol_gamma:
            if not have_valid or _mc_metric_less_numba(metric, best_valid_metric):
                have_valid = True
                best_valid_count = nact
                for m in range(5):
                    best_valid_metric[m] = metric[m]
                for p in range(3):
                    best_valid_sig[p] = sig_corr[p]
                    best_valid_ids[p] = subset_ids[p] if p < nact else -1
                    best_valid_gamma[p] = gamma[p] if p < nact else 0.0
                for row in range(6):
                    best_valid_vals[row] = vals_corr[row]
        else:
            if not have_near or _mc_metric_less_numba(metric, best_near_metric):
                have_near = True
                best_near_count = nact
                for m in range(5):
                    best_near_metric[m] = metric[m]
                for p in range(3):
                    best_near_sig[p] = sig_corr[p]
                    best_near_ids[p] = subset_ids[p] if p < nact else -1
                    best_near_gamma[p] = gamma[p] if p < nact else 0.0
                for row in range(6):
                    best_near_vals[row] = vals_corr[row]

    if have_valid:
        return True, best_valid_sig, best_valid_ids, best_valid_count, best_valid_gamma, best_valid_vals
    if (
        have_near
        and best_near_metric[0] <= 100.0 * tol_active
        and best_near_metric[1] <= 100.0 * tol_active
        and best_near_metric[2] <= 100.0 * tol_gamma
    ):
        return True, best_near_sig, best_near_ids, best_near_count, best_near_gamma, best_near_vals
    return False, best_near_sig, best_near_ids, best_near_count, best_near_gamma, best_near_vals


@njit(cache=True)
def _mc_regularized_projection_return_mapping_numba(
    sig_tr_p: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
    seed_sig: np.ndarray,
    seed_active_ids: np.ndarray,
    seed_gamma: np.ndarray,
    seed_count: int,
) -> tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, np.ndarray, float, float, float]:
    """Exact multisurface projection for an otherwise inadmissible apex state.

    The ordinary update first exhausts the configured non-associated active
    sets.  With zero dilation, a tension-side state can have no admissible
    return because the configured flow directions cannot change mean stress.
    The apex policy is therefore an explicit associated multisurface submodel,
    solved in one active-set pass.  Unlike the former bounded sequential loop,
    this path must satisfy the yield and complementarity residuals before it is
    accepted.
    """

    (
        ok,
        corrected,
        active_ids,
        active_count,
        gamma_return,
        vals_return,
    ) = _mc_return_mapping_principal_numba(
        sig_tr_p,
        yield_coeffs,
        yield_coeffs,
        cohesion_term,
        Cn,
        hardening,
        kappa,
        tol,
    )

    norm_trial_sq = 0.0
    for i in range(3):
        norm_trial_sq += sig_tr_p[i] * sig_tr_p[i]
    norm_trial = math.sqrt(norm_trial_sq)
    scale = max(1.0, abs(cohesion_term), norm_trial)
    verified_tol = max(10.0 * tol, 1.0e-12 * scale)
    if not ok or active_count <= 0:
        return (
            False,
            corrected,
            active_ids,
            active_count,
            gamma_return,
            active_count,
            vals_return,
            0.0,
            0.0,
            verified_tol,
        )

    max_violation = max(0.0, float(np.max(vals_return)))
    active_residual = 0.0
    min_gamma = gamma_return[0]
    for slot in range(active_count):
        active_residual = max(
            active_residual,
            abs(float(vals_return[active_ids[slot]])),
        )
        min_gamma = min(min_gamma, gamma_return[slot])
    gamma_tol = 1.0e-12 * max(1.0, norm_trial)
    finite = True
    for i in range(3):
        finite = finite and math.isfinite(corrected[i])
    if (
        not finite
        or max_violation > verified_tol
        or active_residual > verified_tol
        or min_gamma < -gamma_tol
    ):
        return (
            False,
            corrected,
            active_ids,
            active_count,
            gamma_return,
            active_count,
            vals_return,
            max(max_violation, active_residual),
            max(max_violation, active_residual) / scale,
            verified_tol,
        )
    verified_residual = max(max_violation, active_residual)
    return (
        True,
        corrected,
        active_ids,
        active_count,
        gamma_return,
        active_count,
        vals_return,
        verified_residual,
        verified_residual / scale,
        verified_tol,
    )


@njit(cache=True)
def _mc_legacy_bounded_projection_return_mapping_numba(
    sig_tr_p: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
    seed_sig: np.ndarray,
    seed_active_ids: np.ndarray,
    seed_gamma: np.ndarray,
    seed_count: int,
) -> tuple[bool, np.ndarray, np.ndarray, int, np.ndarray, int, np.ndarray, float, float, float]:
    """Compatibility projection used by previously calibrated SRM models."""

    sig = seed_sig.copy()
    finite_seed = sig.shape[0] == 3
    for component in range(3):
        finite_seed = finite_seed and math.isfinite(sig[component])
    empty_ids = np.full(3, -1, dtype=np.int64)
    empty_gamma = np.zeros(3, dtype=np.float64)
    empty_vals = np.zeros(6, dtype=np.float64)
    if not finite_seed:
        return False, sig, empty_ids, 0, empty_gamma, 0, empty_vals, 0.0, 0.0, 0.0

    norm_trial_sq = 0.0
    for component in range(3):
        norm_trial_sq += sig_tr_p[component] * sig_tr_p[component]
    norm_trial = math.sqrt(norm_trial_sq)
    scale = max(1.0, abs(cohesion_term), norm_trial)
    relaxed_tol = max(1.0e3 * tol, 1.0e-8 * scale)
    total_gamma = 0.0
    gamma_acc = np.zeros(6, dtype=np.float64)
    for slot in range(min(seed_count, 3)):
        value = max(seed_gamma[slot], 0.0)
        total_gamma += value
        active_id = seed_active_ids[slot]
        if 0 <= active_id < 6:
            gamma_acc[active_id] += value

    vals = yield_coeffs @ sig - cohesion_term - hardening * (kappa + total_gamma)
    best_sig = sig.copy()
    best_vals = vals.copy()
    best_violation = max(0.0, float(np.max(vals)))
    for _iteration in range(80):
        vals = yield_coeffs @ sig - cohesion_term - hardening * (kappa + total_gamma)
        finite_vals = True
        for plane in range(6):
            finite_vals = finite_vals and math.isfinite(vals[plane])
        if not finite_vals:
            break
        max_id = int(np.argmax(vals))
        violation = max(0.0, float(vals[max_id]))
        if violation < best_violation:
            best_violation = violation
            best_sig = sig.copy()
            best_vals = vals.copy()
        if violation <= relaxed_tol:
            best_violation = violation
            best_sig = sig.copy()
            best_vals = vals.copy()
            break
        direction = Cn @ yield_coeffs[max_id]
        denom = float(yield_coeffs[max_id] @ direction + hardening)
        if not math.isfinite(denom) or denom <= np.finfo(np.float64).eps:
            break
        dgamma = float(vals[max_id]) / denom
        if not math.isfinite(dgamma) or dgamma <= 0.0:
            break
        sig = sig - direction * dgamma
        gamma_acc[max_id] += dgamma
        total_gamma += dgamma

    acceptance_limit = max(relaxed_tol, 0.25 * max(1.0, abs(cohesion_term)))
    if best_violation > acceptance_limit:
        return (
            False,
            best_sig,
            empty_ids,
            0,
            empty_gamma,
            0,
            best_vals,
            best_violation,
            best_violation / scale,
            relaxed_tol,
        )

    active_ids = np.full(3, -1, dtype=np.int64)
    gamma_return = np.zeros(3, dtype=np.float64)
    threshold = max(tol, 1.0e-14 * max(1.0, norm_trial))
    active_count = 0
    used = np.zeros(6, dtype=np.bool_)
    for slot in range(3):
        best_id = -1
        best_gamma = threshold
        for plane in range(6):
            if not used[plane] and gamma_acc[plane] > best_gamma:
                best_id = plane
                best_gamma = gamma_acc[plane]
        if best_id < 0:
            break
        used[best_id] = True
        active_ids[active_count] = best_id
        gamma_return[active_count] = best_gamma
        active_count += 1
    if active_count == 0 and total_gamma > 0.0:
        gamma_return[0] = total_gamma
    return (
        True,
        best_sig,
        active_ids,
        active_count,
        gamma_return,
        active_count,
        best_vals,
        best_violation,
        best_violation / scale,
        relaxed_tol,
    )

@njit(cache=True)
def _mc_principal_jacobian_numba(
    active_ids: np.ndarray,
    active_count: int,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    Cn: np.ndarray,
    hardening: float,
) -> tuple[bool, np.ndarray]:
    P = np.eye(3, dtype=np.float64)
    if active_count <= 0:
        return True, P
    H = np.zeros((3, 3), dtype=np.float64)
    for aidx in range(active_count):
        active_a = active_ids[aidx]
        for bidx in range(active_count):
            active_b = active_ids[bidx]
            value = 0.0
            for p in range(3):
                tmp = 0.0
                for q in range(3):
                    tmp += Cn[p, q] * flow_coeffs[active_b, q]
                value += yield_coeffs[active_a, p] * tmp
            if aidx == bidx:
                value += hardening
            H[aidx, bidx] = value

    Y = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for col in range(3):
        for row in range(active_count):
            rhs[row] = yield_coeffs[active_ids[row], col]
        ok, solution = _mc_solve_active_system_numba(H, rhs, active_count)
        if not ok:
            return False, P
        for row in range(active_count):
            Y[row, col] = solution[row]

    for i in range(3):
        for j in range(3):
            value = 1.0 if i == j else 0.0
            for aidx in range(active_count):
                active_a = active_ids[aidx]
                cm = 0.0
                for q in range(3):
                    cm += Cn[i, q] * flow_coeffs[active_a, q]
                value -= cm * Y[aidx, j]
            P[i, j] = value
            if not math.isfinite(value):
                return False, P
    return True, P


@njit(cache=True)
def _mc_principal_jacobian_precomputed_numba(
    active_ids: np.ndarray,
    active_count: int,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    Cn: np.ndarray,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
) -> tuple[bool, np.ndarray]:
    """Build the fixed-active-set Jacobian from a validated cached inverse."""

    P = np.eye(3, dtype=np.float64)
    if active_count <= 0:
        return True, P
    if active_count > 3:
        return False, P

    inverse = np.zeros((3, 3), dtype=np.float64)
    if active_count == 1:
        first = active_ids[0]
        inverse[0, 0] = operator1[first]
        candidate_index = first
    elif active_count == 2:
        first = active_ids[0]
        second = active_ids[1]
        inverse[:2, :2] = operator2[first, second, :, :]
        candidate_index = 6 + first * 6 + second
    else:
        first = active_ids[0]
        second = active_ids[1]
        third = active_ids[2]
        inverse[:, :] = operator3[first, second, third, :, :]
        candidate_index = 42 + first * 36 + second * 6 + third

    inverse_norm = 0.0
    residual_norm = 0.0
    for row in range(active_count):
        for column in range(active_count):
            value = inverse[row, column]
            if not math.isfinite(value):
                return False, P
            inverse_norm = max(inverse_norm, abs(value))
            product = 0.0
            for inner in range(active_count):
                product += (
                    candidate_h[candidate_index, row, inner]
                    * inverse[inner, column]
                )
            expected = 1.0 if row == column else 0.0
            residual_norm = max(residual_norm, abs(product - expected))
    if inverse_norm <= 0.0 or inverse_norm > 1.0e14 or residual_norm > 1.0e-8:
        return False, P

    Y = np.zeros((3, 3), dtype=np.float64)
    for row in range(active_count):
        for column in range(3):
            value = 0.0
            for inner in range(active_count):
                value += (
                    inverse[row, inner]
                    * yield_coeffs[active_ids[inner], column]
                )
            Y[row, column] = value

    for i in range(3):
        for j in range(3):
            value = 1.0 if i == j else 0.0
            for active_index in range(active_count):
                active = active_ids[active_index]
                cm = 0.0
                for q in range(3):
                    cm += Cn[i, q] * flow_coeffs[active, q]
                value -= cm * Y[active_index, j]
            if not math.isfinite(value):
                return False, P
            P[i, j] = value
    return True, P


@njit(cache=True)
def _stress4_column_to_tensor_numba(D4: np.ndarray, col: int) -> np.ndarray:
    tensor = np.zeros((3, 3), dtype=np.float64)
    tensor[0, 0] = D4[0, col]
    tensor[1, 1] = D4[1, col]
    tensor[2, 2] = D4[2, col]
    tensor[0, 1] = D4[3, col]
    tensor[1, 0] = D4[3, col]
    return tensor


@njit(cache=True)
def _tensor_to_stress4_numba(tensor: np.ndarray) -> np.ndarray:
    out = np.empty(4, dtype=np.float64)
    out[0] = tensor[0, 0]
    out[1] = tensor[1, 1]
    out[2] = tensor[2, 2]
    out[3] = tensor[0, 1]
    return out


@njit(cache=True)
def _mc_consistent_tangent_spectral_numba(
    sig_tr_p: np.ndarray,
    sig_corr_p: np.ndarray,
    vecs: np.ndarray,
    active_ids: np.ndarray,
    active_count: int,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    Cn: np.ndarray,
    D4: np.ndarray,
    hardening: float,
) -> tuple[bool, np.ndarray]:
    ok, P = _mc_principal_jacobian_numba(active_ids, active_count, yield_coeffs, flow_coeffs, Cn, hardening)
    if not ok:
        return False, np.zeros((4, 4), dtype=np.float64)
    return _mc_consistent_tangent_from_principal_jacobian_numba(
        sig_tr_p,
        sig_corr_p,
        vecs,
        P,
        D4,
    )


@njit(cache=True)
def _mc_consistent_tangent_spectral_precomputed_numba(
    sig_tr_p: np.ndarray,
    sig_corr_p: np.ndarray,
    vecs: np.ndarray,
    active_ids: np.ndarray,
    active_count: int,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    Cn: np.ndarray,
    D4: np.ndarray,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
) -> tuple[bool, np.ndarray]:
    ok, P = _mc_principal_jacobian_precomputed_numba(
        active_ids,
        active_count,
        yield_coeffs,
        flow_coeffs,
        Cn,
        operator1,
        operator2,
        operator3,
        candidate_h,
    )
    if not ok:
        return False, np.zeros((4, 4), dtype=np.float64)
    return _mc_consistent_tangent_from_principal_jacobian_numba(
        sig_tr_p,
        sig_corr_p,
        vecs,
        P,
        D4,
    )


@njit(cache=True)
def _mc_consistent_tangent_from_principal_jacobian_numba(
    sig_tr_p: np.ndarray,
    sig_corr_p: np.ndarray,
    vecs: np.ndarray,
    P: np.ndarray,
    D4: np.ndarray,
) -> tuple[bool, np.ndarray]:
    tangent = np.zeros((4, 4), dtype=np.float64)

    scale_ref = 1.0
    for i in range(3):
        scale_ref = max(scale_ref, abs(sig_tr_p[i]))
    tol_lam = 1.0e-12 * scale_ref
    coeff = np.zeros((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            den = sig_tr_p[i] - sig_tr_p[j]
            if abs(den) > tol_lam:
                coeff[i, j] = (sig_corr_p[i] - sig_corr_p[j]) / den
            else:
                coeff[i, j] = 0.5 * (P[i, i] - P[i, j] - P[j, i] + P[j, j])

    for col in range(4):
        dS_tr = _stress4_column_to_tensor_numba(D4, col)
        Hp = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            for j in range(3):
                value = 0.0
                for a in range(3):
                    for b in range(3):
                        value += vecs[a, i] * dS_tr[a, b] * vecs[b, j]
                Hp[i, j] = value

        dYp = np.zeros((3, 3), dtype=np.float64)
        for i in range(3):
            value = 0.0
            for j in range(3):
                value += P[i, j] * Hp[j, j]
            dYp[i, i] = value
        for i in range(3):
            for j in range(i + 1, 3):
                dYp[i, j] = coeff[i, j] * Hp[i, j]
                dYp[j, i] = coeff[j, i] * Hp[j, i]

        dY = np.zeros((3, 3), dtype=np.float64)
        for a in range(3):
            for b in range(3):
                value = 0.0
                for i in range(3):
                    for j in range(3):
                        value += vecs[a, i] * dYp[i, j] * vecs[b, j]
                dY[a, b] = value
        out = _tensor_to_stress4_numba(dY)
        for row in range(4):
            tangent[row, col] = out[row]
            if not math.isfinite(out[row]):
                return False, tangent
    return True, tangent


def principal_stresses(stress4: np.ndarray) -> np.ndarray:
    return _principal_stresses_plane_strain_numba(np.ascontiguousarray(stress4, dtype=np.float64))


def _uses_plastic_strength_model(material: ElasticPlaneStrainMaterial) -> bool:
    return material.model in {"drucker_prager", "dp", "von_mises", "j2", "mohr_coulomb", "mc"}


_ADVANCED_MATERIAL_MODELS = {
    "nonlinear_elastic",
    "hardin_drnevich",
    "duncan_chang",
    "ramberg_osgood",
    "uw_clay",
    "pastor_zienkiewicz_sand",
    "pastor_zienkiewicz_clay",
    "liquefaction",
    "bilinear_liquefaction",
}

_ADV_MODEL_GENERIC = 0
_ADV_MODEL_NONLINEAR_ELASTIC = 1
_ADV_MODEL_HARDIN_DRNEVICH = 2
_ADV_MODEL_DUNCAN_CHANG = 3
_ADV_MODEL_RAMBERG_OSGOOD = 4
_ADV_MODEL_UW_CLAY = 5
_ADV_MODEL_PZ_SAND = 6
_ADV_MODEL_PZ_CLAY = 7
_ADV_MODEL_LIQUEFACTION = 8
_ADV_MODEL_BILINEAR_LIQUEFACTION = 9

_ADVANCED_MODEL_ID_MAP = {
    "nonlinear_elastic": _ADV_MODEL_NONLINEAR_ELASTIC,
    "hardin_drnevich": _ADV_MODEL_HARDIN_DRNEVICH,
    "duncan_chang": _ADV_MODEL_DUNCAN_CHANG,
    "ramberg_osgood": _ADV_MODEL_RAMBERG_OSGOOD,
    "uw_clay": _ADV_MODEL_UW_CLAY,
    "pastor_zienkiewicz_sand": _ADV_MODEL_PZ_SAND,
    "pastor_zienkiewicz_clay": _ADV_MODEL_PZ_CLAY,
    "liquefaction": _ADV_MODEL_LIQUEFACTION,
    "bilinear_liquefaction": _ADV_MODEL_BILINEAR_LIQUEFACTION,
}

_ADV_STATE_FIELDS = (
    "gamma_eq",
    "delta_gamma",
    "cyclic_strain",
    "cycle_increment",
    "cycles",
    "ru",
    "ru_generation_increment",
    "ru_dissipation_increment",
    "ru_dissipation_rate",
    "liquefaction_FL",
    "modulus_ratio",
    "effective_G",
    "effective_E",
    "hardening_variable",
    "dilatancy",
)
_ADV_STATE_GAMMA_EQ = 0
_ADV_STATE_DELTA_GAMMA = 1
_ADV_STATE_CYCLIC_STRAIN = 2
_ADV_STATE_CYCLE_INCREMENT = 3
_ADV_STATE_CYCLES = 4
_ADV_STATE_RU = 5
_ADV_STATE_RU_GENERATION_INCREMENT = 6
_ADV_STATE_RU_DISSIPATION_INCREMENT = 7
_ADV_STATE_RU_DISSIPATION_RATE = 8
_ADV_STATE_LIQUEFACTION_FL = 9
_ADV_STATE_MODULUS_RATIO = 10
_ADV_STATE_EFFECTIVE_G = 11
_ADV_STATE_EFFECTIVE_E = 12
_ADV_STATE_HARDENING_VARIABLE = 13
_ADV_STATE_DILATANCY = 14

_ADV_PARAM_GAMMA_REF = 0
_ADV_PARAM_EXPONENT = 1
_ADV_PARAM_ALPHA = 2
_ADV_PARAM_R = 3
_ADV_PARAM_RF = 4
_ADV_PARAM_N = 5
_ADV_PARAM_MIN_RATIO = 6
_ADV_PARAM_G0 = 7
_ADV_PARAM_RU_INITIAL = 8
_ADV_PARAM_LIQ_ENABLED = 9
_ADV_PARAM_CYCLE_INCREMENT = 10
_ADV_PARAM_RU_DIRECT = 11
_ADV_PARAM_CSR = 12
_ADV_PARAM_CRR = 13
_ADV_PARAM_GENERATION = 14
_ADV_PARAM_DISSIPATION = 15
_ADV_PARAM_POST_RATIO = 16
_ADV_PARAM_POST_ENABLED = 17
_ADV_PARAM_NU = 18
_ADV_PARAM_COUNT = 19


MC_ORDERED_PAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))


def _apply_tension_cutoff(stress4: np.ndarray, material: ElasticPlaneStrainMaterial) -> tuple[np.ndarray, bool, float]:
    if not material.tension_cutoff:
        return np.asarray(stress4, dtype=float), False, 0.0
    ft = float(material.tensile_strength)
    if not math.isfinite(ft):
        return np.asarray(stress4, dtype=float), False, 0.0
    updated, clipped, excess = _tension_cutoff_plane_strain_numba(np.ascontiguousarray(stress4, dtype=np.float64), ft)
    return updated, bool(clipped), float(excess)


def _is_advanced_material(material: ElasticPlaneStrainMaterial) -> bool:
    return bool(material.advanced_model) or material.model in _ADVANCED_MATERIAL_MODELS


def _advanced_model_id(source_model: str) -> int:
    return int(_ADVANCED_MODEL_ID_MAP.get(str(source_model).lower().strip(), _ADV_MODEL_GENERIC))


def _advanced_params_array(material: ElasticPlaneStrainMaterial, source_model: str) -> np.ndarray:
    params = material.advanced_params or {}
    liq = params.get("liquefaction")
    liq_map = liq if isinstance(liq, Mapping) else params
    liq_enabled = 1.0 if isinstance(liq_map, Mapping) and (
        source_model in {"liquefaction", "bilinear_liquefaction"} or "cyclic_resistance_ratio" in liq_map or "CRR" in liq_map
    ) else 0.0
    out = np.zeros(_ADV_PARAM_COUNT, dtype=float)
    out[_ADV_PARAM_GAMMA_REF] = max(_param_float(params, ("gamma_ref", "reference_strain", "gamma50"), 1.0e-3), 1.0e-12)
    out[_ADV_PARAM_EXPONENT] = max(_param_float(params, ("exponent", "m"), 1.0), 1.0e-6)
    out[_ADV_PARAM_ALPHA] = max(_param_float(params, ("alpha", "a"), 1.0), 0.0)
    out[_ADV_PARAM_R] = max(_param_float(params, ("r", "n", "exponent"), 2.0), 1.0)
    out[_ADV_PARAM_RF] = max(0.0, min(_param_float(params, ("Rf", "failure_ratio"), 0.9), 0.99))
    out[_ADV_PARAM_N] = max(_param_float(params, ("n", "exponent"), 1.0), 1.0e-6)
    out[_ADV_PARAM_MIN_RATIO] = max(0.0, min(_param_float(params, ("min_stiffness_ratio", "Gmin_ratio"), 0.02), 1.0))
    out[_ADV_PARAM_G0] = _param_float(params, ("G0", "Gmax", "G_max", "initial_shear_modulus"), material.shear_mu)
    out[_ADV_PARAM_RU_INITIAL] = _liquefaction_ru(liq_map if isinstance(liq_map, Mapping) else params)
    out[_ADV_PARAM_LIQ_ENABLED] = liq_enabled
    out[_ADV_PARAM_CYCLE_INCREMENT] = max(_param_float(liq_map, ("cycle_increment", "cycles_per_step", "dN"), 0.0), 0.0) if isinstance(liq_map, Mapping) else 0.0
    out[_ADV_PARAM_RU_DIRECT] = _param_float(liq_map, ("ru", "excess_pore_pressure_ratio"), math.nan) if isinstance(liq_map, Mapping) else math.nan
    out[_ADV_PARAM_CSR] = _param_float(liq_map, ("cyclic_stress_ratio", "CSR"), 0.0) if isinstance(liq_map, Mapping) else 0.0
    out[_ADV_PARAM_CRR] = _param_float(liq_map, ("cyclic_resistance_ratio", "CRR", "RL20"), 0.0) if isinstance(liq_map, Mapping) else 0.0
    out[_ADV_PARAM_GENERATION] = max(_param_float(liq_map, ("generation_rate", "ru_generation_rate"), 0.25), 0.0) if isinstance(liq_map, Mapping) else 0.0
    out[_ADV_PARAM_DISSIPATION] = max(_param_float(liq_map, ("dissipation_rate", "ru_dissipation_rate"), 0.0), 0.0) if isinstance(liq_map, Mapping) else 0.0
    out[_ADV_PARAM_POST_RATIO] = max(0.0, min(_param_float(liq_map, ("post_liquefaction_stiffness_ratio", "G_post_ratio"), out[_ADV_PARAM_MIN_RATIO]), 1.0)) if isinstance(liq_map, Mapping) else out[_ADV_PARAM_MIN_RATIO]
    out[_ADV_PARAM_POST_ENABLED] = 1.0 if isinstance(liq_map, Mapping) else 0.0
    out[_ADV_PARAM_NU] = float(material.nu)
    return out


def _advanced_state_array_from_vars(state_vars: Mapping[str, object] | None, *, ru_default: float = 0.0) -> np.ndarray:
    source = state_vars or {}
    out = np.zeros(len(_ADV_STATE_FIELDS), dtype=float)
    out[_ADV_STATE_RU] = max(0.0, min(_state_float(source, "ru", ru_default), 0.99))
    out[_ADV_STATE_LIQUEFACTION_FL] = _state_float(source, "liquefaction_FL", math.inf)
    for index, key in enumerate(_ADV_STATE_FIELDS):
        if index in {_ADV_STATE_RU, _ADV_STATE_LIQUEFACTION_FL}:
            continue
        out[index] = _state_float(source, key, 0.0)
    return out


@njit(cache=True)
def _advanced_modulus_ratio_numba(model_id: int, gamma_eq: float, gamma_ref: float, params: np.ndarray) -> float:
    x = max(gamma_eq / gamma_ref, 0.0)
    if (
        model_id == _ADV_MODEL_NONLINEAR_ELASTIC
        or model_id == _ADV_MODEL_HARDIN_DRNEVICH
        or model_id == _ADV_MODEL_UW_CLAY
        or model_id == _ADV_MODEL_PZ_SAND
        or model_id == _ADV_MODEL_PZ_CLAY
        or model_id == _ADV_MODEL_LIQUEFACTION
        or model_id == _ADV_MODEL_BILINEAR_LIQUEFACTION
    ):
        exponent = max(params[_ADV_PARAM_EXPONENT], 1.0e-6)
        return 1.0 / (1.0 + x**exponent)
    if model_id == _ADV_MODEL_RAMBERG_OSGOOD:
        alpha = max(params[_ADV_PARAM_ALPHA], 0.0)
        r = max(params[_ADV_PARAM_R], 1.0)
        return 1.0 / (1.0 + alpha * x ** max(r - 1.0, 0.0))
    if model_id == _ADV_MODEL_DUNCAN_CHANG:
        rf = max(0.0, min(params[_ADV_PARAM_RF], 0.99))
        n = max(params[_ADV_PARAM_N], 1.0e-6)
        return max(1.0 - rf * x / (1.0 + x), 0.0) ** n
    return 1.0


@njit(cache=True)
def _advanced_history_state_numba(strain4: np.ndarray, previous: np.ndarray, params: np.ndarray, model_id: int) -> np.ndarray:
    history = np.zeros(15, dtype=np.float64)
    gamma_eq = _advanced_equivalent_shear_strain_numba(strain4)
    gamma_ref = max(params[_ADV_PARAM_GAMMA_REF], 1.0e-12)
    old_gamma = previous[_ADV_STATE_GAMMA_EQ]
    delta_gamma = abs(gamma_eq - old_gamma)
    cycle_increment = delta_gamma / max(4.0 * gamma_ref, 1.0e-12)
    ru_old = previous[_ADV_STATE_RU]
    if not math.isfinite(ru_old):
        ru_old = params[_ADV_PARAM_RU_INITIAL]
    ru_old = max(0.0, min(ru_old, 0.99))
    ru = ru_old
    ru_generation_increment = 0.0
    ru_dissipation_increment = 0.0
    dissipation = 0.0
    if params[_ADV_PARAM_LIQ_ENABLED] > 0.5:
        cycle_increment += max(params[_ADV_PARAM_CYCLE_INCREMENT], 0.0)
        ru_direct = params[_ADV_PARAM_RU_DIRECT]
        if math.isfinite(ru_direct):
            ru = max(0.0, min(ru_direct, 0.99))
        else:
            csr = params[_ADV_PARAM_CSR]
            crr = max(params[_ADV_PARAM_CRR], 1.0e-12)
            generation = max(params[_ADV_PARAM_GENERATION], 0.0)
            dissipation = max(params[_ADV_PARAM_DISSIPATION], 0.0)
            ru_generation_increment = generation * max(csr / crr, 0.0) * cycle_increment
            ru_dissipation_increment = dissipation * ru_old
            ru = max(0.0, min(0.99, ru_old + ru_generation_increment - ru_dissipation_increment))
    cycles = previous[_ADV_STATE_CYCLES] + cycle_increment
    ratio = _advanced_modulus_ratio_numba(model_id, gamma_eq, gamma_ref, params)
    min_ratio = max(0.0, min(params[_ADV_PARAM_MIN_RATIO], 1.0))
    if params[_ADV_PARAM_POST_ENABLED] > 0.5 or model_id == _ADV_MODEL_LIQUEFACTION or model_id == _ADV_MODEL_BILINEAR_LIQUEFACTION:
        post_ratio = max(0.0, min(params[_ADV_PARAM_POST_RATIO], 1.0))
        ratio = max(post_ratio, ratio * max(0.0, 1.0 - ru))
    ratio = max(min_ratio, min(ratio, 1.0))
    effective_g = max(params[_ADV_PARAM_G0] * ratio, np.finfo(np.float64).eps)
    effective_e = max(2.0 * effective_g * (1.0 + params[_ADV_PARAM_NU]), np.finfo(np.float64).eps)
    csr = params[_ADV_PARAM_CSR]
    crr = params[_ADV_PARAM_CRR]
    fl = math.inf if csr <= 0.0 else crr / csr
    history[_ADV_STATE_GAMMA_EQ] = gamma_eq
    history[_ADV_STATE_DELTA_GAMMA] = delta_gamma
    history[_ADV_STATE_CYCLIC_STRAIN] = previous[_ADV_STATE_CYCLIC_STRAIN] + delta_gamma
    history[_ADV_STATE_CYCLE_INCREMENT] = cycle_increment
    history[_ADV_STATE_CYCLES] = cycles
    history[_ADV_STATE_RU] = ru
    history[_ADV_STATE_RU_GENERATION_INCREMENT] = ru_generation_increment
    history[_ADV_STATE_RU_DISSIPATION_INCREMENT] = ru_dissipation_increment
    history[_ADV_STATE_RU_DISSIPATION_RATE] = dissipation
    history[_ADV_STATE_LIQUEFACTION_FL] = fl
    history[_ADV_STATE_MODULUS_RATIO] = ratio
    history[_ADV_STATE_EFFECTIVE_G] = effective_g
    history[_ADV_STATE_EFFECTIVE_E] = effective_e
    history[_ADV_STATE_HARDENING_VARIABLE] = previous[_ADV_STATE_HARDENING_VARIABLE]
    history[_ADV_STATE_DILATANCY] = previous[_ADV_STATE_DILATANCY]
    return history


def _advanced_history_array(
    material: ElasticPlaneStrainMaterial,
    strain4: np.ndarray,
    state: PlasticState2D,
    source_model: str,
) -> np.ndarray:
    params = _advanced_params_array(material, source_model)
    previous = _advanced_state_array_from_vars(state.state_vars, ru_default=float(params[_ADV_PARAM_RU_INITIAL]))
    return _advanced_history_state_numba(
        np.ascontiguousarray(strain4, dtype=np.float64),
        np.ascontiguousarray(previous, dtype=np.float64),
        np.ascontiguousarray(params, dtype=np.float64),
        _advanced_model_id(source_model),
    )


def _advanced_history_dict_from_array(source_model: str, history: np.ndarray) -> dict[str, float | str]:
    out: dict[str, float | str] = {"advanced_model": source_model}
    for index, key in enumerate(_ADV_STATE_FIELDS):
        out[key] = float(history[index])
    return out


def _advanced_effective_material_from_history_array(material: ElasticPlaneStrainMaterial, history: np.ndarray) -> ElasticPlaneStrainMaterial:
    return replace(material, E=float(max(history[_ADV_STATE_EFFECTIVE_E], np.finfo(float).eps)), advanced_model="", model="elastic")


def _advanced_material_update(
    material: ElasticPlaneStrainMaterial,
    strain4: np.ndarray,
    *,
    state: PlasticState2D,
    initial_stress: np.ndarray,
    strength_factor: float,
) -> StressUpdate2D:
    source_model = material.advanced_model or material.model
    history_array = _advanced_history_array(material, strain4, state, source_model)
    history = _advanced_history_dict_from_array(source_model, history_array)
    effective = _advanced_effective_material_from_history_array(material, history_array)
    if source_model in {"nonlinear_elastic", "hardin_drnevich", "duncan_chang", "ramberg_osgood"}:
        stress = effective.D4 @ strain4 + initial_stress
        corrected, clipped, excess = _apply_tension_cutoff(stress, material)
        p, q, _mean, _dev = _stress_pq_mean_dev(corrected)
        return StressUpdate2D(corrected, clipped, float(max(excess, 0.0)), p, q, state.plastic_strain.copy(), float(max(state.kappa, history["gamma_eq"])), state_vars=history)

    strength_material = _advanced_strength_material(effective, strain4, source_model, history)
    base = update_plane_strain_stress(
        strength_material,
        strain4,
        state=state,
        initial_stress=initial_stress,
        strength_factor=strength_factor,
    )
    plastic_multiplier = max(float(base.kappa) - float(state.kappa), 0.0)
    hardening_rate = _param_float(material.advanced_params or {}, ("advanced_hardening", "pz_hardening", "hardening_rate"), max(material.hardening, 0.0))
    history["plastic_multiplier"] = plastic_multiplier
    history["hardening_variable"] = float(history.get("hardening_variable", 0.0)) + plastic_multiplier * hardening_rate
    history["dilatancy"] = _advanced_dilatancy(material, source_model, history)
    history["yield_surface"] = strength_material.model
    history["base_active_set"] = "/".join(str(value) for value in base.active_set)
    return replace(base, state_vars=history)


def _advanced_history_state(
    material: ElasticPlaneStrainMaterial,
    strain4: np.ndarray,
    state: PlasticState2D,
    source_model: str,
) -> dict[str, float | str]:
    history = _advanced_history_array(material, strain4, state, source_model)
    return _advanced_history_dict_from_array(source_model, history)


def _advanced_effective_material(material: ElasticPlaneStrainMaterial, strain4: np.ndarray, state_vars: Mapping[str, object] | None = None) -> ElasticPlaneStrainMaterial:
    params = material.advanced_params or {}
    if state_vars is None:
        history_array = _advanced_history_array(material, strain4, PlasticState2D(), material.advanced_model or material.model)
        return _advanced_effective_material_from_history_array(material, history_array)
    g0 = _param_float(params, ("G0", "Gmax", "G_max", "initial_shear_modulus"), material.shear_mu)
    ratio = _state_float(state_vars, "modulus_ratio", 1.0)
    g_eff = max(g0 * ratio, np.finfo(float).eps)
    e_eff = max(2.0 * g_eff * (1.0 + material.nu), np.finfo(float).eps)
    return replace(material, E=float(e_eff), advanced_model="", model="elastic")


def _advanced_strength_model_name(params: Mapping[str, object] | None) -> str:
    source = params or {}
    nested = source.get("strength")
    candidates: list[object] = []
    if isinstance(nested, Mapping):
        candidates.extend(
            [
                nested.get("model"),
                nested.get("strength_model"),
                nested.get("yield_surface"),
                nested.get("plastic_model"),
            ]
        )
    candidates.extend(
        [
            source.get("strength_model"),
            source.get("yield_surface"),
            source.get("plastic_model"),
            source.get("base_strength_model"),
        ]
    )
    for candidate in candidates:
        if candidate is None:
            continue
        name = str(candidate).lower().strip().replace("-", "_").replace(" ", "_")
        if name:
            return name
    return ""


def _advanced_strength_material(material: ElasticPlaneStrainMaterial, strain4: np.ndarray, source_model: str, state_vars: Mapping[str, object]) -> ElasticPlaneStrainMaterial:
    params = material.advanced_params or {}
    use_mc = _advanced_strength_model_name(params) in {"mohr_coulomb", "mc"}
    hardening_variable = _state_float(state_vars, "hardening_variable", 0.0)
    if source_model == "uw_clay":
        su = _param_float(params, ("su", "cu", "undrained_shear_strength"), material.cohesion if material.cohesion > 0.0 else max(material.yield_stress / math.sqrt(3.0), 1.0))
        if use_mc:
            return replace(material, model="mohr_coulomb", advanced_model="", cohesion=max(su + hardening_variable, 0.0), friction_angle=0.0, dilation_angle=0.0)
        return replace(material, model="drucker_prager", advanced_model="", cohesion=max(su + hardening_variable, 0.0), friction_angle=0.0, dilation_angle=0.0)
    if source_model == "pastor_zienkiewicz_clay":
        su = _param_float(params, ("su", "cu", "undrained_shear_strength"), material.cohesion if material.cohesion > 0.0 else 5.0)
        phi = _param_float(params, ("phi_cs", "critical_state_phi", "friction_angle"), material.friction_angle)
        psi = _advanced_dilatancy(material, source_model, state_vars)
        if use_mc:
            return replace(material, model="mohr_coulomb", advanced_model="", cohesion=max(su + hardening_variable, material.cohesion, 0.0), friction_angle=phi, dilation_angle=min(psi, phi))
        return replace(material, model="drucker_prager", advanced_model="", cohesion=max(su + hardening_variable, material.cohesion, 0.0), friction_angle=phi, dilation_angle=min(psi, phi))
    if source_model == "pastor_zienkiewicz_sand":
        phi = _param_float(params, ("phi_cs", "critical_state_phi", "friction_angle"), material.friction_angle if material.friction_angle > 0.0 else 32.0)
        psi = _advanced_dilatancy(material, source_model, state_vars)
        cohesion = max(material.cohesion + hardening_variable, _param_float(params, ("c", "cohesion"), 0.0))
        if use_mc:
            return replace(material, model="mohr_coulomb", advanced_model="", cohesion=cohesion, friction_angle=phi, dilation_angle=min(psi, phi))
        return replace(material, model="drucker_prager", advanced_model="", cohesion=cohesion, friction_angle=phi, dilation_angle=min(psi, phi))
    liq = params.get("liquefaction")
    liq_map = liq if isinstance(liq, Mapping) else params
    ru = _state_float(state_vars, "ru", _liquefaction_ru(liq_map))
    strength_ratio = max(_param_float(liq_map, ("residual_strength_ratio", "post_liquefaction_strength_ratio"), 0.05), 1.0 - ru)
    phi = material.friction_angle * strength_ratio
    cohesion = material.cohesion * strength_ratio
    if use_mc:
        if cohesion <= 0.0 and material.yield_stress > 0.0:
            cohesion = material.yield_stress * strength_ratio / math.sqrt(3.0)
        return replace(material, model="mohr_coulomb", advanced_model="", cohesion=max(cohesion, 0.0), friction_angle=max(phi, 0.0), dilation_angle=min(material.dilation_angle, max(phi, 0.0)))
    if material.yield_stress > 0.0:
        return replace(material, model="von_mises", advanced_model="", yield_stress=max(material.yield_stress * strength_ratio, np.finfo(float).eps))
    return replace(material, model="drucker_prager", advanced_model="", cohesion=max(cohesion, 0.0), friction_angle=max(phi, 0.0), dilation_angle=min(material.dilation_angle, max(phi, 0.0)))


def _advanced_dilatancy(material: ElasticPlaneStrainMaterial, source_model: str, state_vars: Mapping[str, object]) -> float:
    params = material.advanced_params or {}
    base = _param_float(params, ("dilation_angle", "psi"), material.dilation_angle)
    if source_model not in {"pastor_zienkiewicz_sand", "pastor_zienkiewicz_clay"}:
        return base
    gamma_eq = _state_float(state_vars, "gamma_eq", 0.0)
    gamma_ref = max(_param_float(params, ("gamma_ref", "reference_strain", "gamma50"), 1.0e-3), 1.0e-12)
    peak = _param_float(params, ("peak_dilation_angle", "psi_peak"), base)
    residual = _param_float(params, ("residual_dilation_angle", "psi_residual"), min(base, 0.0))
    weight = math.exp(-max(gamma_eq / gamma_ref, 0.0))
    return residual + (peak - residual) * weight


def _advanced_modulus_ratio(model: str, gamma_eq: float, gamma_ref: float, params: Mapping[str, object]) -> float:
    x = max(gamma_eq / gamma_ref, 0.0)
    if model in {"hardin_drnevich", "nonlinear_elastic", "uw_clay", "pastor_zienkiewicz_sand", "pastor_zienkiewicz_clay", "liquefaction", "bilinear_liquefaction"}:
        exponent = max(_param_float(params, ("exponent", "m"), 1.0), 1.0e-6)
        return 1.0 / (1.0 + x**exponent)
    if model == "ramberg_osgood":
        alpha = max(_param_float(params, ("alpha", "a"), 1.0), 0.0)
        r = max(_param_float(params, ("r", "n", "exponent"), 2.0), 1.0)
        return 1.0 / (1.0 + alpha * x ** max(r - 1.0, 0.0))
    if model == "duncan_chang":
        rf = max(0.0, min(_param_float(params, ("Rf", "failure_ratio"), 0.9), 0.99))
        n = max(_param_float(params, ("n", "exponent"), 1.0), 1.0e-6)
        return max(1.0 - rf * x / (1.0 + x), 0.0) ** n
    return 1.0


def _equivalent_shear_strain(strain4: np.ndarray) -> float:
    return float(_advanced_equivalent_shear_strain_numba(np.ascontiguousarray(strain4, dtype=np.float64)))


def _liquefaction_ru(params: Mapping[str, object]) -> float:
    direct = _param_float(params, ("ru", "excess_pore_pressure_ratio"), math.nan)
    if math.isfinite(direct):
        return max(0.0, min(direct, 0.99))
    csr = _param_float(params, ("cyclic_stress_ratio", "CSR"), 0.0)
    crr = _param_float(params, ("cyclic_resistance_ratio", "CRR", "RL20"), 0.0)
    if crr > 0.0 and csr > 0.0:
        return max(0.0, min(csr / crr - 1.0, 0.99))
    return 0.0


def _state_float(params: Mapping[str, object], key: str, default: float) -> float:
    try:
        value = float(params.get(key, default))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _param_float(params: Mapping[str, object], keys: tuple[str, ...], default: float) -> float:
    for key in keys:
        if key not in params:
            continue
        try:
            value = float(params[key])  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return default


def update_plane_strain_stress(
    material: ElasticPlaneStrainMaterial,
    strain4: np.ndarray,
    *,
    state: PlasticStateLike | None = None,
    initial_stress: np.ndarray | None = None,
    strength_factor: float = 1.0,
    diagnostic_context: tuple[Any, int] | None = None,
) -> StressUpdate2D:
    if strength_factor <= 0.0:
        raise FEM2DError("strength_factor must be positive")
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    if initial.shape != (4,):
        raise FEM2DError("initial stress must have 4 components")
    old_state = state or PlasticState2D()
    plastic_strain = np.asarray(old_state.plastic_strain, dtype=float)
    if plastic_strain.shape != (4,):
        raise FEM2DError("plastic strain state must have 4 components")
    if _is_advanced_material(material):
        return _advanced_material_update(
            material,
            strain4,
            state=old_state,
            initial_stress=initial,
            strength_factor=strength_factor,
        )
    trial = material.D4 @ (strain4 - plastic_strain) + initial
    p, q, mean, dev = _stress_pq_mean_dev(trial)
    if not _uses_plastic_strength_model(material):
        corrected, clipped, excess = _apply_tension_cutoff(trial, material)
        p_new, q_new, _mean_new, _dev_new = _stress_pq_mean_dev(corrected)
        return StressUpdate2D(corrected, clipped, float(max(excess, 0.0)), p_new, q_new, plastic_strain.copy(), float(old_state.kappa))

    if material.model in {"mohr_coulomb", "mc"}:
        return _mohr_coulomb_active_set_update(
            material,
            strain4,
            plastic_strain,
            initial,
            old_state,
            strength_factor,
            diagnostic_context,
        )

    alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
    yield_value = q - alpha * p - cohesion_term - material.hardening * old_state.kappa
    if yield_value <= max(1.0e-10, 1.0e-10 * max(q, abs(alpha * p) + cohesion_term, 1.0)):
        corrected, clipped, excess = _apply_tension_cutoff(trial, material)
        p_new, q_new, _mean_new, _dev_new = _stress_pq_mean_dev(corrected)
        return StressUpdate2D(corrected, clipped, float(max(yield_value, excess)), p_new, q_new, plastic_strain.copy(), float(old_state.kappa))

    denom = max(3.0 * material.shear_mu + material.hardening, np.finfo(float).eps)
    dgamma = max(yield_value / denom, 0.0)
    kappa_new = float(old_state.kappa + dgamma)
    q_limit = max(alpha * p + cohesion_term + material.hardening * kappa_new, 0.0)
    scale = 0.0 if q <= np.finfo(float).eps else min(q_limit / q, 1.0)
    corrected = np.array([mean + scale * dev[0], mean + scale * dev[1], mean + scale * dev[2], scale * dev[3]], dtype=float)
    corrected, clipped, excess = _apply_tension_cutoff(corrected, material)
    elastic_strain_new = np.linalg.solve(material.D4, corrected - initial)
    plastic_strain_new = strain4 - elastic_strain_new
    p_new, q_new, _mean_new, _dev_new = _stress_pq_mean_dev(corrected)
    return StressUpdate2D(corrected, True, float(max(yield_value, excess)), p_new, q_new, plastic_strain_new, kappa_new)


def _mohr_coulomb_active_set_update(
    material: ElasticPlaneStrainMaterial,
    strain4: np.ndarray,
    plastic_strain: np.ndarray,
    initial: np.ndarray,
    old_state: PlasticStateLike,
    strength_factor: float,
    diagnostic_context: tuple[Any, int] | None = None,
) -> StressUpdate2D:
    trial = material.D4 @ (strain4 - plastic_strain) + initial
    if material.mohr_coulomb_apex_policy == "rankine_cap":
        trial, _clipped, _excess = _apply_tension_cutoff(trial, material)
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    cohesion_term = 2.0 * c * math.cos(phi)
    f_trial_fast, tol_fast = _mc_trial_yield_numba(
        np.ascontiguousarray(trial, dtype=np.float64),
        math.sin(phi),
        cohesion_term,
        float(material.hardening),
        float(old_state.kappa),
    )
    if f_trial_fast <= tol_fast:
        corrected, clipped, excess = _apply_tension_cutoff(trial, material)
        p_new, q_new, _mean_new, _dev_new = _stress_pq_mean_dev(corrected)
        return StressUpdate2D(corrected, clipped, float(max(f_trial_fast, excess)), p_new, q_new, plastic_strain.copy(), float(old_state.kappa))

    tensor = _stress4_to_tensor(trial)
    sig_tr_p, vecs = np.linalg.eigh(tensor)
    yield_coeffs = _mc_plane_coeffs(phi)
    flow_coeffs = _mc_plane_coeffs(psi)
    vals_tr = yield_coeffs @ sig_tr_p - cohesion_term - material.hardening * float(old_state.kappa)
    f_trial = float(np.max(vals_tr))
    tol = _mc_yield_tol(sig_tr_p, cohesion_term)
    if f_trial <= tol:
        corrected, clipped, excess = _apply_tension_cutoff(trial, material)
        p_new, q_new, _mean_new, _dev_new = _stress_pq_mean_dev(corrected)
        return StressUpdate2D(corrected, clipped, float(max(f_trial, excess)), p_new, q_new, plastic_strain.copy(), float(old_state.kappa))

    sig_corr_p, active_ids, gamma, vals_corr = _mc_return_mapping_principal(
        sig_tr_p,
        yield_coeffs=yield_coeffs,
        flow_coeffs=flow_coeffs,
        cohesion_term=cohesion_term,
        Cn=material.D4[:3, :3],
        hardening=material.hardening,
        kappa=float(old_state.kappa),
        tol=tol,
        diagnostic_context=diagnostic_context,
        apex_policy=material.mohr_coulomb_apex_policy,
    )
    corrected_tensor = vecs @ np.diag(sig_corr_p) @ vecs.T
    corrected = _tensor_to_stress4(corrected_tensor)
    corrected, clipped, excess = _apply_tension_cutoff(corrected, material)
    elastic_strain_new = np.linalg.solve(material.D4, corrected - initial)
    plastic_strain_new = strain4 - elastic_strain_new
    dgamma = float(np.sum(gamma)) if gamma.size else 0.0
    kappa_new = float(old_state.kappa + max(dgamma, 0.0))
    residual = float(max(np.max(vals_corr), excess))
    p_new, q_new, _mean_new, _dev_new = _stress_pq_mean_dev(corrected)
    return StressUpdate2D(corrected, True, residual, p_new, q_new, plastic_strain_new, kappa_new, tuple(active_ids))


def _strength_parameter_signature(material: ElasticPlaneStrainMaterial) -> tuple[Any, ...]:
    return (
        str(material.model).lower().strip(),
        float(material.cohesion),
        float(material.friction_angle),
        float(material.dilation_angle),
        float(material.yield_stress),
    )


def _strength_factor_key(strength_factor: float) -> float:
    return round(float(strength_factor), 12)


def _mc_reduced_parameters(material: ElasticPlaneStrainMaterial, strength_factor: float) -> tuple[float, float, float]:
    return _mc_reduced_parameters_cached(_strength_parameter_signature(material), _strength_factor_key(strength_factor))


@lru_cache(maxsize=4096)
def _mc_reduced_parameters_cached(signature: tuple[Any, ...], strength_factor: float) -> tuple[float, float, float]:
    _model, cohesion, friction_angle, dilation_angle, _yield_stress = signature
    c = float(cohesion) / strength_factor
    phi0 = _angle_radians(float(friction_angle))
    psi0 = _angle_radians(float(dilation_angle))
    phi = math.atan(math.tan(phi0) / strength_factor)
    psi = math.atan(math.tan(psi0) / strength_factor)
    return c, phi, psi


def _mc_plane_coeffs(angle: float) -> np.ndarray:
    s = math.sin(angle)
    out = np.zeros((6, 3), dtype=float)
    for idx, (i, j) in enumerate(MC_ORDERED_PAIRS):
        out[idx, i] += 1.0 + s
        out[idx, j] -= 1.0 - s
    return out


def _mc_yield_tol(sig_p: np.ndarray, cohesion_term: float) -> float:
    return float(1.0e-10 * max(1.0, abs(cohesion_term), float(np.linalg.norm(sig_p))))


def _mc_candidate_active_sets(order: np.ndarray) -> Iterable[tuple[int, ...]]:
    ids = [int(i) for i in np.asarray(order, dtype=int).tolist()]
    for ia, a in enumerate(ids):
        yield (a,)
        for ib in range(ia + 1, len(ids)):
            b = ids[ib]
            yield (a, b)
            for ic in range(ib + 1, len(ids)):
                yield (a, b, ids[ic])


@lru_cache(maxsize=256)
def _mc_python_candidate_matrix_cache(
    yield_coeffs_bytes: bytes,
    flow_coeffs_bytes: bytes,
    Cn_bytes: bytes,
    hardening: float,
) -> dict[
    tuple[int, ...],
    tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        bool,
        tuple[np.ndarray, np.ndarray] | None,
        np.ndarray | None,
    ],
]:
    """Cache factor-invariant matrices used by Python Mohr-Coulomb fallback scans."""

    yield_coeffs = np.frombuffer(yield_coeffs_bytes, dtype=np.float64).reshape(6, 3)
    flow_coeffs = np.frombuffer(flow_coeffs_bytes, dtype=np.float64).reshape(6, 3)
    Cn = np.frombuffer(Cn_bytes, dtype=np.float64).reshape(3, 3)
    cached: dict[
        tuple[int, ...],
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            bool,
            tuple[np.ndarray, np.ndarray] | None,
            np.ndarray | None,
        ],
    ] = {}
    plane_ids = tuple(range(6))
    for count in (1, 2, 3):
        for subset in permutations(plane_ids, count):
            idx = np.asarray(subset, dtype=int)
            N = np.ascontiguousarray(yield_coeffs[idx, :], dtype=float)
            M = np.ascontiguousarray(flow_coeffs[idx, :], dtype=float)
            H = np.ascontiguousarray(N @ Cn @ M.T, dtype=float)
            if hardening != 0.0:
                H = H.copy()
                H.flat[:: H.shape[0] + 1] += hardening
            try:
                np.linalg.solve(H, np.zeros(count, dtype=float))
                use_lstsq = False
            except np.linalg.LinAlgError:
                use_lstsq = True
            lu_data = None if use_lstsq else lu_factor(H, check_finite=False)
            solution_operator = (
                np.linalg.lstsq(H, np.eye(count, dtype=float), rcond=None)[0]
                if use_lstsq
                else np.linalg.solve(H, np.eye(count, dtype=float))
            )
            cached[subset] = (idx, N, M, H, use_lstsq, lu_data, solution_operator)
    return cached


def _mc_python_candidate_from_gamma(
    sig_tr_p: np.ndarray,
    *,
    yield_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    subset: tuple[int, ...],
    idx: np.ndarray,
    M: np.ndarray,
    gamma: np.ndarray,
) -> tuple[
    tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray],
    tuple[float, float, float, float, int],
]:
    correction = Cn @ (M.T @ gamma)
    sig_corr = sig_tr_p - correction
    vals_corr = (
        yield_coeffs @ sig_corr
        - cohesion_term
        - hardening * (kappa + float(np.sum(gamma)))
    )
    inactive_pos = float(max(0.0, np.max(vals_corr)))
    active_res = float(np.max(np.abs(vals_corr[idx]))) if idx.size else 0.0
    neg_gamma = float(max(0.0, -np.min(gamma))) if gamma.size else 0.0
    metric = (
        inactive_pos,
        active_res,
        neg_gamma,
        float(np.linalg.norm(correction)),
        len(subset),
    )
    candidate = (
        sig_corr.copy(),
        tuple(int(i) for i in subset),
        gamma.copy(),
        vals_corr.copy(),
    )
    return candidate, metric


def _mc_shortlisted_singular_candidate_scan(
    sig_tr_p: np.ndarray,
    *,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
) -> tuple[
    tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray] | None,
    tuple[float, float, float, float, int] | None,
    tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray] | None,
    tuple[float, float, float, float, int] | None,
] | None:
    """Rank singular candidates cheaply, then verify the shortlist exactly."""

    vals_tr = yield_coeffs @ sig_tr_p - cohesion_term - hardening * kappa
    order = np.argsort(-vals_tr, kind="mergesort")
    tol_active = 10.0 * tol
    tol_gamma = 1.0e-12 * max(1.0, float(np.linalg.norm(sig_tr_p)))
    yield_coeffs_bytes = np.ascontiguousarray(yield_coeffs, dtype=np.float64).tobytes()
    flow_coeffs_bytes = np.ascontiguousarray(flow_coeffs, dtype=np.float64).tobytes()
    Cn_bytes = np.ascontiguousarray(Cn, dtype=np.float64).tobytes()
    candidate_cache = _mc_python_candidate_matrix_cache(
        yield_coeffs_bytes,
        flow_coeffs_bytes,
        Cn_bytes,
        float(hardening),
    )
    records: list[tuple[Any, ...]] = []
    for position, subset in enumerate(_mc_candidate_active_sets(order)):
        idx, _N, M, _H, use_lstsq, _lu_data, solution_operator = candidate_cache[subset]
        if solution_operator is None:
            return None
        rhs = vals_tr[idx].astype(float, copy=True)
        gamma = np.asarray(solution_operator @ rhs, dtype=float).reshape(len(subset))
        if not np.all(np.isfinite(gamma)):
            return None
        _candidate, metric = _mc_python_candidate_from_gamma(
            sig_tr_p,
            yield_coeffs=yield_coeffs,
            cohesion_term=cohesion_term,
            Cn=Cn,
            hardening=hardening,
            kappa=kappa,
            subset=subset,
            idx=idx,
            M=M,
            gamma=gamma,
        )
        valid = (
            metric[0] <= tol_active
            and metric[1] <= tol_active
            and metric[2] <= tol_gamma
        )
        records.append((position, subset, use_lstsq, metric, valid))

    selected_positions: set[int] = set()
    approximate_ranks: dict[tuple[bool, int], int] = {}
    shortlist_size = 8
    for valid in (True, False):
        ranked = sorted(
            (record for record in records if bool(record[4]) is valid),
            key=lambda record: record[3],
        )
        for rank, record in enumerate(ranked):
            approximate_ranks[(valid, int(record[0]))] = rank
            if rank < shortlist_size:
                selected_positions.add(int(record[0]))
    if not selected_positions:
        return None

    exact_records: list[tuple[int, Any, Any, bool]] = []
    for record in records:
        position, subset, use_lstsq, _approximate_metric, approximate_valid = record
        if int(position) not in selected_positions:
            continue
        idx, _N, M, H, _use_lstsq, lu_data, _solution_operator = candidate_cache[subset]
        rhs = vals_tr[idx].astype(float, copy=True)
        if use_lstsq:
            gamma, _res, _rank, _sing = np.linalg.lstsq(H, rhs, rcond=None)
        elif lu_data is not None:
            gamma = lu_solve(lu_data, rhs, check_finite=False)
        else:
            gamma = np.linalg.solve(H, rhs)
        gamma = np.asarray(gamma, dtype=float).reshape(len(subset))
        if not np.all(np.isfinite(gamma)):
            return None
        candidate, metric = _mc_python_candidate_from_gamma(
            sig_tr_p,
            yield_coeffs=yield_coeffs,
            cohesion_term=cohesion_term,
            Cn=Cn,
            hardening=hardening,
            kappa=kappa,
            subset=subset,
            idx=idx,
            M=M,
            gamma=gamma,
        )
        exact_valid = (
            metric[0] <= tol_active
            and metric[1] <= tol_active
            and metric[2] <= tol_gamma
        )
        if exact_valid != bool(approximate_valid):
            return None
        exact_records.append((int(position), candidate, metric, exact_valid))

    valid_records = [record for record in exact_records if record[3]]
    near_records = [record for record in exact_records if not record[3]]
    best_valid_record = min(valid_records, key=lambda record: record[2]) if valid_records else None
    best_near_record = min(near_records, key=lambda record: record[2]) if near_records else None
    chosen = best_valid_record if best_valid_record is not None else best_near_record
    if chosen is None:
        return None
    chosen_rank = approximate_ranks.get((bool(chosen[3]), int(chosen[0])), shortlist_size)
    if chosen_rank >= shortlist_size - 1:
        return None
    return (
        best_valid_record[1] if best_valid_record is not None else None,
        best_valid_record[2] if best_valid_record is not None else None,
        best_near_record[1] if best_near_record is not None else None,
        best_near_record[2] if best_near_record is not None else None,
    )


def _mc_return_mapping_principal_python(
    sig_tr_p: np.ndarray,
    *,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
    diagnostic_context: tuple[Any, int] | None = None,
    shortlist_singular_candidates: bool = False,
    apex_policy: str = "legacy_bounded",
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray]:
    vals_tr = yield_coeffs @ sig_tr_p - cohesion_term - hardening * kappa
    order = np.argsort(-vals_tr, kind="mergesort")
    tol_active = 10.0 * tol
    tol_gamma = 1.0e-12 * max(1.0, float(np.linalg.norm(sig_tr_p)))
    best_valid: tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray] | None = None
    best_metric: tuple[float, float, float, float, int] | None = None
    best_near: tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray] | None = None
    best_near_metric: tuple[float, float, float, float, int] | None = None
    shortlisted = (
        _mc_shortlisted_singular_candidate_scan(
            sig_tr_p,
            yield_coeffs=yield_coeffs,
            flow_coeffs=flow_coeffs,
            cohesion_term=cohesion_term,
            Cn=Cn,
            hardening=hardening,
            kappa=kappa,
            tol=tol,
        )
        if shortlist_singular_candidates
        else None
    )
    if shortlisted is not None:
        best_valid, best_metric, best_near, best_near_metric = shortlisted
    else:
        candidate_cache = _mc_python_candidate_matrix_cache(
            np.ascontiguousarray(yield_coeffs, dtype=np.float64).tobytes(),
            np.ascontiguousarray(flow_coeffs, dtype=np.float64).tobytes(),
            np.ascontiguousarray(Cn, dtype=np.float64).tobytes(),
            float(hardening),
        )
        for subset in _mc_candidate_active_sets(order):
            idx, _N, M, H, use_lstsq, lu_data, _lstsq_operator = candidate_cache[subset]
            rhs = vals_tr[idx].astype(float, copy=True)
            if use_lstsq:
                gamma, _res, _rank, _sing = np.linalg.lstsq(H, rhs, rcond=None)
            elif lu_data is not None:
                gamma = lu_solve(lu_data, rhs, check_finite=False)
            else:
                gamma = np.linalg.solve(H, rhs)
            gamma = np.asarray(gamma, dtype=float).reshape(len(subset))
            if not np.all(np.isfinite(gamma)):
                continue
            candidate, metric = _mc_python_candidate_from_gamma(
                sig_tr_p,
                yield_coeffs=yield_coeffs,
                cohesion_term=cohesion_term,
                Cn=Cn,
                hardening=hardening,
                kappa=kappa,
                subset=subset,
                idx=idx,
                M=M,
                gamma=gamma,
            )
            inactive_pos, active_res, neg_gamma, _corr_norm, _nact = metric
            if inactive_pos <= tol_active and active_res <= tol_active and neg_gamma <= tol_gamma:
                if best_metric is None or metric < best_metric:
                    best_valid = candidate
                    best_metric = metric
            elif best_near_metric is None or metric < best_near_metric:
                best_near = candidate
                best_near_metric = metric

    if best_valid is not None:
        return best_valid
    if best_near is not None and best_near_metric is not None:
        inactive_pos, active_res, neg_gamma, _corr_norm, _nact = best_near_metric
        if inactive_pos <= 100.0 * tol_active and active_res <= 100.0 * tol_active and neg_gamma <= 100.0 * tol_gamma:
            return best_near
    regularized = _mc_regularized_projection_return_mapping(
        sig_tr_p,
        yield_coeffs=yield_coeffs,
        flow_coeffs=flow_coeffs,
        cohesion_term=cohesion_term,
        Cn=Cn,
        hardening=hardening,
        kappa=kappa,
        tol=tol,
        seed=best_near,
        diagnostic_context=diagnostic_context,
        apex_policy=apex_policy,
    )
    if regularized is not None:
        return regularized
    raise FEM2DError("Mohr-Coulomb active-set return mapping failed")


def _mc_legacy_bounded_projection_return_mapping(
    sig_tr_p: np.ndarray,
    *,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
    seed: tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray] | None,
    diagnostic_context: tuple[Any, int] | None = None,
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray] | None:
    """Historical bounded cone-tip projection retained for compatibility.

    The active-set solver is exact for the usual one-to-three plane returns, but
    non-associated Mohr-Coulomb can hit near-singular corner/tension-side states.
    In a zero-dilation tension-side apex no exact non-associated return may
    exist.  Use the historical bounded sequential cone-tip projection only for
    that exceptional state.  Telemetry marks the result as a constitutive
    regularization instead of silently certifying it as the configured flow
    rule.
    """

    seed_sig = np.asarray(seed[0] if seed is not None else sig_tr_p, dtype=float)
    if seed_sig.shape != (3,) or not np.all(np.isfinite(seed_sig)):
        return None
    scale = max(1.0, abs(cohesion_term), float(np.linalg.norm(sig_tr_p)))
    relaxed_tol = max(1.0e3 * tol, 1.0e-8 * scale)
    sig = seed_sig.copy()
    seed_total_gamma = (
        float(np.sum(np.maximum(np.asarray(seed[2], dtype=float), 0.0)))
        if seed is not None and seed[2].size
        else 0.0
    )
    total_gamma = seed_total_gamma
    gamma_acc = np.zeros(6, dtype=float)
    if seed is not None:
        seed_active_ids = tuple(int(index) for index in seed[1])
        seed_gamma = np.asarray(seed[2], dtype=float).reshape(-1)
        for slot, active_id in enumerate(seed_active_ids[: seed_gamma.size]):
            if 0 <= active_id < gamma_acc.size:
                gamma_acc[active_id] += max(float(seed_gamma[slot]), 0.0)

    best_sig = sig.copy()
    best_vals = yield_coeffs @ sig - cohesion_term - hardening * (kappa + total_gamma)
    best_violation = float(max(0.0, np.max(best_vals)))
    for _iteration in range(80):
        vals = yield_coeffs @ sig - cohesion_term - hardening * (kappa + total_gamma)
        if not np.all(np.isfinite(vals)):
            return None
        max_id = int(np.argmax(vals))
        violation = float(max(0.0, vals[max_id]))
        if violation < best_violation:
            best_violation = violation
            best_sig = sig.copy()
            best_vals = vals.copy()
        if violation <= relaxed_tol:
            best_violation = violation
            best_sig = sig.copy()
            best_vals = vals.copy()
            break
        direction = Cn @ yield_coeffs[max_id, :]
        denom = float(yield_coeffs[max_id, :] @ direction + hardening)
        if not math.isfinite(denom) or denom <= np.finfo(float).eps:
            break
        dgamma = float(vals[max_id]) / denom
        if not math.isfinite(dgamma) or dgamma <= 0.0:
            break
        sig = sig - direction * dgamma
        gamma_acc[max_id] += dgamma
        total_gamma += dgamma
    if best_violation > max(relaxed_tol, 0.25 * max(1.0, abs(cohesion_term))):
        return None
    if not np.all(np.isfinite(best_sig)):
        return None
    relative_yield_violation = best_violation / scale
    _record_mohr_coulomb_fallback(
        "regularized_projection",
        diagnostic_context=diagnostic_context,
        yield_violation=best_violation,
        relative_yield_violation=relative_yield_violation,
        relaxed_tolerance=relaxed_tol,
        regularization_method=_MC_LEGACY_REGULARIZATION_METHOD,
        configured_apex_policy_verified=False,
    )
    threshold = max(tol, 1.0e-14 * max(1.0, float(np.linalg.norm(sig_tr_p))))
    active = np.flatnonzero(gamma_acc > threshold)
    if active.size:
        active = active[np.argsort(-gamma_acc[active], kind="mergesort")[:3]]
        active_ids = tuple(int(index) for index in active)
        gamma_return = gamma_acc[active].copy()
    else:
        active_ids = ()
        gamma_return = np.asarray([max(seed_total_gamma, 0.0)], dtype=float)
    return best_sig, active_ids, gamma_return, best_vals.copy()


def _mc_regularized_projection_return_mapping(
    sig_tr_p: np.ndarray,
    *,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
    seed: tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray] | None,
    diagnostic_context: tuple[Any, int] | None = None,
    apex_policy: str = "legacy_bounded",
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray] | None:
    """Apply the configured policy after the non-associated active sets fail."""

    policy = str(apex_policy).lower().strip().replace("-", "_")
    if policy == "strict_nonassociated":
        return None
    if policy == "legacy_bounded":
        return _mc_legacy_bounded_projection_return_mapping(
            sig_tr_p,
            yield_coeffs=yield_coeffs,
            flow_coeffs=flow_coeffs,
            cohesion_term=cohesion_term,
            Cn=Cn,
            hardening=hardening,
            kappa=kappa,
            tol=tol,
            seed=seed,
            diagnostic_context=diagnostic_context,
        )

    seed_sig = np.ascontiguousarray(
        seed[0] if seed is not None else sig_tr_p,
        dtype=np.float64,
    )
    seed_ids = np.full(3, -1, dtype=np.int64)
    seed_gamma = np.zeros(3, dtype=np.float64)
    seed_count = 0
    if seed is not None:
        seed_count = min(len(seed[1]), 3)
        for slot in range(seed_count):
            seed_ids[slot] = int(seed[1][slot])
            if slot < int(seed[2].size):
                seed_gamma[slot] = float(seed[2][slot])
    (
        ok,
        corrected,
        active_ids_array,
        active_count,
        gamma,
        gamma_count,
        vals_corr,
        yield_violation,
        relative_yield_violation,
        verified_tolerance,
    ) = _mc_regularized_projection_return_mapping_numba(
        np.ascontiguousarray(sig_tr_p, dtype=np.float64),
        np.ascontiguousarray(yield_coeffs, dtype=np.float64),
        np.ascontiguousarray(flow_coeffs, dtype=np.float64),
        float(cohesion_term),
        np.ascontiguousarray(Cn, dtype=np.float64),
        float(hardening),
        float(kappa),
        float(tol),
        seed_sig,
        seed_ids,
        seed_gamma,
        int(seed_count),
    )
    if not ok:
        return None
    _record_mohr_coulomb_fallback(
        "regularized_projection",
        diagnostic_context=diagnostic_context,
        yield_violation=float(yield_violation),
        relative_yield_violation=float(relative_yield_violation),
        relaxed_tolerance=float(verified_tolerance),
    )
    active_ids = tuple(
        int(active_ids_array[slot]) for slot in range(int(active_count))
    )
    return (
        corrected.copy(),
        active_ids,
        gamma[: int(gamma_count)].copy(),
        vals_corr.copy(),
    )


def _mc_return_mapping_principal(
    sig_tr_p: np.ndarray,
    *,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: float,
    Cn: np.ndarray,
    hardening: float,
    kappa: float,
    tol: float,
    diagnostic_context: tuple[Any, int] | None = None,
    apex_policy: str = "legacy_bounded",
) -> tuple[np.ndarray, tuple[int, ...], np.ndarray, np.ndarray]:
    ok, sig_corr, active_ids_array, active_count, gamma, vals_corr = _mc_return_mapping_principal_numba(
        np.ascontiguousarray(sig_tr_p, dtype=np.float64),
        np.ascontiguousarray(yield_coeffs, dtype=np.float64),
        np.ascontiguousarray(flow_coeffs, dtype=np.float64),
        float(cohesion_term),
        np.ascontiguousarray(Cn, dtype=np.float64),
        float(hardening),
        float(kappa),
        float(tol),
    )
    if ok:
        active_ids = tuple(int(active_ids_array[i]) for i in range(int(active_count)))
        return sig_corr.copy(), active_ids, gamma[: int(active_count)].copy(), vals_corr.copy()
    _record_mohr_coulomb_fallback("numba_to_python", diagnostic_context=diagnostic_context)
    friction_sine = 0.5 * abs(float(np.sum(np.asarray(yield_coeffs, dtype=float)[0, :])))
    use_shortlisted_fallback = bool(
        float(hardening) == 0.0
        and np.all(np.sum(np.asarray(flow_coeffs, dtype=float), axis=1) == 0.0)
        and friction_sine >= 0.42
    )
    return _mc_return_mapping_principal_python(
        sig_tr_p,
        yield_coeffs=yield_coeffs,
        flow_coeffs=flow_coeffs,
        cohesion_term=cohesion_term,
        Cn=Cn,
        hardening=hardening,
        kappa=kappa,
        tol=tol,
        diagnostic_context=diagnostic_context,
        shortlist_singular_candidates=use_shortlisted_fallback,
        apex_policy=apex_policy,
    )


def mohr_coulomb_principal_return_feasibility(
    material: ElasticPlaneStrainMaterial,
    trial_principal_stress: np.ndarray,
    *,
    strength_factor: float = 1.0,
    kappa: float = 0.0,
) -> dict[str, Any]:
    """Replay a material point and report strict/apex return feasibility."""

    sig_tr_p = np.ascontiguousarray(trial_principal_stress, dtype=np.float64)
    if sig_tr_p.shape != (3,) or not np.all(np.isfinite(sig_tr_p)):
        raise FEM2DError("trial principal stress must contain three finite values")
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    yield_coeffs = np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64)
    flow_coeffs = np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64)
    cohesion_term = float(2.0 * c * math.cos(phi))
    Cn = np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64)
    tol = float(_mc_yield_tol(sig_tr_p, cohesion_term))

    def _result_metrics(
        ok: bool,
        active_ids: np.ndarray,
        active_count: int,
        gamma: np.ndarray,
        values: np.ndarray,
    ) -> dict[str, Any]:
        max_violation = max(0.0, float(np.max(values)))
        active_residual = max(
            (
                abs(float(values[int(active_ids[slot])]))
                for slot in range(int(active_count))
            ),
            default=0.0,
        )
        min_gamma = min(
            (float(gamma[slot]) for slot in range(int(active_count))),
            default=0.0,
        )
        scale = max(1.0, abs(cohesion_term), float(np.linalg.norm(sig_tr_p)))
        exact_tol = max(10.0 * tol, 1.0e-12 * scale)
        exact = bool(
            ok
            and int(active_count) > 0
            and max_violation <= exact_tol
            and active_residual <= exact_tol
            and min_gamma >= -1.0e-12 * scale
        )
        return {
            "solver_returned": bool(ok),
            "exact_complementarity": exact,
            "active_planes": tuple(
                int(active_ids[slot]) for slot in range(int(active_count))
            ),
            "max_yield_violation": max_violation,
            "max_relative_yield_violation": max_violation / scale,
            "active_residual": active_residual,
            "minimum_multiplier": min_gamma,
            "verification_tolerance": exact_tol,
        }

    strict = _mc_return_mapping_principal_numba(
        sig_tr_p,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        Cn,
        float(material.hardening),
        float(kappa),
        tol,
    )
    associated = _mc_return_mapping_principal_numba(
        sig_tr_p,
        yield_coeffs,
        yield_coeffs,
        cohesion_term,
        Cn,
        float(material.hardening),
        float(kappa),
        tol,
    )
    return {
        "configured_apex_policy": material.mohr_coulomb_apex_policy,
        "strict_nonassociated": _result_metrics(
            strict[0], strict[2], strict[3], strict[4], strict[5]
        ),
        "associated_multisurface_apex": _result_metrics(
            associated[0], associated[2], associated[3], associated[4], associated[5]
        ),
    }


def _stress4_to_tensor(stress4: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [stress4[0], stress4[3], 0.0],
            [stress4[3], stress4[1], 0.0],
            [0.0, 0.0, stress4[2]],
        ],
        dtype=float,
    )


def _tensor_to_stress4(tensor: np.ndarray) -> np.ndarray:
    return np.array([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1]], dtype=float)


def algorithmic_material_tangent(
    material: ElasticPlaneStrainMaterial,
    strain4: np.ndarray,
    *,
    state: PlasticStateLike | None = None,
    initial_stress: np.ndarray | None = None,
    strength_factor: float = 1.0,
    method: str = "analytic",
) -> np.ndarray:
    method_norm = str(method or "analytic").lower().strip()
    if method_norm in {"numerical", "finite_difference", "finite-difference", "fd"}:
        return numerical_material_tangent(material, strain4, state=state, initial_stress=initial_stress, strength_factor=strength_factor)
    if method_norm not in {"analytic", "analytical", "consistent"}:
        raise FEM2DError(f"unsupported 2D material tangent method '{method}'")
    if _is_advanced_material(material):
        return numerical_material_tangent(material, strain4, state=state, initial_stress=initial_stress, strength_factor=strength_factor)
    if material.tension_cutoff and math.isfinite(float(material.tensile_strength)) and material.model in {"mohr_coulomb", "mc"}:
        return _mohr_coulomb_tension_cutoff_consistent_tangent(
            material,
            strain4,
            state=state,
            initial_stress=initial_stress,
            strength_factor=strength_factor,
        )
    if material.tension_cutoff:
        return numerical_material_tangent(material, strain4, state=state, initial_stress=initial_stress, strength_factor=strength_factor)
    if not _uses_plastic_strength_model(material):
        return material.D4
    if material.model in {"mohr_coulomb", "mc"}:
        return _mohr_coulomb_consistent_tangent(material, strain4, state=state, initial_stress=initial_stress, strength_factor=strength_factor)

    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    old_state = state or PlasticState2D()
    plastic_strain = np.asarray(old_state.plastic_strain, dtype=float)
    trial = material.D4 @ (strain4 - plastic_strain) + initial
    p, q, _mean, dev = _stress_pq_mean_dev(trial)
    alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
    yield_value = q - alpha * p - cohesion_term - material.hardening * old_state.kappa
    if yield_value <= max(1.0e-10, 1.0e-10 * max(q, abs(alpha * p) + cohesion_term, 1.0)):
        return material.D4
    if q <= np.finfo(float).eps:
        return _mean_stress_projector() @ material.D4

    denom = max(3.0 * material.shear_mu + material.hardening, np.finfo(float).eps)
    eta = material.hardening / denom
    beta = 1.0 - eta
    kappa_new = float(old_state.kappa + max(yield_value / denom, 0.0))
    q_limit = max(alpha * p + cohesion_term + material.hardening * kappa_new, 0.0)
    scale = min(q_limit / q, 1.0)
    if q_limit <= 0.0:
        dscale_dtrial = np.zeros(4, dtype=float)
        scale = 0.0
    else:
        mean_grad = _mean_stress_gradient()
        dq = _dq_dstress(dev, q)
        dscale_dtrial = (-alpha * beta * mean_grad + (eta - scale) * dq) / q

    jac_trial = _mean_stress_projector() + scale * _deviatoric_projector() + np.outer(dev, dscale_dtrial)
    return jac_trial @ material.D4


def _mohr_coulomb_consistent_tangent(
    material: ElasticPlaneStrainMaterial,
    strain4: np.ndarray,
    *,
    state: PlasticStateLike | None = None,
    initial_stress: np.ndarray | None = None,
    strength_factor: float = 1.0,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    old_state = state or PlasticState2D()
    plastic_strain = np.asarray(old_state.plastic_strain, dtype=float)
    trial = material.D4 @ (strain4 - plastic_strain) + initial
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    cohesion_term = 2.0 * c * math.cos(phi)
    f_trial_fast, tol_fast = _mc_trial_yield_numba(
        np.ascontiguousarray(trial, dtype=np.float64),
        math.sin(phi),
        cohesion_term,
        float(material.hardening),
        float(old_state.kappa),
    )
    if f_trial_fast <= tol_fast:
        return material.D4

    sig_tr_p, vecs = np.linalg.eigh(_stress4_to_tensor(trial))
    yield_coeffs = _mc_plane_coeffs(phi)
    flow_coeffs = _mc_plane_coeffs(psi)
    vals_tr = yield_coeffs @ sig_tr_p - cohesion_term - material.hardening * float(old_state.kappa)
    tol = _mc_yield_tol(sig_tr_p, cohesion_term)
    if float(np.max(vals_tr)) <= tol:
        return material.D4

    direct_mapping = _mc_return_mapping_principal_numba(
        np.ascontiguousarray(sig_tr_p, dtype=np.float64),
        np.ascontiguousarray(yield_coeffs, dtype=np.float64),
        np.ascontiguousarray(flow_coeffs, dtype=np.float64),
        float(cohesion_term),
        np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
        float(material.hardening),
        float(old_state.kappa),
        float(tol),
    )
    apex_mapping = False
    tangent_flow_coeffs = flow_coeffs
    if not bool(direct_mapping[0]):
        if material.mohr_coulomb_apex_policy != "associated_multisurface":
            return numerical_material_tangent(
                material,
                strain4,
                state=state,
                initial_stress=initial_stress,
                strength_factor=strength_factor,
            )
        apex_return = _mc_regularized_projection_return_mapping_numba(
            np.ascontiguousarray(sig_tr_p, dtype=np.float64),
            np.ascontiguousarray(yield_coeffs, dtype=np.float64),
            np.ascontiguousarray(flow_coeffs, dtype=np.float64),
            float(cohesion_term),
            np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
            float(material.hardening),
            float(old_state.kappa),
            float(tol),
            np.ascontiguousarray(direct_mapping[1], dtype=np.float64),
            np.ascontiguousarray(direct_mapping[2], dtype=np.int64),
            np.ascontiguousarray(direct_mapping[4], dtype=np.float64),
            int(direct_mapping[3]),
        )
        if not bool(apex_return[0]):
            return numerical_material_tangent(
                material,
                strain4,
                state=state,
                initial_stress=initial_stress,
                strength_factor=strength_factor,
            )
        direct_mapping = (
            apex_return[0],
            apex_return[1],
            apex_return[2],
            apex_return[3],
            apex_return[4],
            apex_return[6],
        )
        apex_mapping = True
        tangent_flow_coeffs = yield_coeffs
    sig_corr_p = np.asarray(direct_mapping[1], dtype=float)
    active_ids = tuple(
        int(direct_mapping[2][slot]) for slot in range(int(direct_mapping[3]))
    )
    if not active_ids:
        return material.D4
    active_ids_array = np.full(3, -1, dtype=np.int64)
    for idx, active_id in enumerate(active_ids[:3]):
        active_ids_array[idx] = int(active_id)
    tangent_ok, tangent = _mc_consistent_tangent_spectral_numba(
        np.ascontiguousarray(sig_tr_p, dtype=np.float64),
        np.ascontiguousarray(sig_corr_p, dtype=np.float64),
        np.ascontiguousarray(vecs, dtype=np.float64),
        active_ids_array,
        int(len(active_ids)),
        np.ascontiguousarray(yield_coeffs, dtype=np.float64),
        np.ascontiguousarray(tangent_flow_coeffs, dtype=np.float64),
        np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        float(material.hardening),
    )
    if not tangent_ok or not np.all(np.isfinite(tangent)):
        return numerical_material_tangent(material, strain4, state=state, initial_stress=initial_stress, strength_factor=strength_factor)
    if apex_mapping or abs(phi - psi) <= 1.0e-12:
        tangent = _symmetrize(tangent)
    return tangent


def _mohr_coulomb_tension_cutoff_consistent_tangent(
    material: ElasticPlaneStrainMaterial,
    strain4: np.ndarray,
    *,
    state: PlasticStateLike | None = None,
    initial_stress: np.ndarray | None = None,
    strength_factor: float = 1.0,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    old_state = state or PlasticState2D()
    plastic_strain = np.asarray(old_state.plastic_strain, dtype=float)
    if initial.shape != (4,) or plastic_strain.shape != (4,):
        return numerical_material_tangent(material, strain4, state=state, initial_stress=initial_stress, strength_factor=strength_factor)
    c, phi, psi = _mc_reduced_parameters(material, strength_factor)
    cohesion_term = 2.0 * c * math.cos(phi)
    ok, tangent = _mc_tension_cutoff_consistent_tangent_numba(
        np.ascontiguousarray(strain4, dtype=np.float64),
        np.ascontiguousarray(plastic_strain, dtype=np.float64),
        np.ascontiguousarray(material.D4, dtype=np.float64),
        np.ascontiguousarray(initial, dtype=np.float64),
        float(math.sin(phi)),
        float(cohesion_term),
        float(material.hardening),
        float(old_state.kappa),
        np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
        np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
        np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
        float(material.tensile_strength),
    )
    if ok and np.all(np.isfinite(tangent)):
        return tangent
    return numerical_material_tangent(material, strain4, state=state, initial_stress=initial_stress, strength_factor=strength_factor)


def _mc_principal_jacobian(active_ids: tuple[int, ...], yield_coeffs: np.ndarray, flow_coeffs: np.ndarray, Cn: np.ndarray, hardening: float) -> np.ndarray:
    ids = tuple(int(i) for i in active_ids)
    if not ids:
        return np.eye(3, dtype=float)
    idx = np.asarray(ids, dtype=int)
    N = yield_coeffs[idx, :]
    M = flow_coeffs[idx, :]
    H = N @ Cn @ M.T
    if hardening != 0.0:
        H = H.copy()
        H.flat[:: H.shape[0] + 1] += hardening
    try:
        Y = np.linalg.solve(H, N)
    except np.linalg.LinAlgError:
        Y, _res, _rank, _sing = np.linalg.lstsq(H, N, rcond=None)
    P = np.eye(3, dtype=float) - Cn @ M.T @ Y
    if not np.all(np.isfinite(P)):
        return np.eye(3, dtype=float)
    return P


def _mc_spectral_offdiag_limit(P: np.ndarray, i: int, j: int) -> float:
    return 0.5 * (float(P[i, i]) - float(P[i, j]) - float(P[j, i]) + float(P[j, j]))


def numerical_material_tangent(
    material: ElasticPlaneStrainMaterial,
    strain4: np.ndarray,
    *,
    state: PlasticStateLike | None = None,
    initial_stress: np.ndarray | None = None,
    strength_factor: float = 1.0,
) -> np.ndarray:
    initial = np.zeros(4, dtype=float) if initial_stress is None else np.asarray(initial_stress, dtype=float)
    old_state = state or PlasticState2D()
    if (
        material.tension_cutoff
        and math.isfinite(float(material.tensile_strength))
        and not _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and not old_state.state_vars
        and initial.shape == (4,)
    ):
        plastic_strain = np.asarray(old_state.plastic_strain, dtype=float)
        if plastic_strain.shape == (4,) and float(old_state.kappa) == 0.0 and np.allclose(plastic_strain, 0.0):
            return _elastic_tension_cutoff_numerical_tangent_numba(
                np.ascontiguousarray(strain4, dtype=np.float64),
                np.ascontiguousarray(material.D4, dtype=np.float64),
                np.ascontiguousarray(initial, dtype=np.float64),
                float(material.tensile_strength),
            )
    if (
        material.tension_cutoff
        and math.isfinite(float(material.tensile_strength))
        and not _is_advanced_material(material)
        and material.model in {"von_mises", "j2", "drucker_prager", "dp"}
        and not old_state.state_vars
        and initial.shape == (4,)
    ):
        plastic_strain = np.asarray(old_state.plastic_strain, dtype=float)
        if plastic_strain.shape == (4,):
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            return _j2dp_tension_cutoff_numerical_tangent_numba(
                np.ascontiguousarray(strain4, dtype=np.float64),
                np.ascontiguousarray(plastic_strain, dtype=np.float64),
                np.ascontiguousarray(material.D4, dtype=np.float64),
                np.ascontiguousarray(initial, dtype=np.float64),
                float(alpha),
                float(cohesion_term),
                float(material.hardening),
                float(material.shear_mu),
                float(old_state.kappa),
                float(material.tensile_strength),
            )
    if (
        material.tension_cutoff
        and math.isfinite(float(material.tensile_strength))
        and not _is_advanced_material(material)
        and material.model in {"mohr_coulomb", "mc"}
        and not old_state.state_vars
        and initial.shape == (4,)
    ):
        plastic_strain = np.asarray(old_state.plastic_strain, dtype=float)
        if plastic_strain.shape == (4,):
            c, phi, psi = _mc_reduced_parameters(material, strength_factor)
            cohesion_term = 2.0 * c * math.cos(phi)
            ok, tangent = _mc_tension_cutoff_numerical_tangent_numba(
                np.ascontiguousarray(strain4, dtype=np.float64),
                np.ascontiguousarray(plastic_strain, dtype=np.float64),
                np.ascontiguousarray(material.D4, dtype=np.float64),
                np.ascontiguousarray(initial, dtype=np.float64),
                float(math.sin(phi)),
                float(cohesion_term),
                float(material.hardening),
                float(old_state.kappa),
                np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64),
                np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64),
                np.ascontiguousarray(material.D4[:3, :3], dtype=np.float64),
                float(material.tensile_strength),
            )
            if ok and np.all(np.isfinite(tangent)):
                return tangent
    base = update_plane_strain_stress(material, strain4, state=state, initial_stress=initial_stress, strength_factor=strength_factor).stress
    delta = 1.0e-8 * max(1.0, float(np.linalg.norm(strain4)))
    tangent = np.zeros((4, 4), dtype=float)
    for i in range(4):
        perturbed = np.array(strain4, dtype=float)
        perturbed[i] += delta
        plus = update_plane_strain_stress(material, perturbed, state=state, initial_stress=initial_stress, strength_factor=strength_factor).stress
        tangent[:, i] = (plus - base) / delta
    return tangent


def _stress_pq_mean_dev(stress4: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    mean = float((stress4[0] + stress4[1] + stress4[2]) / 3.0)
    dev = np.array([stress4[0] - mean, stress4[1] - mean, stress4[2] - mean, stress4[3]], dtype=float)
    j2 = 0.5 * float(dev[0] ** 2 + dev[1] ** 2 + dev[2] ** 2) + float(dev[3] ** 2)
    q = math.sqrt(max(3.0 * j2, 0.0))
    p = -mean
    return p, q, mean, dev


def _mean_stress_gradient() -> np.ndarray:
    return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=float)


def _mean_stress_projector() -> np.ndarray:
    return np.outer(np.array([1.0, 1.0, 1.0, 0.0], dtype=float), _mean_stress_gradient())


def _deviatoric_projector() -> np.ndarray:
    return np.eye(4, dtype=float) - _mean_stress_projector()


def _dq_dstress(dev: np.ndarray, q: float) -> np.ndarray:
    if q <= np.finfo(float).eps:
        return np.zeros(4, dtype=float)
    return np.array([1.5 * dev[0] / q, 1.5 * dev[1] / q, 1.5 * dev[2] / q, 3.0 * dev[3] / q], dtype=float)


def _yield_surface_parameters(material: ElasticPlaneStrainMaterial, strength_factor: float) -> tuple[float, float]:
    return _yield_surface_parameters_cached(_strength_parameter_signature(material), _strength_factor_key(strength_factor))


@lru_cache(maxsize=4096)
def _yield_surface_parameters_cached(signature: tuple[Any, ...], strength_factor: float) -> tuple[float, float]:
    model, cohesion, friction_angle, _dilation_angle, yield_stress_raw = signature
    if model in {"von_mises", "j2"}:
        yield_stress = yield_stress_raw if yield_stress_raw > 0.0 else max(cohesion * math.sqrt(3.0), 0.0)
        if yield_stress <= 0.0:
            raise FEM2DError("von_mises material requires yield_stress or cohesion")
        return 0.0, yield_stress / strength_factor

    c = cohesion / strength_factor
    phi = _angle_radians(friction_angle)
    tan_phi_reduced = math.tan(phi) / strength_factor
    phi_reduced = math.atan(tan_phi_reduced)
    sin_phi = math.sin(phi_reduced)
    denom = max(math.sqrt(3.0) * (3.0 - sin_phi), np.finfo(float).eps)
    alpha = 2.0 * sin_phi / denom
    cohesion_term = 6.0 * c * math.cos(phi_reduced) / denom
    if cohesion_term <= 0.0 and yield_stress_raw > 0.0:
        cohesion_term = yield_stress_raw / strength_factor
    return alpha, cohesion_term


def clear_strength_parameter_cache() -> None:
    _yield_surface_parameters_cached.cache_clear()
    _mc_reduced_parameters_cached.cache_clear()


def strength_parameter_cache_info() -> dict[str, Any]:
    y = _yield_surface_parameters_cached.cache_info()
    mc = _mc_reduced_parameters_cached.cache_info()
    return {
        "yield_surface": {"hits": y.hits, "misses": y.misses, "maxsize": y.maxsize, "currsize": y.currsize},
        "mohr_coulomb": {"hits": mc.hits, "misses": mc.misses, "maxsize": mc.maxsize, "currsize": mc.currsize},
    }


def _angle_radians(value: float) -> float:
    return math.radians(value) if abs(value) > math.pi / 2.0 else value


def _material_k0(material: ElasticPlaneStrainMaterial) -> float:
    if material.k0 is not None:
        return material.k0
    return material.nu / max(1.0 - material.nu, np.finfo(float).eps)


def _plastic_state_key(element_id: str, gp_index: int) -> str:
    return f"{element_id}:{gp_index}"


def _plastic_state_for_gp(state: Mapping[str, PlasticState2D] | None, element_id: str, gp_index: int) -> PlasticState2D:
    if not state:
        return PlasticState2D()
    return state.get(_plastic_state_key(element_id, gp_index), PlasticState2D())

__all__ = [
    "principal_stresses",
    "_uses_plastic_strength_model",
    "_is_advanced_material",
    "_advanced_material_update",
    "_advanced_history_array",
    "_advanced_history_state_numba",
    "_advanced_history_state",
    "_advanced_history_dict_from_array",
    "_advanced_state_array_from_vars",
    "_advanced_params_array",
    "_advanced_model_id",
    "_advanced_strength_model_name",
    "_advanced_equivalent_shear_strain_numba",
    "_ADV_STATE_FIELDS",
    "_ADV_STATE_GAMMA_EQ",
    "_ADV_STATE_RU",
    "_ADV_STATE_MODULUS_RATIO",
    "_ADV_STATE_EFFECTIVE_E",
    "_advanced_effective_material",
    "_advanced_strength_material",
    "_equivalent_shear_strain",
    "MC_ORDERED_PAIRS",
    "_apply_tension_cutoff",
    "_principal_stresses_plane_strain_numba",
    "_tension_cutoff_plane_strain_numba",
    "_tension_cutoff_projection_jacobian_numba",
    "_elastic_tension_cutoff_numerical_tangent_numba",
    "_j2dp_tension_cutoff_stress_numba",
    "_j2dp_tension_cutoff_numerical_tangent_numba",
    "_stress4_eigh_plane_strain_numba",
    "_mc_tension_cutoff_stress_numba",
    "_mc_tension_cutoff_numerical_tangent_numba",
    "_mc_tension_cutoff_consistent_tangent_numba",
    "_mc_trial_yield_numba",
    "update_plane_strain_stress",
    "_mohr_coulomb_active_set_update",
    "_mc_reduced_parameters",
    "_mc_plane_coeffs",
    "_mc_yield_tol",
    "_mc_candidate_active_sets",
    "_mc_return_mapping_principal_numba",
    "_mc_return_mapping_principal_python",
    "_mc_return_mapping_principal",
    "mohr_coulomb_principal_return_feasibility",
    "_mc_principal_jacobian_numba",
    "_mc_principal_jacobian_precomputed_numba",
    "_mc_consistent_tangent_spectral_numba",
    "_mc_consistent_tangent_spectral_precomputed_numba",
    "_stress4_to_tensor",
    "_tensor_to_stress4",
    "algorithmic_material_tangent",
    "mohr_coulomb_adaptive_tangent_counters",
    "mohr_coulomb_fallback_telemetry",
    "reset_mohr_coulomb_fallback_telemetry",
    "_mohr_coulomb_consistent_tangent",
    "_mohr_coulomb_tension_cutoff_consistent_tangent",
    "_mc_principal_jacobian",
    "_mc_spectral_offdiag_limit",
    "numerical_material_tangent",
    "_stress_pq_mean_dev",
    "_mean_stress_gradient",
    "_mean_stress_projector",
    "_deviatoric_projector",
    "_dq_dstress",
    "_yield_surface_parameters",
    "clear_strength_parameter_cache",
    "strength_parameter_cache_info",
    "_angle_radians",
    "_material_k0",
    "_plastic_state_key",
    "_plastic_state_for_gp",
]

