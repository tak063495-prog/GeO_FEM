"""Element interpolation, integration, and QUAD4 acceleration kernels."""

from __future__ import annotations

from collections.abc import Mapping as MappingABC
import math
from typing import Any, Mapping

import numpy as np

from .fem2d_materials import (
    _ADV_MODEL_BILINEAR_LIQUEFACTION,
    _ADV_MODEL_LIQUEFACTION,
    _ADV_MODEL_PZ_CLAY,
    _ADV_MODEL_PZ_SAND,
    _ADV_MODEL_UW_CLAY,
    _ADV_PARAM_GAMMA_REF,
    _ADV_PARAM_RU_INITIAL,
    _ADV_STATE_DILATANCY,
    _ADV_STATE_EFFECTIVE_E,
    _ADV_STATE_HARDENING_VARIABLE,
    _ADV_STATE_RU,
    _is_advanced_material,
    _angle_radians,
    _advanced_history_state_numba,
    _advanced_model_id,
    _advanced_params_array,
    _advanced_strength_model_name,
    _advanced_state_array_from_vars,
    _j2dp_tension_cutoff_numerical_tangent_numba,
    _j2dp_tension_cutoff_stress_numba,
    _mc_consistent_tangent_spectral_numba,
    _mc_plane_coeffs,
    _mc_reduced_parameters,
    _mc_return_mapping_principal_numba,
    _mc_tension_cutoff_consistent_tangent_numba,
    _mc_tension_cutoff_stress_numba,
    _plastic_state_for_gp,
    _plastic_state_key,
    _param_float,
    _tension_cutoff_plane_strain_numba,
    _uses_plastic_strength_model,
    _yield_surface_parameters,
    principal_stresses,
    update_plane_strain_stress,
)
from .fem2d_types import ElasticPlaneStrainMaterial, Element2D, FEM2DError, Mesh2D, PlasticState2D, PlasticStateView2D, njit, normalize_integration, _symmetrize
from .fem2d_element_state_output import (
    _average_material_state_outputs,
    _default_material_state_output,
    _inactive_material_state_output,
    _material_state_output,
)
from .fem2d_element_fast_paths import (
    _quad4_j2dp_bbar_post_fast_path,
    _quad4_j2dp_post_fast_path,
    _quad4_mc_bbar_post_fast_path,
    _quad4_mc_post_fast_path,
    _quad4_post_state_arrays,
    _quad8_j2dp_bbar_post_fast_path,
    _quad8_j2dp_post_fast_path,
    _quad8_mc_bbar_post_fast_path,
    _quad8_mc_post_fast_path,
    _quad8_post_state_arrays,
)
from .fem2d_plastic_state_arrays import ArrayBackedPlasticStateMapping, PlasticStateArrayCache, build_plastic_state_array_cache
from .fem2d_element_interpolation import (
    axisymmetric_strain_displacement_matrix,
    integration_points,
    shape_functions,
    strain_displacement_matrix,
)
from .fem2d_element_numba_primitives import (
    _quad4_add_btcb_numba,
    _quad4_b_det_numba,
    _quad4_project_b_numba,
    _quad4_shape_grad_numba,
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
from .fem2d_element_elastic_post import (
    _quad4_elastic_bbar_post_fast,
    _quad4_elastic_bbar_post_numba,
    _quad4_elastic_post_fast,
    _quad4_elastic_post_numba,
    _quad4_post_principal_values_numba,
    _quad8_elastic_bbar_post_fast,
    _quad8_elastic_bbar_post_numba,
    _quad8_elastic_post_fast,
    _quad8_elastic_post_numba,
    _quad8_elastic_tension_bbar_post_fast,
    _quad8_elastic_tension_bbar_post_numba,
    _quad8_elastic_tension_post_fast,
    _quad8_elastic_tension_post_numba,
)
from .fem2d_element_advanced_elastic_post import (
    _quad4_advanced_elastic_stress_numba,
    _quad4_fill_advanced_post_row_numba,
    _quad4_advanced_elastic_post_numba,
    _quad4_advanced_elastic_bbar_post_numba,
    _quad4_fill_advanced_tension_post_row_numba,
    _quad4_advanced_elastic_tension_post_numba,
    _quad4_advanced_elastic_tension_bbar_post_numba,
    _quad8_fill_advanced_post_row_numba,
    _quad8_fill_advanced_tension_post_row_numba,
    _quad8_advanced_elastic_post_numba,
    _quad8_advanced_elastic_bbar_post_numba,
    _quad8_advanced_elastic_tension_post_numba,
    _quad8_advanced_elastic_tension_bbar_post_numba,
    _quad4_advanced_elastic_post_fast,
    _quad4_advanced_elastic_bbar_post_fast,
    _quad4_advanced_elastic_tension_post_fast,
    _quad4_advanced_elastic_tension_bbar_post_fast,
    _quad8_advanced_elastic_post_fast,
    _quad8_advanced_elastic_bbar_post_fast,
    _quad8_advanced_elastic_tension_post_fast,
    _quad8_advanced_elastic_tension_bbar_post_fast,
)
from .fem2d_element_elastic_kernels import (
    _quad4_axisymmetric_b_matrix_numba,
    _quad4_axisymmetric_biot_matrix_fast,
    _quad4_axisymmetric_biot_matrix_numba,
    _quad4_axisymmetric_element_stiffness_fast,
    _quad4_axisymmetric_element_stiffness_numba,
    _quad4_axisymmetric_internal_force_elastic_fast,
    _quad4_axisymmetric_internal_force_elastic_numba,
    _quad4_axisymmetric_pressure_matrices_fast,
    _quad4_axisymmetric_pressure_matrices_numba,
    _quad4_biot_matrix_fast,
    _quad4_biot_matrix_numba,
    _quad4_consistent_mass_matrix_fast,
    _quad4_consistent_mass_matrix_numba,
    _quad4_element_stiffness_fast,
    _quad4_element_stiffness_numba,
    _quad4_internal_force_elastic_fast,
    _quad4_internal_force_elastic_numba,
    _quad4_pressure_matrices_fast,
    _quad4_pressure_matrices_numba,
    _quad8_axisymmetric_b_matrix_numba,
    _quad8_axisymmetric_biot_matrix_fast,
    _quad8_axisymmetric_biot_matrix_numba,
    _quad8_axisymmetric_edge_traction_fast,
    _quad8_axisymmetric_edge_traction_numba,
    _quad8_axisymmetric_element_stiffness_fast,
    _quad8_axisymmetric_element_stiffness_numba,
    _quad8_axisymmetric_internal_force_elastic_fast,
    _quad8_axisymmetric_internal_force_elastic_numba,
    _quad8_axisymmetric_pressure_matrices_fast,
    _quad8_axisymmetric_pressure_matrices_numba,
    _quad8_biot_matrix_fast,
    _quad8_biot_matrix_numba,
    _quad8_consistent_mass_matrix_fast,
    _quad8_consistent_mass_matrix_numba,
    _quad8_element_stiffness_fast,
    _quad8_element_stiffness_numba,
    _quad8_internal_force_elastic_fast,
    _quad8_internal_force_elastic_numba,
    _quad8_pressure_matrices_fast,
    _quad8_pressure_matrices_numba,
)
from .fem2d_element_tension_cutoff_kernels import (
    _quad8_elastic_tension_tangent_force_fast,
    _quad8_elastic_tension_tangent_force_numba,
)
from .fem2d_element_j2dp_kernels import (
    _j2dp_maybe_tension_stress_tangent_numba,
    _j2dp_post_update_numba,
    _j2dp_stress_tangent_numba,
    _j2dp_tension_cutoff_post_update_numba,
    _quad4_axisymmetric_j2dp_tangent_force_fast,
    _quad4_axisymmetric_j2dp_tangent_force_numba,
    _quad4_j2dp_bbar_post_fast,
    _quad4_j2dp_bbar_post_numba,
    _quad4_j2dp_post_fast,
    _quad4_j2dp_post_numba,
    _quad4_j2dp_tangent_force_fast,
    _quad4_j2dp_tangent_force_numba,
    _quad8_axisymmetric_j2dp_bbar_post_fast,
    _quad8_axisymmetric_j2dp_bbar_post_numba,
    _quad8_axisymmetric_j2dp_post_fast,
    _quad8_axisymmetric_j2dp_post_numba,
    _quad8_axisymmetric_j2dp_tangent_force_fast,
    _quad8_axisymmetric_j2dp_tangent_force_numba,
    _quad8_j2dp_bbar_post_fast,
    _quad8_j2dp_bbar_post_numba,
    _quad8_j2dp_post_fast,
    _quad8_j2dp_post_numba,
    _quad8_j2dp_tangent_force_fast,
    _quad8_j2dp_tangent_force_numba,
)
from .fem2d_element_mohr_coulomb_kernels import (
    _quad4_mc_plane_coeffs_numba,
    _quad4_mc_principal_frame_numba,
    _quad4_mc_pq_numba,
    _quad4_mc_post_update_numba,
    _quad4_mc_tension_cutoff_post_update_numba,
    _quad4_mc_maybe_tension_post_update_numba,
    _quad4_mc_stress_tangent_numba,
    _quad4_mc_maybe_tension_stress_tangent_numba,
    _quad4_mc_tangent_force_numba,
    _quad4_mc_internal_force_numba,
    _quad8_mc_post_numba,
    _quad8_mc_bbar_post_numba,
    _quad4_mc_post_numba,
    _quad4_mc_bbar_post_numba,
    _quad8_mc_tangent_force_numba,
    _quad8_mc_internal_force_numba,
    _quad4_mc_post_fast,
    _quad4_mc_bbar_post_fast,
    _quad8_mc_post_fast,
    _quad8_mc_bbar_post_fast,
    _quad4_mc_tangent_force_fast,
    _quad4_mc_internal_force_fast,
    _quad8_mc_tangent_force_fast,
    _quad8_mc_internal_force_fast,
)
from .fem2d_element_advanced_strength_kernels import (
    _angle_radians_numba,
    _quad4_effective_d4_s4_numba,
    _advanced_strength_dilatancy_numba,
    _advanced_strength_yield_params_numba,
    _quad4_fill_advanced_strength_post_row_numba,
    _quad4_advanced_strength_j2dp_post_numba,
    _quad4_advanced_strength_j2dp_bbar_post_numba,
    _quad8_advanced_strength_j2dp_post_numba,
    _quad8_advanced_strength_j2dp_bbar_post_numba,
    _advanced_strength_j2dp_stress_numba,
    _advanced_strength_j2dp_stress_tangent_numba,
    _quad4_advanced_strength_j2dp_tangent_force_numba,
    _quad8_advanced_strength_j2dp_tangent_force_numba,
    _advanced_strength_mc_params_numba,
    _advanced_strength_mc_stress_numba,
    _advanced_strength_mc_post_update_numba,
    _advanced_strength_mc_stress_tangent_numba,
    _quad4_advanced_strength_mc_tangent_force_numba,
    _quad8_advanced_strength_mc_post_numba,
    _quad8_advanced_strength_mc_bbar_post_numba,
    _quad8_advanced_strength_mc_tangent_force_numba,
    _advanced_strength_params_array,
    _quad4_advanced_strength_j2dp_post_fast,
    _quad4_advanced_strength_j2dp_bbar_post_fast,
    _quad8_advanced_strength_j2dp_post_fast,
    _quad8_advanced_strength_j2dp_bbar_post_fast,
    _quad8_advanced_strength_mc_post_fast,
    _quad8_advanced_strength_mc_bbar_post_fast,
    _quad4_advanced_strength_j2dp_tangent_force_fast,
    _quad8_advanced_strength_j2dp_tangent_force_fast,
    _quad4_advanced_strength_mc_tangent_force_fast,
    _quad8_advanced_strength_mc_tangent_force_fast,
)
from .fem2d_element_result_rows import (
    _inactive_element_result,
    _inactive_integration_point_result as _inactive_ip_result_row,
    _quad4_elastic_post_result_rows,
    _quad4_j2dp_post_result_rows,
    _quad4_mc_post_result_rows,
    _quad8_elastic_tension_post_result_rows,
)
from .fem2d_element_fast_paths import (
    _quad4_advanced_state_arrays,
    _quad4_post_state_arrays,
    _quad8_advanced_state_arrays,
    _quad8_post_state_arrays,
)
from .fem2d_utils import _dofs_from_node_indices, _element_node_indices

_QUAD4_MODE_FULL = 0


_QUAD4_MODE_SRI = 1


_QUAD4_MODE_BBAR = 2

def _quad4_mode_code(mode: str) -> int:
    if mode == "FULL":
        return _QUAD4_MODE_FULL
    if mode == "SRI":
        return _QUAD4_MODE_SRI
    if mode == "B-BAR":
        return _QUAD4_MODE_BBAR
    raise FEM2DError(f"unsupported integration '{mode}'")































































































































































































































































































































































def element_stiffness(element_type: str, coords: np.ndarray, material: ElasticPlaneStrainMaterial, integration: str) -> np.ndarray:
    etype = element_type.upper()
    mode = normalize_integration(integration)
    if etype == "QUAD4":
        return _quad4_element_stiffness_fast(coords, material, mode)
    if etype == "QUAD8":
        return _quad8_element_stiffness_fast(coords, material, mode)
    nnode = coords.shape[0]
    ke = np.zeros((2 * nnode, 2 * nnode), dtype=float)
    full_points = integration_points(etype, "FULL")
    Pvol = material.volumetric_projector

    if mode == "FULL":
        for gp in full_points:
            B4, detJ, _N = strain_displacement_matrix(etype, coords, gp)
            dV = detJ * gp[2] * material.thickness
            ke += B4.T @ material.D4 @ B4 * dV
        return _symmetrize(ke)

    if mode == "SRI":
        for gp in full_points:
            B4, detJ, _N = strain_displacement_matrix(etype, coords, gp)
            dV = detJ * gp[2] * material.thickness
            ke += B4.T @ material.C_dev @ B4 * dV
        for gp in integration_points(etype, "REDUCED"):
            B4, detJ, _N = strain_displacement_matrix(etype, coords, gp)
            dV = detJ * gp[2] * material.thickness
            Bv = Pvol @ B4
            ke += Bv.T @ material.C_vol @ Bv * dV
        return _symmetrize(ke)

    if mode == "B-BAR":
        volume = 0.0
        Bv_acc = np.zeros((4, 2 * nnode), dtype=float)
        cached: list[tuple[np.ndarray, float]] = []
        for gp in full_points:
            B4, detJ, _N = strain_displacement_matrix(etype, coords, gp)
            dV = detJ * gp[2] * material.thickness
            volume += dV
            Bv_acc += (Pvol @ B4) * dV
            cached.append((B4, dV))
        if volume <= 0.0:
            raise FEM2DError(f"{etype}: non-positive element measure")
        Bv_bar = Bv_acc / volume
        for B4, dV in cached:
            Bdev = (np.eye(4) - Pvol) @ B4
            ke += Bdev.T @ material.C_dev @ Bdev * dV
            ke += Bv_bar.T @ material.C_vol @ Bv_bar * dV
        return _symmetrize(ke)

    raise FEM2DError(f"unsupported integration '{integration}'")


def axisymmetric_element_stiffness(element_type: str, coords: np.ndarray, material: ElasticPlaneStrainMaterial, integration: str) -> np.ndarray:
    etype = element_type.upper()
    mode = normalize_integration(integration)
    if etype == "QUAD4" and mode == "FULL":
        return _quad4_axisymmetric_element_stiffness_fast(coords, material)
    if etype == "QUAD8":
        return _quad8_axisymmetric_element_stiffness_fast(coords, material, mode)
    nnode = coords.shape[0]
    ke = np.zeros((2 * nnode, 2 * nnode), dtype=float)
    full_points = integration_points(etype, "FULL")
    Pvol = material.volumetric_projector

    if mode == "FULL":
        for gp in full_points:
            B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix(etype, coords, gp)
            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
            ke += B4.T @ material.D4 @ B4 * dV
        return _symmetrize(ke)

    if mode == "SRI":
        for gp in full_points:
            B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix(etype, coords, gp)
            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
            ke += B4.T @ material.C_dev @ B4 * dV
        for gp in integration_points(etype, "REDUCED"):
            B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix(etype, coords, gp)
            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
            Bv = Pvol @ B4
            ke += Bv.T @ material.C_vol @ Bv * dV
        return _symmetrize(ke)

    if mode == "B-BAR":
        volume = 0.0
        Bv_acc = np.zeros((4, 2 * nnode), dtype=float)
        cached: list[tuple[np.ndarray, float]] = []
        for gp in full_points:
            B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix(etype, coords, gp)
            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
            volume += dV
            Bv_acc += (Pvol @ B4) * dV
            cached.append((B4, dV))
        if volume <= 0.0:
            raise FEM2DError(f"{etype}: non-positive axisymmetric element measure")
        Bv_bar = Bv_acc / volume
        Pdev = np.eye(4) - Pvol
        for B4, dV in cached:
            Bdev = Pdev @ B4
            ke += Bdev.T @ material.C_dev @ Bdev * dV
            ke += Bv_bar.T @ material.C_vol @ Bv_bar * dV
        return _symmetrize(ke)

    raise FEM2DError(f"unsupported integration '{integration}'")


def compute_element_results(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
) -> list[dict[str, Any]]:
    rows, _state = compute_element_results_and_state(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        strength_factor=strength_factor,
        plastic_state=plastic_state,
    )
    return rows


def _fast_post_data_for_state_commit(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    coords: np.ndarray,
    ue: np.ndarray,
    initial: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    strength_factor: float,
) -> np.ndarray | None:
    mode = normalize_integration(element.integration)
    try:
        if _quad4_j2dp_bbar_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad4_post_state_arrays(element.id, plastic_state)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            return _quad4_j2dp_bbar_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
        if _quad4_j2dp_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad4_post_state_arrays(element.id, plastic_state)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            return _quad4_j2dp_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
        if _quad8_j2dp_bbar_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad8_post_state_arrays(element.id, plastic_state)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            return _quad8_j2dp_bbar_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
        if mode != "SRI" and _quad8_j2dp_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad8_post_state_arrays(element.id, plastic_state)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            return _quad8_j2dp_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
        if _quad4_mc_bbar_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad4_post_state_arrays(element.id, plastic_state)
            return _quad4_mc_bbar_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
        if _quad4_mc_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad4_post_state_arrays(element.id, plastic_state)
            return _quad4_mc_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
        if _quad8_mc_bbar_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad8_post_state_arrays(element.id, plastic_state)
            return _quad8_mc_bbar_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
        if mode != "SRI" and _quad8_mc_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad8_post_state_arrays(element.id, plastic_state)
            return _quad8_mc_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
    except FEM2DError:
        return None
    return None


def _fast_elastic_post_data(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    coords: np.ndarray,
    ue: np.ndarray,
    initial: np.ndarray,
) -> np.ndarray | None:
    if material.is_plastic:
        return None
    mode = normalize_integration(element.integration)
    try:
        if element.type == "QUAD4":
            if mode == "B-BAR":
                return _quad4_elastic_bbar_post_fast(coords, ue, material, initial_stress=initial)
            return _quad4_elastic_post_fast(coords, ue, material, initial_stress=initial)
        if element.type == "QUAD8":
            if material.tension_cutoff:
                if mode == "B-BAR":
                    return _quad8_elastic_tension_bbar_post_fast(coords, ue, material, initial_stress=initial)
                return _quad8_elastic_tension_post_fast(coords, ue, material, initial_stress=initial)
            if mode == "B-BAR":
                return _quad8_elastic_bbar_post_fast(coords, ue, material, initial_stress=initial)
            return _quad8_elastic_post_fast(coords, ue, material, initial_stress=initial)
    except FEM2DError:
        return None
    return None


def _commit_post_data_state(updated_state: dict[str, PlasticState2D], element: Element2D, data: np.ndarray) -> None:
    values = np.asarray(data, dtype=float)
    for gp_index, row_data in enumerate(values):
        updated_state[_plastic_state_key(element.id, gp_index)] = PlasticState2D(row_data[23:27].copy(), float(row_data[22]))


def _postprocess_array_cache_from_source(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
) -> PlasticStateArrayCache | None:
    if plastic_state_cache is not None:
        return plastic_state_cache
    if isinstance(plastic_state, ArrayBackedPlasticStateMapping):
        return plastic_state.to_array_cache(mesh, materials)
    if not any(material.is_plastic for material in materials.values()):
        return None
    return build_plastic_state_array_cache(mesh, materials, plastic_state)


def _commit_post_data_state_arrays(
    cache: PlasticStateArrayCache,
    strains: np.ndarray,
    kappas: np.ndarray,
    present: np.ndarray,
    state_var_flags: np.ndarray,
    state_objects: np.ndarray,
    element: Element2D,
    data: np.ndarray,
) -> int:
    row = cache.element_row.get(str(element.id))
    if row is None:
        return 0
    values = np.asarray(data, dtype=float)
    used = min(int(cache.state_point_counts[row]), values.shape[0], strains.shape[1])
    if used <= 0:
        return 0
    strains[row, :used, :] = values[:used, 23:27]
    kappas[row, :used] = values[:used, 22]
    present[row, :used] = True
    state_var_flags[row, :used] = False
    state_objects[row, :used] = None
    return int(used)


def _commit_update_state_arrays(
    cache: PlasticStateArrayCache,
    strains: np.ndarray,
    kappas: np.ndarray,
    present: np.ndarray,
    state_var_flags: np.ndarray,
    state_objects: np.ndarray,
    element: Element2D,
    gp_index: int,
    update: Any,
) -> bool:
    row = cache.element_row.get(str(element.id))
    gp = int(gp_index)
    if row is None or gp < 0 or gp >= strains.shape[1] or gp >= int(cache.state_point_counts[row]):
        return False
    strains[row, gp, :] = np.asarray(update.plastic_strain, dtype=float)
    kappas[row, gp] = float(update.kappa)
    present[row, gp] = True
    if getattr(update, "state_vars", None):
        state_var_flags[row, gp] = True
        state_objects[row, gp] = PlasticState2D(np.asarray(update.plastic_strain, dtype=float).copy(), float(update.kappa), dict(update.state_vars))
    else:
        state_var_flags[row, gp] = False
        state_objects[row, gp] = None
    return True


def _state_for_gp_from_cache_or_mapping(
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    element_id: str,
    gp_index: int,
) -> PlasticState2D | PlasticStateView2D:
    if plastic_state_cache is not None:
        cached = plastic_state_cache.state_view_for_gp(element_id, gp_index)
        if cached is not None:
            return cached
    return _plastic_state_for_gp(plastic_state, element_id, gp_index)


def compute_plastic_state_array_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    postprocess_info: dict[str, Any] | None = None,
) -> PlasticStateArrayCache:
    base = plastic_state_cache if plastic_state_cache is not None else build_plastic_state_array_cache(mesh, materials, plastic_state)
    strains = np.zeros_like(base.plastic_strains, dtype=float)
    kappas = np.zeros_like(base.kappas, dtype=float)
    present = np.zeros_like(base.present, dtype=bool)
    state_var_flags = np.zeros_like(base.state_var_flags, dtype=bool)
    state_objects = np.empty_like(base.state_objects, dtype=object)
    state_objects[:, :] = None
    array_committed = 0
    loop_committed = 0
    state_var_points = 0
    node_index = mesh.node_index
    for element in mesh.elements:
        if not element.active:
            continue
        row = base.element_row.get(str(element.id))
        if row is None:
            continue
        count = min(int(base.state_point_counts[row]), strains.shape[1])
        if count <= 0:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        initial = np.asarray(initial_stresses.get(element.id, np.zeros(4, dtype=float)), dtype=float) if initial_stresses else np.zeros(4, dtype=float)
        if initial.shape != (4,):
            raise FEM2DError(f"element {element.id}: initial stress must have 4 components")
        fast_post_data = _fast_post_data_for_state_commit(element, material, coords, ue, initial, plastic_state, strength_factor) if plastic_state_cache is None else None
        if fast_post_data is not None:
            values = np.asarray(fast_post_data, dtype=float)
            used = min(count, values.shape[0])
            if used:
                strains[row, :used, :] = values[:used, 23:27]
                kappas[row, :used] = values[:used, 22]
                present[row, :used] = True
                array_committed += 1
            continue
        mode = normalize_integration(element.integration)
        points = integration_points(element.type, "FULL")
        Pvol = material.volumetric_projector
        if mode == "B-BAR":
            volume = 0.0
            epsv_acc = np.zeros(4, dtype=float)
            cached: list[tuple[np.ndarray, float, int]] = []
            for gp_index, gp in enumerate(points):
                B4, detJ, _N = strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness
                eps = B4 @ ue
                volume += dV
                epsv_acc += (Pvol @ eps) * dV
                cached.append((eps, dV, gp_index))
            epsv_bar = epsv_acc / max(volume, np.finfo(float).eps)
            for eps, _dV, gp_index in cached:
                if gp_index >= count:
                    continue
                eps_eff = (np.eye(4) - Pvol) @ eps + epsv_bar
                old_state = _state_for_gp_from_cache_or_mapping(plastic_state, plastic_state_cache, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps_eff, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                strains[row, gp_index, :] = update.plastic_strain
                kappas[row, gp_index] = float(update.kappa)
                present[row, gp_index] = True
                if update.state_vars:
                    state_var_flags[row, gp_index] = True
                    state_objects[row, gp_index] = PlasticState2D(update.plastic_strain.copy(), float(update.kappa), dict(update.state_vars))
                    state_var_points += 1
                loop_committed += 1
        else:
            for gp_index, gp in enumerate(points):
                if gp_index >= count:
                    continue
                B4, _detJ, _N = strain_displacement_matrix(element.type, coords, gp)
                eps = B4 @ ue
                old_state = _state_for_gp_from_cache_or_mapping(plastic_state, plastic_state_cache, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                strains[row, gp_index, :] = update.plastic_strain
                kappas[row, gp_index] = float(update.kappa)
                present[row, gp_index] = True
                if update.state_vars:
                    state_var_flags[row, gp_index] = True
                    state_objects[row, gp_index] = PlasticState2D(update.plastic_strain.copy(), float(update.kappa), dict(update.state_vars))
                    state_var_points += 1
                loop_committed += 1
            if mode == "SRI":
                offset = len(points)
                for red_index, gp in enumerate(integration_points(element.type, "REDUCED")):
                    gp_index = offset + red_index
                    if gp_index >= count:
                        continue
                    B4, _detJ, _N = strain_displacement_matrix(element.type, coords, gp)
                    eps = B4 @ ue
                    old_state = _state_for_gp_from_cache_or_mapping(plastic_state, plastic_state_cache, element.id, gp_index)
                    update = update_plane_strain_stress(
                        material, eps, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                        diagnostic_context=(element.id, gp_index),
                    )
                    strains[row, gp_index, :] = update.plastic_strain
                    kappas[row, gp_index] = float(update.kappa)
                    present[row, gp_index] = True
                    if update.state_vars:
                        state_var_flags[row, gp_index] = True
                        state_objects[row, gp_index] = PlasticState2D(update.plastic_strain.copy(), float(update.kappa), dict(update.state_vars))
                        state_var_points += 1
                    loop_committed += 1
    cache = PlasticStateArrayCache(
        element_ids=base.element_ids,
        element_row=dict(base.element_row),
        state_point_counts=base.state_point_counts.copy(),
        plastic_strains=np.ascontiguousarray(strains, dtype=np.float64),
        kappas=np.ascontiguousarray(kappas, dtype=np.float64),
        present=present,
        state_var_flags=state_var_flags,
        state_objects=state_objects,
        source_state_count=int(base.source_state_count),
    )
    if postprocess_info is not None:
        postprocess_info.update(
            {
                "state_commit": "array_only",
                "array_state_elements": int(len(base.element_ids)),
                "array_state_points": int(np.sum(base.state_point_counts)) if base.state_point_counts.size else 0,
                "array_committed_elements": array_committed,
                "loop_committed_points": loop_committed,
                "state_var_points": state_var_points,
                "row_generation": "skipped",
                "plastic_ratio_source": "plastic_state_array_cache",
                "array_postprocess_enabled": True,
                "dict_materialized": False,
            }
        )
    return cache


def _element_result_row_from_post_data(element: Element2D, material: ElasticPlaneStrainMaterial, data: np.ndarray) -> dict[str, Any]:
    values = np.asarray(data, dtype=float)
    weights = np.asarray(values[:, 5], dtype=float)
    if not np.any(weights > 0.0):
        weights = np.ones(values.shape[0], dtype=float)
    eps_avg = np.average(values[:, 6:10], axis=0, weights=weights)
    sig_avg = np.average(values[:, 10:14], axis=0, weights=weights)
    principal = principal_stresses(sig_avg)
    active_sets: set[str] = set()
    if values.shape[1] >= 31:
        for row_data in values:
            active_count = max(0, min(3, int(round(float(row_data[27])))))
            active_ids = [str(int(round(float(row_data[28 + index])))) for index in range(active_count)]
            if active_ids:
                active_sets.add("/".join(active_ids))
    row = {
        "element_id": element.id,
        "active": 1.0,
        "type": element.type,
        "material": element.material,
        "integration": normalize_integration(element.integration),
        "eps_x": float(eps_avg[0]),
        "eps_y": float(eps_avg[1]),
        "eps_z": float(eps_avg[2]),
        "gamma_xy": float(eps_avg[3]),
        "sigma_x": float(sig_avg[0]),
        "sigma_y": float(sig_avg[1]),
        "sigma_z": float(sig_avg[2]),
        "tau_xy": float(sig_avg[3]),
        "sigma_1": float(principal[0]),
        "sigma_2": float(principal[1]),
        "sigma_3": float(principal[2]),
        "tau_max": float((principal[0] - principal[2]) / 2.0),
        "plastic": float(np.average(values[:, 20], weights=weights)),
        "yield_value": float(np.average(values[:, 21], weights=weights)),
        "p": float(np.average(values[:, 18], weights=weights)),
        "q": float(np.average(values[:, 19], weights=weights)),
        "active_set": ";".join(sorted(active_sets)) if active_sets else "",
    }
    row.update(_default_material_state_output(material))
    return row


def _element_result_row_from_elastic_post_data(element: Element2D, material: ElasticPlaneStrainMaterial, data: np.ndarray) -> dict[str, Any]:
    values = np.asarray(data, dtype=float)
    weights = np.asarray(values[:, 5], dtype=float)
    if not np.any(weights > 0.0):
        weights = np.ones(values.shape[0], dtype=float)
    eps_avg = np.average(values[:, 6:10], axis=0, weights=weights)
    sig_avg = np.average(values[:, 10:14], axis=0, weights=weights)
    principal = principal_stresses(sig_avg)
    clipped = values.shape[1] > 20 and np.any(values[:, 20] > 0.0)
    yield_value = float(np.average(values[:, 21], weights=weights)) if values.shape[1] > 21 else 0.0
    row = {
        "element_id": element.id,
        "active": 1.0,
        "type": element.type,
        "material": element.material,
        "integration": normalize_integration(element.integration),
        "eps_x": float(eps_avg[0]),
        "eps_y": float(eps_avg[1]),
        "eps_z": float(eps_avg[2]),
        "gamma_xy": float(eps_avg[3]),
        "sigma_x": float(sig_avg[0]),
        "sigma_y": float(sig_avg[1]),
        "sigma_z": float(sig_avg[2]),
        "tau_xy": float(sig_avg[3]),
        "sigma_1": float(principal[0]),
        "sigma_2": float(principal[1]),
        "sigma_3": float(principal[2]),
        "tau_max": float((principal[0] - principal[2]) / 2.0),
        "plastic": float(np.average(values[:, 20], weights=weights)) if values.shape[1] > 20 else 0.0,
        "yield_value": yield_value,
        "p": float(np.average(values[:, 18], weights=weights)),
        "q": float(np.average(values[:, 19], weights=weights)),
        "active_set": "tension_cutoff" if clipped else "",
    }
    row.update(_default_material_state_output(material))
    return row


class _LazyElementResultRow(MappingABC):
    __slots__ = ("_element", "_material", "_data", "_elastic", "_row")

    def __init__(self, element: Element2D, material: ElasticPlaneStrainMaterial, data: np.ndarray, *, elastic: bool) -> None:
        self._element = element
        self._material = material
        self._data = np.asarray(data, dtype=float)
        self._elastic = bool(elastic)
        self._row: dict[str, Any] | None = None

    def _materialize(self) -> dict[str, Any]:
        if self._row is None:
            if self._elastic:
                self._row = _element_result_row_from_elastic_post_data(self._element, self._material, self._data)
            else:
                self._row = _element_result_row_from_post_data(self._element, self._material, self._data)
        return self._row

    def __getitem__(self, key: str) -> Any:
        return self._materialize()[key]

    def __iter__(self):
        return iter(self._materialize())

    def __len__(self) -> int:
        return len(self._materialize())


def _integration_point_rows_from_fast_post_data(element: Element2D, material: ElasticPlaneStrainMaterial, data: np.ndarray) -> list[dict[str, Any]]:
    model = _advanced_strength_model_name(material.advanced_params or {}) if material.advanced_model else (material.model or "").lower()
    if model in {"mohr_coulomb", "mc"}:
        return _quad4_mc_post_result_rows(element, material, data)
    return _quad4_j2dp_post_result_rows(element, material, data)


def _integration_point_rows_from_elastic_post_data(element: Element2D, material: ElasticPlaneStrainMaterial, data: np.ndarray) -> list[dict[str, Any]]:
    if np.asarray(data).shape[1] > 20:
        return _quad8_elastic_tension_post_result_rows(element, material, data)
    return _quad4_elastic_post_result_rows(element, material, data)


def _integration_point_result_row_from_update(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    gp_index: int,
    gp: tuple[float, float, float],
    x: float,
    y: float,
    dV: float,
    strain: np.ndarray,
    update: Any,
) -> dict[str, Any]:
    principal = principal_stresses(update.stress)
    plastic_strain = np.asarray(update.plastic_strain, dtype=float)
    row = {
        "element_id": element.id,
        "ip": gp_index + 1,
        "state_key": _plastic_state_key(element.id, gp_index),
        "xi": float(gp[0]),
        "eta": float(gp[1]),
        "weight": float(gp[2]),
        "x": float(x),
        "y": float(y),
        "dV": float(dV),
        "active": 1.0,
        "type": element.type,
        "material": element.material,
        "integration": normalize_integration(element.integration),
        "eps_x": float(strain[0]),
        "eps_y": float(strain[1]),
        "eps_z": float(strain[2]),
        "gamma_xy": float(strain[3]),
        "sigma_x": float(update.stress[0]),
        "sigma_y": float(update.stress[1]),
        "sigma_z": float(update.stress[2]),
        "tau_xy": float(update.stress[3]),
        "sigma_1": float(principal[0]),
        "sigma_2": float(principal[1]),
        "sigma_3": float(principal[2]),
        "tau_max": float((principal[0] - principal[2]) / 2.0),
        "plastic": 1.0 if update.plastic else 0.0,
        "yield_value": float(update.yield_value),
        "p": float(update.p),
        "q": float(update.q),
        "active_set": "/".join(str(i) for i in update.active_set),
        "kappa": float(update.kappa),
        "plastic_strain_x": float(plastic_strain[0]),
        "plastic_strain_y": float(plastic_strain[1]),
        "plastic_strain_z": float(plastic_strain[2]),
        "plastic_strain_gamma_xy": float(plastic_strain[3]),
    }
    row.update(_material_state_output(material, update))
    return row


def compute_element_results_and_state(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    collect_results: bool = True,
    postprocess_info: dict[str, Any] | None = None,
    use_array_postprocess: bool = True,
    collect_integration_point_rows: bool = False,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> tuple[list[dict[str, Any]], Mapping[str, PlasticState2D]]:
    node_index = mesh.node_index
    rows: list[dict[str, Any]] = []
    integration_rows: list[dict[str, Any]] | None = [] if collect_integration_point_rows else None
    array_state_base = _postprocess_array_cache_from_source(mesh, materials, plastic_state, plastic_state_cache) if use_array_postprocess else None
    array_strains = array_state_base.plastic_strains.copy() if array_state_base is not None else None
    array_kappas = array_state_base.kappas.copy() if array_state_base is not None else None
    array_present = array_state_base.present.copy() if array_state_base is not None else None
    array_state_var_flags = array_state_base.state_var_flags.copy() if array_state_base is not None else None
    array_state_objects = array_state_base.state_objects.copy() if array_state_base is not None else None
    state_overlay = None if isinstance(plastic_state, ArrayBackedPlasticStateMapping) else plastic_state
    state_source: Mapping[str, PlasticState2D] | None = ArrayBackedPlasticStateMapping(array_state_base, state_overlay) if array_state_base is not None else plastic_state
    updated_state: dict[str, PlasticState2D] = {} if array_state_base is not None else dict(plastic_state or {})
    array_committed = 0
    array_committed_points = 0
    array_elastic_elements = 0
    array_row_elements = 0
    array_integration_row_elements = 0
    loop_committed = 0
    array_loop_committed = 0
    elastic_state_skipped_points = 0
    for element in mesh.elements:
        if not element.active:
            if collect_results:
                rows.append(_inactive_element_result(element))
            if integration_rows is not None:
                conn = _element_node_indices(element.nodes, node_index)
                coords = mesh.coords[conn]
                for gp_index, gp in enumerate(integration_points(element.type, "FULL")):
                    _B4, _detJ, N = strain_displacement_matrix(element.type, coords, gp)
                    xy = N @ coords
                    integration_rows.append(_inactive_ip_result_row(element, gp_index, gp, float(xy[0]), float(xy[1])))
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        points = integration_points(element.type, "FULL")
        strains: list[np.ndarray] = []
        stresses: list[np.ndarray] = []
        plastic_flags: list[float] = []
        yield_values: list[float] = []
        p_values: list[float] = []
        q_values: list[float] = []
        active_sets: list[str] = []
        material_states: list[dict[str, Any]] = []
        weights: list[float] = []
        Pvol = material.volumetric_projector
        initial = np.asarray(initial_stresses.get(element.id, np.zeros(4, dtype=float)), dtype=float) if initial_stresses else np.zeros(4, dtype=float)
        if initial.shape != (4,):
            raise FEM2DError(f"element {element.id}: initial stress must have 4 components")
        mode = normalize_integration(element.integration)
        elastic_post_data = _fast_elastic_post_data(element, material, coords, ue, initial) if use_array_postprocess else None
        if elastic_post_data is not None:
            array_elastic_elements += 1
            elastic_state_skipped_points += int(np.asarray(elastic_post_data).shape[0])
            if collect_results:
                rows.append(_LazyElementResultRow(element, material, elastic_post_data, elastic=True))
                array_row_elements += 1
            if integration_rows is not None:
                integration_rows.extend(_integration_point_rows_from_elastic_post_data(element, material, elastic_post_data))
                array_integration_row_elements += 1
            continue
        fast_post_data = _fast_post_data_for_state_commit(element, material, coords, ue, initial, state_source, strength_factor) if use_array_postprocess else None
        if fast_post_data is not None:
            if array_state_base is not None and array_strains is not None and array_kappas is not None and array_present is not None and array_state_var_flags is not None and array_state_objects is not None:
                array_committed_points += _commit_post_data_state_arrays(
                    array_state_base,
                    array_strains,
                    array_kappas,
                    array_present,
                    array_state_var_flags,
                    array_state_objects,
                    element,
                    fast_post_data,
                )
            else:
                _commit_post_data_state(updated_state, element, fast_post_data)
            array_committed += 1
            if collect_results:
                rows.append(_LazyElementResultRow(element, material, fast_post_data, elastic=False))
                array_row_elements += 1
            if integration_rows is not None:
                integration_rows.extend(_integration_point_rows_from_fast_post_data(element, material, fast_post_data))
                array_integration_row_elements += 1
            continue
        if mode == "B-BAR":
            volume = 0.0
            epsv_acc = np.zeros(4, dtype=float)
            cached: list[tuple[np.ndarray, float, int, tuple[float, float, float], np.ndarray]] = []
            for gp_index, gp in enumerate(points):
                B4, detJ, N = strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness
                eps = B4 @ ue
                volume += dV
                epsv_acc += (Pvol @ eps) * dV
                cached.append((eps, dV, gp_index, gp, N))
            epsv_bar = epsv_acc / max(volume, np.finfo(float).eps)
            for eps, dV, gp_index, gp, N in cached:
                eps_eff = (np.eye(4) - Pvol) @ eps + epsv_bar
                state_key = _plastic_state_key(element.id, gp_index)
                old_state = _state_for_gp_from_cache_or_mapping(state_source, array_state_base, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps_eff, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                if array_state_base is not None and array_strains is not None and array_kappas is not None and array_present is not None and array_state_var_flags is not None and array_state_objects is not None:
                    if _commit_update_state_arrays(array_state_base, array_strains, array_kappas, array_present, array_state_var_flags, array_state_objects, element, gp_index, update):
                        array_loop_committed += 1
                    else:
                        updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), update.kappa, dict(update.state_vars))
                else:
                    updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), update.kappa, dict(update.state_vars))
                loop_committed += 1
                if integration_rows is not None:
                    xy = N @ coords
                    integration_rows.append(_integration_point_result_row_from_update(element, material, gp_index, gp, float(xy[0]), float(xy[1]), dV, eps_eff, update))
                if collect_results:
                    strains.append(eps_eff)
                    stresses.append(update.stress)
                    plastic_flags.append(1.0 if update.plastic else 0.0)
                    yield_values.append(update.yield_value)
                    p_values.append(update.p)
                    q_values.append(update.q)
                    active_sets.append("/".join(str(i) for i in update.active_set))
                    material_states.append(_material_state_output(material, update))
                    weights.append(dV)
        elif mode == "SRI":
            for gp_index, gp in enumerate(points):
                B4, detJ, _N = strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness
                eps = B4 @ ue
                state_key = _plastic_state_key(element.id, gp_index)
                old_state = _state_for_gp_from_cache_or_mapping(state_source, array_state_base, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                if array_state_base is not None and array_strains is not None and array_kappas is not None and array_present is not None and array_state_var_flags is not None and array_state_objects is not None:
                    if _commit_update_state_arrays(array_state_base, array_strains, array_kappas, array_present, array_state_var_flags, array_state_objects, element, gp_index, update):
                        array_loop_committed += 1
                    else:
                        updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), update.kappa, dict(update.state_vars))
                else:
                    updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), update.kappa, dict(update.state_vars))
                loop_committed += 1
                if integration_rows is not None:
                    xy = _N @ coords
                    integration_rows.append(_integration_point_result_row_from_update(element, material, gp_index, gp, float(xy[0]), float(xy[1]), dV, eps, update))
                if collect_results:
                    strains.append(eps)
                    stresses.append(update.stress)
                    plastic_flags.append(1.0 if update.plastic else 0.0)
                    yield_values.append(update.yield_value)
                    p_values.append(update.p)
                    q_values.append(update.q)
                    active_sets.append("/".join(str(i) for i in update.active_set))
                    material_states.append(_material_state_output(material, update))
                    weights.append(dV)
            offset = len(points)
            for red_index, gp in enumerate(integration_points(element.type, "REDUCED")):
                gp_index = offset + red_index
                B4, _detJ, _N = strain_displacement_matrix(element.type, coords, gp)
                eps = B4 @ ue
                state_key = _plastic_state_key(element.id, gp_index)
                old_state = _state_for_gp_from_cache_or_mapping(state_source, array_state_base, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                if array_state_base is not None and array_strains is not None and array_kappas is not None and array_present is not None and array_state_var_flags is not None and array_state_objects is not None:
                    if _commit_update_state_arrays(array_state_base, array_strains, array_kappas, array_present, array_state_var_flags, array_state_objects, element, gp_index, update):
                        array_loop_committed += 1
                    else:
                        updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), update.kappa, dict(update.state_vars))
                else:
                    updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), update.kappa, dict(update.state_vars))
                loop_committed += 1
        else:
            for gp_index, gp in enumerate(points):
                B4, detJ, _N = strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness
                eps = B4 @ ue
                state_key = _plastic_state_key(element.id, gp_index)
                old_state = _state_for_gp_from_cache_or_mapping(state_source, array_state_base, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                if array_state_base is not None and array_strains is not None and array_kappas is not None and array_present is not None and array_state_var_flags is not None and array_state_objects is not None:
                    if _commit_update_state_arrays(array_state_base, array_strains, array_kappas, array_present, array_state_var_flags, array_state_objects, element, gp_index, update):
                        array_loop_committed += 1
                    else:
                        updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), update.kappa, dict(update.state_vars))
                else:
                    updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), update.kappa, dict(update.state_vars))
                loop_committed += 1
                if integration_rows is not None:
                    xy = _N @ coords
                    integration_rows.append(_integration_point_result_row_from_update(element, material, gp_index, gp, float(xy[0]), float(xy[1]), dV, eps, update))
                if collect_results:
                    strains.append(eps)
                    stresses.append(update.stress)
                    plastic_flags.append(1.0 if update.plastic else 0.0)
                    yield_values.append(update.yield_value)
                    p_values.append(update.p)
                    q_values.append(update.q)
                    active_sets.append("/".join(str(i) for i in update.active_set))
                    material_states.append(_material_state_output(material, update))
                    weights.append(dV)
        if not collect_results:
            continue
        w = np.asarray(weights, dtype=float)
        eps_avg = np.average(np.vstack(strains), axis=0, weights=w)
        sig_avg = np.average(np.vstack(stresses), axis=0, weights=w)
        principal = principal_stresses(sig_avg)
        row = {
                "element_id": element.id,
                "active": 1.0,
                "type": element.type,
                "material": element.material,
                "integration": normalize_integration(element.integration),
                "eps_x": float(eps_avg[0]),
                "eps_y": float(eps_avg[1]),
                "eps_z": float(eps_avg[2]),
                "gamma_xy": float(eps_avg[3]),
                "sigma_x": float(sig_avg[0]),
                "sigma_y": float(sig_avg[1]),
                "sigma_z": float(sig_avg[2]),
                "tau_xy": float(sig_avg[3]),
                "sigma_1": float(principal[0]),
                "sigma_2": float(principal[1]),
                "sigma_3": float(principal[2]),
                "tau_max": float((principal[0] - principal[2]) / 2.0),
                "plastic": float(np.average(np.asarray(plastic_flags, dtype=float), weights=w)),
                "yield_value": float(np.average(np.asarray(yield_values, dtype=float), weights=w)),
                "p": float(np.average(np.asarray(p_values, dtype=float), weights=w)),
                "q": float(np.average(np.asarray(q_values, dtype=float), weights=w)),
                "active_set": ";".join(sorted({value for value in active_sets if value})),
            }
        row.update(_average_material_state_outputs(material_states, w, material))
        rows.append(row)
    state_array_cache: PlasticStateArrayCache | None = None
    state_result: Mapping[str, PlasticState2D]
    if array_state_base is not None and array_strains is not None and array_kappas is not None and array_present is not None and array_state_var_flags is not None and array_state_objects is not None:
        state_array_cache = PlasticStateArrayCache(
            element_ids=array_state_base.element_ids,
            element_row=dict(array_state_base.element_row),
            state_point_counts=array_state_base.state_point_counts.copy(),
            plastic_strains=np.ascontiguousarray(array_strains, dtype=np.float64),
            kappas=np.ascontiguousarray(array_kappas, dtype=np.float64),
            present=array_present,
            state_var_flags=array_state_var_flags,
            state_objects=array_state_objects,
            source_state_count=int(array_state_base.source_state_count),
        )
        state_result = ArrayBackedPlasticStateMapping(state_array_cache, updated_state)
    else:
        state_result = updated_state
    if postprocess_info is not None:
        integration_source = "not_requested"
        if integration_rows is not None:
            integration_source = "array_post_data" if array_integration_row_elements else "same_pass_loop"
            postprocess_info["_integration_point_rows"] = integration_rows
        if state_array_cache is not None:
            postprocess_info["_plastic_state_array_cache"] = state_array_cache
        postprocess_info.update(
            {
                "state_commit": "array_batch" if array_committed else ("elastic_array_no_state" if array_elastic_elements else "element_loop"),
                "array_committed_elements": array_committed,
                "array_committed_points": array_committed_points,
                "array_loop_committed_points": array_loop_committed,
                "array_elastic_elements": array_elastic_elements,
                "array_row_elements": array_row_elements,
                "array_integration_row_elements": array_integration_row_elements,
                "elastic_state_skipped_points": elastic_state_skipped_points,
                "loop_committed_points": loop_committed,
                "row_generation": "array_backed_lazy" if array_row_elements else ("element_loop" if collect_results else "skipped"),
                "integration_point_row_generation": integration_source,
                "integration_point_rows": 0 if integration_rows is None else len(integration_rows),
                "integration_point_second_pass_skipped": integration_rows is not None,
                "plastic_ratio_source": "plastic_state",
                "array_postprocess_enabled": bool(use_array_postprocess),
                "state_mapping": "array_backed_lazy" if state_array_cache is not None else "dict",
                "dict_materialized": state_array_cache is None,
            }
        )
    return rows, state_result


def compute_integration_point_results(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
) -> list[dict[str, Any]]:
    from .fem2d_element_post_processing import compute_integration_point_results as _compute_integration_point_results

    return _compute_integration_point_results(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        strength_factor=strength_factor,
        plastic_state=plastic_state,
    )





__all__ = [
    "_QUAD4_MODE_FULL",
    "_QUAD4_MODE_SRI",
    "_QUAD4_MODE_BBAR",
    "_quad4_mode_code",
    "_quad4_b_det_numba",
    "_quad4_shape_grad_numba",
    "_quad4_project_b_numba",
    "_quad4_add_btcb_numba",
    "_quad4_symmetrize_numba",
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
    "_quad8_shape_grad_numba",
    "_quad8_b_det_numba",
    "_quad8_project_b_numba",
    "_quad8_add_btcb_numba",
    "_quad8_add_btlcbr_numba",
    "_quad8_add_btstress_numba",
    "_quad8_symmetrize_numba",
    "_quad8_element_stiffness_numba",
    "_quad8_consistent_mass_matrix_numba",
    "_quad8_pressure_matrices_numba",
    "_quad8_biot_matrix_numba",
    "_quad8_internal_force_elastic_numba",
    "_quad8_elastic_tension_tangent_force_numba",
    "_quad8_elastic_post_numba",
    "_quad8_elastic_bbar_post_numba",
    "_quad8_elastic_tension_post_numba",
    "_quad8_elastic_tension_bbar_post_numba",
    "_quad8_axisymmetric_b_matrix_numba",
    "_quad8_axisymmetric_element_stiffness_numba",
    "_quad8_axisymmetric_internal_force_elastic_numba",
    "_quad8_axisymmetric_pressure_matrices_numba",
    "_quad8_axisymmetric_biot_matrix_numba",
    "_quad8_axisymmetric_edge_traction_numba",
    "_quad8_axisymmetric_j2dp_tangent_force_numba",
    "_quad8_axisymmetric_j2dp_post_numba",
    "_quad8_axisymmetric_j2dp_bbar_post_numba",
    "_quad4_elastic_post_numba",
    "_quad4_elastic_bbar_post_numba",
    "_quad4_advanced_elastic_post_numba",
    "_quad4_advanced_elastic_bbar_post_numba",
    "_quad4_advanced_elastic_tension_post_numba",
    "_quad4_advanced_elastic_tension_bbar_post_numba",
    "_quad8_advanced_elastic_post_numba",
    "_quad8_advanced_elastic_bbar_post_numba",
    "_quad8_advanced_elastic_tension_post_numba",
    "_quad8_advanced_elastic_tension_bbar_post_numba",
    "_quad4_advanced_strength_j2dp_post_numba",
    "_quad4_advanced_strength_j2dp_bbar_post_numba",
    "_quad8_advanced_strength_j2dp_post_numba",
    "_quad8_advanced_strength_j2dp_bbar_post_numba",
    "_quad4_advanced_strength_j2dp_tangent_force_numba",
    "_quad8_advanced_strength_j2dp_tangent_force_numba",
    "_quad4_advanced_strength_mc_tangent_force_numba",
    "_quad8_advanced_strength_mc_post_numba",
    "_quad8_advanced_strength_mc_bbar_post_numba",
    "_quad8_advanced_strength_mc_tangent_force_numba",
    "_quad4_j2dp_post_numba",
    "_quad4_j2dp_bbar_post_numba",
    "_quad8_j2dp_post_numba",
    "_quad8_j2dp_bbar_post_numba",
    "_quad8_mc_post_numba",
    "_quad8_mc_bbar_post_numba",
    "_quad4_mc_post_numba",
    "_quad4_mc_bbar_post_numba",
    "_j2dp_stress_tangent_numba",
    "_quad4_mc_stress_tangent_numba",
    "_quad4_mc_tangent_force_numba",
    "_quad4_mc_internal_force_numba",
    "_quad8_mc_tangent_force_numba",
    "_quad8_mc_internal_force_numba",
    "_quad4_j2dp_tangent_force_numba",
    "_quad8_j2dp_tangent_force_numba",
    "_quad4_axisymmetric_j2dp_tangent_force_numba",
    "_quad4_element_stiffness_fast",
    "_quad4_pressure_matrices_fast",
    "_quad4_biot_matrix_fast",
    "_quad4_consistent_mass_matrix_fast",
    "_quad4_axisymmetric_pressure_matrices_fast",
    "_quad4_axisymmetric_biot_matrix_fast",
    "_quad4_axisymmetric_element_stiffness_fast",
    "_quad4_axisymmetric_internal_force_elastic_fast",
    "_quad4_internal_force_elastic_fast",
    "_quad8_element_stiffness_fast",
    "_quad8_pressure_matrices_fast",
    "_quad8_biot_matrix_fast",
    "_quad8_consistent_mass_matrix_fast",
    "_quad8_internal_force_elastic_fast",
    "_quad8_elastic_tension_tangent_force_fast",
    "_quad8_j2dp_tangent_force_fast",
    "_quad8_mc_tangent_force_fast",
    "_quad8_mc_internal_force_fast",
    "_quad8_elastic_post_fast",
    "_quad8_elastic_bbar_post_fast",
    "_quad8_elastic_tension_post_fast",
    "_quad8_elastic_tension_bbar_post_fast",
    "_quad8_axisymmetric_element_stiffness_fast",
    "_quad8_axisymmetric_pressure_matrices_fast",
    "_quad8_axisymmetric_biot_matrix_fast",
    "_quad8_axisymmetric_internal_force_elastic_fast",
    "_quad8_axisymmetric_edge_traction_fast",
    "_quad8_axisymmetric_j2dp_tangent_force_fast",
    "_quad8_axisymmetric_j2dp_post_fast",
    "_quad8_axisymmetric_j2dp_bbar_post_fast",
    "_quad4_elastic_post_fast",
    "_quad4_elastic_bbar_post_fast",
    "_quad4_advanced_elastic_post_fast",
    "_quad4_advanced_elastic_bbar_post_fast",
    "_quad4_advanced_elastic_tension_post_fast",
    "_quad4_advanced_elastic_tension_bbar_post_fast",
    "_quad8_advanced_elastic_post_fast",
    "_quad8_advanced_elastic_bbar_post_fast",
    "_quad8_advanced_elastic_tension_post_fast",
    "_quad8_advanced_elastic_tension_bbar_post_fast",
    "_quad4_advanced_strength_j2dp_post_fast",
    "_quad4_advanced_strength_j2dp_bbar_post_fast",
    "_quad8_advanced_strength_j2dp_post_fast",
    "_quad8_advanced_strength_j2dp_bbar_post_fast",
    "_quad4_advanced_strength_j2dp_tangent_force_fast",
    "_quad8_advanced_strength_j2dp_tangent_force_fast",
    "_quad4_advanced_strength_mc_tangent_force_fast",
    "_quad8_advanced_strength_mc_post_fast",
    "_quad8_advanced_strength_mc_bbar_post_fast",
    "_quad8_advanced_strength_mc_tangent_force_fast",
    "_quad4_j2dp_post_fast",
    "_quad4_j2dp_bbar_post_fast",
    "_quad8_j2dp_post_fast",
    "_quad8_j2dp_bbar_post_fast",
    "_quad8_mc_post_fast",
    "_quad8_mc_bbar_post_fast",
    "_quad4_mc_post_fast",
    "_quad4_mc_bbar_post_fast",
    "_quad4_mc_tangent_force_fast",
    "_quad4_mc_internal_force_fast",
    "_quad4_j2dp_tangent_force_fast",
    "_quad4_axisymmetric_j2dp_tangent_force_fast",
    "element_stiffness",
    "axisymmetric_element_stiffness",
    "compute_element_results",
    "compute_element_results_and_state",
    "compute_plastic_state_array_cache",
    "compute_integration_point_results",
    "_inactive_element_result",
    "strain_displacement_matrix",
    "axisymmetric_strain_displacement_matrix",
    "shape_functions",
    "integration_points",
]

