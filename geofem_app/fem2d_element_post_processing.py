"""Integration-point post-processing orchestration for FEM2D elements."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from . import fem2d_elements as _element_kernels
from .fem2d_element_elastic_post import (
    _quad4_elastic_bbar_post_fast,
    _quad4_elastic_post_fast,
    _quad8_elastic_bbar_post_fast,
    _quad8_elastic_post_fast,
    _quad8_elastic_tension_bbar_post_fast,
    _quad8_elastic_tension_post_fast,
)
from .fem2d_element_fast_paths import (
    _quad4_advanced_elastic_bbar_post_fast_path,
    _quad4_advanced_elastic_post_fast_path,
    _quad4_advanced_elastic_tension_bbar_post_fast_path,
    _quad4_advanced_elastic_tension_post_fast_path,
    _quad4_advanced_strength_j2dp_bbar_post_fast_path,
    _quad4_advanced_strength_j2dp_post_fast_path,
    _quad4_elastic_bbar_post_fast_path,
    _quad4_elastic_post_fast_path,
    _quad4_j2dp_bbar_post_fast_path,
    _quad4_j2dp_post_fast_path,
    _quad4_mc_bbar_post_fast_path,
    _quad4_mc_post_fast_path,
    _quad4_post_state_arrays,
    _quad8_advanced_elastic_bbar_post_fast_path,
    _quad8_advanced_elastic_post_fast_path,
    _quad8_advanced_elastic_tension_bbar_post_fast_path,
    _quad8_advanced_elastic_tension_post_fast_path,
    _quad8_advanced_strength_j2dp_bbar_post_fast_path,
    _quad8_advanced_strength_j2dp_post_fast_path,
    _quad8_advanced_strength_mc_bbar_post_fast_path,
    _quad8_advanced_strength_mc_post_fast_path,
    _quad8_elastic_bbar_post_fast_path,
    _quad8_elastic_post_fast_path,
    _quad8_elastic_tension_bbar_post_fast_path,
    _quad8_elastic_tension_post_fast_path,
    _quad8_j2dp_bbar_post_fast_path,
    _quad8_j2dp_post_fast_path,
    _quad8_mc_bbar_post_fast_path,
    _quad8_mc_post_fast_path,
    _quad8_post_state_arrays,
)
from .fem2d_element_interpolation import integration_points, strain_displacement_matrix
from .fem2d_element_result_rows import (
    _inactive_integration_point_result,
    _integration_point_result_row,
    _quad4_advanced_elastic_post_result_rows,
    _quad4_advanced_elastic_tension_post_result_rows,
    _quad4_advanced_strength_j2dp_post_result_rows,
    _quad4_elastic_post_result_rows,
    _quad4_j2dp_post_result_rows,
    _quad4_mc_post_result_rows,
    _quad8_elastic_tension_post_result_rows,
)
from .fem2d_materials import _yield_surface_parameters
from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, Mesh2D, PlasticState2D, normalize_integration
from .fem2d_utils import _dofs_from_node_indices, _element_node_indices


ELEMENT_POST_PROCESSING_FUNCTIONS = (
    "element_post_processing_contract",
    "compute_integration_point_results",
)


def element_post_processing_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_post_processing.v1",
        "module": "geofem_app.fem2d_element_post_processing",
        "function_count": len(ELEMENT_POST_PROCESSING_FUNCTIONS),
        "functions": list(ELEMENT_POST_PROCESSING_FUNCTIONS),
        "covered_surfaces": [
            "integration_point_post_orchestration",
            "fast_post_kernel_dispatch",
            "bbar_fallback_post_processing",
            "inactive_integration_point_rows",
        ],
        "kernel_provider": "geofem_app.fem2d_elements",
    }


def compute_integration_point_results(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
) -> list[dict[str, Any]]:
    node_index = mesh.node_index
    rows: list[dict[str, Any]] = []
    for element in mesh.elements:
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        points = integration_points(element.type, "FULL")
        if not element.active:
            for gp_index, gp in enumerate(points):
                _B4, _detJ, N = strain_displacement_matrix(element.type, coords, gp)
                xy = N @ coords
                rows.append(_inactive_integration_point_result(element, gp_index, gp, float(xy[0]), float(xy[1])))
            continue
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        Pvol = material.volumetric_projector
        initial = np.asarray(initial_stresses.get(element.id, np.zeros(4, dtype=float)), dtype=float) if initial_stresses else np.zeros(4, dtype=float)
        if initial.shape != (4,):
            raise FEM2DError(f"element {element.id}: initial stress must have 4 components")
        if _quad4_advanced_elastic_tension_bbar_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_elastic_tension_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad4_advanced_elastic_tension_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    ),
                )
            )
            continue
        if _quad4_advanced_elastic_tension_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_elastic_tension_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad4_advanced_elastic_tension_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    ),
                )
            )
            continue
        if _quad8_advanced_elastic_tension_bbar_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_elastic_tension_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad8_advanced_elastic_tension_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    ),
                )
            )
            continue
        if _quad8_advanced_elastic_tension_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_elastic_tension_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad8_advanced_elastic_tension_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    ),
                )
            )
            continue
        if _quad8_elastic_tension_bbar_post_fast_path(element, material, plastic_state):
            rows.extend(
                _quad8_elastic_tension_post_result_rows(
                    element,
                    material,
                    _quad8_elastic_tension_bbar_post_fast(coords, ue, material, initial),
                )
            )
            continue
        if _quad8_elastic_tension_post_fast_path(element, material, plastic_state):
            rows.extend(
                _quad8_elastic_tension_post_result_rows(
                    element,
                    material,
                    _quad8_elastic_tension_post_fast(coords, ue, material, initial),
                )
            )
            continue
        if _quad4_advanced_strength_j2dp_bbar_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_strength_j2dp_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad4_advanced_strength_j2dp_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                        strength_factor=strength_factor,
                    ),
                )
            )
            continue
        if _quad4_advanced_strength_j2dp_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_strength_j2dp_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad4_advanced_strength_j2dp_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                        strength_factor=strength_factor,
                    ),
                )
            )
            continue
        if _quad8_advanced_strength_mc_bbar_post_fast_path(element, material):
            data = _element_kernels._quad8_advanced_strength_mc_bbar_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            if data is not None:
                rows.extend(_quad4_advanced_strength_j2dp_post_result_rows(element, material, data))
                continue
        if _quad8_advanced_strength_mc_post_fast_path(element, material):
            data = _element_kernels._quad8_advanced_strength_mc_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            if data is not None:
                rows.extend(_quad4_advanced_strength_j2dp_post_result_rows(element, material, data))
                continue
        if _quad8_advanced_strength_j2dp_bbar_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_strength_j2dp_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad8_advanced_strength_j2dp_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                        strength_factor=strength_factor,
                    ),
                )
            )
            continue
        if _quad8_advanced_strength_j2dp_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_strength_j2dp_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad8_advanced_strength_j2dp_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                        strength_factor=strength_factor,
                    ),
                )
            )
            continue
        if _quad4_advanced_elastic_bbar_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_elastic_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad4_advanced_elastic_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    ),
                )
            )
            continue
        if _quad4_advanced_elastic_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_elastic_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad4_advanced_elastic_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    ),
                )
            )
            continue
        if _quad8_advanced_elastic_bbar_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_elastic_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad8_advanced_elastic_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    ),
                )
            )
            continue
        if _quad8_advanced_elastic_post_fast_path(element, material):
            rows.extend(
                _quad4_advanced_elastic_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad8_advanced_elastic_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_state=plastic_state,
                        element_id=element.id,
                    ),
                )
            )
            continue
        if _quad4_elastic_bbar_post_fast_path(element, material, plastic_state):
            rows.extend(_quad4_elastic_post_result_rows(element, material, _quad4_elastic_bbar_post_fast(coords, ue, material, initial)))
            continue
        if _quad4_elastic_post_fast_path(element, material, plastic_state):
            rows.extend(_quad4_elastic_post_result_rows(element, material, _quad4_elastic_post_fast(coords, ue, material, initial)))
            continue
        if _quad8_elastic_bbar_post_fast_path(element, material, plastic_state):
            rows.extend(_quad4_elastic_post_result_rows(element, material, _quad8_elastic_bbar_post_fast(coords, ue, material, initial)))
            continue
        if _quad8_elastic_post_fast_path(element, material, plastic_state):
            rows.extend(_quad4_elastic_post_result_rows(element, material, _quad8_elastic_post_fast(coords, ue, material, initial)))
            continue
        if _quad8_j2dp_bbar_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad8_post_state_arrays(element.id, plastic_state)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            rows.extend(
                _quad4_j2dp_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad8_j2dp_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_strains=plastic_strains,
                        kappas=kappas,
                        alpha=alpha,
                        cohesion_term=cohesion_term,
                    ),
                )
            )
            continue
        if _quad8_j2dp_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad8_post_state_arrays(element.id, plastic_state)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            rows.extend(
                _quad4_j2dp_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad8_j2dp_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_strains=plastic_strains,
                        kappas=kappas,
                        alpha=alpha,
                        cohesion_term=cohesion_term,
                    ),
                )
            )
            continue
        if _quad8_mc_bbar_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad8_post_state_arrays(element.id, plastic_state)
            data = _element_kernels._quad8_mc_bbar_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if data is not None:
                rows.extend(_quad4_mc_post_result_rows(element, material, data))
                continue
        if _quad8_mc_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad8_post_state_arrays(element.id, plastic_state)
            data = _element_kernels._quad8_mc_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if data is not None:
                rows.extend(_quad4_mc_post_result_rows(element, material, data))
                continue
        if _quad4_j2dp_bbar_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad4_post_state_arrays(element.id, plastic_state)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            rows.extend(
                _quad4_j2dp_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad4_j2dp_bbar_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_strains=plastic_strains,
                        kappas=kappas,
                        alpha=alpha,
                        cohesion_term=cohesion_term,
                    ),
                )
            )
            continue
        if _quad4_j2dp_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad4_post_state_arrays(element.id, plastic_state)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            rows.extend(
                _quad4_j2dp_post_result_rows(
                    element,
                    material,
                    _element_kernels._quad4_j2dp_post_fast(
                        coords,
                        ue,
                        material,
                        initial_stress=initial,
                        plastic_strains=plastic_strains,
                        kappas=kappas,
                        alpha=alpha,
                        cohesion_term=cohesion_term,
                    ),
                )
            )
            continue
        if _quad4_mc_bbar_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad4_post_state_arrays(element.id, plastic_state)
            data = _element_kernels._quad4_mc_bbar_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if data is not None:
                rows.extend(_quad4_mc_post_result_rows(element, material, data))
                continue
        if _quad4_mc_post_fast_path(element, material, plastic_state):
            plastic_strains, kappas = _quad4_post_state_arrays(element.id, plastic_state)
            data = _element_kernels._quad4_mc_post_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if data is not None:
                rows.extend(_quad4_mc_post_result_rows(element, material, data))
                continue
        if normalize_integration(element.integration) == "B-BAR":
            volume = 0.0
            epsv_acc = np.zeros(4, dtype=float)
            cached: list[tuple[int, tuple[float, float, float], np.ndarray, float, np.ndarray]] = []
            for gp_index, gp in enumerate(points):
                B4, detJ, N = strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness
                eps = B4 @ ue
                volume += dV
                epsv_acc += (Pvol @ eps) * dV
                cached.append((gp_index, gp, eps, dV, N))
            epsv_bar = epsv_acc / max(volume, np.finfo(float).eps)
            for gp_index, gp, eps, dV, N in cached:
                eps_eff = (np.eye(4) - Pvol) @ eps + epsv_bar
                xy = N @ coords
                rows.append(
                    _integration_point_result_row(
                        element,
                        material,
                        gp_index,
                        gp,
                        float(xy[0]),
                        float(xy[1]),
                        dV,
                        eps_eff,
                        initial,
                        strength_factor,
                        plastic_state,
                    )
                )
        else:
            for gp_index, gp in enumerate(points):
                B4, detJ, N = strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness
                eps = B4 @ ue
                xy = N @ coords
                rows.append(
                    _integration_point_result_row(
                        element,
                        material,
                        gp_index,
                        gp,
                        float(xy[0]),
                        float(xy[1]),
                        dV,
                        eps,
                        initial,
                        strength_factor,
                        plastic_state,
                    )
                )
    return rows


__all__ = [
    "element_post_processing_contract",
    "compute_integration_point_results",
]
