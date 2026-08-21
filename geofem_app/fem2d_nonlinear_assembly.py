"""Nonlinear tangent and internal force assembly helpers for 2D FEM solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.sparse import csr_matrix

from .fem2d_elements import (
    _quad4_advanced_strength_j2dp_tangent_force_fast,
    _quad4_advanced_strength_mc_tangent_force_fast,
    _quad4_axisymmetric_element_stiffness_fast,
    _quad4_axisymmetric_internal_force_elastic_fast,
    _quad4_axisymmetric_j2dp_tangent_force_fast,
    _quad4_element_stiffness_fast,
    _quad4_internal_force_elastic_fast,
    _quad4_j2dp_tangent_force_fast,
    _quad4_mc_internal_force_fast,
    _quad4_mc_tangent_force_fast,
    _quad8_advanced_strength_j2dp_tangent_force_fast,
    _quad8_advanced_strength_mc_tangent_force_fast,
    _quad8_axisymmetric_element_stiffness_fast,
    _quad8_axisymmetric_internal_force_elastic_fast,
    _quad8_axisymmetric_j2dp_tangent_force_fast,
    _quad8_elastic_tension_tangent_force_fast,
    _quad8_element_stiffness_fast,
    _quad8_internal_force_elastic_fast,
    _quad8_j2dp_tangent_force_fast,
    _quad8_mc_internal_force_fast,
    _quad8_mc_tangent_force_fast,
    axisymmetric_strain_displacement_matrix,
    integration_points,
    strain_displacement_matrix,
)
from .fem2d_interfaces import interface_force_tangent
from .fem2d_materials import (
    _advanced_strength_model_name,
    _is_advanced_material,
    _plastic_state_for_gp,
    _plastic_state_key,
    _uses_plastic_strength_model,
    _yield_surface_parameters,
    algorithmic_material_tangent,
    update_plane_strain_stress,
)
from .fem2d_plastic_batch import (
    PLASTIC_BATCH_STATUS_OK,
    MohrCoulombActiveSetCache,
    Quad4MCGeometryCache,
    build_plastic_element_blocks,
    evaluate_mc_internal_force_candidates,
    evaluate_plastic_tangent_block,
)
from .fem2d_plastic_state_arrays import PlasticStateArrayCache, build_plastic_state_array_cache
from .fem2d_structural import (
    structural_element_dofs,
    structural_element_force_tangent,
    structural_rotation_dof_map,
    structural_total_dofs,
)
from .fem2d_types import FEM2DError, ElasticPlaneStrainMaterial, Interface2D, Mesh2D, PlasticState2D, PlasticStateView2D, StructuralElement2D, normalize_integration
from .fem2d_utils import _dofs_from_node_indices, _element_dofs, _element_node_indices
from .sparse_assembly import SparseAssemblyBuilder, SparseAssemblyPattern

_ZERO_INITIAL_STRESS = np.zeros(4, dtype=float)


@dataclass(frozen=True)
class InitialStressArrayCache:
    values: np.ndarray
    element_index_to_row: np.ndarray
    element_ids: tuple[str, ...]
    present_elements: int
    layout: str = "active_element_major"

    def solver_info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "layout": self.layout,
            "active_elements": int(self.values.shape[0]),
            "present_elements": int(self.present_elements),
            "component_count": int(self.values.shape[1]) if self.values.ndim == 2 else 0,
            "index_map_size": int(self.element_index_to_row.size),
        }


NONLINEAR_ASSEMBLY_FUNCTIONS = (
    "InitialStressArrayCache",
    "build_initial_stress_array_cache",
    "initial_stress_array_cache_info",
    "_axisymmetric_quad4_elastic_fast_path",
    "_quad4_elastic_fast_path",
    "_quad8_elastic_tension_fast_path",
    "_quad4_j2dp_fast_path",
    "_quad4_advanced_strength_j2dp_fast_path",
    "_quad4_advanced_strength_mc_fast_path",
    "_quad4_mc_fast_path",
    "_quad4_plastic_state_arrays",
    "_quad8_state_point_count",
    "_quad8_j2dp_fast_path",
    "_quad8_j2dp_tension_fast_path",
    "_quad8_mc_fast_path",
    "_quad8_plastic_state_arrays",
    "_axisymmetric_quad4_j2dp_fast_path",
    "_axisymmetric_quad4_state_arrays",
    "_axisymmetric_quad8_j2dp_fast_path",
    "_axisymmetric_quad8_state_arrays",
    "_axisymmetric_quad8_post_state_arrays",
    "_quad4_sri_bbar_plastic_batch_blocks",
    "_element_gp_kinematics",
    "_algorithmic_tangent_block",
    "_internal_force_block",
    "_tangent_internal_block",
    "assemble_axisymmetric_algorithmic_tangent_stiffness",
    "assemble_axisymmetric_internal_force",
    "assemble_axisymmetric_tangent_and_internal_force",
    "assemble_algorithmic_tangent_stiffness",
    "assemble_tangent_and_internal_force",
    "assemble_internal_force_candidates",
    "assemble_internal_force",
)


def build_initial_stress_array_cache(
    mesh: Mesh2D,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    *,
    active_element_ids: list[str] | tuple[str, ...] | None = None,
) -> InitialStressArrayCache:
    element_index_by_id = {str(element.id): index for index, element in enumerate(mesh.elements)}
    if active_element_ids is None:
        ordered_ids = tuple(str(element.id) for element in mesh.elements if element.active)
    else:
        ordered_ids = tuple(str(element_id) for element_id in active_element_ids)
    values = np.zeros((len(ordered_ids), 4), dtype=np.float64)
    element_index_to_row = np.full(len(mesh.elements), -1, dtype=np.int64)
    present = 0
    for row, element_id in enumerate(ordered_ids):
        element_index = element_index_by_id.get(element_id)
        if element_index is None:
            raise FEM2DError(f"initial stress cache references unknown element '{element_id}'")
        element_index_to_row[element_index] = row
        if initial_stresses and element_id in initial_stresses:
            initial = np.asarray(initial_stresses[element_id], dtype=float)
            if initial.shape != (4,):
                raise FEM2DError(f"element {element_id}: initial stress must have 4 components")
            values[row, :] = initial
            present += 1
    return InitialStressArrayCache(
        values=np.ascontiguousarray(values, dtype=np.float64),
        element_index_to_row=element_index_to_row,
        element_ids=ordered_ids,
        present_elements=present,
    )


def initial_stress_array_cache_info(cache: InitialStressArrayCache | None) -> dict[str, Any]:
    if cache is None:
        return {"enabled": False}
    return cache.solver_info()


def _initial_stress_for_element(
    element_index: int,
    element_id: str,
    initial_stresses: Mapping[str, np.ndarray] | None,
    initial_stress_cache: InitialStressArrayCache | None = None,
) -> np.ndarray:
    if initial_stress_cache is not None:
        if 0 <= element_index < initial_stress_cache.element_index_to_row.size:
            row = int(initial_stress_cache.element_index_to_row[element_index])
            if row >= 0:
                return initial_stress_cache.values[row]
        return _ZERO_INITIAL_STRESS
    if initial_stresses:
        initial = np.asarray(initial_stresses.get(element_id, _ZERO_INITIAL_STRESS), dtype=float)
        if initial.shape != (4,):
            raise FEM2DError(f"element {element_id}: initial stress must have 4 components")
        return initial
    return _ZERO_INITIAL_STRESS


def nonlinear_assembly_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.nonlinear_assembly.v1",
        "module": "geofem_app.fem2d_nonlinear_assembly",
        "function_count": len(NONLINEAR_ASSEMBLY_FUNCTIONS),
        "functions": list(NONLINEAR_ASSEMBLY_FUNCTIONS),
        "covered_surfaces": [
            "plane_strain_algorithmic_tangent",
            "plane_strain_internal_force",
            "plane_strain_combined_tangent_internal",
            "initial_stress_active_element_array_cache",
            "axisymmetric_algorithmic_tangent",
            "axisymmetric_internal_force",
            "fast_path_selection",
            "plastic_state_arrays",
            "interface_and_structural_element_coupling",
        ],
    }


def _axisymmetric_quad4_elastic_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    return (
        tangent_norm in {"analytic", "analytical", "consistent"}
        and not plastic_state
        and not _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and not bool(material.tension_cutoff)
    )


def _quad4_elastic_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    return (
        tangent_norm in {"analytic", "analytical", "consistent"}
        and not plastic_state
        and not _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and not bool(material.tension_cutoff)
    )


def _quad8_elastic_tension_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    return (
        tangent_norm in {"analytic", "analytical", "consistent"}
        and not plastic_state
        and not _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and bool(material.tension_cutoff)
        and math.isfinite(float(material.tensile_strength))
    )


def _quad4_j2dp_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
    element_id: str | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    model = str(material.model).lower().strip()
    if tangent_norm not in {"analytic", "analytical", "consistent"}:
        return False
    if _is_advanced_material(material) or bool(material.tension_cutoff):
        return False
    if model not in {"von_mises", "j2", "drucker_prager", "dp"}:
        return False
    if element_id is not None and _element_has_state_vars(plastic_state, element_id, 4, plastic_state_cache):
        return False
    return True


def _quad4_advanced_strength_j2dp_fast_path(
    material: ElasticPlaneStrainMaterial,
    *,
    tangent_method: str = "analytic",
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    source_model = str(material.advanced_model or material.model).lower().strip()
    return (
        tangent_norm in {"analytic", "analytical", "consistent", "numerical", "finite_difference", "finite-difference", "fd"}
        and _is_advanced_material(material)
        and (not bool(material.tension_cutoff) or math.isfinite(float(material.tensile_strength)))
        and _advanced_strength_model_name(material.advanced_params or {}) not in {"mohr_coulomb", "mc"}
        and source_model in {"uw_clay", "pastor_zienkiewicz_sand", "pastor_zienkiewicz_clay", "liquefaction", "bilinear_liquefaction"}
    )


def _quad4_advanced_strength_mc_fast_path(
    material: ElasticPlaneStrainMaterial,
    *,
    tangent_method: str = "analytic",
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    source_model = str(material.advanced_model or material.model).lower().strip()
    return (
        tangent_norm in {"analytic", "analytical", "consistent", "numerical", "finite_difference", "finite-difference", "fd"}
        and _is_advanced_material(material)
        and (not bool(material.tension_cutoff) or math.isfinite(float(material.tensile_strength)))
        and _advanced_strength_model_name(material.advanced_params or {}) in {"mohr_coulomb", "mc"}
        and source_model in {"uw_clay", "pastor_zienkiewicz_sand", "pastor_zienkiewicz_clay", "liquefaction", "bilinear_liquefaction"}
    )


def _quad4_mc_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
    element_id: str | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    model = str(material.model).lower().strip()
    if tangent_norm not in {"analytic", "analytical", "consistent"}:
        return False
    if _is_advanced_material(material) or bool(material.tension_cutoff):
        return False
    if model not in {"mohr_coulomb", "mc"}:
        return False
    if element_id is not None and _element_has_state_vars(plastic_state, element_id, 4, plastic_state_cache):
        return False
    return True


def _quad4_plastic_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if plastic_state_cache is not None:
        return plastic_state_cache.state_arrays(element_id, 4)
    plastic_strains = np.zeros((4, 4), dtype=float)
    kappas = np.zeros(4, dtype=float)
    if not plastic_state:
        return plastic_strains, kappas
    for gp_index in range(4):
        state = plastic_state.get(_plastic_state_key(element_id, gp_index))
        if state is None:
            continue
        strain = np.asarray(state.plastic_strain, dtype=float)
        if strain.shape == (4,):
            plastic_strains[gp_index, :] = strain
        kappas[gp_index] = float(state.kappa)
    return plastic_strains, kappas


def _quad8_state_point_count(integration: str) -> int:
    mode = normalize_integration(integration)
    return 13 if mode == "SRI" else 9


def _quad8_j2dp_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
    element_id: str | None = None,
    integration: str = "FULL",
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    model = str(material.model).lower().strip()
    if tangent_norm not in {"analytic", "analytical", "consistent"}:
        return False
    if _is_advanced_material(material) or bool(material.tension_cutoff):
        return False
    if model not in {"von_mises", "j2", "drucker_prager", "dp"}:
        return False
    if element_id is not None and _element_has_state_vars(plastic_state, element_id, _quad8_state_point_count(integration), plastic_state_cache):
        return False
    return True


def _quad8_j2dp_tension_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
    element_id: str | None = None,
    integration: str = "FULL",
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    model = str(material.model).lower().strip()
    if tangent_norm not in {"analytic", "analytical", "consistent"}:
        return False
    if _is_advanced_material(material) or not bool(material.tension_cutoff) or not math.isfinite(float(material.tensile_strength)):
        return False
    if model not in {"von_mises", "j2", "drucker_prager", "dp"}:
        return False
    if element_id is not None and _element_has_state_vars(plastic_state, element_id, _quad8_state_point_count(integration), plastic_state_cache):
        return False
    return True


def _quad8_mc_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
    element_id: str | None = None,
    integration: str = "FULL",
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> bool:
    tangent_norm = str(tangent_method or "analytic").lower().strip()
    model = str(material.model).lower().strip()
    if tangent_norm not in {"analytic", "analytical", "consistent"}:
        return False
    if _is_advanced_material(material):
        return False
    if bool(material.tension_cutoff) and not math.isfinite(float(material.tensile_strength)):
        return False
    if model not in {"mohr_coulomb", "mc"}:
        return False
    if element_id is not None and _element_has_state_vars(plastic_state, element_id, _quad8_state_point_count(integration), plastic_state_cache):
        return False
    return True


def _quad8_plastic_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
    integration: str,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    count = _quad8_state_point_count(integration)
    if plastic_state_cache is not None:
        return plastic_state_cache.state_arrays(element_id, count)
    plastic_strains = np.zeros((count, 4), dtype=float)
    kappas = np.zeros(count, dtype=float)
    if not plastic_state:
        return plastic_strains, kappas
    for gp_index in range(count):
        state = plastic_state.get(_plastic_state_key(element_id, gp_index))
        if state is None:
            continue
        strain = np.asarray(state.plastic_strain, dtype=float)
        if strain.shape == (4,):
            plastic_strains[gp_index, :] = strain
        kappas[gp_index] = float(state.kappa)
    return plastic_strains, kappas


def _axisymmetric_quad4_j2dp_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
    element_id: str | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> bool:
    return _quad4_j2dp_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element_id, plastic_state_cache=plastic_state_cache)


def _axisymmetric_quad4_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _quad4_plastic_state_arrays(element_id, plastic_state, plastic_state_cache)


def _axisymmetric_quad8_j2dp_fast_path(
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    *,
    tangent_method: str = "analytic",
    element_id: str | None = None,
    integration: str = "FULL",
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> bool:
    return _quad8_j2dp_fast_path(
        material,
        plastic_state,
        tangent_method=tangent_method,
        element_id=element_id,
        integration=integration,
        plastic_state_cache=plastic_state_cache,
    )


def _axisymmetric_quad8_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
    integration: str,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _quad8_plastic_state_arrays(element_id, plastic_state, integration, plastic_state_cache)


def _axisymmetric_quad8_post_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    return _quad8_plastic_state_arrays(element_id, plastic_state, "FULL", plastic_state_cache)


def _element_has_state_vars(
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    point_count: int,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> bool:
    if plastic_state_cache is not None:
        return plastic_state_cache.has_state_vars(element_id, point_count)
    if not plastic_state:
        return False
    for gp_index in range(point_count):
        state = plastic_state.get(_plastic_state_key(element_id, gp_index))
        if state is not None and state.state_vars:
            return True
    return False


def _plastic_state_for_gp_cached(
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    gp_index: int,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> PlasticState2D | PlasticStateView2D | None:
    if plastic_state_cache is not None:
        return plastic_state_cache.state_view_for_gp(element_id, gp_index)
    return _plastic_state_for_gp(plastic_state, element_id, gp_index)


def _quad4_sri_bbar_plastic_batch_blocks(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
    strength_factor: float,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    initial_stress_cache: InitialStressArrayCache | None = None,
    mohr_coulomb_active_set_cache: MohrCoulombActiveSetCache | None = None,
    quad4_mc_geometry_cache: Quad4MCGeometryCache | None = None,
    tangent_method: str = "analytic",
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if str(tangent_method or "analytic").lower().strip() not in {"analytic", "analytical", "consistent"}:
        return {}
    batched: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for block in build_plastic_element_blocks(mesh, materials):
        if block.element_type != "QUAD4" or block.integration not in {"SRI", "B-BAR"}:
            continue
        if block.material_model not in {"j2", "drucker_prager", "mohr_coulomb"} or block.tension_cutoff:
            continue
        result = evaluate_plastic_tangent_block(
            block,
            mesh,
            materials,
            u,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            strength_factor=strength_factor,
            mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
            quad4_mc_geometry_cache=quad4_mc_geometry_cache,
        )
        for row, element_index in enumerate(block.element_indices):
            if result.status_flags[row] != PLASTIC_BATCH_STATUS_OK:
                continue
            dofs = np.asarray(result.dofs[row], dtype=np.int64)
            dof_count = int(dofs.size)
            batched[int(element_index)] = (
                dofs,
                np.asarray(result.ke_values[row], dtype=np.float64).reshape((dof_count, dof_count)),
                np.asarray(result.internal_force_values[row], dtype=np.float64).reshape((dof_count,)),
            )
    return batched


def _element_gp_kinematics(
    element_type: str,
    coords: np.ndarray,
    gp: tuple[float, float, float],
    material: ElasticPlaneStrainMaterial,
    *,
    axisymmetric: bool = False,
) -> tuple[np.ndarray, float]:
    if axisymmetric:
        B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix(element_type, coords, gp)
        dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
    else:
        B4, detJ, _N = strain_displacement_matrix(element_type, coords, gp)
        dV = detJ * gp[2] * material.thickness
    return B4, dV


def _algorithmic_tangent_block(
    element: Any,
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial: np.ndarray,
    *,
    strength_factor: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
    tangent_method: str,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    axisymmetric: bool = False,
) -> np.ndarray:
    mode = normalize_integration(element.integration)
    full_points = integration_points(element.type, "FULL")
    ke = np.zeros((ue.size, ue.size), dtype=float)
    if mode == "B-BAR":
        Pvol = material.volumetric_projector
        Pdev = np.eye(4) - Pvol
        volume = 0.0
        Bv_acc = np.zeros((4, ue.size), dtype=float)
        cached: list[tuple[int, np.ndarray, float]] = []
        for gp_index, gp in enumerate(full_points):
            B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
            volume += dV
            Bv_acc += (Pvol @ B4) * dV
            cached.append((gp_index, B4, dV))
        if volume <= 0.0:
            raise FEM2DError(f"{element.type}: non-positive element measure")
        Bv_bar = Bv_acc / volume
        for gp_index, B4, dV in cached:
            B_eff = Pdev @ B4 + Bv_bar
            strain = B_eff @ ue
            state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
            tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor, method=tangent_method)
            ke += B_eff.T @ tangent @ B_eff * dV
        return ke

    if mode == "SRI":
        Pvol = material.volumetric_projector
        Pdev = np.eye(4) - Pvol
        for gp_index, gp in enumerate(full_points):
            B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
            strain = B4 @ ue
            state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
            tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor, method=tangent_method)
            Bdev = Pdev @ B4
            ke += Bdev.T @ tangent @ B4 * dV
        offset = len(full_points)
        for red_index, gp in enumerate(integration_points(element.type, "REDUCED")):
            gp_index = offset + red_index
            B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
            strain = B4 @ ue
            state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
            tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor, method=tangent_method)
            Bv = Pvol @ B4
            ke += Bv.T @ tangent @ B4 * dV
        return ke

    for gp_index, gp in enumerate(full_points):
        B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
        strain = B4 @ ue
        state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
        tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor, method=tangent_method)
        ke += B4.T @ tangent @ B4 * dV
    return ke


def _internal_force_block(
    element: Any,
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial: np.ndarray,
    *,
    strength_factor: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    axisymmetric: bool = False,
) -> np.ndarray:
    mode = normalize_integration(element.integration)
    full_points = integration_points(element.type, "FULL")
    fe = np.zeros(ue.size, dtype=float)
    if mode == "B-BAR":
        Pvol = material.volumetric_projector
        Pdev = np.eye(4) - Pvol
        volume = 0.0
        Bv_acc = np.zeros((4, ue.size), dtype=float)
        cached: list[tuple[int, np.ndarray, float]] = []
        for gp_index, gp in enumerate(full_points):
            B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
            volume += dV
            Bv_acc += (Pvol @ B4) * dV
            cached.append((gp_index, B4, dV))
        if volume <= 0.0:
            raise FEM2DError(f"{element.type}: non-positive element measure")
        Bv_bar = Bv_acc / volume
        for gp_index, B4, dV in cached:
            B_eff = Pdev @ B4 + Bv_bar
            eps = B_eff @ ue
            state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
            update = update_plane_strain_stress(
                material, eps, state=state, initial_stress=initial, strength_factor=strength_factor,
                diagnostic_context=(element.id, gp_index),
            )
            fe += B_eff.T @ update.stress * dV
        return fe

    if mode == "SRI":
        Pvol = material.volumetric_projector
        Pdev = np.eye(4) - Pvol
        for gp_index, gp in enumerate(full_points):
            B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
            eps = B4 @ ue
            state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
            update = update_plane_strain_stress(
                material, eps, state=state, initial_stress=initial, strength_factor=strength_factor,
                diagnostic_context=(element.id, gp_index),
            )
            Bdev = Pdev @ B4
            fe += Bdev.T @ update.stress * dV
        offset = len(full_points)
        for red_index, gp in enumerate(integration_points(element.type, "REDUCED")):
            gp_index = offset + red_index
            B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
            eps = B4 @ ue
            state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
            update = update_plane_strain_stress(
                material, eps, state=state, initial_stress=initial, strength_factor=strength_factor,
                diagnostic_context=(element.id, gp_index),
            )
            Bv = Pvol @ B4
            fe += Bv.T @ update.stress * dV
        return fe

    for gp_index, gp in enumerate(full_points):
        B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
        eps = B4 @ ue
        state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
        update = update_plane_strain_stress(
            material, eps, state=state, initial_stress=initial, strength_factor=strength_factor,
            diagnostic_context=(element.id, gp_index),
        )
        fe += B4.T @ update.stress * dV
    return fe


def _tangent_internal_block(
    element: Any,
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial: np.ndarray,
    *,
    strength_factor: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
    tangent_method: str,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    axisymmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    mode = normalize_integration(element.integration)
    full_points = integration_points(element.type, "FULL")
    ke = np.zeros((ue.size, ue.size), dtype=float)
    fe = np.zeros(ue.size, dtype=float)
    if mode == "B-BAR":
        Pvol = material.volumetric_projector
        Pdev = np.eye(4) - Pvol
        volume = 0.0
        Bv_acc = np.zeros((4, ue.size), dtype=float)
        cached: list[tuple[int, np.ndarray, float]] = []
        for gp_index, gp in enumerate(full_points):
            B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
            volume += dV
            Bv_acc += (Pvol @ B4) * dV
            cached.append((gp_index, B4, dV))
        if volume <= 0.0:
            raise FEM2DError(f"{element.type}: non-positive element measure")
        Bv_bar = Bv_acc / volume
        for gp_index, B4, dV in cached:
            B_eff = Pdev @ B4 + Bv_bar
            strain = B_eff @ ue
            state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
            update = update_plane_strain_stress(
                material, strain, state=state, initial_stress=initial, strength_factor=strength_factor,
                diagnostic_context=(element.id, gp_index),
            )
            tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor, method=tangent_method)
            fe += B_eff.T @ update.stress * dV
            ke += B_eff.T @ tangent @ B_eff * dV
        return ke, fe

    if mode == "SRI":
        Pvol = material.volumetric_projector
        Pdev = np.eye(4) - Pvol
        for gp_index, gp in enumerate(full_points):
            B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
            strain = B4 @ ue
            state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
            update = update_plane_strain_stress(
                material, strain, state=state, initial_stress=initial, strength_factor=strength_factor,
                diagnostic_context=(element.id, gp_index),
            )
            tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor, method=tangent_method)
            Bdev = Pdev @ B4
            fe += Bdev.T @ update.stress * dV
            ke += Bdev.T @ tangent @ B4 * dV
        offset = len(full_points)
        for red_index, gp in enumerate(integration_points(element.type, "REDUCED")):
            gp_index = offset + red_index
            B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
            strain = B4 @ ue
            state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
            update = update_plane_strain_stress(
                material, strain, state=state, initial_stress=initial, strength_factor=strength_factor,
                diagnostic_context=(element.id, gp_index),
            )
            tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor, method=tangent_method)
            Bv = Pvol @ B4
            fe += Bv.T @ update.stress * dV
            ke += Bv.T @ tangent @ B4 * dV
        return ke, fe

    for gp_index, gp in enumerate(full_points):
        B4, dV = _element_gp_kinematics(element.type, coords, gp, material, axisymmetric=axisymmetric)
        strain = B4 @ ue
        state = _plastic_state_for_gp_cached(plastic_state, element.id, gp_index, plastic_state_cache)
        update = update_plane_strain_stress(
            material, strain, state=state, initial_stress=initial, strength_factor=strength_factor,
            diagnostic_context=(element.id, gp_index),
        )
        tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor, method=tangent_method)
        fe += B4.T @ update.stress * dV
        ke += B4.T @ tangent @ B4 * dV
    return ke, fe


def assemble_axisymmetric_algorithmic_tangent_stiffness(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: InitialStressArrayCache | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    tangent_method: str = "analytic",
    sparse_pattern: SparseAssemblyPattern | None = None,
) -> csr_matrix:
    if plastic_state_cache is None and plastic_state:
        plastic_state_cache = build_plastic_state_array_cache(mesh, materials, plastic_state)
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements, axisymmetric=True)
    builder = SparseAssemblyBuilder() if sparse_pattern is None else None
    flat_values = sparse_pattern.empty_flat_values() if sparse_pattern is not None else None
    block_index = 0

    def add_block(row_dofs: Any, col_dofs: Any, block: np.ndarray) -> None:
        nonlocal block_index
        if sparse_pattern is not None:
            assert flat_values is not None
            sparse_pattern.fill_block(flat_values, block_index, block)
            block_index += 1
        else:
            assert builder is not None
            builder.add_block(row_dofs, col_dofs, block)

    for element_index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        initial = _initial_stress_for_element(element_index, element.id, initial_stresses, initial_stress_cache)
        etype = element.type.upper()
        mode = normalize_integration(element.integration)
        if etype == "QUAD4" and _axisymmetric_quad4_j2dp_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _axisymmetric_quad4_state_arrays(element.id, plastic_state, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, _fe = _quad4_axisymmetric_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            continue
        if etype == "QUAD4" and _axisymmetric_quad4_elastic_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke = _quad4_axisymmetric_element_stiffness_fast(coords, material)
            add_block(dofs, dofs, ke)
            continue
        if etype == "QUAD8" and _axisymmetric_quad8_j2dp_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _axisymmetric_quad8_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, _fe = _quad8_axisymmetric_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            continue
        if etype == "QUAD8" and _axisymmetric_quad4_elastic_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke = _quad8_axisymmetric_element_stiffness_fast(coords, material, mode)
            add_block(dofs, dofs, ke)
            continue
        ke = _algorithmic_tangent_block(
            element,
            coords,
            ue,
            material,
            initial,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            tangent_method=tangent_method,
            plastic_state_cache=plastic_state_cache,
            axisymmetric=True,
        )
        add_block(dofs, dofs, ke)
    for interface in interfaces or []:
        if not interface.active:
            continue
        dofs = _element_dofs((*interface.minus_nodes, *interface.plus_nodes), node_index)
        idofs, _fe, ke = interface_force_tangent(interface, mesh, u[dofs], axisymmetric=True)
        add_block(idofs, idofs, ke)
    for structural in structural_elements or []:
        if not structural.active:
            continue
        dofs = _element_dofs(structural.nodes, node_index)
        idofs, _fe, ke = structural_element_force_tangent(structural, mesh, materials, u[dofs], axisymmetric=True)
        add_block(idofs, idofs, ke)
    if sparse_pattern is not None:
        assert flat_values is not None
        sparse_pattern.validate_filled_block_count(block_index)
        return sparse_pattern.assemble_flat_values(flat_values)
    assert builder is not None
    return builder.to_csr((ndof, ndof))


def assemble_axisymmetric_internal_force(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: InitialStressArrayCache | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
) -> np.ndarray:
    if plastic_state_cache is None and plastic_state:
        plastic_state_cache = build_plastic_state_array_cache(mesh, materials, plastic_state)
    node_index = mesh.node_index
    fint = np.zeros(structural_total_dofs(mesh, structural_elements, axisymmetric=True), dtype=float)
    for element_index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        initial = _initial_stress_for_element(element_index, element.id, initial_stresses, initial_stress_cache)
        etype = element.type.upper()
        mode = normalize_integration(element.integration)
        if etype == "QUAD4" and _axisymmetric_quad4_j2dp_fast_path(material, plastic_state, element_id=element.id, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _axisymmetric_quad4_state_arrays(element.id, plastic_state, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            _ke, fe = _quad4_axisymmetric_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            fint[dofs] += fe
            continue
        if etype == "QUAD4" and _axisymmetric_quad4_elastic_fast_path(material, plastic_state):
            fe = _quad4_axisymmetric_internal_force_elastic_fast(coords, ue, material, initial)
            fint[dofs] += fe
            continue
        if etype == "QUAD8" and _axisymmetric_quad8_j2dp_fast_path(material, plastic_state, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _axisymmetric_quad8_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            _ke, fe = _quad8_axisymmetric_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            fint[dofs] += fe
            continue
        if etype == "QUAD8" and _axisymmetric_quad4_elastic_fast_path(material, plastic_state):
            fe = _quad8_axisymmetric_internal_force_elastic_fast(coords, ue, material, mode, initial)
            fint[dofs] += fe
            continue
        fe = _internal_force_block(
            element,
            coords,
            ue,
            material,
            initial,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            axisymmetric=True,
        )
        fint[dofs] += fe
    for interface in interfaces or []:
        if not interface.active:
            continue
        dofs = _element_dofs((*interface.minus_nodes, *interface.plus_nodes), node_index)
        idofs, fe, _ke = interface_force_tangent(interface, mesh, u[dofs], axisymmetric=True)
        fint[idofs] += fe
    for structural in structural_elements or []:
        if not structural.active:
            continue
        dofs = _element_dofs(structural.nodes, node_index)
        idofs, fe, _ke = structural_element_force_tangent(structural, mesh, materials, u[dofs], axisymmetric=True)
        fint[idofs] += fe
    return fint


def assemble_axisymmetric_tangent_and_internal_force(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: InitialStressArrayCache | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    tangent_method: str = "analytic",
    sparse_pattern: SparseAssemblyPattern | None = None,
) -> tuple[csr_matrix, np.ndarray]:
    if plastic_state_cache is None and plastic_state:
        plastic_state_cache = build_plastic_state_array_cache(mesh, materials, plastic_state)
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements, axisymmetric=True)
    fint = np.zeros(ndof, dtype=float)
    builder = SparseAssemblyBuilder() if sparse_pattern is None else None
    flat_values = sparse_pattern.empty_flat_values() if sparse_pattern is not None else None
    block_index = 0

    def add_block(row_dofs: Any, col_dofs: Any, block: np.ndarray) -> None:
        nonlocal block_index
        if sparse_pattern is not None:
            assert flat_values is not None
            sparse_pattern.fill_block(flat_values, block_index, block)
            block_index += 1
        else:
            assert builder is not None
            builder.add_block(row_dofs, col_dofs, block)

    def add_force(row_dofs: Any, fe: np.ndarray) -> None:
        fint[row_dofs] += fe

    for element_index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        initial = _initial_stress_for_element(element_index, element.id, initial_stresses, initial_stress_cache)
        etype = element.type.upper()
        mode = normalize_integration(element.integration)
        if etype == "QUAD4" and _axisymmetric_quad4_j2dp_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _axisymmetric_quad4_state_arrays(element.id, plastic_state, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, fe = _quad4_axisymmetric_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if etype == "QUAD4" and _axisymmetric_quad4_elastic_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke = _quad4_axisymmetric_element_stiffness_fast(coords, material)
            fe = _quad4_axisymmetric_internal_force_elastic_fast(coords, ue, material, initial)
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if etype == "QUAD8" and _axisymmetric_quad8_j2dp_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _axisymmetric_quad8_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, fe = _quad8_axisymmetric_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if etype == "QUAD8" and _axisymmetric_quad4_elastic_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke = _quad8_axisymmetric_element_stiffness_fast(coords, material, mode)
            fe = _quad8_axisymmetric_internal_force_elastic_fast(coords, ue, material, mode, initial)
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        ke, fe = _tangent_internal_block(
            element,
            coords,
            ue,
            material,
            initial,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            tangent_method=tangent_method,
            plastic_state_cache=plastic_state_cache,
            axisymmetric=True,
        )
        add_block(dofs, dofs, ke)
        add_force(dofs, fe)
    for interface in interfaces or []:
        if not interface.active:
            continue
        dofs = _element_dofs((*interface.minus_nodes, *interface.plus_nodes), node_index)
        idofs, fe, ke = interface_force_tangent(interface, mesh, u[dofs], axisymmetric=True)
        add_block(idofs, idofs, ke)
        add_force(idofs, fe)
    for structural in structural_elements or []:
        if not structural.active:
            continue
        dofs = _element_dofs(structural.nodes, node_index)
        idofs, fe, ke = structural_element_force_tangent(structural, mesh, materials, u[dofs], axisymmetric=True)
        add_block(idofs, idofs, ke)
        add_force(idofs, fe)
    if sparse_pattern is not None:
        assert flat_values is not None
        sparse_pattern.validate_filled_block_count(block_index)
        return sparse_pattern.assemble_flat_values(flat_values), fint
    assert builder is not None
    return builder.to_csr((ndof, ndof)), fint


def assemble_algorithmic_tangent_stiffness(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: InitialStressArrayCache | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    mohr_coulomb_active_set_cache: MohrCoulombActiveSetCache | None = None,
    tangent_method: str = "analytic",
    sparse_pattern: SparseAssemblyPattern | None = None,
) -> csr_matrix:
    if plastic_state_cache is None and plastic_state:
        plastic_state_cache = build_plastic_state_array_cache(mesh, materials, plastic_state)
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements)
    rotation_dofs = structural_rotation_dof_map(mesh, structural_elements)
    builder = SparseAssemblyBuilder() if sparse_pattern is None else None
    flat_values = sparse_pattern.empty_flat_values() if sparse_pattern is not None else None
    block_index = 0
    plastic_batch_blocks = _quad4_sri_bbar_plastic_batch_blocks(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        initial_stress_cache=initial_stress_cache,
        plastic_state=plastic_state,
        plastic_state_cache=plastic_state_cache,
        strength_factor=strength_factor,
        mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
        tangent_method=tangent_method,
    )

    def add_block(row_dofs: Any, col_dofs: Any, block: np.ndarray) -> None:
        nonlocal block_index
        if sparse_pattern is not None:
            assert flat_values is not None
            sparse_pattern.fill_block(flat_values, block_index, block)
            block_index += 1
        else:
            assert builder is not None
            builder.add_block(row_dofs, col_dofs, block)

    for element_index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        batched = plastic_batch_blocks.get(element_index)
        if batched is not None:
            dofs, ke, _fe = batched
            add_block(dofs, dofs, ke)
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        initial = _initial_stress_for_element(element_index, element.id, initial_stresses, initial_stress_cache)
        etype = element.type.upper()
        mode = normalize_integration(element.integration)
        fast_full = etype == "QUAD4" and mode == "FULL"
        if fast_full and _quad4_advanced_strength_mc_fast_path(material, tangent_method=tangent_method):
            fast = _quad4_advanced_strength_mc_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            if fast is not None:
                ke, _fe = fast
                add_block(dofs, dofs, ke)
                continue
        if fast_full and _quad4_advanced_strength_j2dp_fast_path(material, tangent_method=tangent_method):
            ke, _fe = _quad4_advanced_strength_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            add_block(dofs, dofs, ke)
            continue
        if fast_full and _quad4_j2dp_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad4_plastic_state_arrays(element.id, plastic_state, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, _fe = _quad4_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            continue
        if fast_full and _quad4_mc_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad4_plastic_state_arrays(element.id, plastic_state, plastic_state_cache)
            fast = _quad4_mc_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if fast is not None:
                ke, _fe = fast
                add_block(dofs, dofs, ke)
                continue
        if etype == "QUAD8" and _quad4_advanced_strength_mc_fast_path(material, tangent_method=tangent_method):
            fast = _quad8_advanced_strength_mc_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            if fast is not None:
                ke, _fe = fast
                add_block(dofs, dofs, ke)
                continue
        if etype == "QUAD8" and _quad4_advanced_strength_j2dp_fast_path(material, tangent_method=tangent_method):
            ke, _fe = _quad8_advanced_strength_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            add_block(dofs, dofs, ke)
            continue
        if etype == "QUAD8" and _quad8_j2dp_tension_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad8_plastic_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, _fe = _quad8_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            continue
        if etype == "QUAD8" and _quad8_j2dp_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad8_plastic_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, _fe = _quad8_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            continue
        if etype == "QUAD8" and _quad8_mc_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad8_plastic_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            fast = _quad8_mc_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if fast is not None:
                ke, _fe = fast
                add_block(dofs, dofs, ke)
                continue
        if etype == "QUAD8" and _quad8_elastic_tension_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke, _fe = _quad8_elastic_tension_tangent_force_fast(coords, ue, material, mode, initial)
            add_block(dofs, dofs, ke)
            continue
        if fast_full and _quad4_elastic_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke = _quad4_element_stiffness_fast(coords, material, "FULL")
            add_block(dofs, dofs, ke)
            continue
        if etype == "QUAD8" and _quad4_elastic_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke = _quad8_element_stiffness_fast(coords, material, mode)
            add_block(dofs, dofs, ke)
            continue
        ke = _algorithmic_tangent_block(
            element,
            coords,
            ue,
            material,
            initial,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            tangent_method=tangent_method,
            plastic_state_cache=plastic_state_cache,
        )
        add_block(dofs, dofs, ke)
    for interface in interfaces or []:
        if not interface.active:
            continue
        dofs = _element_dofs((*interface.minus_nodes, *interface.plus_nodes), node_index)
        idofs, _fe, ke = interface_force_tangent(interface, mesh, u[dofs])
        add_block(idofs, idofs, ke)
    for structural in structural_elements or []:
        if not structural.active:
            continue
        dofs = structural_element_dofs(structural, mesh, rotation_dofs)
        idofs, _fe, ke = structural_element_force_tangent(structural, mesh, materials, u[dofs], rotation_dofs=rotation_dofs)
        add_block(idofs, idofs, ke)
    if sparse_pattern is not None:
        assert flat_values is not None
        sparse_pattern.validate_filled_block_count(block_index)
        return sparse_pattern.assemble_flat_values(flat_values)
    assert builder is not None
    return builder.to_csr((ndof, ndof))


def assemble_tangent_and_internal_force(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: InitialStressArrayCache | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    mohr_coulomb_active_set_cache: MohrCoulombActiveSetCache | None = None,
    quad4_mc_geometry_cache: Quad4MCGeometryCache | None = None,
    tangent_method: str = "analytic",
    sparse_pattern: SparseAssemblyPattern | None = None,
) -> tuple[csr_matrix, np.ndarray]:
    if plastic_state_cache is None and plastic_state:
        plastic_state_cache = build_plastic_state_array_cache(mesh, materials, plastic_state)
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements)
    fint = np.zeros(ndof, dtype=float)
    rotation_dofs = structural_rotation_dof_map(mesh, structural_elements)
    builder = SparseAssemblyBuilder() if sparse_pattern is None else None
    flat_values = sparse_pattern.empty_flat_values() if sparse_pattern is not None else None
    block_index = 0
    plastic_batch_blocks = _quad4_sri_bbar_plastic_batch_blocks(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        initial_stress_cache=initial_stress_cache,
        plastic_state=plastic_state,
        plastic_state_cache=plastic_state_cache,
        strength_factor=strength_factor,
        mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
        quad4_mc_geometry_cache=quad4_mc_geometry_cache,
        tangent_method=tangent_method,
    )

    def add_block(row_dofs: Any, col_dofs: Any, block: np.ndarray) -> None:
        nonlocal block_index
        if sparse_pattern is not None:
            assert flat_values is not None
            sparse_pattern.fill_block(flat_values, block_index, block)
            block_index += 1
        else:
            assert builder is not None
            builder.add_block(row_dofs, col_dofs, block)

    def add_force(row_dofs: Any, fe: np.ndarray) -> None:
        fint[row_dofs] += fe

    for element_index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        batched = plastic_batch_blocks.get(element_index)
        if batched is not None:
            dofs, ke, fe = batched
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        initial = _initial_stress_for_element(element_index, element.id, initial_stresses, initial_stress_cache)
        etype = element.type.upper()
        mode = normalize_integration(element.integration)
        fast_full = etype == "QUAD4" and mode == "FULL"
        if fast_full and _quad4_advanced_strength_mc_fast_path(material, tangent_method=tangent_method):
            fast = _quad4_advanced_strength_mc_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            if fast is not None:
                ke, fe = fast
                add_block(dofs, dofs, ke)
                add_force(dofs, fe)
                continue
        if fast_full and _quad4_advanced_strength_j2dp_fast_path(material, tangent_method=tangent_method):
            ke, fe = _quad4_advanced_strength_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if fast_full and _quad4_j2dp_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad4_plastic_state_arrays(element.id, plastic_state, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, fe = _quad4_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if fast_full and _quad4_mc_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad4_plastic_state_arrays(element.id, plastic_state, plastic_state_cache)
            fast = _quad4_mc_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if fast is not None:
                ke, fe = fast
                add_block(dofs, dofs, ke)
                add_force(dofs, fe)
                continue
        if etype == "QUAD8" and _quad4_advanced_strength_mc_fast_path(material, tangent_method=tangent_method):
            fast = _quad8_advanced_strength_mc_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            if fast is not None:
                ke, fe = fast
                add_block(dofs, dofs, ke)
                add_force(dofs, fe)
                continue
        if etype == "QUAD8" and _quad4_advanced_strength_j2dp_fast_path(material, tangent_method=tangent_method):
            ke, fe = _quad8_advanced_strength_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if etype == "QUAD8" and _quad8_j2dp_tension_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad8_plastic_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, fe = _quad8_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if etype == "QUAD8" and _quad8_j2dp_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad8_plastic_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            ke, fe = _quad8_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if etype == "QUAD8" and _quad8_mc_fast_path(material, plastic_state, tangent_method=tangent_method, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad8_plastic_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            fast = _quad8_mc_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if fast is not None:
                ke, fe = fast
                add_block(dofs, dofs, ke)
                add_force(dofs, fe)
                continue
        if etype == "QUAD8" and _quad8_elastic_tension_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke, fe = _quad8_elastic_tension_tangent_force_fast(coords, ue, material, mode, initial)
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if fast_full and _quad4_elastic_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke = _quad4_element_stiffness_fast(coords, material, "FULL")
            fe = _quad4_internal_force_elastic_fast(coords, ue, material, initial)
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        if etype == "QUAD8" and _quad4_elastic_fast_path(material, plastic_state, tangent_method=tangent_method):
            ke = _quad8_element_stiffness_fast(coords, material, mode)
            fe = _quad8_internal_force_elastic_fast(coords, ue, material, mode, initial)
            add_block(dofs, dofs, ke)
            add_force(dofs, fe)
            continue
        ke, fe = _tangent_internal_block(
            element,
            coords,
            ue,
            material,
            initial,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            tangent_method=tangent_method,
            plastic_state_cache=plastic_state_cache,
        )
        add_block(dofs, dofs, ke)
        add_force(dofs, fe)
    for interface in interfaces or []:
        if not interface.active:
            continue
        dofs = _element_dofs((*interface.minus_nodes, *interface.plus_nodes), node_index)
        idofs, fe, ke = interface_force_tangent(interface, mesh, u[dofs])
        add_block(idofs, idofs, ke)
        add_force(idofs, fe)
    for structural in structural_elements or []:
        if not structural.active:
            continue
        dofs = structural_element_dofs(structural, mesh, rotation_dofs)
        idofs, fe, ke = structural_element_force_tangent(structural, mesh, materials, u[dofs], rotation_dofs=rotation_dofs)
        add_block(idofs, idofs, ke)
        add_force(idofs, fe)
    if sparse_pattern is not None:
        assert flat_values is not None
        sparse_pattern.validate_filled_block_count(block_index)
        return sparse_pattern.assemble_flat_values(flat_values), fint
    assert builder is not None
    return builder.to_csr((ndof, ndof)), fint


def assemble_internal_force_candidates(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    displacements: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: InitialStressArrayCache | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    mohr_coulomb_active_set_cache: MohrCoulombActiveSetCache | None = None,
    quad4_mc_geometry_cache: Quad4MCGeometryCache | None = None,
) -> np.ndarray | None:
    """Fast force-only line-search path for homogeneous QUAD4 SRI/B-bar MC."""

    candidates = np.ascontiguousarray(displacements, dtype=np.float64)
    if (
        candidates.ndim != 2
        or candidates.shape[0] == 0
        or interfaces
        or structural_elements
    ):
        return None
    expected_dofs = 2 * len(mesh.node_ids)
    if candidates.shape[1] != expected_dofs:
        return None
    active_indices = {
        index for index, element in enumerate(mesh.elements) if element.active
    }
    blocks = [
        block
        for block in build_plastic_element_blocks(mesh, materials)
        if block.element_type == "QUAD4"
        and block.integration in {"SRI", "B-BAR"}
        and block.material_model == "mohr_coulomb"
        and not block.tension_cutoff
    ]
    covered_indices = {
        int(element_index)
        for block in blocks
        for element_index in block.element_indices
    }
    if not active_indices or covered_indices != active_indices:
        return None
    if plastic_state_cache is None and plastic_state:
        plastic_state_cache = build_plastic_state_array_cache(
            mesh, materials, plastic_state
        )

    assembled = np.zeros(
        (candidates.shape[0], expected_dofs), dtype=np.float64
    )
    for block in blocks:
        result = evaluate_mc_internal_force_candidates(
            block,
            mesh,
            materials,
            candidates,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            strength_factor=strength_factor,
            mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
            quad4_mc_geometry_cache=quad4_mc_geometry_cache,
        )
        if result is None or np.any(
            result.status_flags != PLASTIC_BATCH_STATUS_OK
        ):
            return None
        for row in range(result.dofs.shape[0]):
            for local_dof, global_dof in enumerate(result.dofs[row]):
                assembled[:, int(global_dof)] += result.internal_force_values[
                    :, row, local_dof
                ]
    return assembled


def assemble_internal_force(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: InitialStressArrayCache | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    mohr_coulomb_active_set_cache: MohrCoulombActiveSetCache | None = None,
    quad4_mc_geometry_cache: Quad4MCGeometryCache | None = None,
) -> np.ndarray:
    if plastic_state_cache is None and plastic_state:
        plastic_state_cache = build_plastic_state_array_cache(mesh, materials, plastic_state)
    node_index = mesh.node_index
    fint = np.zeros(structural_total_dofs(mesh, structural_elements), dtype=float)
    rotation_dofs = structural_rotation_dof_map(mesh, structural_elements)
    plastic_batch_blocks = _quad4_sri_bbar_plastic_batch_blocks(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        initial_stress_cache=initial_stress_cache,
        plastic_state=plastic_state,
        plastic_state_cache=plastic_state_cache,
        strength_factor=strength_factor,
        mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
        quad4_mc_geometry_cache=quad4_mc_geometry_cache,
    )
    for element_index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        batched = plastic_batch_blocks.get(element_index)
        if batched is not None:
            dofs, _ke, fe = batched
            fint[dofs] += fe
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        initial = _initial_stress_for_element(element_index, element.id, initial_stresses, initial_stress_cache)
        etype = element.type.upper()
        mode = normalize_integration(element.integration)
        fast_full = etype == "QUAD4" and mode == "FULL"
        if fast_full and _quad4_advanced_strength_mc_fast_path(material):
            fast = _quad4_advanced_strength_mc_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            if fast is not None:
                _ke, fe = fast
                fint[dofs] += fe
                continue
        if fast_full and _quad4_advanced_strength_j2dp_fast_path(material):
            _ke, fe = _quad4_advanced_strength_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            fint[dofs] += fe
            continue
        if fast_full and _quad4_j2dp_fast_path(material, plastic_state, element_id=element.id, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad4_plastic_state_arrays(element.id, plastic_state, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            _ke, fe = _quad4_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            fint[dofs] += fe
            continue
        if fast_full and _quad4_mc_fast_path(material, plastic_state, element_id=element.id, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad4_plastic_state_arrays(element.id, plastic_state, plastic_state_cache)
            fe = _quad4_mc_internal_force_fast(
                coords,
                ue,
                material,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if fe is not None:
                fint[dofs] += fe
                continue
        if etype == "QUAD8" and _quad4_advanced_strength_mc_fast_path(material):
            fast = _quad8_advanced_strength_mc_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            if fast is not None:
                _ke, fe = fast
                fint[dofs] += fe
                continue
        if etype == "QUAD8" and _quad4_advanced_strength_j2dp_fast_path(material):
            _ke, fe = _quad8_advanced_strength_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_state=plastic_state,
                element_id=element.id,
                strength_factor=strength_factor,
            )
            fint[dofs] += fe
            continue
        if etype == "QUAD8" and _quad8_j2dp_tension_fast_path(material, plastic_state, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad8_plastic_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            _ke, fe = _quad8_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            fint[dofs] += fe
            continue
        if etype == "QUAD8" and _quad8_j2dp_fast_path(material, plastic_state, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad8_plastic_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            _ke, fe = _quad8_j2dp_tangent_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                alpha=alpha,
                cohesion_term=cohesion_term,
            )
            fint[dofs] += fe
            continue
        if etype == "QUAD8" and _quad8_mc_fast_path(material, plastic_state, element_id=element.id, integration=mode, plastic_state_cache=plastic_state_cache):
            plastic_strains, kappas = _quad8_plastic_state_arrays(element.id, plastic_state, mode, plastic_state_cache)
            fe = _quad8_mc_internal_force_fast(
                coords,
                ue,
                material,
                mode,
                initial_stress=initial,
                plastic_strains=plastic_strains,
                kappas=kappas,
                strength_factor=strength_factor,
            )
            if fe is not None:
                fint[dofs] += fe
                continue
        if etype == "QUAD8" and _quad8_elastic_tension_fast_path(material, plastic_state):
            _ke, fe = _quad8_elastic_tension_tangent_force_fast(coords, ue, material, mode, initial)
            fint[dofs] += fe
            continue
        if fast_full and _quad4_elastic_fast_path(material, plastic_state):
            fe = _quad4_internal_force_elastic_fast(coords, ue, material, initial)
            fint[dofs] += fe
            continue
        if etype == "QUAD8" and _quad4_elastic_fast_path(material, plastic_state):
            fe = _quad8_internal_force_elastic_fast(coords, ue, material, mode, initial)
            fint[dofs] += fe
            continue
        fe = _internal_force_block(
            element,
            coords,
            ue,
            material,
            initial,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
        )
        fint[dofs] += fe
    for interface in interfaces or []:
        if not interface.active:
            continue
        dofs = _element_dofs((*interface.minus_nodes, *interface.plus_nodes), node_index)
        idofs, fe, _ke = interface_force_tangent(interface, mesh, u[dofs])
        fint[idofs] += fe
    for structural in structural_elements or []:
        if not structural.active:
            continue
        dofs = structural_element_dofs(structural, mesh, rotation_dofs)
        idofs, fe, _ke = structural_element_force_tangent(structural, mesh, materials, u[dofs], rotation_dofs=rotation_dofs)
        fint[idofs] += fe
    return fint
