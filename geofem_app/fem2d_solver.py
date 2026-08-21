"""2D FEM solver orchestration and matrix assembly."""

from __future__ import annotations

import csv
import copy
import hashlib
import json
import multiprocessing as mp
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping
import math
import time
from time import perf_counter as _perf_counter

import numpy as np
from scipy.sparse import bmat, csr_matrix

from .fem2d_config import plane_strain_materials, validate_2d_core_scope
from .fem2d_constraints import (
    _add_inactive_node_constraints,
    assemble_mpc_penalty,
    collect_constraints,
    mpc_violation,
)
from .fem2d_elements import (
    _inactive_element_result,
    _average_material_state_outputs,
    _default_material_state_output,
    _inactive_material_state_output,
    _material_state_output,
    _quad4_axisymmetric_biot_matrix_fast,
    _quad4_axisymmetric_element_stiffness_fast,
    _quad4_axisymmetric_internal_force_elastic_fast,
    _quad4_axisymmetric_j2dp_tangent_force_fast,
    _quad4_axisymmetric_pressure_matrices_fast,
    _quad4_advanced_strength_mc_tangent_force_fast,
    _quad4_advanced_strength_j2dp_tangent_force_fast,
    _quad8_advanced_strength_mc_tangent_force_fast,
    _quad8_advanced_strength_j2dp_tangent_force_fast,
    _quad4_biot_matrix_fast,
    _quad4_consistent_mass_matrix_fast,
    _quad4_element_stiffness_fast,
    _quad4_internal_force_elastic_fast,
    _quad4_j2dp_tangent_force_fast,
    _quad4_mc_internal_force_fast,
    _quad4_mc_tangent_force_fast,
    _quad4_pressure_matrices_fast,
    _quad8_biot_matrix_fast,
    _quad8_axisymmetric_biot_matrix_fast,
    _quad8_axisymmetric_edge_traction_fast,
    _quad8_axisymmetric_element_stiffness_fast,
    _quad8_axisymmetric_internal_force_elastic_fast,
    _quad8_axisymmetric_j2dp_bbar_post_fast,
    _quad8_axisymmetric_j2dp_post_fast,
    _quad8_axisymmetric_j2dp_tangent_force_fast,
    _quad8_axisymmetric_pressure_matrices_fast,
    _quad8_consistent_mass_matrix_fast,
    _quad8_element_stiffness_fast,
    _quad8_elastic_tension_tangent_force_fast,
    _quad8_internal_force_elastic_fast,
    _quad8_j2dp_tangent_force_fast,
    _quad8_mc_internal_force_fast,
    _quad8_mc_tangent_force_fast,
    _quad8_pressure_matrices_fast,
    axisymmetric_element_stiffness,
    axisymmetric_strain_displacement_matrix,
    compute_element_results_and_state,
    compute_integration_point_results,
    compute_plastic_state_array_cache,
    element_stiffness,
    integration_points,
    shape_functions,
    strain_displacement_matrix,
)
from .fem2d_interfaces import assemble_interface_hydraulic_transfer, compute_interface_results, interface_force_tangent, update_interface_histories
from .fem2d_dynamic import (
    _acceleration_to_g,
    _apply_dynamic_profile,
    _dynamic_boundary_unit_scale,
    _dynamic_boundary_value_at,
    _dynamic_history_row,
    _dynamic_initial_vector,
    _dynamic_load_scale,
    _dynamic_solver_config,
    _dynamic_stage_settings,
    _dynamic_time_vector,
    _dynamic_up_enabled,
    _float_sequence,
    _history_axis_k_at,
    _history_row_k,
    _history_value_at,
    _integrated_history_rows,
    _integrated_history_value,
    _rayleigh_coefficients_from_damping_spec,
    _rayleigh_damping_matrix,
    _regularized_dynamic_mass,
    _row_first_float,
    _time_history_rows,
)
from .fem2d_hydro import (
    _attach_stage_load_hydro_info,
    _collect_pressure_constraints,
    _convert_pressure_unit,
    _hydro_mapping,
    _initial_pore_pressure,
    _merge_hydro_maps,
    _normalize_pressure_spec,
    _pore_pressure_from_hydro,
    _prepare_stage_hydro,
    _pressure_source_name,
    _pressure_value_for_node,
    _row_time_value,
    _seepage_pressure_specs,
    _seepage_rows,
    _select_time_rows,
    _stage_has_hydro_coupling,
    _stage_with_hydro,
    _water_level_pressure_specs,
)
from .fem2d_hydro_iteration import (
    SeepageActiveSetState,
    advance_seepage_active_set,
    observe_seepage_active_set,
    seepage_outer_iteration_limit,
)
from .fem2d_io import write_run_summary, write_stage_outputs
from .fem2d_linear_solver import (
    check_linear_solution as _linear_check_linear_solution,
    clear_linear_factor_cache,
    linear_factor_cache_info,
    linear_solver_settings as _linear_solver_settings_core,
    solve_linear_system as _solve_linear_system_core,
    solve_reduced_linear_system as _solve_reduced_linear_system_core,
    solve_sparse_with_constraints as _solve_sparse_with_constraints_core,
)
from .fem2d_result_annotations import (
    _attach_integration_point_results,
    _attach_matrix_profile,
    _attach_stage_runtime,
    _liquefaction_effective_stress_reference,
    _liquefaction_state_summary,
    _material_has_liquefaction,
    _update_liquefaction_state_from_pore_pressure,
)
from .fem2d_mpc import (
    MPCStagePlan,
    lagrange_mpc_projected_residual as _lagrange_mpc_projected_residual,
    mpc_arc_length_stage_plan,
    mpc_constraint_matrix as _mpc_constraint_matrix,
    mpc_stage_plan,
    solve_arc_length_lagrange_correction as _solve_arc_length_lagrange_correction,
    solve_lagrange_augmented_system as _solve_lagrange_augmented_system,
    solve_lagrange_mpc_correction as _solve_lagrange_mpc_correction,
    solve_linear_system_with_mpc_elimination,
    solve_linear_system_with_mpc_lagrange,
)
from .fem2d_pressure import (
    _axisymmetric_edge_measure,
    _solve_scalar_constraints,
    assemble_axisymmetric_biot_coupling_matrix,
    assemble_axisymmetric_pressure_boundary_terms,
    assemble_axisymmetric_pressure_matrices,
    assemble_biot_coupling_matrix,
    assemble_biot_coupling_matrix_cached,
    assemble_liquefaction_pressure_terms,
    assemble_pore_pressure_load,
    assemble_pressure_boundary_terms_cached,
    assemble_pressure_boundary_terms,
    assemble_pressure_matrices_cached,
    assemble_pressure_matrices,
    build_biot_coupling_assembly_cache,
    assemble_pore_pressure_load_cached,
    build_pore_pressure_load_cache,
    build_pressure_boundary_term_cache,
    build_pressure_matrix_assembly_cache,
    solve_consolidation_pressure,
)
from .fem2d_nonlinear_assembly import (
    _axisymmetric_quad8_j2dp_fast_path,
    _axisymmetric_quad8_post_state_arrays,
    InitialStressArrayCache,
    assemble_algorithmic_tangent_stiffness,
    assemble_axisymmetric_algorithmic_tangent_stiffness,
    assemble_axisymmetric_internal_force,
    assemble_axisymmetric_tangent_and_internal_force,
    assemble_internal_force,
    assemble_internal_force_candidates,
    assemble_tangent_and_internal_force,
    build_initial_stress_array_cache,
    initial_stress_array_cache_info,
)
from .fem2d_solver_controls import (
    has_nonlinear_interfaces as _has_nonlinear_interfaces_core,
    increment_settings as _increment_settings_core,
    newton_settings as _newton_settings_core,
    plastic_ratio as _plastic_ratio_core,
    scale_boundary_conditions as _scale_boundary_conditions_core,
    scale_load_item as _scale_load_item_core,
    scale_loads as _scale_loads_core,
    solver_without_increments as _solver_without_increments_core,
    srm_factors as _srm_factors_core,
    tangent_method as _tangent_method_core,
)
from .fem2d_convergence import (
    dynamic_residual_metrics as _dynamic_residual_metrics,
    newton_convergence_metrics as _newton_convergence_metrics,
    newton_convergence_with_force_norm as _newton_convergence_with_force_norm,
    riks_convergence_metrics as _riks_convergence_metrics,
)
from .fem2d_solver_progress import (
    stage_boundary_conditions,
    stage_display_name,
    stage_loads,
    stage_mpc_constraints,
    stage_riks_solver_config,
    stage_sequence_from_config,
    stage_solver_config,
    stage_srm_solver_config,
    stage_state_after_result,
    stage_time as _stage_time,
    stage_type as _stage_type,
)
from .fem2d_structural_assembly import (
    _add_body_weight,
    _add_edge_traction,
    assemble_axisymmetric_load_vector,
    assemble_axisymmetric_stiffness,
    assemble_axisymmetric_stiffness_cached,
    assemble_global_stiffness,
    assemble_global_stiffness_cached,
    assemble_load_vector,
    assemble_load_vector_cached,
    assemble_mass_matrix,
    assemble_mass_matrix_cached,
    build_axisymmetric_stiffness_assembly_cache,
    build_load_vector_assembly_cache,
    build_global_stiffness_assembly_cache,
    build_mass_matrix_assembly_cache,
)
from .load_combinations import configured_load_combinations
from .fem2d_materials import (
    _advanced_strength_model_name,
    _equivalent_shear_strain,
    _is_advanced_material,
    _material_k0,
    _param_float,
    _plastic_state_for_gp,
    _plastic_state_key,
    _uses_plastic_strength_model,
    _yield_surface_parameters,
    algorithmic_material_tangent,
    mohr_coulomb_adaptive_tangent_counters,
    mohr_coulomb_fallback_telemetry,
    reset_mohr_coulomb_fallback_telemetry,
    update_plane_strain_stress,
)
from .fem2d_mesh import (
    _collect_warnings,
    _edge_consistent_robin_matrix,
    _edge_length,
    _edge_lumped_weights,
    _edge_set,
    _generate_rectangle_mesh,
    _mesh_with_active_elements,
    _pressure_edges,
    _target_elements,
    _target_nodes,
    _validate_material_references,
    _validate_mesh,
    interfaces_from_config,
    mesh_from_config,
    validate_mesh_quality_for_solve,
)
from .fem2d_structural import (
    compute_structural_results,
    structural_element_dofs,
    structural_element_equivalent_load,
    structural_element_force_tangent,
    structural_elements_from_config,
    structural_elements_with_active,
    structural_extra_dof_labels,
    structural_has_nonlinear,
    structural_rotation_dof_map,
    structural_total_dofs,
    update_structural_element_histories,
)
from .fem2d_types import (
    CONSOLIDATION_2D_STAGE_TYPES,
    AXISYMMETRIC_2D_STAGE_TYPES,
    DEACTIVATION_2D_STAGE_TYPES,
    DOF_NAMES,
    DYNAMIC_2D_STAGE_TYPES,
    Element2D,
    FEM2DError,
    GEOSTATIC_2D_STAGE_TYPES,
    LARGE_DEFORMATION_2D_STAGE_TYPES,
    Interface2D,
    Mesh2D,
    PlasticState2D,
    RIKS_2D_STAGE_TYPES,
    SRM_2D_STAGE_TYPES,
    ElasticPlaneStrainMaterial,
    SolveResult2D,
    StageResult2D,
    StructuralElement2D,
    normalize_integration,
)
from .fem2d_large_deformation import (
    LargeDeformationStepCache,
    build_large_deformation_step_cache,
    fill_updated_coords,
    large_deformation_common_solver_info,
    large_deformation_settings as _large_deformation_settings_core,
    max_displacement_norm,
    mesh_diagonal_length,
    mesh_with_updated_coords,
    solver_without_large_deformation as _solver_without_large_deformation_core,
    temporary_mesh_coords,
)
from .fem2d_performance_contract import deformation_mode_from_config
from .fem2d_plastic_state_arrays import ArrayBackedPlasticStateMapping, PlasticStateArrayCache, build_plastic_state_array_cache, plastic_state_array_cache_info
from .fem2d_plastic_batch import (
    MohrCoulombActiveSetCache,
    Quad4MCGeometryCache,
    build_quad4_mc_geometry_cache,
)
from .fem2d_utils import (
    _dofs_from_node_indices,
    _element_dofs,
    _element_node_indices,
    _ensure_list,
    _merge_solver_config,
    _require_sequence,
    _safe_name,
)
from .geofeas_public import annotate_public_stage_result, public_profile_run_warnings, write_stage_public_profile
from .reduced_matrix_cache import ReducedMatrixCache, build_reduced_matrix_cache_from_csr, build_reduced_matrix_cache_from_pattern
from .sparse_assembly import SparseAssemblyBuilder, SparseAssemblyPattern


def _with_run_output_config(solver: Mapping[str, Any] | None, output_config: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(output_config, Mapping) or not output_config:
        return solver
    updated = dict(solver or {})
    updated["_run_output_config"] = dict(output_config)
    return updated


def _stage_output_config_from_solver(solver: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(solver, Mapping):
        return None
    output = solver.get("_run_output_config")
    return output if isinstance(output, Mapping) else None


@dataclass(frozen=True)
class SmallDeformationStepCache:
    ndof: int
    constrained: dict[int, float]
    free_dofs: np.ndarray
    fixed_dofs: np.ndarray
    active_elements: list[str]
    stiffness_cache: Any | None = None
    load_scale: float = 1.0
    base_load_vector: np.ndarray | None = None
    load_vector_cache_reason: str = ""
    mpc_penalty_matrix: csr_matrix | None = None
    mpc_penalty_vector: np.ndarray | None = None
    mpc_info: dict[str, Any] | None = None
    mpc_penalty_cache_reason: str = ""
    reduced_matrix_cache: ReducedMatrixCache | None = None
    quad4_mc_geometry_cache: Quad4MCGeometryCache | None = None

    def with_constraint_scale(self, scale: float) -> "SmallDeformationStepCache":
        factor = float(scale)
        return replace(self, constrained={int(dof): float(value) * factor for dof, value in self.constrained.items()}, load_scale=factor)

    def solver_info(self) -> dict[str, Any]:
        stiffness_info = {} if self.stiffness_cache is None else self.stiffness_cache.info()
        linear_batches = stiffness_info.get("precomputed_linear_batches", {})
        if not isinstance(linear_batches, Mapping):
            linear_batches = {}
        return {
            "enabled": True,
            "cache_kind": "small_deformation_step_cache",
            "geometry_mode": "small_deformation",
            "ndof": int(self.ndof),
            "constrained_dofs": len(self.constrained),
            "free_dofs": int(self.free_dofs.size),
            "active_elements": len(self.active_elements),
            "stiffness_pattern_cached": self.stiffness_cache is not None,
            "stiffness_blocks": 0 if self.stiffness_cache is None else self.stiffness_cache.block_count,
            "batched_elastic_elements": int(stiffness_info.get("batched_elastic_elements", 0) or 0),
            "batched_quad4_elastic_elements": int(stiffness_info.get("batched_quad4_elastic_elements", 0) or 0),
            "batched_quad8_elastic_elements": int(stiffness_info.get("batched_quad8_elastic_elements", 0) or 0),
            "batched_tri3_elastic_elements": int(stiffness_info.get("batched_tri3_elastic_elements", 0) or 0),
            "batched_tri6_elastic_elements": int(stiffness_info.get("batched_tri6_elastic_elements", 0) or 0),
            "precomputed_interface_blocks": int(stiffness_info.get("precomputed_interface_blocks", 0) or 0),
            "precomputed_structural_blocks": int(stiffness_info.get("precomputed_structural_blocks", 0) or 0),
            "precomputed_linear_stiffness_blocks": int(stiffness_info.get("precomputed_linear_blocks", 0) or 0),
            "precomputed_linear_stiffness_batch_count": int(linear_batches.get("batch_count", 0) or 0),
            "precomputed_linear_stiffness_batched_blocks": int(linear_batches.get("batched_blocks", 0) or 0),
            "stiffness_assembly_cache": stiffness_info,
            "reduced_matrix_cached": self.reduced_matrix_cache is not None,
            "reduced_matrix_cache": {"enabled": False} if self.reduced_matrix_cache is None else self.reduced_matrix_cache.info(),
            "quad4_mc_geometry_cache": (
                {"enabled": False, "reason": "no_supported_quad4_mc_block"}
                if self.quad4_mc_geometry_cache is None
                else self.quad4_mc_geometry_cache.solver_info()
            ),
            "cache_inputs": ["load_factor", "strength_factor", "initial_stress", "pore_pressure"],
            "factor_invariant_load_mpc_cache": {
                "enabled": self.base_load_vector is not None or (self.mpc_penalty_matrix is not None and self.mpc_penalty_vector is not None and self.mpc_info is not None),
                "load_vector_cached": self.base_load_vector is not None,
                "load_vector_size": 0 if self.base_load_vector is None else int(self.base_load_vector.size),
                "load_scale": float(self.load_scale),
                "load_vector_reason": self.load_vector_cache_reason,
                "mpc_penalty_cached": self.mpc_penalty_matrix is not None and self.mpc_penalty_vector is not None and self.mpc_info is not None,
                "mpc_equations": 0 if self.mpc_info is None else int(self.mpc_info.get("count", 0) or 0),
                "mpc_penalty_reason": self.mpc_penalty_cache_reason,
                "constraint_scale_template_cached": True,
            },
        }


StepCache2D = LargeDeformationStepCache | SmallDeformationStepCache


@dataclass(frozen=True)
class DynamicMassStepCache:
    stiffness_cache: Any | None
    mass_cache: Any | None
    load_vector_cache: Any | None
    lumped_mass: bool = False

    def solver_info(self) -> dict[str, Any]:
        stiffness_info = {} if self.stiffness_cache is None else self.stiffness_cache.info()
        mass_info = {} if self.mass_cache is None else self.mass_cache.info()
        load_info = {} if self.load_vector_cache is None else self.load_vector_cache.info()
        return {
            "enabled": self.stiffness_cache is not None or self.mass_cache is not None or self.load_vector_cache is not None,
            "cache_kind": "dynamic_mass_step_cache",
            "stiffness_cache": {"enabled": False} if self.stiffness_cache is None else stiffness_info,
            "mass_cache": {"enabled": False} if self.mass_cache is None else mass_info,
            "load_vector_cache": {"enabled": False} if self.load_vector_cache is None else load_info,
            "lumped_mass": bool(self.lumped_mass),
            "batched_mass_elements": int(mass_info.get("batched_elements", 0) or 0),
            "direct_mass_fill": bool((mass_info.get("direct_fill", {}) if isinstance(mass_info.get("direct_fill", {}), Mapping) else {}).get("enabled", False)),
            "batched_stiffness_elements": int(stiffness_info.get("batched_elastic_elements", 0) or 0),
        }


@dataclass(frozen=True)
class AxisymmetricStepCache:
    ndof: int
    constrained: dict[int, float]
    free_dofs: np.ndarray
    fixed_dofs: np.ndarray
    active_elements: list[str]
    stiffness_pattern: SparseAssemblyPattern | None
    stiffness_cache: Any | None
    reduced_matrix_cache: ReducedMatrixCache | None

    def solver_info(self) -> dict[str, Any]:
        stiffness_info = {} if self.stiffness_cache is None else self.stiffness_cache.info()
        linear_batches = stiffness_info.get("precomputed_linear_batches", {})
        if not isinstance(linear_batches, Mapping):
            linear_batches = {}
        reduced = {"enabled": False} if self.reduced_matrix_cache is None else self.reduced_matrix_cache.info()
        return {
            "enabled": True,
            "cache_kind": "axisymmetric_step_cache",
            "geometry_mode": "axisymmetric",
            "ndof": int(self.ndof),
            "constrained_dofs": len(self.constrained),
            "free_dofs": int(self.free_dofs.size),
            "active_elements": len(self.active_elements),
            "stiffness_pattern_cached": self.stiffness_pattern is not None,
            "stiffness_blocks": 0 if self.stiffness_pattern is None else self.stiffness_pattern.block_count,
            "batched_axisymmetric_elastic_elements": int(stiffness_info.get("batched_axisymmetric_elastic_elements", 0) or 0),
            "batched_quad4_axisymmetric_elastic_elements": int(stiffness_info.get("batched_quad4_axisymmetric_elastic_elements", 0) or 0),
            "batched_quad8_axisymmetric_elastic_elements": int(stiffness_info.get("batched_quad8_axisymmetric_elastic_elements", 0) or 0),
            "batched_tri3_axisymmetric_elastic_elements": int(stiffness_info.get("batched_tri3_axisymmetric_elastic_elements", 0) or 0),
            "batched_tri6_axisymmetric_elastic_elements": int(stiffness_info.get("batched_tri6_axisymmetric_elastic_elements", 0) or 0),
            "precomputed_interface_blocks": int(stiffness_info.get("precomputed_interface_blocks", 0) or 0),
            "precomputed_structural_blocks": int(stiffness_info.get("precomputed_structural_blocks", 0) or 0),
            "precomputed_linear_stiffness_blocks": int(stiffness_info.get("precomputed_linear_blocks", 0) or 0),
            "precomputed_linear_stiffness_batch_count": int(linear_batches.get("batch_count", 0) or 0),
            "precomputed_linear_stiffness_batched_blocks": int(linear_batches.get("batched_blocks", 0) or 0),
            "stiffness_assembly_cache": {"enabled": False} if self.stiffness_cache is None else stiffness_info,
            "reduced_matrix_cached": self.reduced_matrix_cache is not None,
            "reduced_matrix_cache": reduced,
        }


def build_axisymmetric_step_cache(
    mesh: Mesh2D,
    boundary_conditions: Any,
    *,
    materials: Mapping[str, ElasticPlaneStrainMaterial] | None = None,
    interfaces: list[Any] | None = None,
    structural_elements: list[Any] | None = None,
    precompute_stiffness_pattern: bool = True,
) -> AxisymmetricStepCache:
    ndof = structural_total_dofs(mesh, structural_elements, axisymmetric=True)
    constrained = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
    _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)
    fixed = np.asarray(sorted(constrained), dtype=np.int64)
    if fixed.size and (int(fixed[0]) < 0 or int(fixed[-1]) >= ndof):
        raise FEM2DError("axisymmetric arc-length cached constraint dof is outside the model dof range")
    mask = np.zeros(ndof, dtype=bool)
    if fixed.size:
        mask[fixed] = True
    free = np.nonzero(~mask)[0].astype(np.int64, copy=False)
    pattern: SparseAssemblyPattern | None = None
    stiffness_cache: Any | None = None
    if precompute_stiffness_pattern:
        if materials is not None:
            stiffness_cache = build_axisymmetric_stiffness_assembly_cache(
                mesh,
                materials,
                interfaces=interfaces,
                structural_elements=structural_elements,
                precompute_linear_element_stiffness=True,
            )
            pattern = stiffness_cache.pattern
        else:
            node_index = mesh.node_index
            dof_blocks: list[np.ndarray] = []
            for element in mesh.elements:
                if not element.active:
                    continue
                dof_blocks.append(_dofs_from_node_indices(_element_node_indices(element.nodes, node_index)))
            for interface in interfaces or []:
                if not interface.active:
                    continue
                dof_blocks.append(_element_dofs((*interface.minus_nodes, *interface.plus_nodes), node_index))
            for structural in structural_elements or []:
                if not structural.active:
                    continue
                dof_blocks.append(structural_element_dofs(structural, mesh, axisymmetric=True))
            pattern = SparseAssemblyPattern.from_square_blocks(dof_blocks, (ndof, ndof))
    reduced_matrix_cache = (
        build_reduced_matrix_cache_from_pattern(
            pattern.shape,
            pattern.indptr,
            pattern.indices,
            free,
            fixed,
            source="axisymmetric_stiffness_pattern",
        )
        if pattern is not None
        else None
    )
    return AxisymmetricStepCache(
        ndof=ndof,
        constrained=dict(constrained),
        free_dofs=free,
        fixed_dofs=fixed,
        active_elements=[element.id for element in mesh.elements if element.active],
        stiffness_pattern=pattern,
        stiffness_cache=stiffness_cache,
        reduced_matrix_cache=reduced_matrix_cache,
    )


build_axisymmetric_arc_length_step_cache = build_axisymmetric_step_cache


def _static_step_cache_settings(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    raw_solver = solver if isinstance(solver, Mapping) else {}
    raw = raw_solver.get("static_step_cache", raw_solver.get("nonlinear_step_cache", raw_solver.get("step_cache", {})))
    if isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", raw.get("cache", True)))
        precompute = bool(raw.get("precompute_stiffness_pattern", raw.get("precompute_sparse_pattern", True)))
    elif raw is None:
        enabled = True
        precompute = True
    else:
        enabled = bool(raw)
        precompute = True
    return {
        "enabled": enabled,
        "precompute_stiffness_pattern": precompute,
        "source": "static_step_cache",
    }


def _disabled_static_step_cache_info(reason: str, *, geometry_mode: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "cache_kind": "static_step_cache",
        "geometry_mode": geometry_mode,
        "reason": reason,
        "stiffness_pattern_cached": False,
        "reduced_matrix_cached": False,
    }


@dataclass(frozen=True)
class ArcLengthAugmentedMatrixCache:
    free_size: int
    kff_indptr: np.ndarray
    kff_indices: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    kff_source_positions: np.ndarray
    kff_value_positions: np.ndarray
    top_right_positions: np.ndarray
    bottom_left_positions: np.ndarray
    bottom_right_position: int

    @classmethod
    def from_reduced_matrix(cls, matrix: csr_matrix) -> "ArcLengthAugmentedMatrixCache":
        kff = matrix.tocsr()
        if kff.shape[0] != kff.shape[1]:
            raise FEM2DError("arc-length augmented cache requires a square reduced tangent")
        n = int(kff.shape[0])
        indptr = np.empty(n + 2, dtype=np.int64)
        indptr[0] = 0
        indices_parts: list[np.ndarray] = []
        kff_source_parts: list[np.ndarray] = []
        kff_value_parts: list[np.ndarray] = []
        top_right = np.empty(n, dtype=np.int64)
        cursor = 0
        for row in range(n):
            start = int(kff.indptr[row])
            end = int(kff.indptr[row + 1])
            row_indices = np.asarray(kff.indices[start:end], dtype=np.int32)
            if row_indices.size:
                indices_parts.append(row_indices)
                kff_source_parts.append(np.arange(start, end, dtype=np.int64))
                kff_value_parts.append(np.arange(cursor, cursor + row_indices.size, dtype=np.int64))
                cursor += int(row_indices.size)
            indices_parts.append(np.asarray([n], dtype=np.int32))
            top_right[row] = cursor
            cursor += 1
            indptr[row + 1] = cursor
        bottom_left = np.arange(cursor, cursor + n, dtype=np.int64)
        if n:
            indices_parts.append(np.arange(n, dtype=np.int32))
            cursor += n
        indices_parts.append(np.asarray([n], dtype=np.int32))
        bottom_right_position = int(cursor)
        cursor += 1
        indptr[n + 1] = cursor
        indices = np.concatenate(indices_parts).astype(np.int32, copy=False) if indices_parts else np.zeros(0, dtype=np.int32)
        kff_source = np.concatenate(kff_source_parts).astype(np.int64, copy=False) if kff_source_parts else np.zeros(0, dtype=np.int64)
        kff_value = np.concatenate(kff_value_parts).astype(np.int64, copy=False) if kff_value_parts else np.zeros(0, dtype=np.int64)
        return cls(
            free_size=n,
            kff_indptr=np.asarray(kff.indptr, dtype=np.int64).copy(),
            kff_indices=np.asarray(kff.indices, dtype=np.int64).copy(),
            indptr=indptr,
            indices=indices,
            kff_source_positions=kff_source,
            kff_value_positions=kff_value,
            top_right_positions=top_right,
            bottom_left_positions=bottom_left,
            bottom_right_position=bottom_right_position,
        )

    def matches(self, matrix: csr_matrix) -> bool:
        kff = matrix.tocsr()
        return bool(
            kff.shape == (self.free_size, self.free_size)
            and kff.indptr.size == self.kff_indptr.size
            and kff.indices.size == self.kff_indices.size
            and np.array_equal(kff.indptr, self.kff_indptr)
            and np.array_equal(kff.indices, self.kff_indices)
        )

    def assemble(self, matrix: csr_matrix, top_right: np.ndarray, bottom_left: np.ndarray, bottom_right: float) -> csr_matrix:
        kff = matrix.tocsr()
        if not self.matches(kff):
            raise FEM2DError("arc-length augmented cache pattern does not match the reduced tangent")
        rhs_col = np.asarray(top_right, dtype=float).ravel()
        constraint_row = np.asarray(bottom_left, dtype=float).ravel()
        if rhs_col.size != self.free_size or constraint_row.size != self.free_size:
            raise FEM2DError("arc-length augmented cache vector size mismatch")
        data = np.zeros(int(self.indptr[-1]), dtype=float)
        if self.kff_value_positions.size:
            data[self.kff_value_positions] = np.asarray(kff.data, dtype=float)[self.kff_source_positions]
        if self.top_right_positions.size:
            data[self.top_right_positions] = rhs_col
        if self.bottom_left_positions.size:
            data[self.bottom_left_positions] = constraint_row
        data[self.bottom_right_position] = float(bottom_right)
        return csr_matrix((data, self.indices, self.indptr), shape=(self.free_size + 1, self.free_size + 1))

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "free_dofs": int(self.free_size),
            "shape": [int(self.free_size + 1), int(self.free_size + 1)],
            "nnz": int(self.indptr[-1]),
            "reduced_tangent_nnz": int(self.kff_indices.size),
        }


def _arc_length_augmented_cache_summary(events: list[Mapping[str, Any]], cache: ArcLengthAugmentedMatrixCache | None) -> dict[str, Any]:
    if not events and cache is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "solves": len(events),
        "hits": sum(1 for event in events if bool(event.get("reused", False))),
        "builds": sum(1 for event in events if bool(event.get("built", False))),
        "current": {} if cache is None else cache.info(),
    }


def _csr_block_entry_positions(template: csr_matrix, block: csr_matrix, *, row_offset: int = 0, col_offset: int = 0) -> np.ndarray:
    block_csr = block.tocsr()
    positions: list[np.ndarray] = []
    for local_row in range(block_csr.shape[0]):
        start = int(block_csr.indptr[local_row])
        end = int(block_csr.indptr[local_row + 1])
        if start == end:
            continue
        row = int(row_offset + local_row)
        template_start = int(template.indptr[row])
        template_end = int(template.indptr[row + 1])
        template_cols = template.indices[template_start:template_end]
        cols = block_csr.indices[start:end] + int(col_offset)
        local = np.searchsorted(template_cols, cols)
        if np.any(local >= template_cols.size) or np.any(template_cols[local] != cols):
            raise FEM2DError("axisymmetric u-p arc-length cache pattern is inconsistent")
        positions.append((template_start + local).astype(np.int64, copy=False))
    if not positions:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(positions).astype(np.int64, copy=False)


def _csr_dense_entry_positions(template: csr_matrix, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    row_arr = np.asarray(rows, dtype=np.int64).ravel()
    col_arr = np.asarray(cols, dtype=np.int64).ravel()
    if row_arr.size != col_arr.size:
        raise FEM2DError("axisymmetric u-p arc-length cache dense entry shape mismatch")
    positions = np.empty(row_arr.size, dtype=np.int64)
    for index, (row_raw, col_raw) in enumerate(zip(row_arr, col_arr, strict=True)):
        row = int(row_raw)
        col = int(col_raw)
        start = int(template.indptr[row])
        end = int(template.indptr[row + 1])
        local = int(np.searchsorted(template.indices[start:end], col))
        if local >= end - start or int(template.indices[start + local]) != col:
            raise FEM2DError("axisymmetric u-p arc-length cache dense entry is missing from pattern")
        positions[index] = start + local
    return positions


def _csr_pattern_rows_cols(block: csr_matrix, *, row_offset: int = 0, col_offset: int = 0) -> tuple[np.ndarray, np.ndarray]:
    block_csr = block.tocsr()
    if block_csr.nnz == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    rows = np.repeat(np.arange(block_csr.shape[0], dtype=np.int64), np.diff(block_csr.indptr)) + int(row_offset)
    cols = block_csr.indices.astype(np.int64, copy=False) + int(col_offset)
    return rows, cols


@dataclass(frozen=True)
class CoupledUPMonolithicMatrixCache:
    ndof: int
    npress: int
    dt: float
    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray
    stiffness_indptr: np.ndarray
    stiffness_indices: np.ndarray
    biot_indptr: np.ndarray
    biot_indices: np.ndarray
    pressure_indptr: np.ndarray
    pressure_indices: np.ndarray
    stiffness_positions: np.ndarray
    biot_upper_positions: np.ndarray
    biot_lower_positions: np.ndarray
    pressure_positions: np.ndarray

    @classmethod
    def build(cls, stiffness: csr_matrix, biot: csr_matrix, pressure_lhs: csr_matrix, dt: float) -> "CoupledUPMonolithicMatrixCache":
        k = stiffness.tocsr()
        b = biot.tocsr()
        p = pressure_lhs.tocsr()
        ndof = int(k.shape[0])
        npress = int(p.shape[0])
        if k.shape != (ndof, ndof) or p.shape != (npress, npress) or b.shape != (ndof, npress):
            raise FEM2DError("coupled u-p monolithic direct-fill block shape mismatch")
        b_t = b.T.tocsr()
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        for block, row_offset, col_offset in (
            (k, 0, 0),
            (b, 0, ndof),
            (b_t, ndof, 0),
            (p, ndof, ndof),
        ):
            block_rows, block_cols = _csr_pattern_rows_cols(block, row_offset=row_offset, col_offset=col_offset)
            if block_rows.size:
                rows.append(block_rows)
                cols.append(block_cols)
        all_rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        all_cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        template = csr_matrix((np.ones(all_rows.size, dtype=float), (all_rows, all_cols)), shape=(ndof + npress, ndof + npress))
        return cls(
            ndof=ndof,
            npress=npress,
            dt=float(dt),
            shape=template.shape,
            indices=template.indices.copy(),
            indptr=template.indptr.copy(),
            stiffness_indptr=k.indptr.copy(),
            stiffness_indices=k.indices.copy(),
            biot_indptr=b.indptr.copy(),
            biot_indices=b.indices.copy(),
            pressure_indptr=p.indptr.copy(),
            pressure_indices=p.indices.copy(),
            stiffness_positions=_csr_block_entry_positions(template, k),
            biot_upper_positions=_csr_block_entry_positions(template, b, col_offset=ndof),
            biot_lower_positions=_csr_block_entry_positions(template, b_t, row_offset=ndof),
            pressure_positions=_csr_block_entry_positions(template, p, row_offset=ndof, col_offset=ndof),
        )

    def matches(self, stiffness: csr_matrix, biot: csr_matrix, pressure_lhs: csr_matrix, dt: float, *, validate_structure: bool = True) -> bool:
        if abs(float(dt) - self.dt) > 0.0:
            return False
        k = stiffness.tocsr()
        b = biot.tocsr()
        p = pressure_lhs.tocsr()
        if k.shape != (self.ndof, self.ndof) or b.shape != (self.ndof, self.npress) or p.shape != (self.npress, self.npress):
            return False
        if not validate_structure:
            return True
        return (
            np.array_equal(k.indptr, self.stiffness_indptr)
            and np.array_equal(k.indices, self.stiffness_indices)
            and np.array_equal(b.indptr, self.biot_indptr)
            and np.array_equal(b.indices, self.biot_indices)
            and np.array_equal(p.indptr, self.pressure_indptr)
            and np.array_equal(p.indices, self.pressure_indices)
        )

    def assemble(self, stiffness: csr_matrix, biot: csr_matrix, pressure_lhs: csr_matrix) -> csr_matrix:
        data = np.zeros(self.indices.size, dtype=float)
        k = stiffness.tocsr()
        b = biot.tocsr()
        p = pressure_lhs.tocsr()
        data[self.stiffness_positions] = k.data
        data[self.biot_upper_positions] = -b.data
        if self.biot_lower_positions.size:
            data[self.biot_lower_positions] = b.T.tocsr().data / self.dt
        data[self.pressure_positions] = p.data
        return csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "coupled_up_monolithic_csr_pattern_cache",
            "unknowns": int(self.ndof + self.npress),
            "displacement_dofs": int(self.ndof),
            "pressure_dofs": int(self.npress),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "nnz": int(self.indices.size),
            "dt": float(self.dt),
            "direct_fill": {"enabled": True, "mode": "coupled_up_monolithic_flat_fill", "nnz": int(self.indices.size)},
            "cached_blocks": ["K", "-Bp", "Bp.T/dt", "pressure_lhs"],
        }


def _assemble_coupled_up_monolithic_lhs(
    stiffness: csr_matrix,
    biot: csr_matrix,
    pressure_lhs: csr_matrix,
    dt: float,
    *,
    cache: CoupledUPMonolithicMatrixCache | None = None,
    validate_structure: bool = True,
) -> tuple[csr_matrix, CoupledUPMonolithicMatrixCache, dict[str, Any]]:
    reused = bool(cache is not None and cache.matches(stiffness, biot, pressure_lhs, dt, validate_structure=validate_structure))
    active_cache = cache if reused and cache is not None else CoupledUPMonolithicMatrixCache.build(stiffness, biot, pressure_lhs, dt)
    matrix = active_cache.assemble(stiffness, biot, pressure_lhs)
    event = {**active_cache.info(), "reused": reused, "built": not reused}
    return matrix, active_cache, event


def _coupled_up_monolithic_cache_summary(events: list[Mapping[str, Any]], cache: CoupledUPMonolithicMatrixCache | None) -> dict[str, Any]:
    if not events and cache is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "solves": len(events),
        "hits": sum(1 for event in events if bool(event.get("reused", False))),
        "builds": sum(1 for event in events if bool(event.get("built", False))),
        "current": {} if cache is None else cache.info(),
    }


def _record_coupled_up_monolithic_cache_event(stats: dict[str, Any], event: Mapping[str, Any]) -> None:
    if bool(event.get("reused", False)):
        stats["monolithic_lhs_direct_fill_reuses"] = int(stats.get("monolithic_lhs_direct_fill_reuses", 0) or 0) + 1
    if bool(event.get("built", False)):
        stats["monolithic_lhs_direct_fill_builds"] = int(stats.get("monolithic_lhs_direct_fill_builds", 0) or 0) + 1


@dataclass(frozen=True)
class AxisymmetricUPArcLengthCorrectionCache:
    free_u: np.ndarray
    free_p: np.ndarray
    tangent_reduced_cache: ReducedMatrixCache
    pressure_reduced_cache: ReducedMatrixCache | None
    k_up: csr_matrix
    p_u: csr_matrix
    load_column_values: np.ndarray
    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray
    k_uu_positions: np.ndarray
    k_up_positions: np.ndarray
    load_column_positions: np.ndarray
    p_u_positions: np.ndarray
    p_p_positions: np.ndarray
    arc_u_positions: np.ndarray
    arc_l_position: int

    @classmethod
    def build(
        cls,
        tangent: csr_matrix,
        biot: csr_matrix,
        pressure_lhs: csr_matrix,
        reference_load: np.ndarray,
        free_u: np.ndarray,
        free_p: np.ndarray,
        *,
        dt: float,
    ) -> "AxisymmetricUPArcLengthCorrectionCache":
        free_u_arr = np.asarray(free_u, dtype=np.int64).ravel()
        free_p_arr = np.asarray(free_p, dtype=np.int64).ravel()
        empty_fixed = np.zeros(0, dtype=np.int64)
        tangent_csr = tangent.tocsr()
        pressure_csr = pressure_lhs.tocsr()
        tangent_cache = build_reduced_matrix_cache_from_csr(tangent_csr, free_u_arr, empty_fixed, source="axisymmetric_up_riks_tangent")
        k_uu = tangent_cache.extract_free_free(tangent_csr)
        if free_p_arr.size:
            pressure_cache = build_reduced_matrix_cache_from_csr(pressure_csr, free_p_arr, empty_fixed, source="axisymmetric_up_riks_pressure_lhs")
            p_p = pressure_cache.extract_free_free(pressure_csr)
            biot_csr = biot.tocsr()
            k_up = (-biot_csr[free_u_arr][:, free_p_arr]).tocsr()
            p_u = (biot_csr.T[free_p_arr][:, free_u_arr] / float(dt)).tocsr()
        else:
            pressure_cache = None
            p_p = csr_matrix((0, 0), dtype=float)
            k_up = csr_matrix((free_u_arr.size, 0), dtype=float)
            p_u = csr_matrix((0, free_u_arr.size), dtype=float)

        n_u = int(free_u_arr.size)
        n_p = int(free_p_arr.size)
        lambda_col = n_u + n_p
        total = lambda_col + 1
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        for block, row_offset, col_offset in (
            (k_uu, 0, 0),
            (k_up, 0, n_u),
            (p_u, n_u, 0),
            (p_p, n_u, n_u),
        ):
            block_rows, block_cols = _csr_pattern_rows_cols(block, row_offset=row_offset, col_offset=col_offset)
            if block_rows.size:
                rows.append(block_rows)
                cols.append(block_cols)
        if n_u:
            rows.append(np.arange(n_u, dtype=np.int64))
            cols.append(np.full(n_u, lambda_col, dtype=np.int64))
            rows.append(np.full(n_u, lambda_col, dtype=np.int64))
            cols.append(np.arange(n_u, dtype=np.int64))
        rows.append(np.asarray([lambda_col], dtype=np.int64))
        cols.append(np.asarray([lambda_col], dtype=np.int64))
        all_rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        all_cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        template = csr_matrix((np.ones(all_rows.size, dtype=float), (all_rows, all_cols)), shape=(total, total))

        load_rows = np.arange(n_u, dtype=np.int64)
        load_cols = np.full(n_u, lambda_col, dtype=np.int64)
        arc_rows = np.full(n_u, lambda_col, dtype=np.int64)
        arc_cols = np.arange(n_u, dtype=np.int64)
        return cls(
            free_u=free_u_arr.copy(),
            free_p=free_p_arr.copy(),
            tangent_reduced_cache=tangent_cache,
            pressure_reduced_cache=pressure_cache,
            k_up=k_up,
            p_u=p_u,
            load_column_values=-np.asarray(reference_load, dtype=float).ravel()[free_u_arr],
            shape=template.shape,
            indices=template.indices.copy(),
            indptr=template.indptr.copy(),
            k_uu_positions=_csr_block_entry_positions(template, k_uu),
            k_up_positions=_csr_block_entry_positions(template, k_up, col_offset=n_u),
            load_column_positions=_csr_dense_entry_positions(template, load_rows, load_cols),
            p_u_positions=_csr_block_entry_positions(template, p_u, row_offset=n_u),
            p_p_positions=_csr_block_entry_positions(template, p_p, row_offset=n_u, col_offset=n_u),
            arc_u_positions=_csr_dense_entry_positions(template, arc_rows, arc_cols),
            arc_l_position=int(_csr_dense_entry_positions(template, np.asarray([lambda_col]), np.asarray([lambda_col]))[0]),
        )

    def matches(self, tangent: csr_matrix, pressure_lhs: csr_matrix, reference_load: np.ndarray, free_u: np.ndarray, free_p: np.ndarray, *, validate_structure: bool = True) -> bool:
        free_u_arr = np.asarray(free_u, dtype=np.int64).ravel()
        free_p_arr = np.asarray(free_p, dtype=np.int64).ravel()
        if not np.array_equal(free_u_arr, self.free_u) or not np.array_equal(free_p_arr, self.free_p):
            return False
        if not np.allclose(-np.asarray(reference_load, dtype=float).ravel()[free_u_arr], self.load_column_values):
            return False
        empty_fixed = np.zeros(0, dtype=np.int64)
        tangent_csr = tangent.tocsr()
        if not self.tangent_reduced_cache.matches(tangent_csr, self.free_u, empty_fixed, validate_structure=validate_structure):
            return False
        if self.free_p.size:
            if self.pressure_reduced_cache is None:
                return False
            return self.pressure_reduced_cache.matches(pressure_lhs.tocsr(), self.free_p, empty_fixed, validate_structure=validate_structure)
        return True

    def assemble(self, tangent: csr_matrix, pressure_lhs: csr_matrix, du_step: np.ndarray, dl_step: float, psi: float) -> csr_matrix:
        data = np.zeros(self.indices.size, dtype=float)
        tangent_csr = tangent.tocsr()
        k_uu = self.tangent_reduced_cache.extract_free_free(tangent_csr)
        data[self.k_uu_positions] = k_uu.data
        if self.k_up_positions.size:
            data[self.k_up_positions] = self.k_up.data
        if self.load_column_positions.size:
            data[self.load_column_positions] = self.load_column_values
        if self.p_u_positions.size:
            data[self.p_u_positions] = self.p_u.data
        if self.free_p.size and self.pressure_reduced_cache is not None:
            p_p = self.pressure_reduced_cache.extract_free_free(pressure_lhs.tocsr())
            data[self.p_p_positions] = p_p.data
        if self.arc_u_positions.size:
            data[self.arc_u_positions] = 2.0 * np.asarray(du_step, dtype=float).ravel()
        data[self.arc_l_position] = 2.0 * float(psi) * float(psi) * float(dl_step)
        return csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    def info(self) -> dict[str, Any]:
        pressure_info = {"enabled": False} if self.pressure_reduced_cache is None else self.pressure_reduced_cache.info()
        return {
            "enabled": True,
            "cache_kind": "axisymmetric_up_arc_length_correction_cache",
            "free_displacement_dofs": int(self.free_u.size),
            "free_pressure_dofs": int(self.free_p.size),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "nnz": int(self.indices.size),
            "direct_fill": {"enabled": True, "mode": "axisymmetric_up_arc_length_flat_fill", "nnz": int(self.indices.size)},
            "tangent_reduced_matrix_cache": self.tangent_reduced_cache.info(),
            "pressure_reduced_matrix_cache": pressure_info,
            "coupling_blocks_cached": True,
        }


def _axisymmetric_up_arc_length_cache_summary(events: list[Mapping[str, Any]], cache: AxisymmetricUPArcLengthCorrectionCache | None) -> dict[str, Any]:
    if not events and cache is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "solves": len(events),
        "hits": sum(1 for event in events if bool(event.get("reused", False))),
        "builds": sum(1 for event in events if bool(event.get("built", False))),
        "current": {} if cache is None else cache.info(),
    }


@dataclass(frozen=True)
class AxisymmetricUPArcLengthLagrangeCorrectionCache:
    base_cache: AxisymmetricUPArcLengthCorrectionCache
    constraint_matrix: csr_matrix
    constraint_values: np.ndarray
    active_rows: np.ndarray
    c_active: csr_matrix
    c_active_transpose: csr_matrix
    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray
    base_positions: np.ndarray
    c_top_positions: np.ndarray
    c_bottom_positions: np.ndarray

    @classmethod
    def build(
        cls,
        tangent: csr_matrix,
        biot: csr_matrix,
        pressure_lhs: csr_matrix,
        reference_load: np.ndarray,
        free_u: np.ndarray,
        free_p: np.ndarray,
        constraint_matrix: csr_matrix,
        constraint_values: np.ndarray,
        active_rows: np.ndarray,
        *,
        dt: float,
    ) -> "AxisymmetricUPArcLengthLagrangeCorrectionCache":
        base_cache = AxisymmetricUPArcLengthCorrectionCache.build(
            tangent,
            biot,
            pressure_lhs,
            reference_load,
            free_u,
            free_p,
            dt=dt,
        )
        constraint_csr = constraint_matrix.tocsr()
        active_rows_arr = np.asarray(active_rows, dtype=bool).ravel()
        c_active = constraint_csr[:, base_cache.free_u][active_rows_arr].tocsr()
        c_active_transpose = c_active.T.tocsr()
        base_pattern = csr_matrix(
            (np.ones(base_cache.indices.size, dtype=float), base_cache.indices, base_cache.indptr),
            shape=base_cache.shape,
        )
        base_rows, base_cols = _csr_pattern_rows_cols(base_pattern)
        c_rows, c_cols = _csr_pattern_rows_cols(c_active)
        n_base = int(base_cache.shape[0])
        rows: list[np.ndarray] = [base_rows]
        cols: list[np.ndarray] = [base_cols]
        if c_rows.size:
            rows.extend((c_cols, c_rows + n_base))
            cols.extend((c_rows + n_base, c_cols))
        all_rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        all_cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        size = n_base + int(c_active.shape[0])
        template = csr_matrix((np.ones(all_rows.size, dtype=float), (all_rows, all_cols)), shape=(size, size))
        return cls(
            base_cache=base_cache,
            constraint_matrix=constraint_csr,
            constraint_values=np.asarray(constraint_values, dtype=float).ravel().copy(),
            active_rows=active_rows_arr.copy(),
            c_active=c_active,
            c_active_transpose=c_active_transpose,
            shape=template.shape,
            indices=template.indices.copy(),
            indptr=template.indptr.copy(),
            base_positions=_csr_block_entry_positions(template, base_pattern),
            c_top_positions=_csr_block_entry_positions(template, c_active_transpose, col_offset=n_base),
            c_bottom_positions=_csr_block_entry_positions(template, c_active, row_offset=n_base),
        )

    def matches(
        self,
        tangent: csr_matrix,
        pressure_lhs: csr_matrix,
        reference_load: np.ndarray,
        free_u: np.ndarray,
        free_p: np.ndarray,
    ) -> bool:
        return self.base_cache.matches(
            tangent,
            pressure_lhs,
            reference_load,
            free_u,
            free_p,
            validate_structure=True,
        )

    def assemble(
        self,
        tangent: csr_matrix,
        pressure_lhs: csr_matrix,
        du_step: np.ndarray,
        dl_step: float,
        psi: float,
    ) -> csr_matrix:
        base_system = self.base_cache.assemble(tangent, pressure_lhs, du_step, dl_step, psi)
        data = np.zeros(self.indices.size, dtype=float)
        data[self.base_positions] = base_system.data
        if self.c_active.nnz:
            data[self.c_top_positions] = self.c_active_transpose.data
            data[self.c_bottom_positions] = self.c_active.data
        return csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "axisymmetric_up_arc_length_lagrange_correction_cache",
            "active_mpc_rows": int(np.count_nonzero(self.active_rows)),
            "constraint_count": int(self.constraint_values.size),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "nnz": int(self.indices.size),
            "direct_fill": {
                "enabled": True,
                "mode": "axisymmetric_up_arc_length_lagrange_flat_fill",
                "nnz": int(self.indices.size),
            },
            "constraint_matrix_cached": True,
            "active_rows_cached": True,
            "base_coupled_cache": self.base_cache.info(),
        }


def _axisymmetric_up_arc_length_lagrange_cache_summary(
    events: list[Mapping[str, Any]],
    cache: AxisymmetricUPArcLengthLagrangeCorrectionCache | None,
) -> dict[str, Any]:
    if not events and cache is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "solves": len(events),
        "hits": sum(1 for event in events if bool(event.get("reused", False))),
        "builds": sum(1 for event in events if bool(event.get("built", False))),
        "current": {} if cache is None else cache.info(),
    }


@dataclass(frozen=True)
class ArcLengthLagrangeCorrectionCache:
    free: np.ndarray
    active_rows: np.ndarray
    value_count: int
    tangent_reduced_cache: ReducedMatrixCache
    c_active: csr_matrix
    load_column_values: np.ndarray
    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray
    kff_positions: np.ndarray
    load_column_positions: np.ndarray
    arc_u_positions: np.ndarray
    arc_l_position: int
    c_top_positions: np.ndarray
    c_bottom_positions: np.ndarray

    @classmethod
    def build(
        cls,
        tangent: csr_matrix,
        reference_load: np.ndarray,
        c_active: csr_matrix,
        active_rows: np.ndarray,
        value_count: int,
        free: np.ndarray,
    ) -> "ArcLengthLagrangeCorrectionCache":
        free_arr = np.asarray(free, dtype=np.int64).ravel()
        empty_fixed = np.zeros(0, dtype=np.int64)
        tangent_csr = tangent.tocsr()
        tangent_cache = build_reduced_matrix_cache_from_csr(tangent_csr, free_arr, empty_fixed, source="arc_length_lagrange_tangent")
        kff = tangent_cache.extract_free_free(tangent_csr)
        c_csr = c_active.tocsr()
        n_free = int(free_arr.size)
        n_mpc = int(c_csr.shape[0])
        lambda_col = n_free
        mpc_col0 = n_free + 1
        total = n_free + 1 + n_mpc
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        k_rows, k_cols = _csr_pattern_rows_cols(kff)
        if k_rows.size:
            rows.append(k_rows)
            cols.append(k_cols)
        c_rows, c_cols = _csr_pattern_rows_cols(c_csr)
        if c_rows.size:
            rows.append(c_cols)
            cols.append(c_rows + mpc_col0)
            rows.append(c_rows + mpc_col0)
            cols.append(c_cols)
        if n_free:
            rows.append(np.arange(n_free, dtype=np.int64))
            cols.append(np.full(n_free, lambda_col, dtype=np.int64))
            rows.append(np.full(n_free, lambda_col, dtype=np.int64))
            cols.append(np.arange(n_free, dtype=np.int64))
        rows.append(np.asarray([lambda_col], dtype=np.int64))
        cols.append(np.asarray([lambda_col], dtype=np.int64))
        all_rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        all_cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        template = csr_matrix((np.ones(all_rows.size, dtype=float), (all_rows, all_cols)), shape=(total, total))
        load_rows = np.arange(n_free, dtype=np.int64)
        load_cols = np.full(n_free, lambda_col, dtype=np.int64)
        arc_rows = np.full(n_free, lambda_col, dtype=np.int64)
        arc_cols = np.arange(n_free, dtype=np.int64)
        return cls(
            free=free_arr.copy(),
            active_rows=np.asarray(active_rows, dtype=bool).ravel().copy(),
            value_count=int(value_count),
            tangent_reduced_cache=tangent_cache,
            c_active=c_csr,
            load_column_values=-np.asarray(reference_load, dtype=float).ravel()[free_arr],
            shape=template.shape,
            indices=template.indices.copy(),
            indptr=template.indptr.copy(),
            kff_positions=_csr_block_entry_positions(template, kff),
            load_column_positions=_csr_dense_entry_positions(template, load_rows, load_cols),
            arc_u_positions=_csr_dense_entry_positions(template, arc_rows, arc_cols),
            arc_l_position=int(_csr_dense_entry_positions(template, np.asarray([lambda_col]), np.asarray([lambda_col]))[0]),
            c_top_positions=_csr_block_entry_positions(template, c_csr.T.tocsr(), col_offset=mpc_col0),
            c_bottom_positions=_csr_block_entry_positions(template, c_csr, row_offset=mpc_col0),
        )

    def matches(self, tangent: csr_matrix, reference_load: np.ndarray, c_active: csr_matrix, active_rows: np.ndarray, value_count: int, free: np.ndarray) -> bool:
        free_arr = np.asarray(free, dtype=np.int64).ravel()
        if int(value_count) != self.value_count:
            return False
        if not np.array_equal(free_arr, self.free):
            return False
        if not np.array_equal(np.asarray(active_rows, dtype=bool).ravel(), self.active_rows):
            return False
        if not np.allclose(-np.asarray(reference_load, dtype=float).ravel()[free_arr], self.load_column_values):
            return False
        c_csr = c_active.tocsr()
        if c_csr.shape != self.c_active.shape or c_csr.nnz != self.c_active.nnz:
            return False
        if not (np.array_equal(c_csr.indptr, self.c_active.indptr) and np.array_equal(c_csr.indices, self.c_active.indices) and np.allclose(c_csr.data, self.c_active.data)):
            return False
        empty_fixed = np.zeros(0, dtype=np.int64)
        return self.tangent_reduced_cache.matches(tangent.tocsr(), self.free, empty_fixed, validate_structure=True)

    def assemble(self, tangent: csr_matrix, du_step: np.ndarray, dl_step: float, psi: float) -> csr_matrix:
        data = np.zeros(self.indices.size, dtype=float)
        kff = self.tangent_reduced_cache.extract_free_free(tangent.tocsr())
        data[self.kff_positions] = kff.data
        data[self.load_column_positions] = self.load_column_values
        data[self.arc_u_positions] = 2.0 * np.asarray(du_step, dtype=float).ravel()
        data[self.arc_l_position] = 2.0 * float(psi) * float(psi) * float(dl_step)
        if self.c_active.nnz:
            data[self.c_top_positions] = self.c_active.T.tocsr().data
            data[self.c_bottom_positions] = self.c_active.data
        return csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "arc_length_lagrange_correction_cache",
            "free_dofs": int(self.free.size),
            "active_mpc_rows": int(np.count_nonzero(self.active_rows)),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "nnz": int(self.indices.size),
            "direct_fill": {"enabled": True, "mode": "arc_length_lagrange_flat_fill", "nnz": int(self.indices.size)},
            "constraint_matrix_cached": True,
            "active_rows_cached": True,
            "tangent_reduced_matrix_cache": self.tangent_reduced_cache.info(),
        }


def _arc_length_lagrange_cache_summary(events: list[Mapping[str, Any]], cache: ArcLengthLagrangeCorrectionCache | None) -> dict[str, Any]:
    if not events and cache is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "solves": len(events),
        "hits": sum(1 for event in events if bool(event.get("reused", False))),
        "builds": sum(1 for event in events if bool(event.get("built", False))),
        "current": {} if cache is None else cache.info(),
    }


@dataclass(frozen=True)
class LagrangeMPCLinearCorrectionCache:
    free: np.ndarray
    fixed: np.ndarray
    active_rows: np.ndarray
    value_count: int
    tangent_reduced_cache: ReducedMatrixCache
    c_active: csr_matrix
    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray
    kff_positions: np.ndarray
    c_top_positions: np.ndarray
    c_bottom_positions: np.ndarray

    @classmethod
    def build(
        cls,
        matrix: csr_matrix,
        c_active: csr_matrix,
        active_rows: np.ndarray,
        value_count: int,
        free: np.ndarray,
        fixed: np.ndarray,
    ) -> "LagrangeMPCLinearCorrectionCache":
        free_arr = np.asarray(free, dtype=np.int64).ravel()
        fixed_arr = np.asarray(fixed, dtype=np.int64).ravel()
        matrix_csr = matrix.tocsr()
        tangent_cache = build_reduced_matrix_cache_from_csr(matrix_csr, free_arr, fixed_arr, source="mpc_lagrange_linear_correction")
        kff = tangent_cache.extract_free_free(matrix_csr)
        c_csr = c_active.tocsr()
        n_free = int(free_arr.size)
        n_mpc = int(c_csr.shape[0])
        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        k_rows, k_cols = _csr_pattern_rows_cols(kff)
        if k_rows.size:
            rows.append(k_rows)
            cols.append(k_cols)
        c_rows, c_cols = _csr_pattern_rows_cols(c_csr)
        if c_rows.size:
            rows.append(c_cols)
            cols.append(c_rows + n_free)
            rows.append(c_rows + n_free)
            cols.append(c_cols)
        all_rows = np.concatenate(rows) if rows else np.zeros(0, dtype=np.int64)
        all_cols = np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64)
        template = csr_matrix((np.ones(all_rows.size, dtype=float), (all_rows, all_cols)), shape=(n_free + n_mpc, n_free + n_mpc))
        return cls(
            free=free_arr.copy(),
            fixed=fixed_arr.copy(),
            active_rows=np.asarray(active_rows, dtype=bool).ravel().copy(),
            value_count=int(value_count),
            tangent_reduced_cache=tangent_cache,
            c_active=c_csr,
            shape=template.shape,
            indices=template.indices.copy(),
            indptr=template.indptr.copy(),
            kff_positions=_csr_block_entry_positions(template, kff),
            c_top_positions=_csr_block_entry_positions(template, c_csr.T.tocsr(), col_offset=n_free),
            c_bottom_positions=_csr_block_entry_positions(template, c_csr, row_offset=n_free),
        )

    def matches(self, matrix: csr_matrix, c_active: csr_matrix, active_rows: np.ndarray, value_count: int, free: np.ndarray, fixed: np.ndarray) -> bool:
        free_arr = np.asarray(free, dtype=np.int64).ravel()
        fixed_arr = np.asarray(fixed, dtype=np.int64).ravel()
        if int(value_count) != self.value_count:
            return False
        if not np.array_equal(free_arr, self.free) or not np.array_equal(fixed_arr, self.fixed):
            return False
        if not np.array_equal(np.asarray(active_rows, dtype=bool).ravel(), self.active_rows):
            return False
        c_csr = c_active.tocsr()
        if c_csr.shape != self.c_active.shape or c_csr.nnz != self.c_active.nnz:
            return False
        if not (np.array_equal(c_csr.indptr, self.c_active.indptr) and np.array_equal(c_csr.indices, self.c_active.indices) and np.allclose(c_csr.data, self.c_active.data)):
            return False
        return self.tangent_reduced_cache.matches(matrix.tocsr(), self.free, self.fixed, validate_structure=True)

    def assemble(self, matrix: csr_matrix) -> csr_matrix:
        data = np.zeros(self.indices.size, dtype=float)
        kff = self.tangent_reduced_cache.extract_free_free(matrix.tocsr())
        if self.kff_positions.size:
            data[self.kff_positions] = kff.data
        if self.c_active.nnz:
            data[self.c_top_positions] = self.c_active.T.tocsr().data
            data[self.c_bottom_positions] = self.c_active.data
        return csr_matrix((data, self.indices, self.indptr), shape=self.shape)

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "mpc_lagrange_linear_correction_cache",
            "free_dofs": int(self.free.size),
            "fixed_dofs": int(self.fixed.size),
            "active_mpc_rows": int(np.count_nonzero(self.active_rows)),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "nnz": int(self.indices.size),
            "direct_fill": {"enabled": True, "mode": "mpc_lagrange_linear_flat_fill", "nnz": int(self.indices.size)},
            "constraint_matrix_cached": True,
            "active_rows_cached": True,
            "tangent_reduced_matrix_cache": self.tangent_reduced_cache.info(),
        }


def _lagrange_linear_cache_summary(events: list[Mapping[str, Any]], cache: LagrangeMPCLinearCorrectionCache | None) -> dict[str, Any]:
    if not events and cache is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "solves": len(events),
        "hits": sum(1 for event in events if bool(event.get("reused", False))),
        "builds": sum(1 for event in events if bool(event.get("built", False))),
        "current": {} if cache is None else cache.info(),
    }


def _solve_lagrange_mpc_linear_correction_cached(
    matrix: csr_matrix,
    rhs: np.ndarray,
    constrained: Mapping[int, float],
    mpc_info: Mapping[str, Any],
    current: np.ndarray,
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    cache: LagrangeMPCLinearCorrectionCache | None,
) -> tuple[np.ndarray, LagrangeMPCLinearCorrectionCache | None, dict[str, Any]]:
    total_start = _perf_counter()
    profile: dict[str, float] = {
        "constraint_matrix_elapsed_seconds": 0.0,
        "constraint_filter_elapsed_seconds": 0.0,
        "reduced_matrix_elapsed_seconds": 0.0,
        "cache_lookup_elapsed_seconds": 0.0,
        "cache_build_elapsed_seconds": 0.0,
        "bmat_elapsed_seconds": 0.0,
        "rhs_assembly_elapsed_seconds": 0.0,
        "linear_solve_elapsed_seconds": 0.0,
        "fallback_elapsed_seconds": 0.0,
        "total_elapsed_seconds": 0.0,
    }
    n = int(np.asarray(rhs).size)
    constraint_start = _perf_counter()
    constraint_matrix, values = _mpc_constraint_matrix(mpc_info, n, stage_name)
    profile["constraint_matrix_elapsed_seconds"] = max(_perf_counter() - constraint_start, 0.0)
    correction_constraints = {int(dof): float(value) - float(current[int(dof)]) for dof, value in constrained.items()}
    correction_values = values - np.asarray(constraint_matrix @ current).ravel()
    if correction_values.size == 0:
        fallback_start = _perf_counter()
        solution, info = solve_linear_system(matrix, rhs, stage_name=stage_name, solver=solver)
        profile["fallback_elapsed_seconds"] = max(_perf_counter() - fallback_start, 0.0)
        profile["linear_solve_elapsed_seconds"] = float(info.get("linear_solve_elapsed_seconds", 0.0) or 0.0)
        profile["total_elapsed_seconds"] = max(_perf_counter() - total_start, 0.0)
        return solution, cache, {"enabled": False, "reason": "no_mpc_constraints", "profile": profile, **profile}
    free, fixed = _free_index_arrays(n, correction_constraints, stage_name=stage_name)
    fixed_values = np.asarray([correction_constraints[int(dof)] for dof in fixed], dtype=float)
    filter_start = _perf_counter()
    c_free = constraint_matrix[:, free]
    constraint_rhs = correction_values.copy()
    if fixed.size:
        constraint_rhs = constraint_rhs - np.asarray(constraint_matrix[:, fixed] @ fixed_values).ravel()
    active_rows = np.asarray(np.abs(c_free).sum(axis=1)).ravel() > 0.0
    inconsistent = np.logical_not(active_rows) & (np.abs(constraint_rhs) > 1.0e-10)
    if np.any(inconsistent):
        raise FEM2DError(f"{stage_name}: MPC constraints conflict with fixed displacement constraints")
    c_active = c_free[active_rows].tocsr()
    constraint_rhs = constraint_rhs[active_rows]
    profile["constraint_filter_elapsed_seconds"] = max(_perf_counter() - filter_start, 0.0)
    if free.size == 0 or c_active.shape[0] == 0:
        fallback_start = _perf_counter()
        solution, info = _solve_lagrange_mpc_correction(matrix, rhs, constrained, mpc_info, current, stage_name=stage_name, solver=solver)
        profile["fallback_elapsed_seconds"] = max(_perf_counter() - fallback_start, 0.0)
        source_profile = info.get("profile", {}) if isinstance(info.get("profile", {}), Mapping) else info
        if isinstance(source_profile, Mapping):
            for key in ("reduced_matrix_elapsed_seconds", "bmat_elapsed_seconds", "linear_solve_elapsed_seconds"):
                profile[key] = float(source_profile.get(key, profile.get(key, 0.0)) or 0.0)
        profile["total_elapsed_seconds"] = max(_perf_counter() - total_start, 0.0)
        return solution, cache, {"enabled": False, "reason": "empty_free_or_active_constraint_rows", "profile": profile, **profile}
    matrix_csr = matrix.tocsr()
    cache_lookup_start = _perf_counter()
    reused = bool(cache is not None and cache.matches(matrix_csr, c_active, active_rows, values.size, free, fixed))
    profile["cache_lookup_elapsed_seconds"] = max(_perf_counter() - cache_lookup_start, 0.0)
    if reused and cache is not None:
        active_cache = cache
    else:
        cache_build_start = _perf_counter()
        active_cache = LagrangeMPCLinearCorrectionCache.build(matrix_csr, c_active, active_rows, values.size, free, fixed)
        profile["cache_build_elapsed_seconds"] = max(_perf_counter() - cache_build_start, 0.0)
    reduced_start = _perf_counter()
    reduced_rhs = active_cache.tangent_reduced_cache.reduced_rhs(matrix_csr, np.asarray(rhs, dtype=float).ravel(), fixed_values)
    profile["reduced_matrix_elapsed_seconds"] = max(_perf_counter() - reduced_start, 0.0)
    bmat_start = _perf_counter()
    saddle = active_cache.assemble(matrix_csr)
    profile["bmat_elapsed_seconds"] = max(_perf_counter() - bmat_start, 0.0)
    rhs_start = _perf_counter()
    augmented_rhs = _fill_two_block_vector(None, reduced_rhs, constraint_rhs)
    profile["rhs_assembly_elapsed_seconds"] = max(_perf_counter() - rhs_start, 0.0)
    linear_start = _perf_counter()
    augmented_solution, info = solve_linear_system(saddle, augmented_rhs, stage_name=stage_name, solver=solver)
    profile["linear_solve_elapsed_seconds"] = max(_perf_counter() - linear_start, 0.0)
    solution = np.zeros(n, dtype=float)
    if fixed.size:
        solution[fixed] = fixed_values
    if free.size:
        solution[free] = augmented_solution[: free.size]
    raw_multipliers = np.asarray(augmented_solution[free.size :], dtype=float).ravel()
    full_multipliers = np.zeros(values.size, dtype=float)
    full_multipliers[active_rows] = raw_multipliers
    residual_norm = float(np.linalg.norm(matrix_csr @ solution + constraint_matrix.T @ full_multipliers - np.asarray(rhs, dtype=float).ravel()))
    constraint_norm = float(np.linalg.norm(np.asarray(constraint_matrix @ solution).ravel() - correction_values, ord=np.inf))
    profile["total_elapsed_seconds"] = max(_perf_counter() - total_start, 0.0)
    event = {
        **active_cache.info(),
        "method": "mpc_lagrange_linear_correction_cached",
        "reused": reused,
        "built": not reused,
        "linear_method": str(info.get("method", "direct")),
        "iterations": int(info.get("iterations", 1) or 1),
        "residual_norm": residual_norm,
        "constraint_norm": constraint_norm,
        "equilibrated": bool(info.get("equilibrated", False)),
        "multipliers": [float(value) for value in full_multipliers],
        "profile": profile,
        **profile,
    }
    if isinstance(info.get("symbolic_cache"), Mapping):
        event["symbolic_cache"] = info["symbolic_cache"]
    if "factor_cache" in info:
        event["factor_cache"] = info["factor_cache"]
    return solution, active_cache, event


def _solve_arc_length_lagrange_correction_cached(
    tangent: csr_matrix,
    reference_load: np.ndarray,
    residual: np.ndarray,
    constrained: Mapping[int, float],
    mpc_info: Mapping[str, Any],
    u_trial: np.ndarray,
    free: np.ndarray,
    du_step: np.ndarray,
    dl_step: float,
    constraint_value: float,
    psi: float,
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    cache: ArcLengthLagrangeCorrectionCache | None,
) -> tuple[np.ndarray, float, list[float], ArcLengthLagrangeCorrectionCache | None, dict[str, Any]]:
    total_start = _perf_counter()
    profile: dict[str, float] = {
        "constraint_matrix_elapsed_seconds": 0.0,
        "constraint_filter_elapsed_seconds": 0.0,
        "cache_lookup_elapsed_seconds": 0.0,
        "cache_build_elapsed_seconds": 0.0,
        "bmat_elapsed_seconds": 0.0,
        "rhs_assembly_elapsed_seconds": 0.0,
        "linear_solve_elapsed_seconds": 0.0,
        "fallback_elapsed_seconds": 0.0,
        "total_elapsed_seconds": 0.0,
    }
    constraint_start = _perf_counter()
    constraint_matrix, values = _mpc_constraint_matrix(mpc_info, tangent.shape[0], stage_name)
    free_arr = np.asarray(free, dtype=np.int64).ravel()
    c_free = constraint_matrix[:, free_arr]
    rhs_mpc = values - np.asarray(constraint_matrix @ u_trial).ravel()
    active_rows = np.asarray(np.abs(c_free).sum(axis=1)).ravel() > 0.0
    inconsistent = np.logical_not(active_rows) & (np.abs(rhs_mpc) > 1.0e-10)
    if np.any(inconsistent):
        raise FEM2DError(f"{stage_name}: MPC constraints conflict with fixed displacement constraints")
    c_active = c_free[active_rows].tocsr()
    rhs_mpc = rhs_mpc[active_rows]
    profile["constraint_matrix_elapsed_seconds"] = max(_perf_counter() - constraint_start, 0.0)
    if c_active.shape[0] == 0:
        fallback_start = _perf_counter()
        du_corr, dl_corr, multipliers = _solve_arc_length_lagrange_correction(
            tangent,
            reference_load,
            residual,
            constrained,
            mpc_info,
            u_trial,
            free_arr,
            du_step,
            dl_step,
            constraint_value,
            psi,
            stage_name=stage_name,
            solver=solver,
        )
        profile["fallback_elapsed_seconds"] = max(_perf_counter() - fallback_start, 0.0)
        profile["total_elapsed_seconds"] = max(_perf_counter() - total_start, 0.0)
        return du_corr, dl_corr, multipliers, cache, {"enabled": False, "reason": "no_active_mpc_rows", "profile": profile, **profile}
    cache_lookup_start = _perf_counter()
    reused = bool(
        cache is not None
        and cache.matches(tangent, reference_load, c_active, active_rows, values.size, free_arr)
    )
    profile["cache_lookup_elapsed_seconds"] = max(_perf_counter() - cache_lookup_start, 0.0)
    if reused and cache is not None:
        active_cache = cache
    else:
        cache_build_start = _perf_counter()
        active_cache = ArcLengthLagrangeCorrectionCache.build(
            tangent,
            reference_load,
            c_active,
            active_rows,
            values.size,
            free_arr,
        )
        profile["cache_build_elapsed_seconds"] = max(_perf_counter() - cache_build_start, 0.0)
    bmat_start = _perf_counter()
    system = active_cache.assemble(tangent, du_step, dl_step, psi)
    profile["bmat_elapsed_seconds"] = max(_perf_counter() - bmat_start, 0.0)
    rhs_start = _perf_counter()
    rhs = _fill_three_block_vector(None, -np.asarray(residual, dtype=float).ravel()[free_arr], np.asarray([-constraint_value], dtype=float), rhs_mpc)
    profile["rhs_assembly_elapsed_seconds"] = max(_perf_counter() - rhs_start, 0.0)
    linear_start = _perf_counter()
    correction, _info = solve_linear_system(system, rhs, stage_name=stage_name, solver=solver)
    profile["linear_solve_elapsed_seconds"] = max(_perf_counter() - linear_start, 0.0)
    profile["total_elapsed_seconds"] = max(_perf_counter() - total_start, 0.0)
    full_multipliers = np.zeros(values.size, dtype=float)
    full_multipliers[active_rows] = correction[free_arr.size + 1 :]
    event = {**active_cache.info(), "reused": reused, "built": not reused, "profile": profile, **profile}
    return correction[: free_arr.size], float(correction[free_arr.size]), [float(v) for v in full_multipliers], active_cache, event


def _solve_axisymmetric_up_arc_length_correction_cached(
    tangent: csr_matrix,
    biot: csr_matrix,
    pressure_lhs: csr_matrix,
    reference_load: np.ndarray,
    residual: np.ndarray,
    pressure_residual: np.ndarray,
    free_u: np.ndarray,
    free_p: np.ndarray,
    du_step: np.ndarray,
    dl_step: float,
    constraint_value: float,
    psi: float,
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    dt: float,
    cache: AxisymmetricUPArcLengthCorrectionCache | None,
) -> tuple[np.ndarray, np.ndarray, float, AxisymmetricUPArcLengthCorrectionCache, dict[str, Any]]:
    free_u_arr = np.asarray(free_u, dtype=np.int64).ravel()
    free_p_arr = np.asarray(free_p, dtype=np.int64).ravel()
    reused = bool(
        cache is not None
        and cache.matches(tangent, pressure_lhs, reference_load, free_u_arr, free_p_arr, validate_structure=True)
    )
    active_cache = cache if reused and cache is not None else AxisymmetricUPArcLengthCorrectionCache.build(
        tangent,
        biot,
        pressure_lhs,
        reference_load,
        free_u_arr,
        free_p_arr,
        dt=dt,
    )
    system = active_cache.assemble(tangent, pressure_lhs, du_step, dl_step, psi)
    rhs = _fill_three_block_vector(
        None,
        -np.asarray(residual, dtype=float).ravel()[free_u_arr],
        -(np.asarray(pressure_residual, dtype=float).ravel()[free_p_arr] if free_p_arr.size else np.zeros(0, dtype=float)),
        np.asarray([-constraint_value], dtype=float),
    )
    correction, _info = solve_linear_system(system, rhs, stage_name=stage_name, solver=solver)
    event = {**active_cache.info(), "reused": reused, "built": not reused}
    n_u = int(free_u_arr.size)
    n_p = int(free_p_arr.size)
    return correction[:n_u], correction[n_u : n_u + n_p], float(correction[n_u + n_p]), active_cache, event


def _solve_axisymmetric_up_arc_length_lagrange_correction_cached(
    tangent: csr_matrix,
    biot: csr_matrix,
    pressure_lhs: csr_matrix,
    reference_load: np.ndarray,
    residual: np.ndarray,
    pressure_residual: np.ndarray,
    mpc_info: Mapping[str, Any],
    u_trial: np.ndarray,
    free_u: np.ndarray,
    free_p: np.ndarray,
    du_step: np.ndarray,
    dl_step: float,
    constraint_value: float,
    psi: float,
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    dt: float,
    cache: AxisymmetricUPArcLengthLagrangeCorrectionCache | None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    float,
    list[float],
    AxisymmetricUPArcLengthLagrangeCorrectionCache,
    dict[str, Any],
]:
    total_start = _perf_counter()
    profile: dict[str, float] = {
        "constraint_matrix_elapsed_seconds": 0.0,
        "constraint_filter_elapsed_seconds": 0.0,
        "cache_lookup_elapsed_seconds": 0.0,
        "cache_build_elapsed_seconds": 0.0,
        "bmat_elapsed_seconds": 0.0,
        "rhs_assembly_elapsed_seconds": 0.0,
        "linear_solve_elapsed_seconds": 0.0,
        "total_elapsed_seconds": 0.0,
    }
    free_u_arr = np.asarray(free_u, dtype=np.int64).ravel()
    free_p_arr = np.asarray(free_p, dtype=np.int64).ravel()
    if free_u_arr.size == 0:
        raise FEM2DError(f"{stage_name}: coupled axisymmetric Riks requires at least one free displacement dof")

    lookup_start = _perf_counter()
    reused = bool(
        cache is not None
        and cache.matches(tangent, pressure_lhs, reference_load, free_u_arr, free_p_arr)
    )
    profile["cache_lookup_elapsed_seconds"] = max(_perf_counter() - lookup_start, 0.0)
    if reused and cache is not None:
        active_cache = cache
    else:
        constraint_start = _perf_counter()
        constraint_matrix, values = _mpc_constraint_matrix(mpc_info, tangent.shape[0], stage_name)
        c_free = constraint_matrix[:, free_u_arr]
        active_rows = np.asarray(np.abs(c_free).sum(axis=1)).ravel() > 0.0
        profile["constraint_matrix_elapsed_seconds"] = max(_perf_counter() - constraint_start, 0.0)
        build_start = _perf_counter()
        active_cache = AxisymmetricUPArcLengthLagrangeCorrectionCache.build(
            tangent,
            biot,
            pressure_lhs,
            reference_load,
            free_u_arr,
            free_p_arr,
            constraint_matrix,
            values,
            active_rows,
            dt=dt,
        )
        profile["cache_build_elapsed_seconds"] = max(_perf_counter() - build_start, 0.0)

    filter_start = _perf_counter()
    rhs_mpc_all = active_cache.constraint_values - np.asarray(active_cache.constraint_matrix @ u_trial).ravel()
    inconsistent = np.logical_not(active_cache.active_rows) & (np.abs(rhs_mpc_all) > 1.0e-10)
    if np.any(inconsistent):
        raise FEM2DError(f"{stage_name}: MPC constraints conflict with fixed displacement constraints")
    rhs_mpc = rhs_mpc_all[active_cache.active_rows]
    profile["constraint_filter_elapsed_seconds"] = max(_perf_counter() - filter_start, 0.0)

    assembly_start = _perf_counter()
    system = active_cache.assemble(tangent, pressure_lhs, du_step, dl_step, psi)
    profile["bmat_elapsed_seconds"] = max(_perf_counter() - assembly_start, 0.0)
    rhs_start = _perf_counter()
    force_rhs = -np.asarray(residual, dtype=float).ravel()[free_u_arr]
    pressure_rhs = -(
        np.asarray(pressure_residual, dtype=float).ravel()[free_p_arr]
        if free_p_arr.size
        else np.zeros(0, dtype=float)
    )
    n_u = int(free_u_arr.size)
    n_p = int(free_p_arr.size)
    rhs = np.empty(n_u + n_p + 1 + rhs_mpc.size, dtype=float)
    rhs[:n_u] = force_rhs
    rhs[n_u : n_u + n_p] = pressure_rhs
    rhs[n_u + n_p] = -float(constraint_value)
    rhs[n_u + n_p + 1 :] = rhs_mpc
    profile["rhs_assembly_elapsed_seconds"] = max(_perf_counter() - rhs_start, 0.0)
    linear_start = _perf_counter()
    correction, linear_info = solve_linear_system(system, rhs, stage_name=stage_name, solver=solver)
    profile["linear_solve_elapsed_seconds"] = max(_perf_counter() - linear_start, 0.0)
    profile["total_elapsed_seconds"] = max(_perf_counter() - total_start, 0.0)

    full_multipliers = np.zeros(active_cache.constraint_values.size, dtype=float)
    if rhs_mpc.size:
        full_multipliers[active_cache.active_rows] = correction[n_u + n_p + 1 :]
    event: dict[str, Any] = {
        **active_cache.info(),
        "method": "axisymmetric_up_arc_length_lagrange_correction_cached",
        "reused": reused,
        "built": not reused,
        "linear_method": str(linear_info.get("method", "direct")),
        "iterations": int(linear_info.get("iterations", 1) or 1),
        "profile": profile,
        **profile,
    }
    if isinstance(linear_info.get("symbolic_cache"), Mapping):
        event["symbolic_cache"] = linear_info["symbolic_cache"]
    if "factor_cache" in linear_info:
        event["factor_cache"] = linear_info["factor_cache"]
    return (
        correction[:n_u],
        correction[n_u : n_u + n_p],
        float(correction[n_u + n_p]),
        [float(value) for value in full_multipliers],
        active_cache,
        event,
    )


@dataclass(frozen=True)
class ConsolidationStepCache:
    axisymmetric: bool
    ndof: int
    npress: int
    dt: float
    constrained_u: dict[int, float]
    fixed_p: dict[int, float]
    fixed_all: dict[int, float]
    free_all_dofs: np.ndarray
    fixed_all_dofs: np.ndarray
    fixed_all_values: np.ndarray
    free_pressure_dofs: np.ndarray
    fixed_pressure_dofs: np.ndarray
    load_vector: np.ndarray
    mass: csr_matrix
    conductivity: csr_matrix
    interface_matrix: csr_matrix
    biot: csr_matrix
    stiffness: csr_matrix
    boundary_matrix: csr_matrix
    boundary_rhs: np.ndarray
    boundary_info: dict[str, Any]
    pressure_lhs: csr_matrix
    monolithic_lhs: csr_matrix
    monolithic_lhs_cache: CoupledUPMonolithicMatrixCache | None
    mass_over_dt: csr_matrix
    biot_t_over_dt: csr_matrix
    zero_pressure_matrix: csr_matrix
    zero_pressure_vector: np.ndarray
    reduced_matrix_cache: ReducedMatrixCache | None
    hydraulic_assembly_info: dict[str, Any] | None = None
    reason: str = "linear_static_hydraulic_terms"

    def solver_info(self) -> dict[str, Any]:
        reduced_info = {"enabled": False} if self.reduced_matrix_cache is None else self.reduced_matrix_cache.info()
        return {
            "enabled": True,
            "cache_kind": "consolidation_step_cache",
            "axisymmetric": bool(self.axisymmetric),
            "reason": self.reason,
            "dt": float(self.dt),
            "unknowns": int(self.ndof + self.npress),
            "displacement_dofs": int(self.ndof),
            "pressure_dofs": int(self.npress),
            "fixed_displacement_dofs": len(self.constrained_u),
            "fixed_pressure_dofs": len(self.fixed_p),
            "free_coupled_dofs": int(self.free_all_dofs.size),
            "fixed_coupled_dofs": int(self.fixed_all_dofs.size),
            "pressure_lhs_cached": True,
            "monolithic_lhs_cached": True,
            "monolithic_lhs_nnz": int(self.monolithic_lhs.nnz),
            "pressure_lhs_nnz": int(self.pressure_lhs.nnz),
            "monolithic_lhs_direct_fill": {"enabled": False} if self.monolithic_lhs_cache is None else self.monolithic_lhs_cache.info(),
            "reduced_matrix_cache": reduced_info,
            "hydraulic_assembly": dict(self.hydraulic_assembly_info or {}),
            "cached_blocks": ["M/dt", "H", "interface_hydraulic_transfer", "Biot", "K", "pressure_boundary", "monolithic_lhs_pattern", "monolithic_lhs"],
        }


def build_small_deformation_step_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    *,
    interfaces: list[Any] | None = None,
    structural_elements: list[Any] | None = None,
    precompute_stiffness_pattern: bool = True,
) -> SmallDeformationStepCache:
    ndof = structural_total_dofs(mesh, structural_elements)
    constrained = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
    _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)
    fixed = np.asarray(sorted(constrained), dtype=np.int64)
    if fixed.size and (int(fixed[0]) < 0 or int(fixed[-1]) >= ndof):
        raise FEM2DError("small deformation cached constraint dof is outside the model dof range")
    mask = np.zeros(ndof, dtype=bool)
    if fixed.size:
        mask[fixed] = True
    free = np.nonzero(~mask)[0].astype(np.int64, copy=False)
    stiffness_cache = (
        build_global_stiffness_assembly_cache(
            mesh,
            materials,
            interfaces=interfaces,
            structural_elements=structural_elements,
            precompute_linear_element_stiffness=True,
        )
        if precompute_stiffness_pattern
        else None
    )
    reduced_matrix_cache = (
        build_reduced_matrix_cache_from_pattern(
            stiffness_cache.pattern.shape,
            stiffness_cache.pattern.indptr,
            stiffness_cache.pattern.indices,
            free,
            fixed,
            source="small_deformation_stiffness_pattern",
        )
        if stiffness_cache is not None
        else None
    )
    active_elements = [element.id for element in mesh.elements if element.active]
    quad4_mc_geometry_cache = build_quad4_mc_geometry_cache(mesh, materials)
    if not quad4_mc_geometry_cache.blocks:
        quad4_mc_geometry_cache = None
    return SmallDeformationStepCache(
        ndof=ndof,
        constrained=dict(constrained),
        free_dofs=free,
        fixed_dofs=fixed,
        active_elements=active_elements,
        stiffness_cache=stiffness_cache,
        reduced_matrix_cache=reduced_matrix_cache,
        quad4_mc_geometry_cache=quad4_mc_geometry_cache,
    )


def build_dynamic_mass_step_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    loads: Any,
    *,
    interfaces: list[Any] | None = None,
    structural_elements: list[Any] | None = None,
    lumped_mass: bool = False,
    precompute_stiffness_pattern: bool = True,
    precompute_mass_pattern: bool = True,
    precompute_load_vector: bool = True,
) -> DynamicMassStepCache:
    stiffness_cache = (
        build_global_stiffness_assembly_cache(
            mesh,
            materials,
            interfaces=interfaces,
            structural_elements=structural_elements,
            precompute_linear_element_stiffness=True,
        )
        if precompute_stiffness_pattern
        else None
    )
    mass_cache = (
        build_mass_matrix_assembly_cache(mesh, materials, structural_elements=structural_elements)
        if precompute_mass_pattern
        else None
    )
    load_cache = build_load_vector_assembly_cache(mesh, materials, loads, structural_elements=structural_elements) if precompute_load_vector else None
    return DynamicMassStepCache(stiffness_cache=stiffness_cache, mass_cache=mass_cache, load_vector_cache=load_cache, lumped_mass=bool(lumped_mass))


def _attach_load_mpc_step_cache(
    step_cache: StepCache2D | None,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    loads: Any,
    mpc_constraints: Any,
    *,
    interfaces: list[Any] | None = None,
    structural_elements: list[Any] | None = None,
    reason_scope: str = "factor",
) -> StepCache2D | None:
    if step_cache is None:
        return None
    large_deformation = isinstance(step_cache, LargeDeformationStepCache)
    scope = str(reason_scope or "factor").lower().strip()
    load_cached_reason = "cached_large_deformation_increment_base_load_vector" if scope in {"increment", "large_deformation_increment"} else "cached_factor_invariant_base_load_vector"
    mpc_cached_reason = "cached_large_deformation_increment_mpc_penalty" if scope in {"increment", "large_deformation_increment"} else "cached_factor_invariant_mpc_penalty"
    base_load: np.ndarray | None = None
    load_vector_assembly_cache = None
    load_reason = ""
    load_template_reason = ""
    if (not large_deformation) or _loads_reusable_on_updated_geometry(loads, structural_elements):
        base_load = assemble_load_vector(mesh, materials, loads, structural_elements=structural_elements)
        load_reason = load_cached_reason
    else:
        load_vector_assembly_cache = build_load_vector_assembly_cache(mesh, materials, loads, structural_elements=structural_elements)
        if load_vector_assembly_cache is not None:
            load_reason = "cached_geometry_dependent_load_template"
            load_template_reason = "cached_large_deformation_updated_coordinate_load_template"
        else:
            load_reason = "not_cached_geometry_dependent_loads"
            load_template_reason = "unsupported_geometry_dependent_load_template"

    mpc_matrix: csr_matrix | None = None
    mpc_vector: np.ndarray | None = None
    mpc_info: dict[str, Any] | None = None
    mpc_reason = ""
    specs = _ensure_list(mpc_constraints)
    if not specs:
        mpc_matrix = csr_matrix((step_cache.ndof, step_cache.ndof), dtype=float)
        mpc_vector = np.zeros(step_cache.ndof, dtype=float)
        mpc_info = {"count": 0, "penalty": 0.0, "max_violation": 0.0}
        mpc_reason = "cached_empty_mpc_penalty"
    elif (not large_deformation) or _mpc_penalty_reusable_on_updated_geometry(mpc_constraints):
        reference = (
            assemble_global_stiffness_cached(step_cache.stiffness_cache, mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
            if step_cache.stiffness_cache is not None
            else assemble_global_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
        )
        mpc_matrix, mpc_vector, mpc_info = assemble_mpc_penalty(mesh, reference, mpc_constraints)
        mpc_reason = mpc_cached_reason
    else:
        mpc_reason = "not_cached_default_penalty_depends_on_updated_stiffness"

    update: dict[str, Any] = {
        "base_load_vector": None if base_load is None else np.asarray(base_load, dtype=float).copy(),
        "load_vector_cache_reason": load_reason,
        "mpc_penalty_matrix": mpc_matrix,
        "mpc_penalty_vector": None if mpc_vector is None else np.asarray(mpc_vector, dtype=float).copy(),
        "mpc_info": None if mpc_info is None else dict(mpc_info),
        "mpc_penalty_cache_reason": mpc_reason,
    }
    if large_deformation:
        update["load_vector_assembly_cache"] = load_vector_assembly_cache
        update["load_vector_assembly_cache_reason"] = load_template_reason
    return replace(step_cache, **update)


def _loads_reusable_on_updated_geometry(loads: Any, structural_elements: list[Any] | None) -> bool:
    if structural_elements:
        return False
    for load in _ensure_list(loads):
        if not isinstance(load, Mapping):
            return False
        ltype = str(load.get("type", "")).lower().strip()
        if ltype in {"gravity", "self_weight", "body"} or bool(load.get("self_weight", False)):
            return False
        if "edge" in load or "edges" in load:
            return False
    return True


def _mpc_penalty_reusable_on_updated_geometry(mpc_constraints: Any) -> bool:
    for spec in _ensure_list(mpc_constraints):
        if not isinstance(spec, Mapping):
            return False
        if "penalty" not in spec:
            return False
    return True


def _load_vector_for_stage(
    step_cache: StepCache2D | None,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    loads: Any,
    structural_elements: list[Any] | None,
) -> tuple[np.ndarray, bool]:
    cached = None if step_cache is None else step_cache.base_load_vector
    if cached is not None:
        scale = float(getattr(step_cache, "load_scale", 1.0))
        return np.asarray(cached, dtype=float).copy() * scale, True
    load_template = None if step_cache is None else getattr(step_cache, "load_vector_assembly_cache", None)
    if load_template is not None:
        scale = float(getattr(step_cache, "load_scale", 1.0))
        return assemble_load_vector_cached(load_template, mesh, materials) * scale, True
    return assemble_load_vector(mesh, materials, loads, structural_elements=structural_elements), False


def _pore_pressure_load_for_stage(
    step_cache: StepCache2D | None,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    pore_pressure: np.ndarray,
    *,
    ndof: int,
) -> tuple[np.ndarray, bool]:
    cache = None if step_cache is None else getattr(step_cache, "pore_pressure_load_cache", None)
    if cache is not None:
        load = assemble_pore_pressure_load_cached(cache, mesh, materials, pore_pressure)
        reused = True
    else:
        load = assemble_pore_pressure_load(mesh, materials, pore_pressure)
        reused = False
    load = np.asarray(load, dtype=float).reshape(-1)
    if load.size == ndof:
        return load, reused
    if load.size > ndof:
        raise FEM2DError(f"pore-pressure load size {load.size} exceeds displacement dof count {ndof}")
    padded = np.zeros(ndof, dtype=float)
    padded[: load.size] = load
    return padded, reused


def _pore_pressure_load_cache_info(step_cache: StepCache2D | None, *, reused: bool) -> dict[str, Any]:
    cache = None if step_cache is None else getattr(step_cache, "pore_pressure_load_cache", None)
    if cache is None:
        return {"enabled": False, "reused": bool(reused)}
    return {**cache.info(), "reused": bool(reused)}


def _mpc_penalty_for_stage(
    step_cache: StepCache2D | None,
    mesh: Mesh2D,
    reference_stiffness: csr_matrix,
    mpc_constraints: Any,
) -> tuple[csr_matrix, np.ndarray, dict[str, Any], bool]:
    if (
        step_cache is not None
        and step_cache.mpc_penalty_matrix is not None
        and step_cache.mpc_penalty_vector is not None
        and step_cache.mpc_info is not None
        and step_cache.mpc_penalty_matrix.shape == reference_stiffness.shape
        and step_cache.mpc_penalty_vector.shape == (reference_stiffness.shape[0],)
    ):
        return step_cache.mpc_penalty_matrix, np.asarray(step_cache.mpc_penalty_vector, dtype=float).copy(), dict(step_cache.mpc_info), True
    Kmpc, Fmpc, mpc_info = assemble_mpc_penalty(mesh, reference_stiffness, mpc_constraints)
    return Kmpc, Fmpc, mpc_info, False


def _stage_load_mpc_cache_info(step_cache: StepCache2D | None, *, load_vector_reused: bool, mpc_penalty_reused: bool) -> dict[str, Any]:
    base_info = {}
    if step_cache is not None:
        info = step_cache.solver_info()
        raw = info.get("factor_invariant_load_mpc_cache", {})
        if isinstance(raw, Mapping):
            base_info.update(raw)
    return {
        "enabled": step_cache is not None and (bool(base_info.get("load_vector_cached", False)) or bool(base_info.get("load_vector_template_cached", False)) or bool(base_info.get("mpc_penalty_cached", False))),
        **base_info,
        "load_vector_reused": bool(load_vector_reused),
        "mpc_penalty_reused": bool(mpc_penalty_reused),
    }


def _consolidation_cache_enabled(solver: Mapping[str, Any] | None, consolidation: Mapping[str, Any]) -> bool:
    raw_solver = solver if isinstance(solver, Mapping) else {}
    raw = raw_solver.get("consolidation_step_cache", raw_solver.get("consolidation_cache", {}))
    if raw in (None, ""):
        raw = {}
    if isinstance(raw, bool):
        return bool(raw)
    if isinstance(raw, Mapping):
        return bool(raw.get("enabled", raw.get("cache_steps", raw.get("cache", True))))
    hydro = consolidation.get("hydro", consolidation.get("consolidation", consolidation))
    if isinstance(hydro, Mapping):
        raw_hydro = hydro.get("step_cache", hydro.get("cache_steps", None))
        if raw_hydro is not None:
            return bool(raw_hydro)
    return True


def _hydro_has_pressure_dependent_robin(hydro: Mapping[str, Any]) -> bool:
    robin_specs = hydro.get("pore_robin_bcs", hydro.get("robin_bcs", hydro.get("pore_robin", hydro.get("robin", []))))
    for spec in _ensure_list(robin_specs):
        if isinstance(spec, Mapping) and bool(spec.get("seepage_face", spec.get("seepage", False))):
            return True
    return False


def _hydro_has_liquefaction_coupling(mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial], hydro: Mapping[str, Any]) -> bool:
    requested = hydro.get("liquefaction_coupling", hydro.get("coupled_liquefaction", True))
    enabled = bool(requested.get("enabled", True)) if isinstance(requested, Mapping) else bool(requested)
    if not enabled:
        return False
    active_materials = {element.material for element in mesh.elements if element.active and element.material in materials}
    return any(_material_has_liquefaction(materials[name]) for name in active_materials)


def _disabled_consolidation_step_cache_info(reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "cache_kind": "consolidation_step_cache",
        "reason": reason,
        "pressure_lhs_cached": False,
        "monolithic_lhs_cached": False,
        "reduced_matrix_cache": {"enabled": False},
    }


def _consolidation_step_cache_stats() -> dict[str, Any]:
    return {
        "solves": 0,
        "monolithic_lhs_reuses": 0,
        "reduced_cache_reuses": 0,
        "reduced_cache_builds": 0,
        "factor_cache_hits": 0,
        "factor_cache_misses": 0,
        "symbolic_cache_hits": 0,
        "symbolic_cache_misses": 0,
        "preconditioner_reuses": 0,
        "boundary_cache_reuses": 0,
        "pressure_lhs_reuses": 0,
        "monolithic_lhs_direct_fill_builds": 0,
        "monolithic_lhs_direct_fill_reuses": 0,
        "lagrange_linear_cache_builds": 0,
        "lagrange_linear_cache_reuses": 0,
        "seepage_active_set_reuses": 0,
        "seepage_active_set_changes": 0,
    }


def _consolidation_step_cache_info(cache: ConsolidationStepCache | None, disabled_reason: str, stats: Mapping[str, Any]) -> dict[str, Any]:
    info = _disabled_consolidation_step_cache_info(disabled_reason) if cache is None else cache.solver_info()
    for key, value in stats.items():
        if isinstance(value, (int, np.integer)):
            info[key] = int(value)
        else:
            info[key] = value
    return info


def _build_consolidation_step_cache(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    hydro: Mapping[str, Any],
    solver: Mapping[str, Any] | None,
    consolidation: Mapping[str, Any],
    axisymmetric: bool,
    nonlinear: bool,
    dt: float,
    constrained_u: Mapping[int, float],
    fixed_p: Mapping[int, float],
    F: np.ndarray,
    M: csr_matrix,
    H: csr_matrix,
    H_interface: csr_matrix,
    Bp: csr_matrix,
    K: csr_matrix,
    mpc_plan: MPCStagePlan,
    hydraulic_assembly_info: Mapping[str, Any] | None = None,
) -> tuple[ConsolidationStepCache | None, str]:
    if not _consolidation_cache_enabled(solver, consolidation):
        return None, "disabled_by_solver"
    if nonlinear:
        return None, "disabled_for_nonlinear_coupled_stage"
    if _hydro_has_pressure_dependent_robin(hydro):
        return None, "disabled_for_pressure_dependent_seepage_boundary"
    if _hydro_has_liquefaction_coupling(mesh, materials, hydro):
        return None, "disabled_for_liquefaction_coupling"

    ndof = int(F.size)
    npress = int(M.shape[0])
    fixed_all: dict[int, float] = {int(dof): float(value) for dof, value in constrained_u.items()}
    for idx, value in fixed_p.items():
        fixed_all[ndof + int(idx)] = float(value)
    free_all, fixed_all_dofs = _free_index_arrays(ndof + npress, fixed_all, stage_name="consolidation cache")
    fixed_all_values = np.asarray([fixed_all[int(dof)] for dof in fixed_all_dofs], dtype=float)
    free_p, fixed_pressure = _free_index_arrays(npress, fixed_p, stage_name="consolidation cache", label="fixed pressure")
    boundary_cache = build_pressure_boundary_term_cache(mesh, hydro, axisymmetric=axisymmetric)
    Rb, qb, boundary_info = assemble_pressure_boundary_terms_cached(boundary_cache, pressure=None)
    hydraulic_info = dict(hydraulic_assembly_info or {})
    hydraulic_info["boundary_terms"] = boundary_cache.info()
    pressure_lhs = (M / dt + H + Rb + H_interface).tocsr()
    lhs, monolithic_lhs_cache, monolithic_event = _assemble_coupled_up_monolithic_lhs(K, Bp, pressure_lhs, dt)
    hydraulic_info["monolithic_lhs"] = {key: value for key, value in monolithic_event.items() if key not in {"indices", "indptr"}}
    reduced_cache = (
        None
        if mpc_plan.use_lagrange_linear
        else build_reduced_matrix_cache_from_csr(lhs, free_all, fixed_all_dofs, source="consolidation_monolithic_lhs")
    )
    zero_pressure_matrix = csr_matrix((npress, npress), dtype=float)
    return (
        ConsolidationStepCache(
            axisymmetric=axisymmetric,
            ndof=ndof,
            npress=npress,
            dt=float(dt),
            constrained_u={int(k): float(v) for k, v in constrained_u.items()},
            fixed_p={int(k): float(v) for k, v in fixed_p.items()},
            fixed_all=fixed_all,
            free_all_dofs=free_all,
            fixed_all_dofs=fixed_all_dofs,
            fixed_all_values=fixed_all_values,
            free_pressure_dofs=free_p,
            fixed_pressure_dofs=fixed_pressure,
            load_vector=np.asarray(F, dtype=float).copy(),
            mass=M.tocsr(),
            conductivity=H.tocsr(),
            interface_matrix=H_interface.tocsr(),
            biot=Bp.tocsr(),
            stiffness=K.tocsr(),
            boundary_matrix=Rb.tocsr(),
            boundary_rhs=np.asarray(qb, dtype=float).copy(),
            boundary_info=dict(boundary_info),
            pressure_lhs=pressure_lhs,
            monolithic_lhs=lhs,
            monolithic_lhs_cache=monolithic_lhs_cache,
            mass_over_dt=(M / dt).tocsr(),
            biot_t_over_dt=(Bp.T / dt).tocsr(),
            zero_pressure_matrix=zero_pressure_matrix,
            zero_pressure_vector=np.zeros(npress, dtype=float),
            reduced_matrix_cache=reduced_cache,
            hydraulic_assembly_info=hydraulic_info,
        ),
        "enabled",
    )


def _record_consolidation_linear_cache_info(stats: dict[str, Any], info: Mapping[str, Any]) -> None:
    reduced = info.get("reduced_matrix_cache", {}) if isinstance(info.get("reduced_matrix_cache", {}), Mapping) else {}
    if bool(reduced.get("reused", False)):
        stats["reduced_cache_reuses"] += 1
    if bool(reduced.get("built", False)):
        stats["reduced_cache_builds"] += 1
    factor_state = str(info.get("factor_cache", ""))
    if factor_state == "hit":
        stats["factor_cache_hits"] += 1
    elif factor_state == "miss":
        stats["factor_cache_misses"] += 1
    symbolic = info.get("symbolic_cache", {}) if isinstance(info.get("symbolic_cache", {}), Mapping) else {}
    symbolic_state = str(symbolic.get("state", ""))
    if symbolic_state == "hit":
        stats["symbolic_cache_hits"] += 1
    elif symbolic_state == "miss":
        stats["symbolic_cache_misses"] += 1
    stats["linear_solver_method"] = str(info.get("method", ""))
    stats["last_reduced_matrix_elapsed_seconds"] = float(info.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
    preconditioner_info = info.get("preconditioner_info", {})
    if isinstance(preconditioner_info, Mapping):
        stats["linear_preconditioner"] = dict(preconditioner_info)
        if bool(preconditioner_info.get("enabled", False)):
            stats["preconditioner_reuses"] = int(stats.get("preconditioner_reuses", 0) or 0) + 1


def _solve_consolidation_monolithic_system(
    lhs: csr_matrix,
    rhs: np.ndarray,
    fixed_all: Mapping[int, float],
    *,
    cache: ConsolidationStepCache | None,
    free_all_dofs: np.ndarray | None = None,
    fixed_all_dofs: np.ndarray | None = None,
    fixed_all_values: np.ndarray | None = None,
    reduction_cache: ReducedMatrixCache | None = None,
    lagrange_cache: LagrangeMPCLinearCorrectionCache | None = None,
    validate_reduction_cache: bool = True,
    mpc_plan: MPCStagePlan,
    mpc_info: Mapping[str, Any],
    stage_name: str,
    solver: Mapping[str, Any] | None,
    method: str,
    cache_stats: dict[str, Any],
    block_dof_ranges: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, float | None, ReducedMatrixCache | None, LagrangeMPCLinearCorrectionCache | None, dict[str, Any]]:
    if cache is not None or free_all_dofs is not None:
        cache_stats["solves"] += 1
    if cache is not None:
        cache_stats["monolithic_lhs_reuses"] += 1
        if cache.monolithic_lhs_cache is not None:
            cache_stats["monolithic_lhs_direct_fill_reuses"] = int(cache_stats.get("monolithic_lhs_direct_fill_reuses", 0) or 0) + 1
    if mpc_plan.use_lagrange_linear:
        solution, active_lagrange_cache, cache_event = _solve_lagrange_mpc_linear_correction_cached(
            lhs,
            rhs,
            fixed_all,
            mpc_info,
            np.zeros(rhs.size, dtype=float),
            stage_name=stage_name,
            solver=solver,
            cache=lagrange_cache,
        )
        if bool(cache_event.get("reused", False)):
            cache_stats["lagrange_linear_cache_reuses"] = int(cache_stats.get("lagrange_linear_cache_reuses", 0) or 0) + 1
        if bool(cache_event.get("built", False)):
            cache_stats["lagrange_linear_cache_builds"] = int(cache_stats.get("lagrange_linear_cache_builds", 0) or 0) + 1
        constraint_matrix, values = _mpc_constraint_matrix(mpc_info, rhs.size, stage_name)
        constraint_norm = float(np.linalg.norm(np.asarray(constraint_matrix @ solution).ravel() - values, ord=np.inf)) if values.size else 0.0
        mpc_solve_info = {
            "method": method,
            "linear_method": str(cache_event.get("linear_method", "direct")),
            "iterations": 1,
            "residual_norm": constraint_norm,
            "constraint_norm": constraint_norm,
            "lagrange_linear_cache": cache_event,
            "reduced_matrix_cache": active_lagrange_cache.tangent_reduced_cache.info() if active_lagrange_cache is not None else {"enabled": False},
            "profile": cache_event.get("profile", {}),
            **(cache_event.get("profile", {}) if isinstance(cache_event.get("profile", {}), Mapping) else {}),
        }
        _record_consolidation_linear_cache_info(cache_stats, mpc_solve_info)
        return solution, constraint_norm, reduction_cache, active_lagrange_cache, mpc_solve_info
    active_free = cache.free_all_dofs if cache is not None else free_all_dofs
    active_fixed = cache.fixed_all_dofs if cache is not None else fixed_all_dofs
    active_values = cache.fixed_all_values if cache is not None else fixed_all_values
    active_reduction_cache = cache.reduced_matrix_cache if cache is not None else reduction_cache
    solver_for_coupled = _solver_with_up_block_dof_ranges(solver, block_dof_ranges)
    if active_free is not None and active_fixed is not None and active_values is not None:
        free_solution, info, _reused_cache = solve_reduced_linear_system(
            lhs,
            rhs,
            active_free,
            active_fixed,
            fixed_values=active_values,
            reduction_cache=active_reduction_cache,
            stage_name=stage_name,
            solver=solver_for_coupled,
            validate_cache=validate_reduction_cache,
        )
        _record_consolidation_linear_cache_info(cache_stats, info)
        solution = np.zeros(rhs.size, dtype=float)
        if active_fixed.size:
            solution[active_fixed] = active_values
        if active_free.size:
            solution[active_free] = free_solution
        if not np.all(np.isfinite(solution)):
            raise FEM2DError(f"{stage_name}: cached consolidation solve produced non-finite values")
        return solution, None, _reused_cache, lagrange_cache, dict(info)
    solution = _solve_sparse_with_constraints(lhs, rhs, fixed_all, stage_name=stage_name, solver=solver_for_coupled)
    return solution, None, reduction_cache, lagrange_cache, {"method": "constrained_sparse", "reduced_matrix_elapsed_seconds": 0.0}


def _free_index_arrays(
    size: int,
    fixed_values: Any,
    *,
    stage_name: str | None = None,
    label: str = "constrained dof",
) -> tuple[np.ndarray, np.ndarray]:
    if fixed_values is None:
        raw: list[Any] = []
    elif isinstance(fixed_values, Mapping):
        raw = list(fixed_values.keys())
    else:
        raw = list(fixed_values)
    if not raw:
        return np.arange(size, dtype=int), np.zeros(0, dtype=int)
    fixed = np.asarray(sorted({int(value) for value in raw}), dtype=int)
    if fixed.size and (int(fixed[0]) < 0 or int(fixed[-1]) >= size):
        prefix = f"{stage_name}: " if stage_name else ""
        raise FEM2DError(f"{prefix}{label} index is outside the solution vector")
    mask = np.ones(size, dtype=bool)
    mask[fixed] = False
    return np.flatnonzero(mask), fixed


def _fill_two_block_vector(out: np.ndarray | None, first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_arr = np.asarray(first, dtype=float).ravel()
    second_arr = np.asarray(second, dtype=float).ravel()
    total = int(first_arr.size + second_arr.size)
    if out is None or out.shape != (total,):
        out = np.empty(total, dtype=float)
    out[: first_arr.size] = first_arr
    out[first_arr.size :] = second_arr
    return out


def _fill_three_block_vector(out: np.ndarray | None, first: np.ndarray, second: np.ndarray, third: np.ndarray) -> np.ndarray:
    first_arr = np.asarray(first, dtype=float).ravel()
    second_arr = np.asarray(second, dtype=float).ravel()
    third_arr = np.asarray(third, dtype=float).ravel()
    total = int(first_arr.size + second_arr.size + third_arr.size)
    if out is None or out.shape != (total,):
        out = np.empty(total, dtype=float)
    cursor = 0
    out[cursor : cursor + first_arr.size] = first_arr
    cursor += first_arr.size
    out[cursor : cursor + second_arr.size] = second_arr
    cursor += second_arr.size
    out[cursor : cursor + third_arr.size] = third_arr
    return out


def _offset_index_array(indices: np.ndarray, offset: int, out: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(indices, dtype=int).ravel()
    if out is None or out.shape != arr.shape:
        out = np.empty(arr.shape, dtype=int)
    out[:] = arr + int(offset)
    return out


def _materialized_plastic_state_for_postprocess(
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
) -> Mapping[str, PlasticState2D]:
    if plastic_state_cache is not None:
        overlay = None if isinstance(plastic_state, ArrayBackedPlasticStateMapping) else plastic_state
        return ArrayBackedPlasticStateMapping(plastic_state_cache, overlay)
    return plastic_state or {}


def _pop_same_pass_integration_point_rows(postprocess_info: dict[str, Any]) -> list[Mapping[str, Any]] | None:
    rows = postprocess_info.pop("_integration_point_rows", None)
    if rows is None:
        return None
    if not isinstance(rows, list):
        return None
    return [row for row in rows if isinstance(row, Mapping)]


def _pop_postprocess_state_array_cache(postprocess_info: dict[str, Any]) -> PlasticStateArrayCache | None:
    cache = postprocess_info.pop("_plastic_state_array_cache", None)
    return cache if isinstance(cache, PlasticStateArrayCache) else None


def _solver_with_up_block_dof_ranges(solver: Mapping[str, Any] | None, ranges: list[tuple[int, int]] | None) -> Mapping[str, Any] | None:
    if not ranges or not isinstance(solver, Mapping):
        return solver
    raw_linear = solver.get("linear", solver)
    if not isinstance(raw_linear, Mapping):
        return solver
    preconditioner = str(raw_linear.get("preconditioner", raw_linear.get("precond", ""))).lower().strip().replace("-", "_")
    if preconditioner not in {"up_block_jacobi", "up_block_ilu", "block_jacobi", "block_ilu"}:
        return solver
    if "block_dof_ranges" in raw_linear or "block_sizes" in raw_linear:
        return solver
    if "linear" in solver and isinstance(solver.get("linear"), Mapping):
        updated = dict(solver)
        linear = dict(raw_linear)
        linear["block_dof_ranges"] = [[int(start), int(end)] for start, end in ranges]
        updated["linear"] = linear
        return updated
    linear = dict(raw_linear)
    linear["block_dof_ranges"] = [[int(start), int(end)] for start, end in ranges]
    return linear


def solve_plane_strain_config(cfg: Mapping[str, Any], output_dir: str | Path | None = None) -> SolveResult2D:
    validate_2d_core_scope(cfg)
    analysis = cfg.get("analysis", {})
    analysis_type = str(analysis.get("type", "")).lower().strip() if isinstance(analysis, Mapping) else ""
    if analysis_type in AXISYMMETRIC_2D_STAGE_TYPES or (isinstance(analysis, Mapping) and bool(analysis.get("axisymmetric", False))):
        return solve_axisymmetric_config(cfg, output_dir)
    mesh = mesh_from_config(cfg)
    mesh_quality_preflight = validate_mesh_quality_for_solve(mesh, cfg)
    materials = plane_strain_materials(cfg)
    interfaces = interfaces_from_config(cfg, mesh)
    structural_elements = structural_elements_from_config(cfg, mesh)
    warnings = _collect_warnings(mesh, materials)
    _validate_material_references(mesh, materials)

    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), Mapping) else {}
    if output_dir is None:
        output_dir = out_cfg.get("directory", out_cfg.get("dir", "runs/plane_strain"))
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    stages_cfg = stage_sequence_from_config(cfg)

    stage_results: list[StageResult2D] = []
    active_element_ids = {element.id for element in mesh.elements if element.active}
    active_interface_ids = {interface.id for interface in interfaces if interface.active}
    active_structural_ids = {element.id for element in structural_elements if element.active}
    base_solver_cfg = cfg.get("solver", {})
    previous_pressure: np.ndarray | None = None
    plastic_state: dict[str, PlasticState2D] = {}
    public_stage_history: list[Mapping[str, Any]] = []

    def record_public_stage(result: StageResult2D, stage_cfg: Mapping[str, Any], stage_mesh: Mesh2D) -> None:
        result.solver_info.setdefault("mesh_quality_preflight", dict(mesh_quality_preflight))
        annotate_public_stage_result(result, cfg, stage_cfg, stage_mesh, materials, stage_history=public_stage_history)
        write_stage_public_profile(result)
        public_stage_history.append(stage_cfg)

    for idx, stage_cfg in enumerate(stages_cfg, start=1):
        if not isinstance(stage_cfg, Mapping):
            raise FEM2DError(f"stages[{idx}] must be a mapping")
        stage_name = stage_display_name(stage_cfg, idx)
        stage_type = _stage_type(stage_cfg)
        if stage_type in DEACTIVATION_2D_STAGE_TYPES:
            active_element_ids = _apply_deactivation_stage(mesh, active_element_ids, stage_cfg)
        active_interface_ids = _apply_stage_library_activity(active_interface_ids, {interface.id for interface in interfaces}, stage_cfg, "interface")
        active_structural_ids = _apply_stage_library_activity(active_structural_ids, {element.id for element in structural_elements}, stage_cfg, "structural")
        stage_mesh = _mesh_with_active_elements(mesh, active_element_ids)
        stage_interfaces = _interfaces_with_active(interfaces, active_interface_ids)
        stage_structural = structural_elements_with_active(structural_elements, active_structural_ids)
        merged_bc = stage_boundary_conditions(cfg, stage_cfg)
        merged_loads = stage_loads(cfg, stage_cfg, stage_type)
        merged_mpc = stage_mpc_constraints(cfg, stage_cfg)
        elapsed_time = _stage_time(stage_cfg, idx)
        merged_loads, load_info = _prepare_stage_loads(stage_mesh, cfg, stage_cfg, merged_loads, elapsed_time)
        stage_hydro, hydro_info = _prepare_stage_hydro(stage_mesh, cfg, stage_cfg, previous_pressure, elapsed_time)
        initial_stresses = _geostatic_initial_stresses(stage_mesh, materials, stage_cfg) if stage_type in GEOSTATIC_2D_STAGE_TYPES else None
        pore_pressure = None if stage_type in CONSOLIDATION_2D_STAGE_TYPES else _pore_pressure_from_hydro(stage_mesh, stage_hydro, previous_pressure)
        pressure_info: dict[str, Any] = dict(hydro_info)
        consolidation_cfg = _stage_with_hydro(stage_cfg, stage_hydro) if stage_type in CONSOLIDATION_2D_STAGE_TYPES else None
        stage_dir = out_root / _safe_name(stage_name)
        solver_cfg = _with_run_output_config(stage_solver_config(base_solver_cfg, stage_cfg), out_cfg)
        stage_started = time.perf_counter()
        if stage_type in LARGE_DEFORMATION_2D_STAGE_TYPES:
            result = solve_large_deformation_stage(
                mesh=stage_mesh,
                materials=materials,
                boundary_conditions=merged_bc,
                loads=merged_loads,
                mpc_constraints=merged_mpc,
                stage_name=stage_name,
                output_dir=stage_dir,
                solver=solver_cfg,
                stage_config=stage_cfg,
                initial_stresses=initial_stresses,
                interfaces=stage_interfaces,
                structural_elements=stage_structural,
                pore_pressure=pore_pressure,
                time=elapsed_time,
                plastic_state=plastic_state,
            )
            _attach_stage_runtime(result, stage_started, stage_mesh)
            previous_pressure, plastic_state = stage_state_after_result(result, previous_pressure)
            _attach_stage_load_hydro_info(result, load_info, hydro_info)
            record_public_stage(result, stage_cfg, stage_mesh)
            update_structural_element_histories(stage_structural, result.structural_results)
            update_interface_histories(stage_interfaces, result.interface_results)
            stage_results.append(result)
            continue
        if stage_type in DYNAMIC_2D_STAGE_TYPES:
            result = solve_dynamic_time_history_stage(
                mesh=stage_mesh,
                materials=materials,
                boundary_conditions=merged_bc,
                loads=merged_loads,
                mpc_constraints=merged_mpc,
                stage_name=stage_name,
                output_dir=stage_dir,
                solver=solver_cfg,
                stage_config=stage_cfg,
                global_config=cfg,
                interfaces=stage_interfaces,
                structural_elements=stage_structural,
                pore_pressure=pore_pressure,
                hydro=stage_hydro,
                time=elapsed_time,
                initial_stresses=initial_stresses,
                plastic_state=plastic_state,
            )
            _attach_stage_runtime(result, stage_started, stage_mesh)
            previous_pressure, plastic_state = stage_state_after_result(result, previous_pressure)
            _attach_stage_load_hydro_info(result, load_info, hydro_info)
            record_public_stage(result, stage_cfg, stage_mesh)
            update_structural_element_histories(stage_structural, result.structural_results)
            update_interface_histories(stage_interfaces, result.interface_results)
            stage_results.append(result)
            continue
        if stage_type in SRM_2D_STAGE_TYPES:
            solver_cfg = _with_run_output_config(stage_srm_solver_config(solver_cfg, stage_cfg), out_cfg)
            result = solve_srm_stage(
                mesh=stage_mesh,
                materials=materials,
                boundary_conditions=merged_bc,
                loads=merged_loads,
                mpc_constraints=merged_mpc,
                stage_name=stage_name,
                output_dir=stage_dir,
                solver=solver_cfg,
                initial_stresses=initial_stresses,
                interfaces=stage_interfaces,
                structural_elements=stage_structural,
                plastic_state=plastic_state,
            )
            _attach_stage_runtime(result, stage_started, stage_mesh)
            previous_pressure, plastic_state = stage_state_after_result(result, previous_pressure)
            _attach_stage_load_hydro_info(result, load_info, hydro_info)
            record_public_stage(result, stage_cfg, stage_mesh)
            update_structural_element_histories(stage_structural, result.structural_results)
            update_interface_histories(stage_interfaces, result.interface_results)
            stage_results.append(result)
            continue
        if stage_type in RIKS_2D_STAGE_TYPES:
            solver_cfg = _with_run_output_config(stage_riks_solver_config(solver_cfg, stage_cfg), out_cfg)
            result = solve_riks_stage(
                mesh=stage_mesh,
                materials=materials,
                boundary_conditions=merged_bc,
                loads=merged_loads,
                mpc_constraints=merged_mpc,
                stage_name=stage_name,
                output_dir=stage_dir,
                solver=solver_cfg,
                initial_stresses=initial_stresses,
                interfaces=stage_interfaces,
                structural_elements=stage_structural,
                pore_pressure=pore_pressure,
                plastic_state=plastic_state,
            )
            _attach_stage_runtime(result, stage_started, stage_mesh)
            previous_pressure, plastic_state = stage_state_after_result(result, previous_pressure)
            _attach_stage_load_hydro_info(result, load_info, hydro_info)
            record_public_stage(result, stage_cfg, stage_mesh)
            update_structural_element_histories(stage_structural, result.structural_results)
            update_interface_histories(stage_interfaces, result.interface_results)
            stage_results.append(result)
            continue
        result = solve_plane_strain_stage(
            mesh=stage_mesh,
            materials=materials,
            boundary_conditions=merged_bc,
            loads=merged_loads,
            mpc_constraints=merged_mpc,
            stage_name=stage_name,
            output_dir=stage_dir,
            solver=solver_cfg,
            initial_stresses=initial_stresses,
            interfaces=stage_interfaces,
            structural_elements=stage_structural,
            pore_pressure=pore_pressure,
            time=elapsed_time,
            plastic_state=plastic_state,
            consolidation=consolidation_cfg,
            previous_pressure=previous_pressure,
        )
        _attach_stage_runtime(result, stage_started, stage_mesh)
        previous_pressure, plastic_state = stage_state_after_result(result, previous_pressure)
        _attach_stage_load_hydro_info(result, load_info, hydro_info)
        record_public_stage(result, stage_cfg, stage_mesh)
        update_structural_element_histories(stage_structural, result.structural_results)
        update_interface_histories(stage_interfaces, result.interface_results)
        stage_results.append(result)

    warnings.extend(public_profile_run_warnings(cfg, mesh, materials, [stage for stage in stages_cfg if isinstance(stage, Mapping)]))
    summary = SolveResult2D(mesh=mesh, materials=materials, stages=stage_results, output_dir=out_root, interfaces=interfaces, structural_elements=structural_elements, warnings=warnings)
    summary.input_config = dict(cfg)
    write_run_summary(summary)
    return summary


def solve_axisymmetric_config(cfg: Mapping[str, Any], output_dir: str | Path | None = None) -> SolveResult2D:
    validate_2d_core_scope(cfg)
    mesh = mesh_from_config(cfg)
    mesh_quality_preflight = validate_mesh_quality_for_solve(mesh, cfg)
    materials = plane_strain_materials(cfg)
    interfaces = interfaces_from_config(cfg, mesh)
    structural_elements = structural_elements_from_config(cfg, mesh)
    warnings = _collect_warnings(mesh, materials)
    _validate_material_references(mesh, materials)
    out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), Mapping) else {}
    if output_dir is None:
        output_dir = out_cfg.get("directory", out_cfg.get("dir", "runs/axisymmetric"))
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    stages_cfg = stage_sequence_from_config(cfg)
    stage_results: list[StageResult2D] = []
    base_solver_cfg = cfg.get("solver", {})
    plastic_state: dict[str, PlasticState2D] = {}
    active_element_ids = {element.id for element in mesh.elements if element.active}
    active_interface_ids = {interface.id for interface in interfaces if interface.active}
    active_structural_ids = {element.id for element in structural_elements if element.active}
    previous_pressure: np.ndarray | None = None
    public_stage_history: list[Mapping[str, Any]] = []

    def record_public_stage(result: StageResult2D, stage_cfg: Mapping[str, Any], stage_mesh: Mesh2D) -> None:
        result.solver_info.setdefault("mesh_quality_preflight", dict(mesh_quality_preflight))
        annotate_public_stage_result(result, cfg, stage_cfg, stage_mesh, materials, stage_history=public_stage_history)
        write_stage_public_profile(result)
        public_stage_history.append(stage_cfg)

    for idx, stage_cfg in enumerate(stages_cfg, start=1):
        if not isinstance(stage_cfg, Mapping):
            raise FEM2DError(f"stages[{idx}] must be a mapping")
        stage_type = _stage_type(stage_cfg)
        if stage_type not in {"", "static", "linear_static", *AXISYMMETRIC_2D_STAGE_TYPES, *SRM_2D_STAGE_TYPES, *GEOSTATIC_2D_STAGE_TYPES, *DEACTIVATION_2D_STAGE_TYPES, *CONSOLIDATION_2D_STAGE_TYPES, *RIKS_2D_STAGE_TYPES, *DYNAMIC_2D_STAGE_TYPES, *LARGE_DEFORMATION_2D_STAGE_TYPES}:
            raise FEM2DError(f"axisymmetric core stage type '{stage_type}' is not implemented yet")
        if stage_type in DYNAMIC_2D_STAGE_TYPES:
            raise FEM2DError(f"axisymmetric dynamic time history stage '{stage_type}' is not implemented yet")
        if stage_type in LARGE_DEFORMATION_2D_STAGE_TYPES:
            raise FEM2DError(f"axisymmetric large-deformation stage '{stage_type}' is not implemented yet")
        if stage_type in DEACTIVATION_2D_STAGE_TYPES:
            active_element_ids = _apply_deactivation_stage(mesh, active_element_ids, stage_cfg)
        active_interface_ids = _apply_stage_library_activity(active_interface_ids, {interface.id for interface in interfaces}, stage_cfg, "interface")
        active_structural_ids = _apply_stage_library_activity(active_structural_ids, {element.id for element in structural_elements}, stage_cfg, "structural")
        stage_mesh = _mesh_with_active_elements(mesh, active_element_ids)
        stage_interfaces = _interfaces_with_active(interfaces, active_interface_ids)
        stage_structural = structural_elements_with_active(structural_elements, active_structural_ids)
        stage_name = stage_display_name(stage_cfg, idx)
        merged_bc = stage_boundary_conditions(cfg, stage_cfg)
        merged_loads = stage_loads(cfg, stage_cfg, stage_type)
        merged_mpc = stage_mpc_constraints(cfg, stage_cfg)
        elapsed_time = _stage_time(stage_cfg, idx)
        merged_loads, load_info = _prepare_stage_loads(stage_mesh, cfg, stage_cfg, merged_loads, elapsed_time)
        stage_hydro, hydro_info = _prepare_stage_hydro(stage_mesh, cfg, stage_cfg, previous_pressure, elapsed_time)
        initial_stresses = _geostatic_initial_stresses(stage_mesh, materials, stage_cfg) if stage_type in GEOSTATIC_2D_STAGE_TYPES else None
        solver_cfg = _with_run_output_config(stage_solver_config(base_solver_cfg, stage_cfg), out_cfg)
        stage_started = time.perf_counter()
        if stage_type in SRM_2D_STAGE_TYPES:
            solver_cfg = _with_run_output_config(stage_srm_solver_config(solver_cfg, stage_cfg), out_cfg)
            result = solve_axisymmetric_srm_stage(
                mesh=stage_mesh,
                materials=materials,
                boundary_conditions=merged_bc,
                loads=merged_loads,
                mpc_constraints=merged_mpc,
                stage_name=stage_name,
                output_dir=out_root / _safe_name(stage_name),
                solver=solver_cfg,
                plastic_state=plastic_state,
                initial_stresses=initial_stresses,
                interfaces=stage_interfaces,
                structural_elements=stage_structural,
            )
            _attach_stage_runtime(result, stage_started, stage_mesh)
            previous_pressure, plastic_state = stage_state_after_result(result, previous_pressure, copy_pressure=True)
            _attach_stage_load_hydro_info(result, load_info, hydro_info)
            record_public_stage(result, stage_cfg, stage_mesh)
            update_structural_element_histories(stage_structural, result.structural_results)
            update_interface_histories(stage_interfaces, result.interface_results)
            stage_results.append(result)
            continue
        if stage_type in CONSOLIDATION_2D_STAGE_TYPES:
            result = solve_axisymmetric_coupled_up_stage(
                mesh=stage_mesh,
                materials=materials,
                boundary_conditions=merged_bc,
                loads=merged_loads,
                mpc_constraints=merged_mpc,
                stage_name=stage_name,
                output_dir=out_root / _safe_name(stage_name),
                solver=solver_cfg,
                initial_stresses=initial_stresses,
                strength_factor=1.0,
                plastic_state=plastic_state,
                consolidation=_stage_with_hydro(stage_cfg, stage_hydro),
                previous_pressure=previous_pressure,
                interfaces=stage_interfaces,
                structural_elements=stage_structural,
            )
            _attach_stage_runtime(result, stage_started, stage_mesh)
            previous_pressure, plastic_state = stage_state_after_result(result, previous_pressure, copy_pressure=True, missing_pressure="clear")
            _attach_stage_load_hydro_info(result, load_info, hydro_info)
            record_public_stage(result, stage_cfg, stage_mesh)
            update_structural_element_histories(stage_structural, result.structural_results)
            update_interface_histories(stage_interfaces, result.interface_results)
            stage_results.append(result)
            continue
        if stage_type in RIKS_2D_STAGE_TYPES:
            solver_cfg = _with_run_output_config(stage_riks_solver_config(solver_cfg, stage_cfg), out_cfg)
            result = solve_axisymmetric_riks_stage(
                mesh=stage_mesh,
                materials=materials,
                boundary_conditions=merged_bc,
                loads=merged_loads,
                mpc_constraints=merged_mpc,
                stage_name=stage_name,
                output_dir=out_root / _safe_name(stage_name),
                solver=solver_cfg,
                initial_stresses=initial_stresses,
                plastic_state=plastic_state,
                interfaces=stage_interfaces,
                structural_elements=stage_structural,
                consolidation=_stage_with_hydro(stage_cfg, stage_hydro) if _stage_has_hydro_coupling(stage_cfg) or stage_hydro else None,
                previous_pressure=previous_pressure,
            )
            _attach_stage_runtime(result, stage_started, stage_mesh)
            previous_pressure, plastic_state = stage_state_after_result(result, previous_pressure, copy_pressure=True)
            _attach_stage_load_hydro_info(result, load_info, hydro_info)
            record_public_stage(result, stage_cfg, stage_mesh)
            update_structural_element_histories(stage_structural, result.structural_results)
            update_interface_histories(stage_interfaces, result.interface_results)
            stage_results.append(result)
            continue
        result = solve_axisymmetric_stage(
            mesh=stage_mesh,
            materials=materials,
            boundary_conditions=merged_bc,
            loads=merged_loads,
            mpc_constraints=merged_mpc,
            stage_name=stage_name,
            output_dir=out_root / _safe_name(stage_name),
            solver=solver_cfg,
            plastic_state=plastic_state,
            initial_stresses=initial_stresses,
            interfaces=stage_interfaces,
            structural_elements=stage_structural,
        )
        _attach_stage_runtime(result, stage_started, stage_mesh)
        previous_pressure, plastic_state = stage_state_after_result(result, previous_pressure, copy_pressure=True)
        _attach_stage_load_hydro_info(result, load_info, hydro_info)
        record_public_stage(result, stage_cfg, stage_mesh)
        update_structural_element_histories(stage_structural, result.structural_results)
        update_interface_histories(stage_interfaces, result.interface_results)
        stage_results.append(result)
    warnings.extend(public_profile_run_warnings(cfg, mesh, materials, [stage for stage in stages_cfg if isinstance(stage, Mapping)]))
    summary = SolveResult2D(mesh=mesh, materials=materials, stages=stage_results, output_dir=out_root, interfaces=interfaces, structural_elements=structural_elements, warnings=warnings)
    summary.input_config = dict(cfg)
    write_run_summary(summary)
    return summary


def solve_axisymmetric_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str = "Stage-1",
    output_dir: str | Path | None = None,
    solver: Mapping[str, Any] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    initial_displacement: np.ndarray | None = None,
    postprocess_results: bool = True,
) -> StageResult2D:
    _validate_mesh(mesh)
    _validate_material_references(mesh, materials)
    solver_cfg = solver if isinstance(solver, Mapping) else {}
    nonlinear = any(material.is_plastic for material in materials.values()) or _has_nonlinear_interfaces(interfaces) or structural_has_nonlinear(structural_elements)
    axisymmetric_step_cache: AxisymmetricStepCache | None = None
    auto_step_cache = False
    static_step_cache_info: dict[str, Any] | None = None
    cache_build_elapsed = 0.0
    stiffness_assembly_elapsed = 0.0
    load_assembly_elapsed = 0.0
    constraint_collection_elapsed = 0.0
    mpc_assembly_elapsed = 0.0
    mpc_penalty_apply_elapsed = 0.0
    linear_solve_elapsed = 0.0
    nonlinear_solve_elapsed = 0.0
    postprocess_elapsed = 0.0
    io_report_elapsed = 0.0
    stiffness_cache_used = False
    cache_settings = _static_step_cache_settings(solver_cfg)
    if bool(cache_settings.get("enabled", True)):
        cache_start = _perf_counter()
        axisymmetric_step_cache = build_axisymmetric_step_cache(
            mesh,
            boundary_conditions,
            materials=materials,
            interfaces=interfaces,
            structural_elements=structural_elements,
            precompute_stiffness_pattern=bool(cache_settings.get("precompute_stiffness_pattern", True)),
        )
        cache_build_elapsed = max(_perf_counter() - cache_start, 0.0)
        auto_step_cache = True
    else:
        static_step_cache_info = _disabled_static_step_cache_info("disabled_by_solver_setting", geometry_mode="axisymmetric")
    stiffness_start = _perf_counter()
    if axisymmetric_step_cache is not None and axisymmetric_step_cache.stiffness_cache is not None:
        K = assemble_axisymmetric_stiffness_cached(
            axisymmetric_step_cache.stiffness_cache,
            mesh,
            materials,
            interfaces=interfaces,
            structural_elements=structural_elements,
        )
        stiffness_cache_used = True
    else:
        K = assemble_axisymmetric_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    stiffness_assembly_elapsed = max(_perf_counter() - stiffness_start, 0.0)
    load_start = _perf_counter()
    F = assemble_axisymmetric_load_vector(mesh, materials, loads)
    load_assembly_elapsed = max(_perf_counter() - load_start, 0.0)
    if axisymmetric_step_cache is not None:
        constrained = dict(axisymmetric_step_cache.constrained)
    else:
        constraint_start = _perf_counter()
        constrained = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
        _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)
        constraint_collection_elapsed = max(_perf_counter() - constraint_start, 0.0)
    mpc_start = _perf_counter()
    Kmpc, Fmpc, mpc_info = assemble_mpc_penalty(mesh, K, mpc_constraints)
    mpc_assembly_elapsed = max(_perf_counter() - mpc_start, 0.0)
    mpc_plan = mpc_stage_plan(
        mpc_constraints,
        solver_cfg,
        mpc_info,
        nonlinear=nonlinear,
        add_plain_penalty_to_stage_matrix=not nonlinear,
    )
    if mpc_plan.add_penalty_to_stage_matrix:
        mpc_apply_start = _perf_counter()
        K = (K + Kmpc).tocsr()
        F = F + Fmpc
        mpc_penalty_apply_elapsed = max(_perf_counter() - mpc_apply_start, 0.0)
    if nonlinear:
        nonlinear_start = _perf_counter()
        u, reactions, solver_info = solve_axisymmetric_nonlinear_system(
            mesh,
            materials,
            F,
            constrained,
            stage_name=stage_name,
            solver=solver_cfg,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            initial_stresses=initial_stresses,
            interfaces=interfaces,
            structural_elements=structural_elements,
            mpc_stiffness=Kmpc,
            mpc_load=Fmpc,
            mpc_info=mpc_info,
            mpc_lagrange=mpc_plan.lagrange_requested,
            free_dofs=axisymmetric_step_cache.free_dofs if axisymmetric_step_cache is not None else None,
            fixed_dofs=axisymmetric_step_cache.fixed_dofs if axisymmetric_step_cache is not None else None,
            sparse_pattern=axisymmetric_step_cache.stiffness_pattern if axisymmetric_step_cache is not None else None,
            reduced_matrix_cache=axisymmetric_step_cache.reduced_matrix_cache if axisymmetric_step_cache is not None else None,
            initial_displacement=initial_displacement,
        )
        nonlinear_solve_elapsed = max(_perf_counter() - nonlinear_start, 0.0)
    elif mpc_plan.use_elimination_linear:
        linear_start = _perf_counter()
        u, solver_info = solve_linear_system_with_mpc_elimination(K, F, constrained, mpc_info, stage_name=stage_name, solver=solver_cfg)
        linear_solve_elapsed = max(_perf_counter() - linear_start, 0.0)
        reactions = K @ u - F
    elif mpc_plan.use_lagrange_linear:
        linear_start = _perf_counter()
        u, solver_info = solve_linear_system_with_mpc_lagrange(K, F, constrained, mpc_info, stage_name=stage_name, solver=solver_cfg)
        linear_solve_elapsed = max(_perf_counter() - linear_start, 0.0)
        reactions = K @ u - F
    else:
        linear_start = _perf_counter()
        u, linear_info, reduced_cache = _solve_axisymmetric_linear(
            K,
            F,
            constrained,
            stage_name,
            solver_cfg,
            free_dofs=axisymmetric_step_cache.free_dofs if axisymmetric_step_cache is not None else None,
            fixed_dofs=axisymmetric_step_cache.fixed_dofs if axisymmetric_step_cache is not None else None,
            reduction_cache=axisymmetric_step_cache.reduced_matrix_cache if axisymmetric_step_cache is not None else None,
            validate_cache=axisymmetric_step_cache.reduced_matrix_cache is None if axisymmetric_step_cache is not None else True,
        )
        linear_solve_elapsed = max(_perf_counter() - linear_start, 0.0)
        residual = K @ u - F
        solver_info = {
            "method": "axisymmetric_static",
            "linear_method": str(linear_info.get("method", "direct")),
            "linear_method_requested": str(linear_info.get("method_requested", linear_info.get("method", "direct"))),
            "iterations": int(linear_info.get("iterations", 1) or 0),
            "residual_norm": float(np.linalg.norm(residual)),
            "equilibrated": bool(linear_info.get("equilibrated", False)),
            "linear_solver": dict(linear_info),
            "reduced_matrix_cache": {"enabled": False} if reduced_cache is None else {**reduced_cache.info(), **dict(linear_info.get("reduced_matrix_cache", {}))},
        }
        reactions = K @ u - F
    post_start = _perf_counter()
    postprocess_state_info: dict[str, Any] = {}
    element_results, updated_plastic_state = compute_axisymmetric_element_results_and_state(
        mesh,
        materials,
        u,
        strength_factor=strength_factor,
        plastic_state=plastic_state,
        initial_stresses=initial_stresses,
        collect_results=postprocess_results,
        postprocess_info=postprocess_state_info,
    )
    postprocess_elapsed += max(_perf_counter() - post_start, 0.0)
    active_elements = [element.id for element in mesh.elements if element.active]
    solver_info["geometry"] = "axisymmetric"
    solver_info["geometry_mode"] = "axisymmetric"
    solver_info["element_type"] = ",".join(sorted({str(element.type).upper() for element in mesh.elements if element.active}))
    solver_info["integration"] = ",".join(sorted({normalize_integration(element.integration) for element in mesh.elements if element.active}))
    solver_info["material_model"] = ",".join(sorted({str(materials[element.material].model) for element in mesh.elements if element.active and element.material in materials}))
    solver_info["postprocess_results"] = bool(postprocess_results)
    solver_info["postprocess_state_commit"] = postprocess_state_info
    solver_info["plastic_ratio"] = _plastic_ratio_from_state(updated_plastic_state, active_elements)
    solver_info["plastic_ratio_source"] = "plastic_state"
    if axisymmetric_step_cache is not None:
        solver_info["topology_cache"] = {**axisymmetric_step_cache.solver_info(), "auto_generated": auto_step_cache}
    elif static_step_cache_info is not None:
        solver_info["topology_cache"] = static_step_cache_info
    solver_info["batched_elements"] = int(solver_info.get("topology_cache", {}).get("batched_axisymmetric_elastic_elements", 0) or 0)
    solver_info["axisymmetric_linear_static_cache"] = {
        "enabled": axisymmetric_step_cache is not None,
        "stiffness_cache_used": bool(stiffness_cache_used),
        "linear_path": not nonlinear,
    }
    assembly_elapsed = (
        stiffness_assembly_elapsed
        + load_assembly_elapsed
        + constraint_collection_elapsed
        + mpc_assembly_elapsed
        + mpc_penalty_apply_elapsed
    )
    perf = solver_info.setdefault("performance", {})
    if not isinstance(perf, dict):
        perf = {}
        solver_info["performance"] = perf
    perf.update(
        {
            "assembly_elapsed_seconds": assembly_elapsed,
            "stiffness_assembly_elapsed_seconds": stiffness_assembly_elapsed,
            "load_assembly_elapsed_seconds": load_assembly_elapsed,
            "constraint_collection_elapsed_seconds": constraint_collection_elapsed,
            "mpc_assembly_elapsed_seconds": mpc_assembly_elapsed,
            "mpc_penalty_apply_elapsed_seconds": mpc_penalty_apply_elapsed,
            "linear_solve_elapsed_seconds": linear_solve_elapsed,
            "nonlinear_solve_elapsed_seconds": nonlinear_solve_elapsed,
            "postprocess_elapsed_seconds": postprocess_elapsed,
            "cache_build_elapsed_seconds": cache_build_elapsed,
            "cache_reuse_elapsed_seconds": stiffness_assembly_elapsed if stiffness_cache_used else 0.0,
            "coupled_assembly_elapsed_seconds": 0.0,
        }
    )
    _attach_matrix_profile(solver_info, K, F, constrained, label="axisymmetric_global_stiffness")
    result = StageResult2D(stage_name, u, reactions, element_results, constrained, active_elements, solver_info)
    result.plastic_state = updated_plastic_state
    if postprocess_results:
        post_extra_start = _perf_counter()
        result.interface_results = compute_interface_results(mesh, interfaces, u, axisymmetric=True)
        result.structural_results = compute_structural_results(mesh, materials, structural_elements, u, axisymmetric=True, loads=loads)
        _attach_integration_point_results(
            result,
            mesh,
            materials,
            u,
            axisymmetric=True,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            initial_stresses=initial_stresses,
        )
        postprocess_elapsed += max(_perf_counter() - post_extra_start, 0.0)
        perf["postprocess_elapsed_seconds"] = postprocess_elapsed
    if mpc_plan.active:
        applied_method = mpc_plan.applied_method
        multiplier_info = {"multipliers": solver_info.get("multipliers", [])} if "multipliers" in solver_info else {}
        result.solver_info["mpc"] = {**mpc_info, **multiplier_info, "max_violation": mpc_violation(mesh, u, mpc_info), "applied_method": applied_method}
    if output_dir is not None:
        io_start = _perf_counter()
        result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
        io_report_elapsed = max(_perf_counter() - io_start, 0.0)
        perf["io_report_elapsed_seconds"] = io_report_elapsed
    return result


def _run_srm_trial_search(
    factors: list[float],
    srm_cfg: Mapping[str, Any],
    failure_plastic_ratio: float,
    solve_trial: Callable[[float], StageResult2D],
    *,
    progress_stage_name: str = "",
    mesh: Mesh2D | None = None,
    warm_start_supported: bool = True,
    parallel_trials_supported: bool = True,
    parallel_disabled_reason: str = "",
    process_trial_spec: Mapping[str, Any] | None = None,
) -> tuple[StageResult2D | None, float, list[dict[str, Any]], dict[str, Any]]:
    parallel = _srm_parallel_settings(
        srm_cfg,
        len(factors),
        mesh=mesh,
        parallel_trials_supported=parallel_trials_supported,
        parallel_disabled_reason=parallel_disabled_reason,
    )
    mode = _srm_search_mode(srm_cfg)
    adaptive_mode = mode in {"adaptive", "adaptive_bracket", "bracket_bisection", "coarse_bracket", "auto", "auto_bracket", "diagnostic_auto"}
    explicit_linear = srm_cfg.get("factors") is not None and not adaptive_mode
    stage_label = progress_stage_name or "srm"
    topology_cache = _srm_topology_diagnostics_cache(mesh)
    if explicit_linear:
        result, fos, trials, info = _run_srm_linear_trials(
            factors,
            failure_plastic_ratio,
            solve_trial,
            parallel=parallel,
            progress=srm_cfg,
            progress_stage_name=stage_label,
            topology_cache=topology_cache,
        )
        info["search_mode"] = "explicit_factors"
        info.setdefault(
            "parallel",
            {**dict(parallel), "evaluated_trials": len(trials), "reported_trials": len(trials), "strategy": "sequential"},
        )
        _srm_annotate_trial_deltas(trials)
        _srm_attach_material_fallback_search_info(info, trials)
        return result, fos, trials, info
    if adaptive_mode:
        result, fos, trials, info = _run_srm_adaptive_bracket_trials(
            factors,
            srm_cfg,
            failure_plastic_ratio,
            solve_trial,
            parallel=parallel,
            progress=srm_cfg,
            progress_stage_name=stage_label,
            topology_cache=topology_cache,
            warm_start_supported=warm_start_supported,
            process_trial_spec=process_trial_spec,
        )
        info.setdefault(
            "parallel",
            {**dict(parallel), "evaluated_trials": len(trials), "reported_trials": len(trials), "strategy": "sequential"},
        )
        _srm_annotate_trial_deltas(trials)
        _srm_attach_material_fallback_search_info(info, trials)
        return result, fos, trials, info
    if mode not in {"two_branch", "two_branch_bisection", "bisection"}:
        result, fos, trials, info = _run_srm_linear_trials(
            factors,
            failure_plastic_ratio,
            solve_trial,
            parallel=parallel,
            progress=srm_cfg,
            progress_stage_name=stage_label,
            topology_cache=topology_cache,
        )
        info.setdefault(
            "parallel",
            {**dict(parallel), "evaluated_trials": len(trials), "reported_trials": len(trials), "strategy": "sequential"},
        )
        _srm_annotate_trial_deltas(trials)
        _srm_attach_material_fallback_search_info(info, trials)
        return result, fos, trials, info
    result, fos, trials, info = _run_srm_two_branch_trials(
        factors,
        srm_cfg,
        failure_plastic_ratio,
        solve_trial,
        parallel=parallel,
        progress=srm_cfg,
        progress_stage_name=stage_label,
        topology_cache=topology_cache,
    )
    info.setdefault(
        "parallel",
        {**dict(parallel), "evaluated_trials": len(trials), "reported_trials": len(trials), "strategy": "sequential"},
    )
    _srm_annotate_trial_deltas(trials)
    _srm_attach_material_fallback_search_info(info, trials)
    return result, fos, trials, info


def _srm_attach_material_fallback_search_info(info: dict[str, Any], trials: list[dict[str, Any]]) -> None:
    verified_relative_tolerance = 1.0e-7
    regularized_count = sum(max(int(row.get("mc_regularized_projection_count", 0) or 0), 0) for row in trials)
    numba_fallback_count = sum(max(int(row.get("mc_numba_to_python_fallback_count", 0) or 0), 0) for row in trials)
    numba_regularized_count = sum(
        max(int(row.get("mc_numba_regularized_projection_count", 0) or 0), 0)
        for row in trials
    )
    apex_regularization_count = sum(
        max(int(row.get("mc_apex_regularization_count", 0) or 0), 0)
        for row in trials
    )
    associated_apex_count = sum(
        max(int(row.get("mc_associated_apex_projection_count", 0) or 0), 0)
        for row in trials
    )
    legacy_bounded_count = sum(
        max(int(row.get("mc_legacy_bounded_projection_count", 0) or 0), 0)
        for row in trials
    )
    if regularized_count > 0 and associated_apex_count + legacy_bounded_count == 0:
        for row in trials:
            row_count = max(int(row.get("mc_regularized_projection_count", 0) or 0), 0)
            method = str(row.get("mc_regularization_method", "") or "")
            if method == "associated_multisurface_apex":
                associated_apex_count += row_count
            else:
                legacy_bounded_count += row_count
    above_relaxed_count = sum(
        max(int(row.get("mc_regularized_projection_above_relaxed_tolerance_count", 0) or 0), 0)
        for row in trials
    )
    active_set_attempt_count = sum(
        max(int(row.get("mc_active_set_update_attempt_count", 0) or 0), 0)
        for row in trials
    )
    active_set_hit_count = sum(
        max(int(row.get("mc_active_set_update_hit_count", 0) or 0), 0)
        for row in trials
    )
    regularized_active_set_hit_count = sum(
        max(int(row.get("mc_active_set_regularized_update_hit_count", 0) or 0), 0)
        for row in trials
    )
    active_set_full_scan_avoided_count = sum(
        max(int(row.get("mc_active_set_full_scan_avoided_count", 0) or 0), 0)
        for row in trials
    )
    max_violation = max(
        (float(row.get("mc_regularized_projection_max_yield_violation", 0.0) or 0.0) for row in trials),
        default=0.0,
    )
    max_relative_violation = max(
        (float(row.get("mc_regularized_projection_max_relative_yield_violation", 0.0) or 0.0) for row in trials),
        default=0.0,
    )
    samples: list[dict[str, Any]] = []
    for row in trials:
        raw_samples = row.get("mc_regularized_projection_samples", [])
        if not isinstance(raw_samples, list):
            continue
        for raw in raw_samples:
            if isinstance(raw, Mapping) and len(samples) < 32:
                samples.append(dict(raw))
    yield_surface_verified = bool(
        above_relaxed_count == 0
        and math.isfinite(max_relative_violation)
        and max_relative_violation <= verified_relative_tolerance
    )
    configured_policy_verified = bool(
        yield_surface_verified
        and legacy_bounded_count == 0
        and associated_apex_count == regularized_count
    )
    regularization_method = (
        "mixed_apex_projection"
        if associated_apex_count and legacy_bounded_count
        else (
            "bounded_sequential_cone_tip"
            if legacy_bounded_count
            else "associated_multisurface_apex"
        )
    )
    info["mohr_coulomb_fallback"] = {
        "numba_to_python_count": numba_fallback_count,
        "numba_regularized_projection_count": numba_regularized_count,
        "regularized_projection_count": regularized_count,
        "apex_regularization_count": apex_regularization_count,
        "associated_apex_projection_count": associated_apex_count,
        "legacy_bounded_projection_count": legacy_bounded_count,
        "regularized_projection_above_relaxed_tolerance_count": above_relaxed_count,
        "active_set_update_attempt_count": active_set_attempt_count,
        "active_set_update_hit_count": active_set_hit_count,
        "active_set_regularized_update_hit_count": regularized_active_set_hit_count,
        "active_set_full_scan_avoided_count": active_set_full_scan_avoided_count,
        "active_set_update_hit_ratio": (
            float(active_set_hit_count) / float(active_set_attempt_count)
            if active_set_attempt_count > 0
            else 0.0
        ),
        "max_yield_violation": max_violation,
        "max_relative_yield_violation": max_relative_violation,
        "verified_relative_yield_tolerance": verified_relative_tolerance,
        "regularization_method": regularization_method,
        "regularization_quality": (
            "yield_surface_verified"
            if (
                above_relaxed_count == 0
                and max_relative_violation <= verified_relative_tolerance
            )
            else "yield_surface_tolerance_exceeded"
        ),
        "samples": samples,
        "affects_nonassociated_flow_rule": associated_apex_count > 0 or legacy_bounded_count > 0,
        "configured_apex_policy_verified": configured_policy_verified,
        "constitutive_model_fidelity": configured_policy_verified or regularized_count <= 0,
        "flow_rule_verified": configured_policy_verified or regularized_count <= 0,
        "base_nonassociated_flow_rule_verified": regularized_count <= 0,
    }
    if regularized_count <= 0:
        return
    info["material_fallback_within_tolerance"] = yield_surface_verified
    info["material_fallback_verification_required"] = not configured_policy_verified
    info["constitutive_regularization"] = {
        "enabled": True,
        "method": regularization_method,
        "reason": "nonassociated_tension_side_apex_without_exact_return",
        "yield_surface_verified": yield_surface_verified,
        "flow_rule_verified": configured_policy_verified,
        "base_nonassociated_flow_rule_verified": False,
        "constitutive_model_fidelity": configured_policy_verified,
        "max_relative_yield_violation": max_relative_violation,
        "relative_yield_tolerance": verified_relative_tolerance,
    }
    if configured_policy_verified:
        info["factor_of_safety_certification_scope"] = (
            "mohr_coulomb_with_associated_multisurface_apex_policy"
        )
        info["mohr_coulomb_fallback"]["certification_effect"] = (
            "certified_for_configured_apex_policy"
        )
        return
    info["factor_of_safety_certification_scope"] = "unverified_apex_projection_points"
    info["mohr_coulomb_fallback"]["certification_effect"] = "boundary_verification_required"
    current_status = str(info.get("factor_of_safety_status", "") or "")
    if current_status != "nonmonotonic_evidence":
        info["factor_of_safety_status"] = "material_fallback_evidence"
        info["factor_of_safety_confidence"] = "limited"
        info["bracketed"] = False
        info["factor_of_safety_boundary_certified"] = False
        info["factor_of_safety_certified"] = False
        info["factor_of_safety_value_kind"] = "verification_required"
        info["boundary_quality"] = "material_fallback_verification_required"


def _srm_search_mode(srm_cfg: Mapping[str, Any]) -> str:
    raw_mode = str(srm_cfg.get("search_mode", srm_cfg.get("mode", "linear")) or "linear").strip().lower().replace("-", "_")
    adaptive_requested = bool(srm_cfg.get("adaptive", srm_cfg.get("use_bisection", False)))
    raw_auto = srm_cfg.get("auto", False)
    auto_requested = (
        (_srm_bool(raw_auto.get("enabled", True), True) if isinstance(raw_auto, Mapping) else _srm_bool(raw_auto, False))
        and raw_mode in {"linear", "adaptive", "adaptive_bracket", "two_branch", "bisection"}
    )
    if auto_requested:
        return "auto"
    if adaptive_requested and raw_mode in {"linear", "explicit", "explicit_factors", "two_branch", "two_branch_bisection", "bisection"}:
        return "adaptive_bracket"
    return raw_mode


def _srm_parallel_settings(
    srm_cfg: Mapping[str, Any],
    trial_count: int,
    *,
    mesh: Mesh2D | None = None,
    parallel_trials_supported: bool = True,
    parallel_disabled_reason: str = "",
) -> dict[str, Any]:
    raw_parallel = srm_cfg.get("parallel", srm_cfg.get("parallel_trials", srm_cfg.get("parallelization")))
    raw_workers = srm_cfg.get("max_workers", srm_cfg.get("workers", srm_cfg.get("parallel_workers")))
    parallel_cfg = raw_parallel if isinstance(raw_parallel, Mapping) else {}
    runtime_cfg = srm_cfg.get("_runtime", srm_cfg.get("_execution", {}))
    if not isinstance(runtime_cfg, Mapping):
        runtime_cfg = {}
    if isinstance(raw_parallel, Mapping):
        enabled = bool(raw_parallel.get("enabled", raw_parallel.get("parallel", True)))
        raw_workers = raw_parallel.get("max_workers", raw_parallel.get("workers", raw_workers))
        raw_workers = raw_parallel.get("auto_workers", raw_workers) if str(raw_workers).lower() == "auto" else raw_workers
    elif raw_parallel is None:
        enabled = raw_workers is not None
    else:
        enabled = bool(raw_parallel)
    logical_cpu_count = max(int(os.cpu_count() or 1), 1)
    physical_cpu_count = _srm_physical_cpu_count()
    worker_cpu_basis = max(int(physical_cpu_count or logical_cpu_count), 1)
    context = _srm_parallel_context(parallel_cfg, runtime_cfg)
    policy = _srm_parallel_policy(parallel_cfg, runtime_cfg, context)
    strategy = _srm_parallel_strategy(parallel_cfg, runtime_cfg)
    executor_kind = str(
        parallel_cfg.get("executor", parallel_cfg.get("backend", "thread")) or "thread"
    ).strip().lower().replace("-", "_")
    if executor_kind in {"processes", "multiprocess", "multiprocessing", "spawn"}:
        executor_kind = "process"
    if executor_kind in {"threads", "threadpool"}:
        executor_kind = "thread"
    if executor_kind not in {"thread", "process"}:
        raise FEM2DError("srm parallel executor must be thread or process")
    batch_like = policy in {"batch", "cli", "auto", "headless"} or context in {"cli", "batch", "headless"}
    if raw_parallel is None and raw_workers is None and batch_like:
        enabled = True
    reserve_default = 1 if batch_like else _srm_parallel_int(parallel_cfg, runtime_cfg, ("gui_reserve_workers", "gui_cpu_reserve"), default=1)
    reserve_workers = _srm_parallel_int(parallel_cfg, runtime_cfg, ("reserve_workers", "cpu_reserve", "reserved_workers"), default=reserve_default)
    reserve_workers = max(0, min(reserve_workers, max(worker_cpu_basis - 1, 0)))
    interactive_cap = _srm_parallel_int(parallel_cfg, runtime_cfg, ("interactive_cap", "gui_cap"), default=2)
    if batch_like:
        default_workers = max(1, worker_cpu_basis - reserve_workers)
    else:
        default_workers = max(1, min(max(1, interactive_cap), worker_cpu_basis - reserve_workers if worker_cpu_basis > 1 else 1))
    try:
        workers = int(raw_workers) if raw_workers is not None and str(raw_workers).strip().lower() != "auto" else default_workers
    except (TypeError, ValueError):
        workers = default_workers
    requested_workers = max(1, workers)
    available_memory_mb = _srm_available_memory_mb()
    memory_limit_mb = _srm_parallel_float(parallel_cfg, runtime_cfg, ("memory_limit_mb", "memory_budget_mb", "available_memory_mb"), default=None)
    memory_fraction = _srm_parallel_float(parallel_cfg, runtime_cfg, ("memory_fraction", "memory_budget_fraction"), default=0.70 if batch_like else None)
    memory_limit_source = "configured" if memory_limit_mb is not None else "none"
    if memory_limit_mb is None and memory_fraction is not None and available_memory_mb is not None:
        memory_limit_mb = max(0.0, float(available_memory_mb) * max(0.0, min(float(memory_fraction or 0.0), 1.0)))
        memory_limit_source = "available_memory_fraction"
    mesh_stats = _srm_parallel_mesh_stats(mesh)
    memory_per_worker_mb = _srm_parallel_float(parallel_cfg, runtime_cfg, ("memory_per_worker_mb", "worker_memory_mb", "estimated_worker_memory_mb"), default=None)
    memory_per_worker_source = "configured" if memory_per_worker_mb is not None else "none"
    if memory_per_worker_mb is None:
        memory_per_worker_mb = _srm_estimated_trial_memory_mb(mesh_stats)
        memory_per_worker_source = "mesh_heuristic" if memory_per_worker_mb is not None else "none"
    memory_limited = False
    memory_worker_cap: int | None = None
    if memory_limit_mb is not None and memory_per_worker_mb is not None and memory_per_worker_mb > 0.0:
        memory_worker_cap = max(1, int(math.floor(float(memory_limit_mb) / float(memory_per_worker_mb))))
        if requested_workers > memory_worker_cap:
            memory_limited = True
            workers = memory_worker_cap
    workers = max(1, min(workers, max(int(trial_count), 1)))
    parallel_requested = bool(enabled and workers > 1 and trial_count > 1)
    thread_safety_guard_active = bool(not parallel_trials_supported)
    enabled = bool(parallel_requested and parallel_trials_supported)
    if thread_safety_guard_active:
        workers = 1
    thread_settings = _srm_parallel_thread_settings(
        parallel_cfg,
        runtime_cfg,
        worker_count=workers if enabled else 1,
        logical_cpu_count=logical_cpu_count,
        physical_cpu_count=physical_cpu_count,
        batch_like=batch_like,
    )
    selection_reasons = [
        f"context={context}",
        f"policy={policy}",
        f"strategy={strategy or 'none'}",
        f"executor={executor_kind}",
        f"logical_cpus={logical_cpu_count}",
        f"physical_cpus={physical_cpu_count or 'unknown'}",
        f"cpu_basis={worker_cpu_basis}",
        f"requested_workers={requested_workers}",
        f"selected_workers={workers if enabled else 1}",
    ]
    if memory_limit_mb is not None:
        selection_reasons.append(f"memory_limit_mb={float(memory_limit_mb):.3g}:{memory_limit_source}")
    if memory_per_worker_mb is not None:
        selection_reasons.append(f"memory_per_worker_mb={float(memory_per_worker_mb):.3g}:{memory_per_worker_source}")
    if memory_limited:
        selection_reasons.append(f"memory_limited_cap={memory_worker_cap}")
    if thread_safety_guard_active:
        selection_reasons.append(
            f"parallel_disabled={parallel_disabled_reason or 'trial_callable_is_not_thread_safe'}"
        )
    return {
        **dict(parallel_cfg),
        "enabled": enabled,
        "max_workers": workers if enabled else 1,
        "parallel_requested": parallel_requested,
        "thread_safety_guard_active": thread_safety_guard_active,
        "disabled_reason": (
            parallel_disabled_reason or "trial_callable_is_not_thread_safe"
            if thread_safety_guard_active
            else ""
        ),
        "executor": executor_kind,
        "context": context,
        "policy": policy,
        "strategy": strategy,
        "cpu_count": logical_cpu_count,
        "logical_cpu_count": logical_cpu_count,
        "physical_cpu_count": physical_cpu_count,
        "available_memory_mb": None if available_memory_mb is None else float(available_memory_mb),
        "reserve_workers": reserve_workers,
        "interactive_cap": interactive_cap,
        "requested_workers": requested_workers,
        "memory_limit_mb": None if memory_limit_mb is None else float(memory_limit_mb),
        "memory_fraction": memory_fraction,
        "memory_limit_source": memory_limit_source,
        "memory_per_worker_mb": None if memory_per_worker_mb is None else float(memory_per_worker_mb),
        "memory_per_worker_source": memory_per_worker_source,
        "memory_worker_cap": memory_worker_cap,
        "memory_limited": memory_limited,
        "mesh": mesh_stats,
        "selected_threads_per_worker": thread_settings["threads_per_worker"],
        "threads_per_worker": thread_settings["threads_per_worker"],
        "numeric_thread_env": thread_settings["numeric_thread_env"],
        "thread_control": thread_settings,
        "environment": {
            "logical_cpu_count": logical_cpu_count,
            "physical_cpu_count": physical_cpu_count,
            "available_memory_mb": None if available_memory_mb is None else float(available_memory_mb),
            "context": context,
            "policy": policy,
            "strategy": strategy,
        },
        "selection_reasons": selection_reasons,
        "worker_limit_policy": (
            "batch/CLI SRM uses CPU and optional memory budget to raise worker count; "
            "GUI/interactive SRM keeps a low cap unless srm.parallel overrides it"
        ),
    }


def _srm_parallel_context(parallel_cfg: Mapping[str, Any], runtime_cfg: Mapping[str, Any]) -> str:
    value = parallel_cfg.get("context", runtime_cfg.get("context", runtime_cfg.get("execution_context", "")))
    text = str(value or "").strip().lower().replace("-", "_")
    return text or "interactive"


def _srm_parallel_policy(parallel_cfg: Mapping[str, Any], runtime_cfg: Mapping[str, Any], context: str) -> str:
    value = parallel_cfg.get("policy", parallel_cfg.get("profile", runtime_cfg.get("srm_parallel_policy", runtime_cfg.get("profile", ""))))
    text = str(value or "").strip().lower().replace("-", "_")
    if text:
        return text
    if context in {"cli", "batch", "headless"}:
        return "batch"
    return "interactive"


def _srm_parallel_strategy(parallel_cfg: Mapping[str, Any], runtime_cfg: Mapping[str, Any]) -> str:
    value = parallel_cfg.get("strategy", parallel_cfg.get("mode", runtime_cfg.get("srm_parallel_strategy", "")))
    return str(value or "").strip().lower().replace("-", "_")


def _srm_physical_cpu_count() -> int | None:
    try:
        import psutil  # type: ignore

        count = psutil.cpu_count(logical=False)
        if count:
            return max(int(count), 1)
    except Exception:
        return None
    return None


def _srm_parallel_mesh_stats(mesh: Mesh2D | None) -> dict[str, Any] | None:
    if mesh is None:
        return None
    active_elements = [element for element in mesh.elements if element.active]
    element_count = len(mesh.elements)
    active_element_count = len(active_elements)
    node_count = len(mesh.node_ids)
    max_nodes_per_element = max((len(element.nodes) for element in active_elements), default=0)
    return {
        "node_count": node_count,
        "element_count": element_count,
        "active_element_count": active_element_count,
        "dof_count": node_count * 2,
        "max_nodes_per_element": max_nodes_per_element,
    }


def _srm_estimated_trial_memory_mb(mesh_stats: Mapping[str, Any] | None) -> float | None:
    if not isinstance(mesh_stats, Mapping):
        return None
    dof_count = float(mesh_stats.get("dof_count", 0) or 0)
    active_element_count = float(mesh_stats.get("active_element_count", 0) or 0)
    element_count = float(mesh_stats.get("element_count", 0) or 0)
    if dof_count <= 0.0 and active_element_count <= 0.0:
        return None
    # Conservative per-trial working-set estimate for sparse solve, tangent blocks,
    # plastic arrays, and post-trial diagnostics. Users can override it explicitly.
    estimated = 96.0 + dof_count * 0.0015 + active_element_count * 0.006 + element_count * 0.002
    return float(max(256.0, estimated))


def _srm_parallel_thread_settings(
    parallel_cfg: Mapping[str, Any],
    runtime_cfg: Mapping[str, Any],
    *,
    worker_count: int,
    logical_cpu_count: int,
    physical_cpu_count: int | None,
    batch_like: bool,
) -> dict[str, Any]:
    raw_threads: Any = None
    for key in ("threads_per_worker", "worker_threads", "num_threads", "numeric_threads", "blas_threads"):
        raw_threads = parallel_cfg.get(key, runtime_cfg.get(key))
        if raw_threads is not None:
            break
    worker_count = max(int(worker_count or 1), 1)
    logical_cpu_count = max(int(logical_cpu_count or 1), 1)
    source = "auto"
    try:
        text = str(raw_threads).strip().lower() if raw_threads is not None else "auto"
        if raw_threads is not None and text != "auto":
            threads = max(1, int(raw_threads))
            source = "configured"
        elif worker_count > 1:
            threads = max(1, min(2, logical_cpu_count // worker_count))
        else:
            basis = int(physical_cpu_count or logical_cpu_count)
            threads = max(1, min(4 if batch_like else 2, basis))
    except (TypeError, ValueError):
        threads = 1
        source = "fallback"
    oversubscribed = worker_count * threads > logical_cpu_count
    if oversubscribed:
        threads = max(1, logical_cpu_count // worker_count)
    env = {
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        "OPENBLAS_NUM_THREADS": str(threads),
        "NUMBA_NUM_THREADS": str(threads),
    }
    return {
        "threads_per_worker": int(threads),
        "source": source,
        "oversubscription_guard_applied": bool(oversubscribed),
        "numeric_thread_env": env,
        "applied": False,
        "apply_method": "",
        "apply_note": "threadpoolctl applies this limit around parallel SRM trials when available; otherwise environment variables are scoped around the prefetch window and restored",
    }


@contextmanager
def _srm_numeric_thread_context(thread_control: Mapping[str, Any] | None):
    if not isinstance(thread_control, Mapping):
        yield
        return
    try:
        threads = int(thread_control.get("threads_per_worker", 0) or 0)
    except (TypeError, ValueError):
        threads = 0
    mutable_control = thread_control if isinstance(thread_control, dict) else None
    if threads <= 0:
        if mutable_control is not None:
            mutable_control["applied"] = False
            mutable_control["apply_method"] = "disabled"
            mutable_control["apply_error"] = "threads_per_worker is not positive"
        yield
        return
    try:
        from threadpoolctl import threadpool_limits  # type: ignore
    except Exception as exc:
        raw_env = thread_control.get("numeric_thread_env", {})
        env = raw_env if isinstance(raw_env, Mapping) else {}
        if not env:
            env = {
                "OMP_NUM_THREADS": str(threads),
                "MKL_NUM_THREADS": str(threads),
                "OPENBLAS_NUM_THREADS": str(threads),
                "NUMBA_NUM_THREADS": str(threads),
            }
        old_env = {str(key): os.environ.get(str(key)) for key in env}
        if mutable_control is not None:
            mutable_control["applied"] = True
            mutable_control["apply_method"] = "environment"
            mutable_control["threadpoolctl_available"] = False
            mutable_control["threadpoolctl_error"] = str(exc)
            mutable_control["environment_restored"] = False
            mutable_control.pop("apply_error", None)
        try:
            for key, value in env.items():
                os.environ[str(key)] = str(value)
            yield
        finally:
            for key, old_value in old_env.items():
                if old_value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old_value
            if mutable_control is not None:
                mutable_control["environment_restored"] = True
        return
    if mutable_control is not None:
        mutable_control["applied"] = True
        mutable_control["apply_method"] = "threadpoolctl"
        mutable_control["threadpoolctl_available"] = True
        mutable_control["environment_restored"] = ""
        mutable_control.pop("apply_error", None)
        mutable_control.pop("threadpoolctl_error", None)
    with threadpool_limits(limits=threads):
        yield


def _srm_parallel_int(parallel_cfg: Mapping[str, Any], runtime_cfg: Mapping[str, Any], keys: tuple[str, ...], *, default: int) -> int:
    for key in keys:
        value = parallel_cfg.get(key, runtime_cfg.get(key))
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return int(default)


def _srm_parallel_float(parallel_cfg: Mapping[str, Any], runtime_cfg: Mapping[str, Any], keys: tuple[str, ...], *, default: float | None) -> float | None:
    for key in keys:
        value = parallel_cfg.get(key, runtime_cfg.get(key))
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def _srm_available_memory_mb() -> float | None:
    if os.name == "nt":
        try:
            import ctypes

            class _MemoryStatusEx(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemoryStatusEx()
            status.dwLength = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return float(status.ullAvailPhys) / (1024.0 * 1024.0)
        except Exception:
            return None
    try:
        meminfo = Path("/proc/meminfo")
        if meminfo.exists():
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return float(parts[1]) / 1024.0
    except Exception:
        return None
    return None


@dataclass(frozen=True)
class _SRMTopologyDiagnosticsCache:
    active_element_ids: tuple[str, ...]
    adjacency: dict[str, tuple[str, ...]]
    boundary_masks: dict[str, int]
    min_det_j: float


def _srm_topology_diagnostics_cache(mesh: Mesh2D | None) -> _SRMTopologyDiagnosticsCache | None:
    if mesh is None:
        return None
    active_elements = [element for element in mesh.elements if element.active]
    active_ids = tuple(str(element.id) for element in active_elements)
    node_to_elements: dict[str, list[str]] = {}
    for element in active_elements:
        for node_id in element.nodes:
            node_to_elements.setdefault(str(node_id), []).append(str(element.id))
    adjacency_sets: dict[str, set[str]] = {element_id: set() for element_id in active_ids}
    for element_ids in node_to_elements.values():
        if len(element_ids) <= 1:
            continue
        for element_id in element_ids:
            adjacency_sets[element_id].update(other for other in element_ids if other != element_id)
    coords = np.asarray(mesh.coords, dtype=float)
    boundary_masks: dict[str, int] = {}
    if coords.size:
        min_x = float(np.min(coords[:, 0]))
        max_x = float(np.max(coords[:, 0]))
        min_y = float(np.min(coords[:, 1]))
        max_y = float(np.max(coords[:, 1]))
        span = max(max_x - min_x, max_y - min_y, 1.0)
        tol = max(span * 1.0e-9, 1.0e-12)
        node_index = mesh.node_index
        for element in active_elements:
            mask = 0
            for node_id in element.nodes:
                idx = node_index.get(str(node_id))
                if idx is None:
                    continue
                x = float(coords[idx, 0])
                y = float(coords[idx, 1])
                if abs(x - min_x) <= tol:
                    mask |= 1
                if abs(x - max_x) <= tol:
                    mask |= 2
                if abs(y - min_y) <= tol:
                    mask |= 4
                if abs(y - max_y) <= tol:
                    mask |= 8
            boundary_masks[str(element.id)] = mask
    try:
        min_det_j = float(_minimum_element_det_j(mesh))
    except Exception:
        min_det_j = math.nan
    return _SRMTopologyDiagnosticsCache(
        active_element_ids=active_ids,
        adjacency={element_id: tuple(sorted(neighbors)) for element_id, neighbors in adjacency_sets.items()},
        boundary_masks=boundary_masks,
        min_det_j=min_det_j,
    )


def _equivalent_plastic_strain_components(plastic_strain: Any, kappa: Any = 0.0, state_vars: Mapping[str, Any] | None = None) -> float:
    candidates: list[float] = []
    try:
        kappa_value = float(kappa)
        if math.isfinite(kappa_value):
            candidates.append(max(kappa_value, 0.0))
    except (TypeError, ValueError):
        pass
    try:
        strain = np.asarray(plastic_strain, dtype=float).reshape(-1)
        if strain.size >= 4:
            ex, ey, ez, gxy = (float(strain[0]), float(strain[1]), float(strain[2]), float(strain[3]))
            mean = (ex + ey + ez) / 3.0
            sx = ex - mean
            sy = ey - mean
            sz = ez - mean
            shear = 0.5 * gxy
            eq = math.sqrt(max((2.0 / 3.0) * (sx * sx + sy * sy + sz * sz + 2.0 * shear * shear), 0.0))
            if math.isfinite(eq):
                candidates.append(eq)
    except (TypeError, ValueError):
        pass
    if isinstance(state_vars, Mapping):
        for key in ("equivalent_plastic_strain", "eq_plastic_strain", "plastic_strain_eq", "gamma_eq", "hardening_variable"):
            try:
                value = float(state_vars.get(key, 0.0))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                candidates.append(max(value, 0.0))
    return max(candidates) if candidates else 0.0


def _equivalent_plastic_strain_array(strains: np.ndarray, kappas: np.ndarray) -> np.ndarray:
    if strains.size == 0:
        return np.zeros(0, dtype=float)
    strain = np.asarray(strains, dtype=float)
    kappa = np.asarray(kappas, dtype=float).reshape(-1)
    if strain.ndim != 2 or strain.shape[1] < 4:
        return np.maximum(kappa, 0.0)
    mean = np.mean(strain[:, :3], axis=1)
    sx = strain[:, 0] - mean
    sy = strain[:, 1] - mean
    sz = strain[:, 2] - mean
    shear = 0.5 * strain[:, 3]
    eq = np.sqrt(np.maximum((2.0 / 3.0) * (sx * sx + sy * sy + sz * sz + 2.0 * shear * shear), 0.0))
    if kappa.size == eq.size:
        eq = np.maximum(eq, np.maximum(kappa, 0.0))
    return eq


def _srm_plastic_values_and_yielded_elements(
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    active_elements: list[str] | tuple[str, ...] | set[str],
) -> tuple[np.ndarray, set[str]]:
    active = {str(element_id) for element_id in active_elements}
    if not active:
        return np.zeros(0, dtype=float), set()
    yielded: set[str] = set()
    values: list[np.ndarray] = []
    if plastic_state_cache is not None and plastic_state_cache.plastic_strains.ndim == 3:
        for element_id in active:
            row = plastic_state_cache.element_row.get(element_id)
            if row is None:
                continue
            count = min(int(plastic_state_cache.state_point_counts[row]), plastic_state_cache.plastic_strains.shape[1])
            if count <= 0:
                continue
            eq = _equivalent_plastic_strain_array(
                plastic_state_cache.plastic_strains[row, :count, :],
                plastic_state_cache.kappas[row, :count],
            )
            if plastic_state_cache.state_objects.size:
                for gp_index, stored in enumerate(plastic_state_cache.state_objects[row, :count]):
                    if isinstance(stored, PlasticState2D):
                        eq[gp_index] = max(eq[gp_index], _equivalent_plastic_strain_components(stored.plastic_strain, stored.kappa, stored.state_vars))
            if bool(np.any(eq > 0.0)):
                yielded.add(element_id)
            values.append(eq)
        if values:
            return np.concatenate(values), yielded
    if plastic_state:
        scalar_values: list[float] = []
        for raw_key, state in plastic_state.items():
            element_id = str(raw_key).rsplit(":", 1)[0]
            if element_id not in active:
                continue
            value = _equivalent_plastic_strain_components(getattr(state, "plastic_strain", None), getattr(state, "kappa", 0.0), getattr(state, "state_vars", None))
            scalar_values.append(value)
            if value > 0.0 or _plastic_state_is_plastic(state):
                yielded.add(element_id)
        return np.asarray(scalar_values, dtype=float), yielded
    return np.zeros(0, dtype=float), yielded


def _srm_cluster_metrics(yielded_elements: set[str], topology_cache: _SRMTopologyDiagnosticsCache | None) -> dict[str, Any]:
    if topology_cache is None or not yielded_elements:
        return {
            "connected_plastic_cluster_size": 0,
            "plastic_cluster_spans_boundary": False,
            "plastic_cluster_boundary_side_count": 0,
            "plastic_cluster_boundary_sides": [],
            "plastic_cluster_opposite_boundary_span": False,
        }
    yielded = set(yielded_elements)
    visited: set[str] = set()
    largest_size = 0
    largest_mask = 0
    for start in yielded:
        if start in visited:
            continue
        stack = [start]
        visited.add(start)
        size = 0
        mask = 0
        while stack:
            element_id = stack.pop()
            size += 1
            mask |= int(topology_cache.boundary_masks.get(element_id, 0) or 0)
            for neighbor in topology_cache.adjacency.get(element_id, ()):
                if neighbor in yielded and neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        if size > largest_size:
            largest_size = size
            largest_mask = mask
    boundary_side_count = int(largest_mask.bit_count())
    side_names = [name for bit, name in ((1, "left"), (2, "right"), (4, "bottom"), (8, "top")) if largest_mask & bit]
    opposite_span = bool((largest_mask & 1 and largest_mask & 2) or (largest_mask & 4 and largest_mask & 8))
    return {
        "connected_plastic_cluster_size": int(largest_size),
        "plastic_cluster_spans_boundary": boundary_side_count >= 2,
        "plastic_cluster_boundary_side_count": boundary_side_count,
        "plastic_cluster_boundary_sides": side_names,
        "plastic_cluster_opposite_boundary_span": opposite_span,
    }


def _srm_plastic_diagnostics(
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    active_elements: list[str] | tuple[str, ...] | set[str],
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
) -> dict[str, Any]:
    active = [str(element_id) for element_id in active_elements]
    values, yielded = _srm_plastic_values_and_yielded_elements(plastic_state, plastic_state_cache, active)
    finite_values = values[np.isfinite(values)] if values.size else np.zeros(0, dtype=float)
    active_count = len(active)
    yielded_count = len(yielded)
    metrics: dict[str, Any] = {
        "yielded_element_count": yielded_count,
        "max_equivalent_plastic_strain": float(np.max(finite_values)) if finite_values.size else 0.0,
        "mean_equivalent_plastic_strain": float(np.mean(finite_values)) if finite_values.size else 0.0,
        "top_percentile_equivalent_plastic_strain": float(np.percentile(finite_values, 95.0)) if finite_values.size else 0.0,
        "top_percentile_equivalent_plastic_strain_percentile": 95.0,
        "plastic_point_count": int(np.count_nonzero(finite_values > 0.0)) if finite_values.size else 0,
        "active_element_count": active_count,
        "plastic_diagnostics_mode": "array_summary_no_postprocess_rows",
    }
    if active_count:
        metrics["plastic_ratio_from_yielded_elements"] = float(yielded_count / active_count)
    if topology_cache is not None:
        metrics["min_det_j"] = topology_cache.min_det_j
    metrics.update(_srm_cluster_metrics(yielded, topology_cache))
    return metrics


def _srm_trial_ratio_for_delta(row: Mapping[str, Any]) -> float | None:
    for key in ("last_accepted_plastic_ratio", "plastic_ratio"):
        try:
            value = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _srm_annotate_trial_deltas(trials: list[dict[str, Any]]) -> None:
    rows: list[tuple[float, dict[str, Any], float]] = []
    for row in trials:
        try:
            factor = float(row.get("factor"))
        except (TypeError, ValueError):
            continue
        ratio = _srm_trial_ratio_for_delta(row)
        if ratio is None:
            continue
        rows.append((factor, row, ratio))
    rows.sort(key=lambda item: item[0])
    previous: tuple[float, float] | None = None
    for factor, row, ratio in rows:
        if previous is None or factor == previous[0]:
            row["plastic_ratio_delta"] = 0.0
            row["plastic_ratio_delta_reference_factor"] = ""
        else:
            row["plastic_ratio_delta"] = float((ratio - previous[1]) / (factor - previous[0]))
            row["plastic_ratio_delta_reference_factor"] = previous[0]
        previous = (factor, ratio)


def _diagnostic_json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _diagnostic_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_diagnostic_json_value(item) for item in value]
    return value


def _normalized_error_diagnostics(exc: FEM2DError) -> dict[str, Any]:
    raw = getattr(exc, "diagnostics", {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): _diagnostic_json_value(value) for key, value in raw.items()}


def _srm_active_set_cache_diagnostics(
    source: Mapping[str, Any],
) -> dict[str, Any]:
    active_set = source.get("mohr_coulomb_active_set_update")
    if not isinstance(active_set, Mapping):
        return {}
    geometry_cache = active_set.get("geometry_cache", {})
    if not isinstance(geometry_cache, Mapping):
        geometry_cache = {}
    return {
        "mc_active_set_policy": str(active_set.get("policy", "")),
        "mc_active_set_tangent_reuse_enabled": active_set.get(
            "tangent_reuse_enabled", ""
        ),
        "mc_active_set_tangent_reuse_disabled_reason": str(
            active_set.get("tangent_reuse_disabled_reason", "")
        ),
        "mc_active_set_direct_consistent_tangent_enabled": active_set.get(
            "direct_consistent_tangent_enabled", ""
        ),
        "mc_active_set_numerical_tangent_switch_count": int(
            active_set.get("numerical_tangent_switch_count", 0) or 0
        ),
        "mc_active_set_numerical_tangent_switch_reason": str(
            active_set.get("numerical_tangent_switch_reason", "")
        ),
        "mc_active_set_tangent_invalidation_count": int(
            active_set.get("tangent_invalidation_count", 0) or 0
        ),
        "mc_active_set_tangent_invalidated_point_count": int(
            active_set.get("tangent_invalidated_point_count", 0) or 0
        ),
        "mc_active_set_consistent_tangent": str(
            active_set.get("consistent_tangent", "")
        ),
        "mc_active_set_cutback_reset_policy": str(
            active_set.get("cutback_reset_policy", "")
        ),
        "mc_active_set_strict_unstable_points_only": active_set.get(
            "strict_unstable_points_only", ""
        ),
        "mc_geometry_cache_enabled": geometry_cache.get("enabled", ""),
        "mc_geometry_cache_scope": str(geometry_cache.get("scope", "")),
        "mc_geometry_cache_block_hits": int(
            geometry_cache.get("block_cache_hits", 0) or 0
        ),
        "mc_geometry_cache_block_misses": int(
            geometry_cache.get("block_cache_misses", 0) or 0
        ),
        "mc_geometry_cache_element_count": int(
            geometry_cache.get("element_count", 0) or 0
        ),
    }


def _srm_residual_reduction_ratio(history: Any) -> float | None:
    if not isinstance(history, list):
        return None
    residuals: list[float] = []
    for raw in history:
        if not isinstance(raw, Mapping):
            continue
        try:
            value = float(raw.get("residual_norm"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            residuals.append(value)
    if len(residuals) < 2 or residuals[0] == 0.0:
        return None
    return float(residuals[-1] / residuals[0])


def _srm_diagnostic_summary(diagnostics: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, label in (
        ("trial_status", "status"),
        ("last_accepted_load_factor", "last_load"),
        ("attempted_load_factor", "attempt"),
        ("last_accepted_plastic_ratio", "last_pr"),
        ("max_equivalent_plastic_strain", "max_eqp"),
        ("connected_plastic_cluster_size", "cluster"),
        ("cutback_count", "cutbacks"),
        ("residual_norm_final", "residual"),
    ):
        value = diagnostics.get(key)
        if value in (None, ""):
            continue
        parts.append(f"{label}={value}")
    return ", ".join(parts)


def _srm_trial_solver_diagnostics(
    trial: StageResult2D,
    plastic_ratio: float,
    ok: bool,
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
) -> dict[str, Any]:
    solver_info = trial.solver_info if isinstance(trial.solver_info, Mapping) else {}
    converged = bool(solver_info.get("converged", True))
    plastic_metrics = _srm_plastic_diagnostics(trial.plastic_state, trial.plastic_state_array_cache, trial.active_elements, topology_cache)
    residual = solver_info.get("residual_norm", "")
    iterations = solver_info.get("iterations", "")
    line_search_reductions = solver_info.get("line_search_reductions", "")
    line_search_batch = solver_info.get("line_search_batch", {})
    if not isinstance(line_search_batch, Mapping):
        line_search_batch = {}
    performance = solver_info.get("performance", {})
    if not isinstance(performance, Mapping):
        performance = {}
    diagnostics: dict[str, Any] = {
        "trial_status": "stable" if ok else _srm_failure_reason(converged, plastic_ratio, None),
        "last_accepted_plastic_ratio": float(plastic_ratio),
        "last_accepted_plastic_point_count": int(_plastic_point_count(trial.plastic_state, trial.plastic_state_array_cache)),
        "last_accepted_max_displacement": float(max_displacement_norm(trial.displacements)) if trial.displacements.size else 0.0,
        "max_displacement_norm": float(max_displacement_norm(trial.displacements)) if trial.displacements.size else 0.0,
        "displacement_increment_norm": float(max_displacement_norm(trial.displacements)) if trial.displacements.size else 0.0,
        "last_accepted_residual_norm": residual,
        "residual_norm_final": residual,
        "last_accepted_iterations": iterations,
        "newton_iterations_total": iterations,
        "newton_iterations_max": iterations,
        "last_accepted_line_search_reductions": line_search_reductions,
        "line_search_reductions_total": line_search_reductions,
        "line_search_batch_calls_total": int(
            line_search_batch.get("batch_calls", 0) or 0
        ),
        "line_search_batch_candidates_total": int(
            line_search_batch.get("candidate_count", 0) or 0
        ),
        "line_search_batch_fallback_count": int(
            line_search_batch.get("fallback_count", 0) or 0
        ),
        "residual_reduction_ratio": _srm_residual_reduction_ratio(solver_info.get("convergence_history")),
        "internal_external_work_ratio": solver_info.get("internal_external_work_ratio", ""),
        "failure_diagnostic_source": "stage_result",
        "assembly_elapsed_seconds": _srm_record_float(
            performance, "assembly_elapsed_seconds", 0.0
        ),
        "linear_solve_elapsed_seconds": _srm_record_float(
            performance, "linear_solve_elapsed_seconds", 0.0
        ),
        "line_search_elapsed_seconds": _srm_record_float(
            performance, "line_search_elapsed_seconds", 0.0
        ),
        "postprocess_elapsed_seconds": _srm_record_float(
            performance, "postprocess_elapsed_seconds", 0.0
        ),
        **plastic_metrics,
    }
    increments = solver_info.get("increments")
    if isinstance(increments, Mapping):
        increment_log = [row for row in increments.get("log", []) if isinstance(row, Mapping)]
        accepted_rows = [row for row in increment_log if bool(row.get("accepted", False))]
        failed_rows = [row for row in increment_log if not bool(row.get("accepted", False))]
        iteration_values = [int(row.get("iterations", 0) or 0) for row in accepted_rows]
        line_search_values = [int(row.get("line_search_reductions", 0) or 0) for row in accepted_rows]
        line_search_batch_calls = [
            int(row.get("line_search_batch_calls", 0) or 0)
            for row in accepted_rows
        ]
        line_search_batch_candidates = [
            int(row.get("line_search_batch_candidates", 0) or 0)
            for row in accepted_rows
        ]
        last_accepted = accepted_rows[-1] if accepted_rows else {}
        last_failed = failed_rows[-1] if failed_rows else {}
        diagnostics.update(
            {
                "last_accepted_load_factor": increments.get("final_factor", 1.0),
                "accepted_increment_count": increments.get("accepted_steps", ""),
                "cutback_count": increments.get("cutbacks", ""),
                "final_step_size": last_accepted.get("step_size", ""),
                "failed_step_size": last_failed.get("step_size", ""),
                "newton_iterations_total": int(sum(iteration_values)) if iteration_values else diagnostics["newton_iterations_total"],
                "newton_iterations_max": int(max(iteration_values)) if iteration_values else diagnostics["newton_iterations_max"],
                "line_search_reductions_total": int(sum(line_search_values)) if line_search_values else diagnostics["line_search_reductions_total"],
                "line_search_batch_calls_total": int(
                    sum(line_search_batch_calls)
                ),
                "line_search_batch_candidates_total": int(
                    sum(line_search_batch_candidates)
                ),
                "displacement_increment_norm": last_accepted.get("displacement_increment_norm", diagnostics["displacement_increment_norm"]),
                "increment_checkpoint_continuation_requested": bool(
                    increments.get("checkpoint_continuation_requested", False)
                ),
                "increment_checkpoint_continuation_used": bool(
                    increments.get("checkpoint_continuation_used", False)
                ),
                "increment_checkpoint_fallback_reason": str(
                    increments.get("checkpoint_fallback_reason", "")
                ),
                "increment_checkpoint_source_load_factor": increments.get(
                    "checkpoint_source_load_factor", ""
                ),
                "increment_checkpoint_resumed_accepted_steps": increments.get(
                    "checkpoint_resumed_accepted_steps", ""
                ),
                "increment_checkpoint_resumed_cutbacks": increments.get(
                    "checkpoint_resumed_cutbacks", ""
                ),
                "increment_checkpoint_reused_history_rows": increments.get(
                    "checkpoint_reused_history_rows", ""
                ),
            }
        )
        for timing_key in (
            "assembly_elapsed_seconds",
            "linear_solve_elapsed_seconds",
            "line_search_elapsed_seconds",
            "postprocess_elapsed_seconds",
        ):
            logged_total = float(
                sum(
                    max(_srm_record_float(row, timing_key, 0.0), 0.0)
                    for row in increment_log
                )
            )
            if logged_total > 0.0:
                diagnostics[timing_key] = logged_total
    elif isinstance(solver_info.get("large_deformation"), Mapping):
        large_info = solver_info["large_deformation"]
        history = [row for row in large_info.get("history", []) if isinstance(row, Mapping)]
        last_row = history[-1] if history else {}
        iteration_values = [int(row.get("iterations", 0) or 0) for row in history]
        line_search_values = [int(row.get("line_search_reductions", 0) or 0) for row in history]
        diagnostics.update(
            {
                "last_accepted_load_factor": last_row.get("load_end", 1.0 if converged else ""),
                "accepted_increment_count": large_info.get("accepted_steps", len(history)),
                "cutback_count": large_info.get("cutbacks", ""),
                "final_step_size": last_row.get("load_increment", ""),
                "newton_iterations_total": int(sum(iteration_values)) if iteration_values else diagnostics["newton_iterations_total"],
                "newton_iterations_max": int(max(iteration_values)) if iteration_values else diagnostics["newton_iterations_max"],
                "line_search_reductions_total": int(sum(line_search_values)) if line_search_values else diagnostics["line_search_reductions_total"],
                "displacement_increment_norm": last_row.get("max_increment_displacement", diagnostics["displacement_increment_norm"]),
                "max_displacement_norm": last_row.get("max_total_displacement", diagnostics["max_displacement_norm"]),
                "min_det_j": last_row.get("min_detJ", diagnostics.get("min_det_j", "")),
                "internal_external_work_ratio": last_row.get("internal_external_work_ratio", diagnostics.get("internal_external_work_ratio", "")),
            }
        )
    else:
        diagnostics["last_accepted_load_factor"] = 1.0 if converged else ""
        diagnostics["final_step_size"] = 1.0 if converged else ""
    diagnostics["diagnostic_summary"] = _srm_diagnostic_summary(diagnostics)
    return diagnostics


def _srm_failed_trial_diagnostics(exc: FEM2DError) -> dict[str, Any]:
    diagnostics = _normalized_error_diagnostics(exc)
    if not diagnostics:
        return {"trial_status": "solver_error", "failure_diagnostic_source": "exception", "diagnostic_summary": "status=solver_error"}
    if not diagnostics.get("trial_status"):
        diagnostics["trial_status"] = "solver_error"
    if not diagnostics.get("failure_diagnostic_source"):
        diagnostics["failure_diagnostic_source"] = "exception"
    diagnostics.update(_srm_active_set_cache_diagnostics(diagnostics))
    diagnostics["diagnostic_summary"] = _srm_diagnostic_summary(diagnostics)
    return diagnostics


def _increment_failure_diagnostics(
    *,
    status: str,
    settings: Mapping[str, Any],
    trial_target: float,
    target: float,
    accepted: int,
    cutbacks: int,
    step_size: float,
    next_step: float,
    log: list[dict[str, Any]],
    state_current: Mapping[str, PlasticState2D] | None,
    state_current_cache: PlasticStateArrayCache | None,
    active_elements: list[str],
    u_current: np.ndarray | None,
    last_result: StageResult2D | None,
    error: FEM2DError,
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
) -> dict[str, Any]:
    plastic_ratio = _plastic_ratio_from_state_or_array(state_current, state_current_cache, active_elements)
    plastic_metrics = _srm_plastic_diagnostics(state_current, state_current_cache, active_elements, topology_cache)
    diagnostics: dict[str, Any] = {
        "trial_status": status,
        "failure_diagnostic_source": "incremental_stage",
        "attempted_load_factor": float(trial_target),
        "last_accepted_load_factor": float(target),
        "accepted_increment_count": int(accepted),
        "cutback_count": int(cutbacks),
        "max_cutbacks": int(settings.get("max_cutbacks", 0) or 0),
        "failed_step_size": float(step_size),
        "next_step_size": float(next_step),
        "min_step": float(settings.get("min_step", 0.0) or 0.0),
        "cutback_factor": float(settings.get("cutback_factor", 0.0) or 0.0),
        "final_step_size": float(step_size),
        "last_accepted_plastic_ratio": float(plastic_ratio),
        "last_accepted_plastic_point_count": int(_plastic_point_count(state_current, state_current_cache)),
        "last_accepted_max_displacement": float(max_displacement_norm(u_current)) if u_current is not None and u_current.size else 0.0,
        "max_displacement_norm": float(max_displacement_norm(u_current)) if u_current is not None and u_current.size else 0.0,
        "displacement_increment_norm": "",
        "failed_increment_error": str(error),
        "increment_log_tail": [dict(row) for row in log[-5:]],
        **plastic_metrics,
    }
    timing_rows = [row for row in log if isinstance(row, Mapping)]
    diagnostics["solver_elapsed_seconds"] = float(
        sum(
            max(_srm_record_float(row, "elapsed_seconds", 0.0), 0.0)
            for row in timing_rows
        )
    )
    for timing_key in (
        "assembly_elapsed_seconds",
        "linear_solve_elapsed_seconds",
        "line_search_elapsed_seconds",
        "postprocess_elapsed_seconds",
    ):
        diagnostics[timing_key] = float(
            sum(
                max(_srm_record_float(row, timing_key, 0.0), 0.0)
                for row in timing_rows
            )
        )
    accepted_log_rows = [
        row
        for row in log
        if isinstance(row, Mapping) and bool(row.get("accepted", False))
    ]
    if accepted_log_rows:
        iteration_values = [
            int(row.get("iterations", 0) or 0) for row in accepted_log_rows
        ]
        line_search_values = [
            int(row.get("line_search_reductions", 0) or 0)
            for row in accepted_log_rows
        ]
        line_search_batch_calls = [
            int(row.get("line_search_batch_calls", 0) or 0)
            for row in accepted_log_rows
        ]
        line_search_batch_candidates = [
            int(row.get("line_search_batch_candidates", 0) or 0)
            for row in accepted_log_rows
        ]
        last_accepted_log = accepted_log_rows[-1]
        diagnostics.update(
            {
                "last_accepted_residual_norm": last_accepted_log.get(
                    "residual_norm", ""
                ),
                "residual_norm_final": last_accepted_log.get(
                    "residual_norm", ""
                ),
                "last_accepted_iterations": last_accepted_log.get(
                    "iterations", ""
                ),
                "newton_iterations_total": int(sum(iteration_values)),
                "newton_iterations_max": int(max(iteration_values)),
                "last_accepted_line_search_reductions": last_accepted_log.get(
                    "line_search_reductions", ""
                ),
                "line_search_reductions_total": int(
                    sum(line_search_values)
                ),
                "line_search_batch_calls_total": int(
                    sum(line_search_batch_calls)
                ),
                "line_search_batch_candidates_total": int(
                    sum(line_search_batch_candidates)
                ),
                "internal_external_work_ratio": last_accepted_log.get(
                    "internal_external_work_ratio", ""
                ),
                "displacement_increment_norm": last_accepted_log.get(
                    "displacement_increment_norm",
                    diagnostics["displacement_increment_norm"],
                ),
            }
        )
    if last_result is not None:
        solver_info = last_result.solver_info if isinstance(last_result.solver_info, Mapping) else {}
        increments = solver_info.get("increments")
        increment_log = [row for row in increments.get("log", []) if isinstance(row, Mapping)] if isinstance(increments, Mapping) else []
        accepted_rows = [row for row in increment_log if bool(row.get("accepted", False))]
        iteration_values = [int(row.get("iterations", 0) or 0) for row in accepted_rows]
        line_search_values = [int(row.get("line_search_reductions", 0) or 0) for row in accepted_rows]
        last_accepted = accepted_rows[-1] if accepted_rows else {}
        diagnostics.update(
            {
                "last_accepted_residual_norm": solver_info.get("residual_norm", ""),
                "residual_norm_final": solver_info.get("residual_norm", ""),
                "last_accepted_iterations": solver_info.get("iterations", ""),
                "newton_iterations_total": diagnostics.get(
                    "newton_iterations_total",
                    int(sum(iteration_values))
                    if iteration_values
                    else solver_info.get("iterations", ""),
                ),
                "newton_iterations_max": diagnostics.get(
                    "newton_iterations_max",
                    int(max(iteration_values))
                    if iteration_values
                    else solver_info.get("iterations", ""),
                ),
                "last_accepted_line_search_reductions": solver_info.get("line_search_reductions", ""),
                "line_search_reductions_total": diagnostics.get(
                    "line_search_reductions_total",
                    int(sum(line_search_values))
                    if line_search_values
                    else solver_info.get("line_search_reductions", ""),
                ),
                "residual_reduction_ratio": _srm_residual_reduction_ratio(solver_info.get("convergence_history")),
                "internal_external_work_ratio": solver_info.get("internal_external_work_ratio", ""),
                "displacement_increment_norm": last_accepted.get("displacement_increment_norm", diagnostics["displacement_increment_norm"]),
            }
        )
    diagnostics["diagnostic_summary"] = _srm_diagnostic_summary(diagnostics)
    return diagnostics


def _increment_failure_log_row(
    *,
    target: float,
    step_size: float,
    error: FEM2DError,
    elapsed_seconds: float = 0.0,
) -> dict[str, Any]:
    """Keep only scalar convergence evidence needed by checkpoint prediction."""

    diagnostics = _normalized_error_diagnostics(error)
    row: dict[str, Any] = {
        "target": float(target),
        "accepted": False,
        "step_size": float(step_size),
        "error": str(error),
        "elapsed_seconds": max(float(elapsed_seconds), 0.0),
    }
    for key in (
        "trial_status",
        "residual_norm_final",
        "residual_reduction_ratio",
        "newton_iterations_total",
        "newton_iterations_max",
        "line_search_reductions_total",
        "assembly_elapsed_seconds",
        "linear_solve_elapsed_seconds",
        "line_search_elapsed_seconds",
        "postprocess_elapsed_seconds",
        "solver_elapsed_seconds",
    ):
        value = diagnostics.get(key, "")
        if value not in (None, ""):
            row[key] = value
    return row


def _large_deformation_failure_diagnostics(
    *,
    status: str,
    load_fraction: float,
    factor: float,
    next_step: float,
    min_step: float,
    cutback_factor: float,
    cutbacks: int,
    max_cutbacks: int,
    accepted_steps: int,
    state_current: Mapping[str, PlasticState2D] | None,
    state_current_cache: PlasticStateArrayCache | None,
    active_elements: list[str],
    total_u: np.ndarray,
    history: list[dict[str, Any]],
    error: FEM2DError,
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
) -> dict[str, Any]:
    last_history = history[-1] if history else {}
    plastic_ratio = _plastic_ratio_from_state_or_array(state_current, state_current_cache, active_elements)
    plastic_metrics = _srm_plastic_diagnostics(state_current, state_current_cache, active_elements, topology_cache)
    iteration_values = [int(row.get("iterations", 0) or 0) for row in history if isinstance(row, Mapping)]
    line_search_values = [int(row.get("line_search_reductions", 0) or 0) for row in history if isinstance(row, Mapping)]
    diagnostics: dict[str, Any] = {
        "trial_status": status,
        "failure_diagnostic_source": "large_deformation_stage",
        **plastic_metrics,
        "attempted_load_factor": float(load_fraction + factor),
        "last_accepted_load_factor": float(load_fraction),
        "accepted_increment_count": int(accepted_steps),
        "cutback_count": int(cutbacks),
        "max_cutbacks": int(max_cutbacks),
        "failed_step_size": float(factor),
        "next_step_size": float(next_step),
        "min_step": float(min_step),
        "cutback_factor": float(cutback_factor),
        "final_step_size": float(factor),
        "last_accepted_plastic_ratio": float(plastic_ratio),
        "last_accepted_plastic_point_count": int(_plastic_point_count(state_current, state_current_cache)),
        "last_accepted_max_displacement": float(max_displacement_norm(total_u)) if total_u.size else 0.0,
        "max_displacement_norm": float(max_displacement_norm(total_u)) if total_u.size else 0.0,
        "displacement_increment_norm": last_history.get("max_increment_displacement", ""),
        "last_accepted_residual_norm": last_history.get("residual_norm", ""),
        "residual_norm_final": last_history.get("residual_norm", ""),
        "last_accepted_iterations": last_history.get("iterations", ""),
        "newton_iterations_total": int(sum(iteration_values)) if iteration_values else "",
        "newton_iterations_max": int(max(iteration_values)) if iteration_values else "",
        "last_accepted_line_search_reductions": last_history.get("line_search_reductions", ""),
        "line_search_reductions_total": int(sum(line_search_values)) if line_search_values else "",
        "residual_reduction_ratio": _srm_residual_reduction_ratio(last_history.get("convergence_history")),
        "min_det_j": last_history.get("min_detJ", plastic_metrics.get("min_det_j", "")),
        "internal_external_work_ratio": last_history.get("internal_external_work_ratio", ""),
        "failed_increment_error": str(error),
        "increment_log_tail": [dict(row) for row in history[-5:]],
    }
    diagnostics["diagnostic_summary"] = _srm_diagnostic_summary(diagnostics)
    return diagnostics


def _nonlinear_failure_diagnostics(
    *,
    status: str,
    residual_norm: float,
    settings: Mapping[str, Any],
    strength_factor: float,
    convergence_history: list[dict[str, Any]],
    u: np.ndarray,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    active_elements: list[str],
    line_search_reductions: int,
    internal_external_work_ratio: float | None = None,
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
) -> dict[str, Any]:
    plastic_ratio = _plastic_ratio_from_state_or_array(plastic_state, plastic_state_cache, active_elements)
    plastic_metrics = _srm_plastic_diagnostics(plastic_state, plastic_state_cache, active_elements, topology_cache)
    diagnostics: dict[str, Any] = {
        "trial_status": status,
        "failure_diagnostic_source": "newton_solver",
        "strength_factor": float(strength_factor),
        "attempted_load_factor": 1.0,
        "last_accepted_load_factor": "",
        "accepted_increment_count": 0,
        "cutback_count": 0,
        "final_step_size": 1.0,
        "last_accepted_plastic_ratio": float(plastic_ratio),
        "last_accepted_plastic_point_count": int(_plastic_point_count(plastic_state, plastic_state_cache)),
        "last_accepted_max_displacement": float(max_displacement_norm(u)) if u.size else 0.0,
        "max_displacement_norm": float(max_displacement_norm(u)) if u.size else 0.0,
        "displacement_increment_norm": float(max_displacement_norm(u)) if u.size else 0.0,
        "last_accepted_residual_norm": float(residual_norm),
        "residual_norm_final": float(residual_norm),
        "last_accepted_iterations": int(settings.get("max_iter", 0) or 0),
        "last_accepted_line_search_reductions": int(line_search_reductions),
        "newton_iterations_total": int(settings.get("max_iter", 0) or 0),
        "newton_iterations_max": int(settings.get("max_iter", 0) or 0),
        "line_search_reductions_total": int(line_search_reductions),
        "residual_reduction_ratio": _srm_residual_reduction_ratio(convergence_history),
        "internal_external_work_ratio": "" if internal_external_work_ratio is None else float(internal_external_work_ratio),
        "convergence_history_tail": [dict(row) for row in convergence_history[-5:]],
        **plastic_metrics,
    }
    diagnostics["solver_elapsed_seconds"] = float(
        sum(
            max(_srm_record_float(row, "elapsed_seconds", 0.0), 0.0)
            for row in convergence_history
            if isinstance(row, Mapping)
        )
    )
    for timing_key in (
        "assembly_elapsed_seconds",
        "linear_solve_elapsed_seconds",
        "line_search_elapsed_seconds",
        "postprocess_elapsed_seconds",
    ):
        diagnostics[timing_key] = float(
            sum(
                max(_srm_record_float(row, timing_key, 0.0), 0.0)
                for row in convergence_history
                if isinstance(row, Mapping)
            )
        )
    diagnostics["diagnostic_summary"] = _srm_diagnostic_summary(diagnostics)
    return diagnostics


def _srm_call_solve_trial(
    solve_trial: Callable[..., StageResult2D],
    factor: float,
    solver_override: Mapping[str, Any] | None = None,
) -> StageResult2D:
    if solver_override:
        try:
            return solve_trial(factor, solver_override=solver_override)
        except TypeError:
            return solve_trial(factor)
    return solve_trial(factor)


def _srm_mohr_coulomb_fallback_record() -> dict[str, Any]:
    telemetry = mohr_coulomb_fallback_telemetry()
    return {
        "mohr_coulomb_fallback": telemetry,
        "mc_numba_to_python_fallback_count": telemetry["numba_to_python_count"],
        "mc_numba_regularized_projection_count": telemetry["numba_regularized_projection_count"],
        "mc_regularized_projection_count": telemetry["regularized_projection_count"],
        "mc_apex_regularization_count": telemetry["apex_regularization_count"],
        "mc_associated_apex_projection_count": telemetry[
            "associated_apex_projection_count"
        ],
        "mc_legacy_bounded_projection_count": telemetry[
            "legacy_bounded_projection_count"
        ],
        "mc_regularization_method": telemetry["regularization_method"],
        "mc_configured_apex_policy_verified": telemetry[
            "configured_apex_policy_verified"
        ],
        "mc_base_nonassociated_flow_rule_verified": telemetry[
            "base_nonassociated_flow_rule_verified"
        ],
        "mc_constitutive_model_fidelity": telemetry[
            "constitutive_model_fidelity"
        ],
        "mc_regularized_projection_above_relaxed_tolerance_count": telemetry[
            "regularized_projection_above_relaxed_tolerance_count"
        ],
        "mc_active_set_update_attempt_count": telemetry["active_set_update_attempt_count"],
        "mc_active_set_update_hit_count": telemetry["active_set_update_hit_count"],
        "mc_active_set_regularized_update_hit_count": telemetry[
            "active_set_regularized_update_hit_count"
        ],
        "mc_active_set_full_scan_avoided_count": telemetry[
            "active_set_full_scan_avoided_count"
        ],
        "mc_regularized_projection_max_yield_violation": telemetry["max_yield_violation"],
        "mc_regularized_projection_max_relative_yield_violation": telemetry["max_relative_yield_violation"],
        "mc_regularized_projection_samples": telemetry["samples"],
    }


def _srm_trial_record(
    factor: float,
    failure_plastic_ratio: float,
    solve_trial: Callable[..., StageResult2D],
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
    *,
    solver_override: Mapping[str, Any] | None = None,
    retry_of: float | None = None,
    retry_index: int = 0,
) -> dict[str, Any]:
    factor = _round_srm_factor(factor)
    trial_started = _perf_counter()
    reset_mohr_coulomb_fallback_telemetry()
    try:
        trial = _srm_call_solve_trial(solve_trial, factor, solver_override)
        mc_fallback = _srm_mohr_coulomb_fallback_record()
        elapsed = max(_perf_counter() - trial_started, 0.0)
        converged = bool(trial.solver_info.get("converged", True))
        plastic_ratio = _plastic_ratio(trial)
        _srm_compact_lightweight_trial_result(trial)
        ok = converged and plastic_ratio <= failure_plastic_ratio
        solver_elapsed = _srm_trial_solver_elapsed(trial)
        record = {
            "factor": factor,
            "converged": converged,
            "plastic_ratio": plastic_ratio,
            "ok": ok,
            "failure_reason": "" if ok else _srm_failure_reason(converged, plastic_ratio, None),
            "elapsed_seconds": elapsed,
            "solver_elapsed_seconds": solver_elapsed,
            "overhead_elapsed_seconds": max(elapsed - solver_elapsed, 0.0),
            "accounted_elapsed_seconds": min(solver_elapsed, elapsed),
            "unattributed_elapsed_seconds": max(elapsed - solver_elapsed, 0.0),
            "timing_coverage_ratio": (
                min(solver_elapsed / elapsed, 1.0) if elapsed > 0.0 else 1.0
            ),
            **_srm_trial_solver_diagnostics(trial, plastic_ratio, ok, topology_cache),
            **_srm_trial_cache_diagnostics(trial),
            **mc_fallback,
            "result": trial,
        }
        estimate = _srm_estimated_fos_from_last_load(record)
        if estimate is not None:
            record["estimated_fos_from_last_load"] = estimate
        if retry_index:
            record.update({"auto_retry": True, "auto_retry_of": retry_of if retry_of is not None else factor, "auto_retry_index": retry_index})
        _srm_attach_solver_override_diagnostics(record, solver_override)
        return record
    except FEM2DError as exc:
        mc_fallback = _srm_mohr_coulomb_fallback_record()
        elapsed = max(_perf_counter() - trial_started, 0.0)
        diagnostics = _srm_failed_trial_diagnostics(exc)
        diagnostic_solver_elapsed = _srm_diagnostic_solver_elapsed(diagnostics)
        continuation_checkpoint = getattr(
            exc, "_srm_increment_continuation_checkpoint", None
        )
        if not isinstance(
            continuation_checkpoint, _IncrementContinuationCheckpoint
        ):
            continuation_checkpoint = None
        record = {
            "factor": factor,
            "converged": False,
            "plastic_ratio": math.inf,
            "ok": False,
            "error": str(exc),
            "failure_reason": _srm_failure_reason(False, math.inf, str(exc)),
            "elapsed_seconds": elapsed,
            "solver_elapsed_seconds": diagnostic_solver_elapsed,
            "overhead_elapsed_seconds": max(elapsed - diagnostic_solver_elapsed, 0.0),
            "accounted_elapsed_seconds": min(diagnostic_solver_elapsed, elapsed),
            "unattributed_elapsed_seconds": max(
                elapsed - diagnostic_solver_elapsed, 0.0
            ),
            "timing_coverage_ratio": (
                min(diagnostic_solver_elapsed / elapsed, 1.0)
                if elapsed > 0.0
                else 1.0
            ),
            **diagnostics,
            **mc_fallback,
            "diagnostics": diagnostics,
            "result": None,
        }
        record.update(
            _srm_increment_checkpoint_metadata(continuation_checkpoint)
        )
        if continuation_checkpoint is not None:
            record["_increment_continuation_checkpoint"] = (
                continuation_checkpoint
            )
        estimate = _srm_estimated_fos_from_last_load(record)
        if estimate is not None:
            record["estimated_fos_from_last_load"] = estimate
        if retry_index:
            record.update({"auto_retry": True, "auto_retry_of": retry_of if retry_of is not None else factor, "auto_retry_index": retry_index})
        _srm_attach_solver_override_diagnostics(record, solver_override)
        return record


def _srm_attach_solver_override_diagnostics(record: dict[str, Any], solver_override: Mapping[str, Any] | None) -> None:
    if not isinstance(solver_override, Mapping):
        return
    warm_start = solver_override.get("_srm_warm_start")
    if isinstance(warm_start, Mapping) and warm_start:
        displacement = warm_start.get("displacement")
        displacement_size = 0
        try:
            displacement_size = int(np.asarray(displacement, dtype=float).size)
        except (TypeError, ValueError):
            displacement_size = 0
        record.update(
            {
                "warm_start_used": True,
                "warm_start_source": str(warm_start.get("source", "previous_stable_trial")),
                "warm_start_source_factor": warm_start.get("source_factor", ""),
                "warm_start_target_factor": warm_start.get("target_factor", record.get("factor", "")),
                "warm_start_factor_distance": warm_start.get("factor_distance", ""),
                "warm_start_displacement_only": bool(warm_start.get("displacement_only", True)),
                "warm_start_displacement_size": displacement_size,
                "warm_start_max_displacement_norm": warm_start.get("max_displacement_norm", ""),
            }
        )
    policy = solver_override.get("_srm_adaptive_increment_policy")
    if isinstance(policy, Mapping) and policy:
        record.update(
            {
                "adaptive_increment_control": True,
                "adaptive_increment_source": str(policy.get("source", "previous_failed_trial_log")),
                "adaptive_increment_source_factor": policy.get("source_factor", ""),
                "adaptive_increment_target_factor": policy.get("target_factor", record.get("factor", "")),
                "adaptive_increment_reason": str(policy.get("reason", "")),
                "adaptive_increment_last_accepted_load_factor": policy.get("last_accepted_load_factor", ""),
                "adaptive_increment_final_step_size": policy.get("final_step_size", ""),
                "adaptive_increment_cutback_count": policy.get("cutback_count", ""),
                "adaptive_increment_max_cutbacks": policy.get("max_cutbacks", ""),
                "adaptive_increment_cutback_ratio": policy.get("cutback_ratio", ""),
                "adaptive_increment_target_initial_step_factor": policy.get("target_initial_step_factor", ""),
                "adaptive_increment_max_steps_multiplier": policy.get("max_steps_multiplier", ""),
                "adaptive_increment_extra_cutbacks": policy.get("extra_cutbacks", ""),
                "adaptive_increment_min_step_factor": policy.get("min_step_factor", ""),
            }
        )
    retry_policy = solver_override.get("_srm_retry_policy")
    if isinstance(retry_policy, Mapping):
        retry_prediction = retry_policy.get("checkpoint_residual_prediction")
        if isinstance(retry_prediction, Mapping):
            record.update(
                {
                    "checkpoint_residual_prediction_enabled": bool(
                        retry_prediction.get("enabled", False)
                    ),
                    "checkpoint_residual_prediction_reason": str(
                        retry_prediction.get("reason", "")
                    ),
                    "checkpoint_residual_prediction_sample_count": int(
                        retry_prediction.get("sample_count", 0) or 0
                    ),
                    "checkpoint_residual_prediction_ratio": retry_prediction.get(
                        "residual_ratio", ""
                    ),
                    "checkpoint_residual_prediction_extra_cutbacks": int(
                        retry_prediction.get("recommended_extra_cutbacks", 1) or 1
                    ),
                }
            )
    verification = solver_override.get("_srm_boundary_verification")
    if isinstance(verification, Mapping) and verification:
        record.update(
            {
                "boundary_verification": True,
                "boundary_verification_reason": str(verification.get("reason", "")),
                "boundary_verification_of": verification.get("source_factor", record.get("factor", "")),
                "boundary_verification_cold_start": bool(verification.get("cold_start", True)),
                "boundary_checkpoint_continuation_requested": bool(
                    verification.get(
                        "checkpoint_continuation_requested", False
                    )
                ),
                "boundary_checkpoint_continuation_used": bool(
                    record.get(
                        "increment_checkpoint_continuation_used", False
                    )
                ),
                "boundary_checkpoint_fallback_reason": str(
                    record.get("increment_checkpoint_fallback_reason", "")
                ),
                "boundary_verification_early_failure_disabled": bool(
                    verification.get("early_failure_disabled", True)
                ),
            }
        )
        prediction = verification.get("checkpoint_residual_prediction")
        if isinstance(prediction, Mapping):
            record.update(
                {
                    "checkpoint_residual_prediction_enabled": bool(
                        prediction.get("enabled", False)
                    ),
                    "checkpoint_residual_prediction_reason": str(
                        prediction.get("reason", "")
                    ),
                    "checkpoint_residual_prediction_sample_count": int(
                        prediction.get("sample_count", 0) or 0
                    ),
                    "checkpoint_residual_prediction_ratio": prediction.get(
                        "residual_ratio", ""
                    ),
                    "checkpoint_residual_prediction_extra_cutbacks": int(
                        prediction.get("recommended_extra_cutbacks", 1) or 1
                    ),
                }
            )
        if bool(
            record.get("increment_checkpoint_continuation_used", False)
        ):
            record["boundary_verification_cold_start"] = False


def _srm_warm_start_settings(srm_cfg: Mapping[str, Any], *, supported: bool = True) -> dict[str, Any]:
    missing = object()
    raw = srm_cfg.get("warm_start", srm_cfg.get("srm_warm_start", missing))
    raw_map = dict(raw) if isinstance(raw, Mapping) else {}
    configured = raw is not missing
    enabled = _srm_bool(raw_map.get("enabled", raw), False) if configured else False
    if not supported:
        return {
            "enabled": False,
            "configured": configured,
            "supported": False,
            "disabled_reason": "warm_start_not_supported_for_this_srm_solver",
            "displacement_only": True,
            "prefer_stable_source": True,
            "max_factor_distance": 0.05,
        }
    return {
        "enabled": bool(enabled),
        "configured": configured,
        "supported": True,
        "disabled_reason": "" if enabled else "disabled",
        "displacement_only": _srm_bool(raw_map.get("displacement_only", True), True),
        "prefer_stable_source": _srm_bool(raw_map.get("prefer_stable_source", True), True),
        "allow_failed_source": _srm_bool(raw_map.get("allow_failed_source", False), False),
        "max_factor_distance": float(raw_map.get("max_factor_distance", raw_map.get("factor_distance", 0.05)) or 0.05),
    }


def _srm_warm_start_solver_override(
    settings: Mapping[str, Any],
    reference_record: Mapping[str, Any] | None,
    target_factor: float | None,
) -> dict[str, Any] | None:
    if not _srm_bool(settings.get("enabled", False), False):
        return None
    if not _srm_bool(settings.get("displacement_only", True), True):
        return None
    if not isinstance(reference_record, Mapping) or target_factor is None:
        return None
    if not bool(reference_record.get("ok", False)) and not _srm_bool(settings.get("allow_failed_source", False), False):
        return None
    result = reference_record.get("result")
    if not isinstance(result, StageResult2D):
        return None
    source_factor = _srm_record_float(reference_record, "factor")
    target = _round_srm_factor(float(target_factor))
    if not math.isfinite(source_factor) or source_factor <= 0.0 or target <= 0.0:
        return None
    factor_distance = abs(float(target) - float(source_factor))
    max_distance = _srm_record_float(settings, "max_factor_distance", 0.05)
    if math.isfinite(max_distance) and max_distance > 0.0 and factor_distance > max_distance:
        return None
    displacement = np.asarray(result.displacements, dtype=float)
    if displacement.ndim != 1 or displacement.size == 0 or not np.all(np.isfinite(displacement)):
        return None
    return {
        "_srm_warm_start": {
            "enabled": True,
            "source": "previous_stable_trial",
            "source_factor": float(source_factor),
            "target_factor": float(target),
            "factor_distance": float(factor_distance),
            "displacement_only": True,
            "max_displacement_norm": float(max_displacement_norm(displacement)) if displacement.size else 0.0,
            "displacement": displacement.copy(),
        }
    }


def _srm_merge_solver_overrides(*overrides: Mapping[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for override in overrides:
        if isinstance(override, Mapping) and override:
            merged.update(dict(override))
    return merged or None


def _srm_initial_displacement_from_solver_override(solver_override: Mapping[str, Any] | None) -> np.ndarray | None:
    if not isinstance(solver_override, Mapping):
        return None
    warm_start = solver_override.get("_srm_warm_start")
    if not isinstance(warm_start, Mapping) or not _srm_bool(warm_start.get("enabled", True), True):
        return None
    try:
        displacement = np.asarray(warm_start.get("displacement"), dtype=float).copy()
    except (TypeError, ValueError):
        return None
    if displacement.ndim != 1 or displacement.size == 0 or not np.all(np.isfinite(displacement)):
        return None
    return displacement


def _srm_trial_solver_elapsed(trial: StageResult2D) -> float:
    solver_info = trial.solver_info if isinstance(trial.solver_info, Mapping) else {}
    performance = solver_info.get("performance", {}) if isinstance(solver_info.get("performance", {}), Mapping) else {}
    elapsed_candidates: list[float] = []
    elapsed = _srm_record_float(performance, "elapsed_seconds", 0.0)
    if math.isfinite(elapsed) and elapsed > 0.0:
        elapsed_candidates.append(float(elapsed))
    history = solver_info.get("convergence_history")
    if isinstance(history, list):
        total = 0.0
        for row in history:
            if isinstance(row, Mapping):
                total += max(_srm_record_float(row, "elapsed_seconds", 0.0), 0.0)
        if total > 0.0:
            elapsed_candidates.append(float(total))
    increments = solver_info.get("increments")
    if isinstance(increments, Mapping):
        inc_history = increments.get(
            "log",
            increments.get("history", increments.get("increment_history")),
        )
        if isinstance(inc_history, list):
            total = 0.0
            for row in inc_history:
                if isinstance(row, Mapping):
                    total += max(_srm_record_float(row, "elapsed_seconds", 0.0), 0.0)
            if total > 0.0:
                elapsed_candidates.append(float(total))
    large_info = solver_info.get("large_deformation")
    if isinstance(large_info, Mapping):
        large_history = large_info.get("history")
        if isinstance(large_history, list):
            total = sum(
                max(_srm_record_float(row, "elapsed_seconds", 0.0), 0.0)
                for row in large_history
                if isinstance(row, Mapping)
            )
            if total > 0.0:
                elapsed_candidates.append(float(total))
    return max(elapsed_candidates, default=0.0)


def _srm_diagnostic_solver_elapsed(diagnostics: Mapping[str, Any]) -> float:
    elapsed = _srm_record_float(diagnostics, "solver_elapsed_seconds", 0.0)
    if math.isfinite(elapsed) and elapsed > 0.0:
        return float(elapsed)
    total = 0.0
    for key in ("convergence_history_tail", "increment_log_tail"):
        rows = diagnostics.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, Mapping):
                total += max(_srm_record_float(row, "elapsed_seconds", 0.0), 0.0)
    return float(total)


def _srm_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled", "auto"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled", "none", ""}:
        return False
    return default


def _srm_auto_enabled(srm_cfg: Mapping[str, Any]) -> bool:
    mode = _srm_search_mode(srm_cfg)
    if mode in {"auto", "auto_bracket", "diagnostic_auto"}:
        return True
    raw = srm_cfg.get("auto")
    if isinstance(raw, Mapping):
        return _srm_bool(raw.get("enabled", True), True)
    return _srm_bool(raw, False)


def _srm_auto_float(settings: Mapping[str, Any], key: str, default: float) -> float:
    try:
        return float(settings.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _srm_auto_int(settings: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return int(settings.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _srm_auto_settings(srm_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = srm_cfg.get("auto", {})
    raw_map = dict(raw) if isinstance(raw, Mapping) else {}
    settings: dict[str, Any] = {
        "enabled": _srm_auto_enabled(srm_cfg),
        "decision_source": "trial_log_diagnostics",
        "retry_suspect_failures": True,
        "max_suspect_retries": 1,
        "boundary_verification_enabled": False,
        "boundary_verification_strategy": "immediate",
        "boundary_verification_suspect": True,
        "boundary_verification_early_failure": True,
        "boundary_verification_max_recoveries": 4,
        "boundary_verification_defer_min_failure_score": 6,
        "boundary_verification_newton_max_iter_multiplier": 1.25,
        "boundary_verification_max_line_search_multiplier": 1.25,
        "boundary_verification_steps_multiplier": 1.25,
        "boundary_verification_extra_cutbacks": 2,
        "boundary_verification_min_step_factor": 0.5,
        "boundary_checkpoint_continuation_enabled": True,
        "boundary_checkpoint_continuation_extra_cutbacks": 1,
        "boundary_checkpoint_residual_prediction_enabled": True,
        "boundary_checkpoint_residual_prediction_max_extra_cutbacks": 4,
        "boundary_checkpoint_residual_prediction_min_samples": 2,
        "boundary_checkpoint_residual_prediction_max_improving_ratio": 0.90,
        "boundary_checkpoint_residual_prediction_safety_cutbacks": 1,
        "boundary_verification_strict_tangent": True,
        "boundary_verification_cold_retry_on_indeterminate": True,
        "boundary_verification_cold_retry_max_per_factor": 1,
        "retry_strict_tangent": True,
        "factor_tol_enforcement_enabled": False,
        "factor_tol_enforcement_accept_verified_numerical_failure": False,
        "factor_tol_require_physical_failure_evidence": True,
        "factor_tol_enforcement_max_extra_bisections": 8,
        "suspect_last_load_threshold": 0.90,
        "confirmed_last_load_threshold": 0.95,
        "confirmed_cluster_fraction": 0.50,
        "confirmed_plastic_ratio": 0.50,
        "minimum_spanning_cluster_fraction": 0.10,
        "strong_plastic_ratio": 0.70,
        "strong_cluster_fraction": 0.65,
        "stable_regression_ratio": 0.85,
        "retry_newton_max_iter_multiplier": 1.25,
        "retry_max_line_search_multiplier": 1.25,
        "retry_steps_multiplier": 2.0,
        "retry_extra_cutbacks": 6,
        "retry_min_step_factor": 0.5,
        "lower_projection_enabled": True,
        "lower_projection_multipliers": [1.10, 1.20, 0.98],
        "lower_projection_max_probes": 3,
        "lower_projection_skip_coarse_scan_on_bracket": True,
        "early_failure_stop_enabled": True,
        "early_failure_min_cutbacks": 4,
        "early_failure_min_cutback_ratio": 0.75,
        "early_failure_min_last_load": 0.90,
        "early_failure_score_threshold": 5,
        "early_failure_cluster_fraction": 0.50,
        "early_failure_spanning_cluster_fraction": 0.10,
        "early_failure_plastic_ratio": 0.50,
        "early_failure_residual_reduction_min": 0.80,
        "early_failure_line_search_min": 20,
        "early_failure_displacement_increment_min": 0.0,
        "early_failure_strong_collapse_enabled": True,
        "early_failure_strong_min_cutbacks": 4,
        "early_failure_strong_min_cutback_ratio": 0.50,
        "early_failure_strong_max_last_load": 0.85,
        "early_failure_strong_cluster_fraction": 0.80,
        "early_failure_strong_plastic_ratio": 0.70,
        "early_failure_strong_residual_reduction_min": 0.90,
        "early_failure_strong_require_boundary_span": True,
        "adaptive_increment_control_enabled": True,
        "adaptive_increment_min_cutbacks": 4,
        "adaptive_increment_min_cutback_ratio": 0.75,
        "adaptive_increment_min_last_load": 0.0,
        "adaptive_increment_max_factor_distance": 0.25,
        "adaptive_increment_min_initial_step_factor": 0.25,
        "adaptive_increment_max_initial_step_factor": 1.0,
        "adaptive_increment_max_steps_multiplier": 2.0,
        "adaptive_increment_extra_cutbacks": 2,
        "adaptive_increment_min_step_factor": 0.5,
        "adaptive_increment_use_last_accepted_load_factor": True,
        "adaptive_increment_use_final_step_size": True,
        "adaptive_increment_residual_reduction_min": 0.80,
        "adaptive_increment_plastic_ratio": 0.50,
    }
    settings.update(raw_map)
    for adaptive_raw in (
        raw_map.get("adaptive_increment_control"),
        raw_map.get("adaptive_increment"),
        srm_cfg.get("adaptive_increment_control"),
        srm_cfg.get("adaptive_increment"),
    ):
        if not isinstance(adaptive_raw, Mapping):
            continue
        for raw_key, raw_value in adaptive_raw.items():
            key = str(raw_key).strip()
            if not key:
                continue
            if key == "enabled":
                settings["adaptive_increment_control_enabled"] = raw_value
            elif key.startswith("adaptive_increment_"):
                settings[key] = raw_value
            else:
                settings[f"adaptive_increment_{key}"] = raw_value
    settings["enabled"] = _srm_bool(settings.get("enabled", True), True)
    settings["retry_suspect_failures"] = _srm_bool(settings.get("retry_suspect_failures", True), True)
    settings["max_suspect_retries"] = max(0, _srm_auto_int(settings, "max_suspect_retries", 1))
    settings["boundary_verification_enabled"] = _srm_bool(
        settings.get("boundary_verification_enabled", False), False
    )
    verification_strategy = str(
        settings.get("boundary_verification_strategy", "immediate") or "immediate"
    ).strip().lower().replace("-", "_")
    if verification_strategy in {"deferred", "final", "final_only", "defer_final"}:
        verification_strategy = "deferred_final"
    if verification_strategy not in {"immediate", "deferred_final"}:
        raise FEM2DError(
            "srm auto boundary_verification_strategy must be immediate or deferred_final"
        )
    settings["boundary_verification_strategy"] = verification_strategy
    settings["boundary_verification_suspect"] = _srm_bool(
        settings.get("boundary_verification_suspect", True), True
    )
    settings["boundary_verification_early_failure"] = _srm_bool(
        settings.get("boundary_verification_early_failure", True), True
    )
    settings["boundary_checkpoint_continuation_enabled"] = _srm_bool(
        settings.get("boundary_checkpoint_continuation_enabled", True), True
    )
    settings["boundary_checkpoint_continuation_extra_cutbacks"] = max(
        1,
        _srm_auto_int(
            settings,
            "boundary_checkpoint_continuation_extra_cutbacks",
            1,
        ),
    )
    settings["boundary_checkpoint_residual_prediction_enabled"] = _srm_bool(
        settings.get("boundary_checkpoint_residual_prediction_enabled", True),
        True,
    )
    settings["boundary_checkpoint_residual_prediction_max_extra_cutbacks"] = max(
        settings["boundary_checkpoint_continuation_extra_cutbacks"],
        _srm_auto_int(
            settings,
            "boundary_checkpoint_residual_prediction_max_extra_cutbacks",
            4,
        ),
    )
    settings["boundary_checkpoint_residual_prediction_min_samples"] = max(
        2,
        _srm_auto_int(
            settings,
            "boundary_checkpoint_residual_prediction_min_samples",
            2,
        ),
    )
    settings["boundary_checkpoint_residual_prediction_max_improving_ratio"] = min(
        0.999,
        max(
            1.0e-6,
            _srm_auto_float(
                settings,
                "boundary_checkpoint_residual_prediction_max_improving_ratio",
                0.90,
            ),
        ),
    )
    settings["boundary_checkpoint_residual_prediction_safety_cutbacks"] = max(
        0,
        _srm_auto_int(
            settings,
            "boundary_checkpoint_residual_prediction_safety_cutbacks",
            1,
        ),
    )
    settings["boundary_verification_strict_tangent"] = _srm_bool(
        settings.get("boundary_verification_strict_tangent", True),
        True,
    )
    settings["boundary_verification_cold_retry_on_indeterminate"] = _srm_bool(
        settings.get("boundary_verification_cold_retry_on_indeterminate", True),
        True,
    )
    settings["boundary_verification_cold_retry_max_per_factor"] = max(
        0,
        _srm_auto_int(
            settings,
            "boundary_verification_cold_retry_max_per_factor",
            1,
        ),
    )
    settings["retry_strict_tangent"] = _srm_bool(
        settings.get("retry_strict_tangent", True),
        True,
    )
    settings["factor_tol_enforcement_enabled"] = _srm_bool(
        settings.get("factor_tol_enforcement_enabled", False), False
    )
    settings["factor_tol_enforcement_accept_verified_numerical_failure"] = _srm_bool(
        settings.get(
            "factor_tol_enforcement_accept_verified_numerical_failure", False
        ),
        False,
    )
    settings["factor_tol_require_physical_failure_evidence"] = _srm_bool(
        settings.get("factor_tol_require_physical_failure_evidence", True),
        True,
    )
    settings["factor_tol_enforcement_max_extra_bisections"] = max(
        0,
        _srm_auto_int(
            settings,
            "factor_tol_enforcement_max_extra_bisections",
            8,
        ),
    )
    for key in (
        "suspect_last_load_threshold",
        "confirmed_last_load_threshold",
        "confirmed_cluster_fraction",
        "confirmed_plastic_ratio",
        "minimum_spanning_cluster_fraction",
        "strong_plastic_ratio",
        "strong_cluster_fraction",
        "stable_regression_ratio",
        "retry_newton_max_iter_multiplier",
        "retry_max_line_search_multiplier",
        "retry_steps_multiplier",
        "retry_min_step_factor",
        "boundary_verification_newton_max_iter_multiplier",
        "boundary_verification_max_line_search_multiplier",
        "boundary_verification_steps_multiplier",
        "boundary_verification_min_step_factor",
        "early_failure_min_cutback_ratio",
        "early_failure_min_last_load",
        "early_failure_cluster_fraction",
        "early_failure_spanning_cluster_fraction",
        "early_failure_plastic_ratio",
        "early_failure_residual_reduction_min",
        "early_failure_displacement_increment_min",
        "adaptive_increment_min_cutback_ratio",
        "adaptive_increment_min_last_load",
        "adaptive_increment_max_factor_distance",
        "adaptive_increment_min_initial_step_factor",
        "adaptive_increment_max_initial_step_factor",
        "adaptive_increment_max_steps_multiplier",
        "adaptive_increment_min_step_factor",
        "adaptive_increment_residual_reduction_min",
        "adaptive_increment_plastic_ratio",
    ):
        settings[key] = _srm_auto_float(settings, key, float(settings[key]))
    settings["early_failure_stop_enabled"] = _srm_bool(settings.get("early_failure_stop_enabled", True), True)
    settings["early_failure_min_cutbacks"] = max(0, _srm_auto_int(settings, "early_failure_min_cutbacks", int(settings["early_failure_min_cutbacks"])))
    settings["early_failure_score_threshold"] = max(1, _srm_auto_int(settings, "early_failure_score_threshold", int(settings["early_failure_score_threshold"])))
    settings["early_failure_line_search_min"] = max(0, _srm_auto_int(settings, "early_failure_line_search_min", int(settings["early_failure_line_search_min"])))
    settings["early_failure_strong_collapse_enabled"] = _srm_bool(
        settings.get("early_failure_strong_collapse_enabled", True), True
    )
    settings["early_failure_strong_require_boundary_span"] = _srm_bool(
        settings.get("early_failure_strong_require_boundary_span", True), True
    )
    settings["early_failure_strong_min_cutbacks"] = max(
        0,
        _srm_auto_int(
            settings,
            "early_failure_strong_min_cutbacks",
            int(settings["early_failure_strong_min_cutbacks"]),
        ),
    )
    settings["retry_extra_cutbacks"] = max(0, _srm_auto_int(settings, "retry_extra_cutbacks", int(settings["retry_extra_cutbacks"])))
    settings["boundary_verification_extra_cutbacks"] = max(
        0,
        _srm_auto_int(
            settings,
            "boundary_verification_extra_cutbacks",
            int(settings["boundary_verification_extra_cutbacks"]),
        ),
    )
    settings["boundary_verification_max_recoveries"] = max(
        0,
        _srm_auto_int(
            settings,
            "boundary_verification_max_recoveries",
            int(settings["boundary_verification_max_recoveries"]),
        ),
    )
    settings["boundary_verification_defer_min_failure_score"] = max(
        1,
        _srm_auto_int(
            settings,
            "boundary_verification_defer_min_failure_score",
            int(settings["boundary_verification_defer_min_failure_score"]),
        ),
    )
    settings["adaptive_increment_control_enabled"] = _srm_bool(settings.get("adaptive_increment_control_enabled", True), True)
    settings["adaptive_increment_use_last_accepted_load_factor"] = _srm_bool(settings.get("adaptive_increment_use_last_accepted_load_factor", True), True)
    settings["adaptive_increment_use_final_step_size"] = _srm_bool(settings.get("adaptive_increment_use_final_step_size", True), True)
    settings["adaptive_increment_min_cutbacks"] = max(0, _srm_auto_int(settings, "adaptive_increment_min_cutbacks", int(settings["adaptive_increment_min_cutbacks"])))
    settings["adaptive_increment_extra_cutbacks"] = max(0, _srm_auto_int(settings, "adaptive_increment_extra_cutbacks", int(settings["adaptive_increment_extra_cutbacks"])))
    settings["lower_projection_enabled"] = _srm_bool(settings.get("lower_projection_enabled", True), True)
    settings["lower_projection_skip_coarse_scan_on_bracket"] = _srm_bool(settings.get("lower_projection_skip_coarse_scan_on_bracket", True), True)
    raw_multipliers = settings.get("lower_projection_multipliers", [1.10, 1.20, 0.98])
    multipliers: list[float] = []
    if isinstance(raw_multipliers, (list, tuple)):
        for value in raw_multipliers:
            try:
                multiplier = float(value)
            except (TypeError, ValueError):
                continue
            if multiplier > 0.0:
                multipliers.append(multiplier)
    else:
        for key in ("lower_projection_stable_multiplier", "lower_projection_upper_multiplier", "lower_projection_upper_multiplier_2"):
            if key not in settings:
                continue
            try:
                multiplier = float(settings[key])
            except (TypeError, ValueError):
                continue
            if multiplier > 0.0:
                multipliers.append(multiplier)
    settings["lower_projection_multipliers"] = multipliers or [1.10, 1.20, 0.98]
    settings["lower_projection_max_probes"] = max(0, _srm_auto_int(settings, "lower_projection_max_probes", 3))
    return settings


def _srm_record_float(record: Mapping[str, Any], key: str, default: float = math.nan) -> float:
    try:
        value = float(record.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _srm_record_int(record: Mapping[str, Any], key: str, default: int = 0) -> int:
    try:
        return int(record.get(key, default))
    except (TypeError, ValueError):
        return default


def _srm_auto_early_failure_policy(settings: Mapping[str, Any]) -> dict[str, Any]:
    enabled = _srm_bool(settings.get("early_failure_stop_enabled", True), True)
    return {
        "enabled": enabled,
        "source": "srm_auto_log_metric_score",
        "min_cutbacks": max(0, _srm_auto_int(settings, "early_failure_min_cutbacks", 4)),
        "min_cutback_ratio": _srm_auto_float(settings, "early_failure_min_cutback_ratio", 0.75),
        "min_last_load": _srm_auto_float(settings, "early_failure_min_last_load", 0.90),
        "score_threshold": max(1, _srm_auto_int(settings, "early_failure_score_threshold", 5)),
        "cluster_fraction": _srm_auto_float(settings, "early_failure_cluster_fraction", 0.50),
        "spanning_cluster_fraction": _srm_auto_float(settings, "early_failure_spanning_cluster_fraction", 0.10),
        "plastic_ratio": _srm_auto_float(settings, "early_failure_plastic_ratio", 0.50),
        "residual_reduction_min": _srm_auto_float(settings, "early_failure_residual_reduction_min", 0.80),
        "line_search_min": max(0, _srm_auto_int(settings, "early_failure_line_search_min", 20)),
        "displacement_increment_min": _srm_auto_float(settings, "early_failure_displacement_increment_min", 0.0),
        "strong_collapse_enabled": _srm_bool(
            settings.get("early_failure_strong_collapse_enabled", True), True
        ),
        "strong_min_cutbacks": max(
            0,
            _srm_auto_int(settings, "early_failure_strong_min_cutbacks", 4),
        ),
        "strong_min_cutback_ratio": _srm_auto_float(
            settings, "early_failure_strong_min_cutback_ratio", 0.50
        ),
        "strong_max_last_load": _srm_auto_float(
            settings, "early_failure_strong_max_last_load", 0.85
        ),
        "strong_cluster_fraction": _srm_auto_float(
            settings, "early_failure_strong_cluster_fraction", 0.80
        ),
        "strong_plastic_ratio": _srm_auto_float(
            settings, "early_failure_strong_plastic_ratio", 0.70
        ),
        "strong_residual_reduction_min": _srm_auto_float(
            settings, "early_failure_strong_residual_reduction_min", 0.90
        ),
        "strong_require_boundary_span": _srm_bool(
            settings.get("early_failure_strong_require_boundary_span", True),
            True,
        ),
    }


def _srm_auto_adaptive_increment_policy(
    settings: Mapping[str, Any],
    reference_record: Mapping[str, Any] | None,
    target_factor: float | None,
) -> dict[str, Any] | None:
    if not _srm_bool(settings.get("adaptive_increment_control_enabled", True), True):
        return None
    if not isinstance(reference_record, Mapping) or bool(reference_record.get("ok", False)):
        return None
    source_factor = _srm_record_float(reference_record, "factor")
    if not math.isfinite(source_factor) or source_factor <= 0.0 or target_factor is None:
        return None
    target = _round_srm_factor(float(target_factor))
    if target <= 0.0:
        return None
    max_distance = _srm_auto_float(settings, "adaptive_increment_max_factor_distance", 0.25)
    if max_distance > 0.0 and abs(target - source_factor) > max_distance:
        return None

    cutbacks = max(_srm_record_int(reference_record, "cutback_count", 0), 0)
    max_cutbacks = max(_srm_record_int(reference_record, "max_cutbacks", cutbacks), 0)
    cutback_ratio = 1.0 if cutbacks > 0 and max_cutbacks <= 0 else (float(cutbacks) / float(max_cutbacks) if max_cutbacks > 0 else 0.0)
    min_cutbacks = max(0, _srm_auto_int(settings, "adaptive_increment_min_cutbacks", 4))
    min_cutback_ratio = _srm_auto_float(settings, "adaptive_increment_min_cutback_ratio", 0.75)
    last_load = _srm_record_float(reference_record, "last_accepted_load_factor")
    min_last_load = _srm_auto_float(settings, "adaptive_increment_min_last_load", 0.0)
    residual_ratio = _srm_record_float(reference_record, "residual_reduction_ratio")
    plastic_ratio = _srm_auto_plastic_ratio(reference_record)
    spans_boundary = bool(reference_record.get("plastic_cluster_spans_boundary", False))

    reasons: list[str] = []
    evidence = 0
    if cutbacks >= min_cutbacks:
        evidence += 1
        reasons.append(f"cutbacks={cutbacks}")
    if cutback_ratio >= min_cutback_ratio:
        evidence += 1
        reasons.append(f"cutback_ratio={cutback_ratio:.3g}")
    if math.isfinite(last_load) and last_load >= min_last_load and last_load > 0.0:
        evidence += 1
        reasons.append(f"last_load={last_load:.3g}")
    if math.isfinite(residual_ratio) and residual_ratio >= _srm_auto_float(settings, "adaptive_increment_residual_reduction_min", 0.80):
        evidence += 1
        reasons.append(f"residual_reduction={residual_ratio:.3g}")
    if spans_boundary and plastic_ratio >= _srm_auto_float(settings, "adaptive_increment_plastic_ratio", 0.50):
        evidence += 1
        reasons.append("plastic_cluster_near_failure")
    if evidence < 2:
        return None

    min_factor = max(1.0e-3, min(1.0, _srm_auto_float(settings, "adaptive_increment_min_initial_step_factor", 0.25)))
    max_factor = max(min_factor, min(1.0, _srm_auto_float(settings, "adaptive_increment_max_initial_step_factor", 1.0)))
    target_step_factor = max_factor
    if _srm_bool(settings.get("adaptive_increment_use_last_accepted_load_factor", True), True) and math.isfinite(last_load) and 0.0 < last_load < 1.0:
        target_step_factor = min(target_step_factor, max(min_factor, last_load))
    if cutback_ratio >= min_cutback_ratio:
        target_step_factor = min(target_step_factor, 0.5)
    if math.isfinite(residual_ratio) and residual_ratio >= _srm_auto_float(settings, "adaptive_increment_residual_reduction_min", 0.80):
        target_step_factor = min(target_step_factor, 0.5)
    if spans_boundary and plastic_ratio >= _srm_auto_float(settings, "adaptive_increment_plastic_ratio", 0.50):
        target_step_factor = min(target_step_factor, 0.5)
    target_step_factor = min(max_factor, max(min_factor, target_step_factor))
    return {
        "enabled": True,
        "source": "previous_failed_trial_log",
        "source_factor": float(source_factor),
        "target_factor": float(target),
        "reason": "; ".join(reasons),
        "last_accepted_load_factor": "" if not math.isfinite(last_load) else float(last_load),
        "final_step_size": reference_record.get("final_step_size", ""),
        "cutback_count": int(cutbacks),
        "max_cutbacks": int(max_cutbacks),
        "cutback_ratio": float(cutback_ratio),
        "residual_reduction_ratio": "" if not math.isfinite(residual_ratio) else float(residual_ratio),
        "plastic_ratio": "" if not math.isfinite(plastic_ratio) else float(plastic_ratio),
        "target_initial_step_factor": float(target_step_factor),
        "min_initial_step_factor": float(min_factor),
        "max_initial_step_factor": float(max_factor),
        "max_steps_multiplier": max(1.0, _srm_auto_float(settings, "adaptive_increment_max_steps_multiplier", 2.0)),
        "extra_cutbacks": max(0, _srm_auto_int(settings, "adaptive_increment_extra_cutbacks", 2)),
        "min_step_factor": max(1.0e-12, _srm_auto_float(settings, "adaptive_increment_min_step_factor", 0.5)),
        "use_final_step_size": _srm_bool(settings.get("adaptive_increment_use_final_step_size", True), True),
    }


def _srm_auto_trial_solver_override(
    settings: Mapping[str, Any],
    reference_record: Mapping[str, Any] | None = None,
    target_factor: float | None = None,
) -> dict[str, Any] | None:
    override: dict[str, Any] = {}
    if (
        _srm_bool(
            settings.get("boundary_checkpoint_continuation_enabled", True),
            True,
        )
        and (
            _srm_bool(
                settings.get("boundary_verification_enabled", False), False
            )
            or _srm_bool(
                settings.get("retry_suspect_failures", True), True
            )
        )
    ):
        override["_srm_capture_increment_checkpoint"] = True
    policy = _srm_auto_early_failure_policy(settings)
    if bool(policy.get("enabled", False)):
        override["_srm_early_failure_policy"] = policy
    adaptive_policy = _srm_auto_adaptive_increment_policy(settings, reference_record, target_factor)
    if adaptive_policy is not None:
        override["_srm_adaptive_increment_policy"] = adaptive_policy
    return override or None


def _srm_early_failure_policy_from_solver(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = solver.get("_srm_early_failure_policy", {}) if isinstance(solver, Mapping) else {}
    if isinstance(raw, bool):
        return {"enabled": bool(raw)}
    if not isinstance(raw, Mapping):
        return {"enabled": False}
    policy = dict(raw)
    policy["enabled"] = _srm_bool(policy.get("enabled", True), True)
    return policy


def _srm_early_failure_cutback_decision(
    policy: Mapping[str, Any],
    *,
    last_load: float,
    attempted_load: float,
    cutbacks: int,
    max_cutbacks: int,
    state_current: Mapping[str, PlasticState2D] | None,
    state_current_cache: PlasticStateArrayCache | None,
    active_elements: list[str],
    error: FEM2DError,
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
) -> dict[str, Any] | None:
    if not _srm_bool(policy.get("enabled", False), False):
        return None
    effective_cutbacks = max(int(cutbacks) + 1, 0)
    max_cutbacks = max(int(max_cutbacks), 0)
    normal_min_cutbacks = max(
        0, _srm_record_int(policy, "min_cutbacks", 4)
    )
    strong_min_cutbacks = max(
        0, _srm_record_int(policy, "strong_min_cutbacks", 4)
    )
    if effective_cutbacks < min(normal_min_cutbacks, strong_min_cutbacks):
        return None
    cutback_ratio = 1.0 if max_cutbacks <= 0 else float(effective_cutbacks) / float(max_cutbacks)
    min_cutback_ratio = _srm_record_float(policy, "min_cutback_ratio", 0.75)
    min_last_load = _srm_record_float(policy, "min_last_load", 0.90)
    normal_eligible = bool(
        effective_cutbacks >= normal_min_cutbacks
        and cutback_ratio >= min_cutback_ratio
        and (
            not math.isfinite(min_last_load)
            or min_last_load <= 0.0
            or float(last_load) >= min_last_load
        )
    )

    metrics = _srm_plastic_diagnostics(state_current, state_current_cache, active_elements, topology_cache)
    active = max(int(metrics.get("active_element_count", 0) or 0), 1)
    cluster = max(int(metrics.get("connected_plastic_cluster_size", 0) or 0), 0)
    cluster_fraction = float(cluster) / float(active)
    plastic_ratio = _plastic_ratio_from_state_or_array(state_current, state_current_cache, active_elements)
    spans = bool(metrics.get("plastic_cluster_spans_boundary", False))
    qualified_span = bool(
        spans
        and cluster_fraction >= _srm_record_float(policy, "spanning_cluster_fraction", 0.10)
    )
    exc_diag = _normalized_error_diagnostics(error)

    score = 0
    reasons: list[str] = []
    score += 1
    reasons.append(f"cutback_ratio={cutback_ratio:.3g}")
    if qualified_span:
        score += 2
        reasons.append(f"qualified_plastic_boundary_span={cluster_fraction:.3g}")
    if cluster_fraction >= _srm_record_float(policy, "cluster_fraction", 0.50):
        score += 2
        reasons.append(f"cluster_fraction={cluster_fraction:.3g}")
    if plastic_ratio >= _srm_record_float(policy, "plastic_ratio", 0.50):
        score += 2
        reasons.append(f"plastic_ratio={plastic_ratio:.3g}")

    residual_ratio = _srm_record_float(exc_diag, "residual_reduction_ratio")
    if not math.isfinite(residual_ratio):
        residual_ratio = _srm_residual_reduction_ratio(exc_diag.get("convergence_history_tail"))
    residual_min = _srm_record_float(policy, "residual_reduction_min", 0.80)
    if residual_ratio is not None and math.isfinite(float(residual_ratio)) and float(residual_ratio) >= residual_min:
        score += 1
        reasons.append(f"residual_reduction_ratio={float(residual_ratio):.3g}")

    line_search_min = max(0, _srm_record_int(policy, "line_search_min", 20))
    line_search_total = max(
        _srm_record_int(exc_diag, "line_search_reductions_total", 0),
        _srm_record_int(exc_diag, "last_accepted_line_search_reductions", 0),
    )
    if line_search_min > 0 and line_search_total >= line_search_min:
        score += 1
        reasons.append(f"line_search_reductions={line_search_total}")

    strong_enabled = _srm_bool(
        policy.get("strong_collapse_enabled", True), True
    )
    strong_cutback_ratio = _srm_record_float(
        policy, "strong_min_cutback_ratio", 0.50
    )
    strong_max_last_load = _srm_record_float(
        policy, "strong_max_last_load", 0.85
    )
    strong_cluster_fraction = _srm_record_float(
        policy, "strong_cluster_fraction", 0.80
    )
    strong_plastic_ratio = _srm_record_float(
        policy, "strong_plastic_ratio", 0.70
    )
    strong_residual_min = _srm_record_float(
        policy, "strong_residual_reduction_min", 0.90
    )
    strong_requires_span = _srm_bool(
        policy.get("strong_require_boundary_span", True), True
    )
    strong_numerical_difficulty = bool(
        (
            residual_ratio is not None
            and math.isfinite(float(residual_ratio))
            and float(residual_ratio) >= strong_residual_min
        )
        or (line_search_min > 0 and line_search_total >= line_search_min)
    )
    strong_span_ok = bool(not strong_requires_span or spans)
    strong_collapse = bool(
        strong_enabled
        and effective_cutbacks >= strong_min_cutbacks
        and cutback_ratio >= strong_cutback_ratio
        and math.isfinite(float(last_load))
        and 0.0 < float(last_load) <= strong_max_last_load
        and strong_span_ok
        and cluster_fraction >= strong_cluster_fraction
        and plastic_ratio >= strong_plastic_ratio
        and strong_numerical_difficulty
    )
    if strong_collapse:
        strong_reasons = [
            "strong_coarse_collapse",
            f"last_accepted_load_factor={float(last_load):.3g}",
            f"cutback_ratio={cutback_ratio:.3g}",
            f"plastic_cluster_fraction={cluster_fraction:.3g}",
            f"plastic_ratio={plastic_ratio:.3g}",
        ]
        if spans:
            strong_reasons.append("plastic_cluster_spans_boundary")
        if residual_ratio is not None and math.isfinite(float(residual_ratio)):
            strong_reasons.append(
                f"residual_reduction_ratio={float(residual_ratio):.3g}"
            )
        if line_search_total > 0:
            strong_reasons.append(
                f"line_search_reductions={line_search_total}"
            )
        return {
            "early_failure_stop": True,
            "early_failure_policy": "srm_auto_strong_coarse_collapse",
            "early_failure_class": "strong_coarse_collapse",
            "early_failure_score": 8,
            "early_failure_score_threshold": 8,
            "early_failure_reason": "; ".join(strong_reasons),
            "early_failure_cutback_ratio": float(cutback_ratio),
            "early_failure_effective_cutbacks": int(effective_cutbacks),
            "early_failure_qualified_boundary_span": bool(spans),
            "early_failure_cluster_fraction": float(cluster_fraction),
            "early_failure_plastic_ratio": float(plastic_ratio),
            "plastic_cluster_boundary_sides": metrics.get(
                "plastic_cluster_boundary_sides", []
            ),
            "attempted_load_factor": float(attempted_load),
            "last_accepted_load_factor": float(last_load),
        }

    if not normal_eligible:
        return None

    displacement_min = _srm_record_float(policy, "displacement_increment_min", 0.0)
    displacement_increment = _srm_record_float(exc_diag, "displacement_increment_norm")
    if displacement_min > 0.0 and math.isfinite(displacement_increment) and displacement_increment >= displacement_min:
        score += 1
        reasons.append(f"displacement_increment={displacement_increment:.3g}")

    threshold = max(1, _srm_record_int(policy, "score_threshold", 5))
    if score < threshold:
        return None
    return {
        "early_failure_stop": True,
        "early_failure_policy": str(policy.get("source", "srm_auto_log_metric_score")),
        "early_failure_score": int(score),
        "early_failure_score_threshold": int(threshold),
        "early_failure_reason": "; ".join(reasons),
        "early_failure_cutback_ratio": float(cutback_ratio),
        "early_failure_effective_cutbacks": int(effective_cutbacks),
        "early_failure_qualified_boundary_span": qualified_span,
        "plastic_cluster_boundary_sides": metrics.get("plastic_cluster_boundary_sides", []),
        "attempted_load_factor": float(attempted_load),
        "last_accepted_load_factor": float(last_load),
    }


def _srm_auto_plastic_ratio(record: Mapping[str, Any]) -> float:
    ratio = _srm_record_float(record, "plastic_ratio")
    if math.isfinite(ratio):
        return ratio
    return _srm_record_float(record, "last_accepted_plastic_ratio", 0.0)


def _srm_estimated_fos_from_last_load(record: Mapping[str, Any]) -> float | None:
    factor = _srm_record_float(record, "factor")
    last_load = _srm_record_float(record, "last_accepted_load_factor")
    if not (math.isfinite(factor) and factor > 0.0 and math.isfinite(last_load) and last_load > 0.0):
        return None
    return _round_srm_factor(factor * min(last_load, 1.0))


def _srm_auto_classify_trial(
    record: dict[str, Any],
    settings: Mapping[str, Any],
    stable_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if bool(record.get("ok", False)):
        decision = {
            "auto_decision": "stable",
            "auto_failure_class": "stable",
            "srm_trial_state": "stable",
            "auto_failure_score": 0,
            "auto_decision_reason": "full_load_converged",
            "auto_trial_action": "accept_stable",
        }
        record.update(decision)
        return decision

    active = max(_srm_record_int(record, "active_element_count", 0), 1)
    cluster = max(_srm_record_int(record, "connected_plastic_cluster_size", 0), 0)
    cluster_fraction = float(cluster) / float(active)
    plastic_ratio = _srm_auto_plastic_ratio(record)
    last_load = _srm_record_float(record, "last_accepted_load_factor", 0.0)
    cutbacks = max(_srm_record_int(record, "cutback_count", 0), 0)
    max_cutbacks = max(_srm_record_int(record, "max_cutbacks", cutbacks), 0)
    cutback_ratio = 1.0 if cutbacks > 0 and max_cutbacks <= 0 else (float(cutbacks) / float(max_cutbacks) if max_cutbacks > 0 else 0.0)
    spans = bool(record.get("plastic_cluster_spans_boundary", False))
    qualified_span = bool(
        spans
        and cluster_fraction >= _srm_auto_float(settings, "minimum_spanning_cluster_fraction", 0.10)
    )
    reason = str(record.get("failure_reason") or record.get("trial_status") or record.get("error") or "")

    score = 0
    reasons: list[str] = []
    if cutback_ratio >= 1.0:
        score += 2
        reasons.append("cutback_limit")
    if qualified_span:
        score += 2
        reasons.append(f"qualified_plastic_boundary_span={cluster_fraction:.3g}")
    if cluster_fraction >= _srm_auto_float(settings, "confirmed_cluster_fraction", 0.50):
        score += 2
        reasons.append(f"cluster_fraction={cluster_fraction:.3g}")
    if plastic_ratio >= _srm_auto_float(settings, "confirmed_plastic_ratio", 0.50):
        score += 1
        reasons.append(f"plastic_ratio={plastic_ratio:.3g}")
    if last_load >= _srm_auto_float(settings, "confirmed_last_load_threshold", 0.95):
        score += 1
        reasons.append(f"last_load={last_load:.3g}")
    if "detj" in reason.lower() or "displacement" in reason.lower():
        score += 3
        reasons.append("hard_solver_failure")

    stable_regression = False
    if isinstance(stable_reference, Mapping):
        stable_pr = _srm_auto_plastic_ratio(stable_reference)
        stable_cluster = _srm_record_int(stable_reference, "connected_plastic_cluster_size", 0)
        regression_ratio = _srm_auto_float(settings, "stable_regression_ratio", 0.85)
        if stable_pr > 0.0 and plastic_ratio < stable_pr * regression_ratio:
            stable_regression = True
            reasons.append("plastic_ratio_retreated_from_stable")
        if stable_cluster > 0 and cluster < stable_cluster * regression_ratio:
            stable_regression = True
            reasons.append("cluster_retreated_from_stable")

    strong_failure = (
        qualified_span
        and cluster_fraction >= _srm_auto_float(settings, "confirmed_cluster_fraction", 0.50)
        and (plastic_ratio >= _srm_auto_float(settings, "confirmed_plastic_ratio", 0.50) or cutback_ratio >= 1.0)
    ) or (
        qualified_span
        and plastic_ratio >= _srm_auto_float(settings, "strong_plastic_ratio", 0.70)
    ) or (
        cluster_fraction >= _srm_auto_float(settings, "strong_cluster_fraction", 0.65)
        and plastic_ratio >= _srm_auto_float(settings, "strong_plastic_ratio", 0.70)
    )
    weak_failure = (
        (not qualified_span)
        and last_load < _srm_auto_float(settings, "suspect_last_load_threshold", 0.90)
        and plastic_ratio < _srm_auto_float(settings, "strong_plastic_ratio", 0.70)
    ) or (
        stable_regression and last_load < _srm_auto_float(settings, "confirmed_last_load_threshold", 0.95)
    )

    confirmed = bool(strong_failure and not weak_failure)
    failure_class = "confirmed_failure" if confirmed else "suspect_failure"
    action = "use_as_failure_bracket" if confirmed else "retry_or_hold_as_suspect"
    decision = {
        "auto_decision": failure_class,
        "auto_failure_class": failure_class,
        "srm_trial_state": "confirmed_failure" if confirmed else "indeterminate",
        "auto_failure_score": int(score),
        "auto_decision_reason": "; ".join(reasons) or "insufficient_failure_evidence",
        "auto_trial_action": action,
        "auto_cluster_fraction": cluster_fraction,
        "auto_last_load_threshold": _srm_auto_float(settings, "suspect_last_load_threshold", 0.90),
    }
    if last_load > 0.0:
        estimate = _srm_estimated_fos_from_last_load(record)
        if estimate is not None:
            decision["auto_last_accepted_strength_factor_estimate"] = estimate
    record.update(decision)
    return decision


def _srm_trial_decision_state(record: Mapping[str, Any], *, auto_enabled: bool) -> str:
    if bool(record.get("ok", False)):
        return "stable"
    if not auto_enabled:
        return "confirmed_failure"
    if bool(record.get("factor_tol_numerical_failure_boundary", False)):
        return "confirmed_failure"
    state = str(record.get("srm_trial_state", "") or "").strip().lower()
    if state in {"stable", "confirmed_failure", "indeterminate"}:
        return state
    decision = str(record.get("auto_decision", "") or "").strip().lower()
    if decision == "confirmed_failure":
        return "confirmed_failure"
    return "indeterminate"


def _srm_verified_failure_has_physical_evidence(
    record: Mapping[str, Any], settings: Mapping[str, Any]
) -> tuple[bool, str]:
    """Reject solver-only failures as SRM boundaries unless mechanics support them."""

    failure_reason = str(record.get("failure_reason", "") or "").strip().lower()
    if failure_reason in {"invalid_detj", "excessive_displacement"}:
        return True, failure_reason

    active = max(_srm_record_int(record, "active_element_count", 0), 1)
    cluster = max(
        _srm_record_int(record, "connected_plastic_cluster_size", 0), 0
    )
    cluster_fraction = float(cluster) / float(active)
    plastic_ratio = _srm_auto_plastic_ratio(record)
    spans = bool(record.get("plastic_cluster_spans_boundary", False))
    qualified_span = bool(
        spans
        and cluster_fraction
        >= _srm_auto_float(settings, "minimum_spanning_cluster_fraction", 0.10)
    )
    cutbacks = max(_srm_record_int(record, "cutback_count", 0), 0)
    max_cutbacks = max(_srm_record_int(record, "max_cutbacks", cutbacks), 0)
    cutback_exhausted = bool(max_cutbacks > 0 and cutbacks >= max_cutbacks)
    broad_plasticity = bool(
        cluster_fraction
        >= _srm_auto_float(settings, "confirmed_cluster_fraction", 0.50)
        and plastic_ratio
        >= _srm_auto_float(settings, "confirmed_plastic_ratio", 0.50)
    )
    strong_plasticity = bool(
        cluster_fraction
        >= _srm_auto_float(settings, "strong_cluster_fraction", 0.65)
        and plastic_ratio
        >= _srm_auto_float(settings, "strong_plastic_ratio", 0.70)
    )
    supported = bool(
        strong_plasticity
        or (qualified_span and (broad_plasticity or cutback_exhausted))
    )
    reason = (
        "plastic_failure_evidence"
        if supported
        else (
            "no_boundary_spanning_plastic_cluster"
            if not qualified_span
            else "insufficient_plastic_extent"
        )
    )
    return supported, reason


def _srm_checkpoint_residual_prediction(
    record: Mapping[str, Any] | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    baseline = max(
        1,
        _srm_auto_int(
            settings, "boundary_checkpoint_continuation_extra_cutbacks", 1
        ),
    )
    enabled = bool(
        isinstance(record, Mapping)
        and _srm_bool(
            settings.get(
                "boundary_checkpoint_residual_prediction_enabled", True
            ),
            True,
        )
    )
    result: dict[str, Any] = {
        "enabled": enabled,
        "source": "failed_increment_residual_trend",
        "recommended_extra_cutbacks": baseline,
        "sample_count": 0,
        "residual_ratio": "",
        "reason": "disabled_or_no_record" if not enabled else "insufficient_samples",
    }
    if not enabled or not isinstance(record, Mapping):
        return result

    raw_rows = record.get("increment_log_tail", [])
    residuals: list[float] = []
    if isinstance(raw_rows, (list, tuple)):
        for row in raw_rows:
            if not isinstance(row, Mapping) or bool(row.get("accepted", False)):
                continue
            value = _srm_record_float(row, "residual_norm_final")
            if math.isfinite(value) and value > 0.0:
                residuals.append(float(value))
    result["sample_count"] = len(residuals)
    min_samples = max(
        2,
        _srm_auto_int(
            settings,
            "boundary_checkpoint_residual_prediction_min_samples",
            2,
        ),
    )
    if len(residuals) < min_samples:
        return result

    ratios = [
        current / previous
        for previous, current in zip(residuals, residuals[1:])
        if previous > 0.0 and current > 0.0
    ]
    finite_ratios = [ratio for ratio in ratios if math.isfinite(ratio)]
    if not finite_ratios:
        return result
    ratio = float(np.median(np.asarray(finite_ratios, dtype=float)))
    result["residual_ratio"] = ratio
    improving_limit = _srm_auto_float(
        settings,
        "boundary_checkpoint_residual_prediction_max_improving_ratio",
        0.90,
    )
    if ratio <= 0.0 or ratio >= improving_limit:
        result["reason"] = "residual_plateau_or_growth"
        return result

    last_accepted = _srm_record_float(record, "last_accepted_residual_norm")
    if not math.isfinite(last_accepted) or last_accepted <= 0.0:
        last_accepted = max(residuals[-1] * ratio, 1.0e-12)
    target = max(float(last_accepted), 1.0e-12)
    latest = residuals[-1]
    predicted = 1
    if latest > target and 0.0 < ratio < 1.0:
        predicted = max(
            1,
            int(math.ceil(math.log(target / latest) / math.log(ratio))),
        )
    predicted += max(
        0,
        _srm_auto_int(
            settings,
            "boundary_checkpoint_residual_prediction_safety_cutbacks",
            1,
        ),
    )
    maximum = max(
        baseline,
        _srm_auto_int(
            settings,
            "boundary_checkpoint_residual_prediction_max_extra_cutbacks",
            4,
        ),
    )
    recommended = min(maximum, max(baseline, predicted))
    result.update(
        {
            "recommended_extra_cutbacks": int(recommended),
            "latest_residual": float(latest),
            "target_residual": float(target),
            "reason": "predictive_residual_decay",
        }
    )
    return result


def _srm_auto_should_retry(record: Mapping[str, Any], settings: Mapping[str, Any], retry_count: int) -> bool:
    return (
        _srm_bool(settings.get("retry_suspect_failures", True), True)
        and str(record.get("auto_decision", "")) == "suspect_failure"
        and retry_count < max(0, _srm_auto_int(settings, "max_suspect_retries", 1))
    )


def _srm_auto_retry_solver_override(
    settings: Mapping[str, Any],
    retry_index: int,
    *,
    checkpoint: _IncrementContinuationCheckpoint | None = None,
    source_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checkpoint_prediction = _srm_checkpoint_residual_prediction(
        source_record, settings
    )
    override = _srm_auto_trial_solver_override(settings) or {}
    override.update(
        {
            "_srm_auto_retry": True,
            "_srm_retry_index": int(retry_index),
            "_srm_mc_strict_tangent": _srm_bool(
                settings.get("retry_strict_tangent", True),
                True,
            ),
            "_srm_retry_policy": {
                "newton_max_iter_multiplier": _srm_auto_float(settings, "retry_newton_max_iter_multiplier", 1.25),
                "max_line_search_multiplier": _srm_auto_float(settings, "retry_max_line_search_multiplier", 1.25),
                "steps_multiplier": _srm_auto_float(settings, "retry_steps_multiplier", 2.0),
                "extra_cutbacks": _srm_auto_int(settings, "retry_extra_cutbacks", 6),
                "min_step_factor": _srm_auto_float(settings, "retry_min_step_factor", 0.5),
                "checkpoint_continuation_extra_cutbacks": _srm_auto_int(
                    checkpoint_prediction,
                    "recommended_extra_cutbacks",
                    _srm_auto_int(
                        settings,
                        "boundary_checkpoint_continuation_extra_cutbacks",
                        1,
                    ),
                ),
                "checkpoint_residual_prediction": checkpoint_prediction,
            },
        }
    )
    if (
        checkpoint is not None
        and _srm_bool(
            settings.get("boundary_checkpoint_continuation_enabled", True),
            True,
        )
    ):
        override["_srm_increment_checkpoint"] = checkpoint
    return override


def _srm_boundary_verification_reason(
    record: Mapping[str, Any], settings: Mapping[str, Any]
) -> str:
    if not _srm_bool(settings.get("boundary_verification_enabled", False), False):
        return ""
    if bool(record.get("boundary_verification", False)):
        return ""
    reasons: list[str] = []
    if (
        _srm_bool(settings.get("boundary_verification_suspect", True), True)
        and str(record.get("auto_decision", "")) == "suspect_failure"
    ):
        reasons.append("suspect_failure")
    if (
        _srm_bool(settings.get("boundary_verification_early_failure", True), True)
        and bool(record.get("early_failure_stop", False))
    ):
        reasons.append("early_failure_stop")
    return "; ".join(reasons)


def _srm_boundary_verification_solver_override(
    settings: Mapping[str, Any],
    factor: float,
    reason: str,
    *,
    checkpoint: _IncrementContinuationCheckpoint | None = None,
    source_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    continuation_enabled = bool(
        checkpoint is not None
        and _srm_bool(
            settings.get("boundary_checkpoint_continuation_enabled", True),
            True,
        )
    )
    checkpoint_prediction = _srm_checkpoint_residual_prediction(
        source_record, settings
    )
    override = {
        "_srm_mc_strict_tangent": _srm_bool(
            settings.get("boundary_verification_strict_tangent", True),
            True,
        ),
        "_srm_capture_increment_checkpoint": bool(
            _srm_bool(
                settings.get(
                    "boundary_checkpoint_continuation_enabled", True
                ),
                True,
            )
        ),
        "_srm_early_failure_policy": {"enabled": False},
        "_srm_retry_policy": {
            "newton_max_iter_multiplier": _srm_auto_float(
                settings, "boundary_verification_newton_max_iter_multiplier", 1.25
            ),
            "max_line_search_multiplier": _srm_auto_float(
                settings, "boundary_verification_max_line_search_multiplier", 1.25
            ),
            "steps_multiplier": _srm_auto_float(
                settings, "boundary_verification_steps_multiplier", 1.25
            ),
            "extra_cutbacks": _srm_auto_int(
                settings, "boundary_verification_extra_cutbacks", 2
            ),
            "min_step_factor": _srm_auto_float(
                settings, "boundary_verification_min_step_factor", 0.5
            ),
            "checkpoint_continuation_extra_cutbacks": _srm_auto_int(
                checkpoint_prediction,
                "recommended_extra_cutbacks",
                _srm_auto_int(
                    settings,
                    "boundary_checkpoint_continuation_extra_cutbacks",
                    1,
                ),
            ),
            "checkpoint_residual_prediction": checkpoint_prediction,
        },
        "_srm_boundary_verification": {
            "source_factor": float(factor),
            "reason": str(reason),
            "cold_start": not continuation_enabled,
            "checkpoint_continuation_requested": continuation_enabled,
            "early_failure_disabled": True,
            "checkpoint_residual_prediction": checkpoint_prediction,
        },
    }
    if continuation_enabled:
        override["_srm_increment_checkpoint"] = checkpoint
    return override


def _srm_auto_lower_projection_factors(
    failed_record: Mapping[str, Any],
    lower_values: list[float],
    fail_factor: float,
    factor_tol: float,
    settings: Mapping[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    info: dict[str, Any] = {
        "enabled": _srm_bool(settings.get("lower_projection_enabled", True), True),
        "used": False,
        "source_factor": failed_record.get("factor", ""),
        "last_accepted_load_factor": failed_record.get("last_accepted_load_factor", ""),
        "estimated_fos": failed_record.get("estimated_fos_from_last_load", ""),
        "strategy": "strength_factor_times_last_accepted_load_factor",
        "probe_factors": [],
        "skip_reason": "",
    }
    if not info["enabled"]:
        info["skip_reason"] = "disabled"
        return [], info
    estimate = _srm_estimated_fos_from_last_load(failed_record)
    if estimate is None:
        info["skip_reason"] = "missing_last_accepted_load_factor"
        return [], info
    lower_min = min(lower_values) if lower_values else 0.0
    upper_limit = max(float(fail_factor) - max(float(factor_tol), 1.0e-12), 0.0)
    if upper_limit <= 0.0 or estimate <= 0.0:
        info["skip_reason"] = "invalid_projection_bounds"
        return [], info
    max_probes = max(0, _srm_auto_int(settings, "lower_projection_max_probes", 3))
    if max_probes <= 0:
        info["skip_reason"] = "max_probes_zero"
        return [], info
    raw_multipliers = settings.get("lower_projection_multipliers", [1.10, 1.20, 0.98])
    multipliers = [1.10, 1.20, 0.98]
    if isinstance(raw_multipliers, (list, tuple)):
        parsed: list[float] = []
        for value in raw_multipliers:
            try:
                multiplier = float(value)
            except (TypeError, ValueError):
                continue
            if multiplier > 0.0:
                parsed.append(multiplier)
        if parsed:
            multipliers = parsed
    probes: list[float] = []
    seen: set[float] = set()
    for multiplier in multipliers:
        if len(probes) >= max_probes:
            break
        value = _round_srm_factor(float(estimate) * multiplier)
        if value <= 0.0 or value >= upper_limit:
            continue
        if lower_min > 0.0 and value < lower_min:
            value = _round_srm_factor(lower_min)
        if value <= 0.0 or value >= upper_limit or value in seen:
            continue
        probes.append(value)
        seen.add(value)
    if not probes:
        info["skip_reason"] = "no_probe_within_bounds"
        return [], info
    info["used"] = True
    info["probe_factors"] = list(probes)
    return probes, info


def _srm_progress_enabled(srm_cfg: Mapping[str, Any]) -> bool:
    runtime = srm_cfg.get("_runtime", srm_cfg.get("_execution", {}))
    context = str(runtime.get("context", "") if isinstance(runtime, Mapping) else "").lower().strip()
    default = context == "gui"
    raw = srm_cfg.get("progress_stdout", srm_cfg.get("log_progress", srm_cfg.get("progress_log", default)))
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return default
    if isinstance(raw, (int, float)):
        return bool(raw)
    text = str(raw).lower().strip()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled", "none", ""}:
        return False
    return default


_SRM_REQUIRED_TRIAL_LOG_FIELDS = (
    "auto_decision",
    "auto_failure_class",
    "auto_failure_score",
    "auto_decision_reason",
    "auto_trial_action",
    "auto_retry",
    "auto_retry_of",
    "auto_retry_index",
    "auto_retry_planned",
    "auto_retry_reason",
    "auto_retry_result",
    "auto_superseded_by_retry",
    "auto_cluster_fraction",
    "auto_last_accepted_strength_factor_estimate",
    "estimated_fos_from_last_load",
    "warm_start_used",
    "warm_start_source",
    "warm_start_source_factor",
    "warm_start_target_factor",
    "warm_start_factor_distance",
    "warm_start_displacement_only",
    "warm_start_displacement_size",
    "warm_start_max_displacement_norm",
    "adaptive_increment_control",
    "adaptive_increment_source",
    "adaptive_increment_source_factor",
    "adaptive_increment_target_factor",
    "adaptive_increment_reason",
    "adaptive_increment_last_accepted_load_factor",
    "adaptive_increment_final_step_size",
    "adaptive_increment_cutback_count",
    "adaptive_increment_max_cutbacks",
    "adaptive_increment_cutback_ratio",
    "adaptive_increment_target_initial_step_factor",
    "adaptive_increment_max_steps_multiplier",
    "adaptive_increment_extra_cutbacks",
    "adaptive_increment_min_step_factor",
    "elapsed_seconds",
    "solver_elapsed_seconds",
    "overhead_elapsed_seconds",
    "accounted_elapsed_seconds",
    "unattributed_elapsed_seconds",
    "timing_coverage_ratio",
    "assembly_elapsed_seconds",
    "linear_solve_elapsed_seconds",
    "line_search_elapsed_seconds",
    "postprocess_elapsed_seconds",
    "solver_cancel_requested",
    "solver_cancel_checkpoint",
    "solver_cancel_scope",
    "increment_checkpoint_available",
    "increment_checkpoint_schema",
    "increment_checkpoint_fingerprint",
    "increment_checkpoint_load_factor",
    "increment_checkpoint_accepted_steps",
    "increment_checkpoint_cutbacks",
    "increment_checkpoint_continuation_requested",
    "increment_checkpoint_continuation_used",
    "increment_checkpoint_fallback_reason",
    "increment_checkpoint_source_load_factor",
    "increment_checkpoint_resumed_accepted_steps",
    "increment_checkpoint_resumed_cutbacks",
    "increment_checkpoint_reused_history_rows",
    "boundary_checkpoint_continuation_requested",
    "boundary_checkpoint_continuation_used",
    "boundary_checkpoint_fallback_reason",
    "factor_tol_numerical_failure_boundary",
    "factor_tol_numerical_failure_rejected",
    "factor_tol_physical_failure_evidence",
    "factor_tol_enforcement_original_state",
    "factor_tol_enforcement_reason",
    "checkpoint_residual_prediction_enabled",
    "checkpoint_residual_prediction_reason",
    "checkpoint_residual_prediction_sample_count",
    "checkpoint_residual_prediction_ratio",
    "checkpoint_residual_prediction_extra_cutbacks",
    "plastic_ratio_delta",
    "max_equivalent_plastic_strain",
    "mean_equivalent_plastic_strain",
    "top_percentile_equivalent_plastic_strain",
    "yielded_element_count",
    "connected_plastic_cluster_size",
    "plastic_cluster_spans_boundary",
    "last_accepted_load_factor",
    "accepted_increment_count",
    "cutback_count",
    "final_step_size",
    "newton_iterations_total",
    "newton_iterations_max",
    "line_search_reductions_total",
    "residual_norm_final",
    "residual_reduction_ratio",
    "min_det_j",
    "max_displacement_norm",
    "displacement_increment_norm",
    "internal_external_work_ratio",
    "mc_numba_to_python_fallback_count",
    "mc_numba_regularized_projection_count",
    "mc_regularized_projection_count",
    "mc_apex_regularization_count",
    "mc_associated_apex_projection_count",
    "mc_legacy_bounded_projection_count",
    "mc_regularization_method",
    "mc_configured_apex_policy_verified",
    "mc_base_nonassociated_flow_rule_verified",
    "mc_constitutive_model_fidelity",
    "mc_regularized_projection_above_relaxed_tolerance_count",
    "mc_active_set_update_attempt_count",
    "mc_active_set_update_hit_count",
    "mc_active_set_regularized_update_hit_count",
    "mc_active_set_full_scan_avoided_count",
    "mc_active_set_policy",
    "mc_active_set_tangent_reuse_enabled",
    "mc_active_set_tangent_reuse_disabled_reason",
    "mc_active_set_tangent_invalidation_count",
    "mc_active_set_tangent_invalidated_point_count",
    "mc_active_set_consistent_tangent",
    "mc_active_set_cutback_reset_policy",
    "mc_active_set_strict_unstable_points_only",
    "mc_geometry_cache_enabled",
    "mc_geometry_cache_scope",
    "mc_geometry_cache_block_hits",
    "mc_geometry_cache_block_misses",
    "mc_geometry_cache_element_count",
    "mc_regularized_projection_max_yield_violation",
    "mc_regularized_projection_max_relative_yield_violation",
    "mc_regularized_projection_samples",
)


def _srm_emit_progress(
    srm_cfg: Mapping[str, Any] | None,
    *,
    stage_name: str,
    record: Mapping[str, Any],
    index: int,
    total: int | None = None,
    current_fos: float | None = None,
    prefix: str = "trial",
) -> None:
    if not isinstance(srm_cfg, Mapping) or not _srm_progress_enabled(srm_cfg):
        return
    total_text = "" if total is None or total <= 0 else f"/{int(total)}"
    fos_text = "" if current_fos is None else f" current_FOS={current_fos:g}"
    reason = str(record.get("failure_reason") or record.get("error") or "")
    reason_text = f" reason={reason}" if reason else ""
    auto = str(record.get("auto_decision") or "")
    auto_text = f" auto={auto}" if auto else ""
    retry_index = record.get("auto_retry_index", "")
    retry_text = f" retry={retry_index}" if retry_index not in ("", None) else ""
    elapsed = record.get("elapsed_seconds", "")
    elapsed_text = ""
    try:
        elapsed_value = float(elapsed)
        if math.isfinite(elapsed_value):
            elapsed_text = f" elapsed={elapsed_value:.3f}s"
    except (TypeError, ValueError):
        elapsed_text = ""
    estimate = record.get("estimated_fos_from_last_load", "")
    estimate_text = f" est_FOS={estimate}" if estimate not in ("", None) else ""
    diagnostic_summary = str(record.get("diagnostic_summary") or "")
    diagnostic_text = f" diag=({diagnostic_summary})" if diagnostic_summary else ""
    mc_fallback_count = int(record.get("mc_regularized_projection_count", 0) or 0)
    mc_fallback_text = f" mc_regularized_fallback={mc_fallback_count}" if mc_fallback_count else ""
    print(
        f"[GeoFEM][SRM {prefix}] stage={stage_name} trial={index}{total_text} "
        f"factor={record.get('factor', '')} ok={record.get('ok', '')} "
        f"converged={record.get('converged', '')} plastic_ratio={record.get('plastic_ratio', '')}"
        f"{fos_text}{auto_text}{retry_text}{elapsed_text}{estimate_text}{reason_text}{diagnostic_text}{mc_fallback_text}",
        flush=True,
    )


def _srm_trial_cache_diagnostics(trial: StageResult2D) -> dict[str, Any]:
    solver_info = trial.solver_info if isinstance(trial.solver_info, Mapping) else {}
    topology_info: Mapping[str, Any] = {}
    large_info = solver_info.get("large_deformation")
    if isinstance(large_info, Mapping) and isinstance(large_info.get("topology_cache"), Mapping):
        topology_info = large_info["topology_cache"]
    elif isinstance(solver_info.get("topology_cache"), Mapping):
        topology_info = solver_info["topology_cache"]
    diagnostics: dict[str, Any] = {}
    diagnostics.update(_srm_active_set_cache_diagnostics(solver_info))
    if topology_info:
        diagnostics.update(
            {
                "topology_cache_kind": str(topology_info.get("cache_kind", "")),
                "topology_cache_id": str(topology_info.get("topology_cache_id", "")),
                "topology_cache_reuse_scope": str(topology_info.get("reuse_scope", "")),
                "topology_shared_across_srm_factors": bool(topology_info.get("shared_across_srm_factors", False)),
                "topology_stiffness_pattern_cached": bool(topology_info.get("stiffness_pattern_cached", False)),
            }
        )
    if "sparse_pattern_cached" in solver_info:
        diagnostics["sparse_pattern_cached"] = bool(solver_info.get("sparse_pattern_cached", False))
    if "constraint_dofs_cached" in solver_info:
        diagnostics["constraint_dofs_cached"] = bool(solver_info.get("constraint_dofs_cached", False))
    reduced_cache = solver_info.get("reduced_matrix_cache")
    if isinstance(reduced_cache, Mapping):
        diagnostics["reduced_matrix_cache_enabled"] = bool(reduced_cache.get("enabled", False))
        diagnostics["reduced_matrix_cache_hits"] = int(reduced_cache.get("hits", 0) or 0)
        diagnostics["reduced_matrix_cache_builds"] = int(reduced_cache.get("builds", 0) or 0)
    symbolic_cache = solver_info.get("symbolic_ordering_cache", solver_info.get("symbolic_cache"))
    if isinstance(symbolic_cache, Mapping):
        diagnostics["symbolic_ordering_cache_enabled"] = bool(symbolic_cache.get("enabled", False))
        diagnostics["symbolic_ordering_cache_hits"] = int(symbolic_cache.get("hits", 1 if str(symbolic_cache.get("state", "")) == "hit" else 0) or 0)
        diagnostics["symbolic_ordering_cache_misses"] = int(symbolic_cache.get("misses", 1 if str(symbolic_cache.get("state", "")) == "miss" else 0) or 0)
    if "postprocess_results" in solver_info:
        diagnostics["postprocess_results"] = bool(solver_info.get("postprocess_results", True))
    if "plastic_ratio_source" in solver_info:
        diagnostics["plastic_ratio_source"] = str(solver_info.get("plastic_ratio_source", ""))
    lightweight = solver_info.get("srm_lightweight_trial")
    if isinstance(lightweight, Mapping):
        diagnostics["lightweight_trial_compacted"] = bool(lightweight.get("compacted", False))
        diagnostics["lightweight_removed_element_rows"] = int(lightweight.get("removed_element_rows", 0) or 0)
        diagnostics["lightweight_removed_integration_point_rows"] = int(lightweight.get("removed_integration_point_rows", 0) or 0)
        diagnostics["lightweight_removed_interface_rows"] = int(lightweight.get("removed_interface_rows", 0) or 0)
        diagnostics["lightweight_removed_structural_rows"] = int(lightweight.get("removed_structural_rows", 0) or 0)
    diagnostics["trial_element_rows"] = len(trial.element_results)
    diagnostics["trial_integration_point_rows"] = len(trial.integration_point_results)
    diagnostics["trial_interface_rows"] = len(trial.interface_results)
    diagnostics["trial_structural_rows"] = len(trial.structural_results)
    diagnostics["trial_output_dir_materialized"] = trial.output_dir is not None
    return diagnostics


def _srm_compact_lightweight_trial_result(trial: StageResult2D) -> None:
    solver_info = trial.solver_info if isinstance(trial.solver_info, dict) else {}
    if not solver_info or bool(solver_info.get("postprocess_results", True)):
        return
    removed_element_rows = len(trial.element_results)
    removed_ip_rows = len(trial.integration_point_results)
    removed_interface_rows = len(trial.interface_results)
    removed_structural_rows = len(trial.structural_results)
    trial.element_results = []
    trial.integration_point_results = []
    trial.interface_results = []
    trial.structural_results = []
    solver_info["srm_lightweight_trial"] = {
        "enabled": True,
        "compacted": True,
        "removed_element_rows": removed_element_rows,
        "removed_integration_point_rows": removed_ip_rows,
        "removed_interface_rows": removed_interface_rows,
        "removed_structural_rows": removed_structural_rows,
        "artifact_policy": "non-adopted SRM trials keep only convergence, plastic_ratio, cache diagnostics, displacements, reactions, and state arrays",
    }


def _srm_plastic_state_cache_drift(
    retained: PlasticStateArrayCache,
    recomputed: PlasticStateArrayCache,
    *,
    absolute_tolerance: float = 1.0e-11,
    relative_tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    topology_matches = bool(
        retained.element_ids == recomputed.element_ids
        and retained.plastic_strains.shape == recomputed.plastic_strains.shape
        and retained.kappas.shape == recomputed.kappas.shape
        and np.array_equal(retained.state_point_counts, recomputed.state_point_counts)
    )
    if not topology_matches:
        return {
            "topology_matches": False,
            "exceeds_tolerance": True,
            "state_vars_require_reanalysis": False,
            "requires_full_reanalysis": True,
            "reason": "plastic_state_cache_topology_mismatch",
        }

    strain_delta = float(
        np.max(np.abs(recomputed.plastic_strains - retained.plastic_strains))
    ) if retained.plastic_strains.size else 0.0
    kappa_delta = float(
        np.max(np.abs(recomputed.kappas - retained.kappas))
    ) if retained.kappas.size else 0.0
    strain_scale = float(np.max(np.abs(retained.plastic_strains))) if retained.plastic_strains.size else 0.0
    kappa_scale = float(np.max(np.abs(retained.kappas))) if retained.kappas.size else 0.0
    strain_tolerance = float(absolute_tolerance + relative_tolerance * strain_scale)
    kappa_tolerance = float(absolute_tolerance + relative_tolerance * kappa_scale)
    present_mismatch_count = int(np.count_nonzero(retained.present != recomputed.present))
    state_var_flag_mismatch_count = int(
        np.count_nonzero(retained.state_var_flags != recomputed.state_var_flags)
    )
    state_var_point_count = int(
        np.count_nonzero(np.logical_or(retained.state_var_flags, recomputed.state_var_flags))
    )
    state_vars_require_reanalysis = state_var_point_count > 0
    exceeds = bool(
        strain_delta > strain_tolerance
        or kappa_delta > kappa_tolerance
        or present_mismatch_count > 0
        or state_var_flag_mismatch_count > 0
        or state_vars_require_reanalysis
    )
    return {
        "topology_matches": True,
        "exceeds_tolerance": exceeds,
        "max_plastic_strain_delta": strain_delta,
        "max_kappa_delta": kappa_delta,
        "plastic_strain_tolerance": strain_tolerance,
        "kappa_tolerance": kappa_tolerance,
        "present_mismatch_count": present_mismatch_count,
        "state_var_flag_mismatch_count": state_var_flag_mismatch_count,
        "state_var_point_count": state_var_point_count,
        "state_vars_require_reanalysis": state_vars_require_reanalysis,
        "requires_full_reanalysis": state_vars_require_reanalysis,
        "reason": "advanced_state_vars_require_full_reanalysis" if state_vars_require_reanalysis else "",
    }


def _srm_postprocess_retained_trial(
    result: StageResult2D,
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None,
    loads: Any,
    initial_stresses: Mapping[str, np.ndarray] | None,
    initial_plastic_state: Mapping[str, PlasticState2D] | None,
    strength_factor: float,
) -> StageResult2D:
    post_start = _perf_counter()
    retained_state = result.plastic_state
    retained_cache = result.plastic_state_array_cache
    authoritative_cache = retained_cache or build_plastic_state_array_cache(
        mesh, materials, retained_state
    )
    initial_state_for_results = _update_liquefaction_state_from_pore_pressure(
        mesh, materials, result.pore_pressure, initial_plastic_state
    )
    initial_cache = build_plastic_state_array_cache(
        mesh, materials, initial_state_for_results
    )
    state_for_results = _materialized_plastic_state_for_postprocess(
        initial_state_for_results, initial_cache
    )
    postprocess_state_info: dict[str, Any] = {}
    element_results, updated_plastic_state = compute_element_results_and_state(
        mesh,
        materials,
        result.displacements,
        initial_stresses=initial_stresses,
        strength_factor=strength_factor,
        plastic_state=state_for_results,
        collect_results=True,
        postprocess_info=postprocess_state_info,
        collect_integration_point_rows=True,
        plastic_state_cache=initial_cache,
    )
    integration_rows = _pop_same_pass_integration_point_rows(postprocess_state_info)
    updated_cache = _pop_postprocess_state_array_cache(postprocess_state_info)
    if updated_cache is None:
        updated_cache = build_plastic_state_array_cache(mesh, materials, updated_plastic_state)
    state_drift = _srm_plastic_state_cache_drift(authoritative_cache, updated_cache)
    postprocess_state_info.update(
        {
            "state_commit_authority": "retained_solver_trial",
            "output_state_recomputed_from": "stage_initial_state",
            "recomputed_state_committed": False,
            "state_drift": state_drift,
        }
    )
    active_elements = [element.id for element in mesh.elements if element.active]
    result.element_results = element_results
    result.plastic_state = retained_state
    result.plastic_state_array_cache = authoritative_cache
    result.interface_results = compute_interface_results(mesh, interfaces, result.displacements)
    result.structural_results = compute_structural_results(
        mesh, materials, structural_elements, result.displacements, loads=loads
    )
    _attach_structural_extra_dofs(result, mesh, structural_elements)
    if integration_rows is not None:
        result.integration_point_results = integration_rows
    else:
        _attach_integration_point_results(
            result,
            mesh,
            materials,
            result.displacements,
            strength_factor=strength_factor,
            plastic_state=state_for_results,
            initial_stresses=initial_stresses,
        )
    result.solver_info["postprocess_results"] = True
    result.solver_info["postprocess_state_commit"] = postprocess_state_info
    result.solver_info["plastic_ratio"] = authoritative_cache.plastic_ratio(active_elements)
    result.solver_info["plastic_ratio_source"] = "plastic_state_array_cache"
    result.solver_info["plastic_state_array_cache"] = plastic_state_array_cache_info(authoritative_cache)
    result.solver_info["srm_retained_trial_postprocess"] = {
        "enabled": True,
        "strength_factor": float(strength_factor),
        "nonlinear_reanalysis_avoided": not bool(state_drift.get("requires_full_reanalysis", True)),
        "state_commit_authority": "retained_solver_trial",
        "output_state_recomputed_from": "stage_initial_state",
        "state_drift": state_drift,
        "state_drift_exceeds_tolerance": bool(state_drift.get("exceeds_tolerance", True)),
        "requires_full_reanalysis": bool(state_drift.get("requires_full_reanalysis", True)),
    }
    performance = result.solver_info.setdefault("performance", {})
    if isinstance(performance, dict):
        performance["postprocess_elapsed_seconds"] = float(
            performance.get("postprocess_elapsed_seconds", 0.0) or 0.0
        ) + max(_perf_counter() - post_start, 0.0)
    return result


def _srm_public_trial_record(record: Mapping[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in record.items() if key != "result" and not str(key).startswith("_")}
    for key in _SRM_REQUIRED_TRIAL_LOG_FIELDS:
        public.setdefault(key, "")
    return public


def _srm_trial_timing_summary(trials: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = [row for row in trials if isinstance(row, Mapping)]
    elapsed_values = [max(_srm_record_float(row, "elapsed_seconds", 0.0), 0.0) for row in rows]
    solver_values = [max(_srm_record_float(row, "solver_elapsed_seconds", 0.0), 0.0) for row in rows]
    overhead_values = [max(_srm_record_float(row, "overhead_elapsed_seconds", 0.0), 0.0) for row in rows]
    unattributed_values = [
        max(
            _srm_record_float(
                row,
                "unattributed_elapsed_seconds",
                _srm_record_float(row, "overhead_elapsed_seconds", 0.0),
            ),
            0.0,
        )
        for row in rows
    ]
    slowest: Mapping[str, Any] = {}
    if rows:
        slowest = max(rows, key=lambda row: _srm_record_float(row, "elapsed_seconds", 0.0))
    return {
        "schema": "geofem.srm_trial_timing.v2",
        "trial_count": len(rows),
        "total_elapsed_seconds": float(sum(elapsed_values)),
        "total_solver_elapsed_seconds": float(sum(solver_values)),
        "total_overhead_elapsed_seconds": float(sum(overhead_values)),
        "total_unattributed_elapsed_seconds": float(
            sum(unattributed_values)
        ),
        "timing_coverage_ratio": (
            min(float(sum(solver_values)) / float(sum(elapsed_values)), 1.0)
            if sum(elapsed_values) > 0.0
            else 1.0
        ),
        "average_elapsed_seconds": float(sum(elapsed_values) / len(elapsed_values)) if elapsed_values else 0.0,
        "slowest_factor": slowest.get("factor", "") if isinstance(slowest, Mapping) else "",
        "slowest_elapsed_seconds": _srm_record_float(slowest, "elapsed_seconds", 0.0) if isinstance(slowest, Mapping) else 0.0,
        "slowest_status": slowest.get("trial_status", slowest.get("failure_reason", "")) if isinstance(slowest, Mapping) else "",
        "slowest_auto_decision": slowest.get("auto_decision", "") if isinstance(slowest, Mapping) else "",
    }


def _attach_srm_trial_timing(result: StageResult2D, trials: list[Mapping[str, Any]]) -> dict[str, Any]:
    timing = _srm_trial_timing_summary(trials)
    performance = result.solver_info.setdefault("performance", {})
    if not isinstance(performance, dict):
        performance = {}
        result.solver_info["performance"] = performance
    performance["srm_trial_elapsed_seconds"] = float(timing.get("total_elapsed_seconds", 0.0) or 0.0)
    performance["srm_trial_solver_elapsed_seconds"] = float(timing.get("total_solver_elapsed_seconds", 0.0) or 0.0)
    performance["srm_trial_overhead_elapsed_seconds"] = float(timing.get("total_overhead_elapsed_seconds", 0.0) or 0.0)
    performance["srm_trial_unattributed_elapsed_seconds"] = float(
        timing.get("total_unattributed_elapsed_seconds", 0.0) or 0.0
    )
    performance["srm_trial_timing_coverage_ratio"] = float(
        timing.get("timing_coverage_ratio", 0.0) or 0.0
    )
    performance["srm_trial_count"] = int(timing.get("trial_count", 0) or 0)
    performance["srm_slowest_trial_elapsed_seconds"] = float(timing.get("slowest_elapsed_seconds", 0.0) or 0.0)
    return timing


def _srm_cancel_requested(cancel_token: Any) -> bool:
    if cancel_token is None:
        return False
    if callable(cancel_token):
        try:
            return bool(cancel_token())
        except TypeError:
            return False
    for attr in ("cancelled", "canceled", "is_cancelled", "is_canceled", "is_set"):
        marker = getattr(cancel_token, attr, None)
        if marker is None:
            continue
        try:
            return bool(marker() if callable(marker) else marker)
        except TypeError:
            continue
    return bool(cancel_token) if isinstance(cancel_token, bool) else False


def _srm_cancel_token_from_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, os.PathLike)):
        text = str(value).strip()
        if not text:
            return None
        cancel_path = Path(text)
        return lambda cancel_path=cancel_path: cancel_path.exists()
    return value


def _srm_cancel_token_from_config(srm_cfg: Mapping[str, Any], parallel_cfg: Mapping[str, Any]) -> Any:
    for key in ("_cancel_token", "_cancel_requested", "cancel_token", "cancel_requested", "cancel_file", "cancel_path"):
        token = _srm_cancel_token_from_value(srm_cfg.get(key))
        if token is not None:
            return token
    raw_parallel = srm_cfg.get("parallel", srm_cfg.get("parallel_trials", srm_cfg.get("parallelization")))
    if isinstance(raw_parallel, Mapping):
        for key in ("_cancel_token", "_cancel_requested", "cancel_token", "cancel_requested", "cancel_file", "cancel_path"):
            token = _srm_cancel_token_from_value(raw_parallel.get(key))
            if token is not None:
                return token
    for key in ("_cancel_token", "_cancel_requested", "cancel_token", "cancel_requested", "cancel_file", "cancel_path"):
        token = _srm_cancel_token_from_value(parallel_cfg.get(key))
        if token is not None:
            return token
    runtime = srm_cfg.get("_runtime", srm_cfg.get("_execution", {}))
    if isinstance(runtime, Mapping):
        for key in ("_cancel_token", "_cancel_requested", "cancel_token", "cancel_requested", "cancel_file", "cancel_path"):
            token = _srm_cancel_token_from_value(runtime.get(key))
            if token is not None:
                return token
    return None


def _solver_cancel_token_from_config(solver: Mapping[str, Any] | None) -> Any:
    if not isinstance(solver, Mapping):
        return None
    containers: list[Mapping[str, Any]] = [solver]
    for key in ("execution", "runtime", "run_context"):
        value = solver.get(key)
        if isinstance(value, Mapping):
            containers.append(value)
    for container in containers:
        for key in ("_cancel_token", "_cancel_requested", "cancel_token", "cancel_requested", "cancel_file", "cancel_path"):
            token = _srm_cancel_token_from_value(container.get(key))
            if token is not None:
                return token
    return None


def _raise_if_solver_cancel_requested(
    solver: Mapping[str, Any] | None,
    stage_name: str,
    checkpoint: str,
    diagnostics: Mapping[str, Any] | None = None,
    **fields: Any,
) -> None:
    token = _solver_cancel_token_from_config(solver)
    speculative_token = None
    if isinstance(solver, Mapping):
        speculative_token = _srm_cancel_token_from_value(
            solver.get("_srm_speculative_cancel_path")
        )
    user_cancel_requested = _srm_cancel_requested(token)
    speculative_cancel_requested = _srm_cancel_requested(speculative_token)
    if not user_cancel_requested and not speculative_cancel_requested:
        return
    payload = dict(diagnostics or {})
    payload.update(fields)
    payload["trial_status"] = "solver_cancelled"
    payload["failure_diagnostic_source"] = payload.get("failure_diagnostic_source", "solver_cancel")
    payload["solver_cancel_requested"] = True
    payload["solver_cancel_checkpoint"] = checkpoint
    payload["solver_cancel_scope"] = (
        "user"
        if user_cancel_requested
        else "speculative_trial_outside_decision_boundary"
    )
    payload["converged"] = False
    payload["diagnostic_summary"] = _srm_diagnostic_summary(payload)
    raise FEM2DError(f"{stage_name}: solver cancellation requested at {checkpoint}", diagnostics=payload)


def _srm_note_cancellation(
    stats: dict[str, Any] | None,
    *,
    requested: bool = True,
    skipped_count: int = 0,
    canceled_count: int = 0,
    running_after_cancel: int = 0,
    note: str = "",
) -> None:
    if stats is None:
        return
    stats["requested"] = bool(stats.get("requested", False) or requested)
    stats["skipped_count"] = int(stats.get("skipped_count", 0) or 0) + max(int(skipped_count or 0), 0)
    stats["canceled_count"] = int(stats.get("canceled_count", 0) or 0) + max(int(canceled_count or 0), 0)
    stats["running_after_cancel"] = max(int(stats.get("running_after_cancel", 0) or 0), max(int(running_after_cancel or 0), 0))
    if note:
        notes = stats.setdefault("notes", [])
        if isinstance(notes, list) and note not in notes:
            notes.append(note)


_SRM_PROCESS_TRIAL_SPEC: dict[str, Any] | None = None
_SRM_PROCESS_STEP_CACHE: StepCache2D | None = None
_SRM_PROCESS_TOPOLOGY_CACHE: _SRMTopologyDiagnosticsCache | None = None
_SRM_PROCESS_THREAD_CONTROL: dict[str, Any] | None = None


def _srm_process_trial_initializer(
    trial_spec: Mapping[str, Any], thread_control: Mapping[str, Any] | None
) -> None:
    global _SRM_PROCESS_TRIAL_SPEC
    global _SRM_PROCESS_STEP_CACHE
    global _SRM_PROCESS_TOPOLOGY_CACHE
    global _SRM_PROCESS_THREAD_CONTROL
    _SRM_PROCESS_TRIAL_SPEC = dict(trial_spec)
    _SRM_PROCESS_THREAD_CONTROL = dict(thread_control) if isinstance(thread_control, Mapping) else None
    mesh = _SRM_PROCESS_TRIAL_SPEC.get("mesh")
    _SRM_PROCESS_TOPOLOGY_CACHE = (
        _srm_topology_diagnostics_cache(mesh) if isinstance(mesh, Mesh2D) else None
    )
    _SRM_PROCESS_STEP_CACHE = None
    if _SRM_PROCESS_TRIAL_SPEC.get("kind") == "plane_strain" and isinstance(mesh, Mesh2D):
        _SRM_PROCESS_STEP_CACHE = _build_srm_factor_step_cache(
            mesh,
            _SRM_PROCESS_TRIAL_SPEC["materials"],
            _SRM_PROCESS_TRIAL_SPEC.get("boundary_conditions"),
            _SRM_PROCESS_TRIAL_SPEC.get("loads"),
            _SRM_PROCESS_TRIAL_SPEC.get("mpc_constraints"),
            solver=_SRM_PROCESS_TRIAL_SPEC.get("solver"),
            srm_cfg=_SRM_PROCESS_TRIAL_SPEC.get("srm_cfg", {}),
            interfaces=_SRM_PROCESS_TRIAL_SPEC.get("interfaces"),
            structural_elements=_SRM_PROCESS_TRIAL_SPEC.get("structural_elements"),
        )


def _srm_process_solve_trial(
    factor: float, *, solver_override: Mapping[str, Any] | None = None
) -> StageResult2D:
    spec = _SRM_PROCESS_TRIAL_SPEC
    if not isinstance(spec, Mapping):
        raise FEM2DError("SRM process worker was started without a trial specification")
    if spec.get("kind") != "plane_strain":
        raise FEM2DError(f"unsupported SRM process trial kind: {spec.get('kind', '')}")
    mesh = spec.get("mesh")
    if not isinstance(mesh, Mesh2D):
        raise FEM2DError("SRM process trial specification has no mesh")
    trial_solver = _srm_solver_with_retry_override(spec.get("solver"), solver_override)
    warm_start_displacement = _srm_initial_displacement_from_solver_override(solver_override)
    trial_mesh, trial_interfaces = _srm_trial_workspace(
        mesh,
        spec.get("interfaces"),
        independent_geometry=bool(spec.get("large_deformation_trials", False)),
    )
    return solve_plane_strain_stage(
        mesh=trial_mesh,
        materials=spec["materials"],
        boundary_conditions=spec.get("boundary_conditions"),
        loads=spec.get("loads"),
        mpc_constraints=spec.get("mpc_constraints"),
        stage_name=f"{spec.get('stage_name', 'srm')}-FS{factor:g}",
        output_dir=None,
        solver=trial_solver,
        initial_stresses=spec.get("initial_stresses"),
        interfaces=trial_interfaces,
        structural_elements=spec.get("structural_elements"),
        strength_factor=factor,
        plastic_state=spec.get("plastic_state"),
        initial_displacement=warm_start_displacement,
        step_cache=_SRM_PROCESS_STEP_CACHE,
        postprocess_results=not bool(spec.get("lightweight_trials", True)),
    )


def _srm_process_parallel_trial_record(
    factor: float,
    failure_plastic_ratio: float,
    queued_at: float,
    solver_override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    started_at = _perf_counter()
    with _srm_numeric_thread_context(_SRM_PROCESS_THREAD_CONTROL):
        record = _srm_trial_record(
            factor,
            failure_plastic_ratio,
            _srm_process_solve_trial,
            _SRM_PROCESS_TOPOLOGY_CACHE,
            solver_override=solver_override,
        )
    finished_at = _perf_counter()
    record["_parallel_queue_wait_seconds"] = max(started_at - queued_at, 0.0)
    record["_parallel_worker_elapsed_seconds"] = max(finished_at - started_at, 0.0)
    record["_parallel_executor"] = "process"
    record["_parallel_worker_pid"] = int(os.getpid())
    return record


def _srm_evaluate_records_parallel(
    factors: list[float],
    failure_plastic_ratio: float,
    solve_trial: Callable[[float], StageResult2D],
    workers: int,
    *,
    progress: Mapping[str, Any] | None = None,
    progress_stage_name: str = "",
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
    thread_control: Mapping[str, Any] | None = None,
    solver_override: Mapping[str, Any] | None = None,
    solver_overrides: Mapping[float, Mapping[str, Any] | None] | None = None,
    cancel_token: Any = None,
    cancellation_stats: dict[str, Any] | None = None,
    executor_kind: str = "thread",
    process_trial_spec: Mapping[str, Any] | None = None,
    decision_cancel_callback: Callable[
        [int, Mapping[str, Any], list[float]], Any
    ]
    | None = None,
) -> list[dict[str, Any]]:
    if _srm_cancel_requested(cancel_token):
        _srm_note_cancellation(
            cancellation_stats,
            skipped_count=len(factors),
            note="SRM parallel evaluation skipped because cancellation was requested before submission",
        )
        return []
    if len(factors) <= 1 or workers <= 1:
        records: list[dict[str, Any]] = []
        for factor in factors:
            if _srm_cancel_requested(cancel_token):
                _srm_note_cancellation(
                    cancellation_stats,
                    skipped_count=len(factors) - len(records),
                    note="SRM serial evaluation stopped before remaining canceled trials",
                )
                break
            records.append(
                _srm_trial_record(
                    float(factor),
                    failure_plastic_ratio,
                    solve_trial,
                    topology_cache,
                    solver_override=_srm_solver_override_for_factor(solver_override, solver_overrides, float(factor)),
                )
            )
        for index, record in enumerate(records, start=1):
            _srm_emit_progress(progress, stage_name=progress_stage_name, record=record, index=index, total=len(records), prefix="trial")
        return records
    requested_process = str(executor_kind or "thread").strip().lower() == "process"
    use_process = bool(requested_process and isinstance(process_trial_spec, Mapping))
    if requested_process and not use_process:
        _srm_note_cancellation(
            cancellation_stats,
            requested=False,
            note="SRM process executor unavailable for this trial type; thread executor used",
        )

    def run_parallel_batch(process_executor: bool) -> list[dict[str, Any]]:
        if process_executor:
            executor_context = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=_srm_process_trial_initializer,
                initargs=(dict(process_trial_spec or {}), dict(thread_control or {})),
            )
            numeric_context = None
        else:
            executor_context = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="geofem-srm"
            )
            numeric_context = _srm_numeric_thread_context(thread_control)

        if numeric_context is not None:
            numeric_context.__enter__()
        speculative_cancel_dir = (
            tempfile.TemporaryDirectory(prefix="geofem-srm-lookahead-")
            if decision_cancel_callback is not None
            else None
        )
        try:
            with executor_context as executor:
                futures = {}
                futures_by_index = {}
                speculative_cancel_paths: dict[int, Path] = {}
                decision_cancel_requested_indices: set[int] = set()
                skipped_before_submit = 0
                for index, factor in enumerate(factors):
                    if _srm_cancel_requested(cancel_token):
                        skipped_before_submit = len(factors) - index
                        break
                    factor_override = _srm_solver_override_for_factor(
                        solver_override, solver_overrides, float(factor)
                    )
                    if speculative_cancel_dir is not None:
                        cancel_path = (
                            Path(speculative_cancel_dir.name)
                            / f"trial-{index:04d}.cancel"
                        )
                        speculative_cancel_paths[index] = cancel_path
                        factor_override = _srm_merge_solver_overrides(
                            factor_override,
                            {
                                "_srm_speculative_cancel_path": str(
                                    cancel_path
                                )
                            },
                        )
                    if process_executor:
                        future = executor.submit(
                            _srm_process_parallel_trial_record,
                            float(factor),
                            failure_plastic_ratio,
                            _perf_counter(),
                            factor_override,
                        )
                    else:
                        future = executor.submit(
                            _srm_parallel_trial_record,
                            float(factor),
                            failure_plastic_ratio,
                            solve_trial,
                            topology_cache,
                            _perf_counter(),
                            factor_override,
                        )
                    futures[future] = index
                    futures_by_index[index] = future
                if skipped_before_submit:
                    _srm_note_cancellation(
                        cancellation_stats,
                        skipped_count=skipped_before_submit,
                        note="SRM parallel submission stopped before remaining canceled trials",
                    )
                records: list[dict[str, Any] | None] = [None] * len(factors)
                pending = set(futures)
                decision_cursor = 0

                def request_decision_cancellation(indices: Any) -> None:
                    if indices is None:
                        return
                    try:
                        requested_indices = list(indices)
                    except TypeError:
                        return
                    canceled_pending = 0
                    requested_now = 0
                    requested_factors: list[float] = []
                    for raw_index in requested_indices:
                        try:
                            target_index = int(raw_index)
                        except (TypeError, ValueError):
                            continue
                        if (
                            target_index < 0
                            or target_index >= len(factors)
                            or target_index in decision_cancel_requested_indices
                            or records[target_index] is not None
                        ):
                            continue
                        future_to_cancel = futures_by_index.get(target_index)
                        if future_to_cancel is None or future_to_cancel.done():
                            continue
                        decision_cancel_requested_indices.add(target_index)
                        requested_now += 1
                        requested_factors.append(float(factors[target_index]))
                        cancel_path = speculative_cancel_paths.get(target_index)
                        if cancel_path is not None:
                            cancel_path.touch(exist_ok=True)
                        if future_to_cancel.cancel():
                            canceled_pending += 1
                            pending.discard(future_to_cancel)
                    if not requested_now:
                        return
                    if cancellation_stats is not None:
                        cancellation_stats[
                            "decision_linked_requested_count"
                        ] = int(
                            cancellation_stats.get(
                                "decision_linked_requested_count", 0
                            )
                            or 0
                        ) + requested_now
                        factors_log = cancellation_stats.setdefault(
                            "decision_linked_requested_factors", []
                        )
                        if isinstance(factors_log, list):
                            for factor_value in requested_factors:
                                if factor_value not in factors_log:
                                    factors_log.append(factor_value)
                        cancellation_stats[
                            "decision_linked_pending_cancel_count"
                        ] = int(
                            cancellation_stats.get(
                                "decision_linked_pending_cancel_count", 0
                            )
                            or 0
                        ) + canceled_pending
                    _srm_note_cancellation(
                        cancellation_stats,
                        canceled_count=canceled_pending,
                        running_after_cancel=sum(
                            1
                            for pending_future in pending
                            if not pending_future.done()
                        ),
                        note=(
                            "SRM decision boundary was established; "
                            "out-of-bound speculative trials received safe-stop requests"
                        ),
                    )

                def process_ready_decisions() -> None:
                    nonlocal decision_cursor
                    if decision_cancel_callback is None:
                        return
                    while decision_cursor < len(factors):
                        if (
                            decision_cursor
                            in decision_cancel_requested_indices
                        ):
                            decision_cursor += 1
                            continue
                        decision_record = records[decision_cursor]
                        if decision_record is None:
                            break
                        requested = decision_cancel_callback(
                            decision_cursor,
                            decision_record,
                            factors,
                        )
                        decision_cursor += 1
                        request_decision_cancellation(requested)

                for future in as_completed(futures):
                    pending.discard(future)
                    if future.cancelled():
                        process_ready_decisions()
                        continue
                    index = futures[future]
                    record = future.result()
                    _srm_emit_progress(progress, stage_name=progress_stage_name, record=record, index=index + 1, total=len(factors), prefix="parallel-trial")
                    decision_cancelled = bool(
                        index in decision_cancel_requested_indices
                        and str(record.get("trial_status", ""))
                        == "solver_cancelled"
                        and str(record.get("solver_cancel_scope", ""))
                        == "speculative_trial_outside_decision_boundary"
                    )
                    if decision_cancelled:
                        if cancellation_stats is not None:
                            cancellation_stats[
                                "decision_linked_safe_stop_count"
                            ] = int(
                                cancellation_stats.get(
                                    "decision_linked_safe_stop_count", 0
                                )
                                or 0
                            ) + 1
                        _srm_note_cancellation(
                            cancellation_stats,
                            canceled_count=1,
                            note=(
                                "SRM out-of-bound speculative trial stopped "
                                "at an increment or Newton safe checkpoint"
                            ),
                        )
                    else:
                        records[index] = record
                        if (
                            index in decision_cancel_requested_indices
                            and cancellation_stats is not None
                        ):
                            cancellation_stats[
                                "decision_linked_completed_after_request_count"
                            ] = int(
                                cancellation_stats.get(
                                    "decision_linked_completed_after_request_count",
                                    0,
                                )
                                or 0
                            ) + 1
                    process_ready_decisions()
                    if _srm_cancel_requested(cancel_token):
                        canceled_now = 0
                        for pending_future in list(pending):
                            if pending_future.cancel():
                                canceled_now += 1
                                pending.discard(pending_future)
                        _srm_note_cancellation(
                            cancellation_stats,
                            canceled_count=canceled_now,
                            running_after_cancel=len(pending),
                            note="SRM cancellation requested; pending speculative trials were canceled when possible",
                        )
                        break
            return [record for record in records if record is not None]
        finally:
            if speculative_cancel_dir is not None:
                speculative_cancel_dir.cleanup()
            if numeric_context is not None:
                numeric_context.__exit__(None, None, None)

    try:
        records = run_parallel_batch(use_process)
        if cancellation_stats is not None:
            cancellation_stats["executor"] = "process" if use_process else "thread"
        return records
    except Exception as exc:
        if not use_process:
            raise
        if cancellation_stats is not None:
            cancellation_stats["process_executor_fallback"] = True
            cancellation_stats["process_executor_error"] = f"{type(exc).__name__}: {exc}"
        _srm_note_cancellation(
            cancellation_stats,
            requested=False,
            note=f"SRM process executor failed; serial fallback used: {type(exc).__name__}: {exc}",
        )
        return _srm_evaluate_records_parallel(
            factors,
            failure_plastic_ratio,
            solve_trial,
            1,
            progress=progress,
            progress_stage_name=progress_stage_name,
            topology_cache=topology_cache,
            thread_control=thread_control,
            solver_override=solver_override,
            solver_overrides=solver_overrides,
            cancel_token=cancel_token,
            cancellation_stats=cancellation_stats,
            executor_kind="thread",
        )


def _srm_solver_override_for_factor(
    default: Mapping[str, Any] | None,
    overrides: Mapping[float, Mapping[str, Any] | None] | None,
    factor: float,
) -> Mapping[str, Any] | None:
    if isinstance(overrides, Mapping):
        key = _round_srm_factor(float(factor))
        raw = overrides.get(key)
        if raw is None:
            raw = overrides.get(float(factor))
        if isinstance(raw, Mapping):
            return raw
    return default


def _srm_parallel_trial_record(
    factor: float,
    failure_plastic_ratio: float,
    solve_trial: Callable[[float], StageResult2D],
    topology_cache: _SRMTopologyDiagnosticsCache | None,
    queued_at: float,
    solver_override: Mapping[str, Any] | None,
) -> dict[str, Any]:
    started_at = _perf_counter()
    record = _srm_trial_record(
        factor,
        failure_plastic_ratio,
        solve_trial,
        topology_cache,
        solver_override=solver_override,
    )
    finished_at = _perf_counter()
    record["_parallel_queue_wait_seconds"] = max(started_at - queued_at, 0.0)
    record["_parallel_worker_elapsed_seconds"] = max(finished_at - started_at, 0.0)
    return record


def _run_srm_linear_trials(
    factors: list[float],
    failure_plastic_ratio: float,
    solve_trial: Callable[[float], StageResult2D],
    *,
    parallel: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
    progress_stage_name: str = "",
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
) -> tuple[StageResult2D | None, float, list[dict[str, Any]], dict[str, Any]]:
    parallel_cfg = parallel if isinstance(parallel, Mapping) else {"enabled": False, "max_workers": 1}
    if bool(parallel_cfg.get("enabled", False)):
        return _run_srm_linear_trials_parallel(
            factors,
            failure_plastic_ratio,
            solve_trial,
            parallel_cfg,
            progress=progress,
            progress_stage_name=progress_stage_name,
            topology_cache=topology_cache,
        )
    trials: list[dict[str, Any]] = []
    last_ok: StageResult2D | None = None
    last_trial: StageResult2D | None = None
    fos = 0.0
    for factor in factors:
        record = _srm_trial_record(float(factor), failure_plastic_ratio, solve_trial, topology_cache)
        public_record = _srm_public_trial_record(record)
        trials.append(public_record)
        trial = record.get("result")
        if isinstance(trial, StageResult2D):
            last_trial = trial
        if bool(record.get("ok", False)):
            if isinstance(trial, StageResult2D):
                last_ok = trial
            fos = float(record["factor"])
        _srm_emit_progress(progress, stage_name=progress_stage_name, record=record, index=len(trials), total=len(factors), current_fos=fos, prefix="trial")
        if not bool(record.get("ok", False)):
            break
    return last_ok or last_trial, fos, trials, {"search_mode": "linear", "trial_state": "independent_from_stage_start", **_srm_result_state(fos, trials)}


def _run_srm_linear_trials_parallel(
    factors: list[float],
    failure_plastic_ratio: float,
    solve_trial: Callable[[float], StageResult2D],
    parallel: Mapping[str, Any],
    *,
    progress: Mapping[str, Any] | None = None,
    progress_stage_name: str = "",
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
) -> tuple[StageResult2D | None, float, list[dict[str, Any]], dict[str, Any]]:
    workers = int(parallel.get("max_workers", 1) or 1)
    records = _srm_evaluate_records_parallel(
        factors,
        failure_plastic_ratio,
        solve_trial,
        workers,
        progress=progress,
        progress_stage_name=progress_stage_name,
        topology_cache=topology_cache,
        thread_control=parallel.get("thread_control") if isinstance(parallel, Mapping) else None,
    )
    trials: list[dict[str, Any]] = []
    last_ok: StageResult2D | None = None
    last_trial: StageResult2D | None = None
    fos = 0.0
    for record in records:
        trials.append(_srm_public_trial_record(record))
        trial_result = record.get("result")
        if isinstance(trial_result, StageResult2D):
            last_trial = trial_result
        if bool(record.get("ok", False)):
            if isinstance(trial_result, StageResult2D):
                last_ok = trial_result
            fos = float(record["factor"])
            continue
        break
    return last_ok or last_trial, fos, trials, {
        "search_mode": "linear",
        "trial_state": "independent_from_stage_start",
        "parallel": {
            **dict(parallel),
            "evaluated_trials": len(records),
            "reported_trials": len(trials),
            "strategy": "ordered_explicit_factor_batch",
        },
        **_srm_result_state(fos, trials),
    }


def _run_srm_adaptive_bracket_trials(
    factors: list[float],
    srm_cfg: Mapping[str, Any],
    failure_plastic_ratio: float,
    solve_trial: Callable[[float], StageResult2D],
    *,
    parallel: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
    progress_stage_name: str = "",
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
    warm_start_supported: bool = True,
    process_trial_spec: Mapping[str, Any] | None = None,
) -> tuple[StageResult2D | None, float, list[dict[str, Any]], dict[str, Any]]:
    anchor = float(srm_cfg.get("anchor_factor", srm_cfg.get("anchor", 1.0)))
    if anchor <= 0.0:
        raise FEM2DError("srm anchor_factor must be positive")
    base_candidates = {_round_srm_factor(value) for value in factors if float(value) > 0.0}
    explicit_factors = srm_cfg.get("factors") is not None
    if explicit_factors:
        ordered = sorted(value for value in base_candidates if value > 0.0)
        if ordered and _round_srm_factor(anchor) not in set(ordered):
            anchor = ordered[0]
        lower_values = sorted([value for value in ordered if value < anchor], reverse=True)
        upper_values = sorted([value for value in ordered if value > anchor])
    else:
        base_candidates.add(_round_srm_factor(anchor))
        lower_values, upper_values = _srm_two_branch_values(sorted(base_candidates), srm_cfg, anchor)
        base_candidates.update(lower_values)
        base_candidates.update(upper_values)
        ordered = sorted(value for value in base_candidates if value > 0.0)
    auto_enabled = _srm_auto_enabled(srm_cfg)
    if len(ordered) <= 2 and not auto_enabled:
        result, fos, trials, info = _run_srm_linear_trials(
            ordered,
            failure_plastic_ratio,
            solve_trial,
            parallel=parallel,
            progress=progress,
            progress_stage_name=progress_stage_name,
            topology_cache=topology_cache,
        )
        info["search_mode"] = "adaptive_bracket"
        info["adaptive_degenerated_to_linear"] = True
        return result, fos, trials, info
    factor_tol = float(srm_cfg.get("factor_tol", srm_cfg.get("tolerance", srm_cfg.get("tol", 0.005))))
    if factor_tol <= 0.0:
        raise FEM2DError("srm factor_tol must be positive")
    max_bisection = int(srm_cfg.get("max_bisection", srm_cfg.get("bisection_iterations", 8)))
    if max_bisection < 0:
        raise FEM2DError("srm max_bisection must be non-negative")
    auto_settings = _srm_auto_settings(srm_cfg) if auto_enabled else {"enabled": False}
    warm_start_settings = _srm_warm_start_settings(srm_cfg, supported=warm_start_supported)
    stride = _srm_adaptive_bracket_stride(ordered, srm_cfg)
    lower_scan_factors = _srm_adaptive_branch_scan_factors(lower_values, stride)
    upper_scan_factors = _srm_adaptive_branch_scan_factors(upper_values, stride)

    trials: list[dict[str, Any]] = []
    records: dict[float, dict[str, Any]] = {}
    published: set[int] = set()
    published_rows: dict[int, dict[str, Any]] = {}
    retry_counts: dict[float, int] = {}
    last_ok: StageResult2D | None = None
    last_trial: StageResult2D | None = None
    stable_reference_record: dict[str, Any] | None = None
    adaptive_increment_reference_record: dict[str, Any] | None = None
    fos = 0.0
    boundary_verification_strategy = str(
        auto_settings.get("boundary_verification_strategy", "immediate")
    )
    deferred_boundary_verification_enabled = bool(
        auto_enabled
        and auto_settings.get("boundary_verification_enabled", False)
        and boundary_verification_strategy == "deferred_final"
    )
    boundary_verification_deferred_count = 0
    boundary_verification_final_count = 0
    boundary_verification_recovery_count = 0
    boundary_verification_stable_reversal_count = 0
    boundary_verification_cold_retry_count = 0
    boundary_verification_cold_retry_factors: dict[float, int] = {}
    factor_tol_enforcement_enabled = bool(
        auto_enabled
        and auto_settings.get("factor_tol_enforcement_enabled", False)
    )
    factor_tol_accept_verified_numerical_failure = bool(
        factor_tol_enforcement_enabled
        and auto_settings.get(
            "factor_tol_enforcement_accept_verified_numerical_failure", False
        )
    )
    factor_tol_max_extra_bisections = (
        max(
            0,
            int(
                auto_settings.get(
                    "factor_tol_enforcement_max_extra_bisections", 8
                )
                or 0
            ),
        )
        if factor_tol_enforcement_enabled
        else 0
    )
    factor_tol_extra_bisections_used = 0
    factor_tol_numerical_failure_factors: set[float] = set()
    parallel_cfg = parallel if isinstance(parallel, Mapping) else {"enabled": False, "max_workers": 1}
    parallel_enabled = bool(parallel_cfg.get("enabled", False))
    workers = max(1, int(parallel_cfg.get("max_workers", 1) or 1))
    requested_executor_kind = str(parallel_cfg.get("executor", "thread") or "thread").strip().lower()
    effective_executor_kind = (
        "process"
        if requested_executor_kind == "process" and isinstance(process_trial_spec, Mapping)
        else "thread"
    )
    process_executor_disabled_reason = (
        ""
        if requested_executor_kind != "process" or effective_executor_kind == "process"
        else "process_trial_spec_unavailable"
    )
    parallel_strategy = str(parallel_cfg.get("strategy", "") or "").strip().lower().replace("-", "_")
    preserve_order = _srm_bool(parallel_cfg.get("preserve_decision_order", True), True)
    speculative_cancel_token = _srm_cancel_token_from_config(srm_cfg, parallel_cfg)
    lookahead_depth = _srm_parallel_lookahead_depth(srm_cfg, parallel_cfg, workers)
    lookahead_enabled = bool(parallel_enabled and preserve_order and parallel_strategy in {"lookahead", "adaptive_lookahead"} and workers > 1 and lookahead_depth > 1)
    decision_linked_cancellation_enabled = bool(
        lookahead_enabled
        and _srm_bool(
            parallel_cfg.get(
                "decision_linked_cancellation",
                srm_cfg.get("decision_linked_cancellation", True),
            ),
            True,
        )
    )
    bisection_speculation_enabled = bool(
        lookahead_enabled
        and workers >= 3
        and _srm_bool(
            parallel_cfg.get(
                "bisection_speculation",
                parallel_cfg.get("speculative_bisection", srm_cfg.get("bisection_speculation", True)),
            ),
            True,
        )
    )
    cost_aware_lookahead_enabled = bool(
        lookahead_enabled
        and _srm_bool(
            parallel_cfg.get(
                "cost_aware_lookahead",
                srm_cfg.get("cost_aware_lookahead", True),
            ),
            True,
        )
    )
    cost_aware_min_elapsed_seconds = max(
        0.0,
        float(parallel_cfg.get("cost_aware_min_elapsed_seconds", 30.0) or 0.0),
    )
    cost_aware_failure_ratio = max(
        1.0,
        float(parallel_cfg.get("cost_aware_failure_ratio", 3.0) or 3.0),
    )
    cost_aware_escalation_ratio = max(
        1.0,
        float(parallel_cfg.get("cost_aware_escalation_ratio", 4.0) or 4.0),
    )
    cost_aware_min_samples = max(
        1,
        int(parallel_cfg.get("cost_aware_min_samples", 2) or 2),
    )
    event_driven_cost_cancellation_enabled = _srm_bool(
        parallel_cfg.get("event_driven_cost_cancellation", True),
        True,
    )
    speculative_factor_keys: set[float] = set()
    used_speculative_keys: set[float] = set()
    projection_speculative_keys: set[float] = set()
    bisection_speculative_keys: set[float] = set()
    window_evaluated_trials = 0
    speculative_prefetch_call_count = 0
    speculative_prefetch_wall_elapsed_seconds = 0.0
    speculative_trial_elapsed_seconds = 0.0
    speculative_queue_wait_elapsed_seconds = 0.0
    speculative_worker_elapsed_seconds = 0.0
    speculative_estimated_wall_clock_saving_seconds = 0.0
    canceled_speculative_trial_count = 0
    speculative_cancellation_requested = False
    speculative_cancellation_notes: list[str] = []
    decision_linked_requested_count = 0
    decision_linked_pending_cancel_count = 0
    decision_linked_safe_stop_count = 0
    decision_linked_completed_after_request_count = 0
    decision_linked_requested_factors: set[float] = set()
    process_executor_fallback_count = 0
    process_executor_errors: list[str] = []
    cost_aware_depth_limited_count = 0
    cost_aware_asymmetric_bisection_count = 0
    cost_aware_deferred_candidate_count = 0
    event_driven_cost_shrink_count = 0
    event_driven_cost_cancel_candidate_count = 0

    def cost_snapshot() -> dict[str, Any]:
        return _srm_cost_aware_lookahead_snapshot(
            records,
            enabled=cost_aware_lookahead_enabled,
            min_elapsed_seconds=cost_aware_min_elapsed_seconds,
            failure_ratio_threshold=cost_aware_failure_ratio,
            escalation_ratio_threshold=cost_aware_escalation_ratio,
            min_samples=cost_aware_min_samples,
        )

    def scan_window_depth() -> int:
        nonlocal cost_aware_depth_limited_count
        if not lookahead_enabled:
            return 1
        snapshot = cost_snapshot()
        recommended = max(
            1,
            min(
                lookahead_depth,
                int(snapshot.get("recommended_depth", lookahead_depth)),
            ),
        )
        if recommended < lookahead_depth:
            cost_aware_depth_limited_count += 1
        return recommended

    def publish(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal fos, last_ok, last_trial, stable_reference_record
        factor = _round_srm_factor(float(record["factor"]))
        record_key = id(record)
        if record_key in published:
            return record
        trial_result = record.get("result")
        if isinstance(trial_result, StageResult2D):
            last_trial = trial_result
        if bool(record.get("ok", False)):
            if isinstance(trial_result, StageResult2D):
                last_ok = trial_result
            fos = factor
            stable_reference_record = record
        public_record = _srm_public_trial_record(record)
        trials.append(public_record)
        published_rows[record_key] = public_record
        published.add(record_key)
        _srm_emit_progress(progress, stage_name=progress_stage_name, record=record, index=len(trials), total=None, current_fos=fos, prefix="adaptive-trial")
        return record

    def sync_published(record: Mapping[str, Any]) -> None:
        public_record = published_rows.get(id(record))
        if public_record is not None:
            public_record.update(_srm_public_trial_record(record))

    def defer_boundary_verification(record: dict[str, Any], reason: str) -> None:
        nonlocal boundary_verification_deferred_count
        if bool(record.get("boundary_verification_pending", False)):
            return
        record.update(
            {
                "boundary_verification_pending": True,
                "boundary_verification_deferred": True,
                "boundary_verification_deferred_reason": str(reason),
                "boundary_verification_result": "deferred",
            }
        )
        boundary_verification_deferred_count += 1

    def verify_boundary_record(
        source_record: dict[str, Any], *, trigger: str
    ) -> dict[str, Any]:
        nonlocal adaptive_increment_reference_record
        nonlocal boundary_verification_final_count
        nonlocal boundary_verification_stable_reversal_count
        key = _round_srm_factor(float(source_record["factor"]))
        if bool(source_record.get("boundary_verification", False)):
            return source_record
        reason = str(source_record.get("boundary_verification_deferred_reason", "") or "")
        if not reason:
            reason = _srm_boundary_verification_reason(source_record, auto_settings)
        if not reason:
            return source_record
        source_record["boundary_verification_pending"] = False
        source_record["boundary_verification_superseded"] = True
        source_record["boundary_verification_result"] = "scheduled"
        verification_record = _srm_trial_record(
            key,
            failure_plastic_ratio,
            solve_trial,
            topology_cache,
            solver_override=_srm_boundary_verification_solver_override(
                auto_settings,
                key,
                reason,
                checkpoint=source_record.get(
                    "_increment_continuation_checkpoint"
                )
                if isinstance(
                    source_record.get("_increment_continuation_checkpoint"),
                    _IncrementContinuationCheckpoint,
                )
                else None,
                source_record=source_record,
            ),
        )
        _srm_auto_classify_trial(
            verification_record, auto_settings, stable_reference_record
        )
        verification_record["_auto_finalized"] = True
        verification_record["boundary_verification_trigger"] = str(trigger)
        verification_record["boundary_verification_result"] = verification_record.get(
            "srm_trial_state", "indeterminate"
        )
        publish(verification_record)
        records[key] = verification_record
        boundary_verification_final_count += 1
        if bool(verification_record.get("ok", False)):
            boundary_verification_stable_reversal_count += 1
        else:
            adaptive_increment_reference_record = verification_record
        return verification_record

    def cold_verify_indeterminate(
        source_record: dict[str, Any], *, trigger: str
    ) -> dict[str, Any]:
        """Retry a decision-blocking indeterminate factor once from stage start."""

        nonlocal adaptive_increment_reference_record
        nonlocal boundary_verification_cold_retry_count
        nonlocal boundary_verification_stable_reversal_count
        if not _srm_bool(
            auto_settings.get(
                "boundary_verification_cold_retry_on_indeterminate", True
            ),
            True,
        ):
            return source_record
        key = _round_srm_factor(float(source_record["factor"]))
        max_per_factor = max(
            0,
            _srm_auto_int(
                auto_settings,
                "boundary_verification_cold_retry_max_per_factor",
                1,
            ),
        )
        retry_count = int(boundary_verification_cold_retry_factors.get(key, 0))
        if max_per_factor <= 0 or retry_count >= max_per_factor:
            return source_record
        if _srm_trial_decision_state(
            source_record, auto_enabled=auto_enabled
        ) != "indeterminate":
            return source_record

        boundary_verification_cold_retry_factors[key] = retry_count + 1
        source_record["boundary_verification_cold_retry_superseded"] = True
        sync_published(source_record)
        cold_record = _srm_trial_record(
            key,
            failure_plastic_ratio,
            solve_trial,
            topology_cache,
            solver_override=_srm_boundary_verification_solver_override(
                auto_settings,
                key,
                f"cold_indeterminate:{trigger}",
                checkpoint=None,
                source_record=source_record,
            ),
        )
        _srm_auto_classify_trial(
            cold_record, auto_settings, stable_reference_record
        )
        cold_record.update(
            {
                "_auto_finalized": True,
                "boundary_verification_cold_retry": True,
                "boundary_verification_cold_retry_trigger": str(trigger),
                "boundary_verification_cold_retry_source_factor": key,
                "boundary_verification_result": cold_record.get(
                    "srm_trial_state", "indeterminate"
                ),
            }
        )
        publish(cold_record)
        records[key] = cold_record
        boundary_verification_cold_retry_count += 1
        if bool(cold_record.get("ok", False)):
            boundary_verification_stable_reversal_count += 1
        else:
            adaptive_increment_reference_record = cold_record
        return cold_record

    def evaluate(factor: float) -> dict[str, Any]:
        nonlocal adaptive_increment_reference_record
        key = _round_srm_factor(factor)
        record = records.get(key)
        if record is None:
            solver_override = _srm_merge_solver_overrides(
                _srm_auto_trial_solver_override(auto_settings, adaptive_increment_reference_record, key)
                if auto_enabled
                else None,
                _srm_warm_start_solver_override(warm_start_settings, stable_reference_record, key),
            )
            record = _srm_trial_record(key, failure_plastic_ratio, solve_trial, topology_cache, solver_override=solver_override)
        elif key in speculative_factor_keys:
            used_speculative_keys.add(key)
        if auto_enabled and not bool(record.get("_auto_finalized", False)):
            _srm_auto_classify_trial(record, auto_settings, stable_reference_record)
            record["_auto_finalized"] = True
            retry_count = retry_counts.get(key, 0)
            retry_planned = _srm_auto_should_retry(record, auto_settings, retry_count)
            verification_reason = ""
            if retry_planned:
                record["auto_superseded_by_retry"] = True
                record["auto_retry_planned"] = True
            else:
                verification_reason = _srm_boundary_verification_reason(record, auto_settings)
                if (
                    verification_reason
                    and deferred_boundary_verification_enabled
                    and str(record.get("auto_decision", "")) == "confirmed_failure"
                ):
                    defer_boundary_verification(record, verification_reason)
                    verification_reason = ""
                elif verification_reason:
                    record["boundary_verification_superseded"] = True
                    record["boundary_verification_result"] = "scheduled"
            publish(record)
            if retry_planned:
                retry_index = retry_count + 1
                retry_counts[key] = retry_index
                retry_record = _srm_trial_record(
                    key,
                    failure_plastic_ratio,
                    solve_trial,
                    topology_cache,
                    solver_override=_srm_auto_retry_solver_override(
                        auto_settings,
                        retry_index,
                        checkpoint=record.get(
                            "_increment_continuation_checkpoint"
                        )
                        if isinstance(
                            record.get("_increment_continuation_checkpoint"),
                            _IncrementContinuationCheckpoint,
                        )
                        else None,
                        source_record=record,
                    ),
                    retry_of=key,
                    retry_index=retry_index,
                )
                retry_record["auto_retry_reason"] = record.get("auto_decision_reason", "")
                _srm_auto_classify_trial(retry_record, auto_settings, stable_reference_record)
                retry_record["_auto_finalized"] = True
                record["auto_retry_result"] = retry_record.get("auto_decision", "")
                verification_reason = _srm_boundary_verification_reason(retry_record, auto_settings)
                if (
                    verification_reason
                    and deferred_boundary_verification_enabled
                    and str(retry_record.get("auto_decision", "")) == "confirmed_failure"
                ):
                    defer_boundary_verification(retry_record, verification_reason)
                    verification_reason = ""
                elif verification_reason:
                    retry_record["boundary_verification_superseded"] = True
                    retry_record["boundary_verification_result"] = "scheduled"
                publish(retry_record)
                record = retry_record
            if verification_reason:
                record["boundary_verification_deferred_reason"] = verification_reason
                record = verify_boundary_record(record, trigger="immediate")
            if not bool(record.get("ok", False)):
                adaptive_increment_reference_record = record
            records[key] = record
        else:
            publish(record)
            if auto_enabled and not bool(record.get("ok", False)):
                adaptive_increment_reference_record = record
        records[key] = record
        return record

    def prefetch(
        candidate_factors: list[float],
        *,
        decision_policy: str = "none",
    ) -> None:
        nonlocal window_evaluated_trials
        nonlocal speculative_prefetch_call_count, speculative_prefetch_wall_elapsed_seconds
        nonlocal speculative_trial_elapsed_seconds, speculative_queue_wait_elapsed_seconds
        nonlocal speculative_worker_elapsed_seconds, speculative_estimated_wall_clock_saving_seconds
        nonlocal canceled_speculative_trial_count, speculative_cancellation_requested
        nonlocal decision_linked_requested_count
        nonlocal decision_linked_pending_cancel_count
        nonlocal decision_linked_safe_stop_count
        nonlocal decision_linked_completed_after_request_count
        nonlocal process_executor_fallback_count
        nonlocal event_driven_cost_shrink_count
        nonlocal event_driven_cost_cancel_candidate_count
        if not lookahead_enabled:
            return
        normalized = [_round_srm_factor(factor) for factor in candidate_factors if float(factor) > 0.0]
        missing = [factor for factor in normalized if factor not in records]
        if len(missing) <= 1:
            return
        wall_start = _perf_counter()
        solver_overrides = {
            factor: _srm_merge_solver_overrides(
                _srm_auto_trial_solver_override(auto_settings, adaptive_increment_reference_record, factor)
                if auto_enabled
                else None,
                _srm_warm_start_solver_override(warm_start_settings, stable_reference_record, factor),
            )
            for factor in missing
        }
        solver_override = _srm_auto_trial_solver_override(auto_settings) if auto_enabled and not solver_overrides else None
        cancellation_stats: dict[str, Any] = {}
        preview_records: dict[float, dict[str, Any]] = {}

        def decision_cancel_callback(
            index: int,
            raw_record: Mapping[str, Any],
            batch_factors: list[float],
        ) -> list[int]:
            nonlocal event_driven_cost_shrink_count
            nonlocal event_driven_cost_cancel_candidate_count
            if not decision_linked_cancellation_enabled or decision_policy == "none":
                return []
            preview_record = dict(raw_record)
            if auto_enabled:
                _srm_auto_classify_trial(
                    preview_record,
                    auto_settings,
                    stable_reference_record,
                )
            state = _srm_trial_decision_state(
                preview_record,
                auto_enabled=auto_enabled,
            )
            factor_key = _round_srm_factor(
                float(preview_record.get("factor", batch_factors[index]))
            )
            preview_records[factor_key] = preview_record
            requested: set[int] = set()
            if decision_policy == "upper_scan":
                if state == "confirmed_failure":
                    requested.update(range(index + 1, len(batch_factors)))
            elif decision_policy == "lower_scan":
                if state == "stable":
                    requested.update(range(index + 1, len(batch_factors)))
            if decision_policy == "bisection" and index == 0:
                pivot = float(batch_factors[0])
                if state == "stable":
                    requested.update(
                        candidate_index
                        for candidate_index, candidate in enumerate(
                            batch_factors
                        )
                        if float(candidate) < pivot
                    )
                if state == "confirmed_failure":
                    requested.update(
                        candidate_index
                        for candidate_index, candidate in enumerate(
                            batch_factors
                        )
                        if float(candidate) > pivot
                    )

            if event_driven_cost_cancellation_enabled:
                event_snapshot = _srm_cost_aware_lookahead_snapshot(
                    {**records, **preview_records},
                    enabled=cost_aware_lookahead_enabled,
                    min_elapsed_seconds=cost_aware_min_elapsed_seconds,
                    failure_ratio_threshold=cost_aware_failure_ratio,
                    escalation_ratio_threshold=cost_aware_escalation_ratio,
                    min_samples=cost_aware_min_samples,
                )
                recommended = max(
                    1,
                    min(
                        len(batch_factors),
                        int(
                            event_snapshot.get(
                                "recommended_depth", len(batch_factors)
                            )
                        ),
                    ),
                )
                event_candidates: set[int] = set()
                if decision_policy in {"upper_scan", "lower_scan"}:
                    keep_through = index + max(recommended - 1, 0)
                    event_candidates.update(
                        range(keep_through + 1, len(batch_factors))
                    )
                elif decision_policy == "bisection" and index == 0 and recommended <= 1:
                    event_candidates.update(range(1, len(batch_factors)))
                new_event_candidates = event_candidates - requested
                if new_event_candidates:
                    event_driven_cost_shrink_count += 1
                    event_driven_cost_cancel_candidate_count += len(
                        new_event_candidates
                    )
                    requested.update(new_event_candidates)
            return sorted(requested)

        prefetched = _srm_evaluate_records_parallel(
            missing,
            failure_plastic_ratio,
            solve_trial,
            workers,
            topology_cache=topology_cache,
            thread_control=parallel_cfg.get("thread_control") if isinstance(parallel_cfg, Mapping) else None,
            solver_override=solver_override,
            solver_overrides=solver_overrides,
            cancel_token=speculative_cancel_token,
            cancellation_stats=cancellation_stats,
            executor_kind=effective_executor_kind,
            process_trial_spec=process_trial_spec,
            decision_cancel_callback=(
                decision_cancel_callback
                if decision_linked_cancellation_enabled
                and decision_policy != "none"
                else None
            ),
        )
        wall_elapsed = max(_perf_counter() - wall_start, 0.0)
        canceled_now = int(cancellation_stats.get("canceled_count", 0) or 0) + int(cancellation_stats.get("skipped_count", 0) or 0)
        if canceled_now:
            canceled_speculative_trial_count += canceled_now
        if bool(cancellation_stats.get("requested", False)):
            speculative_cancellation_requested = True
        decision_linked_requested_count += int(
            cancellation_stats.get("decision_linked_requested_count", 0) or 0
        )
        decision_linked_pending_cancel_count += int(
            cancellation_stats.get(
                "decision_linked_pending_cancel_count", 0
            )
            or 0
        )
        decision_linked_safe_stop_count += int(
            cancellation_stats.get("decision_linked_safe_stop_count", 0) or 0
        )
        decision_linked_completed_after_request_count += int(
            cancellation_stats.get(
                "decision_linked_completed_after_request_count", 0
            )
            or 0
        )
        requested_factors = cancellation_stats.get(
            "decision_linked_requested_factors", []
        )
        if isinstance(requested_factors, list):
            decision_linked_requested_factors.update(
                _round_srm_factor(float(value))
                for value in requested_factors
            )
        if bool(cancellation_stats.get("process_executor_fallback", False)):
            process_executor_fallback_count += 1
            error_text = str(cancellation_stats.get("process_executor_error", "") or "")
            if error_text and error_text not in process_executor_errors:
                process_executor_errors.append(error_text)
        notes = cancellation_stats.get("notes", [])
        if isinstance(notes, list):
            for note in notes:
                note_text = str(note)
                if note_text and note_text not in speculative_cancellation_notes:
                    speculative_cancellation_notes.append(note_text)
        for record in prefetched:
            key = _round_srm_factor(float(record["factor"]))
            record["_speculative"] = True
            records[key] = record
            speculative_factor_keys.add(key)
        window_evaluated_trials += len(prefetched)
        speculative_prefetch_call_count += 1
        speculative_prefetch_wall_elapsed_seconds += wall_elapsed
        trial_elapsed = sum(max(_srm_record_float(record, "elapsed_seconds", 0.0), 0.0) for record in prefetched)
        queue_wait = sum(max(_srm_record_float(record, "_parallel_queue_wait_seconds", 0.0), 0.0) for record in prefetched)
        worker_elapsed = sum(max(_srm_record_float(record, "_parallel_worker_elapsed_seconds", 0.0), 0.0) for record in prefetched)
        speculative_trial_elapsed_seconds += trial_elapsed
        speculative_queue_wait_elapsed_seconds += queue_wait
        speculative_worker_elapsed_seconds += worker_elapsed
        speculative_estimated_wall_clock_saving_seconds += max(trial_elapsed - wall_elapsed, 0.0)

    bracket_low: float | None = None
    bracket_high: float | None = None
    bounded_by = "none"
    scan_direction = "upper"
    lower_projection_info: dict[str, Any] = {"enabled": False, "used": False}
    indeterminate_factor_keys: set[float] = set()

    def decision_state(record: Mapping[str, Any]) -> str:
        state = _srm_trial_decision_state(record, auto_enabled=auto_enabled)
        if state == "indeterminate":
            indeterminate_factor_keys.add(_round_srm_factor(float(record["factor"])))
        return state

    def accept_verified_numerical_failure(
        record: dict[str, Any],
    ) -> bool:
        if (
            not factor_tol_accept_verified_numerical_failure
            or bool(record.get("ok", False))
            or not bool(record.get("boundary_verification", False))
            or bool(record.get("solver_cancel_requested", False))
        ):
            return False
        failure_reason = str(record.get("failure_reason", "") or "").strip()
        if failure_reason not in {
            "nonconvergence",
            "plastic_divergence",
            "invalid_detJ",
            "excessive_displacement",
        }:
            return False
        require_physical_evidence = _srm_bool(
            auto_settings.get(
                "factor_tol_require_physical_failure_evidence", True
            ),
            True,
        )
        physical_evidence, evidence_reason = (
            _srm_verified_failure_has_physical_evidence(record, auto_settings)
        )
        if require_physical_evidence and not physical_evidence:
            record.update(
                {
                    "factor_tol_numerical_failure_rejected": True,
                    "factor_tol_enforcement_original_state": (
                        _srm_trial_decision_state(
                            record, auto_enabled=auto_enabled
                        )
                    ),
                    "factor_tol_enforcement_reason": evidence_reason,
                }
            )
            sync_published(record)
            return False
        factor = _round_srm_factor(float(record["factor"]))
        original_state = _srm_trial_decision_state(
            record, auto_enabled=auto_enabled
        )
        record.update(
            {
                "factor_tol_numerical_failure_boundary": True,
                "factor_tol_enforcement_original_state": original_state,
                "factor_tol_enforcement_reason": (
                    f"verified_{failure_reason}:{evidence_reason}"
                ),
                "factor_tol_physical_failure_evidence": bool(
                    physical_evidence
                ),
            }
        )
        indeterminate_factor_keys.discard(factor)
        factor_tol_numerical_failure_factors.add(factor)
        sync_published(record)
        return True

    def bisect_bracket(
        ok_factor: float, fail_factor: float
    ) -> tuple[float, float, float | None]:
        nonlocal boundary_verification_recovery_count
        nonlocal factor_tol_extra_bisections_used
        low = min(ok_factor, fail_factor)
        high = max(ok_factor, fail_factor)
        ok_side = ok_factor
        fail_side = fail_factor
        failure_candidates: set[float] = {_round_srm_factor(fail_factor)}
        remaining_bisections = (
            max_bisection + factor_tol_max_extra_bisections
        )
        bisections_used = 0
        max_recoveries = max(
            0,
            _srm_auto_int(auto_settings, "boundary_verification_max_recoveries", 4),
        )

        def prefetch_bisection(mid: float) -> None:
            nonlocal cost_aware_asymmetric_bisection_count
            nonlocal cost_aware_deferred_candidate_count
            if not bisection_speculation_enabled:
                return
            mid_key = _round_srm_factor(mid)
            if mid_key in records:
                return
            candidates = [mid_key]
            snapshot = cost_snapshot()
            avoid_failure_side = bool(
                snapshot.get("failure_side_expensive", False)
            )
            ok_next = _round_srm_factor(0.5 * (mid_key + high))
            if (
                not avoid_failure_side
                and high - mid_key > factor_tol
                and mid_key < ok_next < high
            ):
                candidates.append(ok_next)
            elif (
                avoid_failure_side
                and high - mid_key > factor_tol
                and mid_key < ok_next < high
            ):
                cost_aware_deferred_candidate_count += 1
            fail_next = _round_srm_factor(0.5 * (low + mid_key))
            if mid_key - low > factor_tol and low < fail_next < mid_key:
                candidates.append(fail_next)
            if avoid_failure_side and len(candidates) > 1:
                cost_aware_asymmetric_bisection_count += 1
            before = set(speculative_factor_keys)
            prefetch(candidates, decision_policy="bisection")
            bisection_speculative_keys.update(
                key for key in speculative_factor_keys if key not in before and key in {_round_srm_factor(value) for value in candidates}
            )

        while True:
            while remaining_bisections > 0 and high - low > factor_tol + 1.0e-12:
                mid = _round_srm_factor(0.5 * (low + high))
                if mid <= low or mid >= high:
                    break
                prefetch_bisection(mid)
                record = evaluate(mid)
                remaining_bisections -= 1
                bisections_used += 1
                if bisections_used > max_bisection:
                    factor_tol_extra_bisections_used += 1
                if (
                    bool(record.get("boundary_verification_pending", False))
                    and _srm_record_int(record, "auto_failure_score", 0)
                    < _srm_auto_int(
                        auto_settings,
                        "boundary_verification_defer_min_failure_score",
                        6,
                    )
                ):
                    record = verify_boundary_record(
                        record, trigger="weak_provisional_failure"
                    )
                state = decision_state(record)
                if (
                    state == "indeterminate"
                    and factor_tol_enforcement_enabled
                    and not bool(record.get("boundary_verification", False))
                ):
                    record = verify_boundary_record(
                        record, trigger="factor_tol_enforcement"
                    )
                    state = decision_state(record)
                if (
                    state == "indeterminate"
                    and accept_verified_numerical_failure(record)
                ):
                    state = "confirmed_failure"
                if state == "indeterminate":
                    record = cold_verify_indeterminate(
                        record, trigger="bisection_boundary"
                    )
                    state = decision_state(record)
                    if accept_verified_numerical_failure(record):
                        state = "confirmed_failure"
                if state == "stable":
                    ok_side = mid
                    low = mid
                elif state == "confirmed_failure":
                    fail_side = mid
                    high = mid
                    failure_candidates.add(mid)
                else:
                    break

            failed_record = records.get(_round_srm_factor(fail_side))
            if not isinstance(failed_record, dict) or not bool(
                failed_record.get("boundary_verification_pending", False)
            ):
                return float(ok_side), float(ok_side), float(fail_side)

            verified_record = verify_boundary_record(
                failed_record, trigger="final_failed_endpoint"
            )
            verified_state = decision_state(verified_record)
            if verified_state == "confirmed_failure":
                return float(ok_side), float(ok_side), float(fail_side)

            outer_candidates = sorted(
                candidate for candidate in failure_candidates if candidate > fail_side
            )
            if verified_state == "stable":
                ok_side = fail_side
                low = fail_side
            if not outer_candidates:
                return float(ok_side), float(ok_side), None

            boundary_verification_recovery_count += 1
            fail_side = outer_candidates[0]
            high = fail_side
            if boundary_verification_recovery_count <= max_recoveries:
                continue

            # Recovery bisection is bounded, but an unverified provisional failure
            # must never be reported as a certified boundary.
            for candidate in outer_candidates:
                candidate_record = records.get(_round_srm_factor(candidate))
                if not isinstance(candidate_record, dict):
                    continue
                if bool(candidate_record.get("boundary_verification_pending", False)):
                    candidate_record = verify_boundary_record(
                        candidate_record, trigger="recovery_outer_boundary"
                    )
                candidate_state = decision_state(candidate_record)
                if candidate_state == "confirmed_failure":
                    return float(ok_side), float(ok_side), float(candidate)
                if candidate_state == "stable":
                    ok_side = candidate
                    low = candidate
            return float(ok_side), float(ok_side), None

    def scan_upper_for_failure(
        stable_factor: float,
    ) -> tuple[float, float | None]:
        """Continue the coarse upper scan above a verified stable factor."""

        ok_factor = float(stable_factor)
        index = 0
        while index < len(upper_scan_factors):
            active_depth = scan_window_depth()
            raw_window = upper_scan_factors[index : index + active_depth]
            index += max(len(raw_window), 1)
            window = [
                factor
                for factor in raw_window
                if factor > ok_factor + 1.0e-12
            ]
            if not window:
                continue
            prefetch(window, decision_policy="upper_scan")
            for factor in window:
                record = evaluate(factor)
                state = decision_state(record)
                if state == "stable":
                    ok_factor = max(ok_factor, float(record["factor"]))
                    continue
                if state == "confirmed_failure":
                    return ok_factor, float(record["factor"])
        return ok_factor, None

    def resolve_upper_boundary(
        stable_factor: float,
    ) -> tuple[float, float, float | None, str]:
        """Find and refine an upper failure, resuming after stable reversals."""

        ok_factor = float(stable_factor)
        reversal_factors: set[float] = set()
        while True:
            ok_factor, fail_factor = scan_upper_for_failure(ok_factor)
            if fail_factor is None:
                return ok_factor, ok_factor, None, "upper_max"
            resolved_fos, resolved_low, resolved_high = bisect_bracket(
                ok_factor, fail_factor
            )
            if resolved_high is not None:
                return (
                    float(resolved_fos),
                    float(resolved_low),
                    float(resolved_high),
                    "bracket",
                )
            resumed = _round_srm_factor(float(resolved_fos))
            if resumed in reversal_factors or resumed <= ok_factor + 1.0e-12:
                return resumed, resumed, None, "boundary_verification_reversal"
            reversal_factors.add(resumed)
            ok_factor = resumed

    anchor_record = evaluate(anchor)
    anchor_state = decision_state(anchor_record)
    if anchor_state == "indeterminate":
        anchor_record = cold_verify_indeterminate(
            anchor_record, trigger="anchor_boundary"
        )
        anchor_state = decision_state(anchor_record)
    if anchor_state == "stable":
        fos, bracket_low, bracket_high, bounded_by = resolve_upper_boundary(
            _round_srm_factor(float(anchor_record["factor"]))
        )
    else:
        scan_direction = "lower"
        search_ceiling = _round_srm_factor(float(anchor_record["factor"]))
        confirmed_fail_factor: float | None = search_ceiling if anchor_state == "confirmed_failure" else None
        ok_factor: float | None = None
        projection_bracketed = False
        skip_lower_scan = False
        if auto_enabled:
            projection_factors, lower_projection_info = _srm_auto_lower_projection_factors(anchor_record, lower_values, search_ceiling, factor_tol, auto_settings)
            projection_prefetch_before = set(speculative_factor_keys)
            projection_cost_snapshot = cost_snapshot()
            projection_recommended_depth = int(
                projection_cost_snapshot.get(
                    "recommended_depth",
                    len(projection_factors),
                )
            )
            projection_prefetch_depth = (
                max(1, min(len(projection_factors), projection_recommended_depth))
                if projection_recommended_depth < lookahead_depth
                else len(projection_factors)
            )
            projection_prefetch_factors = projection_factors[:projection_prefetch_depth]
            if len(projection_prefetch_factors) < len(projection_factors):
                cost_aware_deferred_candidate_count += (
                    len(projection_factors) - len(projection_prefetch_factors)
                )
            prefetch(projection_prefetch_factors)
            projection_speculative_keys.update(
                _round_srm_factor(factor)
                for factor in projection_factors
                if _round_srm_factor(factor) in speculative_factor_keys and _round_srm_factor(factor) not in projection_prefetch_before
            )
            if lower_projection_info.get("used", False):
                lower_projection_info["parallel_prefetch_enabled"] = bool(lookahead_enabled and len(projection_factors) > 1)
                lower_projection_info["parallel_prefetch_count"] = len(projection_speculative_keys)
            for factor in projection_factors:
                record = evaluate(factor)
                state = decision_state(record)
                if state == "stable":
                    candidate = float(record["factor"])
                    ok_factor = candidate if ok_factor is None else max(ok_factor, candidate)
                    continue
                if state == "confirmed_failure":
                    candidate = float(record["factor"])
                    confirmed_fail_factor = candidate if confirmed_fail_factor is None else min(confirmed_fail_factor, candidate)
                    search_ceiling = min(search_ceiling, candidate)
                if ok_factor is not None and confirmed_fail_factor is not None:
                    projection_bracketed = True
                    break
            if ok_factor is not None and confirmed_fail_factor is not None and _srm_bool(auto_settings.get("lower_projection_skip_coarse_scan_on_bracket", True), True):
                skip_lower_scan = True
                projection_bracketed = True
            if lower_projection_info.get("used", False):
                lower_projection_info["bracketed"] = bool(projection_bracketed)
                lower_projection_info["coarse_scan_skipped"] = bool(skip_lower_scan)
                if ok_factor is not None:
                    lower_projection_info["bracket_stable_factor"] = float(ok_factor)
                if confirmed_fail_factor is not None:
                    lower_projection_info["bracket_failed_factor"] = float(confirmed_fail_factor)
        if not skip_lower_scan:
            index = 0
            while index < len(lower_scan_factors):
                active_depth = scan_window_depth()
                window = [
                    factor
                    for factor in lower_scan_factors[index : index + active_depth]
                    if _round_srm_factor(factor) not in records and factor < search_ceiling
                ]
                prefetch(window, decision_policy="lower_scan")
                for factor in window:
                    record = evaluate(factor)
                    state = decision_state(record)
                    if state == "stable":
                        candidate = float(record["factor"])
                        ok_factor = candidate if ok_factor is None else max(ok_factor, candidate)
                        break
                    if state == "confirmed_failure":
                        candidate = float(record["factor"])
                        confirmed_fail_factor = candidate if confirmed_fail_factor is None else min(confirmed_fail_factor, candidate)
                        search_ceiling = min(search_ceiling, candidate)
                if ok_factor is not None:
                    break
                index += active_depth
        if ok_factor is None:
            fos = 0.0
            bounded_by = "lower_min"
        elif confirmed_fail_factor is None:
            fos, bracket_low, bracket_high, bounded_by = resolve_upper_boundary(
                ok_factor
            )
            if bracket_high is None and indeterminate_factor_keys:
                bounded_by = "indeterminate_failure"
        else:
            fos, bracket_low, bracket_high = bisect_bracket(ok_factor, confirmed_fail_factor)
            if bracket_high is None:
                fos, bracket_low, bracket_high, bounded_by = resolve_upper_boundary(
                    fos
                )

    unused_speculative_keys = set(speculative_factor_keys) - set(used_speculative_keys)
    bisection_unused_keys = bisection_speculative_keys & unused_speculative_keys
    bisection_used_keys = bisection_speculative_keys & used_speculative_keys
    final_cost_snapshot = cost_snapshot()
    if lower_projection_info.get("used", False):
        lower_projection_info["parallel_prefetch_used_count"] = len(projection_speculative_keys & used_speculative_keys)
        lower_projection_info["parallel_prefetch_unused_count"] = len(projection_speculative_keys & unused_speculative_keys)
    decision_records = [record for key, record in records.items() if key not in unused_speculative_keys]
    state_trials = [_srm_public_trial_record(record) for record in decision_records] if auto_enabled else trials
    result_state = _srm_result_state(fos, state_trials)
    if bracket_low is not None and bracket_high is not None:
        result_state["stable_factor"] = float(bracket_low)
        result_state["failed_factor"] = float(bracket_high)
    failed_factor_for_interval = result_state.get("failed_factor")
    failed_record_for_interval = None
    if failed_factor_for_interval is not None:
        try:
            failed_record_for_interval = records.get(
                _round_srm_factor(float(failed_factor_for_interval))
            )
        except (TypeError, ValueError):
            failed_record_for_interval = None
    result_state.update(
        _srm_factor_interval_state(
            result_state.get("stable_factor", 0.0),
            failed_factor_for_interval,
            failed_record_for_interval,
            factor_tol=factor_tol,
            auto_enabled=auto_enabled,
        )
    )
    stable_evidence = [
        float(record["factor"])
        for record in decision_records
        if _srm_trial_decision_state(record, auto_enabled=auto_enabled) == "stable"
    ]
    failure_evidence = [
        float(record["factor"])
        for record in decision_records
        if _srm_trial_decision_state(record, auto_enabled=auto_enabled) == "confirmed_failure"
    ]
    nonmonotonic_evidence = any(stable > failed for stable in stable_evidence for failed in failure_evidence)
    bracket_exists = bool(
        bracket_low is not None
        and bracket_high is not None
        and not nonmonotonic_evidence
    )
    indeterminate_inside_bracket = sorted(
        factor
        for factor in indeterminate_factor_keys
        if bracket_low is not None
        and bracket_high is not None
        and float(bracket_low) < factor < float(bracket_high)
    )
    bracket_tolerance_met = bool(
        result_state.get("factor_of_safety_tolerance_met", False)
    )
    confirmed_bracket = bool(
        bracket_exists
        and bracket_tolerance_met
        and not indeterminate_inside_bracket
    )
    if nonmonotonic_evidence:
        factor_of_safety_status = "nonmonotonic_evidence"
        factor_of_safety_confidence = "low"
    elif confirmed_bracket:
        factor_of_safety_status = "confirmed_bracket"
        factor_of_safety_confidence = (
            "high" if bool(result_state.get("factor_of_safety_certified", False)) else "limited"
        )
    elif bracket_exists and indeterminate_inside_bracket:
        factor_of_safety_status = "unresolved_indeterminate_interval"
        factor_of_safety_confidence = "limited"
    elif bracket_exists:
        factor_of_safety_status = "bracket_tolerance_not_met"
        factor_of_safety_confidence = "limited"
    elif indeterminate_factor_keys:
        factor_of_safety_status = "lower_bound_indeterminate"
        factor_of_safety_confidence = "limited"
    else:
        factor_of_safety_status = "unbounded_search_limit"
        factor_of_safety_confidence = "limited"
    if not confirmed_bracket:
        result_state["factor_of_safety_certified"] = False
        if indeterminate_inside_bracket:
            result_state["factor_of_safety_value_kind"] = (
                "unresolved_interval_lower_bound"
            )
            result_state["boundary_quality"] = (
                "indeterminate_trial_inside_confirmed_endpoints"
            )
        elif nonmonotonic_evidence:
            result_state["factor_of_safety_value_kind"] = (
                "nonmonotonic_evidence_unconfirmed"
            )
    search_mode_label = "auto" if auto_enabled else "adaptive_bracket"
    info = {
        "search_mode": search_mode_label,
        "trial_state": "independent_from_stage_start",
        "anchor_factor": anchor,
        "factor_tol": factor_tol,
        "max_bisection": max_bisection,
        "bracket_stride": stride,
        "candidate_factors": len(ordered),
        "coarse_scan_factors": upper_scan_factors if scan_direction == "upper" else lower_scan_factors,
        "upper_coarse_scan_factors": upper_scan_factors,
        "lower_coarse_scan_factors": lower_scan_factors,
        "scan_direction": scan_direction,
        "bracketed": bracket_exists,
        "bracket_resolved": confirmed_bracket,
        "indeterminate_factors_inside_bracket": indeterminate_inside_bracket,
        "bounded_by": bounded_by,
        "bracket_stable_factor": bracket_low,
        "bracket_failed_factor": bracket_high,
        "factor_of_safety_status": factor_of_safety_status,
        "factor_of_safety_confidence": factor_of_safety_confidence,
        "nonmonotonic_evidence": nonmonotonic_evidence,
        "indeterminate_trial_count": len(indeterminate_factor_keys),
        "indeterminate_factors": sorted(indeterminate_factor_keys),
        "strength_factor_note": "factor < 1 strengthens the material; factor > 1 reduces c and tan(phi)",
        "lower_projection": lower_projection_info,
        "parallel": {
            **dict(parallel_cfg),
            "enabled": bool(lookahead_enabled),
            "evaluated_trials": len(records),
            "reported_trials": len(trials),
            "strategy": (
                "auto_diagnostic_bracket_lookahead"
                if lookahead_enabled and auto_enabled
                else "adaptive_bracket_lookahead"
                if lookahead_enabled
                else "auto_diagnostic_bracket"
                if auto_enabled
                else "adaptive_bracket_sequential_stop"
            ),
            "requested_strategy": parallel_strategy,
            "requested_executor": requested_executor_kind,
            "effective_executor": effective_executor_kind,
            "process_executor_disabled_reason": process_executor_disabled_reason,
            "process_executor_fallback_count": int(process_executor_fallback_count),
            "process_executor_errors": list(process_executor_errors),
            "preserve_decision_order": preserve_order,
            "lookahead_depth": lookahead_depth,
            "window_evaluated_trials": window_evaluated_trials,
            "speculative_trial_count": len(speculative_factor_keys),
            "used_speculative_trial_count": len(used_speculative_keys),
            "unused_speculative_trial_count": len(unused_speculative_keys),
            "canceled_speculative_trial_count": int(canceled_speculative_trial_count),
            "speculative_cancellation_requested": bool(speculative_cancellation_requested),
            "decision_linked_cancellation_enabled": bool(
                decision_linked_cancellation_enabled
            ),
            "decision_linked_requested_count": int(
                decision_linked_requested_count
            ),
            "decision_linked_pending_cancel_count": int(
                decision_linked_pending_cancel_count
            ),
            "decision_linked_safe_stop_count": int(
                decision_linked_safe_stop_count
            ),
            "decision_linked_completed_after_request_count": int(
                decision_linked_completed_after_request_count
            ),
            "decision_linked_requested_factors": sorted(
                decision_linked_requested_factors
            ),
            "speculative_prefetch_call_count": speculative_prefetch_call_count,
            "speculative_prefetch_wall_elapsed_seconds": float(speculative_prefetch_wall_elapsed_seconds),
            "speculative_trial_elapsed_seconds": float(speculative_trial_elapsed_seconds),
            "speculative_queue_wait_elapsed_seconds": float(speculative_queue_wait_elapsed_seconds),
            "speculative_worker_elapsed_seconds": float(speculative_worker_elapsed_seconds),
            "speculative_estimated_wall_clock_saving_seconds": float(speculative_estimated_wall_clock_saving_seconds),
            "speculative_cancellation_note": (
                "; ".join(speculative_cancellation_notes)
                if speculative_cancellation_notes
                else "adaptive lookahead keeps decision order and can stop out-of-bound speculative trials at increment or Newton safe checkpoints"
            ),
            "bisection_speculation_enabled": bisection_speculation_enabled,
            "bisection_speculative_trial_count": len(bisection_speculative_keys),
            "bisection_used_speculative_trial_count": len(bisection_used_keys),
            "bisection_unused_speculative_trial_count": len(bisection_unused_keys),
            "bisection_speculative_unused_factors": sorted(bisection_unused_keys),
            "cost_aware_lookahead_enabled": bool(
                cost_aware_lookahead_enabled
            ),
            "cost_aware_min_elapsed_seconds": float(
                cost_aware_min_elapsed_seconds
            ),
            "cost_aware_failure_ratio_threshold": float(
                cost_aware_failure_ratio
            ),
            "cost_aware_escalation_ratio_threshold": float(
                cost_aware_escalation_ratio
            ),
            "cost_aware_min_samples": int(cost_aware_min_samples),
            "cost_aware_depth_limited_count": int(
                cost_aware_depth_limited_count
            ),
            "cost_aware_asymmetric_bisection_count": int(
                cost_aware_asymmetric_bisection_count
            ),
            "cost_aware_deferred_candidate_count": int(
                cost_aware_deferred_candidate_count
            ),
            "event_driven_cost_cancellation_enabled": bool(
                event_driven_cost_cancellation_enabled
            ),
            "event_driven_cost_shrink_count": int(
                event_driven_cost_shrink_count
            ),
            "event_driven_cost_cancel_candidate_count": int(
                event_driven_cost_cancel_candidate_count
            ),
            "cost_aware_observation": final_cost_snapshot,
            "speculative_unused_factors": sorted(unused_speculative_keys),
            "speculative_unused_trials": [
                _srm_public_trial_record(records[key]) for key in sorted(unused_speculative_keys) if key in records
            ],
            "disabled_reason": (
                ""
                if lookahead_enabled
                else str(parallel_cfg.get("disabled_reason", ""))
                or "adaptive bracket lookahead requires srm.parallel.enabled=true, strategy=lookahead, preserve_decision_order=true, and max_workers>1"
            ),
        },
        **result_state,
    }
    warm_start_source_factors: set[float] = set()
    for row in trials:
        value = row.get("warm_start_source_factor", "") if isinstance(row, Mapping) else ""
        if value in ("", None):
            continue
        try:
            warm_start_source_factors.add(float(value))
        except (TypeError, ValueError):
            continue
    if auto_enabled:
        info["auto"] = {
            "enabled": True,
            "decision_source": auto_settings.get("decision_source", "trial_log_diagnostics"),
            "suspect_trial_count": sum(1 for row in trials if str(row.get("auto_decision", "")) == "suspect_failure"),
            "confirmed_failure_count": sum(1 for row in trials if str(row.get("auto_decision", "")) == "confirmed_failure"),
            "retry_count": sum(1 for row in trials if bool(row.get("auto_retry", False))),
            "retry_suspect_failures": bool(auto_settings.get("retry_suspect_failures", True)),
            "max_suspect_retries": int(auto_settings.get("max_suspect_retries", 0)),
            "boundary_verification_enabled": bool(
                auto_settings.get("boundary_verification_enabled", False)
            ),
            "boundary_verification_strategy": boundary_verification_strategy,
            "boundary_verification_defer_min_failure_score": int(
                auto_settings.get("boundary_verification_defer_min_failure_score", 6)
            ),
            "boundary_verification_count": sum(
                1 for row in trials if bool(row.get("boundary_verification", False))
            ),
            "boundary_verification_deferred_count": int(
                boundary_verification_deferred_count
            ),
            "boundary_verification_executed_count": int(
                boundary_verification_final_count
            ),
            "boundary_verification_recovery_count": int(
                boundary_verification_recovery_count
            ),
            "boundary_verification_stable_reversal_count": int(
                boundary_verification_stable_reversal_count
            ),
            "boundary_verification_cold_retry_on_indeterminate": bool(
                auto_settings.get(
                    "boundary_verification_cold_retry_on_indeterminate", True
                )
            ),
            "boundary_verification_cold_retry_count": int(
                boundary_verification_cold_retry_count
            ),
            "boundary_verification_cold_retry_factors": sorted(
                boundary_verification_cold_retry_factors.keys()
            ),
            "boundary_checkpoint_continuation_enabled": bool(
                auto_settings.get(
                    "boundary_checkpoint_continuation_enabled", True
                )
            ),
            "boundary_checkpoint_continuation_extra_cutbacks": int(
                auto_settings.get(
                    "boundary_checkpoint_continuation_extra_cutbacks", 1
                )
                or 1
            ),
            "boundary_checkpoint_residual_prediction_enabled": bool(
                auto_settings.get(
                    "boundary_checkpoint_residual_prediction_enabled", True
                )
            ),
            "boundary_checkpoint_residual_prediction_max_extra_cutbacks": int(
                auto_settings.get(
                    "boundary_checkpoint_residual_prediction_max_extra_cutbacks",
                    4,
                )
                or 4
            ),
            "boundary_checkpoint_residual_prediction_used_count": sum(
                1
                for row in trials
                if str(
                    row.get("checkpoint_residual_prediction_reason", "")
                )
                == "predictive_residual_decay"
            ),
            "boundary_verification_strict_tangent": bool(
                auto_settings.get("boundary_verification_strict_tangent", True)
            ),
            "retry_strict_tangent": bool(
                auto_settings.get("retry_strict_tangent", True)
            ),
            "boundary_checkpoint_continuation_requested_count": sum(
                1
                for row in trials
                if bool(
                    row.get(
                        "boundary_checkpoint_continuation_requested", False
                    )
                )
            ),
            "boundary_checkpoint_continuation_used_count": sum(
                1
                for row in trials
                if bool(
                    row.get("boundary_checkpoint_continuation_used", False)
                )
            ),
            "boundary_checkpoint_fallback_count": sum(
                1
                for row in trials
                if bool(
                    row.get(
                        "boundary_checkpoint_continuation_requested", False
                    )
                )
                and not bool(
                    row.get("boundary_checkpoint_continuation_used", False)
                )
            ),
            "factor_tol_enforcement_enabled": bool(
                factor_tol_enforcement_enabled
            ),
            "factor_tol_enforcement_accept_verified_numerical_failure": bool(
                factor_tol_accept_verified_numerical_failure
            ),
            "factor_tol_require_physical_failure_evidence": bool(
                auto_settings.get(
                    "factor_tol_require_physical_failure_evidence", True
                )
            ),
            "factor_tol_enforcement_max_extra_bisections": int(
                factor_tol_max_extra_bisections
            ),
            "factor_tol_enforcement_extra_bisections_used": int(
                factor_tol_extra_bisections_used
            ),
            "factor_tol_numerical_failure_boundary_count": len(
                factor_tol_numerical_failure_factors
            ),
            "factor_tol_numerical_failure_boundary_factors": sorted(
                factor_tol_numerical_failure_factors
            ),
            "indeterminate_search_count": len(indeterminate_factor_keys),
            "indeterminate_search_factors": sorted(indeterminate_factor_keys),
            "suspect_last_load_threshold": float(auto_settings.get("suspect_last_load_threshold", 0.90)),
            "confirmed_cluster_fraction": float(auto_settings.get("confirmed_cluster_fraction", 0.50)),
            "confirmed_plastic_ratio": float(auto_settings.get("confirmed_plastic_ratio", 0.50)),
            "early_failure_stop_enabled": bool(auto_settings.get("early_failure_stop_enabled", True)),
            "early_failure_min_cutback_ratio": float(auto_settings.get("early_failure_min_cutback_ratio", 0.75)),
            "early_failure_min_last_load": float(auto_settings.get("early_failure_min_last_load", 0.90)),
            "early_failure_score_threshold": int(auto_settings.get("early_failure_score_threshold", 5)),
            "adaptive_increment_control_enabled": bool(auto_settings.get("adaptive_increment_control_enabled", True)),
            "adaptive_increment_min_cutback_ratio": float(auto_settings.get("adaptive_increment_min_cutback_ratio", 0.75)),
            "adaptive_increment_max_factor_distance": float(auto_settings.get("adaptive_increment_max_factor_distance", 0.25)),
            "adaptive_increment_max_steps_multiplier": float(auto_settings.get("adaptive_increment_max_steps_multiplier", 2.0)),
        }
    info["warm_start"] = {
        "enabled": bool(warm_start_settings.get("enabled", False)),
        "configured": bool(warm_start_settings.get("configured", False)),
        "supported": bool(warm_start_settings.get("supported", True)),
        "disabled_reason": str(warm_start_settings.get("disabled_reason", "")),
        "displacement_only": bool(warm_start_settings.get("displacement_only", True)),
        "prefer_stable_source": bool(warm_start_settings.get("prefer_stable_source", True)),
        "max_factor_distance": float(warm_start_settings.get("max_factor_distance", 0.05) or 0.05),
        "used_trial_count": sum(1 for row in trials if bool(row.get("warm_start_used", False))),
        "source_factors": sorted(warm_start_source_factors),
    }
    selected_record = records.get(_round_srm_factor(fos)) if fos > 0.0 else None
    selected_result = selected_record.get("result") if isinstance(selected_record, Mapping) else None
    return (
        selected_result if isinstance(selected_result, StageResult2D) else last_ok or last_trial,
        fos,
        trials,
        info,
    )


def _srm_adaptive_bracket_stride(factors: list[float], srm_cfg: Mapping[str, Any]) -> int:
    raw_stride = srm_cfg.get("bracket_stride", srm_cfg.get("adaptive_stride", srm_cfg.get("coarse_stride")))
    if raw_stride is not None:
        try:
            return max(1, int(raw_stride))
        except (TypeError, ValueError):
            raise FEM2DError("srm bracket_stride must be a positive integer") from None
    if len(factors) <= 4:
        return 1
    return max(1, int(math.ceil(math.sqrt(len(factors)))))


def _srm_parallel_lookahead_depth(srm_cfg: Mapping[str, Any], parallel_cfg: Mapping[str, Any], workers: int) -> int:
    raw_depth = parallel_cfg.get(
        "lookahead_depth",
        parallel_cfg.get("prefetch_depth", srm_cfg.get("lookahead_depth", srm_cfg.get("prefetch_depth"))),
    )
    if raw_depth is None or str(raw_depth).strip().lower() == "auto":
        return max(1, int(workers or 1))
    try:
        return max(1, min(int(raw_depth), max(1, int(workers or 1))))
    except (TypeError, ValueError):
        raise FEM2DError("srm parallel lookahead_depth must be a positive integer or auto") from None


def _srm_cost_aware_lookahead_snapshot(
    records: Mapping[float, Mapping[str, Any]],
    *,
    enabled: bool,
    min_elapsed_seconds: float,
    failure_ratio_threshold: float,
    escalation_ratio_threshold: float,
    min_samples: int,
) -> dict[str, Any]:
    observations: list[tuple[float, bool, float]] = []
    for raw_factor, record in records.items():
        if not isinstance(record, Mapping):
            continue
        elapsed = _srm_record_float(record, "elapsed_seconds", 0.0)
        if not math.isfinite(elapsed) or elapsed <= 0.0:
            continue
        try:
            factor = float(record.get("factor", raw_factor))
        except (TypeError, ValueError):
            continue
        observations.append((factor, bool(record.get("ok", False)), elapsed))

    stable = sorted(
        ((factor, elapsed) for factor, ok, elapsed in observations if ok),
        key=lambda item: item[0],
    )
    failed = sorted(
        ((factor, elapsed) for factor, ok, elapsed in observations if not ok),
        key=lambda item: item[0],
    )
    stable_costs = [elapsed for _factor, elapsed in stable]
    failed_costs = [elapsed for _factor, elapsed in failed]
    stable_median = (
        float(np.median(np.asarray(stable_costs, dtype=float)))
        if stable_costs
        else 0.0
    )
    failed_median = (
        float(np.median(np.asarray(failed_costs, dtype=float)))
        if failed_costs
        else 0.0
    )
    failure_ratio = (
        failed_median / max(stable_median, 1.0e-12)
        if failed_median > 0.0 and stable_median > 0.0
        else 0.0
    )
    latest_stable_elapsed = stable[-1][1] if stable else 0.0
    previous_stable_median = 0.0
    stable_escalation_ratio = 0.0
    if len(stable) >= 2:
        previous = np.asarray(
            [elapsed for _factor, elapsed in stable[:-1]],
            dtype=float,
        )
        previous_stable_median = float(np.median(previous))
        stable_escalation_ratio = latest_stable_elapsed / max(
            previous_stable_median,
            1.0e-12,
        )

    enough_samples = len(observations) >= max(1, int(min_samples))
    expensive_failure = bool(
        enabled
        and enough_samples
        and failed_median >= float(min_elapsed_seconds)
        and failure_ratio >= float(failure_ratio_threshold)
    )
    stable_escalation = bool(
        enabled
        and enough_samples
        and latest_stable_elapsed >= float(min_elapsed_seconds)
        and stable_escalation_ratio >= float(escalation_ratio_threshold)
    )
    expensive_warmup = bool(
        enabled
        and observations
        and not enough_samples
        and max(elapsed for _factor, _ok, elapsed in observations)
        >= float(min_elapsed_seconds)
    )
    if expensive_failure or stable_escalation:
        recommended_depth = 1
        reason = (
            "failure_cost_ratio"
            if expensive_failure
            else "stable_cost_escalation"
        )
    elif expensive_warmup:
        recommended_depth = 2
        reason = "expensive_initial_trial"
    else:
        recommended_depth = 2**31 - 1
        reason = "insufficient_or_low_cost_evidence"
    return {
        "enabled": bool(enabled),
        "sample_count": len(observations),
        "stable_sample_count": len(stable),
        "failed_sample_count": len(failed),
        "stable_median_elapsed_seconds": stable_median,
        "failed_median_elapsed_seconds": failed_median,
        "failure_to_stable_cost_ratio": failure_ratio,
        "latest_stable_elapsed_seconds": latest_stable_elapsed,
        "previous_stable_median_elapsed_seconds": previous_stable_median,
        "stable_cost_escalation_ratio": stable_escalation_ratio,
        "failure_side_expensive": bool(
            expensive_failure or stable_escalation
        ),
        "recommended_depth": int(recommended_depth),
        "reason": reason,
    }


def _srm_adaptive_scan_factors(factors: list[float], stride: int) -> list[float]:
    ordered = sorted({_round_srm_factor(value) for value in factors})
    if not ordered:
        return []
    stride = max(1, int(stride))
    values = [ordered[index] for index in range(0, len(ordered), stride)]
    if values[-1] != ordered[-1]:
        values.append(ordered[-1])
    return values


def _srm_adaptive_branch_scan_factors(factors: list[float], stride: int) -> list[float]:
    ordered = [_round_srm_factor(value) for value in factors if value > 0.0]
    if not ordered:
        return []
    stride = max(1, int(stride))
    if stride <= 1:
        return ordered
    values = [ordered[min(stride - 1, len(ordered) - 1)]]
    for index in range(stride * 2 - 1, len(ordered), stride):
        values.append(ordered[index])
    if values[-1] != ordered[-1]:
        values.append(ordered[-1])
    return values


def _run_srm_two_branch_trials(
    factors: list[float],
    srm_cfg: Mapping[str, Any],
    failure_plastic_ratio: float,
    solve_trial: Callable[[float], StageResult2D],
    *,
    parallel: Mapping[str, Any] | None = None,
    progress: Mapping[str, Any] | None = None,
    progress_stage_name: str = "",
    topology_cache: _SRMTopologyDiagnosticsCache | None = None,
) -> tuple[StageResult2D | None, float, list[dict[str, Any]], dict[str, Any]]:
    anchor = float(srm_cfg.get("anchor_factor", srm_cfg.get("anchor", 1.0)))
    if anchor <= 0.0:
        raise FEM2DError("srm anchor_factor must be positive")
    factor_tol = float(srm_cfg.get("factor_tol", srm_cfg.get("tolerance", srm_cfg.get("tol", 0.005))))
    if factor_tol <= 0.0:
        raise FEM2DError("srm factor_tol must be positive")
    max_bisection = int(srm_cfg.get("max_bisection", srm_cfg.get("bisection_iterations", 8)))
    if max_bisection < 0:
        raise FEM2DError("srm max_bisection must be non-negative")
    lower_values, upper_values = _srm_two_branch_values(factors, srm_cfg, anchor)

    parallel_cfg = parallel if isinstance(parallel, Mapping) else {"enabled": False, "max_workers": 1}
    parallel_enabled = bool(parallel_cfg.get("enabled", False))
    workers = max(1, int(parallel_cfg.get("max_workers", 1) or 1))
    candidate_window_size = workers if parallel_enabled else 1
    window_evaluated_trials = 0
    trials: list[dict[str, Any]] = []
    records: dict[float, dict[str, Any]] = {}
    published: set[float] = set()
    last_ok: StageResult2D | None = None
    last_trial: StageResult2D | None = None
    fos = 0.0

    def store_record(record: dict[str, Any]) -> dict[str, Any]:
        records[_round_srm_factor(float(record["factor"]))] = record
        return record

    def record_for(factor: float) -> dict[str, Any]:
        factor = _round_srm_factor(factor)
        record = records.get(factor)
        if record is None:
            record = store_record(_srm_trial_record(factor, failure_plastic_ratio, solve_trial, topology_cache))
        return record

    def publish(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal fos, last_ok, last_trial
        factor = _round_srm_factor(float(record["factor"]))
        if factor in published:
            return record
        trial_result = record.get("result")
        if isinstance(trial_result, StageResult2D):
            last_trial = trial_result
        if bool(record.get("ok", False)):
            if isinstance(trial_result, StageResult2D):
                last_ok = trial_result
            fos = factor
        trials.append(_srm_public_trial_record(record))
        published.add(factor)
        _srm_emit_progress(progress, stage_name=progress_stage_name, record=record, index=len(trials), total=None, current_fos=fos, prefix="two-branch-trial")
        return record

    def prefetch(candidate_factors: list[float]) -> None:
        nonlocal window_evaluated_trials
        normalized = [_round_srm_factor(factor) for factor in candidate_factors]
        missing = [factor for factor in normalized if factor not in records]
        if not missing:
            return
        if parallel_enabled and len(missing) > 1 and workers > 1:
            for record in _srm_evaluate_records_parallel(
                missing,
                failure_plastic_ratio,
                solve_trial,
                workers,
                topology_cache=topology_cache,
                thread_control=parallel_cfg.get("thread_control") if isinstance(parallel_cfg, Mapping) else None,
            ):
                store_record(record)
            window_evaluated_trials += len(missing)
            return
        for factor in missing:
            store_record(_srm_trial_record(factor, failure_plastic_ratio, solve_trial, topology_cache))

    def evaluate(factor: float) -> dict[str, Any]:
        factor = _round_srm_factor(factor)
        return publish(record_for(factor))

    def bisect_bracket(ok_factor: float, fail_factor: float) -> float:
        low = min(ok_factor, fail_factor)
        high = max(ok_factor, fail_factor)
        ok_side = ok_factor
        for _index in range(max_bisection):
            if high - low <= factor_tol:
                break
            mid = _round_srm_factor(0.5 * (low + high))
            if mid <= low or mid >= high:
                break
            mid_record = evaluate(mid)
            if bool(mid_record.get("ok", False)):
                ok_side = mid
                low = mid
            else:
                high = mid
        return ok_side

    anchor_record = evaluate(anchor)
    bracketed = False
    bounded_by = "none"
    if bool(anchor_record.get("ok", False)):
        ok_factor = anchor
        fail_factor: float | None = None
        index = 0
        while index < len(upper_values):
            window = upper_values[index : index + candidate_window_size]
            prefetch(window)
            for factor in window:
                record = evaluate(factor)
                if bool(record.get("ok", False)):
                    ok_factor = float(record["factor"])
                else:
                    fail_factor = float(record["factor"])
                    break
            if fail_factor is not None:
                break
            index += len(window)
        if fail_factor is not None:
            fos = bisect_bracket(ok_factor, fail_factor)
            bracketed = True
        else:
            fos = ok_factor
            bounded_by = "upper_max"
    else:
        fail_factor = anchor
        ok_factor: float | None = None
        index = 0
        while index < len(lower_values):
            window = lower_values[index : index + candidate_window_size]
            prefetch(window)
            for factor in window:
                record = evaluate(factor)
                if bool(record.get("ok", False)):
                    ok_factor = float(record["factor"])
                    break
                fail_factor = float(record["factor"])
            if ok_factor is not None:
                break
            index += len(window)
        if ok_factor is not None:
            fos = bisect_bracket(ok_factor, fail_factor)
            bracketed = True
        else:
            fos = 0.0
            bounded_by = "lower_min"

    info = {
        "search_mode": "two_branch",
        "anchor_factor": anchor,
        "factor_tol": factor_tol,
        "bracketed": bracketed,
        "bounded_by": bounded_by,
        "trial_state": "independent_from_stage_start",
        "strength_factor_note": "factor < 1 strengthens the material; factor > 1 reduces c and tan(phi)",
        **_srm_result_state(fos, trials),
    }
    if parallel_enabled:
        info["parallel"] = {
            **dict(parallel_cfg),
            "evaluated_trials": len(records),
            "reported_trials": len(trials),
            "candidate_window_size": candidate_window_size,
            "window_evaluated_trials": window_evaluated_trials,
            "strategy": "two_branch_candidate_windows",
        }
    return last_ok or last_trial, fos, trials, info


def _srm_failure_reason(converged: bool, plastic_ratio: float, error: str | None) -> str:
    text = (error or "").lower()
    if "cancel" in text:
        return "cancelled"
    if "detj" in text or "detj" in text.replace(" ", ""):
        return "invalid_detJ"
    if "displacement" in text or "non-finite" in text:
        return "excessive_displacement"
    if not converged:
        return "nonconvergence"
    if math.isinf(plastic_ratio) or plastic_ratio > 0.0:
        return "plastic_divergence"
    return "unknown_failure"


def _srm_result_state(fos: float, trials: list[dict[str, Any]]) -> dict[str, Any]:
    auto_semantics = any(
        str(row.get("auto_decision", "") or "") or str(row.get("srm_trial_state", "") or "")
        for row in trials
    )
    states = [(row, _srm_trial_decision_state(row, auto_enabled=auto_semantics)) for row in trials]
    stable = [row for row, state in states if state == "stable"]
    failed = [row for row, state in states if state == "confirmed_failure"]
    indeterminate = [row for row, state in states if state == "indeterminate"]
    stable_factor = max((float(row["factor"]) for row in stable), default=0.0)
    failed_candidates = [float(row["factor"]) for row in failed if float(row["factor"]) > stable_factor]
    failed_factor = min(failed_candidates) if failed_candidates else (min((float(row["factor"]) for row in failed), default=None))
    return {
        "stable_factor": stable_factor,
        "failed_factor": failed_factor,
        "final_display_state": "last_stable" if fos > 0.0 else "last_failed",
        "failure_reasons": [str(row.get("failure_reason", "")) for row in failed if str(row.get("failure_reason", ""))],
        "indeterminate_factors": [float(row["factor"]) for row in indeterminate],
        "indeterminate_reasons": [str(row.get("failure_reason", "")) for row in indeterminate if str(row.get("failure_reason", ""))],
    }


def _srm_factor_interval_state(
    stable_factor: Any,
    failed_factor: Any,
    failed_record: Mapping[str, Any] | None,
    *,
    factor_tol: float | None = None,
    auto_enabled: bool = True,
) -> dict[str, Any]:
    try:
        stable = float(stable_factor)
    except (TypeError, ValueError):
        stable = 0.0
    try:
        failed = float(failed_factor) if failed_factor is not None else None
    except (TypeError, ValueError):
        failed = None
    width = max(failed - stable, 0.0) if failed is not None else None
    interval = {"stable": stable, "failed": failed, "width": width}
    failed_state = (
        _srm_trial_decision_state(failed_record, auto_enabled=auto_enabled)
        if isinstance(failed_record, Mapping)
        else "indeterminate"
    )
    if failed is None:
        quality = "unbounded"
    elif not isinstance(failed_record, Mapping):
        quality = "failure_unclassified"
    elif failed_state != "confirmed_failure":
        quality = (
            "suspect_failure_after_verification"
            if bool(failed_record.get("boundary_verification", False))
            else "suspect_failure_unverified"
        )
    elif bool(failed_record.get("early_failure_stop", False)) and not bool(
        failed_record.get("boundary_verification", False)
    ):
        quality = "early_failure_unverified"
    elif bool(failed_record.get("boundary_verification", False)):
        quality = (
            "verified_numerical_failure_boundary"
            if bool(
                failed_record.get(
                    "factor_tol_numerical_failure_boundary", False
                )
            )
            else "verified_failure_boundary"
        )
    else:
        quality = "confirmed_failure_boundary"
    boundary_certified = bool(failed is not None and failed_state == "confirmed_failure")
    tolerance_met = bool(
        boundary_certified
        and width is not None
        and factor_tol is not None
        and float(width) <= float(factor_tol) + 1.0e-12
    )
    certified = bool(boundary_certified and tolerance_met)
    if certified:
        value_kind = "certified_stable_lower_bound"
    elif boundary_certified:
        value_kind = "confirmed_interval_lower_bound"
    elif stable > 0.0:
        value_kind = "uncertified_stable_lower_bound"
    else:
        value_kind = "not_established"
    return {
        "factor_of_safety_interval": interval,
        "factor_of_safety_boundary_certified": boundary_certified,
        "factor_of_safety_tolerance_met": tolerance_met,
        "factor_of_safety_certified": certified,
        "factor_of_safety_value_kind": value_kind,
        "boundary_quality": quality,
        "boundary_verified": bool(
            isinstance(failed_record, Mapping)
            and failed_record.get("boundary_verification", False)
        ),
        "boundary_failure_class": (
            str(failed_record.get("auto_failure_class", ""))
            if isinstance(failed_record, Mapping)
            else ""
        ),
    }


def _srm_failed_trial_result(
    mesh: Mesh2D,
    structural_elements: list[StructuralElement2D] | None,
    stage_name: str,
    *,
    axisymmetric: bool = False,
) -> StageResult2D:
    ndof = structural_total_dofs(mesh, structural_elements, axisymmetric=axisymmetric)
    active_elements = [str(element.id) for element in mesh.elements if element.active]
    return StageResult2D(
        stage_name,
        np.zeros(ndof, dtype=float),
        np.zeros(ndof, dtype=float),
        [],
        {},
        active_elements,
        {
            "method": "axisymmetric_srm" if axisymmetric else "srm",
            "geometry": "axisymmetric" if axisymmetric else "plane_strain",
            "converged": False,
            "iterations": 0,
            "residual_norm": math.inf,
            "failure_reason": "all_srm_trials_failed",
            "postprocess_results": False,
        },
    )


def _srm_selected_factor(fos: float, trials: list[dict[str, Any]]) -> float | None:
    if fos > 0.0:
        return float(fos)
    for row in reversed(trials):
        if bool(row.get("ok", False)):
            return float(row["factor"])
    return None


def _srm_two_branch_values(factors: list[float], srm_cfg: Mapping[str, Any], anchor: float) -> tuple[list[float], list[float]]:
    lower_cfg = srm_cfg.get("lower_branch", {})
    upper_cfg = srm_cfg.get("upper_branch", {})
    lower_values = _srm_branch_values(lower_cfg, anchor, "lower")
    upper_values = _srm_branch_values(upper_cfg, anchor, "upper")
    if not lower_values:
        lower_values = sorted({_round_srm_factor(value) for value in factors if value < anchor}, reverse=True)
    if not upper_values:
        upper_values = sorted({_round_srm_factor(value) for value in factors if value > anchor})
    return lower_values, upper_values


def _srm_branch_values(raw: Any, anchor: float, branch: str) -> list[float]:
    if not isinstance(raw, Mapping):
        return []
    if raw.get("factors") is not None:
        values = [float(value) for value in _ensure_list(raw.get("factors"))]
        if branch == "lower":
            return sorted({_round_srm_factor(value) for value in values if 0.0 < value < anchor}, reverse=True)
        return sorted({_round_srm_factor(value) for value in values if value > anchor})
    if branch == "lower":
        limit = raw.get("factor_min", raw.get("min_factor", raw.get("min")))
        step = raw.get("factor_step", raw.get("step"))
        if limit is None or step is None:
            return []
        return _srm_descending_values(anchor - float(step), float(limit), float(step))
    limit = raw.get("factor_max", raw.get("max_factor", raw.get("max")))
    step = raw.get("factor_step", raw.get("step"))
    if limit is None or step is None:
        return []
    return _srm_ascending_values(anchor + float(step), float(limit), float(step))


def _srm_descending_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise FEM2DError("srm lower_branch factor_step must be positive")
    values: list[float] = []
    value = start
    while value >= stop - step * 1.0e-9 and value > 0.0:
        values.append(_round_srm_factor(value))
        value -= step
    return values


def _srm_ascending_values(start: float, stop: float, step: float) -> list[float]:
    if step <= 0.0:
        raise FEM2DError("srm upper_branch factor_step must be positive")
    values: list[float] = []
    value = start
    while value <= stop + step * 1.0e-9:
        values.append(_round_srm_factor(value))
        value += step
    return values


def _round_srm_factor(value: float) -> float:
    return round(float(value), 12)


def _srm_with_runtime_context(srm_cfg: Mapping[str, Any], solver_cfg: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(srm_cfg)
    runtime = solver_cfg.get("execution", solver_cfg.get("runtime", solver_cfg.get("run_context", {})))
    if isinstance(runtime, Mapping):
        out["_runtime"] = dict(runtime)
    return out


def _srm_coarse_to_fine_settings(srm_cfg: Mapping[str, Any]) -> dict[str, Any]:
    raw = srm_cfg.get("coarse_to_fine", srm_cfg.get("coarse_mesh", srm_cfg.get("coarse_to_fine_srm")))
    if raw is None:
        return {"enabled": False}
    if isinstance(raw, bool):
        return {"enabled": bool(raw)}
    if not isinstance(raw, Mapping):
        return {"enabled": False, "skip_reason": "coarse_to_fine_must_be_mapping_or_bool"}
    return {**dict(raw), "enabled": bool(raw.get("enabled", True))}


def _srm_build_coarse_mesh(mesh: Mesh2D, srm_cfg: Mapping[str, Any]) -> tuple[Mesh2D | None, dict[str, Any]]:
    settings = _srm_coarse_to_fine_settings(srm_cfg)
    info: dict[str, Any] = {
        "enabled": bool(settings.get("enabled", False)),
        "used": False,
        "mesh_policy": "structured_rectangle_only",
    }
    if not info["enabled"]:
        if "skip_reason" in settings:
            info["skip_reason"] = settings["skip_reason"]
        return None, info
    active = [element for element in mesh.elements if element.active]
    if not active:
        info["skip_reason"] = "no_active_elements"
        return None, info
    element_types = {str(element.type).upper() for element in active}
    materials = {str(element.material) for element in active}
    integrations = {str(element.integration).upper() for element in active}
    if len(element_types) != 1:
        info["skip_reason"] = "mixed_element_types_not_supported"
        return None, info
    if len(materials) != 1:
        info["skip_reason"] = "mixed_materials_not_supported"
        return None, info
    if len(integrations) != 1:
        info["skip_reason"] = "mixed_integration_rules_not_supported"
        return None, info
    etype = next(iter(element_types))
    if etype not in {"QUAD4", "QUAD8", "TRI3", "TRI6"}:
        info["skip_reason"] = "element_type_not_supported"
        return None, info
    x_values, y_values = _srm_structured_corner_axes(mesh, active)
    if len(x_values) < 2 or len(y_values) < 2:
        info["skip_reason"] = "structured_axes_not_detected"
        return None, info
    source_nx = len(x_values) - 1
    source_ny = len(y_values) - 1
    expected_elements = source_nx * source_ny * (2 if etype.startswith("TRI") else 1)
    if expected_elements != len(active):
        info["skip_reason"] = "non_structured_rectangle_not_supported"
        info["source_element_count"] = len(active)
        info["expected_structured_elements"] = expected_elements
        return None, info
    try:
        coarsening = float(settings.get("coarsening_factor", settings.get("factor", settings.get("mesh_size_multiplier", 2.0))))
    except (TypeError, ValueError):
        raise FEM2DError("srm coarse_to_fine.coarsening_factor must be numeric") from None
    if coarsening <= 1.0:
        info["skip_reason"] = "coarsening_factor_must_exceed_one"
        return None, info
    min_nx = max(1, int(settings.get("min_nx", 1)))
    min_ny = max(1, int(settings.get("min_ny", 1)))
    coarse_nx = max(min_nx, int(math.ceil(source_nx / coarsening)))
    coarse_ny = max(min_ny, int(math.ceil(source_ny / coarsening)))
    if coarse_nx >= source_nx and coarse_ny >= source_ny:
        info["skip_reason"] = "coarse_mesh_not_smaller"
        info["source_nx"] = source_nx
        info["source_ny"] = source_ny
        return None, info
    coarse_mesh = _generate_rectangle_mesh(
        {
            "x_range": [float(x_values[0]), float(x_values[-1])],
            "y_range": [float(y_values[0]), float(y_values[-1])],
            "nx": coarse_nx,
            "ny": coarse_ny,
            "element_type": etype,
            "material": next(iter(materials)),
            "integration": next(iter(integrations)),
        }
    )
    info.update(
        {
            "used": True,
            "coarsening_factor": coarsening,
            "source_nx": source_nx,
            "source_ny": source_ny,
            "coarse_nx": coarse_nx,
            "coarse_ny": coarse_ny,
            "source_node_count": len(mesh.node_ids),
            "source_element_count": len(active),
            "coarse_node_count": len(coarse_mesh.node_ids),
            "coarse_element_count": len(coarse_mesh.elements),
        }
    )
    return coarse_mesh, info


def _srm_structured_corner_axes(mesh: Mesh2D, elements: list[Element2D]) -> tuple[list[float], list[float]]:
    x_values: set[float] = set()
    y_values: set[float] = set()
    node_index = mesh.node_index
    for element in elements:
        etype = str(element.type).upper()
        corner_count = 4 if etype.startswith("QUAD") else 3
        for nid in element.nodes[:corner_count]:
            idx = node_index[str(nid)]
            x_values.add(round(float(mesh.coords[idx, 0]), 12))
            y_values.add(round(float(mesh.coords[idx, 1]), 12))
    return sorted(x_values), sorted(y_values)


def _srm_coarse_targets_supported(boundary_conditions: Any, loads: Any, mpc_constraints: Any) -> tuple[bool, str]:
    if _ensure_list(mpc_constraints):
        return False, "mpc_constraints_require_original_node_topology"
    for item in _ensure_list(boundary_conditions):
        if not isinstance(item, Mapping):
            continue
        if any(key in item for key in ("node", "nodes", "edge", "edges", "element", "elements")):
            return False, "boundary_conditions_reference_explicit_topology"
    for item in _ensure_list(loads):
        if not isinstance(item, Mapping):
            continue
        load_type = str(item.get("type", "")).lower().strip()
        if load_type in {"gravity", "self_weight", "body"} or bool(item.get("self_weight", False)):
            continue
        if any(key in item for key in ("node", "nodes", "edge", "edges", "element", "elements")):
            return False, "loads_reference_explicit_topology"
    return True, ""


def _srm_coarse_named_sets_supported(boundary_conditions: Any, loads: Any, coarse_mesh: Mesh2D) -> tuple[bool, str]:
    node_sets = set(coarse_mesh.node_sets)
    element_sets = set(coarse_mesh.element_sets)
    for item in _ensure_list(boundary_conditions):
        if not isinstance(item, Mapping):
            continue
        for key in ("set", "node_set", "nodeSet"):
            if key in item and str(item[key]) not in node_sets:
                return False, f"boundary_set_not_available_on_coarse_mesh:{item[key]}"
        for key in ("element_set", "elementSet"):
            if key in item and str(item[key]) not in element_sets:
                return False, f"boundary_element_set_not_available_on_coarse_mesh:{item[key]}"
    for item in _ensure_list(loads):
        if not isinstance(item, Mapping):
            continue
        for key in ("node_set", "nodeSet", "edge_set", "edgeSet"):
            if key in item and str(item[key]) not in node_sets:
                return False, f"load_set_not_available_on_coarse_mesh:{item[key]}"
        for key in ("element_set", "elementSet"):
            if key in item and str(item[key]) not in element_sets:
                return False, f"load_element_set_not_available_on_coarse_mesh:{item[key]}"
        if "set" in item and str(item["set"]) not in node_sets and str(item["set"]) not in element_sets:
            return False, f"load_set_not_available_on_coarse_mesh:{item['set']}"
    return True, ""


def _srm_coarse_search_config(srm_cfg: Mapping[str, Any], settings: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(srm_cfg)
    cfg["adaptive"] = True
    cfg["search_mode"] = "adaptive_bracket"
    if "max_bisection" in settings or "bisection_iterations" in settings:
        cfg["max_bisection"] = int(settings.get("max_bisection", settings.get("bisection_iterations", cfg.get("max_bisection", 3))))
    else:
        cfg["max_bisection"] = min(int(cfg.get("max_bisection", cfg.get("bisection_iterations", 8))), 3)
    if "factor_tol" in settings:
        cfg["factor_tol"] = float(settings["factor_tol"])
    if "bracket_stride" in settings:
        cfg["bracket_stride"] = int(settings["bracket_stride"])
    return cfg


def _srm_final_factors_from_coarse(
    factors: list[float],
    coarse_info: Mapping[str, Any],
    srm_cfg: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> list[float] | None:
    stable_raw = coarse_info.get("bracket_stable_factor", coarse_info.get("stable_factor"))
    failed_raw = coarse_info.get("bracket_failed_factor", coarse_info.get("failed_factor"))
    if stable_raw is None or failed_raw is None:
        return None
    try:
        stable = float(stable_raw)
        failed = float(failed_raw)
    except (TypeError, ValueError):
        return None
    if not (stable > 0.0 and failed > stable):
        return None
    ordered = sorted({_round_srm_factor(value) for value in factors})
    if len(ordered) <= 2:
        return ordered
    positive_steps = [b - a for a, b in zip(ordered, ordered[1:]) if b > a]
    base_step = min(positive_steps) if positive_steps else float(srm_cfg.get("factor_step", srm_cfg.get("step", 0.05)))
    margin_steps = int(settings.get("fine_margin_steps", settings.get("margin_steps", 1)))
    margin = max(0.0, float(margin_steps) * float(base_step))
    low = max(min(ordered), stable - margin)
    high = min(max(ordered), failed + margin)
    values = [value for value in ordered if low <= value <= high]
    values.extend([stable, failed])
    return sorted({_round_srm_factor(value) for value in values if value > 0.0})


def _srm_solver_with_retry_override(solver: Mapping[str, Any] | None, solver_override: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(solver_override, Mapping) or not solver_override:
        return solver
    base: dict[str, Any] = copy.deepcopy(dict(solver)) if isinstance(solver, Mapping) else {}
    continuation_checkpoint = solver_override.get("_srm_increment_checkpoint")
    if isinstance(continuation_checkpoint, _IncrementContinuationCheckpoint):
        base["_srm_increment_checkpoint"] = continuation_checkpoint
    speculative_cancel_path = solver_override.get(
        "_srm_speculative_cancel_path"
    )
    if speculative_cancel_path not in (None, ""):
        base["_srm_speculative_cancel_path"] = str(speculative_cancel_path)
    if _srm_bool(
        solver_override.get("_srm_capture_increment_checkpoint", False),
        False,
    ):
        base["_srm_capture_increment_checkpoint"] = True
    early_policy = solver_override.get("_srm_early_failure_policy")
    if isinstance(early_policy, Mapping):
        base["_srm_early_failure_policy"] = dict(early_policy)
    elif isinstance(early_policy, bool):
        base["_srm_early_failure_policy"] = {"enabled": bool(early_policy)}
    adaptive_policy = solver_override.get("_srm_adaptive_increment_policy")
    if isinstance(adaptive_policy, Mapping):
        _srm_apply_adaptive_increment_solver_override(base, adaptive_policy)
    if "_srm_mc_strict_tangent" in solver_override:
        base["_srm_mc_strict_tangent"] = _srm_bool(
            solver_override.get("_srm_mc_strict_tangent", False),
            False,
        )
    retry_enabled = _srm_bool(solver_override.get("_srm_auto_retry", False), False) or "_srm_retry_policy" in solver_override
    if not retry_enabled:
        return base
    policy = solver_override.get("_srm_retry_policy", {})
    if not isinstance(policy, Mapping):
        policy = {}
    newton = base.setdefault("newton", {})
    if not isinstance(newton, dict):
        newton = {}
        base["newton"] = newton
    max_iter_multiplier = float(policy.get("newton_max_iter_multiplier", 1.0) or 1.0)
    line_search_multiplier = float(policy.get("max_line_search_multiplier", 1.0) or 1.0)
    try:
        current_max_iter = int(newton.get("max_iter", base.get("max_iter", 60)))
        newton["max_iter"] = max(current_max_iter, int(math.ceil(current_max_iter * max_iter_multiplier)))
    except (TypeError, ValueError):
        newton["max_iter"] = 80
    try:
        current_line_search = int(newton.get("max_line_search", base.get("max_line_search", 20)))
        newton["max_line_search"] = max(current_line_search, int(math.ceil(current_line_search * line_search_multiplier)))
    except (TypeError, ValueError):
        newton["max_line_search"] = 20

    increments = base.setdefault("increments", {})
    if not isinstance(increments, dict):
        increments = {}
        base["increments"] = increments
    steps_multiplier = float(policy.get("steps_multiplier", 1.0) or 1.0)
    extra_cutbacks = int(policy.get("extra_cutbacks", 0) or 0)
    min_step_factor = float(policy.get("min_step_factor", 1.0) or 1.0)
    try:
        current_steps = int(increments.get("steps", 1))
        increments["steps"] = max(current_steps, int(math.ceil(current_steps * steps_multiplier)))
    except (TypeError, ValueError):
        increments["steps"] = 20
    try:
        current_cutbacks = int(increments.get("max_cutbacks", base.get("max_cutbacks", 0)))
        increments["max_cutbacks"] = max(current_cutbacks, current_cutbacks + max(extra_cutbacks, 0))
    except (TypeError, ValueError):
        increments["max_cutbacks"] = max(extra_cutbacks, 0)
    if isinstance(continuation_checkpoint, _IncrementContinuationCheckpoint):
        continuation_extra = max(
            1,
            int(policy.get("checkpoint_continuation_extra_cutbacks", 1) or 1),
        )
        increments["max_cutbacks"] = max(
            int(increments.get("max_cutbacks", 0) or 0),
            int(continuation_checkpoint.cutbacks) + continuation_extra,
        )
    try:
        current_min_step = float(increments.get("min_step", 1.0e-5))
        increments["min_step"] = max(current_min_step * max(min_step_factor, 1.0e-12), 1.0e-12)
    except (TypeError, ValueError):
        increments["min_step"] = 5.0e-6
    boundary_verification = solver_override.get("_srm_boundary_verification")
    if isinstance(boundary_verification, Mapping):
        base["_srm_boundary_verification"] = {
            **dict(boundary_verification),
            "policy": dict(policy),
        }
    else:
        base["_srm_auto_retry"] = {
            "enabled": True,
            "retry_index": solver_override.get("_srm_retry_index", ""),
            "policy": dict(policy),
        }
    return base


def _srm_apply_adaptive_increment_solver_override(base: dict[str, Any], policy: Mapping[str, Any]) -> None:
    if not _srm_bool(policy.get("enabled", True), True):
        return
    increments = base.setdefault("increments", {})
    if not isinstance(increments, dict):
        increments = {}
        base["increments"] = increments
    increments["enabled"] = True
    try:
        current_steps = max(1, int(increments.get("steps", 1) or 1))
    except (TypeError, ValueError):
        current_steps = 1
    target_factor = _srm_record_float(policy, "target_initial_step_factor", 1.0)
    min_factor = max(1.0e-3, min(1.0, _srm_record_float(policy, "min_initial_step_factor", 0.25)))
    max_factor = max(min_factor, min(1.0, _srm_record_float(policy, "max_initial_step_factor", 1.0)))
    if not math.isfinite(target_factor):
        target_factor = max_factor
    target_factor = min(max_factor, max(min_factor, target_factor))
    if _srm_bool(policy.get("use_final_step_size", True), True):
        final_step = _srm_record_float(policy, "final_step_size")
        base_step = 1.0 / float(current_steps)
        if math.isfinite(final_step) and final_step > 0.0 and base_step > 0.0:
            target_factor = min(target_factor, min(max_factor, max(min_factor, float(final_step) / base_step)))
    max_multiplier = max(1.0, _srm_record_float(policy, "max_steps_multiplier", 2.0))
    desired_steps = int(math.ceil(float(current_steps) / max(target_factor, 1.0e-12)))
    capped_steps = int(math.ceil(float(current_steps) * max_multiplier))
    adjusted_steps = max(current_steps, min(max(desired_steps, current_steps), max(capped_steps, current_steps)))
    increments["steps"] = adjusted_steps
    extra_cutbacks = max(0, _srm_record_int(policy, "extra_cutbacks", 0))
    try:
        current_cutbacks = int(increments.get("max_cutbacks", base.get("max_cutbacks", 0)) or 0)
    except (TypeError, ValueError):
        current_cutbacks = 0
    if extra_cutbacks > 0:
        increments["max_cutbacks"] = max(current_cutbacks, current_cutbacks + extra_cutbacks)
    elif current_cutbacks:
        increments["max_cutbacks"] = current_cutbacks
    min_step_factor = max(1.0e-12, _srm_record_float(policy, "min_step_factor", 1.0))
    try:
        current_min_step = float(increments.get("min_step", 1.0e-6) or 1.0e-6)
    except (TypeError, ValueError):
        current_min_step = 1.0e-6
    increments["min_step"] = max(current_min_step * min_step_factor, 1.0e-12)
    base["_srm_adaptive_increment_control"] = {
        "enabled": True,
        "policy": dict(policy),
        "original_steps": int(current_steps),
        "applied_steps": int(adjusted_steps),
        "applied_target_initial_step_factor": float(target_factor),
        "applied_max_cutbacks": increments.get("max_cutbacks", ""),
        "applied_min_step": increments.get("min_step", ""),
    }


def solve_axisymmetric_srm_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    output_dir: str | Path | None,
    solver: Mapping[str, Any] | None,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
) -> StageResult2D:
    cfg = solver if isinstance(solver, Mapping) else {}
    raw_srm_cfg = cfg.get("srm", {}) if isinstance(cfg.get("srm", {}), Mapping) else {}
    srm_cfg = _srm_with_runtime_context(raw_srm_cfg, cfg)
    factors = _srm_factors(srm_cfg)
    failure_plastic_ratio = float(srm_cfg.get("failure_plastic_ratio", srm_cfg.get("plastic_ratio_limit", 0.95)))
    lightweight_trials = bool(srm_cfg.get("lightweight_postprocess", srm_cfg.get("lightweight_trials", True)))
    coarse_to_fine_info: dict[str, Any] = {"enabled": False}

    def solve_trial(
        factor: float,
        *,
        postprocess_results: bool | None = None,
        solver_override: Mapping[str, Any] | None = None,
    ) -> StageResult2D:
        collect_results = (not lightweight_trials) if postprocess_results is None else bool(postprocess_results)
        trial_solver = _srm_solver_with_retry_override(solver, solver_override)
        warm_start_displacement = _srm_initial_displacement_from_solver_override(solver_override)
        return solve_axisymmetric_stage(
            mesh=mesh,
            materials=materials,
            boundary_conditions=boundary_conditions,
            loads=loads,
            mpc_constraints=mpc_constraints,
            stage_name=f"{stage_name}-FS{factor:g}",
            output_dir=None,
            solver=trial_solver,
            strength_factor=factor,
            plastic_state=plastic_state,
            initial_stresses=initial_stresses,
            interfaces=interfaces,
            structural_elements=structural_elements,
            initial_displacement=warm_start_displacement,
            postprocess_results=collect_results,
        )

    search_factors = factors
    search_cfg: Mapping[str, Any] = srm_cfg
    warm_start_supported = True
    coarse_settings = _srm_coarse_to_fine_settings(srm_cfg)
    if bool(coarse_settings.get("enabled", False)):
        coarse_mesh, coarse_to_fine_info = _srm_build_coarse_mesh(mesh, srm_cfg)
        supported, unsupported_reason = _srm_coarse_targets_supported(boundary_conditions, loads, mpc_constraints)
        if initial_stresses:
            supported = False
            unsupported_reason = "initial_stresses_require_original_topology"
        if interfaces or structural_elements:
            supported = False
            unsupported_reason = "interfaces_or_structural_elements_require_original_topology"
        if coarse_mesh is None:
            coarse_to_fine_info.setdefault("used", False)
        elif not supported:
            coarse_to_fine_info.update({"used": False, "skip_reason": unsupported_reason})
        else:
            named_supported, named_reason = _srm_coarse_named_sets_supported(boundary_conditions, loads, coarse_mesh)
            if not named_supported:
                coarse_to_fine_info.update({"used": False, "skip_reason": named_reason})
            else:
                def solve_coarse_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
                    trial_solver = _srm_solver_with_retry_override(solver, solver_override)
                    warm_start_displacement = _srm_initial_displacement_from_solver_override(solver_override)
                    return solve_axisymmetric_stage(
                        mesh=coarse_mesh,
                        materials=materials,
                        boundary_conditions=boundary_conditions,
                        loads=loads,
                        mpc_constraints=mpc_constraints,
                        stage_name=f"{stage_name}-coarse-FS{factor:g}",
                        output_dir=None,
                        solver=trial_solver,
                        strength_factor=factor,
                        plastic_state=None,
                        initial_stresses=None,
                        interfaces=None,
                        structural_elements=None,
                        initial_displacement=warm_start_displacement,
                        postprocess_results=False,
                    )

                coarse_search_cfg = _srm_coarse_search_config(srm_cfg, coarse_settings)
                _coarse_result, coarse_fos, coarse_trials, coarse_search_info = _run_srm_trial_search(
                    factors,
                    coarse_search_cfg,
                    failure_plastic_ratio,
                    solve_coarse_trial,
                    progress_stage_name=f"{stage_name}-coarse",
                    mesh=coarse_mesh,
                    warm_start_supported=warm_start_supported,
                )
                narrowed = _srm_final_factors_from_coarse(factors, coarse_search_info, srm_cfg, coarse_settings)
                coarse_to_fine_info.update(
                    {
                        "used": True,
                        "coarse_factor_of_safety": coarse_fos,
                        "coarse_trial_count": len(coarse_trials),
                        "coarse_search": {key: value for key, value in coarse_search_info.items() if key not in {"coarse_scan_factors"}},
                        "fine_factor_candidates": [] if narrowed is None else list(narrowed),
                        "fine_factor_count_before": len(factors),
                        "fine_factor_count_after": len(factors) if narrowed is None else len(narrowed),
                    }
                )
                if narrowed is not None:
                    search_factors = narrowed
                    final_cfg = dict(srm_cfg)
                    final_cfg["adaptive"] = True
                    final_cfg["search_mode"] = "adaptive_bracket"
                    search_cfg = final_cfg

    result, fos, trials, search_info = _run_srm_trial_search(
        search_factors,
        search_cfg,
        failure_plastic_ratio,
        solve_trial,
        progress_stage_name=stage_name,
        mesh=mesh,
        warm_start_supported=warm_start_supported,
    )
    if result is None:
        if not trials:
            raise FEM2DError(f"{stage_name}: axisymmetric SRM did not run any trial")
        result = _srm_failed_trial_result(mesh, structural_elements, stage_name, axisymmetric=True)
    selected_factor = _srm_selected_factor(fos, trials)
    if lightweight_trials and selected_factor is not None:
        result = solve_trial(selected_factor, postprocess_results=True)
        search_info["lightweight_postprocess"] = {
            "enabled": True,
            "trial_postprocess_results": False,
            "final_factor_reprocessed": selected_factor,
            "trial_plastic_ratio_source": "plastic_state_array_cache",
            "final_plastic_ratio_source": "plastic_state",
        }
    if coarse_to_fine_info.get("enabled", False):
        search_info["coarse_to_fine"] = coarse_to_fine_info
    search_info["trial_timing"] = _attach_srm_trial_timing(result, trials)
    result.name = stage_name
    result.solver_info["srm"] = {"factor_of_safety": fos, "trials": trials, "failure_plastic_ratio": failure_plastic_ratio, **search_info}
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
    return result


def solve_axisymmetric_coupled_up_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    output_dir: str | Path | None,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    strength_factor: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
    consolidation: Mapping[str, Any],
    previous_pressure: np.ndarray | None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
) -> StageResult2D:
    hydro = consolidation.get("hydro", consolidation.get("consolidation", consolidation))
    if not isinstance(hydro, Mapping):
        hydro = {}
    dt = float(hydro.get("dt", hydro.get("time_step", consolidation.get("dt", 1.0))))
    steps = int(hydro.get("steps", hydro.get("n_steps", consolidation.get("steps", 1))))
    if dt <= 0.0 or steps <= 0:
        raise FEM2DError("axisymmetric consolidation dt and steps must be positive")
    storage = float(hydro.get("storage", hydro.get("specific_storage", 1.0)))
    permeability = float(hydro.get("permeability", hydro.get("k", 1.0)))
    biot_alpha = float(hydro.get("biot_alpha", hydro.get("alpha", 1.0)))
    if storage <= 0.0 or permeability < 0.0:
        raise FEM2DError("axisymmetric consolidation storage must be positive and permeability non-negative")
    tangent_method = _tangent_method(solver)
    nonlinear = any(material.is_plastic for material in materials.values()) or _has_nonlinear_interfaces(interfaces) or structural_has_nonlinear(structural_elements)

    assembly_start = _perf_counter()
    constrained_u = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
    _add_inactive_node_constraints(mesh, constrained_u, interfaces=interfaces, structural_elements=structural_elements)
    fixed_p = _collect_pressure_constraints(mesh, hydro.get("pressure_bcs", hydro.get("pore_pressure_bcs", hydro.get("drainage", []))))
    u = np.zeros(structural_total_dofs(mesh, structural_elements), dtype=float)
    for dof, value in constrained_u.items():
        u[dof] = value
    p = _initial_pore_pressure(mesh, hydro, previous_pressure)
    for idx, value in fixed_p.items():
        p[idx] = value

    F = assemble_axisymmetric_load_vector(mesh, materials, loads)
    pressure_matrix_cache = build_pressure_matrix_assembly_cache(mesh, axisymmetric=True)
    M, H = assemble_pressure_matrices_cached(pressure_matrix_cache, mesh, materials, storage=storage, permeability=permeability)
    H_interface, interface_hydro_info = assemble_interface_hydraulic_transfer(mesh, interfaces, axisymmetric=True)
    biot_cache = build_biot_coupling_assembly_cache(mesh, axisymmetric=True)
    Bp = assemble_biot_coupling_matrix_cached(biot_cache, mesh, materials, alpha=biot_alpha)
    boundary_cache = build_pressure_boundary_term_cache(mesh, hydro, axisymmetric=True)
    hydraulic_direct_fill_info = {
        "pressure_matrices": pressure_matrix_cache.info(),
        "biot_coupling": biot_cache.info(),
        "boundary_terms": boundary_cache.info(),
    }
    K = assemble_axisymmetric_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    Kmpc, Fmpc, mpc_info = assemble_mpc_penalty(mesh, K, mpc_constraints)
    mpc_plan = mpc_stage_plan(
        mpc_constraints,
        solver,
        mpc_info,
        nonlinear=False,
        allow_elimination_linear=False,
        allow_lagrange_linear=True,
        add_plain_penalty_to_stage_matrix=True,
        add_penalty_when_exact_linear_blocked=True,
    )
    if mpc_plan.add_penalty_to_stage_matrix:
        K = (K + Kmpc).tocsr()
        F = F + Fmpc

    ndof = u.size
    npress = p.size
    fixed_all: dict[int, float] = dict(constrained_u)
    for idx, value in fixed_p.items():
        fixed_all[ndof + idx] = value
    free_all_dofs, fixed_all_dofs = _free_index_arrays(ndof + npress, fixed_all, stage_name=stage_name)
    fixed_all_values = np.asarray([fixed_all[int(dof)] for dof in fixed_all_dofs], dtype=float)
    free_pressure_dofs, _fixed_pressure_dofs = _free_index_arrays(npress, fixed_p, stage_name=stage_name, label="fixed pressure")
    pressure_base_lhs = (M / dt + H + H_interface).tocsr()

    consolidation_cache, consolidation_cache_reason = _build_consolidation_step_cache(
        mesh=mesh,
        materials=materials,
        hydro=hydro,
        solver=solver,
        consolidation=consolidation,
        axisymmetric=True,
        nonlinear=nonlinear,
        dt=dt,
        constrained_u=constrained_u,
        fixed_p=fixed_p,
        F=F,
        M=M,
        H=H,
        H_interface=H_interface,
        Bp=Bp,
        K=K,
        mpc_plan=mpc_plan,
        hydraulic_assembly_info=hydraulic_direct_fill_info,
    )
    consolidation_cache_stats = _consolidation_step_cache_stats()
    assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
    linear_solve_elapsed = 0.0
    coupled_assembly_elapsed = 0.0
    nonlinear_reduction_cache: ReducedMatrixCache | None = None
    lagrange_linear_cache: LagrangeMPCLinearCorrectionCache | None = None
    monolithic_lhs_pattern_cache: CoupledUPMonolithicMatrixCache | None = None
    monolithic_lhs_pattern_events: list[Mapping[str, Any]] = []
    if consolidation_cache is not None:
        monolithic_lhs_pattern_cache = consolidation_cache.monolithic_lhs_cache
    rhs_buffer: np.ndarray | None = np.empty(ndof + npress, dtype=float)
    linear_solver = _repeated_direct_linear_solver_config(solver)
    residual_norm = math.inf
    pressure_residual_norm = math.inf
    pressure_residual = np.zeros(npress, dtype=float)
    mass_balance_terms: dict[str, float] = {}
    liquefaction_coupling_info: dict[str, Any] = {"enabled": False, "enabled_points": 0}
    boundary_info: dict[str, Any] = {}
    seepage_toggles = 0
    seepage_outer_max = seepage_outer_iteration_limit(hydro)
    last_seepage_signature: tuple[int, int] | None = None
    step_history: list[dict[str, Any]] = []
    for step_index in range(steps):
        step_start = _perf_counter()
        step_coupled_assembly_elapsed = 0.0
        step_linear_solve_elapsed = 0.0
        step_reduced_matrix_elapsed = 0.0
        old_u = u.copy()
        old_p = p.copy()
        seepage_state = SeepageActiveSetState()
        Rb = csr_matrix((npress, npress), dtype=float)
        qb = np.zeros(npress, dtype=float)
        outer_iterations = 0
        for _outer in range(seepage_outer_max):
            outer_iterations = _outer + 1
            coupled_start = _perf_counter()
            if consolidation_cache is not None:
                Rb = consolidation_cache.boundary_matrix
                qb = consolidation_cache.boundary_rhs
                boundary_info = dict(consolidation_cache.boundary_info)
                Cliq = consolidation_cache.zero_pressure_matrix
                qliq = consolidation_cache.zero_pressure_vector
                liquefaction_coupling_info = {"enabled": False, "enabled_points": 0}
                pressure_lhs = consolidation_cache.pressure_lhs
                lhs = consolidation_cache.monolithic_lhs
                rhs_p = consolidation_cache.mass_over_dt @ old_p + consolidation_cache.biot_t_over_dt @ old_u + qb
                rhs = _fill_two_block_vector(rhs_buffer, consolidation_cache.load_vector, rhs_p)
            else:
                if nonlinear:
                    K = assemble_axisymmetric_algorithmic_tangent_stiffness(
                        mesh,
                        materials,
                        u,
                        strength_factor=strength_factor,
                        plastic_state=plastic_state,
                        initial_stresses=initial_stresses,
                        interfaces=interfaces,
                        structural_elements=structural_elements,
                        tangent_method=tangent_method,
                    )
                    if mpc_plan.add_penalty_to_stage_matrix:
                        K = (K + Kmpc).tocsr()
                Rb, qb, boundary_info = assemble_pressure_boundary_terms_cached(boundary_cache, pressure=p)
                consolidation_cache_stats["boundary_cache_reuses"] += 1
                Cliq, qliq, liquefaction_coupling_info = assemble_liquefaction_pressure_terms(
                    mesh,
                    materials,
                    hydro,
                    u,
                    old_u,
                    p,
                    dt=dt,
                    storage=storage,
                    axisymmetric=True,
                )
                pressure_lhs = (pressure_base_lhs + Rb + Cliq).tocsr()
                consolidation_cache_stats["pressure_lhs_reuses"] += 1
                lhs, monolithic_lhs_pattern_cache, monolithic_event = _assemble_coupled_up_monolithic_lhs(
                    K,
                    Bp,
                    pressure_lhs,
                    dt,
                    cache=monolithic_lhs_pattern_cache,
                )
                monolithic_lhs_pattern_events.append(monolithic_event)
                _record_coupled_up_monolithic_cache_event(consolidation_cache_stats, monolithic_event)
                rhs_p = (M @ old_p) / dt + (Bp.T @ old_u) / dt + qb + qliq
                rhs = _fill_two_block_vector(rhs_buffer, F, rhs_p)
            coupled_elapsed = max(_perf_counter() - coupled_start, 0.0)
            coupled_assembly_elapsed += coupled_elapsed
            step_coupled_assembly_elapsed += coupled_elapsed
            seepage_signature = (int(boundary_info.get("seepage_active_edges", 0)), int(boundary_info.get("seepage_inactive_edges", 0)))
            if last_seepage_signature is not None:
                if seepage_signature == last_seepage_signature:
                    consolidation_cache_stats["seepage_active_set_reuses"] += 1
                else:
                    consolidation_cache_stats["seepage_active_set_changes"] += 1
            last_seepage_signature = seepage_signature
            solve_start = _perf_counter()
            solution, mpc_residual_norm, nonlinear_reduction_cache, lagrange_linear_cache, solve_info = _solve_consolidation_monolithic_system(
                lhs,
                rhs,
                fixed_all,
                cache=consolidation_cache,
                free_all_dofs=None if consolidation_cache is not None else free_all_dofs,
                fixed_all_dofs=None if consolidation_cache is not None else fixed_all_dofs,
                fixed_all_values=None if consolidation_cache is not None else fixed_all_values,
                reduction_cache=nonlinear_reduction_cache if consolidation_cache is None else None,
                lagrange_cache=lagrange_linear_cache,
                validate_reduction_cache=True,
                mpc_plan=mpc_plan,
                mpc_info=mpc_info,
                stage_name=stage_name,
                solver=linear_solver,
                method="axisymmetric_monolithic_up_mpc_lagrange",
                cache_stats=consolidation_cache_stats,
                block_dof_ranges=[(0, ndof), (ndof, ndof + npress)],
            )
            solve_elapsed = max(_perf_counter() - solve_start, 0.0)
            linear_solve_elapsed += solve_elapsed
            step_linear_solve_elapsed += solve_elapsed
            step_reduced_matrix_elapsed += float(solve_info.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
            if mpc_residual_norm is not None:
                residual_norm = mpc_residual_norm
            u = solution[:ndof]
            p = solution[ndof : ndof + npress]
            residual = lhs @ solution - rhs
            free = consolidation_cache.free_all_dofs if consolidation_cache is not None else free_all_dofs
            if not mpc_plan.use_lagrange_linear:
                residual_norm = float(np.linalg.norm(residual[free])) if free.size else 0.0
            free_p = consolidation_cache.free_pressure_dofs if consolidation_cache is not None else free_pressure_dofs
            pressure_residual = residual[ndof : ndof + npress]
            pressure_residual_norm = float(np.linalg.norm(pressure_residual[free_p])) if free_p.size else 0.0
            seepage_state, seepage_done = advance_seepage_active_set(seepage_state, boundary_info)
            if seepage_done:
                break
        seepage_toggles += seepage_state.toggle_count
        storage_rate = (M @ (p - old_p)) / dt
        coupling_rate = (Bp.T @ (u - old_u)) / dt
        diffusion_flow = H @ p
        robin_flow = Rb @ p
        interface_transfer_flow = H_interface @ p
        liquefaction_dissipation_flow = Cliq @ p
        mass_balance_terms = {
            "storage_rate": float(np.sum(storage_rate)),
            "coupling_rate": float(np.sum(coupling_rate)),
            "diffusion_flow": float(np.sum(diffusion_flow)),
            "robin_flow": float(np.sum(robin_flow)),
            "interface_transfer_flow": float(np.sum(interface_transfer_flow)),
            "liquefaction_generation_source": float(liquefaction_coupling_info.get("generation_source", 0.0)),
            "liquefaction_dissipation_flow": float(np.sum(liquefaction_dissipation_flow)),
            "boundary_source": float(np.sum(qb)),
            "residual_sum": float(np.sum(pressure_residual)),
            "residual_norm": pressure_residual_norm,
        }
        step_history.append(
            {
                "step": step_index + 1,
                "time": float((step_index + 1) * dt),
                "pressure_residual_norm": pressure_residual_norm,
                "mass_balance_residual_sum": mass_balance_terms["residual_sum"],
                "flow_balance": mass_balance_terms["boundary_source"] - mass_balance_terms["diffusion_flow"] - mass_balance_terms["robin_flow"] - mass_balance_terms["interface_transfer_flow"],
                "max_pore_pressure": float(np.max(p)) if p.size else 0.0,
                "min_pore_pressure": float(np.min(p)) if p.size else 0.0,
                "seepage_toggle_count": seepage_state.toggle_count,
                "outer_iterations": outer_iterations,
                "step_cache_used": consolidation_cache is not None,
                "monolithic_lhs_source": "consolidation_step_cache" if consolidation_cache is not None else "assembled",
                "assembly_elapsed_seconds": step_coupled_assembly_elapsed,
                "coupled_assembly_elapsed_seconds": step_coupled_assembly_elapsed,
                "monolithic_assembly_elapsed_seconds": step_coupled_assembly_elapsed,
                "reduced_matrix_elapsed_seconds": step_reduced_matrix_elapsed,
                "linear_solve_elapsed_seconds": step_linear_solve_elapsed,
                "line_search_elapsed_seconds": 0.0,
                "postprocess_elapsed_seconds": 0.0,
                "elapsed_seconds": max(_perf_counter() - step_start, 0.0),
            }
        )
    effective_external = F + Bp @ p
    if any(material.is_plastic for material in materials.values()) or _has_nonlinear_interfaces(interfaces) or structural_has_nonlinear(structural_elements):
        fint = assemble_axisymmetric_internal_force(
            mesh,
            materials,
            u,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            initial_stresses=initial_stresses,
            interfaces=interfaces,
            structural_elements=structural_elements,
        )
        reactions = fint - effective_external
    else:
        reactions = K @ u - effective_external
    post_start = _perf_counter()
    state_for_results = _update_liquefaction_state_from_pore_pressure(mesh, materials, p, plastic_state)
    postprocess_state_info: dict[str, Any] = {}
    element_results, updated_plastic_state = compute_axisymmetric_element_results_and_state(
        mesh,
        materials,
        u,
        strength_factor=strength_factor,
        plastic_state=state_for_results,
        initial_stresses=initial_stresses,
        postprocess_info=postprocess_state_info,
    )
    active_elements = [element.id for element in mesh.elements if element.active]
    result = StageResult2D(
        stage_name,
        u,
        reactions,
        element_results,
        constrained_u,
        active_elements,
        {
            "method": "axisymmetric_monolithic_up",
            "geometry": "axisymmetric",
            "tangent": tangent_method,
            "iterations": steps,
            "residual_norm": residual_norm,
            "converged": True,
            "postprocess_state_commit": postprocess_state_info,
            "consolidation": {
                "dt": dt,
                "steps": steps,
                "storage": storage,
                "permeability": permeability,
                "biot_alpha": biot_alpha,
                "fixed_pressure_nodes": len(fixed_p),
                "unknowns": int(ndof + npress),
                "boundary": boundary_info,
                "interface_transfer": interface_hydro_info,
                "hydraulic_cache": hydraulic_direct_fill_info,
                "liquefaction_coupling": liquefaction_coupling_info,
                "seepage_toggle_count": seepage_toggles,
                "seepage_active_edges": boundary_info.get("seepage_active_edges", 0),
                "mass_balance": pressure_residual_norm,
                "mass_balance_residual_sum": mass_balance_terms.get("residual_sum", 0.0),
                "mass_balance_terms": mass_balance_terms,
                "pressure_dof_count": npress,
                "drainage_boundary_count": len(fixed_p),
                "flow_balance": step_history[-1]["flow_balance"] if step_history else 0.0,
                "pressure_converged": pressure_residual_norm <= 1.0e-8,
                "step_history": step_history,
                "step_cache": _consolidation_step_cache_info(consolidation_cache, consolidation_cache_reason, consolidation_cache_stats),
                "monolithic_lhs_pattern_cache": _coupled_up_monolithic_cache_summary(monolithic_lhs_pattern_events, monolithic_lhs_pattern_cache),
                "lagrange_linear_cache": {"enabled": False} if lagrange_linear_cache is None else lagrange_linear_cache.info(),
                "history_storage_policy": "intermediate steps store pore pressure extrema, flow balance, and convergence metrics only",
                "rollback_policy": "displacement, pore pressure, plastic state, and time-integration history are committed only after an accepted time step",
            },
        },
    )
    if mpc_info["count"]:
        applied_method = "lagrange" if mpc_plan.lagrange_requested else "penalty"
        result.solver_info["mpc"] = {**mpc_info, "max_violation": mpc_violation(mesh, u, mpc_info), "applied_method": applied_method}
    result.pore_pressure = p
    result.time = dt * steps
    result.plastic_state = updated_plastic_state
    liquefaction_info = _liquefaction_state_summary(updated_plastic_state)
    if liquefaction_info.get("count", 0):
        result.solver_info["liquefaction"] = liquefaction_info
        result.solver_info["consolidation"]["liquefaction"] = liquefaction_info
    result.interface_results = compute_interface_results(mesh, interfaces, u, axisymmetric=True)
    result.structural_results = compute_structural_results(mesh, materials, structural_elements, u, axisymmetric=True, loads=loads)
    _attach_integration_point_results(
        result,
        mesh,
        materials,
        u,
        axisymmetric=True,
        strength_factor=strength_factor,
        plastic_state=state_for_results,
        initial_stresses=initial_stresses,
    )
    perf = result.solver_info.setdefault("performance", {})
    if not isinstance(perf, dict):
        perf = {}
        result.solver_info["performance"] = perf
    perf.update(
        {
            "assembly_elapsed_seconds": assembly_elapsed,
            "linear_solve_elapsed_seconds": linear_solve_elapsed,
            "postprocess_elapsed_seconds": max(_perf_counter() - post_start, 0.0),
            "coupled_assembly_elapsed_seconds": coupled_assembly_elapsed,
        }
    )
    if nonlinear:
        perf["nonlinear_solve_elapsed_seconds"] = linear_solve_elapsed
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
    return result


def solve_axisymmetric_riks_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    output_dir: str | Path | None,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    consolidation: Mapping[str, Any] | None = None,
    previous_pressure: np.ndarray | None = None,
) -> StageResult2D:
    riks_cfg = solver.get("riks", {}) if isinstance(solver, Mapping) and isinstance(solver.get("riks", {}), Mapping) else {}
    return solve_axisymmetric_arc_length_stage(
        mesh=mesh,
        materials=materials,
        boundary_conditions=boundary_conditions,
        loads=loads,
        mpc_constraints=mpc_constraints,
        stage_name=stage_name,
        output_dir=output_dir,
        solver=solver,
        initial_stresses=initial_stresses,
        plastic_state=plastic_state,
        interfaces=interfaces,
        structural_elements=structural_elements,
        consolidation=consolidation,
        previous_pressure=previous_pressure,
        riks_cfg=riks_cfg,
    )


def solve_axisymmetric_arc_length_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    output_dir: str | Path | None,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    consolidation: Mapping[str, Any] | None = None,
    previous_pressure: np.ndarray | None = None,
    riks_cfg: Mapping[str, Any],
) -> StageResult2D:
    steps = int(riks_cfg.get("steps", riks_cfg.get("increments", 10)))
    if steps <= 0:
        raise FEM2DError("axisymmetric Riks steps must be positive")
    lambda_max = float(riks_cfg.get("lambda_max", riks_cfg.get("load_factor", 1.0)))
    if lambda_max <= 0.0:
        raise FEM2DError("axisymmetric Riks lambda_max must be positive")
    psi = float(riks_cfg.get("psi", riks_cfg.get("load_scale", 1.0)))
    max_iter = int(riks_cfg.get("max_iter", 20))
    tol = float(riks_cfg.get("tol", 1.0e-8))
    tangent_method = _tangent_method(solver)
    hydro = _hydro_mapping(consolidation)
    coupled_up = hydro is not None
    dt = 1.0
    storage = 1.0
    permeability = 1.0
    biot_alpha = 1.0
    if coupled_up:
        dt = float(hydro.get("dt", hydro.get("time_step", 1.0)))
        storage = float(hydro.get("storage", hydro.get("specific_storage", 1.0)))
        permeability = float(hydro.get("permeability", hydro.get("k", 1.0)))
        biot_alpha = float(hydro.get("biot_alpha", hydro.get("alpha", 1.0)))
        if dt <= 0.0:
            raise FEM2DError("axisymmetric Riks u-p dt must be positive")
        if storage <= 0.0 or permeability < 0.0:
            raise FEM2DError("axisymmetric Riks u-p storage must be positive and permeability non-negative")

    constrained = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
    _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)
    ndof = len(mesh.node_ids) * 2
    free, fixed = _free_index_arrays(ndof, constrained, stage_name=stage_name)
    if free.size == 0:
        raise FEM2DError(f"{stage_name}: axisymmetric Riks requires at least one free displacement dof")
    axisymmetric_step_cache = build_axisymmetric_step_cache(
        mesh,
        boundary_conditions,
        interfaces=interfaces,
        structural_elements=structural_elements,
        precompute_stiffness_pattern=bool(riks_cfg.get("precompute_stiffness_pattern", riks_cfg.get("precompute_sparse_pattern", True))),
    )
    if axisymmetric_step_cache.ndof == ndof:
        free = axisymmetric_step_cache.free_dofs
        fixed = axisymmetric_step_cache.fixed_dofs
    else:
        axisymmetric_step_cache = None
    sparse_pattern = axisymmetric_step_cache.stiffness_pattern if axisymmetric_step_cache is not None else None

    reference_load = assemble_axisymmetric_load_vector(mesh, materials, loads)
    elastic_reference = assemble_axisymmetric_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    Kmpc, Fmpc, mpc_info = assemble_mpc_penalty(mesh, elastic_reference, mpc_constraints)
    mpc_plan = mpc_arc_length_stage_plan(mpc_constraints, solver, mpc_info)
    u = np.zeros(ndof, dtype=float)
    for dof, value in constrained.items():
        u[dof] = value
    fixed_p: dict[int, float] = {}
    p = np.zeros(len(mesh.node_ids), dtype=float)
    M = csr_matrix((len(mesh.node_ids), len(mesh.node_ids)), dtype=float)
    H = csr_matrix((len(mesh.node_ids), len(mesh.node_ids)), dtype=float)
    H_interface = csr_matrix((len(mesh.node_ids), len(mesh.node_ids)), dtype=float)
    Bp = csr_matrix((ndof, len(mesh.node_ids)), dtype=float)
    interface_hydro_info: dict[str, Any] = {"count": 0, "conductance_total": 0.0}
    boundary_info: dict[str, Any] = {}
    mass_balance_terms: dict[str, float] = {}
    pressure_residual_norm = 0.0
    seepage_toggles = 0
    riks_up_hydraulic_cache_info: dict[str, Any] = {"enabled": False}
    riks_up_boundary_cache_reuses = 0
    pressure_boundary_cache: Any | None = None
    pressure_base_lhs = csr_matrix((len(mesh.node_ids), len(mesh.node_ids)), dtype=float)
    if hydro is not None:
        fixed_p = _collect_pressure_constraints(mesh, hydro.get("pressure_bcs", hydro.get("pore_pressure_bcs", hydro.get("drainage", []))))
        p = _initial_pore_pressure(mesh, hydro, previous_pressure)
        for idx, value in fixed_p.items():
            p[idx] = value
        pressure_matrix_cache = build_pressure_matrix_assembly_cache(mesh, axisymmetric=True)
        M, H = assemble_pressure_matrices_cached(pressure_matrix_cache, mesh, materials, storage=storage, permeability=permeability)
        H_interface, interface_hydro_info = assemble_interface_hydraulic_transfer(mesh, interfaces, axisymmetric=True)
        biot_cache = build_biot_coupling_assembly_cache(mesh, axisymmetric=True)
        Bp = assemble_biot_coupling_matrix_cached(biot_cache, mesh, materials, alpha=biot_alpha)
        pressure_boundary_cache = build_pressure_boundary_term_cache(mesh, hydro, axisymmetric=True)
        pressure_base_lhs = (M / dt + H + H_interface).tocsr()
        riks_up_hydraulic_cache_info = {
            "enabled": True,
            "pressure_matrices": pressure_matrix_cache.info(),
            "biot_coupling": biot_cache.info(),
            "boundary_terms": pressure_boundary_cache.info(),
            "pressure_base_lhs_cached": True,
        }
    lam = 0.0
    state_current: dict[str, PlasticState2D] = dict(plastic_state or {})
    linear_solver = _repeated_direct_linear_solver_config(solver)
    plastic_state_cache = (
        build_plastic_state_array_cache(mesh, materials, state_current)
        if state_current or any(material.is_plastic for material in materials.values())
        else None
    )
    initial_stress_cache = build_initial_stress_array_cache(mesh, initial_stresses) if initial_stresses else None
    reduction_cache: ReducedMatrixCache | None = axisymmetric_step_cache.reduced_matrix_cache if axisymmetric_step_cache is not None else None
    reduction_cache_events: list[Mapping[str, Any]] = []
    symbolic_cache_events: list[Mapping[str, Any]] = []
    augmented_cache: ArcLengthAugmentedMatrixCache | None = None
    augmented_cache_events: list[Mapping[str, Any]] = []
    lagrange_linear_cache: LagrangeMPCLinearCorrectionCache | None = None
    lagrange_linear_cache_events: list[Mapping[str, Any]] = []
    lagrange_correction_cache: ArcLengthLagrangeCorrectionCache | None = None
    lagrange_correction_cache_events: list[Mapping[str, Any]] = []
    up_correction_cache: AxisymmetricUPArcLengthCorrectionCache | None = None
    up_correction_cache_events: list[Mapping[str, Any]] = []
    up_lagrange_correction_cache: AxisymmetricUPArcLengthLagrangeCorrectionCache | None = None
    up_lagrange_correction_cache_events: list[Mapping[str, Any]] = []
    combined_tangent_internal_assembly = False
    tangent0 = assemble_axisymmetric_algorithmic_tangent_stiffness(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        initial_stress_cache=initial_stress_cache,
        plastic_state=state_current,
        plastic_state_cache=plastic_state_cache,
        interfaces=interfaces,
        structural_elements=structural_elements,
        tangent_method=tangent_method,
        sparse_pattern=sparse_pattern,
    )
    if mpc_plan.add_penalty_to_stage_matrix:
        tangent0 = (tangent0 + Kmpc).tocsr()
    if mpc_plan.lagrange_requested:
        du_ref_full, lagrange_linear_cache, _info = _solve_lagrange_mpc_linear_correction_cached(
            tangent0,
            reference_load,
            constrained,
            mpc_info,
            u,
            stage_name=stage_name,
            solver=linear_solver,
            cache=lagrange_linear_cache,
        )
        lagrange_linear_cache_events.append(_info)
        du_ref = du_ref_full[free]
    else:
        du_ref, info, reduction_cache = solve_reduced_linear_system(
            tangent0,
            reference_load,
            free,
            fixed,
            fixed_values=np.zeros(fixed.size, dtype=float),
            reduction_cache=reduction_cache,
            stage_name=stage_name,
            solver=linear_solver,
            validate_cache=sparse_pattern is None or mpc_plan.add_penalty_to_stage_matrix,
        )
        cache_event = info.get("reduced_matrix_cache", {})
        if isinstance(cache_event, Mapping):
            reduction_cache_events.append(cache_event)
        symbolic_event = info.get("symbolic_cache", {})
        if isinstance(symbolic_event, Mapping):
            symbolic_cache_events.append(symbolic_event)
    arc_length = riks_cfg.get("arc_length", riks_cfg.get("ds"))
    if arc_length is None:
        arc_length_value = (lambda_max / steps) * math.sqrt(float(du_ref @ du_ref) + psi * psi)
    else:
        arc_length_value = float(arc_length)
    if arc_length_value <= 0.0:
        raise FEM2DError("axisymmetric Riks arc_length must be positive")

    max_cutbacks = int(riks_cfg.get("max_cutbacks", riks_cfg.get("cutbacks", 0)))
    cutback_factor = float(riks_cfg.get("cutback_factor", 0.5))
    growth = float(riks_cfg.get("growth", 1.0))
    min_arc_length = float(riks_cfg.get("min_arc_length", riks_cfg.get("min_ds", arc_length_value * 1.0e-6)))
    if max_cutbacks < 0:
        raise FEM2DError("axisymmetric Riks max_cutbacks must be non-negative")
    if not (0.0 < cutback_factor < 1.0):
        raise FEM2DError("axisymmetric Riks cutback_factor must satisfy 0 < factor < 1")
    if growth <= 0.0:
        raise FEM2DError("axisymmetric Riks growth must be positive")
    if min_arc_length <= 0.0:
        raise FEM2DError("axisymmetric Riks min_arc_length must be positive")

    path: list[dict[str, Any]] = []
    iteration_history: list[dict[str, Any]] = []
    cutback_log: list[dict[str, Any]] = []
    previous_dlambda: float | None = None
    previous_du_step: np.ndarray | None = None
    direction_flips = 0
    predictor_flips = 0
    negative_dlambda = 0
    total_cutbacks = 0
    current_arc_length = arc_length_value
    for step in range(1, steps + 1):
        step_start = _perf_counter()
        step_iteration_start = len(iteration_history)
        local_cutbacks = 0
        predictor_flip = False
        while True:
            try:
                tangent = assemble_axisymmetric_algorithmic_tangent_stiffness(
                    mesh,
                    materials,
                    u,
                    initial_stresses=initial_stresses,
                    initial_stress_cache=initial_stress_cache,
                    plastic_state=state_current,
                    plastic_state_cache=plastic_state_cache,
                    interfaces=interfaces,
                    structural_elements=structural_elements,
                    tangent_method=tangent_method,
                    sparse_pattern=sparse_pattern,
                )
                if mpc_plan.add_penalty_to_stage_matrix:
                    tangent = (tangent + Kmpc).tocsr()
                if mpc_plan.lagrange_requested:
                    du_load_full, lagrange_linear_cache, _info = _solve_lagrange_mpc_linear_correction_cached(
                        tangent,
                        reference_load,
                        constrained,
                        mpc_info,
                        u,
                        stage_name=stage_name,
                        solver=linear_solver,
                        cache=lagrange_linear_cache,
                    )
                    lagrange_linear_cache_events.append(_info)
                    du_load = du_load_full[free]
                else:
                    du_load, info, reduction_cache = solve_reduced_linear_system(
                        tangent,
                        reference_load,
                        free,
                        fixed,
                        fixed_values=np.zeros(fixed.size, dtype=float),
                        reduction_cache=reduction_cache,
                        stage_name=stage_name,
                        solver=linear_solver,
                        validate_cache=sparse_pattern is None or mpc_plan.add_penalty_to_stage_matrix,
                    )
                    cache_event = info.get("reduced_matrix_cache", {})
                    if isinstance(cache_event, Mapping):
                        reduction_cache_events.append(cache_event)
                    symbolic_event = info.get("symbolic_cache", {})
                    if isinstance(symbolic_event, Mapping):
                        symbolic_cache_events.append(symbolic_event)
                dlam = current_arc_length / max(math.sqrt(float(du_load @ du_load) + psi * psi), np.finfo(float).eps)
                predictor_flip = False
                if previous_du_step is not None and float(du_load @ previous_du_step) < 0.0:
                    dlam = -dlam
                    predictor_flip = True
                    predictor_flips += 1
                u_prev = u.copy()
                p_prev = p.copy()
                lam_prev = lam
                u_trial = u.copy()
                u_trial[free] += dlam * du_load
                p_trial = p.copy()
                lam_trial = lam + dlam
                converged = False
                residual_norm = math.inf
                pressure_residual_norm = 0.0
                constraint_value = math.inf
                iteration = 0
                seepage_state = SeepageActiveSetState()
                pressure_lhs = csr_matrix((len(mesh.node_ids), len(mesh.node_ids)), dtype=float)
                qb = np.zeros(len(mesh.node_ids), dtype=float)
                pressure_residual = np.zeros(len(mesh.node_ids), dtype=float)
                free_p, _fixed_pressure = _free_index_arrays(
                    len(mesh.node_ids),
                    fixed_p,
                    stage_name=stage_name,
                    label="fixed pressure",
                )
                for iteration in range(1, max_iter + 2):
                    iteration_start = _perf_counter()
                    assembly_start = _perf_counter()
                    tangent, fint = assemble_axisymmetric_tangent_and_internal_force(
                        mesh,
                        materials,
                        u_trial,
                        initial_stresses=initial_stresses,
                        initial_stress_cache=initial_stress_cache,
                        plastic_state=state_current,
                        plastic_state_cache=plastic_state_cache,
                        interfaces=interfaces,
                        structural_elements=structural_elements,
                        tangent_method=tangent_method,
                        sparse_pattern=sparse_pattern,
                    )
                    assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
                    combined_tangent_internal_assembly = True
                    pressure_reference_norm = 0.0
                    if hydro is not None:
                        if pressure_boundary_cache is not None:
                            Rb, qb, boundary_info = assemble_pressure_boundary_terms_cached(pressure_boundary_cache, pressure=p_trial)
                            riks_up_boundary_cache_reuses += 1
                        else:
                            Rb, qb, boundary_info = assemble_axisymmetric_pressure_boundary_terms(mesh, hydro, pressure=p_trial)
                        pressure_lhs = (pressure_base_lhs + Rb).tocsr()
                        pressure_rhs = (M @ p_prev) / dt + (Bp.T @ u_prev) / dt + qb
                        pressure_residual = pressure_lhs @ p_trial + (Bp.T @ u_trial) / dt - pressure_rhs
                        pressure_reference_norm = float(np.linalg.norm(pressure_rhs[free_p])) if free_p.size else 0.0
                        seepage_state = observe_seepage_active_set(seepage_state, boundary_info)
                        residual = fint - lam_trial * reference_load - Bp @ p_trial
                    else:
                        residual = fint - lam_trial * reference_load
                    if mpc_plan.add_penalty_to_stage_matrix:
                        residual = residual + Kmpc @ u_trial - Fmpc
                        tangent = (tangent + Kmpc).tocsr()
                    residual_free = residual[free]
                    if hydro is not None:
                        pressure_residual_norm = float(np.linalg.norm(pressure_residual[free_p])) if free_p.size else 0.0
                    du_step = u_trial[free] - u_prev[free]
                    dl_step = lam_trial - lam_prev
                    constraint_value = float(du_step @ du_step + (psi * dl_step) ** 2 - current_arc_length**2)
                    if mpc_plan.lagrange_requested:
                        residual_norm, mpc_constraint_norm, _mpc_multipliers = _lagrange_mpc_projected_residual(residual, constrained, mpc_info, u_trial, stage_name)
                    else:
                        residual_norm = float(np.linalg.norm(residual_free))
                        mpc_constraint_norm = 0.0
                    riks_convergence = _riks_convergence_metrics(
                        force_norm=residual_norm,
                        force_reference=float(np.linalg.norm(reference_load[free])),
                        pressure_norm=pressure_residual_norm,
                        pressure_reference=pressure_reference_norm,
                        pressure_enabled=hydro is not None,
                        arc_residual=constraint_value,
                        arc_reference=current_arc_length**2,
                        mpc_norm=mpc_constraint_norm,
                        mpc_reference=max(float(np.linalg.norm(u_trial[free])), current_arc_length),
                        mpc_enabled=mpc_plan.lagrange_requested,
                        riks_cfg=riks_cfg,
                        legacy_tol=tol,
                    )
                    converged = bool(riks_convergence["converged"])
                    iteration_row: dict[str, Any] = {
                        "step": step,
                        "iteration": iteration,
                        "residual_norm": residual_norm,
                        "pressure_residual_norm": pressure_residual_norm,
                        "constraint_residual": constraint_value,
                        "converged": converged,
                        **riks_convergence,
                        "assembly_elapsed_seconds": assembly_elapsed,
                        "tangent_internal_assembly_elapsed_seconds": assembly_elapsed,
                        "tangent_assembly_elapsed_seconds": assembly_elapsed,
                        "internal_force_assembly_elapsed_seconds": 0.0,
                        "monolithic_assembly_elapsed_seconds": 0.0,
                        "reduced_matrix_elapsed_seconds": 0.0,
                        "linear_solve_elapsed_seconds": 0.0,
                        "line_search_elapsed_seconds": 0.0,
                        "postprocess_elapsed_seconds": 0.0,
                    }
                    if converged:
                        iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
                        iteration_history.append(iteration_row)
                        break
                    if iteration > max_iter:
                        iteration_row["final_correction_check"] = True
                        iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
                        iteration_history.append(iteration_row)
                        break
                    solve_start = _perf_counter()
                    if hydro is not None:
                        if mpc_plan.lagrange_requested:
                            (
                                du_corr,
                                dp_corr,
                                dl_corr,
                                _mpc_multipliers,
                                up_lagrange_correction_cache,
                                cache_event,
                            ) = _solve_axisymmetric_up_arc_length_lagrange_correction_cached(
                                tangent,
                                Bp,
                                pressure_lhs,
                                reference_load,
                                residual,
                                pressure_residual,
                                mpc_info,
                                u_trial,
                                free,
                                free_p,
                                du_step,
                                dl_step,
                                constraint_value,
                                psi,
                                stage_name=stage_name,
                                solver=linear_solver,
                                dt=dt,
                                cache=up_lagrange_correction_cache,
                            )
                            up_lagrange_correction_cache_events.append(cache_event)
                            cache_profile = cache_event.get("profile", {})
                            if isinstance(cache_profile, Mapping):
                                iteration_row["monolithic_assembly_elapsed_seconds"] = float(cache_profile.get("bmat_elapsed_seconds", 0.0) or 0.0)
                                iteration_row["linear_solve_elapsed_seconds"] = float(cache_profile.get("linear_solve_elapsed_seconds", 0.0) or 0.0)
                                iteration_row["lagrange_constraint_matrix_elapsed_seconds"] = float(cache_profile.get("constraint_matrix_elapsed_seconds", 0.0) or 0.0)
                                iteration_row["lagrange_cache_build_elapsed_seconds"] = float(cache_profile.get("cache_build_elapsed_seconds", 0.0) or 0.0)
                        else:
                            du_corr, dp_corr, dl_corr, up_correction_cache, cache_event = _solve_axisymmetric_up_arc_length_correction_cached(
                                tangent,
                                Bp,
                                pressure_lhs,
                                reference_load,
                                residual,
                                pressure_residual,
                                free,
                                free_p,
                                du_step,
                                dl_step,
                                constraint_value,
                                psi,
                                stage_name=stage_name,
                                solver=linear_solver,
                                dt=dt,
                                cache=up_correction_cache,
                            )
                            up_correction_cache_events.append(cache_event)
                        u_trial[free] += du_corr
                        if free_p.size:
                            p_trial[free_p] += dp_corr
                        lam_trial += dl_corr
                    elif mpc_plan.lagrange_requested:
                        du_corr, dl_corr, _mpc_multipliers, lagrange_correction_cache, cache_event = _solve_arc_length_lagrange_correction_cached(
                            tangent,
                            reference_load,
                            residual,
                            constrained,
                            mpc_info,
                            u_trial,
                            free,
                            du_step,
                            dl_step,
                            constraint_value,
                            psi,
                            stage_name=stage_name,
                            solver=linear_solver,
                            cache=lagrange_correction_cache,
                        )
                        lagrange_correction_cache_events.append(cache_event)
                        u_trial[free] += du_corr
                        lam_trial += dl_corr
                    else:
                        reduced_start = _perf_counter()
                        system, reduction_cache, augmented_cache = _arc_length_augmented_system_direct_fill(
                            tangent,
                            free,
                            fixed,
                            reference_load[free],
                            du_step,
                            dl_step,
                            psi,
                            reduction_cache=reduction_cache,
                            reduction_cache_events=reduction_cache_events,
                            augmented_cache=augmented_cache,
                            augmented_cache_events=augmented_cache_events,
                            validate_reduction_cache=sparse_pattern is None or mpc_plan.add_penalty_to_stage_matrix,
                        )
                        iteration_row["reduced_matrix_elapsed_seconds"] = max(_perf_counter() - reduced_start, 0.0)
                        rhs = _fill_two_block_vector(None, -residual_free, np.asarray([-constraint_value], dtype=float))
                        correction, _corr_info = solve_linear_system(system, rhs, stage_name=stage_name, solver=linear_solver)
                        u_trial[free] += correction[:-1]
                        lam_trial += float(correction[-1])
                    for dof, value in constrained.items():
                        u_trial[dof] = value
                    for idx, value in fixed_p.items():
                        p_trial[idx] = value
                    correction_elapsed = max(_perf_counter() - solve_start, 0.0)
                    iteration_row["correction_elapsed_seconds"] = correction_elapsed
                    if float(iteration_row.get("linear_solve_elapsed_seconds", 0.0) or 0.0) <= 0.0:
                        iteration_row["linear_solve_elapsed_seconds"] = correction_elapsed
                    iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
                    iteration_history.append(iteration_row)
                if not converged:
                    raise FEM2DError(f"axisymmetric arc-length step {step} did not converge, residual={residual_norm:.6e}")
                seepage_toggles += seepage_state.toggle_count
                break
            except FEM2DError as exc:
                failed_seepage_state = locals().get("seepage_state")
                if isinstance(failed_seepage_state, SeepageActiveSetState):
                    seepage_toggles += failed_seepage_state.toggle_count
                if local_cutbacks >= max_cutbacks:
                    raise FEM2DError(f"{stage_name}: axisymmetric Riks step {step} failed after {local_cutbacks} cutbacks: {exc}") from exc
                next_arc = current_arc_length * cutback_factor
                if next_arc < min_arc_length:
                    raise FEM2DError(f"{stage_name}: axisymmetric Riks arc-length fell below min_arc_length after cutback: {exc}") from exc
                total_cutbacks += 1
                local_cutbacks += 1
                cutback_log.append({"step": step, "arc_length": current_arc_length, "next_arc_length": next_arc, "error": str(exc)})
                current_arc_length = next_arc
        accepted_dlambda = float(lam_trial - lam_prev)
        direction_flip = previous_dlambda is not None and accepted_dlambda * previous_dlambda < 0.0
        if direction_flip:
            direction_flips += 1
        if accepted_dlambda < 0.0:
            negative_dlambda += 1
        previous_dlambda = accepted_dlambda
        previous_du_step = u_trial[free] - u_prev[free]
        if hydro is not None:
            storage_rate = (M @ (p_trial - p_prev)) / dt
            coupling_rate = (Bp.T @ (u_trial - u_prev)) / dt
            diffusion_flow = H @ p_trial
            robin_flow = (pressure_lhs - (M / dt + H + H_interface)) @ p_trial if pressure_lhs.shape[0] else np.zeros_like(p_trial)
            interface_transfer_flow = H_interface @ p_trial
            mass_balance_terms = {
                "storage_rate": float(np.sum(storage_rate)),
                "coupling_rate": float(np.sum(coupling_rate)),
                "diffusion_flow": float(np.sum(diffusion_flow)),
                "robin_flow": float(np.sum(robin_flow)),
                "interface_transfer_flow": float(np.sum(interface_transfer_flow)),
                "boundary_source": float(np.sum(qb)),
                "residual_sum": float(np.sum(pressure_residual)),
                "residual_norm": pressure_residual_norm,
            }
        postprocess_start = _perf_counter()
        step_postprocess_state_info: dict[str, Any] = {}
        rows, state_current = compute_axisymmetric_element_results_and_state(
            mesh,
            materials,
            u_trial,
            initial_stresses=initial_stresses,
            plastic_state=state_current,
            postprocess_info=step_postprocess_state_info,
        )
        postprocess_elapsed = max(_perf_counter() - postprocess_start, 0.0)
        plastic_state_cache = (
            build_plastic_state_array_cache(mesh, materials, state_current)
            if state_current or any(material.is_plastic for material in materials.values())
            else None
        )
        u = u_trial
        if hydro is not None:
            p = p_trial
        lam = lam_trial
        max_disp = float(max((math.hypot(u[2 * i], u[2 * i + 1]) for i in range(len(mesh.node_ids))), default=0.0))
        step_iterations = iteration_history[step_iteration_start:]
        path.append(
            {
                "step": step,
                "lambda": float(lam),
                "max_displacement": max_disp,
                "iterations": iteration,
                "residual_norm": residual_norm,
                "pressure_residual_norm": pressure_residual_norm,
                "constraint_residual": constraint_value,
                "delta_lambda": accepted_dlambda,
                "arc_length": current_arc_length,
                "cutbacks": local_cutbacks,
                "predictor_flip": predictor_flip,
                "direction_flip": direction_flip,
                "negative_dlambda": accepted_dlambda < 0.0,
                "plastic_ratio": sum(1 for row in rows if float(row.get("plastic", 0.0)) > 0.0) / max(len(rows), 1),
                "max_pore_pressure": float(np.max(p)) if hydro is not None and p.size else 0.0,
                "converged": converged,
                "assembly_elapsed_seconds": sum(float(row.get("assembly_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "tangent_assembly_elapsed_seconds": sum(float(row.get("tangent_assembly_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "internal_force_assembly_elapsed_seconds": sum(float(row.get("internal_force_assembly_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "monolithic_assembly_elapsed_seconds": sum(float(row.get("monolithic_assembly_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "reduced_matrix_elapsed_seconds": sum(float(row.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "augmented_bmat_elapsed_seconds": sum(float(row.get("augmented_bmat_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "lagrange_constraint_matrix_elapsed_seconds": sum(float(row.get("lagrange_constraint_matrix_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "lagrange_bmat_elapsed_seconds": sum(float(row.get("lagrange_bmat_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "lagrange_linear_solve_elapsed_seconds": sum(float(row.get("lagrange_linear_solve_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "linear_solve_elapsed_seconds": sum(float(row.get("linear_solve_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "line_search_elapsed_seconds": 0.0,
                "postprocess_elapsed_seconds": postprocess_elapsed,
                "elapsed_seconds": max(_perf_counter() - step_start, 0.0),
            }
        )
        if current_arc_length < arc_length_value:
            current_arc_length = min(arc_length_value, max(min_arc_length, current_arc_length * growth))

    fint = assemble_axisymmetric_internal_force(mesh, materials, u, initial_stresses=initial_stresses, plastic_state=state_current, interfaces=interfaces, structural_elements=structural_elements)
    effective_pore_load = Bp @ p if hydro is not None else np.zeros_like(reference_load)
    reactions = fint - lam * reference_load - effective_pore_load
    if mpc_plan.add_penalty_to_stage_matrix:
        reactions = reactions + Kmpc @ u - Fmpc
    state_for_output = dict(state_current)
    final_postprocess_state_info: dict[str, Any] = {}
    element_results, state_current = compute_axisymmetric_element_results_and_state(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        plastic_state=state_current,
        postprocess_info=final_postprocess_state_info,
    )
    active_elements = [element.id for element in mesh.elements if element.active]
    result = StageResult2D(stage_name, u, reactions, element_results, constrained, active_elements, {"method": "axisymmetric_arc_length", "geometry": "axisymmetric", "tangent": tangent_method, "iterations": sum(int(p["iterations"]) for p in path), "residual_norm": path[-1]["residual_norm"], "converged": True, "postprocess_state_commit": final_postprocess_state_info})
    result.name = stage_name
    if hydro is not None:
        result.pore_pressure = p
        result.time = dt * steps
    result.plastic_state = state_current
    result.interface_results = compute_interface_results(mesh, interfaces, u, axisymmetric=True)
    result.structural_results = compute_structural_results(mesh, materials, structural_elements, u, axisymmetric=True, loads=loads)
    _attach_integration_point_results(
        result,
        mesh,
        materials,
        u,
        axisymmetric=True,
        plastic_state=state_for_output,
        initial_stresses=initial_stresses,
    )
    result.solver_info["riks"] = {
        "lambda": float(lam),
        "lambda_target": lambda_max,
        "steps": steps,
        "arc_length": arc_length_value,
        "psi": psi,
        "path": path,
        "iteration_history": iteration_history,
        "branch_tracking": "axisymmetric-direction-continuity-cutback",
        "direction_flips": direction_flips,
        "predictor_flips": predictor_flips,
        "negative_dlambda": negative_dlambda,
        "cutbacks": total_cutbacks,
        "cutback_log": cutback_log,
        "cache": {
            "step_cache": {"enabled": False} if axisymmetric_step_cache is None else axisymmetric_step_cache.solver_info(),
            "combined_tangent_internal_assembly": combined_tangent_internal_assembly,
            "sparse_pattern_cached": sparse_pattern is not None,
            "plastic_state_array_cache": plastic_state_array_cache_info(plastic_state_cache),
            "initial_stress_array_cache": initial_stress_array_cache_info(initial_stress_cache),
            "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
            "augmented_matrix_cache": _arc_length_augmented_cache_summary(augmented_cache_events, augmented_cache),
            "lagrange_linear_cache": _lagrange_linear_cache_summary(lagrange_linear_cache_events, lagrange_linear_cache),
            "lagrange_correction_cache": _arc_length_lagrange_cache_summary(lagrange_correction_cache_events, lagrange_correction_cache),
            "axisymmetric_up_correction_cache": _axisymmetric_up_arc_length_cache_summary(up_correction_cache_events, up_correction_cache),
            "axisymmetric_up_lagrange_correction_cache": _axisymmetric_up_arc_length_lagrange_cache_summary(
                up_lagrange_correction_cache_events,
                up_lagrange_correction_cache,
            ),
            "axisymmetric_up_hydraulic_cache": {**riks_up_hydraulic_cache_info, "boundary_cache_reuses": int(riks_up_boundary_cache_reuses)},
            "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
        },
    }
    if hydro is not None:
        result.solver_info["method"] = "axisymmetric_arc_length_up"
        result.solver_info["consolidation"] = {
            "dt": dt,
            "steps": steps,
            "storage": storage,
            "permeability": permeability,
            "biot_alpha": biot_alpha,
            "fixed_pressure_nodes": len(fixed_p),
            "unknowns": int(ndof + len(mesh.node_ids) + 1),
            "boundary": boundary_info,
            "interface_transfer": interface_hydro_info,
            "seepage_toggle_count": seepage_toggles,
            "seepage_active_edges": boundary_info.get("seepage_active_edges", 0),
            "mass_balance": pressure_residual_norm,
            "mass_balance_residual_sum": mass_balance_terms.get("residual_sum", 0.0),
            "mass_balance_terms": mass_balance_terms,
            "coupled_with_riks": True,
            "hydraulic_cache": {**riks_up_hydraulic_cache_info, "boundary_cache_reuses": int(riks_up_boundary_cache_reuses)},
            "correction_cache": (
                _axisymmetric_up_arc_length_lagrange_cache_summary(
                    up_lagrange_correction_cache_events,
                    up_lagrange_correction_cache,
                )
                if mpc_plan.lagrange_requested
                else _axisymmetric_up_arc_length_cache_summary(up_correction_cache_events, up_correction_cache)
            ),
        }
    if mpc_info["count"]:
        applied_method = "lagrange" if mpc_plan.lagrange_requested else "penalty"
        result.solver_info["mpc"] = {**mpc_info, "max_violation": mpc_violation(mesh, u, mpc_info), "applied_method": applied_method}
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
    return result


def solve_plane_strain_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str = "Stage-1",
    output_dir: str | Path | None = None,
    solver: Mapping[str, Any] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: Any | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float = 1.0,
    pore_pressure: np.ndarray | None = None,
    time: float = 0.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    consolidation: Mapping[str, Any] | None = None,
    previous_pressure: np.ndarray | None = None,
    initial_displacement: np.ndarray | None = None,
    postprocess_results: bool = True,
    step_cache: StepCache2D | None = None,
) -> StageResult2D:
    _validate_mesh(mesh)
    _validate_material_references(mesh, materials)
    solver_cfg = solver if isinstance(solver, Mapping) else {}
    large_cfg = _large_deformation_settings(solver_cfg)
    if large_cfg["enabled"] and consolidation is None:
        return solve_large_deformation_stage(
            mesh=mesh,
            materials=materials,
            boundary_conditions=boundary_conditions,
            loads=loads,
            mpc_constraints=mpc_constraints,
            stage_name=stage_name,
            output_dir=output_dir,
            solver=solver_cfg,
            stage_config={"large_deformation": large_cfg},
            initial_stresses=initial_stresses,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            pore_pressure=pore_pressure,
            time=time,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            step_cache=step_cache if isinstance(step_cache, LargeDeformationStepCache) else None,
            postprocess_results=postprocess_results,
        )
    if consolidation is None and _increment_settings(solver_cfg)["enabled"]:
        return solve_incremental_stage(
            mesh=mesh,
            materials=materials,
            boundary_conditions=boundary_conditions,
            loads=loads,
            mpc_constraints=mpc_constraints,
            stage_name=stage_name,
            output_dir=output_dir,
            solver=solver_cfg,
            initial_stresses=initial_stresses,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            pore_pressure=pore_pressure,
            time=time,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            initial_displacement=initial_displacement,
            step_cache=step_cache,
            postprocess_results=postprocess_results,
        )
    nonlinear = any(material.is_plastic for material in materials.values()) or _has_nonlinear_interfaces(interfaces) or structural_has_nonlinear(structural_elements)
    auto_step_cache = False
    static_step_cache_info: dict[str, Any] | None = None
    if step_cache is None and consolidation is None and nonlinear:
        cache_settings = _static_step_cache_settings(solver_cfg)
        if bool(cache_settings.get("enabled", True)):
            step_cache = build_small_deformation_step_cache(
                mesh,
                materials,
                boundary_conditions,
                interfaces=interfaces,
                structural_elements=structural_elements,
                precompute_stiffness_pattern=bool(cache_settings.get("precompute_stiffness_pattern", True)),
            )
            step_cache = _attach_load_mpc_step_cache(
                step_cache,
                mesh,
                materials,
                loads,
                mpc_constraints,
                interfaces=interfaces,
                structural_elements=structural_elements,
                reason_scope="static_stage",
            )
            auto_step_cache = True
        else:
            static_step_cache_info = _disabled_static_step_cache_info("disabled_by_solver_setting", geometry_mode="reference")
    assembly_start = _perf_counter()
    if step_cache is not None and step_cache.stiffness_cache is not None:
        K = assemble_global_stiffness_cached(step_cache.stiffness_cache, mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    else:
        K = assemble_global_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    F, load_vector_reused = _load_vector_for_stage(step_cache, mesh, materials, loads, structural_elements)
    pore_pressure_load_reused = False
    if pore_pressure is not None:
        pore_load, pore_pressure_load_reused = _pore_pressure_load_for_stage(step_cache, mesh, materials, pore_pressure, ndof=F.size)
        F += pore_load
    Kmpc, Fmpc, mpc_info, mpc_penalty_reused = _mpc_penalty_for_stage(step_cache, mesh, K, mpc_constraints)
    mpc_plan = mpc_stage_plan(
        mpc_constraints,
        solver_cfg,
        mpc_info,
        nonlinear=nonlinear,
        allow_lagrange_linear=consolidation is None,
        add_plain_penalty_to_stage_matrix=True,
    )
    if mpc_plan.add_penalty_to_stage_matrix:
        K = (K + Kmpc).tocsr()
        F = F + Fmpc
    if step_cache is not None:
        constrained = dict(step_cache.constrained)
    else:
        constrained = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
        _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)

    ndof = K.shape[0]
    assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
    solver_info: dict[str, Any]
    if consolidation is not None:
        result = solve_coupled_up_stage(
            mesh=mesh,
            materials=materials,
            boundary_conditions=boundary_conditions,
            loads=loads,
            mpc_constraints=mpc_constraints,
            stage_name=stage_name,
            output_dir=output_dir,
            solver=solver_cfg,
            initial_stresses=initial_stresses,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            consolidation=consolidation,
            previous_pressure=previous_pressure,
        )
        perf = result.solver_info.setdefault("performance", {})
        if not isinstance(perf, dict):
            perf = {}
            result.solver_info["performance"] = perf
        perf["assembly_elapsed_seconds"] = float(perf.get("assembly_elapsed_seconds", 0.0) or 0.0) + assembly_elapsed
        return result
    solve_start = _perf_counter()
    if nonlinear:
        nonlinear_initial_stress_cache = initial_stress_cache
        if nonlinear_initial_stress_cache is None and initial_stresses and step_cache is not None:
            nonlinear_initial_stress_cache = build_initial_stress_array_cache(mesh, initial_stresses, active_element_ids=step_cache.active_elements)
        u, reactions, solver_info = solve_nonlinear_system(
            mesh,
            materials,
            K,
            F,
            constrained,
            stage_name=stage_name,
            solver=solver_cfg,
            initial_stresses=initial_stresses,
            initial_stress_cache=nonlinear_initial_stress_cache,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            initial_displacement=initial_displacement,
            mpc_stiffness=Kmpc,
            mpc_load=Fmpc,
            mpc_info=mpc_info,
            mpc_lagrange=mpc_plan.lagrange_requested,
            free_dofs=step_cache.free_dofs if step_cache is not None else None,
            fixed_dofs=step_cache.fixed_dofs if step_cache is not None else None,
            sparse_pattern=step_cache.stiffness_cache.pattern if step_cache is not None and step_cache.stiffness_cache is not None else None,
            reduced_matrix_cache=step_cache.reduced_matrix_cache if step_cache is not None else None,
            quad4_mc_geometry_cache=(
                step_cache.quad4_mc_geometry_cache
                if isinstance(step_cache, SmallDeformationStepCache)
                else None
            ),
        )
    elif mpc_plan.use_elimination_linear:
        u, solver_info = solve_linear_system_with_mpc_elimination(K, F, constrained, mpc_info, stage_name=stage_name, solver=solver_cfg)
        reactions = K @ u - F
    elif mpc_plan.use_lagrange_linear:
        u, solver_info = solve_linear_system_with_mpc_lagrange(K, F, constrained, mpc_info, stage_name=stage_name, solver=solver_cfg)
        reactions = K @ u - F
    elif constrained:
        if step_cache is not None:
            free, fixed = step_cache.free_dofs, step_cache.fixed_dofs
        else:
            free, fixed = _free_index_arrays(ndof, constrained, stage_name=stage_name)
        u = np.zeros(ndof, dtype=float)
        for dof, value in constrained.items():
            u[dof] = value
        if free.size:
            reduction_cache = step_cache.reduced_matrix_cache if step_cache is not None else None
            u[free], solver_info, _reduction_cache = solve_reduced_linear_system(
                K,
                F,
                free,
                fixed,
                fixed_values=u[fixed],
                reduction_cache=reduction_cache,
                stage_name=stage_name,
                solver=solver_cfg,
                validate_cache=mpc_plan.add_penalty_to_stage_matrix or step_cache is None or reduction_cache is None,
            )
            if not np.all(np.isfinite(u[free])):
                raise FEM2DError(f"{stage_name}: linear solve produced non-finite displacements")
        else:
            solver_info = {"method": "none", "iterations": 0, "residual_norm": 0.0}
    else:
        u, solver_info = solve_linear_system(K, F, stage_name=stage_name, solver=solver_cfg)
        if not np.all(np.isfinite(u)):
            raise FEM2DError(f"{stage_name}: linear solve produced non-finite displacements")
    linear_solve_elapsed = max(_perf_counter() - solve_start, 0.0)
    if not nonlinear and not (mpc_plan.use_elimination_linear or mpc_plan.use_lagrange_linear):
        reactions = K @ u - F

    post_start = _perf_counter()
    state_for_results = _update_liquefaction_state_from_pore_pressure(mesh, materials, pore_pressure, plastic_state)
    postprocess_state_info: dict[str, Any] = {}
    same_pass_integration_rows: list[dict[str, Any]] | None = None
    active_elements = list(step_cache.active_elements) if step_cache is not None else [element.id for element in mesh.elements if element.active]
    updated_plastic_state_cache: PlasticStateArrayCache | None = None
    if postprocess_results:
        state_for_results = _materialized_plastic_state_for_postprocess(state_for_results, plastic_state_cache)
        element_results, updated_plastic_state = compute_element_results_and_state(
            mesh,
            materials,
            u,
            initial_stresses=initial_stresses,
            strength_factor=strength_factor,
            plastic_state=state_for_results,
            collect_results=True,
            postprocess_info=postprocess_state_info,
            collect_integration_point_rows=True,
            plastic_state_cache=plastic_state_cache,
        )
        same_pass_integration_rows = _pop_same_pass_integration_point_rows(postprocess_state_info)
        updated_plastic_state_cache = _pop_postprocess_state_array_cache(postprocess_state_info)
        if updated_plastic_state_cache is None:
            updated_plastic_state_cache = build_plastic_state_array_cache(mesh, materials, updated_plastic_state)
        plastic_ratio = updated_plastic_state_cache.plastic_ratio(active_elements) if updated_plastic_state_cache is not None else _plastic_ratio_from_state(updated_plastic_state, active_elements)
        plastic_ratio_source = "plastic_state_array_cache" if updated_plastic_state_cache is not None else "plastic_state"
    else:
        updated_plastic_state_cache = compute_plastic_state_array_cache(
            mesh,
            materials,
            u,
            initial_stresses=initial_stresses,
            strength_factor=strength_factor,
            plastic_state=state_for_results,
            plastic_state_cache=plastic_state_cache,
            postprocess_info=postprocess_state_info,
        )
        element_results = []
        updated_plastic_state = {}
        plastic_ratio = updated_plastic_state_cache.plastic_ratio(active_elements)
        plastic_ratio_source = "plastic_state_array_cache"
    _attach_matrix_profile(solver_info, K, F, constrained, label="plane_strain_global_stiffness")
    solver_info["postprocess_results"] = bool(postprocess_results)
    solver_info["postprocess_state_commit"] = postprocess_state_info
    solver_info["plastic_ratio"] = plastic_ratio
    solver_info["plastic_ratio_source"] = plastic_ratio_source
    solver_info["plastic_state_array_cache"] = plastic_state_array_cache_info(updated_plastic_state_cache)
    if step_cache is not None:
        solver_info["topology_cache"] = {**step_cache.solver_info(), "auto_generated": auto_step_cache}
    elif static_step_cache_info is not None:
        solver_info["topology_cache"] = static_step_cache_info
    solver_info["load_mpc_factor_cache"] = _stage_load_mpc_cache_info(
        step_cache,
        load_vector_reused=load_vector_reused,
        mpc_penalty_reused=mpc_penalty_reused,
    )
    if pore_pressure is not None:
        solver_info["pore_pressure_load_cache"] = _pore_pressure_load_cache_info(
            step_cache,
            reused=pore_pressure_load_reused,
        )
    cache_info = step_cache.solver_info() if step_cache is not None else {}
    solver_info.update(
        large_deformation_common_solver_info(
            mesh,
            materials,
            geometry_mode="reference",
            batched_elements=int(cache_info.get("batched_elastic_elements", 0) or 0),
            hydro_coupled=pore_pressure is not None or consolidation is not None,
        )
    )
    result = StageResult2D(stage_name, u, reactions, element_results, constrained, active_elements, solver_info)
    if mpc_plan.active:
        applied_method = mpc_plan.applied_method
        multiplier_info = {"multipliers": solver_info.get("multipliers", [])} if "multipliers" in solver_info else {}
        result.solver_info["mpc"] = {**mpc_info, **multiplier_info, "max_violation": mpc_violation(mesh, u, mpc_info), "applied_method": applied_method}
    result.pore_pressure = pore_pressure
    result.time = time
    result.plastic_state = updated_plastic_state
    result.plastic_state_array_cache = updated_plastic_state_cache
    if postprocess_results:
        result.interface_results = compute_interface_results(mesh, interfaces, u)
        result.structural_results = compute_structural_results(mesh, materials, structural_elements, u, loads=loads)
        _attach_structural_extra_dofs(result, mesh, structural_elements)
        if same_pass_integration_rows is not None:
            result.integration_point_results = same_pass_integration_rows
        else:
            _attach_integration_point_results(
                result,
                mesh,
                materials,
                u,
                strength_factor=strength_factor,
                plastic_state=state_for_results,
                initial_stresses=initial_stresses,
            )
    perf = result.solver_info.setdefault("performance", {})
    if not isinstance(perf, dict):
        perf = {}
        result.solver_info["performance"] = perf
    perf.update(
        {
            "assembly_elapsed_seconds": assembly_elapsed,
            "linear_solve_elapsed_seconds": linear_solve_elapsed,
            "postprocess_elapsed_seconds": max(_perf_counter() - post_start, 0.0),
            "coupled_assembly_elapsed_seconds": 0.0,
        }
    )
    if nonlinear:
        perf["nonlinear_solve_elapsed_seconds"] = linear_solve_elapsed
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
    return result


def solve_dynamic_time_history_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str = "Dynamic",
    output_dir: str | Path | None = None,
    solver: Mapping[str, Any] | None = None,
    stage_config: Mapping[str, Any] | None = None,
    global_config: Mapping[str, Any] | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    pore_pressure: np.ndarray | None = None,
    hydro: Mapping[str, Any] | None = None,
    time: float = 0.0,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
) -> StageResult2D:
    stage_start = _perf_counter()
    assembly_start = _perf_counter()
    _validate_mesh(mesh)
    _validate_material_references(mesh, materials)
    solver_cfg = solver if isinstance(solver, Mapping) else {}
    stage_cfg = stage_config if isinstance(stage_config, Mapping) else {}
    cfg = global_config if isinstance(global_config, Mapping) else {}
    dynamic_cfg = _dynamic_stage_settings(stage_cfg, solver_cfg)
    dynamic_solver_cfg = _dynamic_solver_config(solver_cfg, dynamic_cfg)

    nonlinear = any(material.is_plastic for material in materials.values()) or _has_nonlinear_interfaces(interfaces) or structural_has_nonlinear(structural_elements)
    lumped_mass = bool(dynamic_cfg.get("lumped_mass", dynamic_cfg.get("lumped", False)))
    dynamic_mass_cache: DynamicMassStepCache | None = None
    cache_build_elapsed = 0.0
    stiffness_assembly_elapsed = 0.0
    mass_assembly_elapsed = 0.0
    mass_regularization_elapsed = 0.0
    damping_assembly_elapsed = 0.0
    load_assembly_elapsed = 0.0
    mpc_assembly_elapsed = 0.0
    pore_pressure_load_elapsed = 0.0
    cache_enabled = bool(dynamic_cfg.get("mass_step_cache", dynamic_cfg.get("cache_mass_matrix", dynamic_cfg.get("dynamic_step_cache", True))))
    if cache_enabled:
        cache_start = _perf_counter()
        dynamic_mass_cache = build_dynamic_mass_step_cache(
            mesh,
            materials,
            loads,
            interfaces=interfaces,
            structural_elements=structural_elements,
            lumped_mass=lumped_mass,
            precompute_stiffness_pattern=bool(dynamic_cfg.get("precompute_stiffness_pattern", dynamic_cfg.get("precompute_sparse_pattern", True))),
            precompute_mass_pattern=bool(dynamic_cfg.get("precompute_mass_pattern", dynamic_cfg.get("precompute_mass_matrix", True))),
            precompute_load_vector=bool(dynamic_cfg.get("precompute_load_vector", True)),
        )
        cache_build_elapsed = max(_perf_counter() - cache_start, 0.0)
    stiffness_start = _perf_counter()
    if dynamic_mass_cache is not None and dynamic_mass_cache.stiffness_cache is not None:
        K = assemble_global_stiffness_cached(dynamic_mass_cache.stiffness_cache, mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    else:
        K = assemble_global_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    stiffness_assembly_elapsed = max(_perf_counter() - stiffness_start, 0.0)
    mass_start = _perf_counter()
    if dynamic_mass_cache is not None and dynamic_mass_cache.mass_cache is not None:
        M_raw = assemble_mass_matrix_cached(dynamic_mass_cache.mass_cache, mesh, materials, structural_elements=structural_elements, lumped=lumped_mass)
    else:
        M_raw = assemble_mass_matrix(mesh, materials, structural_elements=structural_elements, lumped=lumped_mass)
    mass_assembly_elapsed = max(_perf_counter() - mass_start, 0.0)
    mass_regularization_start = _perf_counter()
    M, mass_info = _regularized_dynamic_mass(M_raw, dynamic_cfg, stage_name)
    mass_regularization_elapsed = max(_perf_counter() - mass_regularization_start, 0.0)
    mass_info = dict(mass_info)
    mass_info["assembly_cache"] = {"enabled": False} if dynamic_mass_cache is None or dynamic_mass_cache.mass_cache is None else dynamic_mass_cache.mass_cache.info()
    mass_info["assembly_elapsed_seconds"] = mass_assembly_elapsed
    mass_info["regularization_elapsed_seconds"] = mass_regularization_elapsed
    damping_start = _perf_counter()
    C, damping_cache_event = _dynamic_rayleigh_damping_matrix(M, K, dynamic_cfg)
    damping_assembly_elapsed = max(_perf_counter() - damping_start, 0.0)
    hydro_cfg = hydro if isinstance(hydro, Mapping) else {}
    dynamic_up = _dynamic_up_enabled(stage_cfg, dynamic_cfg, hydro_cfg)

    load_start = _perf_counter()
    if dynamic_mass_cache is not None and dynamic_mass_cache.load_vector_cache is not None:
        F_base = assemble_load_vector_cached(dynamic_mass_cache.load_vector_cache, mesh, materials)
        load_vector_reused = True
    else:
        F_base = assemble_load_vector(mesh, materials, loads, structural_elements=structural_elements)
        load_vector_reused = False
    load_assembly_elapsed = max(_perf_counter() - load_start, 0.0)
    if pore_pressure is not None and not dynamic_up:
        pore_start = _perf_counter()
        F_base[: len(mesh.node_ids) * 2] += assemble_pore_pressure_load(mesh, materials, pore_pressure)
        pore_pressure_load_elapsed = max(_perf_counter() - pore_start, 0.0)

    mpc_start = _perf_counter()
    Kmpc, Fmpc, mpc_info = assemble_mpc_penalty(mesh, K, mpc_constraints)
    mpc_assembly_elapsed = max(_perf_counter() - mpc_start, 0.0)
    dynamic_monolithic = bool(nonlinear or dynamic_up or dynamic_cfg.get("nonlinear", False))
    mpc_plan = mpc_stage_plan(
        mpc_constraints,
        dynamic_solver_cfg,
        mpc_info,
        nonlinear=dynamic_monolithic,
        allow_elimination_linear=True,
        allow_lagrange_linear=True,
        add_plain_penalty_to_stage_matrix=True,
        add_penalty_when_exact_linear_blocked=True,
        add_penalty_when_lagrange_linear_blocked=True,
    )
    if mpc_plan.add_penalty_to_stage_matrix:
        K = (K + Kmpc).tocsr()
        F_base = F_base + Fmpc

    ndof = K.shape[0]
    times = _dynamic_time_vector(dynamic_cfg, _seismic_mapping(cfg, stage_cfg), start_time=time)
    constrained = _collect_dynamic_constraints(mesh, boundary_conditions, structural_elements, float(times[0]))
    _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)
    fixed_p: dict[int, float] = {}
    pressure_m: csr_matrix | None = None
    pressure_h: csr_matrix | None = None
    pressure_biot: csr_matrix | None = None
    pressure_interface: csr_matrix | None = None
    p: np.ndarray | None = None
    pressure_info: dict[str, Any] = {}
    if dynamic_up:
        storage = float(hydro_cfg.get("storage", hydro_cfg.get("specific_storage", dynamic_cfg.get("storage", 1.0))))
        permeability = float(hydro_cfg.get("permeability", hydro_cfg.get("k", dynamic_cfg.get("permeability", 1.0))))
        biot_alpha = float(hydro_cfg.get("biot_alpha", hydro_cfg.get("alpha", dynamic_cfg.get("biot_alpha", 1.0))))
        if storage <= 0.0 or permeability < 0.0:
            raise FEM2DError(f"{stage_name}: dynamic u-p storage must be positive and permeability non-negative")
        pressure_m, pressure_h = assemble_pressure_matrices(mesh, materials, storage=storage, permeability=permeability)
        pressure_interface, interface_hydro_info = assemble_interface_hydraulic_transfer(mesh, interfaces)
        pressure_biot = assemble_biot_coupling_matrix(mesh, materials, alpha=biot_alpha, structural_elements=structural_elements)
        fixed_p = _collect_pressure_constraints(mesh, hydro_cfg.get("pressure_bcs", hydro_cfg.get("pore_pressure_bcs", hydro_cfg.get("drainage", []))))
        p = _initial_pore_pressure(mesh, hydro_cfg, pore_pressure)
        for idx, value in fixed_p.items():
            p[idx] = value
        pressure_info = {
            "enabled": True,
            "storage": storage,
            "permeability": permeability,
            "biot_alpha": biot_alpha,
            "fixed_pressure_nodes": len(fixed_p),
            "interface_transfer": interface_hydro_info,
        }
    u = _dynamic_initial_vector(dynamic_cfg, ndof, "initial_displacement", "u0")
    v = _dynamic_initial_vector(dynamic_cfg, ndof, "initial_velocity", "v0")
    for dof, value in constrained.items():
        u[dof] = value
        v[dof] = 0.0

    load0, seismic0 = _dynamic_load_vector(mesh, materials, structural_elements, cfg, stage_cfg, F_base, dynamic_cfg, float(times[0]))
    a = _dynamic_initial_acceleration(M, C, K, load0, u, v, constrained, stage_name, dynamic_solver_cfg)
    for dof in constrained:
        a[dof] = 0.0

    beta = float(dynamic_cfg.get("newmark_beta", dynamic_cfg.get("beta", 0.25)))
    gamma = float(dynamic_cfg.get("newmark_gamma", dynamic_cfg.get("gamma", 0.5)))
    if beta <= 0.0 or gamma <= 0.0:
        raise FEM2DError(f"{stage_name}: Newmark beta/gamma must be positive")

    history: list[dict[str, Any]] = []
    history.append(_dynamic_history_row(0, float(times[0]), 0.0, u, v, a, M, C, K, load0, seismic0, pressure=p))
    linear_iterations = 0
    linear_method = "direct"
    residual_norm = float(history[-1]["residual_norm"])
    tangent_method = _tangent_method(dynamic_solver_cfg)
    dynamic_plastic_state_cache = (
        build_plastic_state_array_cache(mesh, materials, plastic_state)
        if dynamic_monolithic and (plastic_state or any(material.is_plastic for material in materials.values()))
        else None
    )
    dynamic_initial_stress_cache = (
        build_initial_stress_array_cache(mesh, initial_stresses)
        if dynamic_monolithic and initial_stresses
        else None
    )
    dynamic_reduced_matrix_cache: ReducedMatrixCache | None = None
    dynamic_monolithic_lhs_pattern_cache: CoupledUPMonolithicMatrixCache | None = None
    dynamic_effective_stiffness_combo_cache: dict[str, Any] | None = None
    dynamic_reduced_matrix_cache_events: list[Mapping[str, Any]] = []
    dynamic_symbolic_cache_events: list[Mapping[str, Any]] = []
    dynamic_monolithic_lhs_pattern_events: list[Mapping[str, Any]] = []
    dynamic_effective_stiffness_combo_events: list[Mapping[str, Any]] = []
    dynamic_combined_tangent_internal = False
    dynamic_cutbacks = int(dynamic_cfg.get("max_cutbacks", dynamic_cfg.get("cutbacks", 0)) or 0)
    cutback_factor = float(dynamic_cfg.get("cutback_factor", 0.5) or 0.5)
    min_dt = float(dynamic_cfg.get("min_dt", dynamic_cfg.get("min_time_step", 1.0e-8)) or 1.0e-8)
    if dynamic_cutbacks < 0:
        raise FEM2DError(f"{stage_name}: dynamic max_cutbacks must be non-negative")
    if not (0.0 < cutback_factor < 1.0):
        raise FEM2DError(f"{stage_name}: dynamic cutback_factor must satisfy 0 < factor < 1")
    cutback_log: list[dict[str, Any]] = []
    total_cutbacks = 0
    accepted_steps = 0
    current_time = float(times[0])
    target_index = 1
    trial_dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
    assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
    linear_solve_elapsed = 0.0
    effective_stiffness_assembly_elapsed = 0.0
    effective_stiffness_cache = _dynamic_effective_stiffness_cache_state(
        enabled=not dynamic_monolithic
        and not mpc_plan.use_elimination_linear
        and not mpc_plan.use_lagrange_linear
        and bool(dynamic_cfg.get("cache_effective_stiffness", dynamic_cfg.get("effective_stiffness_cache", True)))
    )
    while target_index < len(times):
        target_time = float(times[target_index])
        if trial_dt <= 0.0:
            trial_dt = target_time - current_time
        t_next = min(current_time + trial_dt, target_time)
        dt = t_next - current_time
        if dt <= 0.0:
            raise FEM2DError(f"{stage_name}: dynamic time vector must be strictly increasing")
        a0 = 1.0 / (beta * dt * dt)
        a1 = gamma / (beta * dt)
        a2 = 1.0 / (beta * dt)
        a3 = 1.0 / (2.0 * beta) - 1.0
        a4 = gamma / beta - 1.0
        a5 = dt * (gamma / (2.0 * beta) - 1.0)
        F_next, seismic_info = _dynamic_load_vector(mesh, materials, structural_elements, cfg, stage_cfg, F_base, dynamic_cfg, t_next)
        constrained_next = _collect_dynamic_constraints(mesh, boundary_conditions, structural_elements, t_next)
        _add_inactive_node_constraints(mesh, constrained_next, interfaces=interfaces, structural_elements=structural_elements)
        try:
            if dynamic_monolithic:
                u_next, v_next, a_next, p_next, reactions_trial, solve_info, dynamic_reduced_matrix_cache, dynamic_monolithic_lhs_pattern_cache, dynamic_effective_stiffness_combo_cache = _solve_dynamic_newmark_monolithic_step(
                    mesh,
                    materials,
                    u,
                    v,
                    a,
                    p,
                    F_next,
                    constrained_next,
                    fixed_p,
                    dt=dt,
                    beta=beta,
                    gamma=gamma,
                    mass=M,
                    damping=C,
                    stiffness=K,
                    mpc_stiffness=Kmpc if mpc_plan.add_penalty_to_stage_matrix else None,
                    mpc_info=mpc_info,
                    pressure_mass=pressure_m,
                    pressure_conductivity=pressure_h,
                    pressure_biot=pressure_biot,
                    pressure_interface=pressure_interface,
                    hydro=hydro_cfg,
                    stage_name=stage_name,
                    solver=dynamic_solver_cfg,
                    initial_stresses=initial_stresses,
                    interfaces=interfaces,
                    structural_elements=structural_elements,
                    strength_factor=1.0,
                    plastic_state=plastic_state,
                    plastic_state_cache=dynamic_plastic_state_cache,
                    initial_stress_cache=dynamic_initial_stress_cache,
                    reduced_matrix_cache=dynamic_reduced_matrix_cache,
                    monolithic_lhs_pattern_cache=dynamic_monolithic_lhs_pattern_cache,
                    effective_stiffness_linear_combination_cache=dynamic_effective_stiffness_combo_cache,
                    tangent_method=tangent_method,
                    dynamic_up=dynamic_up,
                )
                if bool(solve_info.get("combined_tangent_internal_assembly", False)):
                    dynamic_combined_tangent_internal = True
                reduced_events = solve_info.get("reduced_matrix_cache_events", [])
                if isinstance(reduced_events, list):
                    dynamic_reduced_matrix_cache_events.extend(event for event in reduced_events if isinstance(event, Mapping))
                symbolic_events = solve_info.get("symbolic_cache_events", [])
                if isinstance(symbolic_events, list):
                    dynamic_symbolic_cache_events.extend(event for event in symbolic_events if isinstance(event, Mapping))
                monolithic_events = solve_info.get("monolithic_lhs_pattern_cache_events", [])
                if isinstance(monolithic_events, list):
                    dynamic_monolithic_lhs_pattern_events.extend(event for event in monolithic_events if isinstance(event, Mapping))
                combo_events = solve_info.get("effective_stiffness_linear_combination_cache_events", [])
                if isinstance(combo_events, list):
                    dynamic_effective_stiffness_combo_events.extend(event for event in combo_events if isinstance(event, Mapping))
                linear_solve_elapsed += float(solve_info.get("linear_solve_elapsed_seconds", 0.0) or 0.0)
                effective_stiffness_assembly_elapsed += float(solve_info.get("effective_stiffness_assembly_elapsed_seconds", 0.0) or 0.0)
            else:
                effective_start = _perf_counter()
                rhs = F_next + M @ (a0 * u + a2 * v + a3 * a) + C @ (a1 * u + a4 * v + a5 * a)
                K_eff, cache_event = _dynamic_effective_stiffness_matrix(
                    K,
                    M,
                    C,
                    a0=a0,
                    a1=a1,
                    dt=dt,
                    constrained=constrained_next,
                    cache=effective_stiffness_cache,
                )
                effective_assembly = max(_perf_counter() - effective_start, 0.0)
                effective_stiffness_assembly_elapsed += effective_assembly
                solve_start = _perf_counter()
                u_next, solve_info = _solve_dynamic_effective_system(
                    K_eff,
                    rhs,
                    constrained_next,
                    mpc_info,
                    stage_name=stage_name,
                    solver=dynamic_solver_cfg,
                    mpc_plan=mpc_plan,
                    effective_cache=effective_stiffness_cache if bool(cache_event.get("enabled", False)) else None,
                )
                step_linear_elapsed = max(_perf_counter() - solve_start, 0.0)
                linear_solve_elapsed += step_linear_elapsed
                reduced_event = solve_info.get("reduced_matrix_cache", {})
                if isinstance(reduced_event, Mapping):
                    cache_event["reduced_matrix_reused"] = bool(reduced_event.get("reused", False))
                    cache_event["reduced_matrix_built"] = bool(reduced_event.get("built", False))
                cache_event["lu_factor_cache_state"] = str(solve_info.get("factor_cache", ""))
                symbolic_event = solve_info.get("symbolic_cache", {})
                if isinstance(symbolic_event, Mapping):
                    cache_event["symbolic_cache_state"] = str(symbolic_event.get("state", ""))
                solve_info["effective_stiffness_cache"] = cache_event
                solve_info["effective_stiffness_assembly_elapsed_seconds"] = effective_assembly
                solve_info["linear_solve_elapsed_seconds"] = step_linear_elapsed
                solve_info["reduced_matrix_elapsed_seconds"] = float(solve_info.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
                solve_info["elapsed_seconds"] = effective_assembly + step_linear_elapsed
                a_next = a0 * (u_next - u) - a2 * v - a3 * a
                v_next = v + dt * ((1.0 - gamma) * a + gamma * a_next)
                reactions_trial = K @ u_next + C @ v_next + M @ a_next - F_next
                p_next = p
            if not bool(solve_info.get("converged", True)):
                residual = float(solve_info.get("residual_norm", math.inf) or math.inf)
                raise FEM2DError(
                    f"{stage_name}: dynamic step to t={t_next:.6g} did not converge, residual={residual:.6e}"
                )
        except FEM2DError as exc:
            if total_cutbacks >= dynamic_cutbacks:
                raise FEM2DError(f"{stage_name}: dynamic step to t={t_next:.6g} failed after {total_cutbacks} cutbacks: {exc}") from exc
            next_dt = dt * cutback_factor
            if next_dt < min_dt:
                raise FEM2DError(f"{stage_name}: dynamic time step fell below min_dt after cutback: {exc}") from exc
            total_cutbacks += 1
            cutback_log.append({"time": t_next, "dt": dt, "next_dt": next_dt, "error": str(exc)})
            trial_dt = next_dt
            continue
        u, v, a, p = u_next, v_next, a_next, p_next
        constrained = constrained_next
        accepted_steps += 1
        current_time = t_next
        linear_iterations += int(solve_info.get("iterations", 0) or 0)
        linear_method = str(solve_info.get("method", linear_method))
        row = _dynamic_history_row(accepted_steps, t_next, dt, u, v, a, M, C, K, F_next, seismic_info, pressure=p)
        row["cutbacks"] = total_cutbacks
        row["nonlinear_iterations"] = int(solve_info.get("iterations", 0) or 0)
        row["converged"] = bool(solve_info.get("converged", True))
        row["elapsed_seconds"] = float(solve_info.get("elapsed_seconds", 0.0) or 0.0)
        row["assembly_elapsed_seconds"] = float(solve_info.get("assembly_elapsed_seconds", solve_info.get("tangent_internal_assembly_elapsed_seconds", 0.0)) or 0.0)
        row["tangent_assembly_elapsed_seconds"] = float(solve_info.get("tangent_assembly_elapsed_seconds", solve_info.get("tangent_internal_assembly_elapsed_seconds", 0.0)) or 0.0)
        row["internal_force_assembly_elapsed_seconds"] = float(solve_info.get("internal_force_assembly_elapsed_seconds", 0.0) or 0.0)
        row["monolithic_assembly_elapsed_seconds"] = float(solve_info.get("monolithic_assembly_elapsed_seconds", 0.0) or 0.0)
        row["effective_stiffness_assembly_elapsed_seconds"] = float(solve_info.get("effective_stiffness_assembly_elapsed_seconds", 0.0) or 0.0)
        row["mass_step_cache_enabled"] = dynamic_mass_cache is not None and bool(dynamic_mass_cache.solver_info().get("enabled", False))
        row["mass_matrix_cache_enabled"] = dynamic_mass_cache is not None and dynamic_mass_cache.mass_cache is not None
        row["load_vector_reused"] = bool(load_vector_reused)
        row["reduced_matrix_elapsed_seconds"] = float(solve_info.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
        row["linear_solve_elapsed_seconds"] = float(solve_info.get("linear_solve_elapsed_seconds", 0.0) or 0.0)
        row["line_search_elapsed_seconds"] = float(solve_info.get("line_search_elapsed_seconds", 0.0) or 0.0)
        row["postprocess_elapsed_seconds"] = float(solve_info.get("postprocess_elapsed_seconds", 0.0) or 0.0)
        if "effective_stiffness_cache" in solve_info:
            cache_event = solve_info.get("effective_stiffness_cache", {})
            if isinstance(cache_event, Mapping):
                row["effective_stiffness_cache_state"] = str(cache_event.get("state", ""))
                row["reduced_matrix_cache_reused"] = bool(cache_event.get("reduced_matrix_reused", False))
                row["lu_factor_cache_state"] = str(cache_event.get("lu_factor_cache_state", ""))
                row["symbolic_cache_state"] = str(cache_event.get("symbolic_cache_state", ""))
                row["effective_stiffness_linear_combination_state"] = str(cache_event.get("linear_combination_state", ""))
        if "convergence_history" in solve_info and isinstance(solve_info.get("convergence_history"), list):
            row["convergence_history"] = solve_info["convergence_history"]
        history.append(row)
        residual_norm = float(solve_info.get("residual_norm", row["residual_norm"]))
        if abs(current_time - target_time) <= max(1.0e-12, abs(target_time) * 1.0e-12):
            target_index += 1
            if target_index < len(times):
                trial_dt = float(times[target_index] - current_time)
        else:
            trial_dt = min(trial_dt, target_time - current_time)

    F_final, _seismic_final = _dynamic_load_vector(mesh, materials, structural_elements, cfg, stage_cfg, F_base, dynamic_cfg, float(times[-1]))
    reactions = reactions_trial if "reactions_trial" in locals() else K @ u + C @ v + M @ a - F_final
    if dynamic_up and pressure_biot is not None and p is not None:
        reactions = reactions - pressure_biot @ p
    result_pressure = p if dynamic_up else pore_pressure
    state_for_results = _update_liquefaction_state_from_pore_pressure(mesh, materials, result_pressure, plastic_state)
    element_results, updated_plastic_state = compute_element_results_and_state(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        plastic_state=state_for_results,
    )
    solver_info: dict[str, Any] = {
        "method": "newmark",
        "linear_method": linear_method,
        "iterations": linear_iterations,
        "residual_norm": residual_norm,
        "converged": all(bool(row.get("converged", True)) for row in history),
        "geometry_mode": "dynamic_newmark",
        "element_type": ",".join(sorted({str(element.type).upper() for element in mesh.elements if element.active})),
        "integration": ",".join(sorted({normalize_integration(element.integration) for element in mesh.elements if element.active})),
        "material_model": ",".join(sorted({str(materials[element.material].model) for element in mesh.elements if element.active and element.material in materials})),
        "batched_elements": 0 if dynamic_mass_cache is None else int(dynamic_mass_cache.solver_info().get("batched_mass_elements", 0) or 0),
        "fallback_count": 0,
        "fallback_reasons": [],
        "dynamic": {
            "scheme": "newmark",
            "beta": beta,
            "gamma": gamma,
            "time_start": float(times[0]),
            "time_end": float(times[-1]),
            "steps": accepted_steps,
            "target_steps": len(times) - 1,
            "cutbacks": total_cutbacks,
            "cutback_log": cutback_log,
            "rayleigh_alpha": float(dynamic_cfg.get("rayleigh_alpha", dynamic_cfg.get("alpha_m", dynamic_cfg.get("mass_damping", 0.0))) or 0.0),
            "rayleigh_beta": float(dynamic_cfg.get("rayleigh_beta", dynamic_cfg.get("beta_k", dynamic_cfg.get("stiffness_damping", 0.0))) or 0.0),
            "history": history,
            "mass": mass_info,
            "mass_step_cache": {"enabled": False, "reason": "disabled_by_dynamic_setting"} if dynamic_mass_cache is None else dynamic_mass_cache.solver_info(),
            "load_vector_reused": bool(load_vector_reused),
            "stiffness_assembly_elapsed_seconds": stiffness_assembly_elapsed,
            "mass_assembly_elapsed_seconds": mass_assembly_elapsed,
            "damping_assembly_elapsed_seconds": damping_assembly_elapsed,
            "damping_matrix_cache": damping_cache_event,
            "effective_stiffness_cache": _dynamic_effective_stiffness_cache_info(effective_stiffness_cache),
            "nonlinear": bool(nonlinear or dynamic_cfg.get("nonlinear", False)),
            "up_coupled": bool(dynamic_up),
            "profile": dynamic_cfg.get("profile_info", {}),
            "damping_spec": dynamic_cfg.get("damping_spec", dynamic_cfg.get("damping", {})) if isinstance(dynamic_cfg.get("damping_spec", dynamic_cfg.get("damping", {})), Mapping) else {},
        },
        "seismic": {
            "method": "dynamic_time_history",
            "kh": float(history[-1].get("kh", 0.0) or 0.0),
            "kv": float(history[-1].get("kv", 0.0) or 0.0),
            "source": "newmark",
        },
    }
    if dynamic_up:
        solver_info["dynamic"]["up"] = pressure_info
    if nonlinear:
        solver_info["dynamic"]["nonlinear_tangent"] = tangent_method
    if dynamic_monolithic:
        solver_info["dynamic"]["newton_cache"] = {
            "combined_tangent_internal_assembly": dynamic_combined_tangent_internal,
            "plastic_state_array_cache": plastic_state_array_cache_info(dynamic_plastic_state_cache),
            "initial_stress_array_cache": initial_stress_array_cache_info(dynamic_initial_stress_cache),
            "reduced_matrix_cache": _reduced_matrix_cache_summary(dynamic_reduced_matrix_cache_events, dynamic_reduced_matrix_cache),
            "monolithic_lhs_pattern_cache": _coupled_up_monolithic_cache_summary(dynamic_monolithic_lhs_pattern_events, dynamic_monolithic_lhs_pattern_cache),
            "effective_stiffness_linear_combination_cache": _csr_linear_combination_cache_summary(dynamic_effective_stiffness_combo_events, dynamic_effective_stiffness_combo_cache),
            "symbolic_ordering_cache": _symbolic_ordering_cache_summary(dynamic_symbolic_cache_events),
        }
    solver_info["performance"] = {
        "assembly_elapsed_seconds": assembly_elapsed,
        "cache_build_elapsed_seconds": cache_build_elapsed,
        "stiffness_assembly_elapsed_seconds": stiffness_assembly_elapsed,
        "mass_assembly_elapsed_seconds": mass_assembly_elapsed,
        "mass_regularization_elapsed_seconds": mass_regularization_elapsed,
        "damping_assembly_elapsed_seconds": damping_assembly_elapsed,
        "load_assembly_elapsed_seconds": load_assembly_elapsed,
        "mpc_assembly_elapsed_seconds": mpc_assembly_elapsed,
        "pore_pressure_load_assembly_elapsed_seconds": pore_pressure_load_elapsed,
        "linear_solve_elapsed_seconds": linear_solve_elapsed,
        "postprocess_elapsed_seconds": max(_perf_counter() - stage_start - assembly_elapsed - linear_solve_elapsed, 0.0),
        "coupled_assembly_elapsed_seconds": 0.0,
        "effective_stiffness_assembly_elapsed_seconds": effective_stiffness_assembly_elapsed,
        "elapsed_seconds": max(_perf_counter() - stage_start, 0.0),
        "node_count": len(mesh.node_ids),
        "element_count": len(mesh.elements),
        "active_element_count": sum(1 for element in mesh.elements if element.active),
        "dof_count": ndof,
    }
    if mpc_info["count"]:
        multiplier_info = {"multipliers": []}
        solver_info["mpc"] = {**mpc_info, **multiplier_info, "max_violation": mpc_violation(mesh, u, mpc_info), "applied_method": mpc_plan.applied_method}
    active_elements = [element.id for element in mesh.elements if element.active]
    result = StageResult2D(stage_name, u, np.asarray(reactions).ravel(), element_results, constrained, active_elements, solver_info)
    result.pore_pressure = result_pressure
    result.time = float(times[-1])
    result.plastic_state = updated_plastic_state
    result.interface_results = compute_interface_results(mesh, interfaces, u)
    result.structural_results = compute_structural_results(mesh, materials, structural_elements, u, loads=loads)
    _attach_structural_extra_dofs(result, mesh, structural_elements)
    _attach_integration_point_results(
        result,
        mesh,
        materials,
        u,
        plastic_state=state_for_results,
        initial_stresses=initial_stresses,
    )
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
    return result


def _build_srm_factor_step_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any,
    *,
    solver: Mapping[str, Any],
    srm_cfg: Mapping[str, Any],
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
) -> StepCache2D | None:
    if not bool(srm_cfg.get("share_factor_cache", srm_cfg.get("cache_factors", True))):
        return None
    large_cfg = _large_deformation_settings(solver)
    precompute_pattern = bool(srm_cfg.get("precompute_stiffness_pattern", srm_cfg.get("precompute_sparse_pattern", True)))
    if large_cfg["enabled"]:
        precompute_pattern = bool(large_cfg.get("precompute_stiffness_pattern", precompute_pattern))
        cache = build_large_deformation_step_cache(
            mesh,
            materials,
            boundary_conditions,
            interfaces=interfaces,
            structural_elements=structural_elements,
            precompute_stiffness_pattern=precompute_pattern,
            reuse_scope="srm_factor_trials",
            shared_across_srm_factors=True,
        )
        return _attach_load_mpc_step_cache(
            cache,
            mesh,
            materials,
            loads,
            mpc_constraints,
            interfaces=interfaces,
            structural_elements=structural_elements,
            reason_scope="factor",
        )
    cache = build_small_deformation_step_cache(
        mesh,
        materials,
        boundary_conditions,
        interfaces=interfaces,
        structural_elements=structural_elements,
        precompute_stiffness_pattern=precompute_pattern,
    )
    return _attach_load_mpc_step_cache(
        cache,
        mesh,
        materials,
        loads,
        mpc_constraints,
        interfaces=interfaces,
        structural_elements=structural_elements,
        reason_scope="factor",
    )


def _srm_trial_workspace(
    mesh: Mesh2D,
    interfaces: list[Interface2D] | None,
    *,
    independent_geometry: bool,
) -> tuple[Mesh2D, list[Interface2D] | None]:
    """Return per-trial mutable state while retaining shared topology caches."""
    if not independent_geometry:
        return mesh, interfaces
    trial_mesh = replace(mesh, coords=np.asarray(mesh.coords, dtype=float).copy())
    trial_interfaces = None
    if interfaces is not None:
        trial_interfaces = [replace(interface, history=copy.deepcopy(interface.history)) for interface in interfaces]
    return trial_mesh, trial_interfaces


def _srm_process_safe_config(value: Any) -> Any:
    if callable(value):
        return None
    if isinstance(value, Mapping):
        return {
            str(key): sanitized
            for key, item in value.items()
            if not callable(item)
            for sanitized in [_srm_process_safe_config(item)]
        }
    if isinstance(value, list):
        return [_srm_process_safe_config(item) for item in value if not callable(item)]
    if isinstance(value, tuple):
        return tuple(_srm_process_safe_config(item) for item in value if not callable(item))
    return value


def _srm_plane_process_trial_spec(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    srm_cfg: Mapping[str, Any],
    initial_stresses: Mapping[str, np.ndarray] | None,
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
    lightweight_trials: bool,
    large_deformation_trials: bool,
) -> dict[str, Any]:
    return {
        "kind": "plane_strain",
        "mesh": mesh,
        "materials": dict(materials),
        "boundary_conditions": boundary_conditions,
        "loads": loads,
        "mpc_constraints": mpc_constraints,
        "stage_name": str(stage_name),
        "solver": _srm_process_safe_config(dict(solver or {})),
        "srm_cfg": _srm_process_safe_config(dict(srm_cfg)),
        "initial_stresses": initial_stresses,
        "interfaces": interfaces,
        "structural_elements": structural_elements,
        "plastic_state": plastic_state,
        "lightweight_trials": bool(lightweight_trials),
        "large_deformation_trials": bool(large_deformation_trials),
    }


def _srm_factor_cache_info(step_cache: StepCache2D | None) -> dict[str, Any]:
    if step_cache is None:
        return {
            "enabled": False,
            "shared_across_trials": False,
            "material_strength_parameter_cache": _material_strength_parameter_cache_info(),
            "factor_invariant_load_mpc_cache": {"enabled": False},
            "reduced_matrix_cache": {"enabled": False},
        }
    info = step_cache.solver_info()
    return {
        "enabled": True,
        "shared_across_trials": True,
        "shared_across_srm_factors": bool(info.get("shared_across_srm_factors", True)),
        "cache_kind": str(info.get("cache_kind", "large_deformation_step_cache")),
        "topology_cache_id": str(info.get("topology_cache_id", "")),
        "reuse_scope": str(info.get("reuse_scope", "srm_factor_trials")),
        "constrained_dofs": int(info.get("constrained_dofs", 0) or 0),
        "free_dofs": int(info.get("free_dofs", 0) or 0),
        "active_elements": int(info.get("active_elements", 0) or 0),
        "stiffness_pattern_cached": bool(info.get("stiffness_pattern_cached", False)),
        "stiffness_blocks": int(info.get("stiffness_blocks", 0) or 0),
        "reduced_matrix_cached": bool(info.get("reduced_matrix_cached", False)),
        "reduced_matrix_cache": dict(info.get("reduced_matrix_cache", {})) if isinstance(info.get("reduced_matrix_cache", {}), Mapping) else {"enabled": False},
        "connectivity_shape": list(info.get("connectivity_shape", [])) if isinstance(info.get("connectivity_shape", []), list) else [],
        "element_dof_shape": list(info.get("element_dof_shape", [])) if isinstance(info.get("element_dof_shape", []), list) else [],
        "element_type_counts": dict(info.get("element_type_counts", {})) if isinstance(info.get("element_type_counts", {}), Mapping) else {},
        "integration_counts": dict(info.get("integration_counts", {})) if isinstance(info.get("integration_counts", {}), Mapping) else {},
        "batched_elements": _large_deformation_batched_elements(step_cache),
        "material_strength_parameter_cache": _material_strength_parameter_cache_info(),
        "factor_invariant_load_mpc_cache": dict(info.get("factor_invariant_load_mpc_cache", {})) if isinstance(info.get("factor_invariant_load_mpc_cache", {}), Mapping) else {},
    }


def _material_strength_parameter_cache_info() -> dict[str, Any]:
    try:
        from .fem2d_materials import strength_parameter_cache_info
        from .fem2d_plastic_batch import material_strength_parameter_array_cache_info
    except Exception:
        return {"enabled": False}
    return {
        "enabled": True,
        "scope": "material_strength_factor",
        "scalar_parameters": strength_parameter_cache_info(),
        "batch_arrays": material_strength_parameter_array_cache_info(),
    }


def solve_srm_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    output_dir: str | Path | None,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None = None,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
) -> StageResult2D:
    cfg = solver if isinstance(solver, Mapping) else {}
    raw_srm_cfg = cfg.get("srm", {}) if isinstance(cfg.get("srm", {}), Mapping) else {}
    srm_cfg = _srm_with_runtime_context(raw_srm_cfg, cfg)
    factors = _srm_factors(srm_cfg)
    failure_plastic_ratio = float(srm_cfg.get("failure_plastic_ratio", srm_cfg.get("plastic_ratio_limit", 0.95)))
    lightweight_trials = bool(srm_cfg.get("lightweight_postprocess", srm_cfg.get("lightweight_trials", True)))
    coarse_to_fine_info: dict[str, Any] = {"enabled": False}
    factor_step_cache = _build_srm_factor_step_cache(
        mesh,
        materials,
        boundary_conditions,
        loads,
        mpc_constraints,
        solver=cfg,
        srm_cfg=srm_cfg,
        interfaces=interfaces,
        structural_elements=structural_elements,
    )
    large_deformation_trials = bool(_large_deformation_settings(cfg).get("enabled", False))
    process_trial_spec = _srm_plane_process_trial_spec(
        mesh=mesh,
        materials=materials,
        boundary_conditions=boundary_conditions,
        loads=loads,
        mpc_constraints=mpc_constraints,
        stage_name=stage_name,
        solver=solver,
        srm_cfg=srm_cfg,
        initial_stresses=initial_stresses,
        interfaces=interfaces,
        structural_elements=structural_elements,
        plastic_state=plastic_state,
        lightweight_trials=lightweight_trials,
        large_deformation_trials=large_deformation_trials,
    )

    def solve_trial(
        factor: float,
        *,
        postprocess_results: bool | None = None,
        solver_override: Mapping[str, Any] | None = None,
    ) -> StageResult2D:
        collect_results = (not lightweight_trials) if postprocess_results is None else bool(postprocess_results)
        trial_solver = _srm_solver_with_retry_override(solver, solver_override)
        warm_start_displacement = _srm_initial_displacement_from_solver_override(solver_override)
        trial_mesh, trial_interfaces = _srm_trial_workspace(
            mesh,
            interfaces,
            independent_geometry=large_deformation_trials,
        )
        return solve_plane_strain_stage(
            mesh=trial_mesh,
            materials=materials,
            boundary_conditions=boundary_conditions,
            loads=loads,
            mpc_constraints=mpc_constraints,
            stage_name=f"{stage_name}-FS{factor:g}",
            output_dir=None,
            solver=trial_solver,
            initial_stresses=initial_stresses,
            interfaces=trial_interfaces,
            structural_elements=structural_elements,
            strength_factor=factor,
            plastic_state=plastic_state,
            initial_displacement=warm_start_displacement,
            step_cache=factor_step_cache,
            postprocess_results=collect_results,
        )

    search_factors = factors
    search_cfg: Mapping[str, Any] = srm_cfg
    warm_start_supported = not large_deformation_trials
    parallel_trials_supported = True
    parallel_disabled_reason = ""
    coarse_settings = _srm_coarse_to_fine_settings(srm_cfg)
    if bool(coarse_settings.get("enabled", False)):
        coarse_mesh, coarse_to_fine_info = _srm_build_coarse_mesh(mesh, srm_cfg)
        supported, unsupported_reason = _srm_coarse_targets_supported(boundary_conditions, loads, mpc_constraints)
        if coarse_mesh is None:
            coarse_to_fine_info.setdefault("used", False)
        elif not supported:
            coarse_to_fine_info.update({"used": False, "skip_reason": unsupported_reason})
        else:
            named_supported, named_reason = _srm_coarse_named_sets_supported(boundary_conditions, loads, coarse_mesh)
            if not named_supported:
                coarse_to_fine_info.update({"used": False, "skip_reason": named_reason})
            else:
                coarse_cache = _build_srm_factor_step_cache(
                    coarse_mesh,
                    materials,
                    boundary_conditions,
                    loads,
                    mpc_constraints,
                    solver=cfg,
                    srm_cfg=srm_cfg,
                    interfaces=None,
                    structural_elements=None,
                )

                def solve_coarse_trial(factor: float, *, solver_override: Mapping[str, Any] | None = None) -> StageResult2D:
                    trial_solver = _srm_solver_with_retry_override(solver, solver_override)
                    warm_start_displacement = _srm_initial_displacement_from_solver_override(solver_override)
                    trial_mesh, _trial_interfaces = _srm_trial_workspace(
                        coarse_mesh,
                        None,
                        independent_geometry=large_deformation_trials,
                    )
                    return solve_plane_strain_stage(
                        mesh=trial_mesh,
                        materials=materials,
                        boundary_conditions=boundary_conditions,
                        loads=loads,
                        mpc_constraints=mpc_constraints,
                        stage_name=f"{stage_name}-coarse-FS{factor:g}",
                        output_dir=None,
                        solver=trial_solver,
                        initial_stresses=None,
                        interfaces=None,
                        structural_elements=None,
                        strength_factor=factor,
                        plastic_state=None,
                        initial_displacement=warm_start_displacement,
                        step_cache=coarse_cache,
                        postprocess_results=False,
                    )

                coarse_search_cfg = _srm_coarse_search_config(srm_cfg, coarse_settings)
                coarse_process_trial_spec = _srm_plane_process_trial_spec(
                    mesh=coarse_mesh,
                    materials=materials,
                    boundary_conditions=boundary_conditions,
                    loads=loads,
                    mpc_constraints=mpc_constraints,
                    stage_name=f"{stage_name}-coarse",
                    solver=solver,
                    srm_cfg=coarse_search_cfg,
                    initial_stresses=None,
                    interfaces=None,
                    structural_elements=None,
                    plastic_state=None,
                    lightweight_trials=True,
                    large_deformation_trials=large_deformation_trials,
                )
                _coarse_result, coarse_fos, coarse_trials, coarse_search_info = _run_srm_trial_search(
                    factors,
                    coarse_search_cfg,
                    failure_plastic_ratio,
                    solve_coarse_trial,
                    progress_stage_name=f"{stage_name}-coarse",
                    mesh=coarse_mesh,
                    warm_start_supported=warm_start_supported,
                    parallel_trials_supported=parallel_trials_supported,
                    parallel_disabled_reason=parallel_disabled_reason,
                    process_trial_spec=coarse_process_trial_spec,
                )
                narrowed = _srm_final_factors_from_coarse(factors, coarse_search_info, srm_cfg, coarse_settings)
                coarse_to_fine_info.update(
                    {
                        "used": True,
                        "coarse_factor_of_safety": coarse_fos,
                        "coarse_trial_count": len(coarse_trials),
                        "coarse_search": {key: value for key, value in coarse_search_info.items() if key not in {"coarse_scan_factors"}},
                        "fine_factor_candidates": [] if narrowed is None else list(narrowed),
                        "fine_factor_count_before": len(factors),
                        "fine_factor_count_after": len(factors) if narrowed is None else len(narrowed),
                    }
                )
                if narrowed is not None:
                    search_factors = narrowed
                    final_cfg = dict(srm_cfg)
                    final_cfg["adaptive"] = True
                    final_cfg["search_mode"] = "adaptive_bracket"
                    search_cfg = final_cfg

    result, fos, trials, search_info = _run_srm_trial_search(
        search_factors,
        search_cfg,
        failure_plastic_ratio,
        solve_trial,
        progress_stage_name=stage_name,
        mesh=mesh,
        warm_start_supported=warm_start_supported,
        parallel_trials_supported=parallel_trials_supported,
        parallel_disabled_reason=parallel_disabled_reason,
        process_trial_spec=process_trial_spec,
    )
    if result is None:
        if not trials:
            raise FEM2DError(f"{stage_name}: SRM did not run any trial")
        result = _srm_failed_trial_result(mesh, structural_elements, stage_name)
    selected_factor = _srm_selected_factor(fos, trials)
    if lightweight_trials and selected_factor is not None:
        method = str(result.solver_info.get("method", "")).lower().strip()
        retained_postprocess_supported = not bool(
            _large_deformation_settings(cfg).get("enabled", False)
        ) and not method.startswith("axisymmetric")
        if retained_postprocess_supported:
            result = _srm_postprocess_retained_trial(
                result,
                mesh=mesh,
                materials=materials,
                interfaces=interfaces,
                structural_elements=structural_elements,
                loads=loads,
                initial_stresses=initial_stresses,
                initial_plastic_state=plastic_state,
                strength_factor=selected_factor,
            )
            retained_info = result.solver_info.get("srm_retained_trial_postprocess", {})
            requires_reanalysis = bool(
                isinstance(retained_info, Mapping)
                and retained_info.get("requires_full_reanalysis", True)
            )
            if requires_reanalysis:
                drift_diagnostics = dict(retained_info) if isinstance(retained_info, Mapping) else {}
                result = solve_trial(selected_factor, postprocess_results=True)
                result.solver_info["srm_retained_trial_postprocess"] = {
                    **drift_diagnostics,
                    "nonlinear_reanalysis_avoided": False,
                    "full_reanalysis_performed": True,
                    "fallback_reason": str(
                        drift_diagnostics.get("state_drift", {}).get(
                            "reason", "retained_state_requires_full_reanalysis"
                        )
                    ) if isinstance(drift_diagnostics.get("state_drift", {}), Mapping) else "retained_state_requires_full_reanalysis",
                }
                search_info["lightweight_postprocess"] = {
                    "enabled": True,
                    "trial_postprocess_results": False,
                    "final_factor_postprocessed": selected_factor,
                    "final_factor_reprocessed": selected_factor,
                    "nonlinear_reanalysis_avoided": False,
                    "fallback_reason": result.solver_info["srm_retained_trial_postprocess"]["fallback_reason"],
                    "plastic_ratio_source": str(
                        result.solver_info.get("plastic_ratio_source", "plastic_state")
                    ),
                }
            else:
                search_info["lightweight_postprocess"] = {
                    "enabled": True,
                    "trial_postprocess_results": False,
                    "final_factor_postprocessed": selected_factor,
                    "final_factor_reprocessed": None,
                    "nonlinear_reanalysis_avoided": True,
                    "state_drift_exceeds_tolerance": bool(
                        isinstance(retained_info, Mapping)
                        and retained_info.get("state_drift_exceeds_tolerance", False)
                    ),
                    "plastic_ratio_source": "plastic_state_array_cache",
                }
        else:
            result = solve_trial(selected_factor, postprocess_results=True)
            search_info["lightweight_postprocess"] = {
                "enabled": True,
                "trial_postprocess_results": False,
                "final_factor_reprocessed": selected_factor,
                "nonlinear_reanalysis_avoided": False,
                "fallback_reason": "retained_trial_postprocess_not_supported_for_geometry",
                "plastic_ratio_source": str(
                    result.solver_info.get("plastic_ratio_source", "plastic_state")
                ),
            }
    if coarse_to_fine_info.get("enabled", False):
        search_info["coarse_to_fine"] = coarse_to_fine_info
    search_info["trial_workspace"] = {
        "independent_geometry": bool(large_deformation_trials),
        "coordinate_buffer_per_trial": bool(large_deformation_trials),
        "interface_history_copy_per_trial": bool(large_deformation_trials and interfaces),
        "topology_cache_shared": factor_step_cache is not None,
        "thread_safe_parallel_trials": True,
    }
    search_info["trial_timing"] = _attach_srm_trial_timing(result, trials)
    result.name = stage_name
    result.solver_info["srm"] = {
        "factor_of_safety": fos,
        "trials": trials,
        "failure_plastic_ratio": failure_plastic_ratio,
        "factor_cache": _srm_factor_cache_info(factor_step_cache),
        **search_info,
    }
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
    return result


@dataclass
class _LargeDeformationIncrementResult:
    displacements: np.ndarray
    reactions: np.ndarray
    solver_info: dict[str, Any]
    plastic_state: dict[str, PlasticState2D]
    plastic_state_array_cache: PlasticStateArrayCache | None
    constrained: dict[int, float]
    active_elements: list[str]
    element_results: list[dict[str, Any]]
    interface_results: list[dict[str, Any]]
    structural_results: list[dict[str, Any]]
    integration_point_results: list[dict[str, Any]]


def _solve_large_deformation_increment(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float = 1.0,
    pore_pressure: np.ndarray | None = None,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    postprocess_results: bool = False,
    step_cache: LargeDeformationStepCache | None = None,
) -> _LargeDeformationIncrementResult:
    solver_cfg = solver if isinstance(solver, Mapping) else {}
    assembly_start = _perf_counter()
    if step_cache is not None and step_cache.stiffness_cache is not None:
        K = assemble_global_stiffness_cached(step_cache.stiffness_cache, mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    else:
        K = assemble_global_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    F, load_vector_reused = _load_vector_for_stage(step_cache, mesh, materials, loads, structural_elements)
    pore_pressure_load_reused = False
    pore_pressure_load_elapsed = 0.0
    if pore_pressure is not None:
        pore_start = _perf_counter()
        pore_load, pore_pressure_load_reused = _pore_pressure_load_for_stage(step_cache, mesh, materials, pore_pressure, ndof=F.size)
        F += pore_load
        pore_pressure_load_elapsed = max(_perf_counter() - pore_start, 0.0)
    Kmpc, Fmpc, mpc_info, mpc_penalty_reused = _mpc_penalty_for_stage(step_cache, mesh, K, mpc_constraints)
    nonlinear = any(material.is_plastic for material in materials.values()) or _has_nonlinear_interfaces(interfaces) or structural_has_nonlinear(structural_elements)
    mpc_plan = mpc_stage_plan(
        mpc_constraints,
        solver_cfg,
        mpc_info,
        nonlinear=nonlinear,
        allow_lagrange_linear=True,
        add_plain_penalty_to_stage_matrix=True,
    )
    if mpc_plan.add_penalty_to_stage_matrix:
        K = (K + Kmpc).tocsr()
        F = F + Fmpc
    if step_cache is not None:
        constrained = dict(step_cache.constrained)
        free_cached = step_cache.free_dofs
        fixed_cached = step_cache.fixed_dofs
        active_elements = list(step_cache.active_elements)
    else:
        constrained = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
        _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)
        free_cached = None
        fixed_cached = None
        active_elements = [element.id for element in mesh.elements if element.active]

    ndof = K.shape[0]
    assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
    solve_start = _perf_counter()
    if nonlinear:
        initial_stress_cache = (
            build_initial_stress_array_cache(mesh, initial_stresses, active_element_ids=step_cache.active_elements)
            if initial_stresses and step_cache is not None
            else None
        )
        u, reactions, solver_info = solve_nonlinear_system(
            mesh,
            materials,
            K,
            F,
            constrained,
            stage_name=stage_name,
            solver=solver_cfg,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            mpc_stiffness=Kmpc,
            mpc_load=Fmpc,
            mpc_info=mpc_info,
            mpc_lagrange=mpc_plan.lagrange_requested,
            free_dofs=free_cached,
            fixed_dofs=fixed_cached,
            sparse_pattern=step_cache.stiffness_cache.pattern if step_cache is not None and step_cache.stiffness_cache is not None else None,
            reduced_matrix_cache=step_cache.reduced_matrix_cache if step_cache is not None else None,
            quad4_mc_geometry_cache=(
                step_cache.quad4_mc_geometry_cache
                if isinstance(step_cache, SmallDeformationStepCache)
                else None
            ),
        )
    elif mpc_plan.use_elimination_linear:
        u, solver_info = solve_linear_system_with_mpc_elimination(K, F, constrained, mpc_info, stage_name=stage_name, solver=solver_cfg)
        reactions = K @ u - F
    elif mpc_plan.use_lagrange_linear:
        u, solver_info = solve_linear_system_with_mpc_lagrange(K, F, constrained, mpc_info, stage_name=stage_name, solver=solver_cfg)
        reactions = K @ u - F
    elif constrained:
        if free_cached is not None and fixed_cached is not None:
            free, fixed = free_cached, fixed_cached
        else:
            free, fixed = _free_index_arrays(ndof, constrained, stage_name=stage_name)
        u = np.zeros(ndof, dtype=float)
        for dof, value in constrained.items():
            u[dof] = value
        if free.size:
            reduction_cache = step_cache.reduced_matrix_cache if step_cache is not None else None
            u[free], solver_info, _reduction_cache = solve_reduced_linear_system(
                K,
                F,
                free,
                fixed,
                fixed_values=u[fixed],
                reduction_cache=reduction_cache,
                stage_name=stage_name,
                solver=solver_cfg,
                validate_cache=mpc_plan.add_penalty_to_stage_matrix or step_cache is None or reduction_cache is None,
            )
            if not np.all(np.isfinite(u[free])):
                raise FEM2DError(f"{stage_name}: linear solve produced non-finite displacements")
        else:
            solver_info = {"method": "none", "iterations": 0, "residual_norm": 0.0}
    else:
        u, solver_info = solve_linear_system(K, F, stage_name=stage_name, solver=solver_cfg)
        if not np.all(np.isfinite(u)):
            raise FEM2DError(f"{stage_name}: linear solve produced non-finite displacements")
    linear_solve_elapsed = max(_perf_counter() - solve_start, 0.0)
    if not nonlinear and not (mpc_plan.use_elimination_linear or mpc_plan.use_lagrange_linear):
        reactions = K @ u - F

    post_start = _perf_counter()
    state_for_results = _update_liquefaction_state_from_pore_pressure(mesh, materials, pore_pressure, plastic_state)
    postprocess_state_info: dict[str, Any] = {}
    same_pass_integration_rows: list[dict[str, Any]] | None = None
    updated_plastic_state_cache: PlasticStateArrayCache | None = None
    if postprocess_results:
        state_for_results = _materialized_plastic_state_for_postprocess(state_for_results, plastic_state_cache)
        element_results, updated_plastic_state = compute_element_results_and_state(
            mesh,
            materials,
            u,
            initial_stresses=initial_stresses,
            strength_factor=strength_factor,
            plastic_state=state_for_results,
            collect_results=True,
            postprocess_info=postprocess_state_info,
            collect_integration_point_rows=True,
            plastic_state_cache=plastic_state_cache,
        )
        same_pass_integration_rows = _pop_same_pass_integration_point_rows(postprocess_state_info)
        updated_plastic_state_cache = _pop_postprocess_state_array_cache(postprocess_state_info)
        if updated_plastic_state_cache is None:
            updated_plastic_state_cache = build_plastic_state_array_cache(mesh, materials, updated_plastic_state)
        plastic_ratio = updated_plastic_state_cache.plastic_ratio(active_elements) if updated_plastic_state_cache is not None else _plastic_ratio_from_state(updated_plastic_state, active_elements)
        plastic_ratio_source = "plastic_state_array_cache" if updated_plastic_state_cache is not None else "plastic_state"
    else:
        updated_plastic_state_cache = compute_plastic_state_array_cache(
            mesh,
            materials,
            u,
            initial_stresses=initial_stresses,
            strength_factor=strength_factor,
            plastic_state=state_for_results,
            plastic_state_cache=plastic_state_cache,
            postprocess_info=postprocess_state_info,
        )
        element_results = []
        updated_plastic_state = {}
        plastic_ratio = updated_plastic_state_cache.plastic_ratio(active_elements)
        plastic_ratio_source = "plastic_state_array_cache"
    solver_info["postprocess_results"] = bool(postprocess_results)
    solver_info["postprocess_state_commit"] = postprocess_state_info
    solver_info["plastic_ratio"] = plastic_ratio
    solver_info["plastic_ratio_source"] = plastic_ratio_source
    solver_info["plastic_state_array_cache"] = plastic_state_array_cache_info(updated_plastic_state_cache)
    if pore_pressure is not None:
        increment_solver = "large_deformation_up_internal_loop"
    elif nonlinear or strength_factor != 1.0:
        increment_solver = "large_deformation_plastic_internal_loop"
    else:
        increment_solver = "large_deformation_internal_loop"
    solver_info["increment_solver"] = increment_solver
    if step_cache is not None:
        solver_info["topology_cache"] = step_cache.solver_info()
    solver_info["load_mpc_factor_cache"] = _stage_load_mpc_cache_info(
        step_cache,
        load_vector_reused=load_vector_reused,
        mpc_penalty_reused=mpc_penalty_reused,
    )
    if pore_pressure is not None:
        solver_info["pore_pressure_load_cache"] = _pore_pressure_load_cache_info(
            step_cache,
            reused=pore_pressure_load_reused,
        )
    cache_info = step_cache.solver_info() if step_cache is not None else {}
    plastic_batched = sum(int(block.get("batched_elements", 0) or 0) for block in cache_info.get("plastic_blocks", []) if isinstance(block, Mapping))
    solver_info.update(
        large_deformation_common_solver_info(
            mesh,
            materials,
            geometry_mode="updated_lagrangian",
            batched_elements=int(cache_info.get("batched_elastic_elements", 0) or 0) + plastic_batched,
            hydro_coupled=pore_pressure is not None,
        )
    )
    if mpc_plan.active:
        multiplier_info = {"multipliers": solver_info.get("multipliers", [])} if "multipliers" in solver_info else {}
        solver_info["mpc"] = {**mpc_info, **multiplier_info, "max_violation": mpc_violation(mesh, u, mpc_info), "applied_method": mpc_plan.applied_method}

    interface_results: list[dict[str, Any]] = []
    structural_results: list[dict[str, Any]] = []
    integration_point_results: list[dict[str, Any]] = []
    if postprocess_results:
        interface_results = compute_interface_results(mesh, interfaces, u)
        structural_results = compute_structural_results(mesh, materials, structural_elements, u, loads=loads)
        if same_pass_integration_rows is not None:
            integration_point_results = same_pass_integration_rows
        else:
            temp_result = StageResult2D(stage_name, u, np.asarray(reactions).ravel(), element_results, constrained, active_elements, solver_info)
            _attach_integration_point_results(
                temp_result,
                mesh,
                materials,
                u,
                strength_factor=strength_factor,
                plastic_state=state_for_results,
                initial_stresses=initial_stresses,
            )
            integration_point_results = [dict(row) for row in temp_result.integration_point_results]
    perf = solver_info.setdefault("performance", {})
    if not isinstance(perf, dict):
        perf = {}
        solver_info["performance"] = perf
    perf.update(
        {
            "assembly_elapsed_seconds": assembly_elapsed,
            "pore_pressure_load_assembly_elapsed_seconds": pore_pressure_load_elapsed,
            "linear_solve_elapsed_seconds": linear_solve_elapsed,
            "postprocess_elapsed_seconds": max(_perf_counter() - post_start, 0.0),
            "coupled_assembly_elapsed_seconds": 0.0,
        }
    )
    if nonlinear or strength_factor != 1.0:
        perf["nonlinear_solve_elapsed_seconds"] = linear_solve_elapsed

    return _LargeDeformationIncrementResult(
        displacements=u,
        reactions=np.asarray(reactions).ravel(),
        solver_info=solver_info,
        plastic_state=updated_plastic_state,
        plastic_state_array_cache=updated_plastic_state_cache,
        constrained=constrained,
        active_elements=active_elements,
        element_results=element_results,
        interface_results=interface_results,
        structural_results=structural_results,
        integration_point_results=integration_point_results,
    )


def solve_large_deformation_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str = "Large-Deformation",
    output_dir: str | Path | None = None,
    solver: Mapping[str, Any] | None = None,
    stage_config: Mapping[str, Any] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float = 1.0,
    pore_pressure: np.ndarray | None = None,
    time: float = 0.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    step_cache: LargeDeformationStepCache | None = None,
    postprocess_results: bool = True,
) -> StageResult2D:
    settings = _large_deformation_settings(solver, stage_config, default_enabled=True)
    step_solver = _solver_without_increments(_solver_without_large_deformation(solver))
    supplied_step_cache = step_cache is not None
    steps = int(settings["steps"])
    backend = str(settings["backend"])
    total_u = np.zeros(len(mesh.node_ids) * 2, dtype=float)
    total_reactions = np.zeros_like(total_u)
    state_current: dict[str, PlasticState2D] = dict(plastic_state or {})
    state_current_cache: PlasticStateArrayCache | None = plastic_state_cache
    if state_current_cache is None and state_current:
        state_current_cache = build_plastic_state_array_cache(mesh, materials, state_current)
    last_result: _LargeDeformationIncrementResult | None = None
    history: list[dict[str, Any]] = []
    performance_totals = {
        "assembly_elapsed_seconds": 0.0,
        "pore_pressure_load_assembly_elapsed_seconds": 0.0,
        "linear_solve_elapsed_seconds": 0.0,
        "postprocess_elapsed_seconds": 0.0,
        "coupled_assembly_elapsed_seconds": 0.0,
    }
    hydro_summary = _large_deformation_up_summary(mesh, pore_pressure, stage_config)
    diagonal = max(mesh_diagonal_length(mesh), np.finfo(float).eps)
    update_geometry = bool(settings["update_geometry"])
    updated_coords_buffer = np.empty_like(mesh.coords, dtype=float) if update_geometry else None
    geometry_update_calls = 0
    temporary_mesh_views = 0
    skip_intermediate_postprocessing = bool(settings.get("skip_intermediate_postprocessing", True))
    adaptive_steps = bool(settings.get("adaptive_steps", True))
    fixed_step = 1.0 / float(steps)
    step_size = float(settings.get("initial_step", fixed_step)) if adaptive_steps else fixed_step
    min_step = float(settings.get("min_step", fixed_step / 16.0))
    max_step = float(settings.get("max_step", 1.0))
    cutback_factor = float(settings.get("cutback_factor", 0.5))
    growth_factor = float(settings.get("growth_factor", 1.5))
    max_cutbacks = int(settings.get("max_cutbacks", max(steps * 4, 8)))
    max_adaptive_steps = int(settings.get("max_adaptive_steps", max(steps * 8, steps + 8)))
    grow_below_ratio = float(settings.get("grow_below_ratio", 0.01))
    shrink_above_ratio = float(settings.get("shrink_above_ratio", 0.05))
    grow_below_iterations = int(settings.get("grow_below_iterations", 2))
    shrink_above_iterations = int(settings.get("shrink_above_iterations", 8))
    early_failure_policy = _srm_early_failure_policy_from_solver(solver)
    cutback_log: list[dict[str, Any]] = []
    cutbacks = 0
    accepted_steps = 0
    load_fraction = 0.0
    if step_cache is None and bool(settings.get("precompute_topology", True)):
        step_cache = build_large_deformation_step_cache(
            mesh,
            materials,
            boundary_conditions,
            interfaces=interfaces,
            structural_elements=structural_elements,
            precompute_stiffness_pattern=bool(settings.get("precompute_stiffness_pattern", True)),
        )
        step_cache = _attach_load_mpc_step_cache(
            step_cache,
            mesh,
            materials,
            loads,
            mpc_constraints,
            interfaces=interfaces,
            structural_elements=structural_elements,
            reason_scope="large_deformation_increment",
        )

    while load_fraction < 1.0 - 1.0e-12:
        if accepted_steps >= max_adaptive_steps:
            raise FEM2DError(f"{stage_name}: large-deformation adaptive step limit reached before full load")
        remaining = max(1.0 - load_fraction, 0.0)
        factor = min(max(step_size, min_step), max_step, remaining)
        _raise_if_solver_cancel_requested(
            solver,
            stage_name,
            "large_deformation_step_start",
            attempted_load_factor=float(load_fraction + factor),
            last_accepted_load_factor=float(load_fraction),
            accepted_increment_count=int(accepted_steps),
            cutback_count=int(cutbacks),
            final_step_size=float(factor),
        )
        final_attempt = remaining - factor <= 1.0e-12
        postprocess_step = bool(postprocess_results and not (skip_intermediate_postprocessing and not final_attempt))
        solver_step_cache = step_cache.with_constraint_scale(factor) if step_cache is not None else None
        if update_geometry:
            assert updated_coords_buffer is not None
            fill_updated_coords(mesh.coords, total_u, out=updated_coords_buffer, backend=backend)
            geometry_update_calls += 1
        step_pressure = None if pore_pressure is None else np.asarray(pore_pressure, dtype=float) * factor
        plastic_points_before = _plastic_point_count(state_current, state_current_cache)

        def solve_increment_on(active_mesh: Mesh2D) -> _LargeDeformationIncrementResult:
            return _solve_large_deformation_increment(
                mesh=active_mesh,
                materials=materials,
                boundary_conditions=_scale_boundary_conditions(boundary_conditions, factor),
                loads=_scale_loads(loads, factor),
                mpc_constraints=mpc_constraints,
                stage_name=f"{stage_name}-UL{accepted_steps + 1}",
                solver=step_solver,
                initial_stresses=initial_stresses,
                interfaces=interfaces,
                structural_elements=structural_elements,
                strength_factor=strength_factor,
                pore_pressure=step_pressure,
                plastic_state=state_current,
                plastic_state_cache=state_current_cache,
                postprocess_results=postprocess_step,
                step_cache=solver_step_cache,
            )

        try:
            if update_geometry:
                assert updated_coords_buffer is not None
                temporary_mesh_views += 1
                with temporary_mesh_coords(mesh, updated_coords_buffer):
                    trial = solve_increment_on(mesh)
            else:
                trial = solve_increment_on(mesh)
            if not bool(trial.solver_info.get("converged", True)):
                raise FEM2DError(f"{stage_name}: large-deformation increment did not converge")
        except FEM2DError as exc:
            next_step = factor * cutback_factor
            active_elements_for_diagnostics = list(step_cache.active_elements) if step_cache is not None else [str(element.id) for element in mesh.elements if element.active]
            topology_diagnostics = _srm_topology_diagnostics_cache(mesh)
            cancel_diagnostics = _large_deformation_failure_diagnostics(
                status="solver_cancelled",
                load_fraction=load_fraction,
                factor=factor,
                next_step=next_step,
                min_step=min_step,
                cutback_factor=cutback_factor,
                cutbacks=cutbacks,
                max_cutbacks=max_cutbacks,
                accepted_steps=accepted_steps,
                state_current=state_current,
                state_current_cache=state_current_cache,
                active_elements=active_elements_for_diagnostics,
                total_u=total_u,
                history=history,
                error=exc,
                topology_cache=topology_diagnostics,
            )
            _raise_if_solver_cancel_requested(solver, stage_name, "large_deformation_cutback", diagnostics=cancel_diagnostics)
            if not adaptive_steps or cutbacks >= max_cutbacks:
                diagnostics = _large_deformation_failure_diagnostics(
                    status="large_deformation_cutback_limit" if adaptive_steps else "large_deformation_step_failure",
                    load_fraction=load_fraction,
                    factor=factor,
                    next_step=next_step,
                    min_step=min_step,
                    cutback_factor=cutback_factor,
                    cutbacks=cutbacks,
                    max_cutbacks=max_cutbacks,
                    accepted_steps=accepted_steps,
                    state_current=state_current,
                    state_current_cache=state_current_cache,
                    active_elements=active_elements_for_diagnostics,
                    total_u=total_u,
                    history=history,
                    error=exc,
                    topology_cache=topology_diagnostics,
                )
                raise FEM2DError(f"{stage_name}: large-deformation step failed at load fraction {load_fraction + factor:.6g}: {exc}", diagnostics=diagnostics) from exc
            if next_step < min_step:
                diagnostics = _large_deformation_failure_diagnostics(
                    status="large_deformation_min_step",
                    load_fraction=load_fraction,
                    factor=factor,
                    next_step=next_step,
                    min_step=min_step,
                    cutback_factor=cutback_factor,
                    cutbacks=cutbacks,
                    max_cutbacks=max_cutbacks,
                    accepted_steps=accepted_steps,
                    state_current=state_current,
                    state_current_cache=state_current_cache,
                    active_elements=active_elements_for_diagnostics,
                    total_u=total_u,
                    history=history,
                    error=exc,
                    topology_cache=topology_diagnostics,
                )
                raise FEM2DError(f"{stage_name}: large-deformation adaptive step fell below min_step after cutback: {exc}", diagnostics=diagnostics) from exc
            early_failure = _srm_early_failure_cutback_decision(
                early_failure_policy,
                last_load=load_fraction,
                attempted_load=load_fraction + factor,
                cutbacks=cutbacks,
                max_cutbacks=max_cutbacks,
                state_current=state_current,
                state_current_cache=state_current_cache,
                active_elements=active_elements_for_diagnostics,
                error=exc,
                topology_cache=topology_diagnostics,
            )
            if early_failure is not None:
                diagnostics = _large_deformation_failure_diagnostics(
                    status="srm_early_confirmed_failure",
                    load_fraction=load_fraction,
                    factor=factor,
                    next_step=next_step,
                    min_step=min_step,
                    cutback_factor=cutback_factor,
                    cutbacks=int(early_failure.get("early_failure_effective_cutbacks", cutbacks + 1)),
                    max_cutbacks=max_cutbacks,
                    accepted_steps=accepted_steps,
                    state_current=state_current,
                    state_current_cache=state_current_cache,
                    active_elements=active_elements_for_diagnostics,
                    total_u=total_u,
                    history=history,
                    error=exc,
                    topology_cache=topology_diagnostics,
                )
                diagnostics.update(early_failure)
                diagnostics["diagnostic_summary"] = _srm_diagnostic_summary(diagnostics)
                raise FEM2DError(
                    f"{stage_name}: large-deformation SRM trial stopped early as confirmed failure at load fraction {load_fraction + factor:.6g}: {exc}",
                    diagnostics=diagnostics,
                ) from exc
            cutbacks += 1
            cutback_log.append(
                {
                    "load_start": float(load_fraction),
                    "load_increment": float(factor),
                    "next_load_increment": float(next_step),
                    "error": str(exc),
                }
            )
            step_size = next_step
            continue
        if trial.displacements.shape != total_u.shape:
            raise FEM2DError(f"{stage_name}: large-deformation increment returned incompatible displacement size")
        total_u = total_u + trial.displacements
        total_reactions = total_reactions + np.asarray(trial.reactions, dtype=float)
        state_current = dict(trial.plastic_state)
        state_current_cache = trial.plastic_state_array_cache
        last_result = trial
        plastic_points_after = _plastic_point_count(state_current, state_current_cache)
        plastic_point_increment = max(plastic_points_after - plastic_points_before, 0)
        trial_performance = trial.solver_info.get("performance", {}) if isinstance(trial.solver_info.get("performance", {}), Mapping) else {}
        trial_state_cache = trial.solver_info.get("plastic_state_array_cache", {}) if isinstance(trial.solver_info.get("plastic_state_array_cache", {}), Mapping) else {}
        trial_load_mpc_cache = trial.solver_info.get("load_mpc_factor_cache", {}) if isinstance(trial.solver_info.get("load_mpc_factor_cache", {}), Mapping) else {}
        trial_pore_load_cache = trial.solver_info.get("pore_pressure_load_cache", {}) if isinstance(trial.solver_info.get("pore_pressure_load_cache", {}), Mapping) else {}
        trial_reduced_cache = trial.solver_info.get("reduced_matrix_cache", {}) if isinstance(trial.solver_info.get("reduced_matrix_cache", {}), Mapping) else {}
        trial_symbolic_cache = trial.solver_info.get("symbolic_ordering_cache", trial.solver_info.get("symbolic_cache", {}))
        trial_symbolic_cache = trial_symbolic_cache if isinstance(trial_symbolic_cache, Mapping) else {}
        trial_performance_row: dict[str, float] = {}
        for key in performance_totals:
            value = float(trial_performance.get(key, 0.0) or 0.0)
            performance_totals[key] += value
            trial_performance_row[key] = value
        trial_iteration_rows = trial.solver_info.get("convergence_history", [])
        if not isinstance(trial_iteration_rows, list):
            trial_iteration_rows = []
        trial_iteration_rows = [row for row in trial_iteration_rows if isinstance(row, Mapping)]
        trial_iteration_timings = {
            "tangent_assembly_elapsed_seconds": sum(float(row.get("tangent_assembly_elapsed_seconds", row.get("tangent_internal_assembly_elapsed_seconds", 0.0)) or 0.0) for row in trial_iteration_rows),
            "internal_force_assembly_elapsed_seconds": sum(float(row.get("internal_force_assembly_elapsed_seconds", 0.0) or 0.0) for row in trial_iteration_rows),
            "monolithic_assembly_elapsed_seconds": sum(float(row.get("monolithic_assembly_elapsed_seconds", 0.0) or 0.0) for row in trial_iteration_rows),
            "reduced_matrix_elapsed_seconds": sum(float(row.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0) for row in trial_iteration_rows),
            "line_search_elapsed_seconds": sum(float(row.get("line_search_elapsed_seconds", 0.0) or 0.0) for row in trial_iteration_rows),
        }
        max_increment = max_displacement_norm(trial.displacements, backend=backend)
        max_total = max_displacement_norm(total_u, backend=backend)
        increment_ratio = max_increment / diagonal
        if update_geometry:
            assert updated_coords_buffer is not None
            fill_updated_coords(mesh.coords, total_u, out=updated_coords_buffer, backend=backend)
            geometry_update_calls += 1
            temporary_mesh_views += 1
            with temporary_mesh_coords(mesh, updated_coords_buffer):
                min_det_j = _minimum_element_det_j(mesh)
        else:
            min_det_j = _minimum_element_det_j(mesh)
        load_start = load_fraction
        load_fraction = min(load_fraction + factor, 1.0)
        accepted_steps += 1
        iterations = int(trial.solver_info.get("iterations", 0) or 0)
        next_step = fixed_step if not adaptive_steps else factor
        adaptive_action = "fixed"
        if adaptive_steps and load_fraction < 1.0 - 1.0e-12:
            adaptive_action = "keep"
            if min_det_j <= 0.0 or increment_ratio >= shrink_above_ratio or iterations >= shrink_above_iterations or plastic_point_increment >= max(4, len(state_current) // 4):
                next_step = max(min_step, factor * cutback_factor)
                adaptive_action = "shrink"
            elif increment_ratio <= grow_below_ratio and iterations <= grow_below_iterations:
                next_step = min(max_step, factor * growth_factor)
                adaptive_action = "grow"
            next_step = min(max(next_step, min_step), max_step, 1.0 - load_fraction)
        history_row = {
            "step": accepted_steps,
            "load_start": float(load_start),
            "load_end": float(load_fraction),
            "load_increment": factor,
            "next_load_increment": float(next_step if load_fraction < 1.0 - 1.0e-12 else 0.0),
            "adaptive_action": adaptive_action,
            "increment_solver": str(trial.solver_info.get("increment_solver", "large_deformation_internal_loop")),
            "max_increment_displacement": max_increment,
            "max_total_displacement": max_total,
            "increment_displacement_to_model_diagonal": increment_ratio,
            "geometry_change_ratio": max_total / diagonal,
            "min_detJ": float(min_det_j),
            "plastic_points": plastic_points_after,
            "plastic_point_increment": plastic_point_increment,
            "iterations": iterations,
            "line_search_reductions": int(trial.solver_info.get("line_search_reductions", 0) or 0),
            "residual_norm": float(trial.solver_info.get("residual_norm", 0.0) or 0.0),
            "internal_external_work_ratio": trial.solver_info.get("internal_external_work_ratio", ""),
            "converged": bool(trial.solver_info.get("converged", True)),
            "postprocessed": postprocess_step,
            "sparse_pattern_cached": bool(trial.solver_info.get("sparse_pattern_cached", False)),
            "constraint_dofs_cached": bool(trial.solver_info.get("constraint_dofs_cached", False)),
            "load_vector_reused": bool(trial_load_mpc_cache.get("load_vector_reused", False)),
            "mpc_penalty_reused": bool(trial_load_mpc_cache.get("mpc_penalty_reused", False)),
            "reduced_matrix_cache_enabled": bool(trial_reduced_cache.get("enabled", False)),
            "reduced_matrix_cache_hits": int(trial_reduced_cache.get("hits", 0) or 0),
            "reduced_matrix_cache_builds": int(trial_reduced_cache.get("builds", 0) or 0),
            "symbolic_ordering_cache_enabled": bool(trial_symbolic_cache.get("enabled", False)),
            "symbolic_ordering_cache_hits": int(trial_symbolic_cache.get("hits", 0) or 0),
            "symbolic_ordering_cache_misses": int(trial_symbolic_cache.get("misses", 0) or 0),
            "plastic_state_array_cache_enabled": bool(trial_state_cache.get("enabled", False)),
            "plastic_state_array_cache_present_points": int(trial_state_cache.get("present_points", 0) or 0),
            "elapsed_seconds": sum(float(trial_performance.get(key, 0.0) or 0.0) for key in ("assembly_elapsed_seconds", "linear_solve_elapsed_seconds", "postprocess_elapsed_seconds", "coupled_assembly_elapsed_seconds")),
            **trial_performance_row,
            **trial_iteration_timings,
        }
        if trial_iteration_rows:
            history_row["convergence_history"] = [dict(row) for row in trial_iteration_rows]
        trial_topology_cache = trial.solver_info.get("topology_cache", {}) if isinstance(trial.solver_info.get("topology_cache", {}), Mapping) else {}
        if trial_topology_cache:
            history_row.update(
                {
                    "topology_cache_id": str(trial_topology_cache.get("topology_cache_id", "")),
                    "topology_cache_reuse_scope": str(trial_topology_cache.get("reuse_scope", "")),
                    "topology_shared_across_srm_factors": bool(trial_topology_cache.get("shared_across_srm_factors", False)),
                }
            )
        if pore_pressure is not None:
            step_pressure_values = np.asarray(step_pressure, dtype=float).reshape(-1) if step_pressure is not None else np.zeros(len(mesh.node_ids), dtype=float)
            history_row.update(
                {
                    "pressure_dof_count": int(step_pressure_values.size),
                    "drainage_boundary_count": int(hydro_summary.get("drainage_boundary_count", 0) or 0),
                    "flow_balance": 0.0,
                    "pressure_residual_norm": 0.0,
                    "pressure_converged": True,
                    "pore_pressure_load_cache_enabled": bool(trial_pore_load_cache.get("enabled", False)),
                    "pore_pressure_load_reused": bool(trial_pore_load_cache.get("reused", False)),
                    "pore_pressure_load_batched_elements": int(trial_pore_load_cache.get("batched_elements", 0) or 0),
                    "max_pore_pressure": float(np.max(step_pressure_values)) if step_pressure_values.size else 0.0,
                    "min_pore_pressure": float(np.min(step_pressure_values)) if step_pressure_values.size else 0.0,
                    "history_storage_policy": hydro_summary.get("history_storage_policy", ""),
                    "rollback_policy": hydro_summary.get("rollback_policy", ""),
                }
            )
        history.append(history_row)
        _raise_if_solver_cancel_requested(
            solver,
            stage_name,
            "large_deformation_step_accepted",
            attempted_load_factor=float(load_fraction),
            last_accepted_load_factor=float(load_fraction),
            accepted_increment_count=int(accepted_steps),
            cutback_count=int(cutbacks),
            final_step_size=float(factor),
            max_displacement_norm=float(max_total),
            displacement_increment_norm=float(max_increment),
            min_det_j=float(min_det_j),
        )
        step_size = next_step

    if last_result is None:
        raise FEM2DError(f"{stage_name}: large-deformation stage did not run any increment")
    if step_cache is not None:
        constrained = dict(step_cache.constrained)
        active_elements = list(step_cache.active_elements)
    else:
        constrained = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
        _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)
        active_elements = [element.id for element in mesh.elements if element.active]
    element_results = [dict(row, stage=stage_name) if isinstance(row, dict) and "stage" in row else row for row in last_result.element_results]
    integration_rows = [dict(row, stage=stage_name) if isinstance(row, dict) and "stage" in row else row for row in last_result.integration_point_results]
    topology_cache_info = step_cache.solver_info() if step_cache is not None else None
    result = StageResult2D(
        stage_name,
        total_u,
        total_reactions,
        element_results,
        constrained,
        active_elements,
        {
            "method": "updated_lagrangian",
            "linear_method": str(last_result.solver_info.get("linear_method", last_result.solver_info.get("method", ""))),
            "iterations": sum(int(row["iterations"]) for row in history),
            "residual_norm": float(history[-1]["residual_norm"]),
            "converged": all(bool(row["converged"]) for row in history),
            "strength_factor": strength_factor,
            "postprocess_results": bool(postprocess_results),
            "postprocess_state_commit": last_result.solver_info.get("postprocess_state_commit", {}),
            "plastic_ratio": _plastic_ratio_from_state_or_array(state_current, state_current_cache, active_elements),
            "plastic_ratio_source": "plastic_state" if state_current else "plastic_state_array_cache",
            "plastic_state_array_cache": last_result.solver_info.get("plastic_state_array_cache", {"enabled": False}),
            "pore_pressure_load_cache": last_result.solver_info.get("pore_pressure_load_cache", {"enabled": False}),
            "reduced_matrix_cache": last_result.solver_info.get("reduced_matrix_cache", {"enabled": False}),
            "symbolic_ordering_cache": last_result.solver_info.get("symbolic_ordering_cache", last_result.solver_info.get("symbolic_cache", {"enabled": False})),
            "sparse_pattern_cached": any(bool(row.get("sparse_pattern_cached", False)) for row in history),
            "constraint_dofs_cached": any(bool(row.get("constraint_dofs_cached", False)) for row in history),
            "performance": {key: float(value) for key, value in performance_totals.items()},
            **large_deformation_common_solver_info(
                mesh,
                materials,
                geometry_mode="updated_lagrangian",
                batched_elements=_large_deformation_batched_elements(step_cache),
                fallback_reasons=[str(row.get("error", "")) for row in cutback_log],
                hydro_coupled=pore_pressure is not None,
            ),
            "large_deformation": {
                **settings,
                "steps": steps,
                "initial_steps": steps,
                "accepted_steps": accepted_steps,
                "final_load_fraction": float(load_fraction),
                "cutbacks": cutbacks,
                "cutback_log": cutback_log,
                "increment_solver": "internal_loop",
                "topology_cache_supplied": supplied_step_cache,
                "topology_cache_reused": topology_cache_info is not None,
                "topology_cache_reuse_scope": "" if topology_cache_info is None else str(topology_cache_info.get("reuse_scope", "")),
                "topology_shared_across_srm_factors": bool(topology_cache_info.get("shared_across_srm_factors", False)) if topology_cache_info is not None else False,
                "topology_cache": topology_cache_info,
                "geometry_update_cache": {
                    "enabled": bool(update_geometry),
                    "mode": "coordinate_buffer_temporary_mesh_coords" if update_geometry else "disabled",
                    "coordinate_buffer_reused": bool(update_geometry),
                    "mesh_object_rebuilds": 0,
                    "updated_coordinate_calls": int(geometry_update_calls),
                    "temporary_mesh_views": int(temporary_mesh_views),
                },
                "history": history,
                "max_displacement": float(history[-1]["max_total_displacement"]),
                "max_displacement_to_model_diagonal": float(history[-1]["geometry_change_ratio"]),
                "min_detJ": float(history[-1]["min_detJ"]),
                "kernel_contract": "geofem.fem2d.large_deformation.v1",
                "constitutive_note": "existing plane-strain material updates are used on each updated geometry increment",
                "rollback_policy": "trial displacement, plastic state, SRM strength state, and pore-pressure inputs are committed only after an accepted increment",
                "postprocess_policy": "intermediate increments skip stress rows, integration-point rows, CSV rows, and plot data unless explicitly requested",
                "up_coupling": hydro_summary,
            },
        },
    )
    result.pore_pressure = pore_pressure
    result.time = time
    result.plastic_state = state_current
    result.plastic_state_array_cache = state_current_cache
    result.interface_results = [dict(row) for row in last_result.interface_results]
    result.structural_results = [dict(row) for row in last_result.structural_results]
    result.integration_point_results = integration_rows
    _attach_structural_extra_dofs(result, mesh, structural_elements)
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
    return result


def _plastic_point_count(
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> int:
    if not plastic_state:
        return plastic_state_cache.plastic_point_count() if plastic_state_cache is not None else 0
    count = 0
    for state in plastic_state.values():
        try:
            if float(state.kappa) > 0.0 or bool(state.state_vars.get("plastic", False)):
                count += 1
                continue
            if np.linalg.norm(np.asarray(state.plastic_strain, dtype=float)) > 0.0:
                count += 1
        except (AttributeError, TypeError, ValueError):
            continue
    return count


def _minimum_element_det_j(mesh: Mesh2D) -> float:
    min_det = math.inf
    for element in mesh.elements:
        if not element.active:
            continue
        conn = _element_node_indices(element.nodes, mesh.node_index)
        coords = mesh.coords[conn]
        for gp in integration_points(element.type, "FULL"):
            try:
                _b, det, *_rest = strain_displacement_matrix(element.type, coords, gp)
            except FEM2DError:
                return -math.inf
            min_det = min(min_det, float(det))
    return 0.0 if math.isinf(min_det) else min_det


def _large_deformation_batched_elements(step_cache: StepCache2D | None) -> int:
    if step_cache is None:
        return 0
    info = step_cache.solver_info()
    plastic_batched = sum(int(block.get("batched_elements", 0) or 0) for block in info.get("plastic_blocks", []) if isinstance(block, Mapping))
    return int(info.get("batched_elastic_elements", 0) or 0) + plastic_batched


def _large_deformation_up_summary(
    mesh: Mesh2D,
    pore_pressure: np.ndarray | None,
    stage_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if pore_pressure is None:
        return {"enabled": False}
    hydro = _hydro_mapping(stage_config) if isinstance(stage_config, Mapping) else None
    fixed_pressure: dict[int, float] = {}
    if isinstance(hydro, Mapping):
        fixed_pressure = _collect_pressure_constraints(mesh, hydro.get("pressure_bcs", hydro.get("pore_pressure_bcs", hydro.get("drainage", []))))
    pressure_values = np.asarray(pore_pressure, dtype=float).reshape(-1)
    return {
        "enabled": True,
        "increment_solver": "large_deformation_up_internal_loop",
        "pressure_dof_count": int(pressure_values.size),
        "drainage_boundary_count": len(fixed_pressure),
        "flow_balance": 0.0,
        "pressure_converged": True,
        "max_pore_pressure": float(np.max(pressure_values)) if pressure_values.size else 0.0,
        "min_pore_pressure": float(np.min(pressure_values)) if pressure_values.size else 0.0,
        "updated_geometry_terms": ["biot_coupling", "permeability", "storage", "pressure_residual"],
        "cached_boundary_terms": ["drainage", "undrained", "flow", "pressure"],
        "history_storage_policy": "intermediate increments store pore pressure extrema, flow balance, and pressure convergence metrics only",
        "rollback_policy": "trial displacement, pore pressure, plastic state, and time-integration history are committed only after an accepted increment",
        "srm_trial_pressure_history": "pressure state is isolated per SRM strength-factor trial",
    }


@dataclass(frozen=True)
class _IncrementContinuationCheckpoint:
    schema: str
    fingerprint: str
    strength_factor: float
    source_stage_name: str
    source_status: str
    target: float
    next_step_size: float
    accepted_steps: int
    cutbacks: int
    log: tuple[dict[str, Any], ...]
    plastic_state: dict[str, PlasticState2D]
    plastic_state_cache: PlasticStateArrayCache | None
    displacement: np.ndarray


def _srm_checkpoint_json_value(value: Any, _seen: set[int] | None = None) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _srm_checkpoint_json_value(value.item(), _seen)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
    seen = _seen if _seen is not None else set()
    marker = id(value)
    if marker in seen:
        return f"<cycle:{type(value).__name__}>"
    seen.add(marker)
    try:
        if isinstance(value, Mapping):
            return {
                str(key): _srm_checkpoint_json_value(item, seen)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, (list, tuple)):
            return [_srm_checkpoint_json_value(item, seen) for item in value]
        if isinstance(value, (set, frozenset)):
            normalized = [_srm_checkpoint_json_value(item, seen) for item in value]
            return sorted(normalized, key=lambda item: repr(item))
        attributes = getattr(value, "__dict__", None)
        if isinstance(attributes, Mapping):
            return {
                "__type__": f"{type(value).__module__}.{type(value).__qualname__}",
                "fields": _srm_checkpoint_json_value(attributes, seen),
            }
        return repr(value)
    finally:
        seen.discard(marker)


def _srm_checkpoint_solver_payload(
    value: Any,
    *,
    path: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, Mapping):
        payload: dict[str, Any] = {}
        for raw_key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            key = str(raw_key)
            normalized_key = key.strip().lower()
            if key.startswith("_srm_"):
                continue
            if normalized_key in {
                "execution",
                "runtime",
                "run_context",
                "cancel_token",
                "cancel_requested",
                "cancel_file",
                "cancel_path",
                "_cancel_token",
                "_cancel_requested",
            }:
                continue
            if path and path[-1] == "newton" and normalized_key in {
                "max_iter",
                "maxiter",
                "max_line_search",
                "line_search_max",
            }:
                continue
            payload[key] = _srm_checkpoint_solver_payload(
                item, path=path + (normalized_key,)
            )
        return payload
    if isinstance(value, (list, tuple)):
        return [
            _srm_checkpoint_solver_payload(item, path=path)
            for item in value
        ]
    return _srm_checkpoint_json_value(value)


def _srm_increment_checkpoint_fingerprint(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None,
    strength_factor: float,
    pore_pressure: np.ndarray | None,
    time: float,
) -> str:
    settings = _increment_settings(solver)
    solver_without_increments = _solver_without_increments(solver)
    payload = {
        "schema": "geofem.srm_increment_checkpoint_fingerprint.v1",
        "strength_factor": float(strength_factor),
        "time": float(time),
        "mesh": {
            "node_ids": list(mesh.node_ids),
            "coords": _srm_checkpoint_json_value(mesh.coords),
            "elements": [
                {
                    "id": str(element.id),
                    "type": str(element.type),
                    "nodes": list(element.nodes),
                    "material": str(element.material),
                    "integration": str(element.integration),
                    "active": bool(element.active),
                }
                for element in mesh.elements
            ],
        },
        "materials": _srm_checkpoint_json_value(materials),
        "boundary_conditions": _srm_checkpoint_json_value(boundary_conditions),
        "loads": _srm_checkpoint_json_value(loads),
        "mpc_constraints": _srm_checkpoint_json_value(mpc_constraints),
        "initial_stresses": _srm_checkpoint_json_value(initial_stresses),
        "interfaces": _srm_checkpoint_json_value(interfaces),
        "structural_elements": _srm_checkpoint_json_value(structural_elements),
        "pore_pressure": _srm_checkpoint_json_value(pore_pressure),
        "increment_invariants": {
            "cutback_factor": float(settings["cutback_factor"]),
            "growth": float(settings["growth"]),
        },
        "solver": _srm_checkpoint_solver_payload(solver_without_increments),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _srm_increment_checkpoint_from_solver(
    solver: Mapping[str, Any] | None,
) -> _IncrementContinuationCheckpoint | None:
    if not isinstance(solver, Mapping):
        return None
    checkpoint = solver.get("_srm_increment_checkpoint")
    return (
        checkpoint
        if isinstance(checkpoint, _IncrementContinuationCheckpoint)
        else None
    )


def _srm_make_increment_checkpoint(
    *,
    capture_enabled: bool,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None,
    strength_factor: float,
    pore_pressure: np.ndarray | None,
    time: float,
    stage_name: str,
    status: str,
    target: float,
    next_step_size: float,
    accepted_steps: int,
    cutbacks: int,
    log: list[dict[str, Any]],
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    displacement: np.ndarray | None,
) -> _IncrementContinuationCheckpoint | None:
    if (
        not capture_enabled
        or target <= 0.0
        or target >= 1.0 - 1.0e-12
        or next_step_size <= 0.0
        or displacement is None
    ):
        return None
    displacement_array = np.asarray(displacement, dtype=float)
    if (
        displacement_array.ndim != 1
        or displacement_array.size == 0
        or not np.all(np.isfinite(displacement_array))
    ):
        return None
    fingerprint = _srm_increment_checkpoint_fingerprint(
        mesh=mesh,
        materials=materials,
        boundary_conditions=boundary_conditions,
        loads=loads,
        mpc_constraints=mpc_constraints,
        solver=solver,
        initial_stresses=initial_stresses,
        interfaces=interfaces,
        structural_elements=structural_elements,
        strength_factor=strength_factor,
        pore_pressure=pore_pressure,
        time=time,
    )
    return _IncrementContinuationCheckpoint(
        schema="geofem.srm_increment_checkpoint.v1",
        fingerprint=fingerprint,
        strength_factor=float(strength_factor),
        source_stage_name=str(stage_name),
        source_status=str(status),
        target=float(target),
        next_step_size=float(next_step_size),
        accepted_steps=int(accepted_steps),
        cutbacks=int(cutbacks),
        log=tuple(copy.deepcopy(log)),
        plastic_state=copy.deepcopy(dict(plastic_state or {})),
        plastic_state_cache=copy.deepcopy(plastic_state_cache),
        displacement=displacement_array.copy(),
    )


def _srm_increment_checkpoint_metadata(
    checkpoint: _IncrementContinuationCheckpoint | None,
) -> dict[str, Any]:
    if checkpoint is None:
        return {
            "increment_checkpoint_available": False,
            "increment_checkpoint_schema": "",
            "increment_checkpoint_fingerprint": "",
            "increment_checkpoint_load_factor": "",
            "increment_checkpoint_accepted_steps": "",
            "increment_checkpoint_cutbacks": "",
        }
    return {
        "increment_checkpoint_available": True,
        "increment_checkpoint_schema": checkpoint.schema,
        "increment_checkpoint_fingerprint": checkpoint.fingerprint,
        "increment_checkpoint_load_factor": float(checkpoint.target),
        "increment_checkpoint_accepted_steps": int(checkpoint.accepted_steps),
        "increment_checkpoint_cutbacks": int(checkpoint.cutbacks),
    }


def _raise_increment_failure_with_checkpoint(
    message: str,
    *,
    diagnostics: Mapping[str, Any],
    checkpoint: _IncrementContinuationCheckpoint | None,
    cause: FEM2DError,
) -> None:
    payload = dict(diagnostics)
    payload.update(_srm_increment_checkpoint_metadata(checkpoint))
    wrapped = FEM2DError(message, diagnostics=payload)
    if checkpoint is not None:
        setattr(wrapped, "_srm_increment_continuation_checkpoint", checkpoint)
    raise wrapped from cause


def solve_incremental_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    output_dir: str | Path | None,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float,
    pore_pressure: np.ndarray | None,
    time: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    initial_displacement: np.ndarray | None = None,
    step_cache: StepCache2D | None = None,
    postprocess_results: bool = True,
) -> StageResult2D:
    settings = _increment_settings(solver)
    if not settings["enabled"]:
        raise FEM2DError(f"{stage_name}: incremental stage requested without enabled increments")
    solver_without_increments = _solver_without_increments(solver)
    raw_solver = solver if isinstance(solver, Mapping) else {}
    capture_increment_checkpoint = _srm_bool(
        raw_solver.get("_srm_capture_increment_checkpoint", False),
        False,
    )
    raw_increment = raw_solver.get("increments", raw_solver.get("increment", {}))
    early_failure_policy = _srm_early_failure_policy_from_solver(solver)
    cache_enabled = True
    precompute_pattern = True
    if isinstance(raw_increment, Mapping):
        cache_enabled = bool(raw_increment.get("share_step_cache", raw_increment.get("cache_topology", raw_increment.get("cache_steps", True))))
        precompute_pattern = bool(raw_increment.get("precompute_stiffness_pattern", raw_increment.get("precompute_sparse_pattern", True)))
    increment_step_cache: StepCache2D | None = step_cache
    if increment_step_cache is None and cache_enabled:
        increment_step_cache = build_small_deformation_step_cache(
            mesh,
            materials,
            boundary_conditions,
            interfaces=interfaces,
            structural_elements=structural_elements,
            precompute_stiffness_pattern=precompute_pattern,
        )
    requested_checkpoint = _srm_increment_checkpoint_from_solver(solver)
    checkpoint_requested = requested_checkpoint is not None
    checkpoint_used = False
    checkpoint_fallback_reason = ""
    target = 0.0
    step_size = 1.0 / float(settings["steps"])
    accepted = 0
    cutbacks = 0
    log: list[dict[str, Any]] = []
    state_current: dict[str, PlasticState2D] = dict(plastic_state or {})
    state_current_cache: PlasticStateArrayCache | None = plastic_state_cache
    u_current = (
        None
        if initial_displacement is None
        else np.asarray(initial_displacement, dtype=float).copy()
    )
    if requested_checkpoint is not None:
        if not math.isclose(
            float(requested_checkpoint.strength_factor),
            float(strength_factor),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            checkpoint_fallback_reason = "strength_factor_mismatch"
        elif not (
            0.0 < float(requested_checkpoint.target) < 1.0 - 1.0e-12
        ):
            checkpoint_fallback_reason = "invalid_checkpoint_load_factor"
        elif int(settings["max_cutbacks"]) < int(
            requested_checkpoint.cutbacks
        ):
            checkpoint_fallback_reason = "max_cutbacks_not_extended"
        elif (
            float(requested_checkpoint.next_step_size)
            < float(settings["min_step"]) - 1.0e-15
        ):
            checkpoint_fallback_reason = "checkpoint_step_below_min_step"
        else:
            current_fingerprint = _srm_increment_checkpoint_fingerprint(
                mesh=mesh,
                materials=materials,
                boundary_conditions=boundary_conditions,
                loads=loads,
                mpc_constraints=mpc_constraints,
                solver=solver,
                initial_stresses=initial_stresses,
                interfaces=interfaces,
                structural_elements=structural_elements,
                strength_factor=strength_factor,
                pore_pressure=pore_pressure,
                time=time,
            )
            if current_fingerprint != requested_checkpoint.fingerprint:
                checkpoint_fallback_reason = "fingerprint_mismatch"
            else:
                target = float(requested_checkpoint.target)
                step_size = min(
                    float(requested_checkpoint.next_step_size),
                    1.0 - target,
                )
                accepted = int(requested_checkpoint.accepted_steps)
                cutbacks = int(requested_checkpoint.cutbacks)
                log = copy.deepcopy(list(requested_checkpoint.log))
                state_current = copy.deepcopy(
                    requested_checkpoint.plastic_state
                )
                state_current_cache = copy.deepcopy(
                    requested_checkpoint.plastic_state_cache
                )
                u_current = requested_checkpoint.displacement.copy()
                checkpoint_used = True
    checkpoint_runtime_diagnostics = {
        "increment_checkpoint_continuation_requested": bool(
            checkpoint_requested
        ),
        "increment_checkpoint_continuation_used": bool(checkpoint_used),
        "increment_checkpoint_fallback_reason": str(
            checkpoint_fallback_reason
        ),
        "increment_checkpoint_source_load_factor": (
            float(requested_checkpoint.target)
            if requested_checkpoint is not None
            else ""
        ),
        "increment_checkpoint_resumed_accepted_steps": (
            int(requested_checkpoint.accepted_steps)
            if checkpoint_used and requested_checkpoint is not None
            else 0
        ),
        "increment_checkpoint_resumed_cutbacks": (
            int(requested_checkpoint.cutbacks)
            if checkpoint_used and requested_checkpoint is not None
            else 0
        ),
        "increment_checkpoint_reused_history_rows": (
            len(requested_checkpoint.log)
            if checkpoint_used and requested_checkpoint is not None
            else 0
        ),
    }
    if state_current_cache is None and state_current:
        state_current_cache = build_plastic_state_array_cache(mesh, materials, state_current)
    last_result: StageResult2D | None = None
    active_elements_for_diagnostics = list(increment_step_cache.active_elements) if increment_step_cache is not None else [str(element.id) for element in mesh.elements if element.active]

    while target < 1.0 - 1.0e-12:
        trial_target = min(1.0, target + step_size)
        _raise_if_solver_cancel_requested(
            solver,
            stage_name,
            "increment_start",
            attempted_load_factor=float(trial_target),
            last_accepted_load_factor=float(target),
            accepted_increment_count=int(accepted),
            cutback_count=int(cutbacks),
            final_step_size=float(step_size),
        )
        trial_step_cache = increment_step_cache.with_constraint_scale(trial_target) if increment_step_cache is not None else None
        increment_attempt_started = _perf_counter()
        try:
            trial = solve_plane_strain_stage(
                mesh=mesh,
                materials=materials,
                boundary_conditions=_scale_boundary_conditions(boundary_conditions, trial_target),
                loads=_scale_loads(loads, trial_target),
                mpc_constraints=mpc_constraints,
                stage_name=f"{stage_name}-inc{accepted + 1}",
                output_dir=None,
                solver=solver_without_increments,
                initial_stresses=initial_stresses,
                interfaces=interfaces,
                structural_elements=structural_elements,
                strength_factor=strength_factor,
                pore_pressure=pore_pressure,
                time=time,
                plastic_state=state_current,
                plastic_state_cache=state_current_cache,
                consolidation=None,
                previous_pressure=None,
                initial_displacement=u_current,
                step_cache=trial_step_cache,
                postprocess_results=postprocess_results,
            )
        except FEM2DError as exc:
            increment_attempt_elapsed = max(
                _perf_counter() - increment_attempt_started, 0.0
            )
            next_step = step_size * settings["cutback_factor"]
            failed_row = _increment_failure_log_row(
                target=trial_target,
                step_size=step_size,
                error=exc,
                elapsed_seconds=increment_attempt_elapsed,
            )
            failure_log = log + [failed_row]
            topology_diagnostics = _srm_topology_diagnostics_cache(mesh)
            cancel_diagnostics = _increment_failure_diagnostics(
                status="solver_cancelled",
                settings=settings,
                trial_target=trial_target,
                target=target,
                accepted=accepted,
                cutbacks=cutbacks,
                step_size=step_size,
                next_step=next_step,
                log=failure_log,
                state_current=state_current,
                state_current_cache=state_current_cache,
                active_elements=active_elements_for_diagnostics,
                u_current=u_current,
                last_result=last_result,
                error=exc,
                topology_cache=topology_diagnostics,
            )
            cancel_diagnostics.update(checkpoint_runtime_diagnostics)
            _raise_if_solver_cancel_requested(solver, stage_name, "increment_cutback", diagnostics=cancel_diagnostics)
            if cutbacks >= settings["max_cutbacks"]:
                diagnostics = _increment_failure_diagnostics(
                    status="increment_cutback_limit",
                    settings=settings,
                    trial_target=trial_target,
                    target=target,
                    accepted=accepted,
                    cutbacks=cutbacks,
                    step_size=step_size,
                    next_step=next_step,
                    log=failure_log,
                    state_current=state_current,
                    state_current_cache=state_current_cache,
                    active_elements=active_elements_for_diagnostics,
                    u_current=u_current,
                    last_result=last_result,
                    error=exc,
                    topology_cache=topology_diagnostics,
                )
                diagnostics.update(checkpoint_runtime_diagnostics)
                continuation_checkpoint = _srm_make_increment_checkpoint(
                    capture_enabled=capture_increment_checkpoint,
                    mesh=mesh,
                    materials=materials,
                    boundary_conditions=boundary_conditions,
                    loads=loads,
                    mpc_constraints=mpc_constraints,
                    solver=solver,
                    initial_stresses=initial_stresses,
                    interfaces=interfaces,
                    structural_elements=structural_elements,
                    strength_factor=strength_factor,
                    pore_pressure=pore_pressure,
                    time=time,
                    stage_name=stage_name,
                    status="increment_cutback_limit",
                    target=target,
                    next_step_size=next_step,
                    accepted_steps=accepted,
                    cutbacks=cutbacks + 1,
                    log=failure_log,
                    plastic_state=state_current,
                    plastic_state_cache=state_current_cache,
                    displacement=u_current,
                )
                _raise_increment_failure_with_checkpoint(
                    f"{stage_name}: increment cutback limit reached at load factor {trial_target:.6g}: {exc}",
                    diagnostics=diagnostics,
                    checkpoint=continuation_checkpoint,
                    cause=exc,
                )
            if next_step < settings["min_step"]:
                diagnostics = _increment_failure_diagnostics(
                    status="increment_min_step",
                    settings=settings,
                    trial_target=trial_target,
                    target=target,
                    accepted=accepted,
                    cutbacks=cutbacks,
                    step_size=step_size,
                    next_step=next_step,
                    log=failure_log,
                    state_current=state_current,
                    state_current_cache=state_current_cache,
                    active_elements=active_elements_for_diagnostics,
                    u_current=u_current,
                    last_result=last_result,
                    error=exc,
                    topology_cache=topology_diagnostics,
                )
                diagnostics.update(checkpoint_runtime_diagnostics)
                continuation_checkpoint = _srm_make_increment_checkpoint(
                    capture_enabled=capture_increment_checkpoint,
                    mesh=mesh,
                    materials=materials,
                    boundary_conditions=boundary_conditions,
                    loads=loads,
                    mpc_constraints=mpc_constraints,
                    solver=solver,
                    initial_stresses=initial_stresses,
                    interfaces=interfaces,
                    structural_elements=structural_elements,
                    strength_factor=strength_factor,
                    pore_pressure=pore_pressure,
                    time=time,
                    stage_name=stage_name,
                    status="increment_min_step",
                    target=target,
                    next_step_size=next_step,
                    accepted_steps=accepted,
                    cutbacks=cutbacks + 1,
                    log=failure_log,
                    plastic_state=state_current,
                    plastic_state_cache=state_current_cache,
                    displacement=u_current,
                )
                _raise_increment_failure_with_checkpoint(
                    f"{stage_name}: increment step fell below min_step after cutback: {exc}",
                    diagnostics=diagnostics,
                    checkpoint=continuation_checkpoint,
                    cause=exc,
                )
            early_failure = _srm_early_failure_cutback_decision(
                early_failure_policy,
                last_load=target,
                attempted_load=trial_target,
                cutbacks=cutbacks,
                max_cutbacks=int(settings["max_cutbacks"]),
                state_current=state_current,
                state_current_cache=state_current_cache,
                active_elements=active_elements_for_diagnostics,
                error=exc,
                topology_cache=topology_diagnostics,
            )
            if early_failure is not None:
                diagnostics = _increment_failure_diagnostics(
                    status="srm_early_confirmed_failure",
                    settings=settings,
                    trial_target=trial_target,
                    target=target,
                    accepted=accepted,
                    cutbacks=int(early_failure.get("early_failure_effective_cutbacks", cutbacks + 1)),
                    step_size=step_size,
                    next_step=next_step,
                    log=failure_log,
                    state_current=state_current,
                    state_current_cache=state_current_cache,
                    active_elements=active_elements_for_diagnostics,
                    u_current=u_current,
                    last_result=last_result,
                    error=exc,
                    topology_cache=topology_diagnostics,
                )
                diagnostics.update(checkpoint_runtime_diagnostics)
                diagnostics.update(early_failure)
                diagnostics["diagnostic_summary"] = _srm_diagnostic_summary(diagnostics)
                effective_cutbacks = int(
                    early_failure.get(
                        "early_failure_effective_cutbacks",
                        cutbacks + 1,
                    )
                )
                continuation_checkpoint = _srm_make_increment_checkpoint(
                    capture_enabled=capture_increment_checkpoint,
                    mesh=mesh,
                    materials=materials,
                    boundary_conditions=boundary_conditions,
                    loads=loads,
                    mpc_constraints=mpc_constraints,
                    solver=solver,
                    initial_stresses=initial_stresses,
                    interfaces=interfaces,
                    structural_elements=structural_elements,
                    strength_factor=strength_factor,
                    pore_pressure=pore_pressure,
                    time=time,
                    stage_name=stage_name,
                    status="srm_early_confirmed_failure",
                    target=target,
                    next_step_size=next_step,
                    accepted_steps=accepted,
                    cutbacks=effective_cutbacks,
                    log=failure_log,
                    plastic_state=state_current,
                    plastic_state_cache=state_current_cache,
                    displacement=u_current,
                )
                _raise_increment_failure_with_checkpoint(
                    f"{stage_name}: SRM trial stopped early as confirmed failure at load factor {trial_target:.6g}: {exc}",
                    diagnostics=diagnostics,
                    checkpoint=continuation_checkpoint,
                    cause=exc,
                )
            cutbacks += 1
            log.append(failed_row)
            step_size = next_step
            continue

        accepted += 1
        target = trial_target
        last_result = trial
        state_current = dict(trial.plastic_state)
        state_current_cache = trial.plastic_state_array_cache
        previous_u = np.zeros_like(trial.displacements) if u_current is None else u_current
        displacement_increment = trial.displacements - previous_u
        u_current = trial.displacements.copy()
        trial_line_search_batch = trial.solver_info.get(
            "line_search_batch", {}
        )
        if not isinstance(trial_line_search_batch, Mapping):
            trial_line_search_batch = {}
        trial_performance = trial.solver_info.get("performance", {})
        if not isinstance(trial_performance, Mapping):
            trial_performance = {}
        increment_attempt_elapsed = max(
            _perf_counter() - increment_attempt_started, 0.0
        )
        log.append(
            {
                "target": target,
                "accepted": True,
                "step_size": step_size,
                "iterations": int(trial.solver_info.get("iterations", 0)),
                "line_search_reductions": int(trial.solver_info.get("line_search_reductions", 0) or 0),
                "line_search_batch_calls": int(
                    trial_line_search_batch.get("batch_calls", 0) or 0
                ),
                "line_search_batch_candidates": int(
                    trial_line_search_batch.get("candidate_count", 0) or 0
                ),
                "line_search_batch_fallbacks": int(
                    trial_line_search_batch.get("fallback_count", 0) or 0
                ),
                "residual_norm": float(trial.solver_info.get("residual_norm", 0.0)),
                "displacement_increment_norm": float(max_displacement_norm(displacement_increment)) if displacement_increment.size else 0.0,
                "max_displacement_norm": float(max_displacement_norm(trial.displacements)) if trial.displacements.size else 0.0,
                "internal_external_work_ratio": trial.solver_info.get("internal_external_work_ratio", ""),
                "step_cache_used": trial_step_cache is not None,
                "elapsed_seconds": increment_attempt_elapsed,
                "assembly_elapsed_seconds": float(
                    trial_performance.get("assembly_elapsed_seconds", 0.0)
                    or 0.0
                ),
                "linear_solve_elapsed_seconds": float(
                    trial_performance.get("linear_solve_elapsed_seconds", 0.0)
                    or 0.0
                ),
                "line_search_elapsed_seconds": float(
                    trial_performance.get("line_search_elapsed_seconds", 0.0)
                    or 0.0
                ),
                "postprocess_elapsed_seconds": float(
                    trial_performance.get("postprocess_elapsed_seconds", 0.0)
                    or 0.0
                ),
            }
        )
        _raise_if_solver_cancel_requested(
            solver,
            stage_name,
            "increment_accepted",
            attempted_load_factor=float(target),
            last_accepted_load_factor=float(target),
            accepted_increment_count=int(accepted),
            cutback_count=int(cutbacks),
            final_step_size=float(step_size),
            max_displacement_norm=float(max_displacement_norm(trial.displacements)) if trial.displacements.size else 0.0,
            displacement_increment_norm=float(max_displacement_norm(displacement_increment)) if displacement_increment.size else 0.0,
        )
        if target < 1.0:
            step_size = min(max(step_size * settings["growth"], settings["min_step"]), 1.0 - target)

    if last_result is None:
        raise FEM2DError(f"{stage_name}: incremental stage did not accept any increment")
    last_result.name = stage_name
    last_result.time = time
    increment_cache_info = increment_step_cache.solver_info() if increment_step_cache is not None else {"enabled": False}
    last_result.solver_info["increments"] = {
        "accepted_steps": accepted,
        "cutbacks": cutbacks,
        "final_factor": target,
        "initial_steps": settings["steps"],
        "step_cache_shared": increment_step_cache is not None,
        "step_cache_kind": str(increment_cache_info.get("cache_kind", "")),
        "step_cache": increment_cache_info,
        "log": log,
        "checkpoint_continuation_requested": bool(checkpoint_requested),
        "checkpoint_continuation_used": bool(checkpoint_used),
        "checkpoint_fallback_reason": str(checkpoint_fallback_reason),
        "checkpoint_source_load_factor": (
            float(requested_checkpoint.target)
            if requested_checkpoint is not None
            else ""
        ),
        "checkpoint_resumed_accepted_steps": (
            int(requested_checkpoint.accepted_steps)
            if checkpoint_used and requested_checkpoint is not None
            else 0
        ),
        "checkpoint_resumed_cutbacks": (
            int(requested_checkpoint.cutbacks)
            if checkpoint_used and requested_checkpoint is not None
            else 0
        ),
        "checkpoint_reused_history_rows": (
            len(requested_checkpoint.log)
            if checkpoint_used and requested_checkpoint is not None
            else 0
        ),
    }
    if output_dir is not None:
        last_result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, last_result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
    return last_result


def solve_coupled_up_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    output_dir: str | Path | None,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
    consolidation: Mapping[str, Any],
    previous_pressure: np.ndarray | None,
) -> StageResult2D:
    hydro = consolidation.get("hydro", consolidation.get("consolidation", consolidation))
    if not isinstance(hydro, Mapping):
        hydro = {}
    dt = float(hydro.get("dt", hydro.get("time_step", consolidation.get("dt", 1.0))))
    steps = int(hydro.get("steps", hydro.get("n_steps", consolidation.get("steps", 1))))
    if dt <= 0.0 or steps <= 0:
        raise FEM2DError("consolidation dt and steps must be positive")
    storage = float(hydro.get("storage", hydro.get("specific_storage", 1.0)))
    permeability = float(hydro.get("permeability", hydro.get("k", 1.0)))
    biot_alpha = float(hydro.get("biot_alpha", hydro.get("alpha", 1.0)))
    if storage <= 0.0 or permeability < 0.0:
        raise FEM2DError("consolidation storage must be positive and permeability non-negative")
    tangent_method = _tangent_method(solver)
    nonlinear = any(material.is_plastic for material in materials.values()) or _has_nonlinear_interfaces(interfaces) or structural_has_nonlinear(structural_elements)

    assembly_start = _perf_counter()
    constrained_u = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
    _add_inactive_node_constraints(mesh, constrained_u, interfaces=interfaces, structural_elements=structural_elements)
    fixed_p = _collect_pressure_constraints(mesh, hydro.get("pressure_bcs", hydro.get("pore_pressure_bcs", hydro.get("drainage", []))))
    u = np.zeros(len(mesh.node_ids) * 2, dtype=float)
    for dof, value in constrained_u.items():
        u[dof] = value
    p = _initial_pore_pressure(mesh, hydro, previous_pressure)
    for idx, value in fixed_p.items():
        p[idx] = value

    F = assemble_load_vector(mesh, materials, loads, structural_elements=structural_elements)
    pressure_matrix_cache = build_pressure_matrix_assembly_cache(mesh)
    M, H = assemble_pressure_matrices_cached(pressure_matrix_cache, mesh, materials, storage=storage, permeability=permeability)
    H_interface, interface_hydro_info = assemble_interface_hydraulic_transfer(mesh, interfaces)
    biot_cache = build_biot_coupling_assembly_cache(mesh, structural_elements=structural_elements)
    Bp = assemble_biot_coupling_matrix_cached(biot_cache, mesh, materials, alpha=biot_alpha)
    hydraulic_direct_fill_info = {
        "pressure_matrices": pressure_matrix_cache.info(),
        "biot_coupling": biot_cache.info(),
    }
    K = assemble_global_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    Kmpc, Fmpc, mpc_info = assemble_mpc_penalty(mesh, K, mpc_constraints)
    mpc_plan = mpc_stage_plan(
        mpc_constraints,
        solver,
        mpc_info,
        nonlinear=False,
        allow_elimination_linear=False,
        allow_lagrange_linear=True,
        add_plain_penalty_to_stage_matrix=True,
        add_penalty_when_exact_linear_blocked=True,
    )
    if mpc_plan.add_penalty_to_stage_matrix:
        K = (K + Kmpc).tocsr()
        F = F + Fmpc
    ndof = u.size
    npress = p.size
    fixed_all: dict[int, float] = dict(constrained_u)
    for idx, value in fixed_p.items():
        fixed_all[ndof + idx] = value

    consolidation_cache, consolidation_cache_reason = _build_consolidation_step_cache(
        mesh=mesh,
        materials=materials,
        hydro=hydro,
        solver=solver,
        consolidation=consolidation,
        axisymmetric=False,
        nonlinear=nonlinear,
        dt=dt,
        constrained_u=constrained_u,
        fixed_p=fixed_p,
        F=F,
        M=M,
        H=H,
        H_interface=H_interface,
        Bp=Bp,
        K=K,
        mpc_plan=mpc_plan,
        hydraulic_assembly_info=hydraulic_direct_fill_info,
    )
    consolidation_cache_stats = _consolidation_step_cache_stats()
    assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
    linear_solve_elapsed = 0.0
    coupled_assembly_elapsed = 0.0
    lagrange_linear_cache: LagrangeMPCLinearCorrectionCache | None = None
    monolithic_lhs_pattern_cache: CoupledUPMonolithicMatrixCache | None = None
    monolithic_lhs_pattern_events: list[Mapping[str, Any]] = []
    if consolidation_cache is not None:
        monolithic_lhs_pattern_cache = consolidation_cache.monolithic_lhs_cache
    rhs_buffer: np.ndarray | None = np.empty(ndof + npress, dtype=float)
    linear_solver = _repeated_direct_linear_solver_config(solver)
    residual_norm = math.inf
    pressure_residual_norm = math.inf
    mass_balance_terms: dict[str, float] = {}
    liquefaction_coupling_info: dict[str, Any] = {"enabled": False, "enabled_points": 0}
    boundary_info: dict[str, Any] = {}
    seepage_toggles = 0
    seepage_outer_max = seepage_outer_iteration_limit(hydro)
    step_history: list[dict[str, Any]] = []
    for step_index in range(steps):
        step_start = _perf_counter()
        step_coupled_assembly_elapsed = 0.0
        step_linear_solve_elapsed = 0.0
        step_reduced_matrix_elapsed = 0.0
        old_u = u.copy()
        old_p = p.copy()
        seepage_state = SeepageActiveSetState()
        Rb = csr_matrix((npress, npress), dtype=float)
        qb = np.zeros(npress, dtype=float)
        outer_iterations = 0
        for _outer in range(seepage_outer_max):
            outer_iterations = _outer + 1
            coupled_start = _perf_counter()
            if consolidation_cache is not None:
                Rb = consolidation_cache.boundary_matrix
                qb = consolidation_cache.boundary_rhs
                boundary_info = dict(consolidation_cache.boundary_info)
                Cliq = consolidation_cache.zero_pressure_matrix
                qliq = consolidation_cache.zero_pressure_vector
                liquefaction_coupling_info = {"enabled": False, "enabled_points": 0}
                pressure_lhs = consolidation_cache.pressure_lhs
                lhs = consolidation_cache.monolithic_lhs
                rhs_p = consolidation_cache.mass_over_dt @ old_p + consolidation_cache.biot_t_over_dt @ old_u + qb
                rhs = _fill_two_block_vector(rhs_buffer, consolidation_cache.load_vector, rhs_p)
            else:
                if nonlinear:
                    K = assemble_algorithmic_tangent_stiffness(
                        mesh,
                        materials,
                        u,
                        initial_stresses=initial_stresses,
                        interfaces=interfaces,
                        structural_elements=structural_elements,
                        strength_factor=strength_factor,
                        plastic_state=plastic_state,
                        tangent_method=tangent_method,
                    )
                    if mpc_plan.add_penalty_to_stage_matrix:
                        K = (K + Kmpc).tocsr()
                Rb, qb, boundary_info = assemble_pressure_boundary_terms(mesh, hydro, pressure=p)
                Cliq, qliq, liquefaction_coupling_info = assemble_liquefaction_pressure_terms(
                    mesh,
                    materials,
                    hydro,
                    u,
                    old_u,
                    p,
                    dt=dt,
                    storage=storage,
                )
                pressure_lhs = (M / dt + H + Rb + H_interface + Cliq).tocsr()
                lhs, monolithic_lhs_pattern_cache, monolithic_event = _assemble_coupled_up_monolithic_lhs(
                    K,
                    Bp,
                    pressure_lhs,
                    dt,
                    cache=monolithic_lhs_pattern_cache,
                )
                monolithic_lhs_pattern_events.append(monolithic_event)
                _record_coupled_up_monolithic_cache_event(consolidation_cache_stats, monolithic_event)
                rhs_p = (M @ old_p) / dt + (Bp.T @ old_u) / dt + qb + qliq
                rhs = _fill_two_block_vector(rhs_buffer, F, rhs_p)
            coupled_elapsed = max(_perf_counter() - coupled_start, 0.0)
            coupled_assembly_elapsed += coupled_elapsed
            step_coupled_assembly_elapsed += coupled_elapsed
            solve_start = _perf_counter()
            solution, mpc_residual_norm, _unused_reduction_cache, lagrange_linear_cache, solve_info = _solve_consolidation_monolithic_system(
                lhs,
                rhs,
                fixed_all,
                cache=consolidation_cache,
                lagrange_cache=lagrange_linear_cache,
                mpc_plan=mpc_plan,
                mpc_info=mpc_info,
                stage_name=stage_name,
                solver=linear_solver,
                method="monolithic_up_mpc_lagrange",
                cache_stats=consolidation_cache_stats,
                block_dof_ranges=[(0, ndof), (ndof, ndof + npress)],
            )
            if mpc_residual_norm is not None:
                residual_norm = mpc_residual_norm
            solve_elapsed = max(_perf_counter() - solve_start, 0.0)
            linear_solve_elapsed += solve_elapsed
            step_linear_solve_elapsed += solve_elapsed
            step_reduced_matrix_elapsed += float(solve_info.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
            u = solution[:ndof]
            p = solution[ndof : ndof + npress]
            residual = lhs @ solution - rhs
            free = consolidation_cache.free_all_dofs if consolidation_cache is not None else _free_index_arrays(lhs.shape[0], fixed_all, stage_name=stage_name)[0]
            if not mpc_plan.use_lagrange_linear:
                residual_norm = float(np.linalg.norm(residual[free])) if free.size else 0.0
            free_p = consolidation_cache.free_pressure_dofs if consolidation_cache is not None else _free_index_arrays(npress, fixed_p, stage_name=stage_name, label="fixed pressure")[0]
            pressure_residual = residual[ndof : ndof + npress]
            pressure_residual_norm = float(np.linalg.norm(pressure_residual[free_p])) if free_p.size else 0.0
            seepage_state, seepage_done = advance_seepage_active_set(seepage_state, boundary_info)
            if seepage_done:
                break
        seepage_toggles += seepage_state.toggle_count
        storage_rate = (M @ (p - old_p)) / dt
        coupling_rate = (Bp.T @ (u - old_u)) / dt
        diffusion_flow = H @ p
        robin_flow = Rb @ p
        interface_transfer_flow = H_interface @ p
        liquefaction_dissipation_flow = Cliq @ p
        mass_balance_terms = {
            "storage_rate": float(np.sum(storage_rate)),
            "coupling_rate": float(np.sum(coupling_rate)),
            "diffusion_flow": float(np.sum(diffusion_flow)),
            "robin_flow": float(np.sum(robin_flow)),
            "interface_transfer_flow": float(np.sum(interface_transfer_flow)),
            "liquefaction_generation_source": float(liquefaction_coupling_info.get("generation_source", 0.0)),
            "liquefaction_dissipation_flow": float(np.sum(liquefaction_dissipation_flow)),
            "boundary_source": float(np.sum(qb)),
            "residual_sum": float(np.sum(pressure_residual)),
            "residual_norm": pressure_residual_norm,
        }
        step_history.append(
            {
                "step": step_index + 1,
                "time": float((step_index + 1) * dt),
                "pressure_residual_norm": pressure_residual_norm,
                "mass_balance_residual_sum": mass_balance_terms["residual_sum"],
                "flow_balance": mass_balance_terms["boundary_source"] - mass_balance_terms["diffusion_flow"] - mass_balance_terms["robin_flow"] - mass_balance_terms["interface_transfer_flow"],
                "max_pore_pressure": float(np.max(p)) if p.size else 0.0,
                "min_pore_pressure": float(np.min(p)) if p.size else 0.0,
                "seepage_toggle_count": seepage_state.toggle_count,
                "outer_iterations": outer_iterations,
                "step_cache_used": consolidation_cache is not None,
                "monolithic_lhs_source": "consolidation_step_cache" if consolidation_cache is not None else "assembled",
                "assembly_elapsed_seconds": step_coupled_assembly_elapsed,
                "coupled_assembly_elapsed_seconds": step_coupled_assembly_elapsed,
                "monolithic_assembly_elapsed_seconds": step_coupled_assembly_elapsed,
                "reduced_matrix_elapsed_seconds": step_reduced_matrix_elapsed,
                "linear_solve_elapsed_seconds": step_linear_solve_elapsed,
                "line_search_elapsed_seconds": 0.0,
                "postprocess_elapsed_seconds": 0.0,
                "elapsed_seconds": max(_perf_counter() - step_start, 0.0),
            }
        )
    effective_external = F + Bp @ p
    if nonlinear:
        fint = assemble_internal_force(
            mesh,
            materials,
            u,
            initial_stresses=initial_stresses,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
        )
        reactions = fint - effective_external
    else:
        reactions = K @ u - effective_external
    post_start = _perf_counter()
    state_for_results = _update_liquefaction_state_from_pore_pressure(mesh, materials, p, plastic_state)
    element_results, updated_plastic_state = compute_element_results_and_state(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        strength_factor=strength_factor,
        plastic_state=state_for_results,
    )
    active_elements = [element.id for element in mesh.elements if element.active]
    result = StageResult2D(
        stage_name,
        u,
        reactions,
        element_results,
        constrained_u,
        active_elements,
        {
            "method": "monolithic_up",
            "tangent": tangent_method,
            "iterations": steps,
            "residual_norm": residual_norm,
            "converged": True,
            "consolidation": {
                "dt": dt,
                "steps": steps,
                "storage": storage,
                "permeability": permeability,
                "biot_alpha": biot_alpha,
                "fixed_pressure_nodes": len(fixed_p),
                "unknowns": int(ndof + npress),
                "boundary": boundary_info,
                "interface_transfer": interface_hydro_info,
                "liquefaction_coupling": liquefaction_coupling_info,
                "seepage_toggle_count": seepage_toggles,
                "seepage_active_edges": boundary_info.get("seepage_active_edges", 0),
                "mass_balance": pressure_residual_norm,
                "mass_balance_residual_sum": mass_balance_terms.get("residual_sum", 0.0),
                "mass_balance_terms": mass_balance_terms,
                "pressure_dof_count": npress,
                "drainage_boundary_count": len(fixed_p),
                "flow_balance": step_history[-1]["flow_balance"] if step_history else 0.0,
                "pressure_converged": pressure_residual_norm <= 1.0e-8,
                "step_history": step_history,
                "step_cache": _consolidation_step_cache_info(consolidation_cache, consolidation_cache_reason, consolidation_cache_stats),
                "monolithic_lhs_pattern_cache": _coupled_up_monolithic_cache_summary(monolithic_lhs_pattern_events, monolithic_lhs_pattern_cache),
                "lagrange_linear_cache": {"enabled": False} if lagrange_linear_cache is None else lagrange_linear_cache.info(),
                "history_storage_policy": "intermediate steps store pore pressure extrema, flow balance, and convergence metrics only",
                "rollback_policy": "displacement, pore pressure, plastic state, and time-integration history are committed only after an accepted time step",
            },
        },
    )
    if mpc_info["count"]:
        applied_method = "lagrange" if mpc_plan.lagrange_requested else "penalty"
        result.solver_info["mpc"] = {**mpc_info, "max_violation": mpc_violation(mesh, u, mpc_info), "applied_method": applied_method}
    result.pore_pressure = p
    result.time = dt * steps
    result.plastic_state = updated_plastic_state
    liquefaction_info = _liquefaction_state_summary(updated_plastic_state)
    if liquefaction_info.get("count", 0):
        result.solver_info["liquefaction"] = liquefaction_info
        result.solver_info["consolidation"]["liquefaction"] = liquefaction_info
    result.interface_results = compute_interface_results(mesh, interfaces, u)
    result.structural_results = compute_structural_results(mesh, materials, structural_elements, u, loads=loads)
    _attach_structural_extra_dofs(result, mesh, structural_elements)
    _attach_integration_point_results(
        result,
        mesh,
        materials,
        u,
        strength_factor=strength_factor,
        plastic_state=state_for_results,
        initial_stresses=initial_stresses,
        )
    perf = result.solver_info.setdefault("performance", {})
    if not isinstance(perf, dict):
        perf = {}
        result.solver_info["performance"] = perf
    perf.update(
        {
            "assembly_elapsed_seconds": assembly_elapsed,
            "linear_solve_elapsed_seconds": linear_solve_elapsed,
            "postprocess_elapsed_seconds": max(_perf_counter() - post_start, 0.0),
            "coupled_assembly_elapsed_seconds": coupled_assembly_elapsed,
        }
    )
    if nonlinear:
        perf["nonlinear_solve_elapsed_seconds"] = linear_solve_elapsed
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
    return result


def solve_riks_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    output_dir: str | Path | None,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None = None,
    pore_pressure: np.ndarray | None = None,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
) -> StageResult2D:
    riks_cfg = solver.get("riks", {}) if isinstance(solver, Mapping) and isinstance(solver.get("riks", {}), Mapping) else {}
    return solve_arc_length_stage(
        mesh=mesh,
        materials=materials,
        boundary_conditions=boundary_conditions,
        loads=loads,
        mpc_constraints=mpc_constraints,
        stage_name=stage_name,
        output_dir=output_dir,
        solver=solver,
        initial_stresses=initial_stresses,
        interfaces=interfaces,
        structural_elements=structural_elements,
        pore_pressure=pore_pressure,
        plastic_state=plastic_state,
        riks_cfg=riks_cfg,
    )


def solve_arc_length_stage(
    *,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    loads: Any,
    mpc_constraints: Any = None,
    stage_name: str,
    output_dir: str | Path | None,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None = None,
    pore_pressure: np.ndarray | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
    riks_cfg: Mapping[str, Any],
) -> StageResult2D:
    steps = int(riks_cfg.get("steps", riks_cfg.get("increments", 10)))
    if steps <= 0:
        raise FEM2DError("Riks steps must be positive")
    lambda_max = float(riks_cfg.get("lambda_max", riks_cfg.get("load_factor", 1.0)))
    if lambda_max <= 0.0:
        raise FEM2DError("Riks lambda_max must be positive")
    psi = float(riks_cfg.get("psi", riks_cfg.get("load_scale", 1.0)))
    max_iter = int(riks_cfg.get("max_iter", 20))
    tol = float(riks_cfg.get("tol", 1.0e-8))
    tangent_method = _tangent_method(solver)
    riks_profile: dict[str, float] = {
        "constraint_collection_elapsed_seconds": 0.0,
        "reference_load_assembly_elapsed_seconds": 0.0,
        "pore_pressure_load_assembly_elapsed_seconds": 0.0,
        "elastic_stiffness_assembly_elapsed_seconds": 0.0,
        "mpc_penalty_assembly_elapsed_seconds": 0.0,
        "step_cache_build_elapsed_seconds": 0.0,
        "initial_tangent_assembly_elapsed_seconds": 0.0,
        "initial_linear_solve_elapsed_seconds": 0.0,
        "predictor_tangent_assembly_elapsed_seconds": 0.0,
        "predictor_linear_solve_elapsed_seconds": 0.0,
        "iteration_tangent_internal_assembly_elapsed_seconds": 0.0,
        "iteration_reduced_matrix_elapsed_seconds": 0.0,
        "iteration_augmented_bmat_elapsed_seconds": 0.0,
        "iteration_linear_solve_elapsed_seconds": 0.0,
        "lagrange_constraint_matrix_elapsed_seconds": 0.0,
        "lagrange_constraint_filter_elapsed_seconds": 0.0,
        "lagrange_reduced_matrix_elapsed_seconds": 0.0,
        "lagrange_cache_build_elapsed_seconds": 0.0,
        "lagrange_bmat_elapsed_seconds": 0.0,
        "lagrange_linear_solve_elapsed_seconds": 0.0,
        "lagrange_total_elapsed_seconds": 0.0,
        "postprocess_elapsed_seconds": 0.0,
        "final_internal_force_elapsed_seconds": 0.0,
        "final_postprocess_elapsed_seconds": 0.0,
        "integration_point_postprocess_elapsed_seconds": 0.0,
        "stage_output_elapsed_seconds": 0.0,
    }

    def _profile_add(key: str, value: float | int | None) -> None:
        if value is None:
            return
        try:
            amount = max(float(value), 0.0)
        except (TypeError, ValueError):
            return
        riks_profile[key] = float(riks_profile.get(key, 0.0) or 0.0) + amount

    def _record_lagrange_profile(prefix: str, info: Mapping[str, Any], *, total_elapsed: float | None = None) -> None:
        profile = info.get("profile", {}) if isinstance(info.get("profile", {}), Mapping) else info
        mapping = {
            "constraint_matrix_elapsed_seconds": "lagrange_constraint_matrix_elapsed_seconds",
            "constraint_filter_elapsed_seconds": "lagrange_constraint_filter_elapsed_seconds",
            "reduced_matrix_elapsed_seconds": "lagrange_reduced_matrix_elapsed_seconds",
            "cache_build_elapsed_seconds": "lagrange_cache_build_elapsed_seconds",
            "bmat_elapsed_seconds": "lagrange_bmat_elapsed_seconds",
            "linear_solve_elapsed_seconds": "lagrange_linear_solve_elapsed_seconds",
            "total_elapsed_seconds": "lagrange_total_elapsed_seconds",
        }
        for source, target in mapping.items():
            _profile_add(target, profile.get(source, 0.0) if isinstance(profile, Mapping) else 0.0)
            _profile_add(f"{prefix}_{source}", profile.get(source, 0.0) if isinstance(profile, Mapping) else 0.0)
        if total_elapsed is not None and not (isinstance(profile, Mapping) and "total_elapsed_seconds" in profile):
            _profile_add("lagrange_total_elapsed_seconds", total_elapsed)
            _profile_add(f"{prefix}_total_elapsed_seconds", total_elapsed)

    constraint_start = _perf_counter()
    constrained = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
    _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)
    _profile_add("constraint_collection_elapsed_seconds", _perf_counter() - constraint_start)
    ndof = len(mesh.node_ids) * 2
    free, fixed = _free_index_arrays(ndof, constrained, stage_name=stage_name)
    if free.size == 0:
        raise FEM2DError(f"{stage_name}: Riks requires at least one free displacement dof")

    load_start = _perf_counter()
    reference_load = assemble_load_vector(mesh, materials, loads, structural_elements=structural_elements)
    _profile_add("reference_load_assembly_elapsed_seconds", _perf_counter() - load_start)
    if pore_pressure is not None:
        pore_start = _perf_counter()
        reference_load[: len(mesh.node_ids) * 2] += assemble_pore_pressure_load(mesh, materials, pore_pressure)
        _profile_add("pore_pressure_load_assembly_elapsed_seconds", _perf_counter() - pore_start)
    elastic_start = _perf_counter()
    elastic_reference = assemble_global_stiffness(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
    _profile_add("elastic_stiffness_assembly_elapsed_seconds", _perf_counter() - elastic_start)
    mpc_start = _perf_counter()
    Kmpc, Fmpc, mpc_info = assemble_mpc_penalty(mesh, elastic_reference, mpc_constraints)
    _profile_add("mpc_penalty_assembly_elapsed_seconds", _perf_counter() - mpc_start)
    mpc_plan = mpc_arc_length_stage_plan(mpc_constraints, solver, mpc_info)
    u = np.zeros(ndof, dtype=float)
    for dof, value in constrained.items():
        u[dof] = value
    lam = 0.0
    state_current: dict[str, PlasticState2D] = dict(plastic_state or {})
    linear_solver = _repeated_direct_linear_solver_config(solver)
    step_cache_start = _perf_counter()
    riks_step_cache = build_small_deformation_step_cache(
        mesh,
        materials,
        boundary_conditions,
        interfaces=interfaces,
        structural_elements=structural_elements,
        precompute_stiffness_pattern=bool(riks_cfg.get("precompute_stiffness_pattern", riks_cfg.get("precompute_sparse_pattern", True))),
    )
    _profile_add("step_cache_build_elapsed_seconds", _perf_counter() - step_cache_start)
    if riks_step_cache.ndof == ndof:
        free = riks_step_cache.free_dofs
        fixed = riks_step_cache.fixed_dofs
    else:
        riks_step_cache = None
    sparse_pattern = riks_step_cache.stiffness_cache.pattern if riks_step_cache is not None and riks_step_cache.stiffness_cache is not None else None
    reduction_cache = riks_step_cache.reduced_matrix_cache if riks_step_cache is not None else None
    reduction_cache_events: list[Mapping[str, Any]] = []
    symbolic_cache_events: list[Mapping[str, Any]] = []
    augmented_cache: ArcLengthAugmentedMatrixCache | None = None
    augmented_cache_events: list[Mapping[str, Any]] = []
    lagrange_linear_cache: LagrangeMPCLinearCorrectionCache | None = None
    lagrange_linear_cache_events: list[Mapping[str, Any]] = []
    lagrange_correction_cache: ArcLengthLagrangeCorrectionCache | None = None
    lagrange_correction_cache_events: list[Mapping[str, Any]] = []
    plastic_state_cache = (
        build_plastic_state_array_cache(mesh, materials, state_current)
        if state_current or any(material.is_plastic for material in materials.values())
        else None
    )
    initial_stress_cache = (
        build_initial_stress_array_cache(mesh, initial_stresses, active_element_ids=riks_step_cache.active_elements if riks_step_cache is not None else None)
        if initial_stresses
        else None
    )
    combined_tangent_internal_assembly = False
    tangent_start = _perf_counter()
    tangent0 = assemble_algorithmic_tangent_stiffness(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        initial_stress_cache=initial_stress_cache,
        interfaces=interfaces,
        structural_elements=structural_elements,
        plastic_state=state_current,
        plastic_state_cache=plastic_state_cache,
        tangent_method=tangent_method,
        sparse_pattern=sparse_pattern,
    )
    _profile_add("initial_tangent_assembly_elapsed_seconds", _perf_counter() - tangent_start)
    if mpc_plan.add_penalty_to_stage_matrix:
        tangent0 = (tangent0 + Kmpc).tocsr()
    if mpc_plan.lagrange_requested:
        solve_start = _perf_counter()
        du_ref_full, lagrange_linear_cache, _info = _solve_lagrange_mpc_linear_correction_cached(
            tangent0,
            reference_load,
            constrained,
            mpc_info,
            u,
            stage_name=stage_name,
            solver=linear_solver,
            cache=lagrange_linear_cache,
        )
        lagrange_linear_cache_events.append(_info)
        solve_elapsed = max(_perf_counter() - solve_start, 0.0)
        _profile_add("initial_linear_solve_elapsed_seconds", solve_elapsed)
        _record_lagrange_profile("initial_lagrange", _info, total_elapsed=solve_elapsed)
        du_ref = du_ref_full[free]
    else:
        solve_start = _perf_counter()
        du_ref, info, reduction_cache = solve_reduced_linear_system(
            tangent0,
            reference_load,
            free,
            fixed,
            fixed_values=np.zeros(fixed.size, dtype=float),
            reduction_cache=reduction_cache,
            stage_name=stage_name,
            solver=linear_solver,
            validate_cache=sparse_pattern is None or mpc_plan.add_penalty_to_stage_matrix,
        )
        _profile_add("initial_linear_solve_elapsed_seconds", _perf_counter() - solve_start)
        cache_event = info.get("reduced_matrix_cache", {})
        if isinstance(cache_event, Mapping):
            reduction_cache_events.append(cache_event)
        symbolic_event = info.get("symbolic_cache", {})
        if isinstance(symbolic_event, Mapping):
            symbolic_cache_events.append(symbolic_event)
    arc_length = riks_cfg.get("arc_length", riks_cfg.get("ds"))
    if arc_length is None:
        arc_length_value = (lambda_max / steps) * math.sqrt(float(du_ref @ du_ref) + psi * psi)
    else:
        arc_length_value = float(arc_length)
    if arc_length_value <= 0.0:
        raise FEM2DError("Riks arc_length must be positive")

    max_cutbacks = int(riks_cfg.get("max_cutbacks", riks_cfg.get("cutbacks", 0)))
    cutback_factor = float(riks_cfg.get("cutback_factor", 0.5))
    growth = float(riks_cfg.get("growth", 1.0))
    min_arc_length = float(riks_cfg.get("min_arc_length", riks_cfg.get("min_ds", arc_length_value * 1.0e-6)))
    if max_cutbacks < 0:
        raise FEM2DError("Riks max_cutbacks must be non-negative")
    if not (0.0 < cutback_factor < 1.0):
        raise FEM2DError("Riks cutback_factor must satisfy 0 < factor < 1")
    if growth <= 0.0:
        raise FEM2DError("Riks growth must be positive")
    if min_arc_length <= 0.0:
        raise FEM2DError("Riks min_arc_length must be positive")

    path: list[dict[str, Any]] = []
    iteration_history: list[dict[str, Any]] = []
    cutback_log: list[dict[str, Any]] = []
    previous_dlambda: float | None = None
    previous_du_step: np.ndarray | None = None
    direction_flips = 0
    predictor_flips = 0
    negative_dlambda = 0
    total_cutbacks = 0
    current_arc_length = arc_length_value
    for step in range(1, steps + 1):
        step_start = _perf_counter()
        step_iteration_start = len(iteration_history)
        local_cutbacks = 0
        predictor_flip = False
        while True:
            try:
                tangent_start = _perf_counter()
                tangent = assemble_algorithmic_tangent_stiffness(
                    mesh,
                    materials,
                    u,
                    initial_stresses=initial_stresses,
                    initial_stress_cache=initial_stress_cache,
                    interfaces=interfaces,
                    structural_elements=structural_elements,
                    plastic_state=state_current,
                    plastic_state_cache=plastic_state_cache,
                    tangent_method=tangent_method,
                    sparse_pattern=sparse_pattern,
                )
                _profile_add("predictor_tangent_assembly_elapsed_seconds", _perf_counter() - tangent_start)
                if mpc_plan.add_penalty_to_stage_matrix:
                    tangent = (tangent + Kmpc).tocsr()
                if mpc_plan.lagrange_requested:
                    solve_start = _perf_counter()
                    du_load_full, lagrange_linear_cache, _info = _solve_lagrange_mpc_linear_correction_cached(
                        tangent,
                        reference_load,
                        constrained,
                        mpc_info,
                        u,
                        stage_name=stage_name,
                        solver=linear_solver,
                        cache=lagrange_linear_cache,
                    )
                    lagrange_linear_cache_events.append(_info)
                    solve_elapsed = max(_perf_counter() - solve_start, 0.0)
                    _profile_add("predictor_linear_solve_elapsed_seconds", solve_elapsed)
                    _record_lagrange_profile("predictor_lagrange", _info, total_elapsed=solve_elapsed)
                    du_load = du_load_full[free]
                else:
                    solve_start = _perf_counter()
                    du_load, info, reduction_cache = solve_reduced_linear_system(
                        tangent,
                        reference_load,
                        free,
                        fixed,
                        fixed_values=np.zeros(fixed.size, dtype=float),
                        reduction_cache=reduction_cache,
                        stage_name=stage_name,
                        solver=linear_solver,
                        validate_cache=sparse_pattern is None or mpc_plan.add_penalty_to_stage_matrix,
                    )
                    _profile_add("predictor_linear_solve_elapsed_seconds", _perf_counter() - solve_start)
                    cache_event = info.get("reduced_matrix_cache", {})
                    if isinstance(cache_event, Mapping):
                        reduction_cache_events.append(cache_event)
                    symbolic_event = info.get("symbolic_cache", {})
                    if isinstance(symbolic_event, Mapping):
                        symbolic_cache_events.append(symbolic_event)
                dlam = current_arc_length / max(math.sqrt(float(du_load @ du_load) + psi * psi), np.finfo(float).eps)
                predictor_flip = False
                if previous_du_step is not None and float(du_load @ previous_du_step) < 0.0:
                    dlam = -dlam
                    predictor_flip = True
                    predictor_flips += 1
                u_prev = u.copy()
                lam_prev = lam
                u_trial = u.copy()
                u_trial[free] += dlam * du_load
                lam_trial = lam + dlam
                converged = False
                residual_norm = math.inf
                constraint_value = math.inf
                iteration = 0
                for iteration in range(1, max_iter + 2):
                    iteration_start = _perf_counter()
                    assembly_start = _perf_counter()
                    tangent, fint = assemble_tangent_and_internal_force(
                        mesh,
                        materials,
                        u_trial,
                        initial_stresses=initial_stresses,
                        initial_stress_cache=initial_stress_cache,
                        interfaces=interfaces,
                        structural_elements=structural_elements,
                        plastic_state=state_current,
                        plastic_state_cache=plastic_state_cache,
                        tangent_method=tangent_method,
                        sparse_pattern=sparse_pattern,
                    )
                    assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
                    _profile_add("iteration_tangent_internal_assembly_elapsed_seconds", assembly_elapsed)
                    combined_tangent_internal_assembly = True
                    residual = fint - lam_trial * reference_load
                    if mpc_plan.add_penalty_to_stage_matrix:
                        residual = residual + Kmpc @ u_trial - Fmpc
                        tangent = (tangent + Kmpc).tocsr()
                    residual_free = residual[free]
                    du_step = u_trial[free] - u_prev[free]
                    dl_step = lam_trial - lam_prev
                    constraint_value = float(du_step @ du_step + (psi * dl_step) ** 2 - current_arc_length**2)
                    if mpc_plan.lagrange_requested:
                        residual_norm, mpc_constraint_norm, _mpc_multipliers = _lagrange_mpc_projected_residual(residual, constrained, mpc_info, u_trial, stage_name)
                    else:
                        residual_norm = float(np.linalg.norm(residual_free))
                        mpc_constraint_norm = 0.0
                    riks_convergence = _riks_convergence_metrics(
                        force_norm=residual_norm,
                        force_reference=float(np.linalg.norm(reference_load[free])),
                        pressure_norm=0.0,
                        pressure_reference=0.0,
                        pressure_enabled=False,
                        arc_residual=constraint_value,
                        arc_reference=current_arc_length**2,
                        mpc_norm=mpc_constraint_norm,
                        mpc_reference=max(float(np.linalg.norm(u_trial[free])), current_arc_length),
                        mpc_enabled=mpc_plan.lagrange_requested,
                        riks_cfg=riks_cfg,
                        legacy_tol=tol,
                    )
                    converged = bool(riks_convergence["converged"])
                    iteration_row: dict[str, Any] = {
                        "step": step,
                        "iteration": iteration,
                        "residual_norm": residual_norm,
                        "constraint_residual": constraint_value,
                        "converged": converged,
                        **riks_convergence,
                        "assembly_elapsed_seconds": assembly_elapsed,
                        "tangent_internal_assembly_elapsed_seconds": assembly_elapsed,
                        "tangent_assembly_elapsed_seconds": assembly_elapsed,
                        "internal_force_assembly_elapsed_seconds": 0.0,
                        "monolithic_assembly_elapsed_seconds": 0.0,
                        "reduced_matrix_elapsed_seconds": 0.0,
                        "augmented_bmat_elapsed_seconds": 0.0,
                        "lagrange_constraint_matrix_elapsed_seconds": 0.0,
                        "lagrange_constraint_filter_elapsed_seconds": 0.0,
                        "lagrange_reduced_matrix_elapsed_seconds": 0.0,
                        "lagrange_bmat_elapsed_seconds": 0.0,
                        "lagrange_linear_solve_elapsed_seconds": 0.0,
                        "lagrange_total_elapsed_seconds": 0.0,
                        "linear_solve_elapsed_seconds": 0.0,
                        "line_search_elapsed_seconds": 0.0,
                        "postprocess_elapsed_seconds": 0.0,
                    }
                    if converged:
                        iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
                        iteration_history.append(iteration_row)
                        converged = True
                        break
                    if iteration > max_iter:
                        iteration_row["final_correction_check"] = True
                        iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
                        iteration_history.append(iteration_row)
                        break
                    solve_start = _perf_counter()
                    if mpc_plan.lagrange_requested:
                        du_corr, dl_corr, _mpc_multipliers, lagrange_correction_cache, cache_event = _solve_arc_length_lagrange_correction_cached(
                            tangent,
                            reference_load,
                            residual,
                            constrained,
                            mpc_info,
                            u_trial,
                            free,
                            du_step,
                            dl_step,
                            constraint_value,
                            psi,
                            stage_name=stage_name,
                            solver=linear_solver,
                            cache=lagrange_correction_cache,
                        )
                        lagrange_correction_cache_events.append(cache_event)
                        lagrange_profile = cache_event.get("profile", {}) if isinstance(cache_event.get("profile", {}), Mapping) else cache_event
                        if isinstance(lagrange_profile, Mapping):
                            iteration_row["lagrange_constraint_matrix_elapsed_seconds"] = float(lagrange_profile.get("constraint_matrix_elapsed_seconds", 0.0) or 0.0)
                            iteration_row["lagrange_constraint_filter_elapsed_seconds"] = float(lagrange_profile.get("constraint_filter_elapsed_seconds", 0.0) or 0.0)
                            iteration_row["lagrange_reduced_matrix_elapsed_seconds"] = float(lagrange_profile.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
                            iteration_row["lagrange_bmat_elapsed_seconds"] = float(lagrange_profile.get("bmat_elapsed_seconds", 0.0) or 0.0)
                            iteration_row["lagrange_linear_solve_elapsed_seconds"] = float(lagrange_profile.get("linear_solve_elapsed_seconds", 0.0) or 0.0)
                            iteration_row["lagrange_total_elapsed_seconds"] = float(lagrange_profile.get("total_elapsed_seconds", 0.0) or 0.0)
                        _record_lagrange_profile("iteration_lagrange", cache_event)
                        u_trial[free] += du_corr
                        lam_trial += dl_corr
                    else:
                        reduced_start = _perf_counter()
                        system, reduction_cache, augmented_cache = _arc_length_augmented_system_direct_fill(
                            tangent,
                            free,
                            fixed,
                            reference_load[free],
                            du_step,
                            dl_step,
                            psi,
                            reduction_cache=reduction_cache,
                            reduction_cache_events=reduction_cache_events,
                            augmented_cache=augmented_cache,
                            augmented_cache_events=augmented_cache_events,
                            validate_reduction_cache=sparse_pattern is None or mpc_plan.add_penalty_to_stage_matrix,
                        )
                        iteration_row["reduced_matrix_elapsed_seconds"] = max(_perf_counter() - reduced_start, 0.0)
                        iteration_row["augmented_bmat_elapsed_seconds"] = iteration_row["reduced_matrix_elapsed_seconds"]
                        _profile_add("iteration_reduced_matrix_elapsed_seconds", iteration_row["reduced_matrix_elapsed_seconds"])
                        _profile_add("iteration_augmented_bmat_elapsed_seconds", iteration_row["augmented_bmat_elapsed_seconds"])
                        rhs = _fill_two_block_vector(None, -residual_free, np.asarray([-constraint_value], dtype=float))
                        correction, _corr_info = solve_linear_system(system, rhs, stage_name=stage_name, solver=linear_solver)
                        u_trial[free] += correction[:-1]
                        lam_trial += float(correction[-1])
                    for dof, value in constrained.items():
                        u_trial[dof] = value
                    iteration_row["linear_solve_elapsed_seconds"] = max(_perf_counter() - solve_start, 0.0)
                    _profile_add("iteration_linear_solve_elapsed_seconds", iteration_row["linear_solve_elapsed_seconds"])
                    iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
                    iteration_history.append(iteration_row)
                if not converged:
                    raise FEM2DError(f"arc-length step {step} did not converge, residual={residual_norm:.6e}")
                break
            except FEM2DError as exc:
                if local_cutbacks >= max_cutbacks:
                    raise FEM2DError(f"{stage_name}: Riks step {step} failed after {local_cutbacks} cutbacks: {exc}") from exc
                next_arc = current_arc_length * cutback_factor
                if next_arc < min_arc_length:
                    raise FEM2DError(f"{stage_name}: Riks arc-length fell below min_arc_length after cutback: {exc}") from exc
                total_cutbacks += 1
                local_cutbacks += 1
                cutback_log.append({"step": step, "arc_length": current_arc_length, "next_arc_length": next_arc, "error": str(exc)})
                current_arc_length = next_arc
        accepted_dlambda = float(lam_trial - lam_prev)
        direction_flip = previous_dlambda is not None and accepted_dlambda * previous_dlambda < 0.0
        if direction_flip:
            direction_flips += 1
        if accepted_dlambda < 0.0:
            negative_dlambda += 1
        previous_dlambda = accepted_dlambda
        previous_du_step = u_trial[free] - u_prev[free]
        postprocess_start = _perf_counter()
        step_postprocess_state_info: dict[str, Any] = {}
        rows, state_current = compute_element_results_and_state(
            mesh,
            materials,
            u_trial,
            initial_stresses=initial_stresses,
            plastic_state=state_current,
            postprocess_info=step_postprocess_state_info,
            plastic_state_cache=plastic_state_cache,
        )
        postprocess_elapsed = max(_perf_counter() - postprocess_start, 0.0)
        _profile_add("postprocess_elapsed_seconds", postprocess_elapsed)
        plastic_state_cache = _pop_postprocess_state_array_cache(step_postprocess_state_info)
        if plastic_state_cache is None:
            plastic_state_cache = (
                build_plastic_state_array_cache(mesh, materials, state_current)
                if state_current or any(material.is_plastic for material in materials.values())
                else None
            )
        u = u_trial
        lam = lam_trial
        max_disp = float(max((math.hypot(u[2 * i], u[2 * i + 1]) for i in range(len(mesh.node_ids))), default=0.0))
        step_iterations = iteration_history[step_iteration_start:]
        active_elements_for_ratio = [element.id for element in mesh.elements if element.active]
        plastic_ratio = (
            plastic_state_cache.plastic_ratio(active_elements_for_ratio)
            if plastic_state_cache is not None
            else sum(1 for row in rows if float(row.get("plastic", 0.0)) > 0.0) / max(len(rows), 1)
        )
        path.append(
            {
                "step": step,
                "lambda": float(lam),
                "max_displacement": max_disp,
                "iterations": iteration,
                "residual_norm": residual_norm,
                "constraint_residual": constraint_value,
                "delta_lambda": accepted_dlambda,
                "arc_length": current_arc_length,
                "cutbacks": local_cutbacks,
                "predictor_flip": predictor_flip,
                "direction_flip": direction_flip,
                "negative_dlambda": accepted_dlambda < 0.0,
                "plastic_ratio": plastic_ratio,
                "converged": converged,
                "assembly_elapsed_seconds": sum(float(row.get("assembly_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "tangent_assembly_elapsed_seconds": sum(float(row.get("tangent_assembly_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "internal_force_assembly_elapsed_seconds": sum(float(row.get("internal_force_assembly_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "monolithic_assembly_elapsed_seconds": sum(float(row.get("monolithic_assembly_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "reduced_matrix_elapsed_seconds": sum(float(row.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "linear_solve_elapsed_seconds": sum(float(row.get("linear_solve_elapsed_seconds", 0.0) or 0.0) for row in step_iterations),
                "line_search_elapsed_seconds": 0.0,
                "postprocess_elapsed_seconds": postprocess_elapsed,
                "elapsed_seconds": max(_perf_counter() - step_start, 0.0),
            }
        )
        if current_arc_length < arc_length_value:
            current_arc_length = min(arc_length_value, max(min_arc_length, current_arc_length * growth))

    lambda_snap_tol = max(100.0 * tol * max(abs(lambda_max), 1.0), 1.0e-9)
    if abs(lam - lambda_max) <= lambda_snap_tol:
        snap_delta = float(lambda_max - lam)
        lam = lambda_max
        if path:
            path[-1]["lambda"] = float(lambda_max)
            path[-1]["delta_lambda"] = float(path[-1].get("delta_lambda", 0.0) + snap_delta)

    final_force_start = _perf_counter()
    fint = assemble_internal_force(mesh, materials, u, initial_stresses=initial_stresses, interfaces=interfaces, structural_elements=structural_elements, plastic_state=state_current)
    _profile_add("final_internal_force_elapsed_seconds", _perf_counter() - final_force_start)
    reactions = fint - lam * reference_load
    if mpc_plan.add_penalty_to_stage_matrix:
        reactions = reactions + Kmpc @ u - Fmpc
    state_for_output = state_current
    final_postprocess_state_info: dict[str, Any] = {}
    final_post_start = _perf_counter()
    element_results, state_current = compute_element_results_and_state(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        plastic_state=state_current,
        postprocess_info=final_postprocess_state_info,
        plastic_state_cache=plastic_state_cache,
    )
    final_state_cache = _pop_postprocess_state_array_cache(final_postprocess_state_info)
    if final_state_cache is not None:
        plastic_state_cache = final_state_cache
    _profile_add("final_postprocess_elapsed_seconds", _perf_counter() - final_post_start)
    active_elements = [element.id for element in mesh.elements if element.active]
    result = StageResult2D(stage_name, u, reactions, element_results, constrained, active_elements, {"method": "arc_length", "tangent": tangent_method, "iterations": sum(int(p["iterations"]) for p in path), "residual_norm": path[-1]["residual_norm"], "converged": True})
    result.name = stage_name
    result.pore_pressure = pore_pressure
    result.plastic_state = state_current
    result.plastic_state_array_cache = plastic_state_cache
    result.interface_results = compute_interface_results(mesh, interfaces, u)
    result.structural_results = compute_structural_results(mesh, materials, structural_elements, u, loads=loads)
    _attach_structural_extra_dofs(result, mesh, structural_elements)
    integration_post_start = _perf_counter()
    _attach_integration_point_results(
        result,
        mesh,
        materials,
        u,
        plastic_state=state_for_output,
        initial_stresses=initial_stresses,
    )
    _profile_add("integration_point_postprocess_elapsed_seconds", _perf_counter() - integration_post_start)
    assembly_total = (
        riks_profile.get("reference_load_assembly_elapsed_seconds", 0.0)
        + riks_profile.get("pore_pressure_load_assembly_elapsed_seconds", 0.0)
        + riks_profile.get("elastic_stiffness_assembly_elapsed_seconds", 0.0)
        + riks_profile.get("mpc_penalty_assembly_elapsed_seconds", 0.0)
        + riks_profile.get("initial_tangent_assembly_elapsed_seconds", 0.0)
        + riks_profile.get("predictor_tangent_assembly_elapsed_seconds", 0.0)
        + riks_profile.get("iteration_tangent_internal_assembly_elapsed_seconds", 0.0)
        + riks_profile.get("final_internal_force_elapsed_seconds", 0.0)
    )
    linear_total = (
        riks_profile.get("initial_linear_solve_elapsed_seconds", 0.0)
        + riks_profile.get("predictor_linear_solve_elapsed_seconds", 0.0)
        + riks_profile.get("iteration_linear_solve_elapsed_seconds", 0.0)
    )
    post_total = (
        riks_profile.get("postprocess_elapsed_seconds", 0.0)
        + riks_profile.get("final_postprocess_elapsed_seconds", 0.0)
        + riks_profile.get("integration_point_postprocess_elapsed_seconds", 0.0)
    )
    result.solver_info["performance"] = {
        **{key: float(value) for key, value in riks_profile.items()},
        "assembly_elapsed_seconds": float(assembly_total),
        "stiffness_assembly_elapsed_seconds": float(
            riks_profile.get("elastic_stiffness_assembly_elapsed_seconds", 0.0)
            + riks_profile.get("initial_tangent_assembly_elapsed_seconds", 0.0)
            + riks_profile.get("predictor_tangent_assembly_elapsed_seconds", 0.0)
        ),
        "tangent_assembly_elapsed_seconds": float(
            riks_profile.get("initial_tangent_assembly_elapsed_seconds", 0.0)
            + riks_profile.get("predictor_tangent_assembly_elapsed_seconds", 0.0)
            + riks_profile.get("iteration_tangent_internal_assembly_elapsed_seconds", 0.0)
        ),
        "internal_force_assembly_elapsed_seconds": float(riks_profile.get("final_internal_force_elapsed_seconds", 0.0)),
        "nonlinear_iteration_elapsed_seconds": sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in iteration_history),
        "riks_elapsed_seconds": sum(float(row.get("elapsed_seconds", 0.0) or 0.0) for row in path),
        "linear_solve_elapsed_seconds": float(linear_total),
        "reduced_matrix_elapsed_seconds": float(riks_profile.get("iteration_reduced_matrix_elapsed_seconds", 0.0)),
        "bmat_elapsed_seconds": float(riks_profile.get("lagrange_bmat_elapsed_seconds", 0.0) + riks_profile.get("iteration_augmented_bmat_elapsed_seconds", 0.0)),
        "constraint_matrix_elapsed_seconds": float(riks_profile.get("constraint_collection_elapsed_seconds", 0.0) + riks_profile.get("lagrange_constraint_matrix_elapsed_seconds", 0.0)),
        "postprocess_elapsed_seconds": float(post_total),
        "io_report_elapsed_seconds": 0.0,
        "cache_build_elapsed_seconds": float(riks_profile.get("step_cache_build_elapsed_seconds", 0.0) + riks_profile.get("lagrange_cache_build_elapsed_seconds", 0.0)),
    }
    result.solver_info["riks"] = {
        "lambda": float(lam),
        "lambda_target": lambda_max,
        "steps": steps,
        "arc_length": arc_length_value,
        "psi": psi,
        "path": path,
        "iteration_history": iteration_history,
        "branch_tracking": "direction-continuity-cutback",
        "direction_flips": direction_flips,
        "predictor_flips": predictor_flips,
        "negative_dlambda": negative_dlambda,
        "cutbacks": total_cutbacks,
        "cutback_log": cutback_log,
        "profile": dict(result.solver_info["performance"]),
        "cache": {
            "step_cache": {"enabled": False} if riks_step_cache is None else riks_step_cache.solver_info(),
            "combined_tangent_internal_assembly": combined_tangent_internal_assembly,
            "sparse_pattern_cached": sparse_pattern is not None,
            "plastic_state_array_cache": plastic_state_array_cache_info(plastic_state_cache),
            "initial_stress_array_cache": initial_stress_array_cache_info(initial_stress_cache),
            "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
            "augmented_matrix_cache": _arc_length_augmented_cache_summary(augmented_cache_events, augmented_cache),
            "lagrange_linear_cache": _lagrange_linear_cache_summary(lagrange_linear_cache_events, lagrange_linear_cache),
            "lagrange_correction_cache": _arc_length_lagrange_cache_summary(lagrange_correction_cache_events, lagrange_correction_cache),
            "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
        },
    }
    if mpc_info["count"]:
        applied_method = "lagrange" if mpc_plan.lagrange_requested else "penalty"
        result.solver_info["mpc"] = {**mpc_info, "max_violation": mpc_violation(mesh, u, mpc_info), "applied_method": applied_method}
    if output_dir is not None:
        result.output_dir = Path(output_dir)
        stage_output_start = _perf_counter()
        write_stage_outputs(mesh, result, Path(output_dir), output_config=_stage_output_config_from_solver(solver))
        output_elapsed = max(_perf_counter() - stage_output_start, 0.0)
        _profile_add("stage_output_elapsed_seconds", output_elapsed)
        perf = result.solver_info.get("performance", {})
        if isinstance(perf, dict):
            perf["stage_output_elapsed_seconds"] = float(riks_profile.get("stage_output_elapsed_seconds", 0.0))
            perf["io_report_elapsed_seconds"] = float(riks_profile.get("stage_output_elapsed_seconds", 0.0))
            riks_info = result.solver_info.get("riks", {})
            if isinstance(riks_info, dict) and isinstance(riks_info.get("profile", {}), dict):
                riks_info["profile"]["stage_output_elapsed_seconds"] = float(riks_profile.get("stage_output_elapsed_seconds", 0.0))
                riks_info["profile"]["io_report_elapsed_seconds"] = float(riks_profile.get("stage_output_elapsed_seconds", 0.0))
    return result


def solve_linear_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compatibility wrapper for the linear-solver boundary module."""

    return _solve_linear_system_core(matrix, rhs, stage_name=stage_name, solver=solver)


def solve_reduced_linear_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    free_dofs: np.ndarray,
    fixed_dofs: np.ndarray,
    *,
    fixed_values: np.ndarray | None = None,
    reduction_cache: ReducedMatrixCache | None = None,
    stage_name: str,
    solver: Mapping[str, Any] | None = None,
    validate_cache: bool = True,
) -> tuple[np.ndarray, dict[str, Any], ReducedMatrixCache]:
    """Compatibility wrapper for cached free-DOF sparse extraction."""

    return _solve_reduced_linear_system_core(
        matrix,
        rhs,
        free_dofs,
        fixed_dofs,
        fixed_values=fixed_values,
        reduction_cache=reduction_cache,
        stage_name=stage_name,
        solver=solver,
        validate_cache=validate_cache,
    )


def _reduced_matrix_cache_summary(events: list[Mapping[str, Any]], cache: ReducedMatrixCache | None) -> dict[str, Any]:
    if not events and cache is None:
        return {"enabled": False}
    current = cache.info() if cache is not None else {}
    return {
        "enabled": True,
        "solves": len(events),
        "hits": sum(1 for event in events if bool(event.get("reused", False))),
        "builds": sum(1 for event in events if bool(event.get("built", False))),
        "validated_solves": sum(1 for event in events if bool(event.get("validated", False))),
        "trusted_pattern_solves": sum(1 for event in events if not bool(event.get("validated", False))),
        "current": current,
    }


def _arc_length_augmented_system_direct_fill(
    tangent: csr_matrix,
    free: np.ndarray,
    fixed: np.ndarray,
    reference_load_free: np.ndarray,
    du_step: np.ndarray,
    dl_step: float,
    psi: float,
    *,
    reduction_cache: ReducedMatrixCache | None,
    reduction_cache_events: list[Mapping[str, Any]],
    augmented_cache: ArcLengthAugmentedMatrixCache | None,
    augmented_cache_events: list[Mapping[str, Any]],
    validate_reduction_cache: bool = True,
) -> tuple[csr_matrix, ReducedMatrixCache, ArcLengthAugmentedMatrixCache]:
    matrix_csr = tangent.tocsr()
    free_arr = np.asarray(free, dtype=np.int64).ravel()
    fixed_arr = np.asarray(fixed, dtype=np.int64).ravel()
    reduced_reused = bool(
        reduction_cache is not None
        and reduction_cache.matches(matrix_csr, free_arr, fixed_arr, validate_structure=validate_reduction_cache)
    )
    reduced_cache = reduction_cache if reduced_reused and reduction_cache is not None else build_reduced_matrix_cache_from_csr(matrix_csr, free_arr, fixed_arr, source="arc_length_tangent_pattern")
    reduction_cache_events.append(
        {
            **reduced_cache.info(),
            "reused": reduced_reused,
            "built": not reduced_reused,
            "validated": bool(validate_reduction_cache and reduction_cache is not None),
        }
    )
    reduced_tangent = reduced_cache.extract_free_free(matrix_csr)
    augmented_reused = bool(augmented_cache is not None and augmented_cache.matches(reduced_tangent))
    augmented = augmented_cache if augmented_reused and augmented_cache is not None else ArcLengthAugmentedMatrixCache.from_reduced_matrix(reduced_tangent)
    augmented_cache_events.append(
        {
            **augmented.info(),
            "reused": augmented_reused,
            "built": not augmented_reused,
        }
    )
    system = augmented.assemble(
        reduced_tangent,
        -np.asarray(reference_load_free, dtype=float).ravel(),
        2.0 * np.asarray(du_step, dtype=float).ravel(),
        2.0 * float(psi) * float(psi) * float(dl_step),
    )
    return system, reduced_cache, augmented


def _symbolic_ordering_cache_summary(events: list[Mapping[str, Any]]) -> dict[str, Any]:
    if not events:
        return {"enabled": False}
    last = events[-1]
    return {
        "enabled": any(bool(event.get("enabled", False)) for event in events),
        "solves": len(events),
        "hits": sum(1 for event in events if str(event.get("state", "")) == "hit"),
        "misses": sum(1 for event in events if str(event.get("state", "")) == "miss"),
        "permuted_solves": sum(1 for event in events if bool(event.get("permuted", False))),
        "ordering": str(last.get("ordering", "")),
        "permc_spec": str(last.get("permc_spec", "")),
    }


def _is_recoverable_material_trial_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "return mapping failed" in text
        or "active-set" in text
        or "non-finite" in text
        or "nonfinite" in text
    )


def _internal_external_work_ratio(displacement: np.ndarray, internal_force: np.ndarray, external_force: np.ndarray) -> float:
    try:
        u = np.asarray(displacement, dtype=float)
        internal = np.asarray(internal_force, dtype=float)
        external = np.asarray(external_force, dtype=float)
        internal_work = abs(float(u @ internal))
        external_work = abs(float(u @ external))
        return internal_work / max(external_work, np.finfo(float).eps)
    except (TypeError, ValueError):
        return math.nan


def _mc_adaptive_numerical_tangent_decision(
    counters_before: tuple[int, int, int, int],
    counters_after: tuple[int, int, int, int],
    *,
    point_capacity: int,
    config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Decide whether a rapidly changing MC active set needs a full numerical tangent."""

    raw = config if isinstance(config, Mapping) else {}
    regularized_count = max(int(counters_after[0]) - int(counters_before[0]), 0)
    attempt_count = max(int(counters_after[1]) - int(counters_before[1]), 0)
    hit_count = max(int(counters_after[2]) - int(counters_before[2]), 0)
    regularized_hit_count = max(
        int(counters_after[3]) - int(counters_before[3]), 0
    )
    effective_hit_count = min(
        attempt_count, hit_count + regularized_hit_count
    )
    miss_count = max(attempt_count - effective_hit_count, 0)
    capacity = max(int(point_capacity), 1)
    regularized_fraction = float(regularized_count) / float(capacity)
    miss_fraction = (
        float(miss_count) / float(attempt_count) if attempt_count else 0.0
    )

    regularized_enabled = _srm_bool(
        raw.get("regularized_projection_invalidation_enabled", True), True
    )
    regularized_min_count = max(
        1,
        _srm_auto_int(
            raw, "regularized_projection_invalidation_min_count", 32
        ),
    )
    regularized_min_fraction = min(
        1.0,
        max(
            0.0,
            _srm_auto_float(
                raw, "regularized_projection_invalidation_fraction", 0.05
            ),
        ),
    )
    active_set_miss_enabled = _srm_bool(
        raw.get("active_set_miss_invalidation_enabled", True), True
    )
    active_set_miss_min_attempts = max(
        1,
        _srm_auto_int(raw, "active_set_miss_invalidation_min_attempts", 32),
    )
    active_set_miss_fraction = min(
        1.0,
        max(
            0.0,
            _srm_auto_float(raw, "active_set_miss_invalidation_fraction", 0.35),
        ),
    )

    switch_reason = ""
    if (
        regularized_enabled
        and regularized_count >= regularized_min_count
        and regularized_fraction >= regularized_min_fraction
    ):
        switch_reason = "regularized_projection_density"
    elif (
        active_set_miss_enabled
        and attempt_count >= active_set_miss_min_attempts
        and miss_fraction >= active_set_miss_fraction
    ):
        switch_reason = "active_set_churn"

    return {
        "switch_reason": switch_reason,
        "point_capacity": int(point_capacity),
        "regularized_projection_count": regularized_count,
        "regularized_projection_fraction": regularized_fraction,
        "regularized_projection_min_count": regularized_min_count,
        "regularized_projection_fraction_threshold": regularized_min_fraction,
        "active_set_attempt_count": attempt_count,
        "active_set_effective_hit_count": effective_hit_count,
        "active_set_miss_count": miss_count,
        "active_set_miss_fraction": miss_fraction,
        "active_set_miss_min_attempts": active_set_miss_min_attempts,
        "active_set_miss_fraction_threshold": active_set_miss_fraction,
        "switched": False,
        "reassembled": False,
    }


def solve_nonlinear_system(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    tangent: csr_matrix,
    external: np.ndarray,
    constrained: Mapping[int, float],
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: Any | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    initial_displacement: np.ndarray | None = None,
    mpc_stiffness: csr_matrix | None = None,
    mpc_load: np.ndarray | None = None,
    mpc_info: Mapping[str, Any] | None = None,
    mpc_lagrange: bool = False,
    free_dofs: np.ndarray | None = None,
    fixed_dofs: np.ndarray | None = None,
    sparse_pattern: Any | None = None,
    reduced_matrix_cache: ReducedMatrixCache | None = None,
    quad4_mc_geometry_cache: Quad4MCGeometryCache | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    settings = _newton_settings(solver)
    nonlinear_factor_cache_mode = _nonlinear_factor_cache_mode(solver)
    tangent_method = _tangent_method(solver)
    raw_solver = solver if isinstance(solver, Mapping) else {}
    raw_newton = raw_solver.get("newton", {})
    raw_active_set = (
        raw_newton.get("mohr_coulomb_active_set_update", True)
        if isinstance(raw_newton, Mapping)
        else True
    )
    raw_line_search_batch = (
        raw_newton.get("line_search_batch", {})
        if isinstance(raw_newton, Mapping)
        else {}
    )
    if isinstance(raw_line_search_batch, Mapping):
        line_search_batch_requested = _srm_bool(
            raw_line_search_batch.get("enabled", True), True
        )
        try:
            line_search_batch_chunk_size = max(
                2, int(raw_line_search_batch.get("chunk_size", 4) or 4)
            )
        except (TypeError, ValueError):
            line_search_batch_chunk_size = 4
    else:
        line_search_batch_requested = _srm_bool(raw_line_search_batch, True)
        line_search_batch_chunk_size = 4
    strict_mc_tangent = _srm_bool(
        raw_solver.get("_srm_mc_strict_tangent", False),
        False,
    )
    strict_mc_full_numerical_tangent = _srm_bool(
        raw_solver.get("_srm_mc_strict_full_numerical_tangent", False),
        False,
    )
    if isinstance(raw_active_set, Mapping):
        active_set_update_enabled = _srm_bool(
            raw_active_set.get("enabled", True),
            True,
        )
        tangent_reuse_enabled = _srm_bool(
            raw_active_set.get("tangent_reuse_enabled", True),
            True,
        )
        direct_consistent_tangent_enabled = _srm_bool(
            raw_active_set.get("direct_consistent_tangent_enabled", True),
            True,
        )
        adaptive_numerical_tangent_enabled = _srm_bool(
            raw_active_set.get("adaptive_numerical_tangent_enabled", True),
            True,
        )
        line_search_cache_invalidation_enabled = _srm_bool(
            raw_active_set.get("line_search_invalidation_enabled", True),
            True,
        )
        try:
            line_search_cache_invalidation_threshold = max(
                1,
                int(raw_active_set.get("line_search_invalidation_threshold", 4)),
            )
        except (TypeError, ValueError):
            line_search_cache_invalidation_threshold = 4
    else:
        active_set_update_enabled = _srm_bool(raw_active_set, True)
        tangent_reuse_enabled = True
        direct_consistent_tangent_enabled = True
        adaptive_numerical_tangent_enabled = True
        line_search_cache_invalidation_enabled = True
        line_search_cache_invalidation_threshold = 4
    if strict_mc_tangent:
        tangent_reuse_enabled = False
        direct_consistent_tangent_enabled = not strict_mc_full_numerical_tangent
    has_active_set_batch = any(
        bool(element.active)
        and str(element.type).upper().strip() == "QUAD4"
        and normalize_integration(element.integration) in {"SRI", "B-BAR"}
        and str(materials[element.material].model).lower().strip() == "mohr_coulomb"
        and not bool(materials[element.material].tension_cutoff)
        for element in mesh.elements
    )
    active_elements = [element for element in mesh.elements if element.active]
    line_search_batch_disabled_reason = ""
    if not line_search_batch_requested:
        line_search_batch_disabled_reason = "configuration"
    elif interfaces:
        line_search_batch_disabled_reason = "interface_elements_present"
    elif structural_elements:
        line_search_batch_disabled_reason = "structural_elements_present"
    elif not active_elements:
        line_search_batch_disabled_reason = "no_active_elements"
    elif not all(
        str(element.type).upper().strip() == "QUAD4"
        and normalize_integration(element.integration) in {"SRI", "B-BAR"}
        and str(materials[element.material].model).lower().strip()
        == "mohr_coulomb"
        and not bool(materials[element.material].tension_cutoff)
        for element in active_elements
    ):
        line_search_batch_disabled_reason = "mixed_or_unsupported_elements"
    line_search_batch_enabled = bool(
        line_search_batch_requested and not line_search_batch_disabled_reason
    )
    mohr_coulomb_active_set_cache = (
        MohrCoulombActiveSetCache(
            tangent_reuse_enabled=tangent_reuse_enabled,
            direct_consistent_tangent_enabled=direct_consistent_tangent_enabled,
            tangent_reuse_disabled_reason=(
                (
                    "srm_boundary_or_retry_strict_full_numerical_tangent"
                    if strict_mc_full_numerical_tangent
                    else "srm_boundary_or_retry_strict_unstable_points"
                )
                if strict_mc_tangent
                else (
                    "configuration"
                    if not tangent_reuse_enabled
                    else ""
                )
            ),
            strict_unstable_points_only=bool(
                strict_mc_tangent and not strict_mc_full_numerical_tangent
            ),
            geometry_cache=quad4_mc_geometry_cache,
        )
        if active_set_update_enabled and has_active_set_batch
        else None
    )
    ndof = external.size
    if plastic_state_cache is None:
        plastic_state_cache = (
            build_plastic_state_array_cache(mesh, materials, plastic_state)
            if plastic_state or any(material.is_plastic for material in materials.values())
            else None
        )
    plastic_state_cache_info = plastic_state_array_cache_info(plastic_state_cache)
    if initial_stress_cache is None and initial_stresses:
        initial_stress_cache = build_initial_stress_array_cache(mesh, initial_stresses)
    initial_stress_cache_info = initial_stress_array_cache_info(initial_stress_cache)
    if free_dofs is not None and fixed_dofs is not None:
        free = np.asarray(free_dofs, dtype=int)
        fixed = np.asarray(fixed_dofs, dtype=int)
    else:
        free, fixed = _free_index_arrays(ndof, constrained, stage_name=stage_name)
    reduction_cache = reduced_matrix_cache
    reduction_cache_events: list[Mapping[str, Any]] = []
    symbolic_cache_events: list[Mapping[str, Any]] = []
    lagrange_linear_cache: LagrangeMPCLinearCorrectionCache | None = None
    lagrange_linear_cache_events: list[Mapping[str, Any]] = []
    has_mpc = bool(mpc_info and int(mpc_info.get("count", 0)) > 0 and mpc_stiffness is not None and mpc_load is not None)
    has_lagrange_mpc = bool(has_mpc and mpc_lagrange)
    has_penalty_mpc = bool(has_mpc and not has_lagrange_mpc)
    reduction_cache_trusted = reduction_cache is not None and sparse_pattern is not None and not has_penalty_mpc
    if initial_displacement is None:
        u = np.zeros(ndof, dtype=float)
    else:
        u = np.asarray(initial_displacement, dtype=float).copy()
        if u.shape != (ndof,):
            raise FEM2DError(f"{stage_name}: initial_displacement size must match displacement dofs")
    for dof, value in constrained.items():
        u[dof] = value
    if free.size == 0:
        fint = assemble_internal_force(
            mesh,
            materials,
            u,
            initial_stresses=initial_stresses,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            initial_stress_cache=initial_stress_cache,
            mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
            quad4_mc_geometry_cache=quad4_mc_geometry_cache,
        )
        reactions = fint - external
        if has_penalty_mpc:
            reactions = reactions + mpc_stiffness @ u - mpc_load
        work_ratio = _internal_external_work_ratio(u, fint, external)
        return u, reactions, {
            "method": "newton",
            "iterations": 0,
            "residual_norm": 0.0,
            "converged": True,
            "tangent": tangent_method,
            "sparse_pattern_cached": sparse_pattern is not None,
            "constraint_dofs_cached": free_dofs is not None and fixed_dofs is not None,
            "plastic_state_array_cache": plastic_state_cache_info,
            "initial_stress_array_cache": initial_stress_cache_info,
            "mohr_coulomb_active_set_update": (
                {"enabled": False, "reason": "disabled_or_no_supported_block"}
                if mohr_coulomb_active_set_cache is None
                else mohr_coulomb_active_set_cache.solver_info()
            ),
            "combined_tangent_internal_assembly": False,
            "internal_external_work_ratio": work_ratio,
            "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
            "lagrange_linear_cache": _lagrange_linear_cache_summary(lagrange_linear_cache_events, lagrange_linear_cache),
            "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
            "nonlinear_factor_cache": _nonlinear_factor_cache_summary(
                [], mode=nonlinear_factor_cache_mode
            ),
        }

    linear_solver = _repeated_direct_linear_solver_config(solver)
    target_norm = float(np.linalg.norm(external[free]))
    residual_norm = math.inf
    constraint_norm = 0.0
    linear_method = "direct"
    line_search_reductions = 0
    line_search_material_failures = 0
    line_search_batch_call_count = 0
    line_search_batch_candidate_count = 0
    line_search_batch_fallback_count = 0
    last_multipliers: list[float] = []
    convergence_history: list[dict[str, Any]] = []
    previous_update_free = np.zeros(free.size, dtype=float)
    has_previous_update = False
    for iteration in range(1, settings["max_iter"] + 1):
        _raise_if_solver_cancel_requested(
            solver,
            stage_name,
            "newton_iteration_start",
            strength_factor=float(strength_factor),
            attempted_load_factor=1.0,
            newton_iterations_total=int(iteration - 1),
            newton_iterations_max=int(settings["max_iter"]),
            residual_norm_final="" if math.isinf(residual_norm) else float(residual_norm),
            convergence_history_tail=[dict(row) for row in convergence_history[-5:]],
        )
        iteration_start = _perf_counter()
        assembly_start = _perf_counter()
        mc_adaptive_counters_before = (
            mohr_coulomb_adaptive_tangent_counters()
            if mohr_coulomb_active_set_cache is not None
            and adaptive_numerical_tangent_enabled
            else (0, 0, 0, 0)
        )
        tangent, fint = assemble_tangent_and_internal_force(
            mesh,
            materials,
            u,
            initial_stresses=initial_stresses,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            initial_stress_cache=initial_stress_cache,
            tangent_method=tangent_method,
            sparse_pattern=sparse_pattern,
            mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
            quad4_mc_geometry_cache=quad4_mc_geometry_cache,
        )
        mc_adaptive_tangent: dict[str, Any] = {}
        if (
            mohr_coulomb_active_set_cache is not None
            and adaptive_numerical_tangent_enabled
        ):
            cache_info = mohr_coulomb_active_set_cache.solver_info()
            mc_adaptive_tangent = _mc_adaptive_numerical_tangent_decision(
                mc_adaptive_counters_before,
                mohr_coulomb_adaptive_tangent_counters(),
                point_capacity=int(
                    cache_info.get("integration_point_capacity", 0) or 0
                ),
                config=raw_active_set
                if isinstance(raw_active_set, Mapping)
                else None,
            )
            switch_reason = str(
                mc_adaptive_tangent.get("switch_reason", "") or ""
            )
            switch_available = bool(
                mohr_coulomb_active_set_cache.tangent_reuse_enabled
                or mohr_coulomb_active_set_cache.direct_consistent_tangent_enabled
            )
            if switch_reason and switch_available:
                invalidated = mohr_coulomb_active_set_cache.force_numerical_tangent(
                    switch_reason
                )
                tangent, fint = assemble_tangent_and_internal_force(
                    mesh,
                    materials,
                    u,
                    initial_stresses=initial_stresses,
                    interfaces=interfaces,
                    structural_elements=structural_elements,
                    strength_factor=strength_factor,
                    plastic_state=plastic_state,
                    plastic_state_cache=plastic_state_cache,
                    initial_stress_cache=initial_stress_cache,
                    tangent_method=tangent_method,
                    sparse_pattern=sparse_pattern,
                    mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
                    quad4_mc_geometry_cache=quad4_mc_geometry_cache,
                )
                mc_adaptive_tangent.update(
                    {
                        "switched": True,
                        "reassembled": True,
                        "invalidated_point_count": int(invalidated),
                    }
                )
        assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
        residual = external - fint
        if has_penalty_mpc:
            residual = residual + mpc_load - mpc_stiffness @ u
        residual_free = residual[free]
        if has_lagrange_mpc:
            residual_norm, constraint_norm, last_multipliers = _lagrange_mpc_projected_residual(residual, constrained, mpc_info or {}, u, stage_name)
        else:
            residual_norm = float(np.linalg.norm(residual_free))
            constraint_norm = 0.0
        convergence = _newton_convergence_metrics(
            residual_free=residual_free,
            external_free=external[free],
            displacement_free=u[free],
            previous_update_free=previous_update_free,
            has_previous_update=has_previous_update,
            constraint_norm=constraint_norm,
            settings=settings,
        )
        convergence = _newton_convergence_with_force_norm(convergence, residual_norm, settings)
        converged = bool(convergence["converged"])
        iteration_row: dict[str, Any] = {
            "iteration": iteration - 1,
            "residual_norm": residual_norm,
            "constraint_norm": constraint_norm,
            "target_norm": target_norm,
            "linear_method": linear_method,
            "converged": converged,
            "assembly_elapsed_seconds": assembly_elapsed,
            "tangent_internal_assembly_elapsed_seconds": assembly_elapsed,
            "tangent_assembly_elapsed_seconds": assembly_elapsed,
            "internal_force_assembly_elapsed_seconds": 0.0,
            "monolithic_assembly_elapsed_seconds": 0.0,
            "reduced_matrix_elapsed_seconds": 0.0,
            "linear_solve_elapsed_seconds": 0.0,
            "line_search_elapsed_seconds": 0.0,
            "postprocess_elapsed_seconds": 0.0,
            **convergence,
        }
        if mc_adaptive_tangent:
            iteration_row["mc_adaptive_numerical_tangent"] = dict(
                mc_adaptive_tangent
            )
            if bool(mc_adaptive_tangent.get("switched", False)):
                iteration_row["mc_tangent_cache_invalidated_points"] = int(
                    mc_adaptive_tangent.get("invalidated_point_count", 0) or 0
                )
                iteration_row["mc_tangent_cache_invalidation_reason"] = str(
                    mc_adaptive_tangent.get("switch_reason", "") or ""
                )
                iteration_row["mc_numerical_tangent_switch"] = True
        if converged:
            iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
            convergence_history.append(iteration_row)
            reactions = fint - external
            if has_penalty_mpc:
                reactions = reactions + mpc_stiffness @ u - mpc_load
            work_ratio = _internal_external_work_ratio(u, fint, external)
            info = {
                "method": "newton",
                "linear_method": linear_method,
                "iterations": iteration - 1,
                "residual_norm": residual_norm,
                "converged": True,
                "strength_factor": strength_factor,
                "tangent": tangent_method,
                "line_search_reductions": line_search_reductions,
                "line_search_batch": {
                    "requested": bool(line_search_batch_requested),
                    "enabled": bool(line_search_batch_enabled),
                    "disabled_reason": line_search_batch_disabled_reason,
                    "chunk_size": int(line_search_batch_chunk_size),
                    "batch_calls": int(line_search_batch_call_count),
                    "candidate_count": int(line_search_batch_candidate_count),
                    "fallback_count": int(line_search_batch_fallback_count),
                    "kernel": "quad4_sri_bbar_mc_force_candidates_numba",
                },
                "convergence_history": convergence_history,
                "sparse_pattern_cached": sparse_pattern is not None,
                "constraint_dofs_cached": free_dofs is not None and fixed_dofs is not None,
                "plastic_state_array_cache": plastic_state_cache_info,
                "initial_stress_array_cache": initial_stress_cache_info,
                "mohr_coulomb_active_set_update": (
                    {"enabled": False, "reason": "disabled_or_no_supported_block"}
                    if mohr_coulomb_active_set_cache is None
                    else mohr_coulomb_active_set_cache.solver_info()
                ),
                "combined_tangent_internal_assembly": True,
                "internal_external_work_ratio": work_ratio,
                "convergence_criteria": dict(convergence),
                "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
                "lagrange_linear_cache": _lagrange_linear_cache_summary(lagrange_linear_cache_events, lagrange_linear_cache),
                "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
                "nonlinear_factor_cache": _nonlinear_factor_cache_summary(
                    convergence_history, mode=nonlinear_factor_cache_mode
                ),
            }
            if has_lagrange_mpc:
                info["constraint_norm"] = constraint_norm
                info["multipliers"] = last_multipliers
            return u, reactions, info
        if has_penalty_mpc:
            tangent = (tangent + mpc_stiffness).tocsr()
        solve_start = _perf_counter()
        if has_lagrange_mpc:
            correction_full, lagrange_linear_cache, info = _solve_lagrange_mpc_linear_correction_cached(
                tangent,
                residual,
                constrained,
                mpc_info or {},
                u,
                stage_name=stage_name,
                solver=linear_solver,
                cache=lagrange_linear_cache,
            )
            lagrange_linear_cache_events.append(info)
            profile = info.get("profile", {}) if isinstance(info.get("profile", {}), Mapping) else info
            if isinstance(profile, Mapping):
                iteration_row["reduced_matrix_elapsed_seconds"] = float(profile.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
                iteration_row["lagrange_bmat_elapsed_seconds"] = float(profile.get("bmat_elapsed_seconds", 0.0) or 0.0)
            symbolic_event = info.get("symbolic_cache", {})
            if isinstance(symbolic_event, Mapping):
                symbolic_cache_events.append(symbolic_event)
                iteration_row["symbolic_cache_state"] = str(symbolic_event.get("state", ""))
            if "factor_cache" in info:
                iteration_row["lu_factor_cache_state"] = str(info.get("factor_cache", ""))
            iteration_row["lagrange_linear_cache_reused"] = bool(info.get("reused", False))
            iteration_row["lagrange_linear_cache_built"] = bool(info.get("built", False))
            correction = correction_full[free]
            if "multipliers" in info:
                last_multipliers = [float(v) for v in info.get("multipliers", [])]
        else:
            residual_free = residual[free]
            correction, info, reduction_cache = solve_reduced_linear_system(
                tangent,
                residual,
                free,
                fixed,
                reduction_cache=reduction_cache,
                stage_name=stage_name,
                solver=linear_solver,
                validate_cache=not reduction_cache_trusted,
            )
            cache_event = info.get("reduced_matrix_cache", {})
            if isinstance(cache_event, Mapping):
                reduction_cache_events.append(cache_event)
                iteration_row["reduced_matrix_cache_reused"] = bool(cache_event.get("reused", False))
                iteration_row["reduced_matrix_cache_built"] = bool(cache_event.get("built", False))
            iteration_row["reduced_matrix_elapsed_seconds"] = float(info.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
            symbolic_event = info.get("symbolic_cache", {})
            if isinstance(symbolic_event, Mapping):
                symbolic_cache_events.append(symbolic_event)
                iteration_row["symbolic_cache_state"] = str(symbolic_event.get("state", ""))
            if "factor_cache" in info:
                iteration_row["lu_factor_cache_state"] = str(info.get("factor_cache", ""))
            residual_free = residual[free]
        linear_elapsed = max(_perf_counter() - solve_start, 0.0)
        linear_method = str(info.get("method", linear_method))
        iteration_row["linear_method"] = linear_method
        iteration_row["linear_solve_elapsed_seconds"] = linear_elapsed
        line_search_start = _perf_counter()
        line_search_reductions_before = line_search_reductions
        line_search_material_failures_before = line_search_material_failures
        line_search_batch_calls_before = line_search_batch_call_count
        line_search_batch_candidates_before = line_search_batch_candidate_count
        applied_alpha = 1.0
        if settings["line_search"]:
            alpha = 1.0
            base_u_free = u[free].copy()
            accepted = False
            best_alpha = alpha
            best_norm = math.inf
            last_trial_error: FEM2DError | None = None
            attempt = 0
            stop_search = False
            batch_available = bool(line_search_batch_enabled)
            max_attempts = settings["max_line_search"] + 1
            while attempt < max_attempts and not accepted and not stop_search:
                candidate_alphas = [alpha]
                batch_forces: np.ndarray | None = None
                candidate_displacements: np.ndarray | None = None
                if batch_available and attempt > 0:
                    candidate_alphas = []
                    candidate_alpha = alpha
                    for _ in range(
                        min(
                            line_search_batch_chunk_size,
                            max_attempts - attempt,
                        )
                    ):
                        candidate_alphas.append(candidate_alpha)
                        if candidate_alpha <= settings["min_line_search_alpha"]:
                            break
                        candidate_alpha *= settings["line_search_factor"]
                    candidate_displacements = np.repeat(
                        u[np.newaxis, :], len(candidate_alphas), axis=0
                    )
                    for candidate_index, candidate_alpha in enumerate(
                        candidate_alphas
                    ):
                        candidate_displacements[candidate_index, free] = (
                            base_u_free + candidate_alpha * correction
                        )
                        for dof, value in constrained.items():
                            candidate_displacements[candidate_index, dof] = value
                    try:
                        batch_forces = assemble_internal_force_candidates(
                            mesh,
                            materials,
                            candidate_displacements,
                            initial_stresses=initial_stresses,
                            interfaces=interfaces,
                            structural_elements=structural_elements,
                            strength_factor=strength_factor,
                            plastic_state=plastic_state,
                            plastic_state_cache=plastic_state_cache,
                            initial_stress_cache=initial_stress_cache,
                            mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
                            quad4_mc_geometry_cache=quad4_mc_geometry_cache,
                        )
                    except FEM2DError as exc:
                        if not _is_recoverable_material_trial_error(exc):
                            raise
                        batch_forces = None
                    if batch_forces is None:
                        batch_available = False
                        line_search_batch_fallback_count += 1
                        candidate_alphas = [alpha]
                        candidate_displacements = None
                    else:
                        line_search_batch_call_count += 1
                        line_search_batch_candidate_count += len(
                            candidate_alphas
                        )

                for candidate_index, candidate_alpha in enumerate(
                    candidate_alphas
                ):
                    if candidate_displacements is not None:
                        u[:] = candidate_displacements[candidate_index]
                    else:
                        u[free] = base_u_free + candidate_alpha * correction
                        for dof, value in constrained.items():
                            u[dof] = value
                    try:
                        trial_fint = (
                            batch_forces[candidate_index]
                            if batch_forces is not None
                            else assemble_internal_force(
                                mesh,
                                materials,
                                u,
                                initial_stresses=initial_stresses,
                                interfaces=interfaces,
                                structural_elements=structural_elements,
                                strength_factor=strength_factor,
                                plastic_state=plastic_state,
                                plastic_state_cache=plastic_state_cache,
                                initial_stress_cache=initial_stress_cache,
                                mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
                                quad4_mc_geometry_cache=quad4_mc_geometry_cache,
                            )
                        )
                    except FEM2DError as exc:
                        if not _is_recoverable_material_trial_error(exc):
                            raise
                        attempt += 1
                        line_search_material_failures += 1
                        last_trial_error = exc
                        if candidate_alpha <= settings["min_line_search_alpha"]:
                            stop_search = True
                            break
                        alpha = candidate_alpha * settings["line_search_factor"]
                        line_search_reductions += 1
                        continue
                    attempt += 1
                    trial_residual = external - trial_fint
                    if has_penalty_mpc:
                        trial_residual = (
                            trial_residual + mpc_load - mpc_stiffness @ u
                        )
                    if has_lagrange_mpc:
                        trial_norm, _trial_constraint_norm, _trial_multipliers = _lagrange_mpc_projected_residual(
                            trial_residual,
                            constrained,
                            mpc_info or {},
                            u,
                            stage_name,
                        )
                    else:
                        trial_norm = float(
                            np.linalg.norm(trial_residual[free])
                        )
                    if trial_norm < best_norm:
                        best_norm = trial_norm
                        best_alpha = candidate_alpha
                    if (
                        trial_norm <= residual_norm
                        or candidate_alpha <= settings["min_line_search_alpha"]
                    ):
                        residual_norm = trial_norm
                        alpha = candidate_alpha
                        accepted = True
                        break
                    alpha = candidate_alpha * settings["line_search_factor"]
                    line_search_reductions += 1
            if not accepted:
                if not math.isfinite(best_norm) and last_trial_error is not None:
                    raise last_trial_error
                u[free] = base_u_free + best_alpha * correction
                residual_norm = best_norm
                applied_alpha = best_alpha
            else:
                applied_alpha = alpha
        else:
            u[free] += correction
        for dof, value in constrained.items():
            u[dof] = value
        iteration_row["line_search_elapsed_seconds"] = max(_perf_counter() - line_search_start, 0.0)
        iteration_row["line_search_batch_calls"] = int(
            line_search_batch_call_count - line_search_batch_calls_before
        )
        iteration_row["line_search_batch_candidates"] = int(
            line_search_batch_candidate_count
            - line_search_batch_candidates_before
        )
        iteration_line_search_reductions = (
            line_search_reductions - line_search_reductions_before
        )
        iteration_material_failures = (
            line_search_material_failures - line_search_material_failures_before
        )
        if (
            mohr_coulomb_active_set_cache is not None
            and adaptive_numerical_tangent_enabled
            and line_search_cache_invalidation_enabled
            and (
                mohr_coulomb_active_set_cache.tangent_reuse_enabled
                or mohr_coulomb_active_set_cache.direct_consistent_tangent_enabled
            )
            and (
                iteration_line_search_reductions
                >= line_search_cache_invalidation_threshold
                or iteration_material_failures > 0
            )
        ):
            reason = (
                "line_search_material_failure"
                if iteration_material_failures > 0
                else "line_search_reduction_storm"
            )
            iteration_row["mc_tangent_cache_invalidated_points"] = (
                mohr_coulomb_active_set_cache.force_numerical_tangent(reason)
            )
            iteration_row["mc_tangent_cache_invalidation_reason"] = reason
            iteration_row["mc_numerical_tangent_switch"] = True
        iteration_row["line_search_reductions_this_iteration"] = int(
            iteration_line_search_reductions
        )
        if line_search_material_failures:
            iteration_row["line_search_material_failures"] = line_search_material_failures
        previous_update_free = np.asarray(applied_alpha * correction, dtype=float)
        has_previous_update = True
        iteration_row["applied_displacement_increment_norm"] = float(np.linalg.norm(previous_update_free))
        iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
        convergence_history.append(iteration_row)

    fint = assemble_internal_force(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        interfaces=interfaces,
        structural_elements=structural_elements,
        strength_factor=strength_factor,
        plastic_state=plastic_state,
        plastic_state_cache=plastic_state_cache,
        initial_stress_cache=initial_stress_cache,
        mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
        quad4_mc_geometry_cache=quad4_mc_geometry_cache,
    )
    final_residual = external - fint
    if has_penalty_mpc:
        final_residual = final_residual + mpc_load - mpc_stiffness @ u
    if has_lagrange_mpc:
        residual_norm, constraint_norm, last_multipliers = _lagrange_mpc_projected_residual(final_residual, constrained, mpc_info or {}, u, stage_name)
    else:
        residual_norm = float(np.linalg.norm(final_residual[free]))
        constraint_norm = 0.0
    final_convergence = _newton_convergence_metrics(
        residual_free=final_residual[free],
        external_free=external[free],
        displacement_free=u[free],
        previous_update_free=previous_update_free,
        has_previous_update=has_previous_update,
        constraint_norm=constraint_norm,
        settings=settings,
    )
    final_convergence = _newton_convergence_with_force_norm(final_convergence, residual_norm, settings)
    final_converged = bool(final_convergence["converged"])
    if final_converged:
        convergence_history.append(
            {
                "iteration": int(settings["max_iter"]),
                "residual_norm": residual_norm,
                "constraint_norm": constraint_norm,
                "target_norm": target_norm,
                "linear_method": linear_method,
                "converged": True,
                "final_residual_check": True,
                "assembly_elapsed_seconds": 0.0,
                "tangent_internal_assembly_elapsed_seconds": 0.0,
                "tangent_assembly_elapsed_seconds": 0.0,
                "internal_force_assembly_elapsed_seconds": 0.0,
                "monolithic_assembly_elapsed_seconds": 0.0,
                "reduced_matrix_elapsed_seconds": 0.0,
                "linear_solve_elapsed_seconds": 0.0,
                "line_search_elapsed_seconds": 0.0,
                "postprocess_elapsed_seconds": 0.0,
                "elapsed_seconds": 0.0,
                **final_convergence,
            }
        )
    work_ratio = _internal_external_work_ratio(u, fint, external)
    if not final_converged and not bool(settings["allow_nonconvergence"]):
        diagnostics = _nonlinear_failure_diagnostics(
            status="newton_nonconvergence",
            residual_norm=residual_norm,
            settings=settings,
            strength_factor=strength_factor,
            convergence_history=convergence_history,
            u=u,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            active_elements=[str(element.id) for element in mesh.elements if element.active],
            line_search_reductions=line_search_reductions,
            internal_external_work_ratio=work_ratio,
            topology_cache=_srm_topology_diagnostics_cache(mesh),
        )
        diagnostics["final_convergence_criteria"] = dict(final_convergence)
        diagnostics["mohr_coulomb_active_set_update"] = (
            {"enabled": False, "reason": "disabled_or_no_supported_block"}
            if mohr_coulomb_active_set_cache is None
            else mohr_coulomb_active_set_cache.solver_info()
        )
        diagnostics["line_search_batch"] = {
            "requested": bool(line_search_batch_requested),
            "enabled": bool(line_search_batch_enabled),
            "disabled_reason": line_search_batch_disabled_reason,
            "chunk_size": int(line_search_batch_chunk_size),
            "batch_calls": int(line_search_batch_call_count),
            "candidate_count": int(line_search_batch_candidate_count),
            "fallback_count": int(line_search_batch_fallback_count),
        }
        raise FEM2DError(f"{stage_name}: nonlinear solve did not converge, residual={residual_norm:.6e}", diagnostics=diagnostics)
    reactions = fint - external
    if has_penalty_mpc:
        reactions = reactions + mpc_stiffness @ u - mpc_load
    info = {
        "method": "newton",
        "linear_method": linear_method,
        "iterations": settings["max_iter"],
        "residual_norm": residual_norm,
        "converged": final_converged,
        "strength_factor": strength_factor,
        "tangent": tangent_method,
        "line_search_reductions": line_search_reductions,
        "line_search_batch": {
            "requested": bool(line_search_batch_requested),
            "enabled": bool(line_search_batch_enabled),
            "disabled_reason": line_search_batch_disabled_reason,
            "chunk_size": int(line_search_batch_chunk_size),
            "batch_calls": int(line_search_batch_call_count),
            "candidate_count": int(line_search_batch_candidate_count),
            "fallback_count": int(line_search_batch_fallback_count),
            "kernel": "quad4_sri_bbar_mc_force_candidates_numba",
        },
        "convergence_history": convergence_history,
        "sparse_pattern_cached": sparse_pattern is not None,
        "constraint_dofs_cached": free_dofs is not None and fixed_dofs is not None,
        "plastic_state_array_cache": plastic_state_cache_info,
        "initial_stress_array_cache": initial_stress_cache_info,
        "mohr_coulomb_active_set_update": (
            {"enabled": False, "reason": "disabled_or_no_supported_block"}
            if mohr_coulomb_active_set_cache is None
            else mohr_coulomb_active_set_cache.solver_info()
        ),
        "combined_tangent_internal_assembly": True,
        "internal_external_work_ratio": work_ratio,
        "convergence_criteria": dict(final_convergence),
        "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
        "lagrange_linear_cache": _lagrange_linear_cache_summary(lagrange_linear_cache_events, lagrange_linear_cache),
        "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
        "nonlinear_factor_cache": _nonlinear_factor_cache_summary(
            convergence_history, mode=nonlinear_factor_cache_mode
        ),
    }
    if has_lagrange_mpc:
        info["constraint_norm"] = constraint_norm
        info["multipliers"] = last_multipliers
    return u, reactions, info


def _solve_axisymmetric_linear(
    K: csr_matrix,
    F: np.ndarray,
    constrained: Mapping[int, float],
    stage_name: str,
    solver: Mapping[str, Any] | None,
    *,
    free_dofs: np.ndarray | None = None,
    fixed_dofs: np.ndarray | None = None,
    reduction_cache: ReducedMatrixCache | None = None,
    validate_cache: bool = True,
) -> tuple[np.ndarray, dict[str, Any], ReducedMatrixCache | None]:
    ndof = F.size
    if not constrained:
        u, info = solve_linear_system(K, F, stage_name=stage_name, solver=solver)
        info["reduced_matrix_elapsed_seconds"] = 0.0
        info["reduced_matrix_cache"] = {"enabled": False}
        return u, info, None
    if free_dofs is not None and fixed_dofs is not None:
        free = np.asarray(free_dofs, dtype=np.int64).ravel()
        fixed = np.asarray(fixed_dofs, dtype=np.int64).ravel()
    else:
        free, fixed = _free_index_arrays(ndof, constrained, stage_name=stage_name)
    u = np.zeros(ndof, dtype=float)
    fixed_values = np.asarray([float(constrained[int(dof)]) for dof in fixed], dtype=float)
    if fixed.size:
        u[fixed] = fixed_values
    if free.size:
        reduced_solution, info, active_cache = solve_reduced_linear_system(
            K,
            F,
            free,
            fixed,
            fixed_values=fixed_values,
            reduction_cache=reduction_cache,
            stage_name=stage_name,
            solver=solver,
            validate_cache=validate_cache,
        )
        u[free] = reduced_solution
        return u, info, active_cache
    cache_info: dict[str, Any]
    if reduction_cache is None:
        cache_info = {"enabled": False}
    else:
        cache_info = {**reduction_cache.info(), "reused": True, "built": False, "validated": False}
    info = {
        "method": "none",
        "method_requested": "none",
        "iterations": 0,
        "residual_norm": 0.0,
        "equilibrated": False,
        "reduced_matrix_elapsed_seconds": 0.0,
        "reduced_matrix_cache": cache_info,
    }
    return u, info, reduction_cache


def compute_axisymmetric_element_results(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
) -> list[dict[str, Any]]:
    rows, _state = compute_axisymmetric_element_results_and_state(mesh, materials, u, strength_factor=strength_factor, plastic_state=plastic_state, initial_stresses=initial_stresses)
    return rows


def compute_axisymmetric_element_results_and_state(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    collect_results: bool = True,
    postprocess_info: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, PlasticState2D]]:
    node_index = mesh.node_index
    rows: list[dict[str, Any]] = []
    updated_state: dict[str, PlasticState2D] = dict(plastic_state or {})
    active_element_count = 0
    inactive_element_count = 0
    committed_points = 0
    row_count = 0
    for element in mesh.elements:
        if not element.active:
            inactive_element_count += 1
            if collect_results:
                rows.append(_inactive_axisymmetric_result(element))
                row_count += 1
            continue
        active_element_count += 1
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        ue = u[_dofs_from_node_indices(conn)]
        initial = np.asarray(initial_stresses.get(element.id, np.zeros(4, dtype=float)), dtype=float) if initial_stresses else np.zeros(4, dtype=float)
        strains: list[np.ndarray] = []
        stresses: list[np.ndarray] = []
        weights: list[float] = []
        plastic_flags: list[bool] = []
        yield_values: list[float] = []
        active_sets: list[str] = []
        material_states: list[dict[str, Any]] = []
        points = integration_points(element.type, "FULL")
        mode = normalize_integration(element.integration)
        if mode == "B-BAR":
            Pvol = material.volumetric_projector
            Pdev = np.eye(4) - Pvol
            volume = 0.0
            epsv_acc = np.zeros(4, dtype=float)
            cached: list[tuple[int, tuple[float, float, float], np.ndarray, float]] = []
            for gp_index, gp in enumerate(points):
                B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                eps = B4 @ ue
                volume += dV
                epsv_acc += (Pvol @ eps) * dV
                cached.append((gp_index, gp, eps, dV))
            epsv_bar = epsv_acc / max(volume, np.finfo(float).eps)
            for gp_index, _gp, eps, dV in cached:
                eps_eff = Pdev @ eps + epsv_bar
                state_key = _plastic_state_key(element.id, gp_index)
                old_state = _plastic_state_for_gp(plastic_state, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps_eff, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), float(update.kappa), dict(update.state_vars))
                committed_points += 1
                if collect_results:
                    strains.append(eps_eff)
                    stresses.append(update.stress)
                    weights.append(dV)
                    plastic_flags.append(bool(update.plastic))
                    yield_values.append(float(update.yield_value))
                    active_sets.append(",".join(str(value) for value in update.active_set) if update.active_set else "elastic")
                    material_states.append(_material_state_output(material, update))
        elif mode == "SRI":
            for gp_index, gp in enumerate(points):
                B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                eps = B4 @ ue
                state_key = _plastic_state_key(element.id, gp_index)
                old_state = _plastic_state_for_gp(plastic_state, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), float(update.kappa), dict(update.state_vars))
                committed_points += 1
                if collect_results:
                    strains.append(eps)
                    stresses.append(update.stress)
                    weights.append(dV)
                    plastic_flags.append(bool(update.plastic))
                    yield_values.append(float(update.yield_value))
                    active_sets.append(",".join(str(value) for value in update.active_set) if update.active_set else "elastic")
                    material_states.append(_material_state_output(material, update))
            offset = len(points)
            for red_index, gp in enumerate(integration_points(element.type, "REDUCED")):
                gp_index = offset + red_index
                B4, _detJ, _N, _radius = axisymmetric_strain_displacement_matrix(element.type, coords, gp)
                eps = B4 @ ue
                state_key = _plastic_state_key(element.id, gp_index)
                old_state = _plastic_state_for_gp(plastic_state, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), float(update.kappa), dict(update.state_vars))
                committed_points += 1
        else:
            for gp_index, gp in enumerate(points):
                B4, detJ, _N, radius = axisymmetric_strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                eps = B4 @ ue
                state_key = _plastic_state_key(element.id, gp_index)
                old_state = _plastic_state_for_gp(plastic_state, element.id, gp_index)
                update = update_plane_strain_stress(
                    material, eps, state=old_state, initial_stress=initial, strength_factor=strength_factor,
                    diagnostic_context=(element.id, gp_index),
                )
                updated_state[state_key] = PlasticState2D(update.plastic_strain.copy(), float(update.kappa), dict(update.state_vars))
                committed_points += 1
                if collect_results:
                    strains.append(eps)
                    stresses.append(update.stress)
                    weights.append(dV)
                    plastic_flags.append(bool(update.plastic))
                    yield_values.append(float(update.yield_value))
                    active_sets.append(",".join(str(value) for value in update.active_set) if update.active_set else "elastic")
                    material_states.append(_material_state_output(material, update))
        if not collect_results:
            continue
        w = np.asarray(weights, dtype=float)
        eps_avg = np.average(np.vstack(strains), axis=0, weights=w)
        sig_avg = np.average(np.vstack(stresses), axis=0, weights=w)
        principal = np.sort(sig_avg[:3])[::-1]
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
                "plastic": 1.0 if any(plastic_flags) else 0.0,
                "yield_value": float(max(yield_values, default=0.0)),
                "p": float(np.mean(sig_avg[:3])),
                "q": float(math.sqrt(max(((sig_avg[0] - sig_avg[1]) ** 2 + (sig_avg[1] - sig_avg[2]) ** 2 + (sig_avg[2] - sig_avg[0]) ** 2) / 2.0 + 3.0 * sig_avg[3] ** 2, 0.0))),
                "active_set": "axisymmetric:" + "|".join(active_sets),
            }
        row.update(_average_material_state_outputs(material_states, w, material))
        rows.append(row)
        row_count += 1
    if postprocess_info is not None:
        postprocess_info.update(
            {
                "geometry": "axisymmetric",
                "array_postprocess_enabled": False,
                "state_commit": "element_loop",
                "dict_materialized": True,
                "collect_results": bool(collect_results),
                "row_generation": "element_loop" if collect_results else "skipped",
                "integration_point_second_pass_skipped": False,
                "active_elements": int(active_element_count),
                "inactive_elements": int(inactive_element_count),
                "committed_points": int(committed_points),
                "loop_committed_points": int(committed_points),
                "result_rows": int(row_count),
                "integration_point_rows": 0,
                "plastic_state_entries": int(len(updated_state)),
            }
        )
    return rows, updated_state


def compute_axisymmetric_integration_point_results(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
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
                _B4, _detJ, N, radius = axisymmetric_strain_displacement_matrix(element.type, coords, gp)
                xy = N @ coords
                rows.append(_inactive_axisymmetric_integration_point_result(element, gp_index, gp, float(xy[0]), float(xy[1]), radius))
            continue
        ue = u[_dofs_from_node_indices(conn)]
        initial = np.asarray(initial_stresses.get(element.id, np.zeros(4, dtype=float)), dtype=float) if initial_stresses else np.zeros(4, dtype=float)
        if initial.shape != (4,):
            raise FEM2DError(f"element {element.id}: initial stress must have 4 components")
        mode = normalize_integration(element.integration)
        if element.type.upper() == "QUAD8" and _axisymmetric_quad8_j2dp_fast_path(material, plastic_state, element_id=element.id, integration=mode):
            plastic_strains, kappas = _axisymmetric_quad8_post_state_arrays(element.id, plastic_state)
            alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
            if mode == "B-BAR":
                data = _quad8_axisymmetric_j2dp_bbar_post_fast(
                    coords,
                    ue,
                    material,
                    initial_stress=initial,
                    plastic_strains=plastic_strains,
                    kappas=kappas,
                    alpha=alpha,
                    cohesion_term=cohesion_term,
                )
            else:
                data = _quad8_axisymmetric_j2dp_post_fast(
                    coords,
                    ue,
                    material,
                    initial_stress=initial,
                    plastic_strains=plastic_strains,
                    kappas=kappas,
                    alpha=alpha,
                    cohesion_term=cohesion_term,
                )
            rows.extend(_axisymmetric_quad8_j2dp_post_result_rows(element, material, data))
            continue
        if mode == "B-BAR":
            Pvol = material.volumetric_projector
            Pdev = np.eye(4) - Pvol
            volume = 0.0
            epsv_acc = np.zeros(4, dtype=float)
            cached: list[tuple[int, tuple[float, float, float], np.ndarray, float, np.ndarray, float]] = []
            for gp_index, gp in enumerate(points):
                B4, detJ, N, radius = axisymmetric_strain_displacement_matrix(element.type, coords, gp)
                dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
                eps = B4 @ ue
                volume += dV
                epsv_acc += (Pvol @ eps) * dV
                cached.append((gp_index, gp, eps, dV, N, radius))
            epsv_bar = epsv_acc / max(volume, np.finfo(float).eps)
            for gp_index, gp, eps, dV, N, radius in cached:
                eps_eff = Pdev @ eps + epsv_bar
                xy = N @ coords
                rows.append(
                    _axisymmetric_integration_point_result_row(
                        element,
                        material,
                        gp_index,
                        gp,
                        float(xy[0]),
                        float(xy[1]),
                        radius,
                        dV,
                        eps_eff,
                        initial,
                        strength_factor,
                        plastic_state,
                    )
                )
            continue
        for gp_index, gp in enumerate(points):
            B4, detJ, N, radius = axisymmetric_strain_displacement_matrix(element.type, coords, gp)
            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
            eps = B4 @ ue
            xy = N @ coords
            rows.append(
                _axisymmetric_integration_point_result_row(
                    element,
                    material,
                    gp_index,
                    gp,
                    float(xy[0]),
                    float(xy[1]),
                    radius,
                    dV,
                    eps,
                    initial,
                    strength_factor,
                    plastic_state,
                )
            )
    return rows


def _axisymmetric_integration_point_result_row(
    element: Any,
    material: ElasticPlaneStrainMaterial,
    gp_index: int,
    gp: tuple[float, float, float],
    x: float,
    y: float,
    radius: float,
    dV: float,
    strain: np.ndarray,
    initial_stress: np.ndarray,
    strength_factor: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> dict[str, Any]:
    old_state = _plastic_state_for_gp(plastic_state, element.id, gp_index)
    update = update_plane_strain_stress(material, strain, state=old_state, initial_stress=initial_stress, strength_factor=strength_factor)
    principal = np.sort(update.stress[:3])[::-1]
    plastic_strain = np.asarray(update.plastic_strain, dtype=float)
    row = {
        "element_id": element.id,
        "ip": gp_index + 1,
        "state_key": _plastic_state_key(element.id, gp_index),
        "xi": float(gp[0]),
        "eta": float(gp[1]),
        "weight": float(gp[2]),
        "x": x,
        "y": y,
        "radius": float(radius),
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
        "active_set": "axisymmetric:" + ("|".join(str(value) for value in update.active_set) if update.active_set else "elastic"),
        "kappa": float(update.kappa),
        "plastic_strain_x": float(plastic_strain[0]),
        "plastic_strain_y": float(plastic_strain[1]),
        "plastic_strain_z": float(plastic_strain[2]),
        "plastic_strain_gamma_xy": float(plastic_strain[3]),
    }
    row.update(_material_state_output(material, update))
    return row


def _axisymmetric_quad8_j2dp_post_result_rows(
    element: Any,
    material: ElasticPlaneStrainMaterial,
    data: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    integration = normalize_integration(element.integration)
    material_state = _default_material_state_output(material)
    for gp_index, row_data in enumerate(np.asarray(data, dtype=float)):
        row = {
            "element_id": element.id,
            "ip": gp_index + 1,
            "state_key": _plastic_state_key(element.id, gp_index),
            "xi": float(row_data[0]),
            "eta": float(row_data[1]),
            "weight": float(row_data[2]),
            "x": float(row_data[3]),
            "y": float(row_data[4]),
            "radius": float(row_data[27]),
            "dV": float(row_data[5]),
            "active": 1.0,
            "type": element.type,
            "material": element.material,
            "integration": integration,
            "eps_x": float(row_data[6]),
            "eps_y": float(row_data[7]),
            "eps_z": float(row_data[8]),
            "gamma_xy": float(row_data[9]),
            "sigma_x": float(row_data[10]),
            "sigma_y": float(row_data[11]),
            "sigma_z": float(row_data[12]),
            "tau_xy": float(row_data[13]),
            "sigma_1": float(row_data[14]),
            "sigma_2": float(row_data[15]),
            "sigma_3": float(row_data[16]),
            "tau_max": float(row_data[17]),
            "plastic": float(row_data[20]),
            "yield_value": float(row_data[21]),
            "p": float(row_data[18]),
            "q": float(row_data[19]),
            "active_set": "axisymmetric:" + ("plastic" if float(row_data[20]) else "elastic"),
            "kappa": float(row_data[22]),
            "plastic_strain_x": float(row_data[23]),
            "plastic_strain_y": float(row_data[24]),
            "plastic_strain_z": float(row_data[25]),
            "plastic_strain_gamma_xy": float(row_data[26]),
        }
        row.update(material_state)
        rows.append(row)
    return rows


def _inactive_axisymmetric_integration_point_result(
    element: Any,
    gp_index: int,
    gp: tuple[float, float, float],
    x: float,
    y: float,
    radius: float,
) -> dict[str, Any]:
    row = {
        "element_id": element.id,
        "ip": gp_index + 1,
        "state_key": _plastic_state_key(element.id, gp_index),
        "xi": float(gp[0]),
        "eta": float(gp[1]),
        "weight": float(gp[2]),
        "x": x,
        "y": y,
        "radius": float(radius),
        "dV": 0.0,
        "active": 0.0,
        "type": element.type,
        "material": element.material,
        "integration": normalize_integration(element.integration),
        "eps_x": 0.0,
        "eps_y": 0.0,
        "eps_z": 0.0,
        "gamma_xy": 0.0,
        "sigma_x": 0.0,
        "sigma_y": 0.0,
        "sigma_z": 0.0,
        "tau_xy": 0.0,
        "sigma_1": 0.0,
        "sigma_2": 0.0,
        "sigma_3": 0.0,
        "tau_max": 0.0,
        "plastic": 0.0,
        "yield_value": 0.0,
        "p": 0.0,
        "q": 0.0,
        "active_set": "axisymmetric",
        "kappa": 0.0,
        "plastic_strain_x": 0.0,
        "plastic_strain_y": 0.0,
        "plastic_strain_z": 0.0,
        "plastic_strain_gamma_xy": 0.0,
    }
    row.update(_inactive_material_state_output(element.material))
    return row


def _inactive_axisymmetric_result(element: Any) -> dict[str, Any]:
    row = _inactive_element_result(element)
    row["active_set"] = "axisymmetric"
    return row


def solve_axisymmetric_nonlinear_system(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    external: np.ndarray,
    constrained: Mapping[int, float],
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    mpc_stiffness: csr_matrix | None = None,
    mpc_load: np.ndarray | None = None,
    mpc_info: Mapping[str, Any] | None = None,
    mpc_lagrange: bool = False,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    initial_stress_cache: Any | None = None,
    free_dofs: np.ndarray | None = None,
    fixed_dofs: np.ndarray | None = None,
    sparse_pattern: Any | None = None,
    reduced_matrix_cache: ReducedMatrixCache | None = None,
    initial_displacement: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    settings = _newton_settings(solver)
    tangent_method = _tangent_method(solver)
    ndof = external.size
    if free_dofs is not None and fixed_dofs is not None:
        free = np.asarray(free_dofs, dtype=int)
        fixed = np.asarray(fixed_dofs, dtype=int)
    else:
        free, fixed = _free_index_arrays(ndof, constrained, stage_name=stage_name)
    if plastic_state_cache is None:
        plastic_state_cache = (
            build_plastic_state_array_cache(mesh, materials, plastic_state)
            if plastic_state or any(material.is_plastic for material in materials.values())
            else None
        )
    plastic_state_cache_info = plastic_state_array_cache_info(plastic_state_cache)
    if initial_stress_cache is None and initial_stresses:
        initial_stress_cache = build_initial_stress_array_cache(mesh, initial_stresses)
    initial_stress_cache_info = initial_stress_array_cache_info(initial_stress_cache)
    reduction_cache = reduced_matrix_cache
    reduction_cache_events: list[Mapping[str, Any]] = []
    symbolic_cache_events: list[Mapping[str, Any]] = []
    lagrange_linear_cache: LagrangeMPCLinearCorrectionCache | None = None
    lagrange_linear_cache_events: list[Mapping[str, Any]] = []
    has_mpc = bool(mpc_info and int(mpc_info.get("count", 0)) > 0 and mpc_stiffness is not None and mpc_load is not None)
    has_lagrange_mpc = bool(has_mpc and mpc_lagrange)
    has_penalty_mpc = bool(has_mpc and not has_lagrange_mpc)
    reduction_cache_trusted = reduction_cache is not None and sparse_pattern is not None and not has_penalty_mpc
    if initial_displacement is None:
        u = np.zeros(ndof, dtype=float)
    else:
        u = np.asarray(initial_displacement, dtype=float).copy()
        if u.shape != (ndof,):
            raise FEM2DError(f"{stage_name}: initial_displacement size must match displacement dofs")
    for dof, value in constrained.items():
        u[dof] = value
    if free.size == 0:
        fint = assemble_axisymmetric_internal_force(
            mesh,
            materials,
            u,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            interfaces=interfaces,
            structural_elements=structural_elements,
        )
        reactions = fint - external
        if has_penalty_mpc:
            reactions = reactions + mpc_stiffness @ u - mpc_load
        work_ratio = _internal_external_work_ratio(u, fint, external)
        return u, reactions, {
            "method": "axisymmetric_newton",
            "iterations": 0,
            "residual_norm": 0.0,
            "converged": True,
            "strength_factor": strength_factor,
            "tangent": tangent_method,
            "plastic_state_array_cache": plastic_state_cache_info,
            "initial_stress_array_cache": initial_stress_cache_info,
            "combined_tangent_internal_assembly": False,
            "internal_external_work_ratio": work_ratio,
            "sparse_pattern_cached": sparse_pattern is not None,
            "constraint_dofs_cached": free_dofs is not None and fixed_dofs is not None,
            "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
            "lagrange_linear_cache": _lagrange_linear_cache_summary(lagrange_linear_cache_events, lagrange_linear_cache),
            "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
        }

    linear_solver = _repeated_direct_linear_solver_config(solver)
    target_norm = float(np.linalg.norm(external[free]))
    residual_norm = math.inf
    constraint_norm = 0.0
    linear_method = "direct"
    line_search_reductions = 0
    last_multipliers: list[float] = []
    convergence_history: list[dict[str, Any]] = []
    previous_update_free = np.zeros(free.size, dtype=float)
    has_previous_update = False
    for iteration in range(1, settings["max_iter"] + 1):
        _raise_if_solver_cancel_requested(
            solver,
            stage_name,
            "axisymmetric_newton_iteration_start",
            strength_factor=float(strength_factor),
            attempted_load_factor=1.0,
            newton_iterations_total=int(iteration - 1),
            newton_iterations_max=int(settings["max_iter"]),
            residual_norm_final="" if math.isinf(residual_norm) else float(residual_norm),
            convergence_history_tail=[dict(row) for row in convergence_history[-5:]],
        )
        iteration_start = _perf_counter()
        assembly_start = _perf_counter()
        tangent, fint = assemble_axisymmetric_tangent_and_internal_force(
            mesh,
            materials,
            u,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            interfaces=interfaces,
            structural_elements=structural_elements,
            tangent_method=tangent_method,
            sparse_pattern=sparse_pattern,
        )
        assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
        residual = external - fint
        if has_penalty_mpc:
            residual = residual + mpc_load - mpc_stiffness @ u
        residual_free = residual[free]
        if has_lagrange_mpc:
            residual_norm, constraint_norm, last_multipliers = _lagrange_mpc_projected_residual(residual, constrained, mpc_info or {}, u, stage_name)
        else:
            residual_norm = float(np.linalg.norm(residual_free))
            constraint_norm = 0.0
        convergence = _newton_convergence_metrics(
            residual_free=residual_free,
            external_free=external[free],
            displacement_free=u[free],
            previous_update_free=previous_update_free,
            has_previous_update=has_previous_update,
            constraint_norm=constraint_norm,
            settings=settings,
        )
        convergence = _newton_convergence_with_force_norm(convergence, residual_norm, settings)
        converged = bool(convergence["converged"])
        iteration_row: dict[str, Any] = {
            "iteration": iteration - 1,
            "residual_norm": residual_norm,
            "constraint_norm": constraint_norm,
            "target_norm": target_norm,
            "linear_method": linear_method,
            "converged": converged,
            "assembly_elapsed_seconds": assembly_elapsed,
            "tangent_internal_assembly_elapsed_seconds": assembly_elapsed,
            "tangent_assembly_elapsed_seconds": assembly_elapsed,
            "internal_force_assembly_elapsed_seconds": 0.0,
            "monolithic_assembly_elapsed_seconds": 0.0,
            "reduced_matrix_elapsed_seconds": 0.0,
            "linear_solve_elapsed_seconds": 0.0,
            "line_search_elapsed_seconds": 0.0,
            "postprocess_elapsed_seconds": 0.0,
            **convergence,
        }
        if converged:
            iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
            convergence_history.append(iteration_row)
            reactions = fint - external
            if has_penalty_mpc:
                reactions = reactions + mpc_stiffness @ u - mpc_load
            work_ratio = _internal_external_work_ratio(u, fint, external)
            info = {
                "method": "axisymmetric_newton",
                "linear_method": linear_method,
                "iterations": iteration - 1,
                "residual_norm": residual_norm,
                "converged": True,
                "strength_factor": strength_factor,
                "tangent": tangent_method,
                "line_search_reductions": line_search_reductions,
                "convergence_history": convergence_history,
                "plastic_state_array_cache": plastic_state_cache_info,
                "initial_stress_array_cache": initial_stress_cache_info,
                "combined_tangent_internal_assembly": True,
                "internal_external_work_ratio": work_ratio,
                "convergence_criteria": dict(convergence),
                "sparse_pattern_cached": sparse_pattern is not None,
                "constraint_dofs_cached": free_dofs is not None and fixed_dofs is not None,
                "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
                "lagrange_linear_cache": _lagrange_linear_cache_summary(lagrange_linear_cache_events, lagrange_linear_cache),
                "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
            }
            if has_lagrange_mpc:
                info["constraint_norm"] = constraint_norm
                info["multipliers"] = last_multipliers
            return u, reactions, info
        if has_penalty_mpc:
            tangent = (tangent + mpc_stiffness).tocsr()
        solve_start = _perf_counter()
        if has_lagrange_mpc:
            correction_full, lagrange_linear_cache, info = _solve_lagrange_mpc_linear_correction_cached(
                tangent,
                residual,
                constrained,
                mpc_info or {},
                u,
                stage_name=stage_name,
                solver=linear_solver,
                cache=lagrange_linear_cache,
            )
            lagrange_linear_cache_events.append(info)
            profile = info.get("profile", {}) if isinstance(info.get("profile", {}), Mapping) else info
            if isinstance(profile, Mapping):
                iteration_row["reduced_matrix_elapsed_seconds"] = float(profile.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
                iteration_row["lagrange_bmat_elapsed_seconds"] = float(profile.get("bmat_elapsed_seconds", 0.0) or 0.0)
            symbolic_event = info.get("symbolic_cache", {})
            if isinstance(symbolic_event, Mapping):
                symbolic_cache_events.append(symbolic_event)
                iteration_row["symbolic_cache_state"] = str(symbolic_event.get("state", ""))
            if "factor_cache" in info:
                iteration_row["lu_factor_cache_state"] = str(info.get("factor_cache", ""))
            iteration_row["lagrange_linear_cache_reused"] = bool(info.get("reused", False))
            iteration_row["lagrange_linear_cache_built"] = bool(info.get("built", False))
            correction = correction_full[free]
            if "multipliers" in info:
                last_multipliers = [float(v) for v in info.get("multipliers", [])]
        else:
            correction, info, reduction_cache = solve_reduced_linear_system(
                tangent,
                residual,
                free,
                fixed,
                reduction_cache=reduction_cache,
                stage_name=stage_name,
                solver=linear_solver,
                validate_cache=not reduction_cache_trusted,
            )
            cache_event = info.get("reduced_matrix_cache", {})
            if isinstance(cache_event, Mapping):
                reduction_cache_events.append(cache_event)
                iteration_row["reduced_matrix_cache_reused"] = bool(cache_event.get("reused", False))
                iteration_row["reduced_matrix_cache_built"] = bool(cache_event.get("built", False))
            iteration_row["reduced_matrix_elapsed_seconds"] = float(info.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
            symbolic_event = info.get("symbolic_cache", {})
            if isinstance(symbolic_event, Mapping):
                symbolic_cache_events.append(symbolic_event)
                iteration_row["symbolic_cache_state"] = str(symbolic_event.get("state", ""))
            if "factor_cache" in info:
                iteration_row["lu_factor_cache_state"] = str(info.get("factor_cache", ""))
        linear_elapsed = max(_perf_counter() - solve_start, 0.0)
        linear_method = str(info.get("method", linear_method))
        iteration_row["linear_method"] = linear_method
        iteration_row["linear_solve_elapsed_seconds"] = linear_elapsed
        line_search_start = _perf_counter()
        applied_alpha = 1.0
        if settings["line_search"]:
            alpha = 1.0
            base_u_free = u[free].copy()
            accepted = False
            best_alpha = alpha
            best_norm = math.inf
            for _ls in range(settings["max_line_search"] + 1):
                u[free] = base_u_free + alpha * correction
                for dof, value in constrained.items():
                    u[dof] = value
                trial_fint = assemble_axisymmetric_internal_force(
                    mesh,
                    materials,
                    u,
                    strength_factor=strength_factor,
                    plastic_state=plastic_state,
                    plastic_state_cache=plastic_state_cache,
                    initial_stresses=initial_stresses,
                    initial_stress_cache=initial_stress_cache,
                    interfaces=interfaces,
                    structural_elements=structural_elements,
                )
                trial_residual = external - trial_fint
                if has_penalty_mpc:
                    trial_residual = trial_residual + mpc_load - mpc_stiffness @ u
                if has_lagrange_mpc:
                    trial_norm, _trial_constraint_norm, _trial_multipliers = _lagrange_mpc_projected_residual(trial_residual, constrained, mpc_info or {}, u, stage_name)
                else:
                    trial_norm = float(np.linalg.norm(trial_residual[free]))
                if trial_norm < best_norm:
                    best_norm = trial_norm
                    best_alpha = alpha
                if trial_norm <= residual_norm or alpha <= settings["min_line_search_alpha"]:
                    residual_norm = trial_norm
                    accepted = True
                    break
                alpha *= settings["line_search_factor"]
                line_search_reductions += 1
            if not accepted:
                u[free] = base_u_free + best_alpha * correction
                residual_norm = best_norm
                applied_alpha = best_alpha
            else:
                applied_alpha = alpha
        else:
            u[free] += correction
        for dof, value in constrained.items():
            u[dof] = value
        iteration_row["line_search_elapsed_seconds"] = max(_perf_counter() - line_search_start, 0.0)
        previous_update_free = np.asarray(applied_alpha * correction, dtype=float)
        has_previous_update = True
        iteration_row["applied_displacement_increment_norm"] = float(np.linalg.norm(previous_update_free))
        iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
        convergence_history.append(iteration_row)

    fint = assemble_axisymmetric_internal_force(
        mesh,
        materials,
        u,
        strength_factor=strength_factor,
        plastic_state=plastic_state,
        plastic_state_cache=plastic_state_cache,
        initial_stresses=initial_stresses,
        initial_stress_cache=initial_stress_cache,
        interfaces=interfaces,
        structural_elements=structural_elements,
    )
    final_residual = external - fint
    if has_penalty_mpc:
        final_residual = final_residual + mpc_load - mpc_stiffness @ u
    if has_lagrange_mpc:
        residual_norm, constraint_norm, last_multipliers = _lagrange_mpc_projected_residual(final_residual, constrained, mpc_info or {}, u, stage_name)
    else:
        residual_norm = float(np.linalg.norm(final_residual[free]))
        constraint_norm = 0.0
    final_convergence = _newton_convergence_metrics(
        residual_free=final_residual[free],
        external_free=external[free],
        displacement_free=u[free],
        previous_update_free=previous_update_free,
        has_previous_update=has_previous_update,
        constraint_norm=constraint_norm,
        settings=settings,
    )
    final_convergence = _newton_convergence_with_force_norm(final_convergence, residual_norm, settings)
    final_converged = bool(final_convergence["converged"])
    if final_converged:
        convergence_history.append(
            {
                "iteration": int(settings["max_iter"]),
                "residual_norm": residual_norm,
                "constraint_norm": constraint_norm,
                "target_norm": target_norm,
                "linear_method": linear_method,
                "converged": True,
                "final_residual_check": True,
                "assembly_elapsed_seconds": 0.0,
                "tangent_internal_assembly_elapsed_seconds": 0.0,
                "tangent_assembly_elapsed_seconds": 0.0,
                "internal_force_assembly_elapsed_seconds": 0.0,
                "monolithic_assembly_elapsed_seconds": 0.0,
                "reduced_matrix_elapsed_seconds": 0.0,
                "linear_solve_elapsed_seconds": 0.0,
                "line_search_elapsed_seconds": 0.0,
                "postprocess_elapsed_seconds": 0.0,
                "elapsed_seconds": 0.0,
                **final_convergence,
            }
        )
    work_ratio = _internal_external_work_ratio(u, fint, external)
    if not final_converged and not bool(settings["allow_nonconvergence"]):
        diagnostics = _nonlinear_failure_diagnostics(
            status="axisymmetric_newton_nonconvergence",
            residual_norm=residual_norm,
            settings=settings,
            strength_factor=strength_factor,
            convergence_history=convergence_history,
            u=u,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            active_elements=[str(element.id) for element in mesh.elements if element.active],
            line_search_reductions=line_search_reductions,
            internal_external_work_ratio=work_ratio,
            topology_cache=_srm_topology_diagnostics_cache(mesh),
        )
        diagnostics["final_convergence_criteria"] = dict(final_convergence)
        raise FEM2DError(f"{stage_name}: axisymmetric nonlinear solve did not converge, residual={residual_norm:.6e}", diagnostics=diagnostics)
    reactions = fint - external
    if has_penalty_mpc:
        reactions = reactions + mpc_stiffness @ u - mpc_load
    info = {
        "method": "axisymmetric_newton",
        "linear_method": linear_method,
        "iterations": settings["max_iter"],
        "residual_norm": residual_norm,
        "converged": final_converged,
        "strength_factor": strength_factor,
        "tangent": tangent_method,
        "line_search_reductions": line_search_reductions,
        "convergence_history": convergence_history,
        "plastic_state_array_cache": plastic_state_cache_info,
        "initial_stress_array_cache": initial_stress_cache_info,
        "combined_tangent_internal_assembly": True,
        "internal_external_work_ratio": work_ratio,
        "convergence_criteria": dict(final_convergence),
        "sparse_pattern_cached": sparse_pattern is not None,
        "constraint_dofs_cached": free_dofs is not None and fixed_dofs is not None,
        "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
        "lagrange_linear_cache": _lagrange_linear_cache_summary(lagrange_linear_cache_events, lagrange_linear_cache),
        "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
    }
    if has_lagrange_mpc:
        info["constraint_norm"] = constraint_norm
        info["multipliers"] = last_multipliers
    return u, reactions, info


def _geostatic_initial_stresses(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    stage_cfg: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    surface_y = float(stage_cfg.get("surface_y", stage_cfg.get("ground_y", np.max(mesh.coords[:, 1]))))
    scale = float(stage_cfg.get("scale", 1.0))
    k0_override = stage_cfg.get("k0", stage_cfg.get("K0"))
    node_index = mesh.node_index
    out: dict[str, np.ndarray] = {}
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        centroid_y = float(np.mean(coords[:, 1]))
        depth = max(surface_y - centroid_y, 0.0)
        sigma_v = -material.gamma * depth * scale
        k0 = float(k0_override) if k0_override is not None else _material_k0(material)
        out[element.id] = np.array([k0 * sigma_v, sigma_v, k0 * sigma_v, 0.0], dtype=float)
    return out


def _collect_dynamic_constraints(
    mesh: Mesh2D,
    boundary_conditions: Any,
    structural_elements: list[StructuralElement2D] | None,
    time: float,
) -> dict[int, float]:
    processed: list[Any] = []
    for bc in _ensure_list(boundary_conditions):
        if not isinstance(bc, Mapping):
            processed.append(bc)
            continue
        item = dict(bc)
        unit_scale = _dynamic_boundary_unit_scale(item)
        rows = _time_history_rows(item.get("time_history", item.get("displacement_history", item.get("history", item.get("values")))))
        if rows:
            if "dof" in item:
                dof_name = str(item.get("dof", "")).lower()
                item["value"] = unit_scale * _dynamic_boundary_value_at(rows, time, dof_name, default=float(item.get("value", 0.0) or 0.0) / unit_scale)
            else:
                for dof in ("ux", "uy", "rz", "theta", "rotation"):
                    if dof in item or any(dof in row for row in rows):
                        item[dof] = unit_scale * _dynamic_boundary_value_at(rows, time, dof, default=float(item.get(dof, 0.0) or 0.0) / unit_scale)
        scale_rows = _time_history_rows(item.get("scale_history", item.get("factor_history")))
        if scale_rows:
            scale = _history_value_at(scale_rows, time, ("scale", "factor", "value"), default=1.0)
            for key in ("ux", "uy", "rz", "theta", "rotation", "value"):
                if key in item and item[key] not in (None, ""):
                    item[key] = float(item[key]) * scale
        processed.append(item)
    return collect_constraints(mesh, processed, structural_elements=structural_elements)


def _solve_dynamic_newmark_monolithic_step(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u_old: np.ndarray,
    v_old: np.ndarray,
    a_old: np.ndarray,
    p_old: np.ndarray | None,
    external: np.ndarray,
    constrained: Mapping[int, float],
    fixed_p: Mapping[int, float],
    *,
    dt: float,
    beta: float,
    gamma: float,
    mass: csr_matrix,
    damping: csr_matrix,
    stiffness: csr_matrix,
    mpc_stiffness: csr_matrix | None,
    mpc_info: Mapping[str, Any],
    pressure_mass: csr_matrix | None,
    pressure_conductivity: csr_matrix | None,
    pressure_biot: csr_matrix | None,
    pressure_interface: csr_matrix | None,
    hydro: Mapping[str, Any],
    stage_name: str,
    solver: Mapping[str, Any] | None,
    initial_stresses: Mapping[str, np.ndarray] | None,
    interfaces: list[Interface2D] | None,
    structural_elements: list[StructuralElement2D] | None,
    strength_factor: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    initial_stress_cache: InitialStressArrayCache | None = None,
    reduced_matrix_cache: ReducedMatrixCache | None = None,
    monolithic_lhs_pattern_cache: CoupledUPMonolithicMatrixCache | None = None,
    effective_stiffness_linear_combination_cache: dict[str, Any] | None = None,
    tangent_method: str,
    dynamic_up: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, dict[str, Any], ReducedMatrixCache | None, CoupledUPMonolithicMatrixCache | None, dict[str, Any] | None]:
    step_start = _perf_counter()
    settings = _newton_settings(solver)
    if plastic_state_cache is None:
        plastic_state_cache = (
            build_plastic_state_array_cache(mesh, materials, plastic_state)
            if plastic_state or any(material.is_plastic for material in materials.values())
            else None
        )
    plastic_state_cache_info = plastic_state_array_cache_info(plastic_state_cache)
    if initial_stress_cache is None and initial_stresses:
        initial_stress_cache = build_initial_stress_array_cache(mesh, initial_stresses)
    initial_stress_cache_info = initial_stress_array_cache_info(initial_stress_cache)
    reduction_cache = reduced_matrix_cache
    reduction_cache_events: list[Mapping[str, Any]] = []
    symbolic_cache_events: list[Mapping[str, Any]] = []
    monolithic_lhs_pattern_events: list[Mapping[str, Any]] = []
    convergence_history: list[dict[str, Any]] = []
    tangent_internal_assembly_elapsed_total = 0.0
    effective_stiffness_assembly_elapsed_total = 0.0
    effective_stiffness_combo_cache: dict[str, Any] | None = effective_stiffness_linear_combination_cache
    effective_stiffness_combo_events: list[Mapping[str, Any]] = []
    monolithic_assembly_elapsed_total = 0.0
    linear_solve_elapsed_total = 0.0
    a0 = 1.0 / (beta * dt * dt)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    u = u_old + dt * v_old + dt * dt * (0.5 - beta) * a_old
    for dof, value in constrained.items():
        u[dof] = value
    p = None if p_old is None else p_old.copy()
    if dynamic_up:
        if p is None or pressure_mass is None or pressure_conductivity is None or pressure_biot is None:
            raise FEM2DError(f"{stage_name}: dynamic u-p requires pressure state and matrices")
        for idx, value in fixed_p.items():
            p[idx] = value

    n_u = u.size
    free_u, fixed_u = _free_index_arrays(n_u, constrained, stage_name=stage_name)
    free_p: np.ndarray | None = None
    fixed_pressure: np.ndarray | None = None
    free_aug: np.ndarray | None = None
    fixed_aug: np.ndarray | None = None
    fixed_pressure_values: np.ndarray | None = None
    rhs_aug_buffer: np.ndarray | None = None
    fixed_values_buffer: np.ndarray | None = None
    if dynamic_up:
        if p is None:
            raise FEM2DError(f"{stage_name}: dynamic u-p requires pressure state")
        free_p, fixed_pressure = _free_index_arrays(p.size, fixed_p, stage_name=stage_name, label="fixed pressure")
        rhs_aug_buffer = np.empty(n_u + p.size, dtype=float)
        free_pressure_aug = _offset_index_array(free_p, n_u)
        fixed_pressure_aug = _offset_index_array(fixed_pressure, n_u)
        free_aug = np.empty(free_u.size + free_pressure_aug.size, dtype=int)
        free_aug[: free_u.size] = free_u
        free_aug[free_u.size :] = free_pressure_aug
        fixed_aug = np.empty(fixed_u.size + fixed_pressure_aug.size, dtype=int)
        fixed_aug[: fixed_u.size] = fixed_u
        fixed_aug[fixed_u.size :] = fixed_pressure_aug
        fixed_pressure_values = np.asarray([float(fixed_p[int(idx)]) for idx in fixed_pressure], dtype=float)
        fixed_values_buffer = np.zeros(fixed_aug.size, dtype=float)
    linear_solver = _repeated_direct_linear_solver_config(solver)
    force_reference_norm = float(np.linalg.norm(external[free_u])) if free_u.size else 0.0
    target_norm = force_reference_norm
    linear_method = "direct"
    residual_norm = math.inf
    line_search_reductions = 0
    pressure_residual_norm = 0.0
    boundary_info: dict[str, Any] = {}
    mass_balance_terms: dict[str, float] = {}
    pressure_lhs: csr_matrix | None = None
    pressure_rhs_const: np.ndarray | None = None
    qb = np.zeros(0, dtype=float)
    dynamic_convergence = _dynamic_residual_metrics(
        force_norm=math.inf,
        force_reference=force_reference_norm,
        pressure_norm=0.0,
        pressure_reference=0.0,
        pressure_enabled=dynamic_up,
        settings=settings,
    )

    def evaluate_trial_state(
        trial_u: np.ndarray,
        trial_p: np.ndarray | None,
    ) -> tuple[dict[str, Any], float]:
        trial_a = a0 * (trial_u - u_old) - a2 * v_old - a3 * a_old
        trial_v = v_old + dt * ((1.0 - gamma) * a_old + gamma * trial_a)
        trial_fint = assemble_internal_force(
            mesh,
            materials,
            trial_u,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
        )
        resisting_u = trial_fint + damping @ trial_v + mass @ trial_a
        if mpc_stiffness is not None:
            resisting_u = resisting_u + mpc_stiffness @ trial_u
        trial_residual_u = external - resisting_u
        trial_pressure_norm = 0.0
        trial_pressure_reference = 0.0
        if dynamic_up:
            assert (
                trial_p is not None
                and p_old is not None
                and pressure_biot is not None
                and pressure_mass is not None
                and pressure_conductivity is not None
                and free_p is not None
            )
            trial_rb, trial_qb, _ = assemble_pressure_boundary_terms(mesh, hydro, pressure=trial_p)
            trial_interface = pressure_interface if pressure_interface is not None else csr_matrix(pressure_mass.shape, dtype=float)
            trial_pressure_lhs = (pressure_mass / dt + pressure_conductivity + trial_interface + trial_rb).tocsr()
            trial_pressure_rhs = (pressure_mass @ p_old) / dt + (pressure_biot.T @ u_old) / dt + trial_qb
            trial_residual_u = trial_residual_u + pressure_biot @ trial_p
            trial_pressure_residual = trial_pressure_rhs - (pressure_biot.T @ trial_u) / dt - trial_pressure_lhs @ trial_p
            trial_pressure_vec = np.asarray(trial_pressure_residual).ravel()
            trial_pressure_resisting = np.asarray((pressure_biot.T @ trial_u) / dt + trial_pressure_lhs @ trial_p).ravel()
            trial_pressure_rhs_vec = np.asarray(trial_pressure_rhs).ravel()
            trial_pressure_norm = float(np.linalg.norm(trial_pressure_vec[free_p])) if free_p.size else 0.0
            trial_pressure_reference = max(
                float(np.linalg.norm(trial_pressure_rhs_vec[free_p])) if free_p.size else 0.0,
                float(np.linalg.norm(trial_pressure_resisting[free_p])) if free_p.size else 0.0,
            )
            resisting_u = resisting_u - pressure_biot @ trial_p
        trial_force_vec = np.asarray(trial_residual_u).ravel()
        trial_resisting_vec = np.asarray(resisting_u).ravel()
        trial_force_norm = float(np.linalg.norm(trial_force_vec[free_u])) if free_u.size else 0.0
        trial_force_reference = max(
            force_reference_norm,
            float(np.linalg.norm(trial_resisting_vec[free_u])) if free_u.size else 0.0,
        )
        metrics = _dynamic_residual_metrics(
            force_norm=trial_force_norm,
            force_reference=trial_force_reference,
            pressure_norm=trial_pressure_norm,
            pressure_reference=trial_pressure_reference,
            pressure_enabled=dynamic_up,
            settings=settings,
        )
        return metrics, float(metrics["normalized_residual_merit"])

    for iteration in range(1, settings["max_iter"] + 2):
        iteration_start = _perf_counter()
        a_trial = a0 * (u - u_old) - a2 * v_old - a3 * a_old
        v_trial = v_old + dt * ((1.0 - gamma) * a_old + gamma * a_trial)
        assembly_start = _perf_counter()
        tangent, fint = assemble_tangent_and_internal_force(
            mesh,
            materials,
            u,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            interfaces=interfaces,
            structural_elements=structural_elements,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            tangent_method=tangent_method,
        )
        tangent_internal_assembly_elapsed = max(_perf_counter() - assembly_start, 0.0)
        tangent_internal_assembly_elapsed_total += tangent_internal_assembly_elapsed
        residual_u = external - fint - damping @ v_trial - mass @ a_trial
        if mpc_stiffness is not None:
            residual_u = residual_u - mpc_stiffness @ u
        if mpc_stiffness is not None:
            tangent = (tangent + mpc_stiffness).tocsr()
        effective_start = _perf_counter()
        effective_tangent, combo_event, effective_stiffness_combo_cache = _csr_linear_combination_matrix(
            [tangent, damping, mass],
            [1.0, a1, a0],
            cache=effective_stiffness_combo_cache,
        )
        effective_stiffness_combo_events.append(combo_event)
        effective_elapsed = max(_perf_counter() - effective_start, 0.0)
        effective_stiffness_assembly_elapsed_total += effective_elapsed
        iteration_row: dict[str, Any] = {
            "iteration": iteration - 1,
            "target_norm": target_norm,
            "linear_method": linear_method,
            "converged": False,
            "assembly_elapsed_seconds": tangent_internal_assembly_elapsed,
            "tangent_internal_assembly_elapsed_seconds": tangent_internal_assembly_elapsed,
            "tangent_assembly_elapsed_seconds": tangent_internal_assembly_elapsed,
            "internal_force_assembly_elapsed_seconds": 0.0,
            "effective_stiffness_assembly_elapsed_seconds": effective_elapsed,
            "effective_stiffness_linear_combination_state": str(combo_event.get("state", "")),
            "monolithic_assembly_elapsed_seconds": 0.0,
            "reduced_matrix_elapsed_seconds": 0.0,
            "linear_solve_elapsed_seconds": 0.0,
            "line_search_elapsed_seconds": 0.0,
            "postprocess_elapsed_seconds": 0.0,
            "elapsed_seconds": 0.0,
        }
        pressure_reference_norm = 0.0
        if dynamic_up:
            assert p is not None and pressure_biot is not None and pressure_mass is not None and pressure_conductivity is not None
            Rb, qb, boundary_info = assemble_pressure_boundary_terms(mesh, hydro, pressure=p)
            interface_matrix = pressure_interface if pressure_interface is not None else csr_matrix(pressure_mass.shape, dtype=float)
            pressure_lhs = (pressure_mass / dt + pressure_conductivity + interface_matrix + Rb).tocsr()
            pressure_rhs_const = (pressure_mass @ p_old) / dt + (pressure_biot.T @ u_old) / dt + qb
            residual_u = residual_u + pressure_biot @ p
            residual_p = pressure_rhs_const - (pressure_biot.T @ u) / dt - pressure_lhs @ p
            rhs = _fill_two_block_vector(rhs_aug_buffer, residual_u, residual_p)
            assert free_p is not None and fixed_pressure is not None and free_aug is not None
            force_residual_vec = np.asarray(residual_u).ravel()
            residual_norm = float(np.linalg.norm(force_residual_vec[free_u])) if free_u.size else 0.0
            pressure_residual_norm = float(np.linalg.norm(np.asarray(residual_p).ravel()[free_p])) if free_p.size else 0.0
            pressure_resisting = np.asarray((pressure_biot.T @ u) / dt + pressure_lhs @ p).ravel()
            pressure_rhs_vec = np.asarray(pressure_rhs_const).ravel()
            pressure_reference_norm = max(
                float(np.linalg.norm(pressure_rhs_vec[free_p])) if free_p.size else 0.0,
                float(np.linalg.norm(pressure_resisting[free_p])) if free_p.size else 0.0,
            )
            iteration_row["pressure_residual_norm"] = pressure_residual_norm
        else:
            residual_vec = np.asarray(residual_u).ravel()
            dp = None
            residual_norm = float(np.linalg.norm(residual_vec[free_u])) if free_u.size else 0.0
        resisting_u = fint + damping @ v_trial + mass @ a_trial
        if mpc_stiffness is not None:
            resisting_u = resisting_u + mpc_stiffness @ u
        if dynamic_up and pressure_biot is not None and p is not None:
            resisting_u = resisting_u - pressure_biot @ p
        force_reference = max(
            force_reference_norm,
            float(np.linalg.norm(np.asarray(resisting_u).ravel()[free_u])) if free_u.size else 0.0,
        )
        dynamic_convergence = _dynamic_residual_metrics(
            force_norm=residual_norm,
            force_reference=force_reference,
            pressure_norm=pressure_residual_norm,
            pressure_reference=pressure_reference_norm,
            pressure_enabled=dynamic_up,
            settings=settings,
        )
        converged = bool(dynamic_convergence["converged"])
        iteration_row["residual_norm"] = residual_norm
        iteration_row["converged"] = converged
        iteration_row.update(dynamic_convergence)
        if converged:
            iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
            convergence_history.append(iteration_row)
            reactions = fint + damping @ v_trial + mass @ a_trial - external
            if dynamic_up and pressure_biot is not None and p is not None:
                reactions = reactions - pressure_biot @ p
                if pressure_lhs is not None and pressure_rhs_const is not None:
                    storage_rate = pressure_mass @ ((p - p_old) / dt) if pressure_mass is not None and p_old is not None else np.zeros_like(p)
                    coupling_rate = pressure_biot.T @ ((u - u_old) / dt)
                    diffusion_flow = pressure_conductivity @ p if pressure_conductivity is not None else np.zeros_like(p)
                    interface_flow = pressure_interface @ p if pressure_interface is not None else np.zeros_like(p)
                    mass_balance_terms = {
                        "storage_rate": float(np.sum(storage_rate)),
                        "coupling_rate": float(np.sum(coupling_rate)),
                        "diffusion_flow": float(np.sum(diffusion_flow)),
                        "interface_transfer_flow": float(np.sum(interface_flow)),
                        "boundary_source": float(np.sum(qb)) if "qb" in locals() else 0.0,
                        "residual_norm": pressure_residual_norm,
                    }
            info = {
                "method": "newmark_monolithic_up" if dynamic_up else "newmark_newton",
                "iterations": iteration - 1,
                "residual_norm": residual_norm,
                "pressure_residual_norm": pressure_residual_norm,
                "converged": True,
                "convergence_criteria": dict(dynamic_convergence),
                "tangent": tangent_method,
                "line_search_reductions": line_search_reductions,
                "pressure_boundary": boundary_info,
                "mass_balance_terms": mass_balance_terms,
                "convergence_history": convergence_history,
                "combined_tangent_internal_assembly": True,
                "plastic_state_array_cache": plastic_state_cache_info,
                "initial_stress_array_cache": initial_stress_cache_info,
                "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
                "monolithic_lhs_pattern_cache": _coupled_up_monolithic_cache_summary(monolithic_lhs_pattern_events, monolithic_lhs_pattern_cache),
                "monolithic_lhs_pattern_cache_events": list(monolithic_lhs_pattern_events),
                "effective_stiffness_linear_combination_cache": _csr_linear_combination_cache_summary(effective_stiffness_combo_events, effective_stiffness_combo_cache),
                "effective_stiffness_linear_combination_cache_events": list(effective_stiffness_combo_events),
                "reduced_matrix_cache_events": list(reduction_cache_events),
                "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
                "symbolic_cache_events": list(symbolic_cache_events),
                "tangent_internal_assembly_elapsed_seconds": tangent_internal_assembly_elapsed_total,
                "effective_stiffness_assembly_elapsed_seconds": effective_stiffness_assembly_elapsed_total,
                "monolithic_assembly_elapsed_seconds": monolithic_assembly_elapsed_total,
                "linear_solve_elapsed_seconds": linear_solve_elapsed_total,
                "elapsed_seconds": max(_perf_counter() - step_start, 0.0),
            }
            return u, v_trial, a_trial, p, reactions, info, reduction_cache, monolithic_lhs_pattern_cache, effective_stiffness_combo_cache
        if iteration > settings["max_iter"]:
            iteration_row["final_correction_check"] = True
            iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
            convergence_history.append(iteration_row)
            break
        solve_start = _perf_counter()
        if dynamic_up:
            assert p is not None and pressure_biot is not None and pressure_lhs is not None
            monolithic_start = _perf_counter()
            lhs, monolithic_lhs_pattern_cache, monolithic_event = _assemble_coupled_up_monolithic_lhs(
                effective_tangent,
                pressure_biot,
                pressure_lhs,
                dt,
                cache=monolithic_lhs_pattern_cache,
            )
            monolithic_lhs_pattern_events.append(monolithic_event)
            monolithic_elapsed = max(_perf_counter() - monolithic_start, 0.0)
            monolithic_assembly_elapsed_total += monolithic_elapsed
            iteration_row["monolithic_assembly_elapsed_seconds"] = monolithic_elapsed
            iteration_row["monolithic_lhs_direct_fill_reused"] = bool(monolithic_event.get("reused", False))
            iteration_row["monolithic_lhs_direct_fill_built"] = bool(monolithic_event.get("built", False))
            assert fixed_pressure is not None and fixed_aug is not None and fixed_values_buffer is not None and fixed_pressure_values is not None
            fixed_values = fixed_values_buffer
            if fixed_u.size:
                fixed_values[: fixed_u.size] = 0.0
            if fixed_pressure.size:
                fixed_values[fixed_u.size :] = fixed_pressure_values - p[fixed_pressure]
            correction = np.zeros(n_u + p.size, dtype=float)
            if fixed_aug.size:
                correction[fixed_aug] = fixed_values
            if free_aug.size:
                correction_free, info, reduction_cache = solve_reduced_linear_system(
                    lhs,
                    rhs,
                    free_aug,
                    fixed_aug,
                    fixed_values=fixed_values,
                    reduction_cache=reduction_cache,
                    stage_name=stage_name,
                    solver=linear_solver,
                    validate_cache=True,
                )
                correction[free_aug] = correction_free
                cache_event = info.get("reduced_matrix_cache", {})
                if isinstance(cache_event, Mapping):
                    reduction_cache_events.append(cache_event)
                    iteration_row["reduced_matrix_cache_reused"] = bool(cache_event.get("reused", False))
                    iteration_row["reduced_matrix_cache_built"] = bool(cache_event.get("built", False))
                iteration_row["reduced_matrix_elapsed_seconds"] = float(info.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
                symbolic_event = info.get("symbolic_cache", {})
                if isinstance(symbolic_event, Mapping):
                    symbolic_cache_events.append(symbolic_event)
                    iteration_row["symbolic_cache_state"] = str(symbolic_event.get("state", ""))
                if "factor_cache" in info:
                    iteration_row["lu_factor_cache_state"] = str(info.get("factor_cache", ""))
                linear_method = str(info.get("method", linear_method))
            else:
                linear_method = "none"
            du = correction[:n_u]
            dp = correction[n_u:]
        else:
            du = np.zeros(n_u, dtype=float)
            if free_u.size:
                correction_free, info, reduction_cache = solve_reduced_linear_system(
                    effective_tangent,
                    residual_vec,
                    free_u,
                    fixed_u,
                    fixed_values=np.zeros(fixed_u.size, dtype=float),
                    reduction_cache=reduction_cache,
                    stage_name=stage_name,
                    solver=linear_solver,
                    validate_cache=True,
                )
                du[free_u] = correction_free
                cache_event = info.get("reduced_matrix_cache", {})
                if isinstance(cache_event, Mapping):
                    reduction_cache_events.append(cache_event)
                    iteration_row["reduced_matrix_cache_reused"] = bool(cache_event.get("reused", False))
                    iteration_row["reduced_matrix_cache_built"] = bool(cache_event.get("built", False))
                iteration_row["reduced_matrix_elapsed_seconds"] = float(info.get("reduced_matrix_elapsed_seconds", 0.0) or 0.0)
                symbolic_event = info.get("symbolic_cache", {})
                if isinstance(symbolic_event, Mapping):
                    symbolic_cache_events.append(symbolic_event)
                    iteration_row["symbolic_cache_state"] = str(symbolic_event.get("state", ""))
                if "factor_cache" in info:
                    iteration_row["lu_factor_cache_state"] = str(info.get("factor_cache", ""))
                linear_method = str(info.get("method", linear_method))
            else:
                linear_method = "none"
            dp = None
        linear_elapsed = max(_perf_counter() - solve_start, 0.0)
        linear_solve_elapsed_total += linear_elapsed
        iteration_row["linear_method"] = linear_method
        iteration_row["linear_solve_elapsed_seconds"] = linear_elapsed
        base_u = u.copy()
        base_p = None if p is None else p.copy()
        alpha = 1.0
        applied_alpha = 1.0
        line_search_start = _perf_counter()
        if settings["line_search"]:
            base_merit = float(dynamic_convergence["normalized_residual_merit"])
            best_merit = math.inf
            best_alpha = max(float(settings["min_line_search_alpha"]), np.finfo(float).eps)
            best_u: np.ndarray | None = None
            best_p: np.ndarray | None = None
            accepted = False
            reductions_this_iteration = 0
            for _line_search_iteration in range(settings["max_line_search"] + 1):
                trial_u = base_u + alpha * du
                for dof, value in constrained.items():
                    trial_u[dof] = value
                trial_p = base_p
                if dynamic_up and p is not None and dp is not None and base_p is not None:
                    trial_p = base_p + alpha * dp
                    for idx, value in fixed_p.items():
                        trial_p[idx] = value
                try:
                    trial_metrics, trial_merit = evaluate_trial_state(trial_u, trial_p)
                except Exception as exc:
                    if not _is_recoverable_material_trial_error(exc):
                        raise
                    trial_metrics = {"normalized_residual_merit": math.inf}
                    trial_merit = math.inf
                if math.isfinite(trial_merit) and trial_merit < best_merit:
                    best_merit = trial_merit
                    best_alpha = alpha
                    best_u = trial_u
                    best_p = trial_p
                armijo_limit = (1.0 - float(settings["line_search_armijo"]) * alpha) * base_merit
                if math.isfinite(trial_merit) and (trial_merit <= armijo_limit or alpha <= settings["min_line_search_alpha"]):
                    u = trial_u
                    p = trial_p
                    applied_alpha = alpha
                    iteration_row["line_search_trial_merit"] = trial_merit
                    iteration_row["line_search_trial_converged"] = bool(trial_metrics.get("converged", False))
                    accepted = True
                    break
                if alpha <= settings["min_line_search_alpha"]:
                    break
                alpha = max(alpha * settings["line_search_factor"], settings["min_line_search_alpha"])
                reductions_this_iteration += 1
            if not accepted:
                if best_u is None:
                    raise FEM2DError(f"{stage_name}: dynamic Newmark line search did not find a finite trial state")
                u = best_u
                p = best_p
                applied_alpha = best_alpha
                iteration_row["line_search_trial_merit"] = best_merit
                iteration_row["line_search_fallback_to_best"] = True
            line_search_reductions += reductions_this_iteration
            iteration_row["line_search_reductions"] = reductions_this_iteration
            iteration_row["applied_alpha"] = applied_alpha
        else:
            u = base_u + du
            for dof, value in constrained.items():
                u[dof] = value
            if dynamic_up and p is not None and dp is not None and base_p is not None:
                p = base_p + dp
                for idx, value in fixed_p.items():
                    p[idx] = value
        line_search_elapsed = max(_perf_counter() - line_search_start, 0.0)
        iteration_row["line_search_elapsed_seconds"] = line_search_elapsed
        iteration_row["elapsed_seconds"] = max(_perf_counter() - iteration_start, 0.0)
        convergence_history.append(iteration_row)
    if not bool(settings["allow_nonconvergence"]):
        raise FEM2DError(
            f"{stage_name}: dynamic Newmark nonlinear step did not converge, residual={residual_norm:.6e}",
            diagnostics={
                "status": "dynamic_newmark_nonconvergence",
                "residual_norm": residual_norm,
                "pressure_residual_norm": pressure_residual_norm,
                "convergence_criteria": dict(dynamic_convergence),
                "convergence_history": convergence_history,
                "line_search_reductions": line_search_reductions,
            },
        )
    a_trial = a0 * (u - u_old) - a2 * v_old - a3 * a_old
    v_trial = v_old + dt * ((1.0 - gamma) * a_old + gamma * a_trial)
    reactions = assemble_internal_force(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        initial_stress_cache=initial_stress_cache,
        interfaces=interfaces,
        structural_elements=structural_elements,
        strength_factor=strength_factor,
        plastic_state=plastic_state,
        plastic_state_cache=plastic_state_cache,
    ) + damping @ v_trial + mass @ a_trial - external
    info = {
        "method": "newmark_monolithic_up" if dynamic_up else "newmark_newton",
        "iterations": settings["max_iter"],
        "residual_norm": residual_norm,
        "pressure_residual_norm": pressure_residual_norm,
        "converged": False,
        "convergence_criteria": dict(dynamic_convergence),
        "tangent": tangent_method,
        "line_search_reductions": line_search_reductions,
        "pressure_boundary": boundary_info,
        "mass_balance_terms": mass_balance_terms,
        "convergence_history": convergence_history,
        "combined_tangent_internal_assembly": True,
        "plastic_state_array_cache": plastic_state_cache_info,
        "initial_stress_array_cache": initial_stress_cache_info,
        "reduced_matrix_cache": _reduced_matrix_cache_summary(reduction_cache_events, reduction_cache),
        "monolithic_lhs_pattern_cache": _coupled_up_monolithic_cache_summary(monolithic_lhs_pattern_events, monolithic_lhs_pattern_cache),
        "monolithic_lhs_pattern_cache_events": list(monolithic_lhs_pattern_events),
        "effective_stiffness_linear_combination_cache": _csr_linear_combination_cache_summary(effective_stiffness_combo_events, effective_stiffness_combo_cache),
        "effective_stiffness_linear_combination_cache_events": list(effective_stiffness_combo_events),
        "reduced_matrix_cache_events": list(reduction_cache_events),
        "symbolic_ordering_cache": _symbolic_ordering_cache_summary(symbolic_cache_events),
        "symbolic_cache_events": list(symbolic_cache_events),
        "tangent_internal_assembly_elapsed_seconds": tangent_internal_assembly_elapsed_total,
        "effective_stiffness_assembly_elapsed_seconds": effective_stiffness_assembly_elapsed_total,
        "monolithic_assembly_elapsed_seconds": monolithic_assembly_elapsed_total,
        "linear_solve_elapsed_seconds": linear_solve_elapsed_total,
        "elapsed_seconds": max(_perf_counter() - step_start, 0.0),
    }
    return u, v_trial, a_trial, p, reactions, info, reduction_cache, monolithic_lhs_pattern_cache, effective_stiffness_combo_cache


def _dynamic_load_vector(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    structural_elements: list[StructuralElement2D] | None,
    cfg: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
    base_load: np.ndarray,
    dynamic_cfg: Mapping[str, Any],
    time: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    scale = _dynamic_load_scale(dynamic_cfg, time)
    load = np.asarray(base_load, dtype=float).copy() * scale
    seismic_loads, seismic_info = _response_seismic_loads(mesh, cfg, _dynamic_seismic_stage_cfg(stage_cfg, dynamic_cfg), time)
    processed: list[dict[str, Any]] = []
    skipped = 0
    for item in seismic_loads:
        factor = _load_case_factor(cfg, stage_cfg, item.get("load_case", item.get("case")))
        if factor is None:
            skipped += 1
            continue
        processed.append(_scale_load_item(item, factor))
    if processed:
        load += assemble_load_vector(mesh, materials, processed, structural_elements=structural_elements)
    info = dict(seismic_info)
    info["load_scale"] = scale
    info["skipped_inactive_loads"] = skipped
    return load, info


def _dynamic_seismic_stage_cfg(stage_cfg: Mapping[str, Any], dynamic_cfg: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(stage_cfg)
    seismic = _seismic_mapping({}, stage_cfg)
    if isinstance(seismic, Mapping):
        seismic = dict(seismic)
        seismic.setdefault("time_history_mode", dynamic_cfg.get("time_history_mode", "linear"))
        out["seismic"] = seismic
    return out


def _dynamic_initial_acceleration(
    mass: csr_matrix,
    damping: csr_matrix,
    stiffness: csr_matrix,
    load: np.ndarray,
    displacement: np.ndarray,
    velocity: np.ndarray,
    constrained: Mapping[int, float],
    stage_name: str,
    solver: Mapping[str, Any] | None,
) -> np.ndarray:
    fixed_acc = {int(dof): 0.0 for dof in constrained}
    rhs = load - damping @ velocity - stiffness @ displacement
    return _solve_sparse_with_constraints(mass, np.asarray(rhs).ravel(), fixed_acc, stage_name=f"{stage_name}-initial-acceleration", solver=solver)


def _solve_dynamic_effective_system(
    matrix: csr_matrix,
    rhs: np.ndarray,
    constrained: Mapping[int, float],
    mpc_info: Mapping[str, Any],
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
    mpc_plan: MPCStagePlan,
    effective_cache: dict[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if int(mpc_info.get("count", 0) or 0) > 0 and mpc_plan.use_elimination_linear:
        return solve_linear_system_with_mpc_elimination(matrix, rhs, constrained, mpc_info, stage_name=stage_name, solver=solver)
    if int(mpc_info.get("count", 0) or 0) > 0 and mpc_plan.use_lagrange_linear:
        return solve_linear_system_with_mpc_lagrange(matrix, rhs, constrained, mpc_info, stage_name=stage_name, solver=solver)
    if effective_cache is not None and bool(effective_cache.get("enabled", False)):
        free, fixed = _free_index_arrays(matrix.shape[0], constrained, stage_name=stage_name)
        solution = np.zeros(matrix.shape[0], dtype=float)
        for dof, value in constrained.items():
            solution[int(dof)] = float(value)
        reduction_cache = effective_cache.get("reduced_matrix_cache")
        linear_solver = _dynamic_effective_linear_solver_config(solver)
        if free.size:
            solution[free], info, reduction_cache = solve_reduced_linear_system(
                matrix,
                rhs,
                free,
                fixed,
                fixed_values=solution[fixed],
                reduction_cache=reduction_cache if isinstance(reduction_cache, ReducedMatrixCache) else None,
                stage_name=stage_name,
                solver=linear_solver,
                validate_cache=not bool(effective_cache.get("matrix_reused", False)),
            )
            effective_cache["reduced_matrix_cache"] = reduction_cache
        else:
            info = {"method": "none", "iterations": 0, "residual_norm": 0.0, "reduced_matrix_cache": {"enabled": bool(reduction_cache is not None), "reused": bool(reduction_cache is not None), "built": False}}
        residual = float(np.linalg.norm(matrix @ solution - rhs))
        info = dict(info)
        info["method"] = "newmark_effective_cached"
        info["iterations"] = int(info.get("iterations", 1) or 1)
        info["residual_norm"] = residual
        info["converged"] = True
        _update_dynamic_effective_factor_cache(effective_cache, info)
        return solution, info
    solution = _solve_sparse_with_constraints(matrix, rhs, constrained, stage_name=stage_name, solver=solver)
    residual = float(np.linalg.norm(matrix @ solution - rhs))
    return solution, {"method": "newmark_effective", "iterations": 1, "residual_norm": residual, "converged": True}


def _csr_pattern_signature(matrix: csr_matrix) -> tuple[tuple[int, int], int, int]:
    csr = matrix.tocsr()
    return (tuple(int(v) for v in csr.shape), int(csr.nnz), int(csr.indptr.size))


def _csr_linear_combination_cache_matches(cache: Mapping[str, Any] | None, matrices: list[csr_matrix]) -> bool:
    if not isinstance(cache, Mapping):
        return False
    patterns = cache.get("source_patterns")
    if not isinstance(patterns, list) or len(patterns) != len(matrices):
        return False
    for matrix, pattern in zip(matrices, patterns, strict=True):
        if not isinstance(pattern, Mapping):
            return False
        if tuple(pattern.get("shape", ())) != tuple(matrix.shape):
            return False
        indptr = pattern.get("indptr")
        indices = pattern.get("indices")
        if not isinstance(indptr, np.ndarray) or not isinstance(indices, np.ndarray):
            return False
        if not np.array_equal(indptr, matrix.indptr) or not np.array_equal(indices, matrix.indices):
            return False
    return True


def _build_csr_linear_combination_cache(matrices: list[csr_matrix]) -> dict[str, Any]:
    if not matrices:
        raise FEM2DError("linear combination cache requires at least one matrix")
    shape = matrices[0].shape
    for matrix in matrices:
        if matrix.shape != shape:
            raise FEM2DError("linear combination matrices must have the same shape")
    nrows = int(shape[0])
    union_indices: list[np.ndarray] = []
    indptr = np.zeros(nrows + 1, dtype=np.int64)
    for row in range(nrows):
        row_cols: list[np.ndarray] = []
        for matrix in matrices:
            start = int(matrix.indptr[row])
            end = int(matrix.indptr[row + 1])
            if end > start:
                row_cols.append(matrix.indices[start:end])
        if row_cols:
            cols = np.unique(np.concatenate(row_cols)).astype(np.int32, copy=False)
        else:
            cols = np.zeros(0, dtype=np.int32)
        union_indices.append(cols)
        indptr[row + 1] = indptr[row] + int(cols.size)
    indices = np.concatenate(union_indices).astype(np.int32, copy=False) if union_indices else np.zeros(0, dtype=np.int32)
    positions: list[np.ndarray] = []
    for matrix in matrices:
        pos = np.empty(matrix.nnz, dtype=np.int64)
        for row in range(nrows):
            start = int(matrix.indptr[row])
            end = int(matrix.indptr[row + 1])
            if end <= start:
                continue
            union_start = int(indptr[row])
            union_end = int(indptr[row + 1])
            pos[start:end] = union_start + np.searchsorted(indices[union_start:union_end], matrix.indices[start:end])
        positions.append(pos)
    return {
        "shape": tuple(int(v) for v in shape),
        "indices": indices,
        "indptr": indptr,
        "positions": positions,
        "source_patterns": [
            {"shape": tuple(int(v) for v in matrix.shape), "indices": matrix.indices.copy(), "indptr": matrix.indptr.copy()}
            for matrix in matrices
        ],
        "builds": 1,
        "hits": 0,
    }


def _csr_linear_combination_matrix(
    matrices: list[csr_matrix],
    coefficients: list[float],
    *,
    cache: dict[str, Any] | None = None,
) -> tuple[csr_matrix, dict[str, Any], dict[str, Any]]:
    csr_matrices = [matrix.tocsr() for matrix in matrices]
    if len(csr_matrices) != len(coefficients):
        raise FEM2DError("linear combination coefficient count must match matrix count")
    reused = _csr_linear_combination_cache_matches(cache, csr_matrices)
    combo_cache = cache if reused and cache is not None else _build_csr_linear_combination_cache(csr_matrices)
    if reused:
        combo_cache["hits"] = int(combo_cache.get("hits", 0) or 0) + 1
    positions = combo_cache.get("positions")
    if not isinstance(positions, list):
        raise FEM2DError("linear combination cache is missing positions")
    data = np.zeros(int(np.asarray(combo_cache["indices"]).size), dtype=np.float64)
    for matrix, coefficient, pos in zip(csr_matrices, coefficients, positions, strict=True):
        scale = float(coefficient)
        if scale == 0.0 or matrix.nnz == 0:
            continue
        data[np.asarray(pos, dtype=np.int64)] += scale * matrix.data
    matrix = csr_matrix((data, combo_cache["indices"], combo_cache["indptr"]), shape=tuple(combo_cache["shape"]))
    return matrix, {"enabled": True, "state": "hit" if reused else "miss", "nnz": int(matrix.nnz), "terms": len(csr_matrices)}, combo_cache


def _csr_linear_combination_cache_summary(events: list[Mapping[str, Any]], cache: Mapping[str, Any] | None) -> dict[str, Any]:
    if not events and cache is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "solves": len(events),
        "hits": sum(1 for event in events if str(event.get("state", "")) == "hit"),
        "builds": sum(1 for event in events if str(event.get("state", "")) == "miss"),
        "terms": int(events[-1].get("terms", 0) or 0) if events else 0,
        "nnz": int(events[-1].get("nnz", 0) or 0) if events else int(np.asarray(cache.get("indices", []) if isinstance(cache, Mapping) else []).size),
    }


def _dynamic_rayleigh_coefficients(dynamic_cfg: Mapping[str, Any]) -> tuple[float, float]:
    alpha = float(dynamic_cfg.get("rayleigh_alpha", dynamic_cfg.get("alpha_m", dynamic_cfg.get("mass_damping", 0.0))) or 0.0)
    beta = float(dynamic_cfg.get("rayleigh_beta", dynamic_cfg.get("beta_k", dynamic_cfg.get("stiffness_damping", 0.0))) or 0.0)
    damping_spec = dynamic_cfg.get("damping", dynamic_cfg.get("damping_spec"))
    if isinstance(damping_spec, Mapping):
        computed_alpha, computed_beta = _rayleigh_coefficients_from_damping_spec(damping_spec)
        if computed_alpha is not None:
            alpha = computed_alpha
        if computed_beta is not None:
            beta = computed_beta
    zeta = dynamic_cfg.get("damping_ratio")
    freq = dynamic_cfg.get("target_frequency_hz", dynamic_cfg.get("frequency_hz"))
    if zeta not in (None, "") and freq not in (None, ""):
        omega = 2.0 * math.pi * float(freq)
        if omega > 0.0:
            alpha += 2.0 * float(zeta) * omega
    return alpha, beta


def _dynamic_rayleigh_damping_matrix(
    mass: csr_matrix,
    stiffness: csr_matrix,
    dynamic_cfg: Mapping[str, Any],
) -> tuple[csr_matrix, dict[str, Any]]:
    alpha, beta = _dynamic_rayleigh_coefficients(dynamic_cfg)
    if alpha == 0.0 and beta == 0.0:
        return csr_matrix(mass.shape, dtype=np.float64), {"enabled": False, "state": "zero", "alpha": alpha, "beta": beta}
    matrix, event, _cache = _csr_linear_combination_matrix([mass, stiffness], [alpha, beta])
    event.update({"alpha": alpha, "beta": beta, "mode": "csr_linear_combination_direct_data"})
    return matrix, event


def _dynamic_effective_stiffness_cache_state(*, enabled: bool) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "matrix": None,
        "dt": None,
        "constraint_key": None,
        "linear_combination_cache": None,
        "reduced_matrix_cache": None,
        "hits": 0,
        "builds": 0,
        "fallbacks": 0,
        "linear_combination_hits": 0,
        "linear_combination_builds": 0,
        "factor_cache_hits": 0,
        "factor_cache_misses": 0,
        "symbolic_cache_hits": 0,
        "symbolic_cache_misses": 0,
        "unused_reasons": [] if enabled else ["disabled_or_exact_mpc_or_nonlinear_dynamic"],
        "matrix_reused": False,
    }


def _dynamic_effective_constraint_key(constrained: Mapping[int, float]) -> tuple[int, ...]:
    return tuple(sorted(int(dof) for dof in constrained.keys()))


def _dynamic_effective_stiffness_matrix(
    stiffness: csr_matrix,
    mass: csr_matrix,
    damping: csr_matrix,
    *,
    a0: float,
    a1: float,
    dt: float,
    constrained: Mapping[int, float],
    cache: dict[str, Any],
) -> tuple[csr_matrix, dict[str, Any]]:
    if not bool(cache.get("enabled", False)):
        cache["fallbacks"] = int(cache.get("fallbacks", 0) or 0) + 1
        return (stiffness + mass * a0 + damping * a1).tocsr(), {
            "enabled": False,
            "state": "disabled",
            "reason": ";".join(str(reason) for reason in cache.get("unused_reasons", []) if str(reason)),
            "reduced_matrix_reused": False,
        }
    key = _dynamic_effective_constraint_key(constrained)
    matrix = cache.get("matrix")
    reused = (
        isinstance(matrix, csr_matrix)
        and cache.get("dt") == float(dt)
        and cache.get("constraint_key") == key
    )
    if reused:
        cache["hits"] = int(cache.get("hits", 0) or 0) + 1
        cache["matrix_reused"] = True
        return matrix, {
            "enabled": True,
            "state": "hit",
            "dt": float(dt),
            "constraint_dofs": len(key),
            "reduced_matrix_reused": False,
            "linear_combination_state": "matrix_reused",
        }
    combo_cache = cache.get("linear_combination_cache")
    matrix, combo_event, combo_cache = _csr_linear_combination_matrix(
        [stiffness, mass, damping],
        [1.0, float(a0), float(a1)],
        cache=combo_cache if isinstance(combo_cache, dict) else None,
    )
    cache["linear_combination_cache"] = combo_cache
    if str(combo_event.get("state", "")) == "hit":
        cache["linear_combination_hits"] = int(cache.get("linear_combination_hits", 0) or 0) + 1
    else:
        cache["linear_combination_builds"] = int(cache.get("linear_combination_builds", 0) or 0) + 1
    cache["matrix"] = matrix
    cache["dt"] = float(dt)
    cache["constraint_key"] = key
    cache["reduced_matrix_cache"] = None
    cache["matrix_reused"] = False
    cache["builds"] = int(cache.get("builds", 0) or 0) + 1
    return matrix, {
        "enabled": True,
        "state": "miss",
        "dt": float(dt),
        "constraint_dofs": len(key),
        "reduced_matrix_reused": False,
        "linear_combination_state": str(combo_event.get("state", "")),
        "linear_combination_terms": int(combo_event.get("terms", 0) or 0),
    }


def _dynamic_effective_stiffness_cache_info(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(cache.get("enabled", False)),
        "cache_kind": "dynamic_effective_stiffness_cache",
        "hits": int(cache.get("hits", 0) or 0),
        "builds": int(cache.get("builds", 0) or 0),
        "fallbacks": int(cache.get("fallbacks", 0) or 0),
        "linear_combination_hits": int(cache.get("linear_combination_hits", 0) or 0),
        "linear_combination_builds": int(cache.get("linear_combination_builds", 0) or 0),
        "factor_cache_hits": int(cache.get("factor_cache_hits", 0) or 0),
        "factor_cache_misses": int(cache.get("factor_cache_misses", 0) or 0),
        "symbolic_cache_hits": int(cache.get("symbolic_cache_hits", 0) or 0),
        "symbolic_cache_misses": int(cache.get("symbolic_cache_misses", 0) or 0),
        "reduced_matrix_cached": isinstance(cache.get("reduced_matrix_cache"), ReducedMatrixCache),
        "dt": cache.get("dt"),
        "constraint_dofs": 0 if cache.get("constraint_key") is None else len(cache.get("constraint_key", ())),
        "unused_reasons": list(cache.get("unused_reasons", [])) if isinstance(cache.get("unused_reasons", []), list) else [],
    }


def _dynamic_effective_linear_solver_config(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(solver) if isinstance(solver, Mapping) else {}
    linear_source = raw.get("linear") if isinstance(raw.get("linear"), Mapping) else raw
    linear = dict(linear_source) if isinstance(linear_source, Mapping) else {}
    method = str(linear.get("method", linear.get("type", "direct"))).lower().strip().replace("-", "_")
    if method in {"", "direct", "spsolve", "superlu", "lu"}:
        linear.setdefault("cache_factorization", True)
        linear.setdefault("cache_symbolic", True)
        linear.setdefault("cache_min_size", 0)
        linear.setdefault("symbolic_cache_min_size", 0)
    raw["linear"] = linear
    return raw


def _repeated_direct_linear_solver_config(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(solver) if isinstance(solver, Mapping) else {}
    linear_source = raw.get("linear") if isinstance(raw.get("linear"), Mapping) else raw
    linear = dict(linear_source) if isinstance(linear_source, Mapping) else {}
    method = str(linear.get("method", linear.get("type", "direct"))).lower().strip().replace("-", "_")
    if method in {"", "direct", "spsolve", "superlu", "lu"}:
        linear.setdefault("cache_factorization", True)
        linear.setdefault("cache_symbolic", True)
        linear.setdefault("cache_min_size", 0)
        linear.setdefault("symbolic_cache_min_size", 0)
    raw["linear"] = linear
    return raw


def _nonlinear_factor_cache_mode(solver: Mapping[str, Any] | None) -> str:
    raw = dict(solver) if isinstance(solver, Mapping) else {}
    linear_source = raw.get("linear") if isinstance(raw.get("linear"), Mapping) else raw
    linear = dict(linear_source) if isinstance(linear_source, Mapping) else {}
    method = str(linear.get("method", linear.get("type", "direct"))).lower().strip().replace("-", "_")
    if method not in {"", "direct", "spsolve", "superlu", "lu"}:
        return "not_applicable"
    if "cache_factorization" not in linear and "factor_cache" not in linear:
        return "auto"
    return "configured" if _srm_bool(
        linear.get("cache_factorization", linear.get("factor_cache", True)), True
    ) else "disabled"


def _nonlinear_factor_cache_summary(
    history: list[Mapping[str, Any]],
    *,
    mode: str,
) -> dict[str, Any]:
    states = [
        str(row.get("lu_factor_cache_state", "") or "").strip().lower()
        for row in history
        if isinstance(row, Mapping) and str(row.get("lu_factor_cache_state", "") or "").strip()
    ]
    hits = 0
    misses = 0
    auto_disabled = 0
    disabled = 0
    consecutive_misses = 0
    max_consecutive_misses = 0
    for state in states:
        if state == "hit" or state.endswith("_hit"):
            hits += 1
            consecutive_misses = 0
        elif state == "miss" or state.endswith("_miss"):
            misses += 1
            consecutive_misses += 1
            max_consecutive_misses = max(max_consecutive_misses, consecutive_misses)
        elif state == "auto_disabled" or state.endswith("_auto_disabled"):
            auto_disabled += 1
            consecutive_misses = 0
        elif state == "disabled" or state.endswith("_disabled"):
            disabled += 1
            consecutive_misses = 0
        else:
            consecutive_misses = 0
    disabled_after_misses = auto_disabled > 0
    return {
        "mode": str(mode),
        "states": states,
        "solves": len(states),
        "hits": hits,
        "misses": misses,
        "auto_disabled_solves": auto_disabled,
        "disabled_solves": disabled,
        "max_consecutive_misses": max_consecutive_misses,
        "disabled_after_misses": disabled_after_misses,
        "disable_reason": "consecutive_factorization_cache_misses" if disabled_after_misses else "",
        "control_scope": "sparse_pattern_global_with_periodic_reprobe",
    }


def _update_dynamic_effective_factor_cache(cache: dict[str, Any], info: Mapping[str, Any]) -> None:
    factor_state = str(info.get("factor_cache", ""))
    if factor_state == "hit":
        cache["factor_cache_hits"] = int(cache.get("factor_cache_hits", 0) or 0) + 1
    elif factor_state == "miss":
        cache["factor_cache_misses"] = int(cache.get("factor_cache_misses", 0) or 0) + 1
    symbolic = info.get("symbolic_cache", {})
    if isinstance(symbolic, Mapping):
        symbolic_state = str(symbolic.get("state", ""))
        if symbolic_state == "hit":
            cache["symbolic_cache_hits"] = int(cache.get("symbolic_cache_hits", 0) or 0) + 1
        elif symbolic_state == "miss":
            cache["symbolic_cache_misses"] = int(cache.get("symbolic_cache_misses", 0) or 0) + 1


def _prepare_stage_loads(
    mesh: Mesh2D,
    cfg: Mapping[str, Any],
    stage_cfg: Mapping[str, Any],
    loads: Any,
    time: float,
) -> tuple[list[Any], dict[str, Any]]:
    processed: list[Any] = []
    skipped = 0
    for load in _ensure_list(loads):
        if not isinstance(load, Mapping):
            processed.append(load)
            continue
        factor = _load_case_factor(cfg, stage_cfg, load.get("load_case", load.get("case")))
        if factor is None:
            skipped += 1
            continue
        processed.append(_scale_load_item(load, factor))
    seismic_loads: list[dict[str, Any]] = []
    seismic_info: dict[str, Any] = {}
    if _stage_type(stage_cfg) not in DYNAMIC_2D_STAGE_TYPES:
        seismic_loads, seismic_info = _response_seismic_loads(mesh, cfg, stage_cfg, time)
    for load in seismic_loads:
        factor = _load_case_factor(cfg, stage_cfg, load.get("load_case", load.get("case")))
        if factor is None:
            skipped += 1
            continue
        processed.append(_scale_load_item(load, factor))
    info: dict[str, Any] = {
        "load_count": len(processed),
        "skipped_inactive_loads": skipped,
        "load_combination": _selected_load_combination_name(stage_cfg),
    }
    if seismic_info:
        info["seismic"] = seismic_info
    return processed, info


def _load_case_factor(cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any], case_name: Any) -> float | None:
    name = str(case_name or "").strip()
    if not name:
        return 1.0
    cases = _load_case_map(cfg)
    case = cases.get(name, {})
    if case and not bool(case.get("active", True)):
        return None
    scale = float(case.get("scale", 1.0) if case else 1.0)
    combo = _selected_load_combination(cfg, stage_cfg)
    if combo:
        factors = combo.get("factors", {})
        if isinstance(factors, Mapping):
            if name not in factors:
                return None if bool(combo.get("strict", False)) else 0.0
            scale *= float(factors[name])
    return scale


def _load_case_map(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in _ensure_list(cfg.get("load_cases", [])):
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name", raw.get("case", ""))).strip()
        if name:
            out[name] = dict(raw)
    return out


def _selected_load_combination_name(stage_cfg: Mapping[str, Any]) -> str:
    raw = stage_cfg.get("load_combination", stage_cfg.get("combination", ""))
    if isinstance(raw, Mapping):
        return str(raw.get("name", ""))
    return str(raw or "")


def _selected_load_combination(cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = stage_cfg.get("load_combination", stage_cfg.get("combination"))
    if isinstance(raw, Mapping):
        return raw
    name = str(raw or "").strip()
    if not name:
        return None
    for combo in configured_load_combinations(cfg):
        if isinstance(combo, Mapping) and str(combo.get("name", combo.get("id", ""))).strip() == name:
            return combo
    return None


def _response_seismic_loads(mesh: Mesh2D, cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any], time: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seismic = _seismic_mapping(cfg, stage_cfg)
    if not seismic:
        return [], {}
    kh, kv, source = _seismic_coefficients(seismic, time)
    layers = _ensure_list(seismic.get("layers", seismic.get("layer_design_seismic", seismic.get("design_seismic_layers", []))))
    load_case = seismic.get("load_case", stage_cfg.get("load_case", ""))
    loads: list[dict[str, Any]] = []
    if layers:
        for index, layer in enumerate(layers, start=1):
            if not isinstance(layer, Mapping):
                continue
            lkh = float(layer.get("kh", layer.get("horizontal_design_seismic", kh)))
            lkv = float(layer.get("kv", layer.get("vertical_design_seismic", kv)))
            ids = _elements_in_y_layer(mesh, layer)
            if not ids:
                continue
            loads.append({"type": "gravity", "gx": lkh, "gy": lkv, "scale": 1.0, "elements": ids, "load_case": load_case, "seismic": {"method": "response_seismic", "layer": layer.get("name", index)}})
    else:
        loads.append({"type": "gravity", "gx": kh, "gy": kv, "scale": 1.0, "load_case": load_case, "seismic": {"method": str(seismic.get("method", "response_seismic"))}})
    return loads, {"method": str(seismic.get("method", "response_seismic")), "kh": kh, "kv": kv, "source": source, "layer_count": len(layers), "generated_loads": len(loads)}


def _seismic_mapping(cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for source in (stage_cfg, cfg):
        for key in ("response_seismic", "seismic_response", "seismic", "earthquake"):
            value = source.get(key)
            if isinstance(value, Mapping):
                method = str(value.get("method", key)).lower().replace("-", "_")
                if method in {"response_seismic", "seismic_response", "pseudo_static", "earthquake", "response_acceleration"} or any(k in value for k in ("kh", "kv", "time_history", "horizontal_response_acceleration", "layers")):
                    return value
    return None


def _seismic_coefficients(seismic: Mapping[str, Any], time: float) -> tuple[float, float, str]:
    kh = float(seismic.get("kh", seismic.get("horizontal_seismic_coefficient", 0.0)) or 0.0)
    kv = float(seismic.get("kv", seismic.get("vertical_seismic_coefficient", 0.0)) or 0.0)
    unit = str(seismic.get("acceleration_unit", seismic.get("unit", "g"))).lower()
    if "horizontal_response_acceleration" in seismic or "ax" in seismic:
        kh += _acceleration_to_g(float(seismic.get("horizontal_response_acceleration", seismic.get("ax", 0.0)) or 0.0), unit)
    if "vertical_response_acceleration" in seismic or "ay" in seismic:
        kv += _acceleration_to_g(float(seismic.get("vertical_response_acceleration", seismic.get("ay", 0.0)) or 0.0), unit)
    hkh, hkv, source = _seismic_history_coefficients(seismic, time)
    return kh + hkh, kv + hkv, source


def _seismic_history_coefficients(seismic: Mapping[str, Any], time: float) -> tuple[float, float, str]:
    rows = _time_history_rows(seismic.get("time_history", seismic.get("acceleration_history", seismic.get("history"))))
    if not rows:
        return 0.0, 0.0, "constant"
    mode = str(seismic.get("time_history_mode", seismic.get("history_mode", "nearest"))).lower()
    unit = str(seismic.get("acceleration_unit", seismic.get("unit", "g"))).lower()
    if mode in {"envelope", "max", "maximum"}:
        kh = max((_history_row_k(row, "x", unit) for row in rows), key=abs, default=0.0)
        kv = max((_history_row_k(row, "y", unit) for row in rows), key=abs, default=0.0)
        return float(kh), float(kv), "time_history_envelope"
    if mode in {"linear", "interpolate", "interpolated"}:
        kh = _history_axis_k_at(rows, time, "x", unit)
        kv = _history_axis_k_at(rows, time, "y", unit)
        return kh, kv, "time_history_linear"
    selected = min(rows, key=lambda row: abs(float(row.get("time", row.get("t", 0.0)) or 0.0) - time))
    return _history_row_k(selected, "x", unit), _history_row_k(selected, "y", unit), "time_history_nearest"


def _elements_in_y_layer(mesh: Mesh2D, layer: Mapping[str, Any]) -> list[str]:
    ymin = float(layer.get("y_min", layer.get("bottom", -math.inf)))
    ymax = float(layer.get("y_max", layer.get("top", math.inf)))
    target_set = str(layer.get("element_set", layer.get("set", "")) or "")
    allowed = set(mesh.element_sets.get(target_set, [])) if target_set else {element.id for element in mesh.elements}
    ids: list[str] = []
    for element in mesh.elements:
        if element.id not in allowed or not element.active:
            continue
        conn = [mesh.node_index[nid] for nid in element.nodes]
        cy = float(np.mean(mesh.coords[conn, 1]))
        if ymin <= cy <= ymax:
            ids.append(element.id)
    return ids


def _apply_deactivation_stage(mesh: Mesh2D, active_element_ids: set[str], stage_cfg: Mapping[str, Any]) -> set[str]:
    target_ids = set(_target_elements(mesh, _deactivation_target_spec(stage_cfg)))
    unknown = sorted(target_ids - {element.id for element in mesh.elements})
    if unknown:
        raise FEM2DError(f"deactivation stage references unknown elements: {', '.join(unknown)}")
    return set(active_element_ids) - target_ids


def _interfaces_with_active(interfaces: list[Interface2D], active_ids: set[str]) -> list[Interface2D]:
    return [replace(interface, active=interface.id in active_ids) for interface in interfaces]


def _apply_stage_library_activity(active_ids: set[str], all_ids: set[str], stage_cfg: Mapping[str, Any], library: str) -> set[str]:
    active = set(active_ids) & set(all_ids)
    deactivate = _stage_library_action_ids(stage_cfg, library, "deactivate", all_ids)
    activate = _stage_library_action_ids(stage_cfg, library, "activate", all_ids)
    unknown = sorted((deactivate | activate) - set(all_ids))
    if unknown:
        raise FEM2DError(f"{library} stage activity references unknown ids: {', '.join(unknown)}")
    active.difference_update(deactivate)
    active.update(activate)
    return active


def _stage_library_action_ids(stage_cfg: Mapping[str, Any], library: str, action: str, all_ids: set[str]) -> set[str]:
    keys = _library_action_keys(library, action)
    ids: set[str] = set()
    for key in keys:
        if key in stage_cfg:
            ids.update(_ids_from_library_spec(stage_cfg[key], all_ids, library=library, action=action))
    for section_key in _library_activity_section_keys(library):
        section = stage_cfg.get(section_key)
        if isinstance(section, Mapping):
            for action_key in _generic_action_keys(action):
                if action_key in section:
                    ids.update(_ids_from_library_spec(section[action_key], all_ids, library=library, action=action))
    for generic_key in _generic_action_keys(action):
        generic = stage_cfg.get(generic_key)
        if isinstance(generic, Mapping):
            for target_key in _library_target_keys(library):
                if target_key in generic:
                    ids.update(_ids_from_library_spec(generic[target_key], all_ids, library=library, action=action))
    return ids


def _ids_from_library_spec(spec: Any, all_ids: set[str], *, library: str, action: str) -> set[str]:
    if spec is None:
        return set()
    if isinstance(spec, str):
        text = spec.strip()
        return set(all_ids) if text.lower() in {"*", "all"} else ({text} if text else set())
    if isinstance(spec, Mapping):
        if bool(spec.get("all", False)):
            return set(all_ids)
        ids: set[str] = set()
        for key in ("id", "ids", "element", "elements", "element_id", "element_ids"):
            if key in spec:
                ids.update(_ids_from_library_spec(spec[key], all_ids, library=library, action=action))
        for key in _library_target_keys(library):
            if key in spec:
                ids.update(_ids_from_library_spec(spec[key], all_ids, library=library, action=action))
        if not ids:
            for key, value in spec.items():
                skey = str(key)
                if skey not in all_ids:
                    continue
                if isinstance(value, Mapping):
                    active_value = value.get("active")
                    if active_value is None or bool(active_value) == (action == "activate"):
                        ids.add(skey)
                elif bool(value):
                    ids.add(skey)
        return ids
    values = spec if isinstance(spec, list) else [spec]
    ids = {str(value).strip() for value in values if str(value).strip()}
    if any(value.lower() in {"*", "all"} for value in ids):
        return set(all_ids)
    return ids


def _library_action_keys(library: str, action: str) -> tuple[str, ...]:
    if library == "structural":
        return (
            ("structural_activate", "activate_structural", "structural_birth", "birth_structural", "add_structural", "line_activate", "line_birth")
            if action == "activate"
            else ("structural_deactivate", "deactivate_structural", "structural_death", "death_structural", "remove_structural", "line_deactivate", "line_death")
        )
    return (
        ("interface_activate", "activate_interface", "joint_activate", "activate_joint", "interface_birth", "joint_birth")
        if action == "activate"
        else ("interface_deactivate", "deactivate_interface", "joint_deactivate", "deactivate_joint", "interface_death", "joint_death")
    )


def _generic_action_keys(action: str) -> tuple[str, ...]:
    return ("activate", "birth", "add") if action == "activate" else ("deactivate", "death", "remove")


def _library_activity_section_keys(library: str) -> tuple[str, ...]:
    if library == "structural":
        return ("structural_activity", "structural_stage", "line_element_activity")
    return ("interface_activity", "joint_activity", "interface_stage")


def _library_target_keys(library: str) -> tuple[str, ...]:
    if library == "structural":
        return ("structural", "structural_elements", "line_elements", "frame_elements", "beams", "bars", "springs")
    return ("interface", "interfaces", "interface_elements", "joints", "joint_elements")


def _deactivation_target_spec(stage_cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    target = stage_cfg.get("target")
    if isinstance(target, Mapping):
        return target
    for key in ("deactivate", "excavation", "death", "remove"):
        value = stage_cfg.get(key)
        if isinstance(value, Mapping):
            return value
    if any(key in stage_cfg for key in ("element", "elements", "element_id", "element_ids", "set")):
        return stage_cfg
    return {"elements": []}


def _attach_structural_extra_dofs(result: StageResult2D, mesh: Mesh2D, structural_elements: list[StructuralElement2D] | None) -> None:
    labels = structural_extra_dof_labels(mesh, structural_elements)
    if labels:
        result.solver_info["extra_dofs"] = {str(dof): label for dof, label in labels.items()}


def _solve_sparse_with_constraints(
    matrix: csr_matrix,
    rhs: np.ndarray,
    fixed_values: Mapping[int, float],
    *,
    stage_name: str,
    solver: Mapping[str, Any] | None,
) -> np.ndarray:
    return _solve_sparse_with_constraints_core(matrix, rhs, fixed_values, stage_name=stage_name, solver=solver)


def _linear_solver_settings(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    return _linear_solver_settings_core(solver)


def _newton_settings(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    return _newton_settings_core(solver)


def _tangent_method(solver: Mapping[str, Any] | None) -> str:
    return _tangent_method_core(solver)


def _increment_settings(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    return _increment_settings_core(solver)


def _solver_without_increments(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    return _solver_without_increments_core(solver)


def _large_deformation_settings(
    solver: Mapping[str, Any] | None,
    stage_config: Mapping[str, Any] | None = None,
    *,
    default_enabled: bool = False,
) -> dict[str, Any]:
    return _large_deformation_settings_core(solver, stage_config, default_enabled=default_enabled)


def _solver_without_large_deformation(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    return _solver_without_large_deformation_core(solver)


def _scale_loads(loads: Any, factor: float) -> list[Any]:
    return _scale_loads_core(loads, factor)


def _scale_load_item(load: Mapping[str, Any], factor: float) -> dict[str, Any]:
    return _scale_load_item_core(load, factor)


def _scale_boundary_conditions(boundary_conditions: Any, factor: float) -> list[Any]:
    return _scale_boundary_conditions_core(boundary_conditions, factor)


def _has_nonlinear_interfaces(interfaces: list[Interface2D] | None) -> bool:
    return _has_nonlinear_interfaces_core(interfaces)


def _check_linear_solution(stage_name: str, method: str, x: np.ndarray) -> None:
    _linear_check_linear_solution(stage_name, method, x)


def _srm_factors(cfg: Mapping[str, Any]) -> list[float]:
    return _srm_factors_core(cfg)


def _plastic_ratio(result: StageResult2D) -> float:
    if result.element_results:
        return _plastic_ratio_core(result)
    solver_info = result.solver_info if isinstance(result.solver_info, Mapping) else {}
    raw_ratio = solver_info.get("plastic_ratio")
    if raw_ratio is not None:
        try:
            return float(raw_ratio)
        except (TypeError, ValueError):
            pass
    return _plastic_ratio_from_state(result.plastic_state, result.active_elements)


def _plastic_ratio_from_state(plastic_state: Mapping[str, PlasticState2D] | None, active_elements: list[str] | tuple[str, ...] | set[str]) -> float:
    active = {str(element_id) for element_id in active_elements}
    if not active or not plastic_state:
        return 0.0
    plastic_elements: set[str] = set()
    for raw_key, state in plastic_state.items():
        element_id = str(raw_key).rsplit(":", 1)[0]
        if element_id not in active:
            continue
        if _plastic_state_is_plastic(state):
            plastic_elements.add(element_id)
    return len(plastic_elements) / len(active)


def _plastic_ratio_from_state_or_array(
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    active_elements: list[str] | tuple[str, ...] | set[str],
) -> float:
    if plastic_state:
        return _plastic_ratio_from_state(plastic_state, active_elements)
    if plastic_state_cache is not None:
        return plastic_state_cache.plastic_ratio(active_elements)
    return 0.0


def _plastic_state_is_plastic(state: PlasticState2D) -> bool:
    try:
        if bool(state.state_vars.get("plastic", False)):
            return True
        if float(state.kappa) > 0.0:
            return True
        return bool(np.linalg.norm(np.asarray(state.plastic_strain, dtype=float)) > 0.0)
    except (AttributeError, TypeError, ValueError):
        return False


__all__ = [
    "solve_plane_strain_config",
    "solve_axisymmetric_config",
    "solve_plane_strain_stage",
    "solve_axisymmetric_stage",
    "solve_axisymmetric_srm_stage",
    "solve_axisymmetric_coupled_up_stage",
    "solve_axisymmetric_riks_stage",
    "solve_axisymmetric_arc_length_stage",
    "solve_dynamic_time_history_stage",
    "solve_srm_stage",
    "solve_large_deformation_stage",
    "solve_incremental_stage",
    "solve_coupled_up_stage",
    "solve_riks_stage",
    "solve_arc_length_stage",
    "SmallDeformationStepCache",
    "AxisymmetricStepCache",
    "ConsolidationStepCache",
    "DynamicMassStepCache",
    "InitialStressArrayCache",
    "build_small_deformation_step_cache",
    "build_dynamic_mass_step_cache",
    "build_axisymmetric_step_cache",
    "build_initial_stress_array_cache",
    "initial_stress_array_cache_info",
    "solve_linear_system",
    "solve_reduced_linear_system",
    "clear_linear_factor_cache",
    "linear_factor_cache_info",
    "solve_nonlinear_system",
    "assemble_global_stiffness",
    "assemble_mass_matrix",
    "build_mass_matrix_assembly_cache",
    "assemble_mass_matrix_cached",
    "assemble_axisymmetric_stiffness",
    "assemble_axisymmetric_algorithmic_tangent_stiffness",
    "assemble_axisymmetric_tangent_and_internal_force",
    "assemble_algorithmic_tangent_stiffness",
    "assemble_tangent_and_internal_force",
    "assemble_load_vector",
    "assemble_axisymmetric_load_vector",
    "assemble_axisymmetric_biot_coupling_matrix",
    "assemble_axisymmetric_pressure_matrices",
    "assemble_axisymmetric_pressure_boundary_terms",
    "build_pressure_matrix_assembly_cache",
    "assemble_pressure_matrices_cached",
    "build_biot_coupling_assembly_cache",
    "assemble_biot_coupling_matrix_cached",
    "build_pore_pressure_load_cache",
    "assemble_pore_pressure_load_cached",
    "build_pressure_boundary_term_cache",
    "assemble_pressure_boundary_terms_cached",
    "assemble_internal_force",
    "assemble_pore_pressure_load",
    "assemble_biot_coupling_matrix",
    "solve_consolidation_pressure",
    "assemble_pressure_matrices",
    "assemble_pressure_boundary_terms",
    "assemble_liquefaction_pressure_terms",
    "collect_constraints",
    "assemble_mpc_penalty",
    "mpc_violation",
    "solve_linear_system_with_mpc_elimination",
    "solve_linear_system_with_mpc_lagrange",
    "compute_axisymmetric_element_results",
    "compute_axisymmetric_element_results_and_state",
    "compute_axisymmetric_integration_point_results",
    "assemble_axisymmetric_internal_force",
    "solve_axisymmetric_nonlinear_system",
    "_axisymmetric_edge_measure",
    "_add_body_weight",
    "_add_edge_traction",
    "_geostatic_initial_stresses",
    "_stage_type",
    "_apply_deactivation_stage",
    "_deactivation_target_spec",
    "_add_inactive_node_constraints",
    "_initial_pore_pressure",
    "_collect_pressure_constraints",
    "_solve_scalar_constraints",
    "_solve_sparse_with_constraints",
    "_linear_solver_settings",
    "_newton_settings",
    "_tangent_method",
    "_increment_settings",
    "_solver_without_increments",
    "_large_deformation_settings",
    "_solver_without_large_deformation",
    "_scale_loads",
    "_scale_boundary_conditions",
    "_has_nonlinear_interfaces",
    "_check_linear_solution",
    "_srm_factors",
    "_plastic_ratio",
]

