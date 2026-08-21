"""Large-deformation helpers for the 2D FEM core.

This module intentionally keeps the geometry-update operations array-based so
they can be exercised through NumPy first and accelerated with Numba without
changing solver orchestration code.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from itertools import count
from typing import Any, Iterator, Mapping

import numpy as np

from .fem2d_types import FEM2DError, Mesh2D, njit
from .fem2d_constraints import _add_inactive_node_constraints, collect_constraints
from .fem2d_performance_contract import common_solver_info_fields
from .fem2d_plastic_batch import build_plastic_element_blocks
from .fem2d_pressure import PorePressureLoadAssemblyCache, build_pore_pressure_load_cache
from .fem2d_structural import structural_total_dofs
from .fem2d_structural_assembly import GlobalStiffnessAssemblyCache, build_global_stiffness_assembly_cache
from .fem2d_types import ElasticPlaneStrainMaterial
from .fem2d_utils import _dofs_from_node_indices, _element_node_indices
from .reduced_matrix_cache import ReducedMatrixCache, build_reduced_matrix_cache_from_pattern


_LARGE_DEFORMATION_CACHE_IDS = count(1)


@dataclass(frozen=True)
class LargeDeformationStepCache:
    ndof: int
    constrained: dict[int, float]
    free_dofs: np.ndarray
    fixed_dofs: np.ndarray
    active_elements: list[str]
    stiffness_cache: GlobalStiffnessAssemblyCache | None = None
    active_element_indices: np.ndarray | None = None
    element_connectivity: np.ndarray | None = None
    element_dofs: np.ndarray | None = None
    element_node_counts: np.ndarray | None = None
    element_type_codes: np.ndarray | None = None
    material_ids: np.ndarray | None = None
    integration_codes: np.ndarray | None = None
    state_point_counts: np.ndarray | None = None
    material_id_map: dict[str, int] | None = None
    plastic_blocks: tuple[dict[str, Any], ...] = ()
    hydro_cache: dict[str, Any] | None = None
    pore_pressure_load_cache: PorePressureLoadAssemblyCache | None = None
    topology_cache_id: str = ""
    reuse_scope: str = "stage"
    shared_across_srm_factors: bool = False
    load_scale: float = 1.0
    base_load_vector: np.ndarray | None = None
    load_vector_cache_reason: str = ""
    load_vector_assembly_cache: Any | None = None
    load_vector_assembly_cache_reason: str = ""
    mpc_penalty_matrix: Any | None = None
    mpc_penalty_vector: np.ndarray | None = None
    mpc_info: dict[str, Any] | None = None
    mpc_penalty_cache_reason: str = ""
    reduced_matrix_cache: ReducedMatrixCache | None = None

    def with_constraint_scale(self, scale: float) -> "LargeDeformationStepCache":
        factor = float(scale)
        return replace(self, constrained={int(dof): float(value) * factor for dof, value in self.constrained.items()}, load_scale=factor)

    def solver_info(self) -> dict[str, Any]:
        stiffness_info = {} if self.stiffness_cache is None else self.stiffness_cache.info()
        connectivity_shape = [] if self.element_connectivity is None else [int(v) for v in self.element_connectivity.shape]
        dof_shape = [] if self.element_dofs is None else [int(v) for v in self.element_dofs.shape]
        integration_counts = _code_counts(self.integration_codes, {1: "FULL", 2: "SRI", 3: "B-BAR"})
        element_type_counts = _code_counts(self.element_type_codes, {3: "TRI3", 4: "QUAD4", 6: "TRI6", 8: "QUAD8"})
        return {
            "enabled": True,
            "cache_kind": "large_deformation_step_cache",
            "topology_cache_id": self.topology_cache_id,
            "reuse_scope": self.reuse_scope,
            "shared_across_srm_factors": bool(self.shared_across_srm_factors),
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
            "reduced_matrix_cached": self.reduced_matrix_cache is not None,
            "reduced_matrix_cache": {"enabled": False} if self.reduced_matrix_cache is None else self.reduced_matrix_cache.info(),
            "active_element_array_size": 0 if self.active_element_indices is None else int(self.active_element_indices.size),
            "connectivity_shape": connectivity_shape,
            "element_dof_shape": dof_shape,
            "element_type_counts": element_type_counts,
            "integration_counts": integration_counts,
            "material_id_count": 0 if self.material_id_map is None else len(self.material_id_map),
            "state_point_count_max": 0 if self.state_point_counts is None or self.state_point_counts.size == 0 else int(np.max(self.state_point_counts)),
            "plastic_blocks": [dict(block) for block in self.plastic_blocks],
            "plastic_state_layout": {
                "order": "element_major_integration_point_minor",
                "state_components": ["plastic_strain_x", "plastic_strain_y", "plastic_strain_z", "plastic_strain_gamma_xy", "kappa"],
                "quad4_points": 4,
                "quad8_full_points": 9,
                "quad8_sri_points": 13,
                "quad8_bbar_points": 9,
            },
            "cache_inputs": ["load_increment", "strength_factor", "tension_cutoff", "initial_stress", "pore_pressure"],
            "hydro_cache": {
                **dict(self.hydro_cache or {}),
                "pore_pressure_load_cache": {"enabled": False} if self.pore_pressure_load_cache is None else self.pore_pressure_load_cache.info(),
            },
            "pore_pressure_load_cached": self.pore_pressure_load_cache is not None,
            "pore_pressure_load_cache": {"enabled": False} if self.pore_pressure_load_cache is None else self.pore_pressure_load_cache.info(),
            "factor_invariant_load_mpc_cache": {
                "enabled": self.base_load_vector is not None or self.load_vector_assembly_cache is not None or (self.mpc_penalty_matrix is not None and self.mpc_penalty_vector is not None and self.mpc_info is not None),
                "load_vector_cached": self.base_load_vector is not None,
                "load_vector_template_cached": self.load_vector_assembly_cache is not None,
                "load_vector_size": 0 if self.base_load_vector is None else int(self.base_load_vector.size),
                "load_scale": float(self.load_scale),
                "load_vector_reason": self.load_vector_cache_reason,
                "load_vector_template_reason": self.load_vector_assembly_cache_reason,
                "load_vector_template": {"enabled": False} if self.load_vector_assembly_cache is None else self.load_vector_assembly_cache.info(),
                "mpc_penalty_cached": self.mpc_penalty_matrix is not None and self.mpc_penalty_vector is not None and self.mpc_info is not None,
                "mpc_equations": 0 if self.mpc_info is None else int(self.mpc_info.get("count", 0) or 0),
                "mpc_penalty_reason": self.mpc_penalty_cache_reason,
                "constraint_scale_template_cached": True,
            },
        }


@njit(cache=True)
def _fill_updated_coords_numba(coords: np.ndarray, displacement: np.ndarray, scale: float, out: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    for i in range(n):
        out[i, 0] = coords[i, 0] + scale * displacement[2 * i]
        out[i, 1] = coords[i, 1] + scale * displacement[2 * i + 1]
    return out


@njit(cache=True)
def _updated_coords_numba(coords: np.ndarray, displacement: np.ndarray, scale: float) -> np.ndarray:
    out = np.empty_like(coords)
    return _fill_updated_coords_numba(coords, displacement, scale, out)


@njit(cache=True)
def _max_displacement_norm_numba(displacement: np.ndarray) -> float:
    max_value = 0.0
    n = displacement.shape[0] // 2
    for i in range(n):
        ux = displacement[2 * i]
        uy = displacement[2 * i + 1]
        value = (ux * ux + uy * uy) ** 0.5
        if value > max_value:
            max_value = value
    return max_value


def updated_coords_vectorized(coords: np.ndarray, displacement: np.ndarray, *, scale: float = 1.0) -> np.ndarray:
    coords_arr = np.asarray(coords, dtype=float)
    disp = _validated_displacement(displacement, len(coords_arr))
    out = np.empty_like(coords_arr, dtype=float)
    return _fill_updated_coords_vectorized(coords_arr, disp, out, scale=float(scale))


def _fill_updated_coords_vectorized(coords: np.ndarray, displacement: np.ndarray, out: np.ndarray, *, scale: float) -> np.ndarray:
    out[:, 0] = coords[:, 0] + float(scale) * displacement[0::2]
    out[:, 1] = coords[:, 1] + float(scale) * displacement[1::2]
    return out


def fill_updated_coords(
    coords: np.ndarray,
    displacement: np.ndarray,
    *,
    out: np.ndarray | None = None,
    scale: float = 1.0,
    backend: str = "auto",
) -> np.ndarray:
    coords_arr = np.asarray(coords, dtype=float)
    disp = _validated_displacement(displacement, len(coords_arr))
    out_arr = np.empty_like(coords_arr, dtype=float) if out is None else np.asarray(out)
    if out_arr.shape != coords_arr.shape:
        raise FEM2DError(f"updated coordinate buffer must have shape {coords_arr.shape}, got {out_arr.shape}")
    if not np.issubdtype(out_arr.dtype, np.floating):
        raise FEM2DError("updated coordinate buffer must use a floating dtype")
    if str(backend or "auto").lower() in {"numba", "auto"}:
        return _fill_updated_coords_numba(
            np.ascontiguousarray(coords_arr, dtype=np.float64),
            np.ascontiguousarray(disp, dtype=np.float64),
            float(scale),
            out_arr,
        )
    return _fill_updated_coords_vectorized(coords_arr, disp, out_arr, scale=float(scale))


def updated_coords_fast(coords: np.ndarray, displacement: np.ndarray, *, scale: float = 1.0, backend: str = "auto") -> np.ndarray:
    coords_arr = np.asarray(coords, dtype=float)
    disp = _validated_displacement(displacement, len(coords_arr))
    if str(backend or "auto").lower() in {"numba", "auto"}:
        return _updated_coords_numba(np.ascontiguousarray(coords_arr, dtype=np.float64), np.ascontiguousarray(disp, dtype=np.float64), float(scale))
    return updated_coords_vectorized(coords_arr, disp, scale=scale)


def mesh_with_updated_coords(mesh: Mesh2D, displacement: np.ndarray, *, scale: float = 1.0, backend: str = "auto") -> Mesh2D:
    coords = updated_coords_fast(mesh.coords, displacement, scale=scale, backend=backend)
    return replace(mesh, coords=coords)


@contextmanager
def temporary_mesh_coords(mesh: Mesh2D, coords: np.ndarray) -> Iterator[Mesh2D]:
    coords_arr = np.asarray(coords, dtype=float)
    if coords_arr.shape != mesh.coords.shape:
        raise FEM2DError(f"temporary mesh coordinates must have shape {mesh.coords.shape}, got {coords_arr.shape}")
    original = mesh.coords
    mesh.coords = coords_arr
    try:
        yield mesh
    finally:
        mesh.coords = original


def max_displacement_norm(displacement: np.ndarray, *, backend: str = "auto") -> float:
    disp = np.asarray(displacement, dtype=float).reshape(-1)
    if disp.size == 0:
        return 0.0
    if disp.size % 2:
        raise FEM2DError("large deformation displacement vector must contain ux,uy pairs")
    if str(backend or "auto").lower() in {"numba", "auto"}:
        return float(_max_displacement_norm_numba(np.ascontiguousarray(disp, dtype=np.float64)))
    pairs = disp.reshape((-1, 2))
    return float(np.max(np.linalg.norm(pairs, axis=1))) if len(pairs) else 0.0


def mesh_diagonal_length(mesh: Mesh2D) -> float:
    if not mesh.node_ids:
        return 0.0
    coords = np.asarray(mesh.coords, dtype=float)
    width = float(np.max(coords[:, 0]) - np.min(coords[:, 0]))
    height = float(np.max(coords[:, 1]) - np.min(coords[:, 1]))
    return float((width * width + height * height) ** 0.5)


def large_deformation_kernel_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.large_deformation.v1",
        "formulation": "updated_lagrangian_geometry_update",
        "vectorized_functions": ["updated_coords_vectorized", "fill_updated_coords", "max_displacement_norm"],
        "numba_functions": ["_updated_coords_numba", "_fill_updated_coords_numba", "_max_displacement_norm_numba"],
        "topology_cache": "LargeDeformationStepCache precomputes constraints, free/fixed dofs, active elements, fixed connectivity/dof/material/integration arrays, stiffness scatter pattern, reduced free-DOF extraction, plastic block metadata, pore-pressure load scatter blocks, and reusable load/MPC templates when they do not depend on updated geometry; SRM can reuse one cache instance across strength-factor trials",
        "geometry_update_cache": "solve_large_deformation_stage reuses a coordinate buffer and temporarily swaps Mesh2D.coords instead of rebuilding Mesh objects for each updated-Lagrangian increment",
        "adaptive_steps": "steps is treated as the initial subdivision; accepted increments can grow or cut back by convergence and deformation size",
        "state": "geometry update kernels are isolated from solver orchestration",
        "common_solver_info": ["geometry_mode", "element_type", "integration", "material_model", "batched_elements", "fallback_count", "fallback_reasons"],
    }


def large_deformation_settings(
    solver: Mapping[str, Any] | None,
    stage_config: Mapping[str, Any] | None = None,
    *,
    default_enabled: bool = False,
) -> dict[str, Any]:
    raw_solver = solver if isinstance(solver, Mapping) else {}
    raw_stage = stage_config if isinstance(stage_config, Mapping) else {}
    raw = raw_solver.get("large_deformation", raw_solver.get("finite_deformation", {}))
    if not raw:
        raw = raw_stage.get("large_deformation", raw_stage.get("finite_deformation", raw_stage.get("updated_lagrangian", {})))
    if isinstance(raw, bool):
        enabled = raw
        raw_map: Mapping[str, Any] = {}
    elif isinstance(raw, (int, float)):
        enabled = True
        raw_map = {"steps": int(raw)}
    elif isinstance(raw, Mapping):
        enabled = bool(raw.get("enabled", default_enabled))
        raw_map = raw
    else:
        enabled = default_enabled
        raw_map = {}
    inc = raw_solver.get("increments", raw_solver.get("increment", {}))
    inc_steps = int(inc.get("steps", 1)) if isinstance(inc, Mapping) else (int(inc) if isinstance(inc, (int, float)) and not isinstance(inc, bool) else 1)
    steps = int(raw_map.get("steps", raw_map.get("increments", raw_map.get("load_steps", max(inc_steps, 8)))))
    if steps <= 0:
        raise FEM2DError("large_deformation.steps must be positive")
    initial_step = float(raw_map.get("initial_step", raw_map.get("initial_load_increment", 1.0 / float(steps))))
    max_step = float(raw_map.get("max_step", raw_map.get("max_load_increment", min(1.0, max(initial_step, 0.5)))))
    min_step = float(raw_map.get("min_step", raw_map.get("min_load_increment", initial_step / 16.0)))
    growth_factor = float(raw_map.get("growth_factor", raw_map.get("step_growth_factor", 1.5)))
    cutback_factor = float(raw_map.get("cutback_factor", raw_map.get("step_cutback_factor", 0.5)))
    max_cutbacks = int(raw_map.get("max_cutbacks", raw_map.get("cutbacks", max(steps * 4, 8))))
    max_adaptive_steps = int(raw_map.get("max_adaptive_steps", raw_map.get("max_steps", max(steps * 8, steps + 8))))
    grow_below_ratio = float(raw_map.get("grow_below_ratio", raw_map.get("small_increment_ratio", 0.01)))
    shrink_above_ratio = float(raw_map.get("shrink_above_ratio", raw_map.get("large_increment_ratio", 0.05)))
    grow_below_iterations = int(raw_map.get("grow_below_iterations", 2))
    shrink_above_iterations = int(raw_map.get("shrink_above_iterations", 8))
    if initial_step <= 0.0 or max_step <= 0.0 or min_step <= 0.0:
        raise FEM2DError("large_deformation adaptive step sizes must be positive")
    if min_step > max_step:
        raise FEM2DError("large_deformation.min_step must not exceed max_step")
    if growth_factor < 1.0:
        raise FEM2DError("large_deformation.growth_factor must be at least 1.0")
    if not (0.0 < cutback_factor < 1.0):
        raise FEM2DError("large_deformation.cutback_factor must satisfy 0 < factor < 1")
    if max_cutbacks < 0 or max_adaptive_steps <= 0:
        raise FEM2DError("large_deformation max_cutbacks must be non-negative and max_adaptive_steps positive")
    backend = str(raw_map.get("backend", raw_map.get("kernel", "auto")) or "auto").lower()
    if backend not in {"auto", "numba", "vectorized", "numpy"}:
        raise FEM2DError("large_deformation.backend must be auto, numba, or vectorized")
    return {
        "enabled": enabled,
        "steps": steps,
        "backend": "vectorized" if backend == "numpy" else backend,
        "adaptive_steps": bool(raw_map.get("adaptive_steps", raw_map.get("adaptive", True))),
        "initial_step": min(max(initial_step, min_step), max_step),
        "min_step": min_step,
        "max_step": max_step,
        "growth_factor": growth_factor,
        "cutback_factor": cutback_factor,
        "max_cutbacks": max_cutbacks,
        "max_adaptive_steps": max_adaptive_steps,
        "grow_below_ratio": grow_below_ratio,
        "shrink_above_ratio": shrink_above_ratio,
        "grow_below_iterations": grow_below_iterations,
        "shrink_above_iterations": shrink_above_iterations,
        "formulation": str(raw_map.get("formulation", "updated_lagrangian_geometry_update")),
        "update_geometry": bool(raw_map.get("update_geometry", True)),
        "precompute_topology": bool(raw_map.get("precompute_topology", True)),
        "precompute_stiffness_pattern": bool(raw_map.get("precompute_stiffness_pattern", raw_map.get("precompute_sparse_pattern", True))),
        "skip_intermediate_postprocessing": bool(raw_map.get("skip_intermediate_postprocessing", True)),
        "strain_measure": str(raw_map.get("strain_measure", "incremental_small_strain")),
        "material_frame": str(raw_map.get("material_frame", "existing_plane_strain_materials")),
    }


def solver_without_large_deformation(solver: Mapping[str, Any] | None) -> dict[str, Any]:
    clean = dict(solver or {})
    clean.pop("large_deformation", None)
    clean.pop("finite_deformation", None)
    clean.pop("updated_lagrangian", None)
    return clean


def build_large_deformation_step_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    boundary_conditions: Any,
    *,
    interfaces: list[Any] | None = None,
    structural_elements: list[Any] | None = None,
    precompute_stiffness_pattern: bool = True,
    reuse_scope: str = "stage",
    shared_across_srm_factors: bool = False,
) -> LargeDeformationStepCache:
    ndof = structural_total_dofs(mesh, structural_elements)
    constrained = collect_constraints(mesh, boundary_conditions, structural_elements=structural_elements)
    _add_inactive_node_constraints(mesh, constrained, interfaces=interfaces, structural_elements=structural_elements)
    fixed = np.asarray(sorted(constrained), dtype=np.int64)
    if fixed.size and (int(fixed[0]) < 0 or int(fixed[-1]) >= ndof):
        raise FEM2DError("large deformation cached constraint dof is outside the model dof range")
    mask = np.zeros(ndof, dtype=bool)
    if fixed.size:
        mask[fixed] = True
    free = np.nonzero(~mask)[0].astype(np.int64, copy=False)
    stiffness_cache = (
        build_global_stiffness_assembly_cache(mesh, materials, interfaces=interfaces, structural_elements=structural_elements)
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
            source="large_deformation_stiffness_pattern",
        )
        if stiffness_cache is not None
        else None
    )
    active_elements = [element.id for element in mesh.elements if element.active]
    topology_arrays = _large_deformation_topology_arrays(mesh, materials)
    plastic_blocks = tuple(block.solver_info() for block in build_plastic_element_blocks(mesh, materials))
    return LargeDeformationStepCache(
        ndof=ndof,
        constrained=dict(constrained),
        free_dofs=free,
        fixed_dofs=fixed,
        active_elements=active_elements,
        stiffness_cache=stiffness_cache,
        active_element_indices=topology_arrays["active_element_indices"],
        element_connectivity=topology_arrays["element_connectivity"],
        element_dofs=topology_arrays["element_dofs"],
        element_node_counts=topology_arrays["element_node_counts"],
        element_type_codes=topology_arrays["element_type_codes"],
        material_ids=topology_arrays["material_ids"],
        integration_codes=topology_arrays["integration_codes"],
        state_point_counts=topology_arrays["state_point_counts"],
        material_id_map=topology_arrays["material_id_map"],
        plastic_blocks=plastic_blocks,
        hydro_cache=_large_deformation_hydro_cache(mesh, materials),
        pore_pressure_load_cache=build_pore_pressure_load_cache(mesh, structural_elements=structural_elements),
        topology_cache_id=f"large-deformation-step-cache-{next(_LARGE_DEFORMATION_CACHE_IDS)}",
        reuse_scope=str(reuse_scope or "stage"),
        shared_across_srm_factors=bool(shared_across_srm_factors),
        reduced_matrix_cache=reduced_matrix_cache,
    )


def large_deformation_common_solver_info(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    *,
    geometry_mode: str,
    batched_elements: int = 0,
    fallback_reasons: list[str] | tuple[str, ...] | None = None,
    hydro_coupled: bool = False,
) -> dict[str, Any]:
    return common_solver_info_fields(
        mesh,
        materials,
        geometry_mode=geometry_mode,
        batched_elements=batched_elements,
        fallback_reasons=fallback_reasons,
        hydro_coupled=hydro_coupled,
    )


def _validated_displacement(displacement: np.ndarray, node_count: int) -> np.ndarray:
    disp = np.asarray(displacement, dtype=float).reshape(-1)
    expected = 2 * int(node_count)
    if disp.size != expected:
        raise FEM2DError(f"large deformation displacement size must be {expected}, got {disp.size}")
    if not np.all(np.isfinite(disp)):
        raise FEM2DError("large deformation displacement contains non-finite values")
    return disp


def _large_deformation_topology_arrays(mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial]) -> dict[str, Any]:
    active = [(index, element) for index, element in enumerate(mesh.elements) if element.active]
    max_nodes = max((len(element.nodes) for _index, element in active), default=0)
    max_dofs = max_nodes * 2
    n = len(active)
    conn = np.full((n, max_nodes), -1, dtype=np.int64)
    dofs = np.full((n, max_dofs), -1, dtype=np.int64)
    node_counts = np.zeros(n, dtype=np.int64)
    type_codes = np.zeros(n, dtype=np.int64)
    material_ids = np.zeros(n, dtype=np.int64)
    integration_codes = np.zeros(n, dtype=np.int64)
    state_point_counts = np.zeros(n, dtype=np.int64)
    material_names = sorted({element.material for _index, element in active})
    material_id_map = {name: idx for idx, name in enumerate(material_names)}
    for row, (element_index, element) in enumerate(active):
        indices = _element_node_indices(element.nodes, mesh.node_index)
        element_dofs = _dofs_from_node_indices(indices)
        conn[row, : len(indices)] = indices
        dofs[row, : len(element_dofs)] = element_dofs
        node_counts[row] = len(indices)
        type_codes[row] = _element_type_code(element.type)
        material_ids[row] = material_id_map[element.material]
        integration_codes[row] = _integration_code(element.integration)
        state_point_counts[row] = _state_point_count(element.type, element.integration, materials.get(element.material))
    return {
        "active_element_indices": np.asarray([index for index, _element in active], dtype=np.int64),
        "element_connectivity": conn,
        "element_dofs": dofs,
        "element_node_counts": node_counts,
        "element_type_codes": type_codes,
        "material_ids": material_ids,
        "integration_codes": integration_codes,
        "state_point_counts": state_point_counts,
        "material_id_map": material_id_map,
    }


def _large_deformation_hydro_cache(mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial]) -> dict[str, Any]:
    pressure_dofs = len(mesh.node_ids)
    displacement_dofs = len(mesh.node_ids) * 2
    coupled = any(_material_hydro_enabled(material) for material in materials.values())
    return {
        "enabled": coupled,
        "displacement_dofs": displacement_dofs,
        "pressure_dofs": pressure_dofs,
        "blocks": ["kuu", "kup", "kpu", "kpp"],
        "boundary_cache_slots": ["drained", "undrained", "flow", "pressure"],
        "time_integration_slots": ["dt", "theta", "storage", "compressibility", "biot"],
    }


def _material_hydro_enabled(material: ElasticPlaneStrainMaterial) -> bool:
    params = material.advanced_params or {}
    return any(str(key).lower().strip() in {"permeability", "kx", "ky", "biot", "storage", "compressibility"} for key in params)


def _element_type_code(value: str) -> int:
    text = str(value).upper().strip()
    if text == "QUAD4":
        return 4
    if text == "QUAD8":
        return 8
    if text == "TRI3":
        return 3
    if text == "TRI6":
        return 6
    return 0


def _integration_code(value: str) -> int:
    text = str(value).upper().strip().replace("_", "-")
    if text == "SRI":
        return 2
    if text in {"B-BAR", "BBAR"}:
        return 3
    return 1


def _state_point_count(element_type: str, integration: str, material: ElasticPlaneStrainMaterial | None) -> int:
    etype = str(element_type).upper().strip()
    mode = str(integration).upper().strip().replace("_", "-")
    if etype == "QUAD8":
        return 13 if mode == "SRI" else 9
    if etype == "TRI6":
        return 3
    return 4 if material is not None and material.is_plastic else 0


def _code_counts(values: np.ndarray | None, labels: Mapping[int, str]) -> dict[str, int]:
    if values is None:
        return {}
    out: dict[str, int] = {}
    for value in np.asarray(values, dtype=np.int64).ravel():
        label = labels.get(int(value), str(int(value)))
        out[label] = out.get(label, 0) + 1
    return out


__all__ = [
    "large_deformation_kernel_contract",
    "large_deformation_common_solver_info",
    "large_deformation_settings",
    "build_large_deformation_step_cache",
    "LargeDeformationStepCache",
    "max_displacement_norm",
    "mesh_diagonal_length",
    "mesh_with_updated_coords",
    "solver_without_large_deformation",
    "fill_updated_coords",
    "temporary_mesh_coords",
    "updated_coords_fast",
    "updated_coords_vectorized",
    "_max_displacement_norm_numba",
    "_fill_updated_coords_numba",
    "_updated_coords_numba",
]
