"""Hydraulic pressure matrix assembly helpers for 2D FEM solvers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .fem2d_elements import (
    _QUAD4_MODE_BBAR,
    _quad4_axisymmetric_biot_matrix_fast,
    _quad4_axisymmetric_biot_matrix_numba,
    _quad4_axisymmetric_pressure_matrices_fast,
    _quad4_axisymmetric_pressure_matrices_numba,
    _quad4_biot_matrix_fast,
    _quad4_biot_matrix_numba,
    _quad4_mode_code,
    _quad4_pressure_matrices_fast,
    _quad4_pressure_matrices_numba,
    _quad4_shape_grad_numba,
    _quad8_axisymmetric_biot_matrix_fast,
    _quad8_axisymmetric_biot_matrix_numba,
    _quad8_axisymmetric_pressure_matrices_fast,
    _quad8_axisymmetric_pressure_matrices_numba,
    _quad8_biot_matrix_fast,
    _quad8_biot_matrix_numba,
    _quad8_gp_full,
    _quad8_pressure_matrices_fast,
    _quad8_pressure_matrices_numba,
    _quad8_shape_grad_numba,
    axisymmetric_strain_displacement_matrix,
    integration_points,
    shape_functions,
    strain_displacement_matrix,
)
from .fem2d_hydro import _collect_pressure_constraints, _initial_pore_pressure
from .fem2d_linear_solver import solve_sparse_with_constraints as _solve_sparse_with_constraints_core
from .fem2d_materials import _equivalent_shear_strain, _param_float
from .fem2d_mesh import _edge_consistent_robin_matrix, _edge_length, _edge_lumped_weights, _pressure_edges
from .fem2d_result_annotations import _liquefaction_effective_stress_reference, _material_has_liquefaction
from .fem2d_structural import structural_total_dofs
from .fem2d_types import FEM2DError, ElasticPlaneStrainMaterial, Mesh2D, StructuralElement2D, njit, normalize_integration
from .fem2d_utils import _dofs_from_node_indices, _element_node_indices, _ensure_list
from .sparse_assembly import SparseAssemblyBuilder, SparseAssemblyPattern

_PRESSURE_BATCH_MIN_ELEMENTS = 8

PRESSURE_ASSEMBLY_FUNCTIONS = (
    "assemble_liquefaction_pressure_terms",
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
    "_axisymmetric_edge_measure",
    "assemble_pore_pressure_load",
    "assemble_biot_coupling_matrix",
    "solve_consolidation_pressure",
    "assemble_pressure_matrices",
    "assemble_pressure_boundary_terms",
    "_solve_scalar_constraints",
)


def pressure_assembly_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.pressure_assembly.v1",
        "module": "geofem_app.fem2d_pressure",
        "function_count": len(PRESSURE_ASSEMBLY_FUNCTIONS),
        "functions": list(PRESSURE_ASSEMBLY_FUNCTIONS),
        "covered_surfaces": [
            "plane_strain_pressure_matrices",
            "axisymmetric_pressure_matrices",
            "quad4_quad8_pressure_matrix_batching",
            "tri3_tri6_pressure_matrix_batching",
            "biot_coupling",
            "quad4_quad8_biot_coupling_batching",
            "tri3_tri6_biot_coupling_batching",
            "pore_pressure_load",
            "pore_pressure_load_direct_vector_scatter",
            "pressure_boundary_terms",
            "liquefaction_pressure_terms",
            "standalone_consolidation_pressure_solve",
        ],
    }


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


@dataclass(frozen=True)
class PressureElementMatrixBlockCache:
    element_index: int
    conn: np.ndarray


@dataclass(frozen=True)
class PressureElementMatrixBatchCache:
    element_type: str
    block_indices: np.ndarray
    element_indices: np.ndarray
    conn: np.ndarray


@dataclass(frozen=True)
class PressureMatrixAssemblyCache:
    axisymmetric: bool
    shape: tuple[int, int]
    pattern: SparseAssemblyPattern
    element_blocks: tuple[PressureElementMatrixBlockCache, ...]
    batch_blocks: tuple[PressureElementMatrixBatchCache, ...] = ()

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "pressure_matrix_assembly_cache",
            "axisymmetric": bool(self.axisymmetric),
            "element_blocks": len(self.element_blocks),
            "batch_min_elements": _PRESSURE_BATCH_MIN_ELEMENTS,
            "batched_elements": _batched_element_count(self.batch_blocks),
            "batch_groups": _batch_group_info(self.batch_blocks),
            "direct_fill": self.pattern.direct_fill_info(),
        }


@dataclass(frozen=True)
class BiotCouplingBlockCache:
    element_index: int
    displacement_dofs: np.ndarray
    pressure_nodes: np.ndarray


@dataclass(frozen=True)
class BiotCouplingBatchCache:
    element_type: str
    block_indices: np.ndarray
    element_indices: np.ndarray
    pressure_nodes: np.ndarray


@dataclass(frozen=True)
class BiotCouplingAssemblyCache:
    axisymmetric: bool
    shape: tuple[int, int]
    pattern: SparseAssemblyPattern
    element_blocks: tuple[BiotCouplingBlockCache, ...]
    batch_blocks: tuple[BiotCouplingBatchCache, ...] = ()

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "biot_coupling_assembly_cache",
            "axisymmetric": bool(self.axisymmetric),
            "element_blocks": len(self.element_blocks),
            "batch_min_elements": _PRESSURE_BATCH_MIN_ELEMENTS,
            "batched_elements": _batched_element_count(self.batch_blocks),
            "batch_groups": _batch_group_info(self.batch_blocks),
            "direct_fill": self.pattern.direct_fill_info(),
        }


@dataclass(frozen=True)
class PressureFluxEdgeCache:
    node_indices: np.ndarray
    weights: np.ndarray
    measure: float
    flux: float


@dataclass(frozen=True)
class PressureRobinEdgeCache:
    node_indices: np.ndarray
    weights: np.ndarray
    measure: float
    beta: float
    pressure: float
    seepage: bool
    local_matrix: np.ndarray


@dataclass(frozen=True)
class PressureBoundaryTermCache:
    axisymmetric: bool
    shape: tuple[int, int]
    pattern: SparseAssemblyPattern
    flux_edges: tuple[PressureFluxEdgeCache, ...]
    robin_edges: tuple[PressureRobinEdgeCache, ...]
    pressure_dependent: bool

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "pressure_boundary_term_cache",
            "axisymmetric": bool(self.axisymmetric),
            "flux_edges": len(self.flux_edges),
            "robin_edges": len(self.robin_edges),
            "pressure_dependent": bool(self.pressure_dependent),
            "direct_fill": self.pattern.direct_fill_info(),
        }


@dataclass(frozen=True)
class PorePressureLoadAssemblyCache:
    biot_cache: BiotCouplingAssemblyCache

    @property
    def shape(self) -> tuple[int, int]:
        return self.biot_cache.shape

    @property
    def element_blocks(self) -> tuple[BiotCouplingBlockCache, ...]:
        return self.biot_cache.element_blocks

    @property
    def batch_blocks(self) -> tuple[BiotCouplingBatchCache, ...]:
        return self.biot_cache.batch_blocks

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "pore_pressure_load_assembly_cache",
            "axisymmetric": bool(self.biot_cache.axisymmetric),
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "element_blocks": len(self.element_blocks),
            "batch_min_elements": _PRESSURE_BATCH_MIN_ELEMENTS,
            "batched_elements": _batched_element_count(self.batch_blocks),
            "batch_groups": _batch_group_info(self.batch_blocks),
            "direct_vector_scatter": {
                "enabled": True,
                "mode": "cached_biot_blocks_pressure_vector_scatter",
            },
            "biot_coupling_cache": self.biot_cache.info(),
        }


def _batched_element_count(batch_blocks: tuple[Any, ...]) -> int:
    return int(sum(int(batch.block_indices.size) for batch in batch_blocks))


def _batch_group_info(batch_blocks: tuple[Any, ...]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for batch in batch_blocks:
        indices = np.asarray(batch.block_indices, dtype=np.int64)
        contiguous = bool(indices.size == 0 or np.all(np.diff(indices) == 1))
        groups.append(
            {
                "element_type": batch.element_type,
                "elements": int(indices.size),
                "contiguous_blocks": contiguous,
            }
        )
    return groups


def _build_pressure_element_matrix_batches(
    mesh: Mesh2D,
    blocks: tuple[PressureElementMatrixBlockCache, ...],
) -> tuple[PressureElementMatrixBatchCache, ...]:
    grouped: dict[str, dict[str, list[Any]]] = {
        "TRI3": {"block_indices": [], "element_indices": [], "conn": []},
        "TRI6": {"block_indices": [], "element_indices": [], "conn": []},
        "QUAD4": {"block_indices": [], "element_indices": [], "conn": []},
        "QUAD8": {"block_indices": [], "element_indices": [], "conn": []},
    }
    for block_index, block in enumerate(blocks):
        element = mesh.elements[block.element_index]
        element_type = str(element.type).upper()
        if element_type not in grouped:
            continue
        grouped[element_type]["block_indices"].append(block_index)
        grouped[element_type]["element_indices"].append(block.element_index)
        grouped[element_type]["conn"].append(np.asarray(block.conn, dtype=np.int64))

    batches: list[PressureElementMatrixBatchCache] = []
    for element_type in ("TRI3", "TRI6", "QUAD4", "QUAD8"):
        raw = grouped[element_type]
        if len(raw["block_indices"]) < _PRESSURE_BATCH_MIN_ELEMENTS:
            continue
        batches.append(
            PressureElementMatrixBatchCache(
                element_type=element_type,
                block_indices=np.asarray(raw["block_indices"], dtype=np.int64),
                element_indices=np.asarray(raw["element_indices"], dtype=np.int64),
                conn=np.vstack(raw["conn"]).astype(np.int64, copy=False),
            )
        )
    return tuple(batches)


def _build_biot_coupling_batches(
    mesh: Mesh2D,
    blocks: tuple[BiotCouplingBlockCache, ...],
) -> tuple[BiotCouplingBatchCache, ...]:
    grouped: dict[str, dict[str, list[Any]]] = {
        "TRI3": {"block_indices": [], "element_indices": [], "pressure_nodes": []},
        "TRI6": {"block_indices": [], "element_indices": [], "pressure_nodes": []},
        "QUAD4": {"block_indices": [], "element_indices": [], "pressure_nodes": []},
        "QUAD8": {"block_indices": [], "element_indices": [], "pressure_nodes": []},
    }
    for block_index, block in enumerate(blocks):
        element = mesh.elements[block.element_index]
        element_type = str(element.type).upper()
        if element_type not in grouped:
            continue
        grouped[element_type]["block_indices"].append(block_index)
        grouped[element_type]["element_indices"].append(block.element_index)
        grouped[element_type]["pressure_nodes"].append(np.asarray(block.pressure_nodes, dtype=np.int64))

    batches: list[BiotCouplingBatchCache] = []
    for element_type in ("TRI3", "TRI6", "QUAD4", "QUAD8"):
        raw = grouped[element_type]
        if len(raw["block_indices"]) < _PRESSURE_BATCH_MIN_ELEMENTS:
            continue
        batches.append(
            BiotCouplingBatchCache(
                element_type=element_type,
                block_indices=np.asarray(raw["block_indices"], dtype=np.int64),
                element_indices=np.asarray(raw["element_indices"], dtype=np.int64),
                pressure_nodes=np.vstack(raw["pressure_nodes"]).astype(np.int64, copy=False),
            )
        )
    return tuple(batches)


def _pressure_batch_inputs(
    batch: PressureElementMatrixBatchCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.ascontiguousarray(mesh.coords[batch.conn], dtype=np.float64)
    thickness = np.empty(batch.element_indices.size, dtype=np.float64)
    for local_index, element_index in enumerate(batch.element_indices):
        element = mesh.elements[int(element_index)]
        thickness[local_index] = float(materials[element.material].thickness)
    return coords, thickness


def _biot_batch_inputs(
    batch: BiotCouplingBatchCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.ascontiguousarray(mesh.coords[batch.pressure_nodes], dtype=np.float64)
    thickness = np.empty(batch.element_indices.size, dtype=np.float64)
    for local_index, element_index in enumerate(batch.element_indices):
        element = mesh.elements[int(element_index)]
        thickness[local_index] = float(materials[element.material].thickness)
    return coords, thickness


def _quad8_biot_batch_material_inputs(
    batch: BiotCouplingBatchCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> tuple[np.ndarray, np.ndarray]:
    mode_codes = np.zeros(batch.element_indices.size, dtype=np.int64)
    volumetric_projectors = np.zeros((batch.element_indices.size, 4, 4), dtype=np.float64)
    for local_index, element_index in enumerate(batch.element_indices):
        element = mesh.elements[int(element_index)]
        material = materials[element.material]
        mode_codes[local_index] = int(_quad4_mode_code(normalize_integration(element.integration)))
        volumetric_projectors[local_index, :, :] = np.asarray(material.volumetric_projector, dtype=np.float64)
    return mode_codes, np.ascontiguousarray(volumetric_projectors)


def _fill_batch_flat_values(pattern: SparseAssemblyPattern, flat_values: np.ndarray, block_indices: np.ndarray, blocks_flat: np.ndarray) -> None:
    indices = np.asarray(block_indices, dtype=np.int64)
    if indices.size == 0:
        return
    values = np.asarray(blocks_flat, dtype=np.float64)
    if bool(np.all(np.diff(indices) == 1)):
        pattern.fill_blocks_flat(flat_values, int(indices[0]), values, int(indices.size))
        return
    for local_index, block_index in enumerate(indices):
        pattern.fill_block(flat_values, int(block_index), values[local_index])


def _validate_batched_geometry(element_type: str, min_det: float, min_radius: float, axisymmetric: bool) -> None:
    if min_det <= 0.0:
        raise FEM2DError(f"{element_type}: detJ must be positive, got {min_det:.6e}")
    if axisymmetric and min_radius <= 0.0:
        raise FEM2DError(f"{element_type}: axisymmetric radius must be positive, got {min_radius:.6e}")


def _tri_pressure_matrices_fast(
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    storage: float,
    permeability: float,
    axisymmetric: bool,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = int(np.asarray(coords).shape[0])
    if node_count not in (3, 6):
        raise FEM2DError(f"TRI pressure kernel expects 3 or 6 nodes, got {node_count}")
    me, ke, min_det, min_radius = _tri_pressure_matrices_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(storage),
        float(permeability),
        float(material.thickness),
        bool(axisymmetric),
        node_count,
    )
    element_type = "TRI3" if node_count == 3 else "TRI6"
    _validate_batched_geometry(element_type, float(min_det), float(min_radius), bool(axisymmetric))
    return np.asarray(me, dtype=float), np.asarray(ke, dtype=float)


def _tri_biot_matrix_fast(
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    alpha: float,
    *,
    axisymmetric: bool,
) -> np.ndarray:
    node_count = int(np.asarray(coords).shape[0])
    if node_count not in (3, 6):
        raise FEM2DError(f"TRI Biot kernel expects 3 or 6 nodes, got {node_count}")
    block, min_det, min_radius = _tri_biot_coupling_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(alpha),
        float(material.thickness),
        bool(axisymmetric),
        node_count,
    )
    element_type = "TRI3" if node_count == 3 else "TRI6"
    _validate_batched_geometry(element_type, float(min_det), float(min_radius), bool(axisymmetric))
    return np.asarray(block, dtype=float)


@njit(cache=True)
def _tri_pressure_gp_numba(coords: np.ndarray, node_count: int, xi: float, eta: float) -> tuple[np.ndarray, np.ndarray, float]:
    N = np.zeros(node_count, dtype=np.float64)
    dxi = np.zeros(node_count, dtype=np.float64)
    deta = np.zeros(node_count, dtype=np.float64)
    grad = np.zeros((2, node_count), dtype=np.float64)
    if node_count == 3:
        N[0] = 1.0 - xi - eta
        N[1] = xi
        N[2] = eta
        dxi[0] = -1.0
        dxi[1] = 1.0
        dxi[2] = 0.0
        deta[0] = -1.0
        deta[1] = 0.0
        deta[2] = 1.0
    else:
        l1 = 1.0 - xi - eta
        l2 = xi
        l3 = eta
        N[0] = l1 * (2.0 * l1 - 1.0)
        N[1] = l2 * (2.0 * l2 - 1.0)
        N[2] = l3 * (2.0 * l3 - 1.0)
        N[3] = 4.0 * l1 * l2
        N[4] = 4.0 * l2 * l3
        N[5] = 4.0 * l3 * l1
        dxi[0] = -(4.0 * l1 - 1.0)
        dxi[1] = 4.0 * l2 - 1.0
        dxi[2] = 0.0
        dxi[3] = 4.0 * (l1 - l2)
        dxi[4] = 4.0 * l3
        dxi[5] = -4.0 * l3
        deta[0] = -(4.0 * l1 - 1.0)
        deta[1] = 0.0
        deta[2] = 4.0 * l3 - 1.0
        deta[3] = -4.0 * l2
        deta[4] = 4.0 * l2
        deta[5] = 4.0 * (l1 - l3)

    j00 = 0.0
    j01 = 0.0
    j10 = 0.0
    j11 = 0.0
    for a in range(node_count):
        j00 += dxi[a] * coords[a, 0]
        j01 += dxi[a] * coords[a, 1]
        j10 += deta[a] * coords[a, 0]
        j11 += deta[a] * coords[a, 1]
    det = j00 * j11 - j01 * j10
    if det <= 0.0:
        return N, grad, det
    inv00 = j11 / det
    inv01 = -j01 / det
    inv10 = -j10 / det
    inv11 = j00 / det
    for a in range(node_count):
        grad[0, a] = inv00 * dxi[a] + inv01 * deta[a]
        grad[1, a] = inv10 * dxi[a] + inv11 * deta[a]
    return N, grad, det


@njit(cache=True)
def _tri_pressure_full_gp_numba(node_count: int, gp_index: int) -> tuple[float, float, float]:
    if node_count == 3:
        return 1.0 / 3.0, 1.0 / 3.0, 0.5
    if gp_index == 0:
        return 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0
    if gp_index == 1:
        return 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0
    return 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0


@njit(cache=True)
def _tri_pressure_matrices_numba(
    coords: np.ndarray,
    storage: float,
    permeability: float,
    thickness: float,
    axisymmetric: bool,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    mass = np.zeros((node_count, node_count), dtype=np.float64)
    conductivity = np.zeros((node_count, node_count), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    gp_count = 1 if node_count == 3 else 3
    for gp in range(gp_count):
        xi, eta, weight = _tri_pressure_full_gp_numba(node_count, gp)
        N, grad, det = _tri_pressure_gp_numba(coords, node_count, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            continue
        radius = 1.0
        if axisymmetric:
            radius = 0.0
            for i in range(node_count):
                radius += N[i] * coords[i, 0]
            if radius < min_radius:
                min_radius = radius
            if radius <= 0.0:
                continue
        dV = det * weight * thickness
        if axisymmetric:
            dV *= 2.0 * math.pi * radius
        for i in range(node_count):
            for j in range(node_count):
                mass[i, j] += storage * N[i] * N[j] * dV
                conductivity[i, j] += permeability * (grad[0, i] * grad[0, j] + grad[1, i] * grad[1, j]) * dV
    return mass, conductivity, min_det, min_radius


@njit(cache=True)
def _tri_pressure_matrices_batch_numba(
    coords: np.ndarray,
    storage: float,
    permeability: float,
    thickness: np.ndarray,
    axisymmetric: bool,
    node_count: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    count = coords.shape[0]
    flat_size = node_count * node_count
    mass = np.zeros((count, flat_size), dtype=np.float64)
    conductivity = np.zeros((count, flat_size), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    for element_index in range(count):
        me, ke, det, radius = _tri_pressure_matrices_numba(
            coords[element_index],
            storage,
            permeability,
            thickness[element_index],
            axisymmetric,
            node_count,
        )
        if det < min_det:
            min_det = det
        if axisymmetric and radius < min_radius:
            min_radius = radius
        for i in range(node_count):
            for j in range(node_count):
                flat_index = i * node_count + j
                mass[element_index, flat_index] = me[i, j]
                conductivity[element_index, flat_index] = ke[i, j]
    return mass, conductivity, min_det, min_radius


@njit(cache=True)
def _tri_biot_coupling_numba(
    coords: np.ndarray,
    alpha: float,
    thickness: float,
    axisymmetric: bool,
    node_count: int,
) -> tuple[np.ndarray, float, float]:
    values = np.zeros((node_count * 2, node_count), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    gp_count = 1 if node_count == 3 else 3
    for gp in range(gp_count):
        xi, eta, weight = _tri_pressure_full_gp_numba(node_count, gp)
        N, grad, det = _tri_pressure_gp_numba(coords, node_count, xi, eta)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            continue
        radius = 1.0
        if axisymmetric:
            radius = 0.0
            for i in range(node_count):
                radius += N[i] * coords[i, 0]
            if radius < min_radius:
                min_radius = radius
            if radius <= 0.0:
                continue
        dV = det * weight * thickness
        if axisymmetric:
            dV *= 2.0 * math.pi * radius
        for a in range(node_count):
            radial_term = grad[0, a]
            if axisymmetric:
                radial_term += N[a] / radius
            axial_term = grad[1, a]
            for j in range(node_count):
                scale = alpha * N[j] * dV
                values[2 * a, j] += radial_term * scale
                values[2 * a + 1, j] += axial_term * scale
    return values, min_det, min_radius


@njit(cache=True)
def _tri_biot_coupling_batch_numba(
    coords: np.ndarray,
    alpha: float,
    thickness: np.ndarray,
    axisymmetric: bool,
    node_count: int,
) -> tuple[np.ndarray, float, float]:
    count = coords.shape[0]
    dof_count = node_count * 2
    values = np.zeros((count, dof_count * node_count), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    for element_index in range(count):
        block, det, radius = _tri_biot_coupling_numba(
            coords[element_index],
            alpha,
            thickness[element_index],
            axisymmetric,
            node_count,
        )
        if det < min_det:
            min_det = det
        if axisymmetric and radius < min_radius:
            min_radius = radius
        for i in range(dof_count):
            for j in range(node_count):
                values[element_index, i * node_count + j] = block[i, j]
    return values, min_det, min_radius


@njit(cache=True)
def _quad4_pressure_matrices_batch_numba(
    coords: np.ndarray,
    storage: float,
    permeability: float,
    thickness: np.ndarray,
    axisymmetric: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    count = coords.shape[0]
    mass = np.zeros((count, 16), dtype=np.float64)
    conductivity = np.zeros((count, 16), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    for element_index in range(count):
        if axisymmetric:
            me, ke, det, radius = _quad4_axisymmetric_pressure_matrices_numba(
                coords[element_index],
                storage,
                permeability,
                thickness[element_index],
            )
            if radius < min_radius:
                min_radius = radius
        else:
            me, ke, det = _quad4_pressure_matrices_numba(
                coords[element_index],
                storage,
                permeability,
                thickness[element_index],
            )
        if det < min_det:
            min_det = det
        for i in range(4):
            for j in range(4):
                flat_index = i * 4 + j
                mass[element_index, flat_index] = me[i, j]
                conductivity[element_index, flat_index] = ke[i, j]
    return mass, conductivity, min_det, min_radius


@njit(cache=True)
def _quad8_pressure_matrices_batch_numba(
    coords: np.ndarray,
    storage: float,
    permeability: float,
    thickness: np.ndarray,
    axisymmetric: bool,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    count = coords.shape[0]
    mass = np.zeros((count, 64), dtype=np.float64)
    conductivity = np.zeros((count, 64), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    for element_index in range(count):
        if axisymmetric:
            me, ke, det, radius = _quad8_axisymmetric_pressure_matrices_numba(
                coords[element_index],
                storage,
                permeability,
                thickness[element_index],
            )
            if radius < min_radius:
                min_radius = radius
        else:
            me, ke, det = _quad8_pressure_matrices_numba(
                coords[element_index],
                storage,
                permeability,
                thickness[element_index],
            )
        if det < min_det:
            min_det = det
        for i in range(8):
            for j in range(8):
                flat_index = i * 8 + j
                mass[element_index, flat_index] = me[i, j]
                conductivity[element_index, flat_index] = ke[i, j]
    return mass, conductivity, min_det, min_radius


@njit(cache=True)
def _quad4_biot_coupling_batch_numba(
    coords: np.ndarray,
    alpha: float,
    thickness: np.ndarray,
    axisymmetric: bool,
) -> tuple[np.ndarray, float, float]:
    count = coords.shape[0]
    values = np.zeros((count, 32), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    for element_index in range(count):
        if axisymmetric:
            block, det, radius = _quad4_axisymmetric_biot_matrix_numba(
                coords[element_index],
                alpha,
                thickness[element_index],
            )
            if radius < min_radius:
                min_radius = radius
        else:
            block, det = _quad4_biot_matrix_numba(
                coords[element_index],
                alpha,
                thickness[element_index],
            )
        if det < min_det:
            min_det = det
        for i in range(8):
            for j in range(4):
                values[element_index, i * 4 + j] = block[i, j]
    return values, min_det, min_radius


@njit(cache=True)
def _quad8_biot_coupling_batch_numba(
    coords: np.ndarray,
    alpha: float,
    thickness: np.ndarray,
    mode_codes: np.ndarray,
    volumetric_projectors: np.ndarray,
    axisymmetric: bool,
) -> tuple[np.ndarray, float, float, float]:
    count = coords.shape[0]
    values = np.zeros((count, 128), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    min_bbar_volume = 1.0e300
    for element_index in range(count):
        if axisymmetric:
            block, det, radius = _quad8_axisymmetric_biot_matrix_numba(
                coords[element_index],
                alpha,
                thickness[element_index],
            )
            if radius < min_radius:
                min_radius = radius
        else:
            block, det, volume = _quad8_biot_matrix_numba(
                coords[element_index],
                alpha,
                thickness[element_index],
                volumetric_projectors[element_index],
                mode_codes[element_index],
            )
            if mode_codes[element_index] == _QUAD4_MODE_BBAR and volume < min_bbar_volume:
                min_bbar_volume = volume
        if det < min_det:
            min_det = det
        for i in range(16):
            for j in range(8):
                values[element_index, i * 8 + j] = block[i, j]
    return values, min_det, min_radius, min_bbar_volume


@njit(cache=True)
def _equivalent_shear_strain4_numba(e0: float, e1: float, e2: float, e3: float) -> float:
    mean = (e0 + e1 + e2) / 3.0
    dev0 = e0 - mean
    dev1 = e1 - mean
    dev2 = e2 - mean
    dev3 = 0.5 * e3
    j2e = 0.5 * (dev0 * dev0 + dev1 * dev1 + dev2 * dev2) + dev3 * dev3
    return math.sqrt(max(4.0 * j2e / 3.0, 0.0))


@njit(cache=True)
def _quad4_liquefaction_pressure_batch_numba(
    coords: np.ndarray,
    du: np.ndarray,
    thickness: np.ndarray,
    unit_weight: np.ndarray,
    initial_effective_stress: np.ndarray,
    crr: np.ndarray,
    csr: np.ndarray,
    generation_rate: np.ndarray,
    dissipation_rate: np.ndarray,
    gamma_ref: np.ndarray,
    manual_cycle_increment: np.ndarray,
    storage: float,
    dt: float,
    axisymmetric: bool,
    y_top: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    count = coords.shape[0]
    rhs = np.zeros((count, 4), dtype=np.float64)
    diss = np.zeros((count, 16), dtype=np.float64)
    metrics = np.zeros(6, dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    a = 1.0 / math.sqrt(3.0)
    for element_index in range(count):
        for gp in range(4):
            xi = -a if gp == 0 or gp == 3 else a
            eta = -a if gp == 0 or gp == 1 else a
            N, grad, det = _quad4_shape_grad_numba(coords[element_index], xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                continue
            radius = 1.0
            if axisymmetric:
                radius = 0.0
                for i in range(4):
                    radius += N[i] * coords[element_index, i, 0]
                if radius < min_radius:
                    min_radius = radius
                if radius <= 0.0:
                    continue
            e0 = 0.0
            e1 = 0.0
            e2 = 0.0
            e3 = 0.0
            y_gp = 0.0
            for i in range(4):
                ur = du[element_index, 2 * i]
                uz = du[element_index, 2 * i + 1]
                e0 += grad[0, i] * ur
                e1 += grad[1, i] * uz
                e3 += grad[1, i] * ur + grad[0, i] * uz
                y_gp += N[i] * coords[element_index, i, 1]
                if axisymmetric:
                    e2 += N[i] * ur / radius
            gamma_increment = _equivalent_shear_strain4_numba(e0, e1, e2, e3)
            cycle_increment = gamma_increment / max(4.0 * gamma_ref[element_index], 1.0e-12)
            if manual_cycle_increment[element_index] > cycle_increment:
                cycle_increment = manual_cycle_increment[element_index]
            if cycle_increment <= 0.0 and dissipation_rate[element_index] <= 0.0:
                continue
            dV = det * thickness[element_index]
            if axisymmetric:
                dV *= 2.0 * math.pi * radius
            effective_ref = initial_effective_stress[element_index]
            if effective_ref <= 0.0:
                effective_ref = abs(unit_weight[element_index]) * max(y_top - y_gp, 0.0)
            effective_ref = max(effective_ref, 1.0)
            demand = 0.0
            if crr[element_index] > 0.0:
                demand = max(csr[element_index] / crr[element_index], 0.0)
            ru_generation_increment = min(0.99, generation_rate[element_index] * demand * cycle_increment)
            pressure_increment = effective_ref * ru_generation_increment
            if pressure_increment > 0.0:
                source_scale = storage * pressure_increment * dV / dt
                for i in range(4):
                    rhs[element_index, i] += N[i] * source_scale
                metrics[0] += source_scale
                metrics[1] += pressure_increment
            if dissipation_rate[element_index] > 0.0:
                local_scale = storage * dissipation_rate[element_index] * dV
                for i in range(4):
                    for j in range(4):
                        value = local_scale * N[i] * N[j]
                        diss[element_index, i * 4 + j] += value
                        metrics[2] += value
            metrics[3] += 1.0
            metrics[4] += cycle_increment
            if cycle_increment > metrics[5]:
                metrics[5] = cycle_increment
    return rhs, diss, metrics, min_det, min_radius


@njit(cache=True)
def _quad8_liquefaction_pressure_batch_numba(
    coords: np.ndarray,
    du: np.ndarray,
    thickness: np.ndarray,
    unit_weight: np.ndarray,
    initial_effective_stress: np.ndarray,
    crr: np.ndarray,
    csr: np.ndarray,
    generation_rate: np.ndarray,
    dissipation_rate: np.ndarray,
    gamma_ref: np.ndarray,
    manual_cycle_increment: np.ndarray,
    storage: float,
    dt: float,
    axisymmetric: bool,
    y_top: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    count = coords.shape[0]
    rhs = np.zeros((count, 8), dtype=np.float64)
    diss = np.zeros((count, 64), dtype=np.float64)
    metrics = np.zeros(6, dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    for element_index in range(count):
        for gp in range(9):
            xi, eta, weight = _quad8_gp_full(gp)
            N, grad, det = _quad8_shape_grad_numba(coords[element_index], xi, eta)
            if det < min_det:
                min_det = det
            if det <= 0.0:
                continue
            radius = 1.0
            if axisymmetric:
                radius = 0.0
                for i in range(8):
                    radius += N[i] * coords[element_index, i, 0]
                if radius < min_radius:
                    min_radius = radius
                if radius <= 0.0:
                    continue
            e0 = 0.0
            e1 = 0.0
            e2 = 0.0
            e3 = 0.0
            y_gp = 0.0
            for i in range(8):
                ur = du[element_index, 2 * i]
                uz = du[element_index, 2 * i + 1]
                e0 += grad[0, i] * ur
                e1 += grad[1, i] * uz
                e3 += grad[1, i] * ur + grad[0, i] * uz
                y_gp += N[i] * coords[element_index, i, 1]
                if axisymmetric:
                    e2 += N[i] * ur / radius
            gamma_increment = _equivalent_shear_strain4_numba(e0, e1, e2, e3)
            cycle_increment = gamma_increment / max(4.0 * gamma_ref[element_index], 1.0e-12)
            if manual_cycle_increment[element_index] > cycle_increment:
                cycle_increment = manual_cycle_increment[element_index]
            if cycle_increment <= 0.0 and dissipation_rate[element_index] <= 0.0:
                continue
            dV = det * weight * thickness[element_index]
            if axisymmetric:
                dV *= 2.0 * math.pi * radius
            effective_ref = initial_effective_stress[element_index]
            if effective_ref <= 0.0:
                effective_ref = abs(unit_weight[element_index]) * max(y_top - y_gp, 0.0)
            effective_ref = max(effective_ref, 1.0)
            demand = 0.0
            if crr[element_index] > 0.0:
                demand = max(csr[element_index] / crr[element_index], 0.0)
            ru_generation_increment = min(0.99, generation_rate[element_index] * demand * cycle_increment)
            pressure_increment = effective_ref * ru_generation_increment
            if pressure_increment > 0.0:
                source_scale = storage * pressure_increment * dV / dt
                for i in range(8):
                    rhs[element_index, i] += N[i] * source_scale
                metrics[0] += source_scale
                metrics[1] += pressure_increment
            if dissipation_rate[element_index] > 0.0:
                local_scale = storage * dissipation_rate[element_index] * dV
                for i in range(8):
                    for j in range(8):
                        value = local_scale * N[i] * N[j]
                        diss[element_index, i * 8 + j] += value
                        metrics[2] += value
            metrics[3] += 1.0
            metrics[4] += cycle_increment
            if cycle_increment > metrics[5]:
                metrics[5] = cycle_increment
    return rhs, diss, metrics, min_det, min_radius


def build_pressure_matrix_assembly_cache(mesh: Mesh2D, *, axisymmetric: bool = False) -> PressureMatrixAssemblyCache:
    nnode_total = len(mesh.node_ids)
    blocks: list[PressureElementMatrixBlockCache] = []
    dof_blocks: list[np.ndarray] = []
    for element_index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        conn = np.asarray(_element_node_indices(element.nodes, mesh.node_index), dtype=np.int64)
        blocks.append(PressureElementMatrixBlockCache(element_index=element_index, conn=conn))
        dof_blocks.append(conn)
    pattern = SparseAssemblyPattern.from_square_blocks(dof_blocks, (nnode_total, nnode_total))
    block_tuple = tuple(blocks)
    return PressureMatrixAssemblyCache(
        axisymmetric=axisymmetric,
        shape=(nnode_total, nnode_total),
        pattern=pattern,
        element_blocks=block_tuple,
        batch_blocks=_build_pressure_element_matrix_batches(mesh, block_tuple),
    )


def assemble_pressure_matrices_cached(
    cache: PressureMatrixAssemblyCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    *,
    storage: float,
    permeability: float,
) -> tuple[csr_matrix, csr_matrix]:
    mass_values = cache.pattern.empty_flat_values()
    conductivity_values = cache.pattern.empty_flat_values()
    batched = np.zeros(len(cache.element_blocks), dtype=bool)
    for batch in cache.batch_blocks:
        coords, thickness = _pressure_batch_inputs(batch, mesh, materials)
        if batch.element_type == "TRI3":
            mass_blocks, conductivity_blocks, min_det, min_radius = _tri_pressure_matrices_batch_numba(
                coords,
                float(storage),
                float(permeability),
                thickness,
                bool(cache.axisymmetric),
                3,
            )
        elif batch.element_type == "TRI6":
            mass_blocks, conductivity_blocks, min_det, min_radius = _tri_pressure_matrices_batch_numba(
                coords,
                float(storage),
                float(permeability),
                thickness,
                bool(cache.axisymmetric),
                6,
            )
        elif batch.element_type == "QUAD4":
            mass_blocks, conductivity_blocks, min_det, min_radius = _quad4_pressure_matrices_batch_numba(
                coords,
                float(storage),
                float(permeability),
                thickness,
                bool(cache.axisymmetric),
            )
        elif batch.element_type == "QUAD8":
            mass_blocks, conductivity_blocks, min_det, min_radius = _quad8_pressure_matrices_batch_numba(
                coords,
                float(storage),
                float(permeability),
                thickness,
                bool(cache.axisymmetric),
            )
        else:
            continue
        _validate_batched_geometry(batch.element_type, float(min_det), float(min_radius), cache.axisymmetric)
        _fill_batch_flat_values(cache.pattern, mass_values, batch.block_indices, mass_blocks)
        _fill_batch_flat_values(cache.pattern, conductivity_values, batch.block_indices, conductivity_blocks)
        batched[batch.block_indices] = True
    for block_index, block in enumerate(cache.element_blocks):
        if batched[block_index]:
            continue
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        coords = mesh.coords[block.conn]
        me, ke = _pressure_matrix_block(element.type, coords, material, storage=storage, permeability=permeability, axisymmetric=cache.axisymmetric)
        cache.pattern.fill_block(mass_values, block_index, me)
        cache.pattern.fill_block(conductivity_values, block_index, ke)
    cache.pattern.validate_filled_block_count(len(cache.element_blocks))
    return cache.pattern.assemble_flat_values(mass_values), cache.pattern.assemble_flat_values(conductivity_values)


def build_biot_coupling_assembly_cache(
    mesh: Mesh2D,
    structural_elements: list[StructuralElement2D] | None = None,
    *,
    axisymmetric: bool = False,
) -> BiotCouplingAssemblyCache:
    npress = len(mesh.node_ids)
    ndof = len(mesh.node_ids) * 2 if axisymmetric else structural_total_dofs(mesh, structural_elements)
    blocks: list[BiotCouplingBlockCache] = []
    row_blocks: list[np.ndarray] = []
    col_blocks: list[np.ndarray] = []
    for element_index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        conn = np.asarray(_element_node_indices(element.nodes, mesh.node_index), dtype=np.int64)
        dofs = _dofs_from_node_indices(conn)
        blocks.append(BiotCouplingBlockCache(element_index=element_index, displacement_dofs=dofs, pressure_nodes=conn))
        row_blocks.append(dofs)
        col_blocks.append(conn)
    pattern = SparseAssemblyPattern.from_blocks(row_blocks, col_blocks, (ndof, npress))
    block_tuple = tuple(blocks)
    return BiotCouplingAssemblyCache(
        axisymmetric=axisymmetric,
        shape=(ndof, npress),
        pattern=pattern,
        element_blocks=block_tuple,
        batch_blocks=_build_biot_coupling_batches(mesh, block_tuple),
    )


def assemble_biot_coupling_matrix_cached(
    cache: BiotCouplingAssemblyCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    alpha: float = 1.0,
) -> csr_matrix:
    values = cache.pattern.empty_flat_values()
    batched = np.zeros(len(cache.element_blocks), dtype=bool)
    for batch in cache.batch_blocks:
        coords, thickness = _biot_batch_inputs(batch, mesh, materials)
        if batch.element_type == "TRI3":
            blocks_flat, min_det, min_radius = _tri_biot_coupling_batch_numba(
                coords,
                float(alpha),
                thickness,
                bool(cache.axisymmetric),
                3,
            )
        elif batch.element_type == "TRI6":
            blocks_flat, min_det, min_radius = _tri_biot_coupling_batch_numba(
                coords,
                float(alpha),
                thickness,
                bool(cache.axisymmetric),
                6,
            )
        elif batch.element_type == "QUAD4":
            blocks_flat, min_det, min_radius = _quad4_biot_coupling_batch_numba(
                coords,
                float(alpha),
                thickness,
                bool(cache.axisymmetric),
            )
        elif batch.element_type == "QUAD8":
            mode_codes, volumetric_projectors = _quad8_biot_batch_material_inputs(batch, mesh, materials)
            blocks_flat, min_det, min_radius, min_bbar_volume = _quad8_biot_coupling_batch_numba(
                coords,
                float(alpha),
                thickness,
                mode_codes,
                volumetric_projectors,
                bool(cache.axisymmetric),
            )
            if not cache.axisymmetric and min_bbar_volume <= 0.0:
                raise FEM2DError("QUAD8: non-positive element measure")
        else:
            continue
        _validate_batched_geometry(batch.element_type, float(min_det), float(min_radius), cache.axisymmetric)
        _fill_batch_flat_values(cache.pattern, values, batch.block_indices, blocks_flat)
        batched[batch.block_indices] = True
    for block_index, block in enumerate(cache.element_blocks):
        if batched[block_index]:
            continue
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        coords = mesh.coords[block.pressure_nodes]
        local = _biot_coupling_block(element.type, coords, material, alpha, normalize_integration(element.integration), axisymmetric=cache.axisymmetric)
        cache.pattern.fill_block(values, block_index, local)
    cache.pattern.validate_filled_block_count(len(cache.element_blocks))
    return cache.pattern.assemble_flat_values(values)


def build_pore_pressure_load_cache(
    mesh: Mesh2D,
    structural_elements: list[StructuralElement2D] | None = None,
    *,
    axisymmetric: bool = False,
) -> PorePressureLoadAssemblyCache:
    return PorePressureLoadAssemblyCache(
        build_biot_coupling_assembly_cache(
            mesh,
            structural_elements=structural_elements,
            axisymmetric=axisymmetric,
        )
    )


def build_pressure_boundary_term_cache(mesh: Mesh2D, hydro: Mapping[str, Any], *, axisymmetric: bool = False) -> PressureBoundaryTermCache:
    nnode_total = len(mesh.node_ids)
    flux_edges: list[PressureFluxEdgeCache] = []
    robin_edges: list[PressureRobinEdgeCache] = []
    robin_blocks: list[np.ndarray] = []
    pressure_dependent = False
    flux_specs = hydro.get("pore_flux_bcs", hydro.get("flux_bcs", hydro.get("pore_flux", hydro.get("flux", []))))
    robin_specs = hydro.get("pore_robin_bcs", hydro.get("robin_bcs", hydro.get("pore_robin", hydro.get("robin", []))))
    for spec in _ensure_list(flux_specs):
        if not isinstance(spec, Mapping):
            raise FEM2DError("each pore flux boundary condition must be a mapping")
        q = float(spec.get("flux", spec.get("q", spec.get("value", 0.0))))
        thickness = float(spec.get("thickness", 1.0))
        for edge in _pressure_edges(mesh, spec):
            indices = np.asarray([mesh.node_index[nid] for nid in edge], dtype=np.int64)
            weights = _edge_lumped_weights(edge)
            measure = _pressure_edge_measure(mesh, edge, thickness, axisymmetric=axisymmetric)
            flux_edges.append(PressureFluxEdgeCache(node_indices=indices, weights=weights, measure=measure, flux=q))

    for spec in _ensure_list(robin_specs):
        if not isinstance(spec, Mapping):
            raise FEM2DError("each pore Robin boundary condition must be a mapping")
        beta = float(spec.get("beta", spec.get("conductance", spec.get("leakage", spec.get("coefficient", 0.0)))))
        pref = float(spec.get("pressure", spec.get("p_ext", spec.get("p_ref", spec.get("value", 0.0)))))
        thickness = float(spec.get("thickness", 1.0))
        seepage = bool(spec.get("seepage_face", spec.get("seepage", False)))
        if beta < 0.0:
            raise FEM2DError("pore Robin conductance must be non-negative")
        pressure_dependent = pressure_dependent or seepage
        for edge in _pressure_edges(mesh, spec):
            indices = np.asarray([mesh.node_index[nid] for nid in edge], dtype=np.int64)
            weights = _edge_lumped_weights(edge)
            measure = _pressure_edge_measure(mesh, edge, thickness, axisymmetric=axisymmetric)
            local = beta * measure * _edge_consistent_robin_matrix(edge)
            robin_edges.append(
                PressureRobinEdgeCache(
                    node_indices=indices,
                    weights=weights,
                    measure=measure,
                    beta=beta,
                    pressure=pref,
                    seepage=seepage,
                    local_matrix=np.asarray(local, dtype=float),
                )
            )
            robin_blocks.append(indices)
    pattern = SparseAssemblyPattern.from_square_blocks(robin_blocks, (nnode_total, nnode_total))
    return PressureBoundaryTermCache(axisymmetric=axisymmetric, shape=(nnode_total, nnode_total), pattern=pattern, flux_edges=tuple(flux_edges), robin_edges=tuple(robin_edges), pressure_dependent=pressure_dependent)


def assemble_pressure_boundary_terms_cached(cache: PressureBoundaryTermCache, pressure: np.ndarray | None = None) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    rhs = np.zeros(cache.shape[0], dtype=float)
    values = cache.pattern.empty_flat_values()
    flux_total = 0.0
    robin_total = 0.0
    seepage_count = 0
    seepage_active_edges = 0
    seepage_inactive_edges = 0
    for edge in cache.flux_edges:
        rhs[edge.node_indices] += edge.flux * edge.measure * edge.weights
        flux_total += edge.flux * edge.measure
    for block_index, edge in enumerate(cache.robin_edges):
        active = True
        if edge.seepage:
            seepage_count += 1
            if pressure is not None:
                p_avg = float(np.dot(edge.weights, np.asarray(pressure, dtype=float)[edge.node_indices]))
                if p_avg <= edge.pressure:
                    active = False
                    seepage_inactive_edges += 1
            if active:
                seepage_active_edges += 1
        if active:
            rhs[edge.node_indices] += edge.pressure * edge.beta * edge.measure * edge.weights
            cache.pattern.fill_block(values, block_index, edge.local_matrix)
            robin_total += edge.beta * edge.measure
        else:
            cache.pattern.fill_block(values, block_index, np.zeros_like(edge.local_matrix))
    cache.pattern.validate_filled_block_count(len(cache.robin_edges))
    matrix = cache.pattern.assemble_flat_values(values)
    info = {
        "flux_total": flux_total,
        "robin_conductance_total": robin_total,
        "flux_count": len(cache.flux_edges),
        "robin_count": len(cache.robin_edges),
        "seepage_count": seepage_count,
        "seepage_active_edges": seepage_active_edges,
        "seepage_inactive_edges": seepage_inactive_edges,
        "direct_fill": cache.info(),
    }
    return matrix, rhs, info


def _pressure_edge_measure(mesh: Mesh2D, edge: tuple[str, ...], thickness: float, *, axisymmetric: bool) -> float:
    if axisymmetric:
        return _axisymmetric_edge_measure(mesh, edge, thickness)
    return _edge_length(mesh, edge) * thickness


def _pressure_matrix_block(
    element_type: str,
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    storage: float,
    permeability: float,
    axisymmetric: bool,
) -> tuple[np.ndarray, np.ndarray]:
    etype = element_type.upper()
    nnode = int(coords.shape[0])
    if axisymmetric:
        if etype == "QUAD4":
            return _quad4_axisymmetric_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability)
        if etype == "QUAD8":
            return _quad8_axisymmetric_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability)
        if etype in {"TRI3", "TRI6"}:
            return _tri_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability, axisymmetric=True)
        me = np.zeros((nnode, nnode), dtype=float)
        ke = np.zeros((nnode, nnode), dtype=float)
        for gp in integration_points(element_type, "FULL"):
            xi, eta, weight = gp
            N, dN_dnatural = shape_functions(element_type, xi, eta)
            jac = dN_dnatural @ coords
            detJ = float(np.linalg.det(jac))
            if detJ <= 0.0:
                raise FEM2DError(f"{element_type}: detJ must be positive, got {detJ:.6e}")
            grad = np.linalg.inv(jac) @ dN_dnatural
            radius = float(N @ coords[:, 0])
            if radius <= 0.0:
                raise FEM2DError(f"{element_type}: axisymmetric radius must be positive, got {radius:.6e}")
            dV = detJ * weight * material.thickness * 2.0 * math.pi * radius
            me += storage * np.outer(N, N) * dV
            ke += permeability * (grad.T @ grad) * dV
        return me, ke

    if etype == "QUAD4":
        return _quad4_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability)
    if etype == "QUAD8":
        return _quad8_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability)
    if etype in {"TRI3", "TRI6"}:
        return _tri_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability, axisymmetric=False)
    me = np.zeros((nnode, nnode), dtype=float)
    ke = np.zeros((nnode, nnode), dtype=float)
    for gp in integration_points(element_type, "FULL"):
        xi, eta, weight = gp
        N, dN_dnatural = shape_functions(element_type, xi, eta)
        jac = dN_dnatural @ coords
        detJ = float(np.linalg.det(jac))
        if detJ <= 0.0:
            raise FEM2DError(f"{element_type}: detJ must be positive, got {detJ:.6e}")
        grad = np.linalg.inv(jac) @ dN_dnatural
        dV = detJ * weight * material.thickness
        me += storage * np.outer(N, N) * dV
        ke += permeability * (grad.T @ grad) * dV
    return me, ke


def _biot_coupling_block(
    element_type: str,
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    alpha: float,
    integration: str,
    *,
    axisymmetric: bool,
) -> np.ndarray:
    etype = element_type.upper()
    m = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
    if axisymmetric:
        if etype == "QUAD4":
            return _quad4_axisymmetric_biot_matrix_fast(coords, material, alpha)
        if etype == "QUAD8":
            return _quad8_axisymmetric_biot_matrix_fast(coords, material, alpha)
        if etype in {"TRI3", "TRI6"}:
            return _tri_biot_matrix_fast(coords, material, alpha, axisymmetric=True)
        nnode = int(coords.shape[0])
        block = np.zeros((2 * nnode, nnode), dtype=float)
        for gp in integration_points(element_type, "FULL"):
            B4, detJ, N, radius = axisymmetric_strain_displacement_matrix(element_type, coords, gp)
            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
            block += B4.T @ (alpha * np.outer(m, N)) * dV
        return block

    if etype == "QUAD4":
        return _quad4_biot_matrix_fast(coords, material, alpha)
    if etype == "QUAD8":
        return _quad8_biot_matrix_fast(coords, material, alpha, integration)
    if etype in {"TRI3", "TRI6"}:
        return _tri_biot_matrix_fast(coords, material, alpha, axisymmetric=False)
    nnode = int(coords.shape[0])
    block = np.zeros((2 * nnode, nnode), dtype=float)
    for gp in integration_points(element_type, "FULL"):
        B4, detJ, N = strain_displacement_matrix(element_type, coords, gp)
        dV = detJ * gp[2] * material.thickness
        block += B4.T @ (alpha * np.outer(m, N)) * dV
    return block


def assemble_liquefaction_pressure_terms(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    hydro: Mapping[str, Any],
    u: np.ndarray,
    old_u: np.ndarray,
    p: np.ndarray,
    *,
    dt: float,
    storage: float,
    axisymmetric: bool = False,
) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    """Return pressure residual terms for cyclic ru generation and dissipation.

    The pressure equation is augmented as
    ``M(p-p_old)/dt + ... + C_liq p = q_liq``.  Generation is driven by the
    cyclic shear increment (or an explicit cycles_per_step input), while
    dissipation remains on the left-hand side so it participates in the
    coupled pressure solve instead of being applied as a post-process.
    """

    requested = hydro.get("liquefaction_coupling", hydro.get("coupled_liquefaction", True))
    if isinstance(requested, Mapping):
        enabled = bool(requested.get("enabled", True))
        hydro_liq = requested
    else:
        enabled = bool(requested)
        hydro_liq = hydro.get("liquefaction", {})
    nnode_total = len(mesh.node_ids)
    if not enabled:
        return csr_matrix((nnode_total, nnode_total), dtype=float), np.zeros(nnode_total, dtype=float), {"enabled": False, "enabled_points": 0}

    node_index = mesh.node_index
    rhs = np.zeros(nnode_total, dtype=float)
    y_top = float(np.max(mesh.coords[:, 1])) if mesh.coords.size else 0.0
    info: dict[str, Any] = {
        "enabled": True,
        "axisymmetric": axisymmetric,
        "enabled_points": 0,
        "batched_elements": 0,
        "fallback_elements": 0,
        "batch_min_elements": _PRESSURE_BATCH_MIN_ELEMENTS,
        "batch_groups": [],
        "generation_source": 0.0,
        "generation_pressure_increment_sum": 0.0,
        "dissipation_matrix_sum": 0.0,
        "cyclic_increment_sum": 0.0,
        "max_cycle_increment": 0.0,
    }
    override = hydro_liq if isinstance(hydro_liq, Mapping) else {}
    element_blocks: list[dict[str, Any]] = []
    matrix_block_lookup: dict[int, int] = {}
    matrix_conn_blocks: list[np.ndarray] = []
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        if not _material_has_liquefaction(material):
            continue
        conn = _element_node_indices(element.nodes, node_index)
        conn_arr = np.asarray(conn, dtype=int)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        ue = u[dofs]
        old_ue = old_u[dofs]
        params = material.advanced_params or {}
        liq = params.get("liquefaction")
        liq_map = dict(liq if isinstance(liq, Mapping) else params)
        liq_map.update(override)
        crr = max(_param_float(liq_map, ("cyclic_resistance_ratio", "CRR", "RL20"), 0.0), 0.0)
        csr = max(_param_float(liq_map, ("cyclic_stress_ratio", "CSR"), 0.0), 0.0)
        generation_rate = max(_param_float(liq_map, ("generation_rate", "ru_generation_rate"), 0.25), 0.0)
        dissipation_rate = max(_param_float(liq_map, ("dissipation_rate", "ru_dissipation_rate"), 0.0), 0.0)
        gamma_ref = max(_param_float(liq_map, ("gamma_ref", "reference_strain", "gamma50"), _param_float(params, ("gamma_ref", "reference_strain", "gamma50"), 1.0e-3)), 1.0e-12)
        manual_cycle_increment = max(_param_float(liq_map, ("cycle_increment", "cycles_per_step", "dN"), 0.0), 0.0)
        if crr <= 0.0 and generation_rate > 0.0:
            crr = 1.0
        block_index = len(element_blocks)
        element_blocks.append(
            {
                "element_type": str(element.type).upper(),
                "conn": conn_arr,
                "dofs": np.asarray(dofs, dtype=np.int64),
                "coords": np.asarray(coords, dtype=float),
                "du": np.asarray(ue - old_ue, dtype=float),
                "material": material,
                "liq_map": liq_map,
                "thickness": float(material.thickness),
                "unit_weight": float(material.gamma),
                "initial_effective_stress": _param_float(liq_map, ("initial_effective_stress", "sigma_v_eff", "effective_stress"), 0.0),
                "crr": float(crr),
                "csr": float(csr),
                "generation_rate": float(generation_rate),
                "dissipation_rate": float(dissipation_rate),
                "gamma_ref": float(gamma_ref),
                "manual_cycle_increment": float(manual_cycle_increment),
            }
        )
        if dissipation_rate > 0.0:
            matrix_block_lookup[block_index] = len(matrix_conn_blocks)
            matrix_conn_blocks.append(conn_arr)
    shape = (nnode_total, nnode_total)
    pattern = SparseAssemblyPattern.from_square_blocks(matrix_conn_blocks, shape) if matrix_conn_blocks else None
    matrix_values = pattern.empty_flat_values() if pattern is not None else np.zeros(0, dtype=float)
    batched = np.zeros(len(element_blocks), dtype=bool)

    for element_type in ("QUAD4", "QUAD8"):
        block_indices = [index for index, block in enumerate(element_blocks) if block["element_type"] == element_type]
        if len(block_indices) < _PRESSURE_BATCH_MIN_ELEMENTS:
            continue
        conn_stack = np.stack([element_blocks[index]["conn"] for index in block_indices]).astype(np.int64, copy=False)
        coords_stack = np.ascontiguousarray(np.stack([element_blocks[index]["coords"] for index in block_indices]), dtype=np.float64)
        du_stack = np.ascontiguousarray(np.stack([element_blocks[index]["du"] for index in block_indices]), dtype=np.float64)
        thickness = np.asarray([element_blocks[index]["thickness"] for index in block_indices], dtype=np.float64)
        unit_weight = np.asarray([element_blocks[index]["unit_weight"] for index in block_indices], dtype=np.float64)
        initial_effective = np.asarray([element_blocks[index]["initial_effective_stress"] for index in block_indices], dtype=np.float64)
        crr_values = np.asarray([element_blocks[index]["crr"] for index in block_indices], dtype=np.float64)
        csr_values = np.asarray([element_blocks[index]["csr"] for index in block_indices], dtype=np.float64)
        generation_values = np.asarray([element_blocks[index]["generation_rate"] for index in block_indices], dtype=np.float64)
        dissipation_values = np.asarray([element_blocks[index]["dissipation_rate"] for index in block_indices], dtype=np.float64)
        gamma_ref_values = np.asarray([element_blocks[index]["gamma_ref"] for index in block_indices], dtype=np.float64)
        manual_cycle_values = np.asarray([element_blocks[index]["manual_cycle_increment"] for index in block_indices], dtype=np.float64)
        if element_type == "QUAD4":
            rhs_blocks, diss_blocks, metrics, min_det, min_radius = _quad4_liquefaction_pressure_batch_numba(
                coords_stack,
                du_stack,
                thickness,
                unit_weight,
                initial_effective,
                crr_values,
                csr_values,
                generation_values,
                dissipation_values,
                gamma_ref_values,
                manual_cycle_values,
                float(storage),
                float(dt),
                bool(axisymmetric),
                float(y_top),
            )
        else:
            rhs_blocks, diss_blocks, metrics, min_det, min_radius = _quad8_liquefaction_pressure_batch_numba(
                coords_stack,
                du_stack,
                thickness,
                unit_weight,
                initial_effective,
                crr_values,
                csr_values,
                generation_values,
                dissipation_values,
                gamma_ref_values,
                manual_cycle_values,
                float(storage),
                float(dt),
                bool(axisymmetric),
                float(y_top),
            )
        _validate_batched_geometry(element_type, float(min_det), float(min_radius), axisymmetric)
        np.add.at(rhs, conn_stack.ravel(), rhs_blocks.ravel())
        if pattern is not None:
            local_matrix_indices = [local for local, index in enumerate(block_indices) if index in matrix_block_lookup]
            if local_matrix_indices:
                pattern_indices = np.asarray([matrix_block_lookup[block_indices[local]] for local in local_matrix_indices], dtype=np.int64)
                _fill_batch_flat_values(pattern, matrix_values, pattern_indices, diss_blocks[np.asarray(local_matrix_indices, dtype=np.int64)])
        batched[np.asarray(block_indices, dtype=np.int64)] = True
        info["batched_elements"] = int(info["batched_elements"]) + len(block_indices)
        info["batch_groups"].append({"element_type": element_type, "elements": len(block_indices), "axisymmetric": bool(axisymmetric)})
        info["generation_source"] = float(info["generation_source"]) + float(metrics[0])
        info["generation_pressure_increment_sum"] = float(info["generation_pressure_increment_sum"]) + float(metrics[1])
        info["dissipation_matrix_sum"] = float(info["dissipation_matrix_sum"]) + float(metrics[2])
        info["enabled_points"] = int(info["enabled_points"]) + int(metrics[3])
        info["cyclic_increment_sum"] = float(info["cyclic_increment_sum"]) + float(metrics[4])
        info["max_cycle_increment"] = max(float(info["max_cycle_increment"]), float(metrics[5]))

    for block_index, block in enumerate(element_blocks):
        if batched[block_index]:
            continue
        info["fallback_elements"] = int(info["fallback_elements"]) + 1
        conn_arr = np.asarray(block["conn"], dtype=int)
        coords = np.asarray(block["coords"], dtype=float)
        du = np.asarray(block["du"], dtype=float)
        nnode = len(conn_arr)
        local_d = np.zeros((nnode, nnode), dtype=float)
        local_rhs = np.zeros(nnode, dtype=float)
        material = block["material"]
        liq_map = block["liq_map"]
        crr = float(block["crr"])
        csr = float(block["csr"])
        generation_rate = float(block["generation_rate"])
        dissipation_rate = float(block["dissipation_rate"])
        gamma_ref = float(block["gamma_ref"])
        manual_cycle_increment = float(block["manual_cycle_increment"])
        for gp in integration_points(str(block["element_type"]), "FULL"):
            if axisymmetric:
                B4, detJ, N, radius = axisymmetric_strain_displacement_matrix(str(block["element_type"]), coords, gp)
                dV = detJ * gp[2] * float(block["thickness"]) * 2.0 * math.pi * radius
            else:
                B4, detJ, N = strain_displacement_matrix(str(block["element_type"]), coords, gp)
                dV = detJ * gp[2] * float(block["thickness"])
            xy = N @ coords
            strain_increment = B4 @ du
            gamma_increment = _equivalent_shear_strain(strain_increment)
            cycle_increment = max(gamma_increment / max(4.0 * gamma_ref, 1.0e-12), manual_cycle_increment)
            if cycle_increment <= 0.0 and dissipation_rate <= 0.0:
                continue
            effective_ref = _liquefaction_effective_stress_reference(material, liq_map, xy, y_top)
            demand = 0.0 if crr <= 0.0 else max(csr / crr, 0.0)
            ru_generation_increment = min(0.99, generation_rate * demand * cycle_increment)
            pressure_increment = effective_ref * ru_generation_increment
            if pressure_increment > 0.0:
                source_scale = storage * pressure_increment * dV / dt
                local_rhs += N * source_scale
                info["generation_source"] = float(info["generation_source"]) + float(source_scale)
                info["generation_pressure_increment_sum"] = float(info["generation_pressure_increment_sum"]) + float(pressure_increment)
            if dissipation_rate > 0.0:
                local = storage * dissipation_rate * np.outer(N, N) * dV
                local_d += local
                info["dissipation_matrix_sum"] = float(info["dissipation_matrix_sum"]) + float(np.sum(local))
            info["enabled_points"] = int(info["enabled_points"]) + 1
            info["cyclic_increment_sum"] = float(info["cyclic_increment_sum"]) + float(cycle_increment)
            info["max_cycle_increment"] = max(float(info["max_cycle_increment"]), float(cycle_increment))
        if np.any(local_rhs):
            rhs[conn_arr] += local_rhs
        matrix_index = matrix_block_lookup.get(block_index)
        if pattern is not None and matrix_index is not None:
            pattern.fill_block(matrix_values, matrix_index, local_d)

    matrix = pattern.assemble_flat_values(matrix_values) if pattern is not None else csr_matrix(shape, dtype=float)
    info["direct_fill"] = {"enabled": False, "reason": "no_dissipation_matrix"} if pattern is None else pattern.direct_fill_info()
    return matrix, rhs, info

def assemble_axisymmetric_biot_coupling_matrix(mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial], alpha: float = 1.0) -> csr_matrix:
    node_index = mesh.node_index
    builder = SparseAssemblyBuilder()
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        pressure_nodes = conn
        block = _biot_coupling_block(element.type, coords, material, alpha, normalize_integration(element.integration), axisymmetric=True)
        builder.add_block(dofs, pressure_nodes, block)
    return builder.to_csr((len(mesh.node_ids) * 2, len(mesh.node_ids)))

def assemble_axisymmetric_pressure_matrices(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    *,
    storage: float,
    permeability: float,
) -> tuple[csr_matrix, csr_matrix]:
    node_index = mesh.node_index
    nnode_total = len(mesh.node_ids)
    mass_builder = SparseAssemblyBuilder()
    cond_builder = SparseAssemblyBuilder()
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        me, ke = _pressure_matrix_block(element.type, coords, material, storage=storage, permeability=permeability, axisymmetric=True)
        conn_arr = np.asarray(conn, dtype=int)
        mass_builder.add_block(conn_arr, conn_arr, me)
        cond_builder.add_block(conn_arr, conn_arr, ke)
    shape = (nnode_total, nnode_total)
    return mass_builder.to_csr(shape), cond_builder.to_csr(shape)

def assemble_axisymmetric_pressure_boundary_terms(mesh: Mesh2D, hydro: Mapping[str, Any], pressure: np.ndarray | None = None) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    nnode_total = len(mesh.node_ids)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(nnode_total, dtype=float)
    flux_total = 0.0
    robin_total = 0.0
    seepage_count = 0
    seepage_active_edges = 0
    seepage_inactive_edges = 0
    flux_specs = hydro.get("pore_flux_bcs", hydro.get("flux_bcs", hydro.get("pore_flux", hydro.get("flux", []))))
    robin_specs = hydro.get("pore_robin_bcs", hydro.get("robin_bcs", hydro.get("pore_robin", hydro.get("robin", []))))

    for spec in _ensure_list(flux_specs):
        if not isinstance(spec, Mapping):
            raise FEM2DError("each axisymmetric pore flux boundary condition must be a mapping")
        q = float(spec.get("flux", spec.get("q", spec.get("value", 0.0))))
        thickness = float(spec.get("thickness", 1.0))
        for edge in _pressure_edges(mesh, spec):
            surface = _axisymmetric_edge_measure(mesh, edge, thickness)
            weights = _edge_lumped_weights(edge)
            for nid, weight in zip(edge, weights):
                rhs[mesh.node_index[nid]] += q * surface * float(weight)
            flux_total += q * surface

    for spec in _ensure_list(robin_specs):
        if not isinstance(spec, Mapping):
            raise FEM2DError("each axisymmetric pore Robin boundary condition must be a mapping")
        beta = float(spec.get("beta", spec.get("conductance", spec.get("leakage", spec.get("coefficient", 0.0)))))
        pref = float(spec.get("pressure", spec.get("p_ext", spec.get("p_ref", spec.get("value", 0.0)))))
        thickness = float(spec.get("thickness", 1.0))
        seepage = bool(spec.get("seepage_face", spec.get("seepage", False)))
        if beta < 0.0:
            raise FEM2DError("axisymmetric pore Robin conductance must be non-negative")
        for edge in _pressure_edges(mesh, spec):
            if seepage:
                seepage_count += 1
                if pressure is not None:
                    edge_indices = [mesh.node_index[nid] for nid in edge]
                    weights = _edge_lumped_weights(edge)
                    p_avg = float(np.dot(weights, pressure[edge_indices]))
                    if p_avg <= pref:
                        seepage_inactive_edges += 1
                        continue
                seepage_active_edges += 1
            surface = _axisymmetric_edge_measure(mesh, edge, thickness)
            edge_indices = [mesh.node_index[nid] for nid in edge]
            weights = _edge_lumped_weights(edge)
            local = beta * surface * _edge_consistent_robin_matrix(edge)
            local_rhs = pref * beta * surface * weights
            for a, row in enumerate(edge_indices):
                rhs[row] += float(local_rhs[a])
                for b, col in enumerate(edge_indices):
                    rows.append(row)
                    cols.append(col)
                    data.append(float(local[a, b]))
            robin_total += beta * surface

    matrix = coo_matrix((data, (rows, cols)), shape=(nnode_total, nnode_total)).tocsr()
    return matrix, rhs, {
        "flux_total": flux_total,
        "robin_conductance_total": robin_total,
        "flux_count": len(_ensure_list(flux_specs)),
        "robin_count": len(_ensure_list(robin_specs)),
        "seepage_count": seepage_count,
        "seepage_active_edges": seepage_active_edges,
        "seepage_inactive_edges": seepage_inactive_edges,
    }

def _axisymmetric_edge_measure(mesh: Mesh2D, edge: tuple[str, ...], thickness: float = 1.0) -> float:
    length = _edge_length(mesh, edge)
    weights = _edge_lumped_weights(edge)
    radii = np.asarray([mesh.coords[mesh.node_index[nid], 0] for nid in edge], dtype=float)
    radius_avg = max(float(np.dot(weights, radii)), np.finfo(float).eps)
    return 2.0 * math.pi * radius_avg * length * thickness


def assemble_pore_pressure_load_cached(
    cache: PorePressureLoadAssemblyCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    pore_pressure: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    pressure = np.asarray(pore_pressure, dtype=float).reshape(-1)
    if pressure.shape != (cache.shape[1],):
        raise FEM2DError("pore_pressure size must match node count")
    load = np.zeros(cache.shape[0], dtype=float)
    batched = np.zeros(len(cache.element_blocks), dtype=bool)
    for batch in cache.batch_blocks:
        coords, thickness = _biot_batch_inputs(batch, mesh, materials)
        if batch.element_type == "TRI3":
            blocks_flat, min_det, min_radius = _tri_biot_coupling_batch_numba(
                coords,
                float(alpha),
                thickness,
                bool(cache.biot_cache.axisymmetric),
                3,
            )
        elif batch.element_type == "TRI6":
            blocks_flat, min_det, min_radius = _tri_biot_coupling_batch_numba(
                coords,
                float(alpha),
                thickness,
                bool(cache.biot_cache.axisymmetric),
                6,
            )
        elif batch.element_type == "QUAD4":
            blocks_flat, min_det, min_radius = _quad4_biot_coupling_batch_numba(
                coords,
                float(alpha),
                thickness,
                bool(cache.biot_cache.axisymmetric),
            )
        elif batch.element_type == "QUAD8":
            mode_codes, volumetric_projectors = _quad8_biot_batch_material_inputs(batch, mesh, materials)
            blocks_flat, min_det, min_radius, min_bbar_volume = _quad8_biot_coupling_batch_numba(
                coords,
                float(alpha),
                thickness,
                mode_codes,
                volumetric_projectors,
                bool(cache.biot_cache.axisymmetric),
            )
            if not cache.biot_cache.axisymmetric and min_bbar_volume <= 0.0:
                raise FEM2DError("QUAD8: non-positive element measure")
        else:
            continue
        _validate_batched_geometry(batch.element_type, float(min_det), float(min_radius), cache.biot_cache.axisymmetric)
        _scatter_biot_batch_pressure_load(load, batch, blocks_flat, pressure)
        batched[batch.block_indices] = True
    for block_index, block in enumerate(cache.element_blocks):
        if batched[block_index]:
            continue
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        coords = mesh.coords[block.pressure_nodes]
        local = _biot_coupling_block(
            element.type,
            coords,
            material,
            alpha,
            normalize_integration(element.integration),
            axisymmetric=cache.biot_cache.axisymmetric,
        )
        np.add.at(load, block.displacement_dofs, local @ pressure[block.pressure_nodes])
    return load


def _scatter_biot_batch_pressure_load(
    out: np.ndarray,
    batch: BiotCouplingBatchCache,
    blocks_flat: np.ndarray,
    pressure: np.ndarray,
) -> None:
    pressure_nodes = np.asarray(batch.pressure_nodes, dtype=np.int64)
    node_count = int(pressure_nodes.shape[1])
    displacement_count = node_count * 2
    blocks = np.asarray(blocks_flat, dtype=np.float64).reshape((-1, displacement_count, node_count))
    local_pressure = np.asarray(pressure, dtype=float)[pressure_nodes]
    local_load = np.einsum("eij,ej->ei", blocks, local_pressure)
    dofs = np.empty((pressure_nodes.shape[0], displacement_count), dtype=np.int64)
    dofs[:, 0::2] = 2 * pressure_nodes
    dofs[:, 1::2] = 2 * pressure_nodes + 1
    np.add.at(out, dofs.ravel(), local_load.ravel())


def assemble_pore_pressure_load(mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial], pore_pressure: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    if pore_pressure.shape != (len(mesh.node_ids),):
        raise FEM2DError("pore_pressure size must match node count")
    cache = build_pore_pressure_load_cache(mesh)
    return assemble_pore_pressure_load_cached(cache, mesh, materials, pore_pressure, alpha=alpha)

def assemble_biot_coupling_matrix(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    alpha: float = 1.0,
    structural_elements: list[StructuralElement2D] | None = None,
) -> csr_matrix:
    node_index = mesh.node_index
    builder = SparseAssemblyBuilder()
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        dofs = _dofs_from_node_indices(conn)
        pressure_nodes = conn
        block = _biot_coupling_block(element.type, coords, material, alpha, normalize_integration(element.integration), axisymmetric=False)
        builder.add_block(dofs, pressure_nodes, block)
    return builder.to_csr((structural_total_dofs(mesh, structural_elements), len(mesh.node_ids)))

def solve_consolidation_pressure(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    stage_cfg: Mapping[str, Any],
    previous_pressure: np.ndarray | None = None,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    hydro = stage_cfg.get("hydro", stage_cfg.get("consolidation", stage_cfg))
    if not isinstance(hydro, Mapping):
        hydro = {}
    dt = float(hydro.get("dt", hydro.get("time_step", stage_cfg.get("dt", 1.0))))
    steps = int(hydro.get("steps", hydro.get("n_steps", stage_cfg.get("steps", 1))))
    if dt <= 0.0 or steps <= 0:
        raise FEM2DError("consolidation dt and steps must be positive")
    storage = float(hydro.get("storage", hydro.get("specific_storage", 1.0)))
    permeability = float(hydro.get("permeability", hydro.get("k", 1.0)))
    if storage <= 0.0 or permeability < 0.0:
        raise FEM2DError("consolidation storage must be positive and permeability non-negative")

    p = _initial_pore_pressure(mesh, hydro, previous_pressure)
    fixed = _collect_pressure_constraints(mesh, hydro.get("pressure_bcs", hydro.get("pore_pressure_bcs", hydro.get("drainage", []))))
    pressure_cache = build_pressure_matrix_assembly_cache(mesh)
    mass, conductivity = assemble_pressure_matrices_cached(pressure_cache, mesh, materials, storage=storage, permeability=permeability)
    boundary_cache = build_pressure_boundary_term_cache(mesh, hydro)
    robin, boundary_rhs, boundary_info = assemble_pressure_boundary_terms_cached(boundary_cache)
    matrix = (mass + dt * (conductivity + robin)).tocsr()
    residual_norm = math.inf
    residual_sum = math.inf
    for _step in range(steps):
        old_p = p.copy()
        rhs = mass @ p + dt * boundary_rhs
        p = _solve_scalar_constraints(matrix, rhs, fixed)
        residual = mass @ (p - old_p) / dt + conductivity @ p + robin @ p - boundary_rhs
        free, _fixed_pressure = _free_index_arrays(len(mesh.node_ids), fixed, label="fixed pressure")
        residual_norm = float(np.linalg.norm(residual[free])) if free.size else 0.0
        residual_sum = float(np.sum(residual))
    return p, dt * steps, {
        "dt": dt,
        "steps": steps,
        "storage": storage,
        "permeability": permeability,
        "fixed_pressure_nodes": len(fixed),
        "boundary": boundary_info,
        "pressure_matrix_cache": pressure_cache.info(),
        "mass_balance": residual_norm,
        "mass_balance_residual_sum": residual_sum,
    }

def assemble_pressure_matrices(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    *,
    storage: float,
    permeability: float,
) -> tuple[csr_matrix, csr_matrix]:
    node_index = mesh.node_index
    nnode_total = len(mesh.node_ids)
    mass_builder = SparseAssemblyBuilder()
    cond_builder = SparseAssemblyBuilder()
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        me, ke = _pressure_matrix_block(element.type, coords, material, storage=storage, permeability=permeability, axisymmetric=False)
        conn_arr = np.asarray(conn, dtype=int)
        mass_builder.add_block(conn_arr, conn_arr, me)
        cond_builder.add_block(conn_arr, conn_arr, ke)
    shape = (nnode_total, nnode_total)
    return mass_builder.to_csr(shape), cond_builder.to_csr(shape)

def assemble_pressure_boundary_terms(mesh: Mesh2D, hydro: Mapping[str, Any], pressure: np.ndarray | None = None) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    nnode_total = len(mesh.node_ids)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    rhs = np.zeros(nnode_total, dtype=float)
    flux_total = 0.0
    robin_total = 0.0
    seepage_count = 0
    seepage_active_edges = 0
    seepage_inactive_edges = 0
    flux_specs = hydro.get("pore_flux_bcs", hydro.get("flux_bcs", hydro.get("pore_flux", hydro.get("flux", []))))
    robin_specs = hydro.get("pore_robin_bcs", hydro.get("robin_bcs", hydro.get("pore_robin", hydro.get("robin", []))))

    for spec in _ensure_list(flux_specs):
        if not isinstance(spec, Mapping):
            raise FEM2DError("each pore flux boundary condition must be a mapping")
        q = float(spec.get("flux", spec.get("q", spec.get("value", 0.0))))
        thickness = float(spec.get("thickness", 1.0))
        for edge in _pressure_edges(mesh, spec):
            length = _edge_length(mesh, edge)
            weights = _edge_lumped_weights(edge)
            for nid, weight in zip(edge, weights):
                rhs[mesh.node_index[nid]] += q * length * thickness * float(weight)
            flux_total += q * length * thickness

    for spec in _ensure_list(robin_specs):
        if not isinstance(spec, Mapping):
            raise FEM2DError("each pore Robin boundary condition must be a mapping")
        beta = float(spec.get("beta", spec.get("conductance", spec.get("leakage", spec.get("coefficient", 0.0)))))
        pref = float(spec.get("pressure", spec.get("p_ext", spec.get("p_ref", spec.get("value", 0.0)))))
        thickness = float(spec.get("thickness", 1.0))
        seepage = bool(spec.get("seepage_face", spec.get("seepage", False)))
        if beta < 0.0:
            raise FEM2DError("pore Robin conductance must be non-negative")
        for edge in _pressure_edges(mesh, spec):
            if seepage:
                seepage_count += 1
                if pressure is not None:
                    edge_indices = [mesh.node_index[nid] for nid in edge]
                    weights = _edge_lumped_weights(edge)
                    p_avg = float(np.dot(weights, pressure[edge_indices]))
                    if p_avg <= pref:
                        seepage_inactive_edges += 1
                        continue
                seepage_active_edges += 1
            length = _edge_length(mesh, edge)
            edge_indices = [mesh.node_index[nid] for nid in edge]
            weights = _edge_lumped_weights(edge)
            local = beta * length * thickness * _edge_consistent_robin_matrix(edge)
            local_rhs = pref * beta * length * thickness * weights
            for a, row in enumerate(edge_indices):
                rhs[row] += float(local_rhs[a])
                for b, col in enumerate(edge_indices):
                    rows.append(row)
                    cols.append(col)
                    data.append(float(local[a, b]))
            robin_total += beta * length * thickness

    matrix = coo_matrix((data, (rows, cols)), shape=(nnode_total, nnode_total)).tocsr()
    info = {
        "flux_total": flux_total,
        "robin_conductance_total": robin_total,
        "flux_count": len(_ensure_list(flux_specs)),
        "robin_count": len(_ensure_list(robin_specs)),
        "seepage_count": seepage_count,
        "seepage_active_edges": seepage_active_edges,
        "seepage_inactive_edges": seepage_inactive_edges,
    }
    return matrix, rhs, info

def _solve_scalar_constraints(matrix: csr_matrix, rhs: np.ndarray, fixed_values: Mapping[int, float]) -> np.ndarray:
    return _solve_sparse_with_constraints_core(matrix, rhs, fixed_values, stage_name="scalar", solver={"linear": {"method": "direct"}})
