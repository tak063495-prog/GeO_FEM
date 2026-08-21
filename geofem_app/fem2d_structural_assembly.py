"""Structural stiffness, mass, and load assembly helpers for 2D FEM solvers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.sparse import csr_matrix, diags

from .fem2d_elements import (
    _quad4_axisymmetric_element_stiffness_fast,
    _quad4_consistent_mass_matrix_fast,
    _quad8_axisymmetric_edge_traction_fast,
    _quad8_consistent_mass_matrix_fast,
    axisymmetric_element_stiffness,
    axisymmetric_strain_displacement_matrix,
    element_stiffness,
    integration_points,
    shape_functions,
    strain_displacement_matrix,
)
from .fem2d_element_elastic_kernels import (
    _QUAD4_MODE_BBAR,
    _quad4_axisymmetric_element_stiffness_numba,
    _quad4_consistent_mass_matrix_numba,
    _quad4_element_stiffness_numba,
    _quad4_mode_code,
    _quad8_axisymmetric_element_stiffness_numba,
    _quad8_consistent_mass_matrix_numba,
    _quad8_element_stiffness_numba,
)
from .fem2d_interfaces import interface_force_tangent
from .fem2d_mesh import _edge_set, _target_nodes
from .fem2d_structural import (
    structural_element_dofs,
    structural_element_equivalent_load,
    structural_element_force_tangent,
    structural_has_nonlinear,
    structural_rotation_dof_map,
    structural_total_dofs,
)
from .fem2d_types import FEM2DError, ElasticPlaneStrainMaterial, Interface2D, Mesh2D, StructuralElement2D, njit, normalize_integration
from .fem2d_utils import _dofs_from_node_indices, _element_node_indices, _ensure_list, _require_sequence
from .sparse_assembly import SparseAssemblyBuilder, SparseAssemblyPattern

STRUCTURAL_ASSEMBLY_FUNCTIONS = (
    "assemble_global_stiffness",
    "assemble_mass_matrix",
    "build_mass_matrix_assembly_cache",
    "assemble_mass_matrix_cached",
    "_material_mass_density",
    "_structural_mass_per_length",
    "assemble_axisymmetric_stiffness",
    "build_axisymmetric_stiffness_assembly_cache",
    "assemble_axisymmetric_stiffness_cached",
    "assemble_axisymmetric_load_vector",
    "_add_axisymmetric_body_weight",
    "_add_axisymmetric_edge_traction",
    "build_global_stiffness_assembly_cache",
    "assemble_global_stiffness_cached",
    "assemble_load_vector",
    "build_load_vector_assembly_cache",
    "assemble_load_vector_cached",
    "_add_body_weight",
    "_body_force_density",
    "_body_load_targets_element",
    "_edge_traction_components",
    "_add_edge_traction",
)

_MASS_BATCH_MIN_ELEMENTS = 8


@dataclass(frozen=True)
class ElementStiffnessBlockCache:
    element_index: int
    conn: np.ndarray
    dofs: np.ndarray
    linear_stiffness: np.ndarray | None = None


@dataclass(frozen=True)
class InterfaceStiffnessBlockCache:
    interface_index: int
    dofs: np.ndarray
    linear_stiffness: np.ndarray | None = None


@dataclass(frozen=True)
class StructuralStiffnessBlockCache:
    structural_index: int
    dofs: np.ndarray
    linear_stiffness: np.ndarray | None = None


@dataclass(frozen=True)
class PrecomputedStiffnessBlockBatch:
    kind: str
    start_block: int
    block_count: int
    values: np.ndarray

    @property
    def flat_value_size(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class ElementMassBlockCache:
    element_index: int
    conn: np.ndarray
    dofs: np.ndarray


@dataclass(frozen=True)
class StructuralMassBlockCache:
    structural_index: int
    dofs: np.ndarray
    mass_matrix: np.ndarray | None = None


@dataclass(frozen=True)
class MassElementBatchCache:
    element_type: str
    block_indices: np.ndarray
    element_indices: np.ndarray
    conn: np.ndarray

    @property
    def element_count(self) -> int:
        return int(self.block_indices.size)


@dataclass(frozen=True)
class Quad4ElasticBatchAssemblyCache:
    conn: np.ndarray
    d4: np.ndarray
    cdev: np.ndarray
    cvol: np.ndarray
    pvol: np.ndarray
    idev: np.ndarray
    thickness: np.ndarray
    mode_codes: np.ndarray

    @property
    def element_count(self) -> int:
        return int(self.conn.shape[0])


@dataclass(frozen=True)
class Quad8ElasticBatchAssemblyCache:
    conn: np.ndarray
    d4: np.ndarray
    cdev: np.ndarray
    cvol: np.ndarray
    pvol: np.ndarray
    idev: np.ndarray
    thickness: np.ndarray
    mode_codes: np.ndarray

    @property
    def element_count(self) -> int:
        return int(self.conn.shape[0])


@dataclass(frozen=True)
class TriElasticBatchAssemblyCache:
    conn: np.ndarray
    d4: np.ndarray
    cdev: np.ndarray
    cvol: np.ndarray
    pvol: np.ndarray
    idev: np.ndarray
    thickness: np.ndarray
    mode_codes: np.ndarray

    @property
    def element_count(self) -> int:
        return int(self.conn.shape[0])


@dataclass(frozen=True)
class Quad4AxisymmetricElasticBatchAssemblyCache:
    conn: np.ndarray
    d4: np.ndarray
    thickness: np.ndarray

    @property
    def element_count(self) -> int:
        return int(self.conn.shape[0])


@dataclass(frozen=True)
class Quad8AxisymmetricElasticBatchAssemblyCache:
    conn: np.ndarray
    d4: np.ndarray
    cdev: np.ndarray
    cvol: np.ndarray
    pvol: np.ndarray
    idev: np.ndarray
    thickness: np.ndarray
    mode_codes: np.ndarray

    @property
    def element_count(self) -> int:
        return int(self.conn.shape[0])


@dataclass(frozen=True)
class TriAxisymmetricElasticBatchAssemblyCache:
    conn: np.ndarray
    d4: np.ndarray
    cdev: np.ndarray
    cvol: np.ndarray
    pvol: np.ndarray
    idev: np.ndarray
    thickness: np.ndarray
    mode_codes: np.ndarray

    @property
    def element_count(self) -> int:
        return int(self.conn.shape[0])


@dataclass(frozen=True)
class GlobalStiffnessAssemblyCache:
    shape: tuple[int, int]
    pattern: SparseAssemblyPattern
    element_blocks: tuple[ElementStiffnessBlockCache, ...]
    interface_blocks: tuple[InterfaceStiffnessBlockCache, ...]
    structural_blocks: tuple[StructuralStiffnessBlockCache, ...]
    rotation_dofs: dict[str, int]
    quad4_elastic_batch: Quad4ElasticBatchAssemblyCache | None = None
    quad8_elastic_batch: Quad8ElasticBatchAssemblyCache | None = None
    tri3_elastic_batch: TriElasticBatchAssemblyCache | None = None
    tri6_elastic_batch: TriElasticBatchAssemblyCache | None = None
    precomputed_element_batches: tuple[PrecomputedStiffnessBlockBatch, ...] = ()
    precomputed_interface_batches: tuple[PrecomputedStiffnessBlockBatch, ...] = ()
    precomputed_structural_batches: tuple[PrecomputedStiffnessBlockBatch, ...] = ()

    @property
    def block_count(self) -> int:
        return len(self.element_blocks) + len(self.interface_blocks) + len(self.structural_blocks)

    @property
    def batched_element_count(self) -> int:
        quad4 = 0 if self.quad4_elastic_batch is None else self.quad4_elastic_batch.element_count
        quad8 = 0 if self.quad8_elastic_batch is None else self.quad8_elastic_batch.element_count
        tri3 = 0 if self.tri3_elastic_batch is None else self.tri3_elastic_batch.element_count
        tri6 = 0 if self.tri6_elastic_batch is None else self.tri6_elastic_batch.element_count
        return quad4 + quad8 + tri3 + tri6

    @property
    def precomputed_interface_count(self) -> int:
        return sum(1 for block in self.interface_blocks if block.linear_stiffness is not None)

    @property
    def precomputed_element_count(self) -> int:
        return sum(1 for block in self.element_blocks if block.linear_stiffness is not None)

    @property
    def precomputed_structural_count(self) -> int:
        return sum(1 for block in self.structural_blocks if block.linear_stiffness is not None)

    @property
    def precomputed_block_count(self) -> int:
        return self.precomputed_element_count + self.precomputed_interface_count + self.precomputed_structural_count

    @property
    def precomputed_linear_batch_count(self) -> int:
        return len(self.precomputed_element_batches) + len(self.precomputed_interface_batches) + len(self.precomputed_structural_batches)

    @property
    def precomputed_linear_batched_block_count(self) -> int:
        return int(sum(batch.block_count for batch in (*self.precomputed_element_batches, *self.precomputed_interface_batches, *self.precomputed_structural_batches)))

    @property
    def precomputed_linear_flat_value_size(self) -> int:
        return int(sum(batch.flat_value_size for batch in (*self.precomputed_element_batches, *self.precomputed_interface_batches, *self.precomputed_structural_batches)))

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "global_stiffness_assembly_cache",
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "block_count": int(self.block_count),
            "element_blocks": len(self.element_blocks),
            "interface_blocks": len(self.interface_blocks),
            "structural_blocks": len(self.structural_blocks),
            "batched_elastic_elements": int(self.batched_element_count),
            "batched_quad4_elastic_elements": 0 if self.quad4_elastic_batch is None else int(self.quad4_elastic_batch.element_count),
            "batched_quad8_elastic_elements": 0 if self.quad8_elastic_batch is None else int(self.quad8_elastic_batch.element_count),
            "batched_tri3_elastic_elements": 0 if self.tri3_elastic_batch is None else int(self.tri3_elastic_batch.element_count),
            "batched_tri6_elastic_elements": 0 if self.tri6_elastic_batch is None else int(self.tri6_elastic_batch.element_count),
            "precomputed_element_blocks": int(self.precomputed_element_count),
            "precomputed_interface_blocks": int(self.precomputed_interface_count),
            "precomputed_structural_blocks": int(self.precomputed_structural_count),
            "precomputed_linear_blocks": int(self.precomputed_block_count),
            "precomputed_linear_batches": {
                "enabled": self.precomputed_linear_batch_count > 0,
                "mode": "contiguous_flat_block_fill",
                "element_batches": len(self.precomputed_element_batches),
                "interface_batches": len(self.precomputed_interface_batches),
                "structural_batches": len(self.precomputed_structural_batches),
                "batch_count": int(self.precomputed_linear_batch_count),
                "batched_blocks": int(self.precomputed_linear_batched_block_count),
                "flat_value_size": int(self.precomputed_linear_flat_value_size),
            },
            "direct_fill": self.pattern.direct_fill_info(),
        }


@dataclass(frozen=True)
class AxisymmetricStiffnessAssemblyCache:
    shape: tuple[int, int]
    pattern: SparseAssemblyPattern
    element_blocks: tuple[ElementStiffnessBlockCache, ...]
    interface_blocks: tuple[InterfaceStiffnessBlockCache, ...]
    structural_blocks: tuple[StructuralStiffnessBlockCache, ...]
    quad4_elastic_batch: Quad4AxisymmetricElasticBatchAssemblyCache | None = None
    quad8_elastic_batch: Quad8AxisymmetricElasticBatchAssemblyCache | None = None
    tri3_elastic_batch: TriAxisymmetricElasticBatchAssemblyCache | None = None
    tri6_elastic_batch: TriAxisymmetricElasticBatchAssemblyCache | None = None
    precomputed_element_batches: tuple[PrecomputedStiffnessBlockBatch, ...] = ()
    precomputed_interface_batches: tuple[PrecomputedStiffnessBlockBatch, ...] = ()
    precomputed_structural_batches: tuple[PrecomputedStiffnessBlockBatch, ...] = ()

    @property
    def block_count(self) -> int:
        return len(self.element_blocks) + len(self.interface_blocks) + len(self.structural_blocks)

    @property
    def batched_element_count(self) -> int:
        quad4 = 0 if self.quad4_elastic_batch is None else self.quad4_elastic_batch.element_count
        quad8 = 0 if self.quad8_elastic_batch is None else self.quad8_elastic_batch.element_count
        tri3 = 0 if self.tri3_elastic_batch is None else self.tri3_elastic_batch.element_count
        tri6 = 0 if self.tri6_elastic_batch is None else self.tri6_elastic_batch.element_count
        return quad4 + quad8 + tri3 + tri6

    @property
    def precomputed_interface_count(self) -> int:
        return sum(1 for block in self.interface_blocks if block.linear_stiffness is not None)

    @property
    def precomputed_element_count(self) -> int:
        return sum(1 for block in self.element_blocks if block.linear_stiffness is not None)

    @property
    def precomputed_structural_count(self) -> int:
        return sum(1 for block in self.structural_blocks if block.linear_stiffness is not None)

    @property
    def precomputed_block_count(self) -> int:
        return self.precomputed_element_count + self.precomputed_interface_count + self.precomputed_structural_count

    @property
    def precomputed_linear_batch_count(self) -> int:
        return len(self.precomputed_element_batches) + len(self.precomputed_interface_batches) + len(self.precomputed_structural_batches)

    @property
    def precomputed_linear_batched_block_count(self) -> int:
        return int(sum(batch.block_count for batch in (*self.precomputed_element_batches, *self.precomputed_interface_batches, *self.precomputed_structural_batches)))

    @property
    def precomputed_linear_flat_value_size(self) -> int:
        return int(sum(batch.flat_value_size for batch in (*self.precomputed_element_batches, *self.precomputed_interface_batches, *self.precomputed_structural_batches)))

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "axisymmetric_stiffness_assembly_cache",
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "block_count": int(self.block_count),
            "element_blocks": len(self.element_blocks),
            "interface_blocks": len(self.interface_blocks),
            "structural_blocks": len(self.structural_blocks),
            "batched_axisymmetric_elastic_elements": int(self.batched_element_count),
            "batched_quad4_axisymmetric_elastic_elements": 0 if self.quad4_elastic_batch is None else int(self.quad4_elastic_batch.element_count),
            "batched_quad8_axisymmetric_elastic_elements": 0 if self.quad8_elastic_batch is None else int(self.quad8_elastic_batch.element_count),
            "batched_tri3_axisymmetric_elastic_elements": 0 if self.tri3_elastic_batch is None else int(self.tri3_elastic_batch.element_count),
            "batched_tri6_axisymmetric_elastic_elements": 0 if self.tri6_elastic_batch is None else int(self.tri6_elastic_batch.element_count),
            "precomputed_element_blocks": int(self.precomputed_element_count),
            "precomputed_interface_blocks": int(self.precomputed_interface_count),
            "precomputed_structural_blocks": int(self.precomputed_structural_count),
            "precomputed_linear_blocks": int(self.precomputed_block_count),
            "precomputed_linear_batches": {
                "enabled": self.precomputed_linear_batch_count > 0,
                "mode": "contiguous_flat_block_fill",
                "element_batches": len(self.precomputed_element_batches),
                "interface_batches": len(self.precomputed_interface_batches),
                "structural_batches": len(self.precomputed_structural_batches),
                "batch_count": int(self.precomputed_linear_batch_count),
                "batched_blocks": int(self.precomputed_linear_batched_block_count),
                "flat_value_size": int(self.precomputed_linear_flat_value_size),
            },
            "direct_fill": self.pattern.direct_fill_info(),
        }


@dataclass(frozen=True)
class MassMatrixAssemblyCache:
    shape: tuple[int, int]
    pattern: SparseAssemblyPattern
    element_blocks: tuple[ElementMassBlockCache, ...]
    structural_blocks: tuple[StructuralMassBlockCache, ...]
    batch_blocks: tuple[MassElementBatchCache, ...] = ()

    @property
    def block_count(self) -> int:
        return len(self.element_blocks) + len(self.structural_blocks)

    @property
    def batched_element_count(self) -> int:
        return int(sum(block.element_count for block in self.batch_blocks))

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "mass_matrix_assembly_cache",
            "shape": [int(self.shape[0]), int(self.shape[1])],
            "block_count": int(self.block_count),
            "element_blocks": len(self.element_blocks),
            "structural_blocks": len(self.structural_blocks),
            "precomputed_structural_mass_blocks": sum(1 for block in self.structural_blocks if block.mass_matrix is not None),
            "batch_min_elements": _MASS_BATCH_MIN_ELEMENTS,
            "batched_elements": int(self.batched_element_count),
            "batch_groups": _mass_batch_group_info(self.batch_blocks),
            "direct_fill": self.pattern.direct_fill_info(),
        }


@dataclass(frozen=True)
class BodyLoadBlockCache:
    element_index: int
    conn: np.ndarray
    dofs: np.ndarray
    gx: float
    gy: float
    scale: float
    target: dict[str, Any]


@dataclass(frozen=True)
class EdgeLoadBlockCache:
    node_indices: np.ndarray
    dofs: np.ndarray
    tx1: float
    ty1: float
    tx2: float
    ty2: float


@dataclass(frozen=True)
class LoadVectorAssemblyCache:
    ndof: int
    node_load_vector: np.ndarray
    body_blocks: tuple[BodyLoadBlockCache, ...]
    edge_blocks: tuple[EdgeLoadBlockCache, ...]
    load_count: int

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "cache_kind": "load_vector_assembly_cache",
            "ndof": int(self.ndof),
            "load_count": int(self.load_count),
            "node_vector_cached": bool(np.any(self.node_load_vector)),
            "body_blocks": len(self.body_blocks),
            "edge_blocks": len(self.edge_blocks),
            "geometry_dependent": bool(self.body_blocks or self.edge_blocks),
            "mode": "fixed_targets_updated_coordinate_evaluation",
        }


@njit(cache=True)
def _quad4_elastic_batch_stiffness_flat_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    d4: np.ndarray,
    cdev: np.ndarray,
    cvol: np.ndarray,
    pvol: np.ndarray,
    idev: np.ndarray,
    thickness: np.ndarray,
    mode_codes: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    n = conn.shape[0]
    values = np.empty(n * 64, dtype=np.float64)
    min_det_global = 1.0e300
    min_volume_global = 1.0e300
    for e in range(n):
        coords = np.empty((4, 2), dtype=np.float64)
        for a in range(4):
            node = conn[e, a]
            coords[a, 0] = coords_all[node, 0]
            coords[a, 1] = coords_all[node, 1]
        ke, min_det, volume = _quad4_element_stiffness_numba(
            coords,
            d4[e],
            cdev[e],
            cvol[e],
            pvol[e],
            idev[e],
            thickness[e],
            mode_codes[e],
        )
        if min_det < min_det_global:
            min_det_global = min_det
        if mode_codes[e] == _QUAD4_MODE_BBAR and volume < min_volume_global:
            min_volume_global = volume
        base = e * 64
        for i in range(8):
            for j in range(8):
                values[base + i * 8 + j] = ke[i, j]
    return values, min_det_global, min_volume_global


@njit(cache=True)
def _quad8_elastic_batch_stiffness_flat_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    d4: np.ndarray,
    cdev: np.ndarray,
    cvol: np.ndarray,
    pvol: np.ndarray,
    idev: np.ndarray,
    thickness: np.ndarray,
    mode_codes: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    n = conn.shape[0]
    values = np.empty(n * 256, dtype=np.float64)
    min_det_global = 1.0e300
    min_volume_global = 1.0e300
    for e in range(n):
        coords = np.empty((8, 2), dtype=np.float64)
        for a in range(8):
            node = conn[e, a]
            coords[a, 0] = coords_all[node, 0]
            coords[a, 1] = coords_all[node, 1]
        ke, min_det, volume = _quad8_element_stiffness_numba(
            coords,
            d4[e],
            cdev[e],
            cvol[e],
            pvol[e],
            idev[e],
            thickness[e],
            mode_codes[e],
        )
        if min_det < min_det_global:
            min_det_global = min_det
        if mode_codes[e] == _QUAD4_MODE_BBAR and volume < min_volume_global:
            min_volume_global = volume
        base = e * 256
        for i in range(16):
            for j in range(16):
                values[base + i * 16 + j] = ke[i, j]
    return values, min_det_global, min_volume_global


@njit(cache=True)
def _quad4_axisymmetric_elastic_batch_stiffness_flat_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    d4: np.ndarray,
    thickness: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    n = conn.shape[0]
    values = np.empty(n * 64, dtype=np.float64)
    min_det_global = 1.0e300
    min_radius_global = 1.0e300
    for e in range(n):
        coords = np.empty((4, 2), dtype=np.float64)
        for a in range(4):
            node = conn[e, a]
            coords[a, 0] = coords_all[node, 0]
            coords[a, 1] = coords_all[node, 1]
        ke, min_det, min_radius = _quad4_axisymmetric_element_stiffness_numba(coords, d4[e], thickness[e])
        if min_det < min_det_global:
            min_det_global = min_det
        if min_radius < min_radius_global:
            min_radius_global = min_radius
        base = e * 64
        for i in range(8):
            for j in range(8):
                values[base + i * 8 + j] = ke[i, j]
    return values, min_det_global, min_radius_global


@njit(cache=True)
def _quad8_axisymmetric_elastic_batch_stiffness_flat_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    d4: np.ndarray,
    cdev: np.ndarray,
    cvol: np.ndarray,
    pvol: np.ndarray,
    idev: np.ndarray,
    thickness: np.ndarray,
    mode_codes: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    n = conn.shape[0]
    values = np.empty(n * 256, dtype=np.float64)
    min_det_global = 1.0e300
    min_radius_global = 1.0e300
    min_volume_global = 1.0e300
    for e in range(n):
        coords = np.empty((8, 2), dtype=np.float64)
        for a in range(8):
            node = conn[e, a]
            coords[a, 0] = coords_all[node, 0]
            coords[a, 1] = coords_all[node, 1]
        ke, min_det, min_radius, volume = _quad8_axisymmetric_element_stiffness_numba(
            coords,
            d4[e],
            cdev[e],
            cvol[e],
            pvol[e],
            idev[e],
            thickness[e],
            mode_codes[e],
        )
        if min_det < min_det_global:
            min_det_global = min_det
        if min_radius < min_radius_global:
            min_radius_global = min_radius
        if mode_codes[e] == _QUAD4_MODE_BBAR and volume < min_volume_global:
            min_volume_global = volume
        base = e * 256
        for i in range(16):
            for j in range(16):
                values[base + i * 16 + j] = ke[i, j]
    return values, min_det_global, min_radius_global, min_volume_global


@njit(cache=True)
def _tri_full_gp_numba(node_count: int, gp_index: int) -> tuple[float, float, float]:
    if node_count == 3:
        return 1.0 / 3.0, 1.0 / 3.0, 0.5
    if gp_index == 0:
        return 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0
    if gp_index == 1:
        return 2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0
    return 1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0


@njit(cache=True)
def _tri_b_matrix_numba(coords: np.ndarray, node_count: int, xi: float, eta: float, axisymmetric: bool) -> tuple[np.ndarray, float, np.ndarray, float]:
    dof_count = node_count * 2
    B = np.zeros((4, dof_count), dtype=np.float64)
    N = np.zeros(node_count, dtype=np.float64)
    dxi = np.zeros(node_count, dtype=np.float64)
    deta = np.zeros(node_count, dtype=np.float64)
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
        return B, det, N, 0.0
    inv00 = j11 / det
    inv01 = -j01 / det
    inv10 = -j10 / det
    inv11 = j00 / det
    radius = 0.0
    if axisymmetric:
        for a in range(node_count):
            radius += N[a] * coords[a, 0]
        if radius <= 0.0:
            return B, det, N, radius
    for a in range(node_count):
        dndr = inv00 * dxi[a] + inv01 * deta[a]
        dndz = inv10 * dxi[a] + inv11 * deta[a]
        c = 2 * a
        B[0, c] = dndr
        B[1, c + 1] = dndz
        if axisymmetric:
            B[2, c] = N[a] / radius
        B[3, c] = dndz
        B[3, c + 1] = dndr
    return B, det, N, radius


@njit(cache=True)
def _tri_add_btcb_numba(ke: np.ndarray, B: np.ndarray, C: np.ndarray, scale: float, dof_count: int) -> None:
    for i in range(dof_count):
        for j in range(dof_count):
            value = 0.0
            for a in range(4):
                cb = 0.0
                for b in range(4):
                    cb += C[a, b] * B[b, j]
                value += B[a, i] * cb
            ke[i, j] += value * scale


@njit(cache=True)
def _tri_project_b_numba(projector: np.ndarray, B: np.ndarray, dof_count: int) -> np.ndarray:
    out = np.zeros((4, dof_count), dtype=np.float64)
    for i in range(4):
        for j in range(dof_count):
            value = 0.0
            for k in range(4):
                value += projector[i, k] * B[k, j]
            out[i, j] = value
    return out


@njit(cache=True)
def _tri_symmetrize_numba(ke: np.ndarray, dof_count: int) -> None:
    for i in range(dof_count):
        for j in range(i + 1, dof_count):
            value = 0.5 * (ke[i, j] + ke[j, i])
            ke[i, j] = value
            ke[j, i] = value


@njit(cache=True)
def _tri_element_stiffness_numba(
    coords: np.ndarray,
    d4: np.ndarray,
    cdev: np.ndarray,
    cvol: np.ndarray,
    pvol: np.ndarray,
    idev: np.ndarray,
    thickness: float,
    mode_code: int,
    node_count: int,
    axisymmetric: bool,
) -> tuple[np.ndarray, float, float, float]:
    dof_count = node_count * 2
    gp_count = 1 if node_count == 3 else 3
    ke = np.zeros((dof_count, dof_count), dtype=np.float64)
    min_det = 1.0e300
    min_radius = 1.0e300
    volume = 0.0

    if mode_code == 0:
        for gp in range(gp_count):
            xi, eta, weight = _tri_full_gp_numba(node_count, gp)
            B, det, _N, radius = _tri_b_matrix_numba(coords, node_count, xi, eta, axisymmetric)
            if det < min_det:
                min_det = det
            if axisymmetric and radius < min_radius:
                min_radius = radius
            if det <= 0.0 or (axisymmetric and radius <= 0.0):
                return ke, min_det, min_radius, volume
            dV = det * weight * thickness
            if axisymmetric:
                dV *= 2.0 * math.pi * radius
            _tri_add_btcb_numba(ke, B, d4, dV, dof_count)
        _tri_symmetrize_numba(ke, dof_count)
        return ke, min_det, min_radius, volume

    if mode_code == 1:
        for gp in range(gp_count):
            xi, eta, weight = _tri_full_gp_numba(node_count, gp)
            B, det, _N, radius = _tri_b_matrix_numba(coords, node_count, xi, eta, axisymmetric)
            if det < min_det:
                min_det = det
            if axisymmetric and radius < min_radius:
                min_radius = radius
            if det <= 0.0 or (axisymmetric and radius <= 0.0):
                return ke, min_det, min_radius, volume
            dV = det * weight * thickness
            if axisymmetric:
                dV *= 2.0 * math.pi * radius
            _tri_add_btcb_numba(ke, B, cdev, dV, dof_count)
        B, det, _N, radius = _tri_b_matrix_numba(coords, node_count, 1.0 / 3.0, 1.0 / 3.0, axisymmetric)
        if det < min_det:
            min_det = det
        if axisymmetric and radius < min_radius:
            min_radius = radius
        if det <= 0.0 or (axisymmetric and radius <= 0.0):
            return ke, min_det, min_radius, volume
        dV = det * 0.5 * thickness
        if axisymmetric:
            dV *= 2.0 * math.pi * radius
        Bv = _tri_project_b_numba(pvol, B, dof_count)
        _tri_add_btcb_numba(ke, Bv, cvol, dV, dof_count)
        _tri_symmetrize_numba(ke, dof_count)
        return ke, min_det, min_radius, volume

    B_cache = np.zeros((3, 4, dof_count), dtype=np.float64)
    dV_cache = np.zeros(3, dtype=np.float64)
    Bv_acc = np.zeros((4, dof_count), dtype=np.float64)
    for gp in range(gp_count):
        xi, eta, weight = _tri_full_gp_numba(node_count, gp)
        B, det, _N, radius = _tri_b_matrix_numba(coords, node_count, xi, eta, axisymmetric)
        if det < min_det:
            min_det = det
        if axisymmetric and radius < min_radius:
            min_radius = radius
        if det <= 0.0 or (axisymmetric and radius <= 0.0):
            return ke, min_det, min_radius, volume
        dV = det * weight * thickness
        if axisymmetric:
            dV *= 2.0 * math.pi * radius
        volume += dV
        dV_cache[gp] = dV
        Bv = _tri_project_b_numba(pvol, B, dof_count)
        for r in range(4):
            for c in range(dof_count):
                B_cache[gp, r, c] = B[r, c]
                Bv_acc[r, c] += Bv[r, c] * dV
    if volume <= 0.0:
        return ke, min_det, min_radius, volume
    Bv_bar = Bv_acc / volume
    for gp in range(gp_count):
        B = B_cache[gp]
        Bdev = _tri_project_b_numba(idev, B, dof_count)
        dV = dV_cache[gp]
        _tri_add_btcb_numba(ke, Bdev, cdev, dV, dof_count)
        _tri_add_btcb_numba(ke, Bv_bar, cvol, dV, dof_count)
    _tri_symmetrize_numba(ke, dof_count)
    return ke, min_det, min_radius, volume


@njit(cache=True)
def _tri_elastic_batch_stiffness_flat_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    d4: np.ndarray,
    cdev: np.ndarray,
    cvol: np.ndarray,
    pvol: np.ndarray,
    idev: np.ndarray,
    thickness: np.ndarray,
    mode_codes: np.ndarray,
    node_count: int,
    axisymmetric: bool,
) -> tuple[np.ndarray, float, float, float]:
    n = conn.shape[0]
    dof_count = node_count * 2
    values = np.empty(n * dof_count * dof_count, dtype=np.float64)
    min_det_global = 1.0e300
    min_radius_global = 1.0e300
    min_volume_global = 1.0e300
    for e in range(n):
        coords = np.empty((node_count, 2), dtype=np.float64)
        for a in range(node_count):
            node = conn[e, a]
            coords[a, 0] = coords_all[node, 0]
            coords[a, 1] = coords_all[node, 1]
        ke, min_det, min_radius, volume = _tri_element_stiffness_numba(
            coords,
            d4[e],
            cdev[e],
            cvol[e],
            pvol[e],
            idev[e],
            thickness[e],
            mode_codes[e],
            node_count,
            axisymmetric,
        )
        if min_det < min_det_global:
            min_det_global = min_det
        if axisymmetric and min_radius < min_radius_global:
            min_radius_global = min_radius
        if mode_codes[e] == _QUAD4_MODE_BBAR and volume < min_volume_global:
            min_volume_global = volume
        base = e * dof_count * dof_count
        for i in range(dof_count):
            for j in range(dof_count):
                values[base + i * dof_count + j] = ke[i, j]
    return values, min_det_global, min_radius_global, min_volume_global


@njit(cache=True)
def _quad4_mass_batch_flat_numba(
    coords: np.ndarray,
    density: np.ndarray,
    thickness: np.ndarray,
) -> tuple[np.ndarray, float]:
    count = coords.shape[0]
    values = np.empty((count, 64), dtype=np.float64)
    min_det_global = 1.0e300
    for element_index in range(count):
        me, min_det = _quad4_consistent_mass_matrix_numba(
            coords[element_index],
            density[element_index],
            thickness[element_index],
        )
        if min_det < min_det_global:
            min_det_global = min_det
        for i in range(8):
            for j in range(8):
                values[element_index, i * 8 + j] = me[i, j]
    return values, min_det_global


@njit(cache=True)
def _quad8_mass_batch_flat_numba(
    coords: np.ndarray,
    density: np.ndarray,
    thickness: np.ndarray,
) -> tuple[np.ndarray, float]:
    count = coords.shape[0]
    values = np.empty((count, 256), dtype=np.float64)
    min_det_global = 1.0e300
    for element_index in range(count):
        me, min_det = _quad8_consistent_mass_matrix_numba(
            coords[element_index],
            density[element_index],
            thickness[element_index],
        )
        if min_det < min_det_global:
            min_det_global = min_det
        for i in range(16):
            for j in range(16):
                values[element_index, i * 16 + j] = me[i, j]
    return values, min_det_global


@njit(cache=True)
def _tri_consistent_mass_matrix_numba(
    coords: np.ndarray,
    density: float,
    thickness: float,
    node_count: int,
) -> tuple[np.ndarray, float]:
    dof_count = int(node_count) * 2
    values = np.zeros((dof_count, dof_count), dtype=np.float64)
    min_det = 1.0e300
    gp_count = 1 if int(node_count) == 3 else 3
    for gp_index in range(gp_count):
        xi, eta, weight = _tri_full_gp_numba(int(node_count), gp_index)
        _B, det, N, _radius = _tri_b_matrix_numba(coords, int(node_count), xi, eta, False)
        if det < min_det:
            min_det = det
        if det <= 0.0:
            continue
        dV = det * weight * thickness
        for a in range(int(node_count)):
            for b in range(int(node_count)):
                value = density * N[a] * N[b] * dV
                ia = 2 * a
                ib = 2 * b
                values[ia, ib] += value
                values[ia + 1, ib + 1] += value
    return values, min_det


@njit(cache=True)
def _tri_mass_batch_flat_numba(
    coords: np.ndarray,
    density: np.ndarray,
    thickness: np.ndarray,
    node_count: int,
) -> tuple[np.ndarray, float]:
    count = coords.shape[0]
    dof_count = int(node_count) * 2
    values = np.zeros((count, dof_count * dof_count), dtype=np.float64)
    min_det_global = 1.0e300
    for element_index in range(count):
        matrix, min_det = _tri_consistent_mass_matrix_numba(
            coords[element_index],
            density[element_index],
            thickness[element_index],
            int(node_count),
        )
        if min_det < min_det_global:
            min_det_global = min_det
        for i in range(dof_count):
            for j in range(dof_count):
                values[element_index, i * dof_count + j] = matrix[i, j]
    return values, min_det_global


def _tri_consistent_mass_matrix_fast(
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    density: float,
) -> np.ndarray:
    node_count = int(np.asarray(coords).shape[0])
    if node_count not in (3, 6):
        raise FEM2DError(f"TRI mass kernel expects 3 or 6 nodes, got {node_count}")
    matrix, min_det = _tri_consistent_mass_matrix_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        float(density),
        float(material.thickness),
        node_count,
    )
    element_type = "TRI3" if node_count == 3 else "TRI6"
    if min_det <= 0.0:
        raise FEM2DError(f"{element_type}: detJ must be positive, got {min_det:.6e}")
    return np.asarray(matrix, dtype=float)


def structural_assembly_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.structural_assembly.v1",
        "module": "geofem_app.fem2d_structural_assembly",
        "function_count": len(STRUCTURAL_ASSEMBLY_FUNCTIONS),
        "functions": list(STRUCTURAL_ASSEMBLY_FUNCTIONS),
        "covered_surfaces": [
            "linear_stiffness_assembly",
            "interface_structural_precomputed_block_batch_fill",
            "tri3_tri6_plane_strain_elastic_stiffness_batching",
            "quad4_quad8_plane_strain_elastic_stiffness_batching",
            "mass_matrix_direct_fill",
            "quad4_quad8_mass_matrix_batching",
            "consistent_and_lumped_mass",
            "plane_strain_load_vector",
            "axisymmetric_stiffness_direct_fill",
            "axisymmetric_tri3_tri6_elastic_batch",
            "axisymmetric_quad4_elastic_batch",
            "axisymmetric_quad8_elastic_batch",
            "axisymmetric_stiffness_assembly",
            "axisymmetric_load_vector",
            "body_and_edge_load_helpers",
        ],
    }


def assemble_global_stiffness(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
) -> csr_matrix:
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements)
    rotation_dofs = structural_rotation_dof_map(mesh, structural_elements)
    builder = SparseAssemblyBuilder()
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        ke = element_stiffness(element.type, coords, material, element.integration)
        dofs = _dofs_from_node_indices(conn)
        builder.add_block(dofs, dofs, ke)
    for interface in interfaces or []:
        if not interface.active:
            continue
        dofs, _fe, ke = interface_force_tangent(interface, mesh)
        builder.add_block(dofs, dofs, ke)
    for structural in structural_elements or []:
        if not structural.active:
            continue
        dofs, _fe, ke = structural_element_force_tangent(structural, mesh, materials, rotation_dofs=rotation_dofs)
        builder.add_block(dofs, dofs, ke)
    return builder.to_csr((ndof, ndof))


def build_global_stiffness_assembly_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    *,
    precompute_linear_element_stiffness: bool = False,
) -> GlobalStiffnessAssemblyCache:
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements)
    rotation_dofs = structural_rotation_dof_map(mesh, structural_elements)
    element_blocks: list[ElementStiffnessBlockCache] = []
    interface_blocks: list[InterfaceStiffnessBlockCache] = []
    structural_blocks: list[StructuralStiffnessBlockCache] = []
    dof_blocks: list[np.ndarray] = []
    for index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        if element.material not in materials:
            raise FEM2DError(f"element {element.id}: material '{element.material}' is not defined")
        conn = _element_node_indices(element.nodes, node_index)
        dofs = _dofs_from_node_indices(conn)
        material = materials[element.material]
        linear_stiffness = None
        if precompute_linear_element_stiffness and not material.is_plastic:
            linear_stiffness = np.asarray(element_stiffness(element.type, mesh.coords[conn], material, element.integration), dtype=float).copy()
        element_blocks.append(ElementStiffnessBlockCache(index, conn, dofs, linear_stiffness))
        dof_blocks.append(dofs)
    for index, interface in enumerate(interfaces or []):
        if not interface.active:
            continue
        dofs = _dofs_from_node_indices(_element_node_indices((*interface.minus_nodes, *interface.plus_nodes), node_index))
        linear_stiffness: np.ndarray | None = None
        if _interface_is_linear(interface):
            _dofs, _fe, ke = interface_force_tangent(interface, mesh)
            linear_stiffness = np.asarray(ke, dtype=float).copy()
        interface_blocks.append(InterfaceStiffnessBlockCache(index, dofs, linear_stiffness))
        dof_blocks.append(dofs)
    for index, structural in enumerate(structural_elements or []):
        if not structural.active:
            continue
        dofs = structural_element_dofs(structural, mesh, rotation_dofs)
        linear_stiffness = None
        if not structural_has_nonlinear([structural]):
            _dofs, _fe, ke = structural_element_force_tangent(structural, mesh, materials, rotation_dofs=rotation_dofs)
            linear_stiffness = np.asarray(ke, dtype=float).copy()
        structural_blocks.append(StructuralStiffnessBlockCache(index, dofs, linear_stiffness))
        dof_blocks.append(dofs)
    pattern = SparseAssemblyPattern.from_square_blocks(dof_blocks, (ndof, ndof))
    element_batches = (
        _build_precomputed_stiffness_block_batches(element_blocks, 0, "element")
        if element_blocks and all(block.linear_stiffness is not None for block in element_blocks)
        else ()
    )
    interface_batches = _build_precomputed_stiffness_block_batches(interface_blocks, len(element_blocks), "interface")
    structural_batches = _build_precomputed_stiffness_block_batches(structural_blocks, len(element_blocks) + len(interface_blocks), "structural")
    return GlobalStiffnessAssemblyCache(
        shape=(ndof, ndof),
        pattern=pattern,
        element_blocks=tuple(element_blocks),
        interface_blocks=tuple(interface_blocks),
        structural_blocks=tuple(structural_blocks),
        rotation_dofs=dict(rotation_dofs),
        quad4_elastic_batch=_build_quad4_elastic_batch_cache(mesh, materials, element_blocks),
        quad8_elastic_batch=_build_quad8_elastic_batch_cache(mesh, materials, element_blocks),
        tri3_elastic_batch=_build_tri_elastic_batch_cache(mesh, materials, element_blocks, "TRI3"),
        tri6_elastic_batch=_build_tri_elastic_batch_cache(mesh, materials, element_blocks, "TRI6"),
        precomputed_element_batches=element_batches,
        precomputed_interface_batches=interface_batches,
        precomputed_structural_batches=structural_batches,
    )


def assemble_global_stiffness_cached(
    cache: GlobalStiffnessAssemblyCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
) -> csr_matrix:
    flat_values = cache.pattern.empty_flat_values()
    block_index = 0
    if cache.precomputed_element_batches:
        block_index = _fill_precomputed_stiffness_batches(cache.pattern, flat_values, block_index, cache.precomputed_element_batches)
    elif cache.quad4_elastic_batch is not None:
        element_values = _assemble_quad4_elastic_batch_flat(cache.quad4_elastic_batch, mesh)
        block_index = cache.pattern.fill_blocks_flat(flat_values, block_index, element_values, len(cache.element_blocks))
    elif cache.quad8_elastic_batch is not None:
        element_values = _assemble_quad8_elastic_batch_flat(cache.quad8_elastic_batch, mesh)
        block_index = cache.pattern.fill_blocks_flat(flat_values, block_index, element_values, len(cache.element_blocks))
    elif cache.tri3_elastic_batch is not None:
        element_values = _assemble_tri_elastic_batch_flat(cache.tri3_elastic_batch, mesh, 3, axisymmetric=False)
        block_index = cache.pattern.fill_blocks_flat(flat_values, block_index, element_values, len(cache.element_blocks))
    elif cache.tri6_elastic_batch is not None:
        element_values = _assemble_tri_elastic_batch_flat(cache.tri6_elastic_batch, mesh, 6, axisymmetric=False)
        block_index = cache.pattern.fill_blocks_flat(flat_values, block_index, element_values, len(cache.element_blocks))
    else:
        for block in cache.element_blocks:
            element = mesh.elements[block.element_index]
            material = materials[element.material]
            coords = mesh.coords[block.conn]
            cache.pattern.fill_block(flat_values, block_index, element_stiffness(element.type, coords, material, element.integration))
            block_index += 1
    block_index = _fill_interface_stiffness_blocks(
        cache.pattern,
        flat_values,
        block_index,
        cache.interface_blocks,
        cache.precomputed_interface_batches,
        interfaces or [],
        mesh,
        axisymmetric=False,
    )
    block_index = _fill_structural_stiffness_blocks(
        cache.pattern,
        flat_values,
        block_index,
        cache.structural_blocks,
        cache.precomputed_structural_batches,
        structural_elements or [],
        mesh,
        materials,
        axisymmetric=False,
        rotation_dofs=cache.rotation_dofs,
    )
    cache.pattern.validate_filled_block_count(block_index)
    return cache.pattern.assemble_flat_values(flat_values)


def _build_quad4_elastic_batch_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    element_blocks: list[ElementStiffnessBlockCache],
) -> Quad4ElasticBatchAssemblyCache | None:
    if not element_blocks:
        return None
    conn_rows: list[np.ndarray] = []
    d4_rows: list[np.ndarray] = []
    cdev_rows: list[np.ndarray] = []
    cvol_rows: list[np.ndarray] = []
    pvol_rows: list[np.ndarray] = []
    idev_rows: list[np.ndarray] = []
    thickness_values: list[float] = []
    mode_values: list[int] = []
    for block in element_blocks:
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        if element.type.upper() != "QUAD4" or material.is_plastic:
            return None
        conn_rows.append(np.asarray(block.conn, dtype=np.int64))
        pvol = np.asarray(material.volumetric_projector, dtype=np.float64)
        d4_rows.append(np.asarray(material.D4, dtype=np.float64))
        cdev_rows.append(np.asarray(material.C_dev, dtype=np.float64))
        cvol_rows.append(np.asarray(material.C_vol, dtype=np.float64))
        pvol_rows.append(pvol)
        idev_rows.append(np.eye(4, dtype=np.float64) - pvol)
        thickness_values.append(float(material.thickness))
        mode_values.append(int(_quad4_mode_code(normalize_integration(element.integration))))
    return Quad4ElasticBatchAssemblyCache(
        conn=np.ascontiguousarray(np.vstack(conn_rows), dtype=np.int64),
        d4=np.ascontiguousarray(np.stack(d4_rows), dtype=np.float64),
        cdev=np.ascontiguousarray(np.stack(cdev_rows), dtype=np.float64),
        cvol=np.ascontiguousarray(np.stack(cvol_rows), dtype=np.float64),
        pvol=np.ascontiguousarray(np.stack(pvol_rows), dtype=np.float64),
        idev=np.ascontiguousarray(np.stack(idev_rows), dtype=np.float64),
        thickness=np.ascontiguousarray(np.asarray(thickness_values, dtype=np.float64)),
        mode_codes=np.ascontiguousarray(np.asarray(mode_values, dtype=np.int64)),
    )


def _assemble_quad4_elastic_batch_flat(cache: Quad4ElasticBatchAssemblyCache, mesh: Mesh2D) -> np.ndarray:
    values, min_det, min_volume = _quad4_elastic_batch_stiffness_flat_numba(
        np.ascontiguousarray(mesh.coords, dtype=np.float64),
        cache.conn,
        cache.d4,
        cache.cdev,
        cache.cvol,
        cache.pvol,
        cache.idev,
        cache.thickness,
        cache.mode_codes,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4 batch: detJ must be positive, got {min_det:.6e}")
    if np.any(cache.mode_codes == _QUAD4_MODE_BBAR) and min_volume <= 0.0:
        raise FEM2DError("QUAD4 batch: non-positive element measure")
    return values


def _build_quad8_elastic_batch_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    element_blocks: list[ElementStiffnessBlockCache],
) -> Quad8ElasticBatchAssemblyCache | None:
    if not element_blocks:
        return None
    conn_rows: list[np.ndarray] = []
    d4_rows: list[np.ndarray] = []
    cdev_rows: list[np.ndarray] = []
    cvol_rows: list[np.ndarray] = []
    pvol_rows: list[np.ndarray] = []
    idev_rows: list[np.ndarray] = []
    thickness_values: list[float] = []
    mode_values: list[int] = []
    for block in element_blocks:
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        if element.type.upper() != "QUAD8" or material.is_plastic:
            return None
        conn_rows.append(np.asarray(block.conn, dtype=np.int64))
        pvol = np.asarray(material.volumetric_projector, dtype=np.float64)
        d4_rows.append(np.asarray(material.D4, dtype=np.float64))
        cdev_rows.append(np.asarray(material.C_dev, dtype=np.float64))
        cvol_rows.append(np.asarray(material.C_vol, dtype=np.float64))
        pvol_rows.append(pvol)
        idev_rows.append(np.eye(4, dtype=np.float64) - pvol)
        thickness_values.append(float(material.thickness))
        mode_values.append(int(_quad4_mode_code(normalize_integration(element.integration))))
    return Quad8ElasticBatchAssemblyCache(
        conn=np.ascontiguousarray(np.vstack(conn_rows), dtype=np.int64),
        d4=np.ascontiguousarray(np.stack(d4_rows), dtype=np.float64),
        cdev=np.ascontiguousarray(np.stack(cdev_rows), dtype=np.float64),
        cvol=np.ascontiguousarray(np.stack(cvol_rows), dtype=np.float64),
        pvol=np.ascontiguousarray(np.stack(pvol_rows), dtype=np.float64),
        idev=np.ascontiguousarray(np.stack(idev_rows), dtype=np.float64),
        thickness=np.ascontiguousarray(np.asarray(thickness_values, dtype=np.float64)),
        mode_codes=np.ascontiguousarray(np.asarray(mode_values, dtype=np.int64)),
    )


def _assemble_quad8_elastic_batch_flat(cache: Quad8ElasticBatchAssemblyCache, mesh: Mesh2D) -> np.ndarray:
    values, min_det, min_volume = _quad8_elastic_batch_stiffness_flat_numba(
        np.ascontiguousarray(mesh.coords, dtype=np.float64),
        cache.conn,
        cache.d4,
        cache.cdev,
        cache.cvol,
        cache.pvol,
        cache.idev,
        cache.thickness,
        cache.mode_codes,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8 batch: detJ must be positive, got {min_det:.6e}")
    if np.any(cache.mode_codes == _QUAD4_MODE_BBAR) and min_volume <= 0.0:
        raise FEM2DError("QUAD8 batch: non-positive element measure")
    return values


def _build_tri_elastic_batch_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    element_blocks: list[ElementStiffnessBlockCache],
    element_type: str,
) -> TriElasticBatchAssemblyCache | None:
    if not element_blocks:
        return None
    target_type = element_type.upper()
    conn_rows: list[np.ndarray] = []
    d4_rows: list[np.ndarray] = []
    cdev_rows: list[np.ndarray] = []
    cvol_rows: list[np.ndarray] = []
    pvol_rows: list[np.ndarray] = []
    idev_rows: list[np.ndarray] = []
    thickness_values: list[float] = []
    mode_values: list[int] = []
    for block in element_blocks:
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        if element.type.upper() != target_type or material.is_plastic:
            return None
        conn_rows.append(np.asarray(block.conn, dtype=np.int64))
        pvol = np.asarray(material.volumetric_projector, dtype=np.float64)
        d4_rows.append(np.asarray(material.D4, dtype=np.float64))
        cdev_rows.append(np.asarray(material.C_dev, dtype=np.float64))
        cvol_rows.append(np.asarray(material.C_vol, dtype=np.float64))
        pvol_rows.append(pvol)
        idev_rows.append(np.eye(4, dtype=np.float64) - pvol)
        thickness_values.append(float(material.thickness))
        mode_values.append(int(_quad4_mode_code(normalize_integration(element.integration))))
    return TriElasticBatchAssemblyCache(
        conn=np.ascontiguousarray(np.vstack(conn_rows), dtype=np.int64),
        d4=np.ascontiguousarray(np.stack(d4_rows), dtype=np.float64),
        cdev=np.ascontiguousarray(np.stack(cdev_rows), dtype=np.float64),
        cvol=np.ascontiguousarray(np.stack(cvol_rows), dtype=np.float64),
        pvol=np.ascontiguousarray(np.stack(pvol_rows), dtype=np.float64),
        idev=np.ascontiguousarray(np.stack(idev_rows), dtype=np.float64),
        thickness=np.ascontiguousarray(np.asarray(thickness_values, dtype=np.float64)),
        mode_codes=np.ascontiguousarray(np.asarray(mode_values, dtype=np.int64)),
    )


def _assemble_tri_elastic_batch_flat(cache: TriElasticBatchAssemblyCache, mesh: Mesh2D, node_count: int, *, axisymmetric: bool) -> np.ndarray:
    values, min_det, min_radius, min_volume = _tri_elastic_batch_stiffness_flat_numba(
        np.ascontiguousarray(mesh.coords, dtype=np.float64),
        cache.conn,
        cache.d4,
        cache.cdev,
        cache.cvol,
        cache.pvol,
        cache.idev,
        cache.thickness,
        cache.mode_codes,
        int(node_count),
        bool(axisymmetric),
    )
    element_name = "TRI3" if int(node_count) == 3 else "TRI6"
    if min_det <= 0.0:
        raise FEM2DError(f"{element_name} batch: detJ must be positive, got {min_det:.6e}")
    if axisymmetric and min_radius <= 0.0:
        raise FEM2DError(f"{element_name} axisymmetric batch: axisymmetric radius must be positive, got {min_radius:.6e}")
    if np.any(cache.mode_codes == _QUAD4_MODE_BBAR) and min_volume <= 0.0:
        raise FEM2DError(f"{element_name} batch: non-positive element measure")
    return values


def _build_precomputed_stiffness_block_batches(
    blocks: list[ElementStiffnessBlockCache] | list[InterfaceStiffnessBlockCache] | list[StructuralStiffnessBlockCache],
    start_block: int,
    kind: str,
) -> tuple[PrecomputedStiffnessBlockBatch, ...]:
    batches: list[PrecomputedStiffnessBlockBatch] = []
    local_index = 0
    while local_index < len(blocks):
        block = blocks[local_index]
        if block.linear_stiffness is None:
            local_index += 1
            continue
        values: list[np.ndarray] = []
        run_start = local_index
        while local_index < len(blocks) and blocks[local_index].linear_stiffness is not None:
            values.append(np.asarray(blocks[local_index].linear_stiffness, dtype=np.float64).ravel())
            local_index += 1
        if values:
            batches.append(
                PrecomputedStiffnessBlockBatch(
                    kind=kind,
                    start_block=int(start_block + run_start),
                    block_count=int(len(values)),
                    values=np.ascontiguousarray(np.concatenate(values), dtype=np.float64),
                )
            )
    return tuple(batches)


def _fill_precomputed_stiffness_batches(
    pattern: SparseAssemblyPattern,
    flat_values: np.ndarray,
    block_index: int,
    batches: tuple[PrecomputedStiffnessBlockBatch, ...],
) -> int:
    for batch in batches:
        if batch.start_block != block_index:
            raise FEM2DError(f"{batch.kind}: precomputed stiffness batch is not contiguous with block fill order")
        block_index = pattern.fill_blocks_flat(flat_values, block_index, batch.values, batch.block_count)
    return block_index


def _fill_interface_stiffness_blocks(
    pattern: SparseAssemblyPattern,
    flat_values: np.ndarray,
    block_index: int,
    blocks: tuple[InterfaceStiffnessBlockCache, ...],
    batches: tuple[PrecomputedStiffnessBlockBatch, ...],
    interfaces: list[Interface2D],
    mesh: Mesh2D,
    *,
    axisymmetric: bool,
) -> int:
    local_index = 0
    batch_index = 0
    while local_index < len(blocks):
        if batch_index < len(batches) and batches[batch_index].start_block == block_index:
            batch = batches[batch_index]
            block_index = pattern.fill_blocks_flat(flat_values, block_index, batch.values, batch.block_count)
            local_index += batch.block_count
            batch_index += 1
            continue
        block = blocks[local_index]
        if block.linear_stiffness is not None:
            pattern.fill_block(flat_values, block_index, block.linear_stiffness)
        else:
            interface = interfaces[block.interface_index]
            _dofs, _fe, ke = interface_force_tangent(interface, mesh, axisymmetric=axisymmetric)
            pattern.fill_block(flat_values, block_index, ke)
        block_index += 1
        local_index += 1
    return block_index


def _fill_structural_stiffness_blocks(
    pattern: SparseAssemblyPattern,
    flat_values: np.ndarray,
    block_index: int,
    blocks: tuple[StructuralStiffnessBlockCache, ...],
    batches: tuple[PrecomputedStiffnessBlockBatch, ...],
    structural_elements: list[StructuralElement2D],
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    *,
    axisymmetric: bool,
    rotation_dofs: Mapping[str, int] | None = None,
) -> int:
    local_index = 0
    batch_index = 0
    while local_index < len(blocks):
        if batch_index < len(batches) and batches[batch_index].start_block == block_index:
            batch = batches[batch_index]
            block_index = pattern.fill_blocks_flat(flat_values, block_index, batch.values, batch.block_count)
            local_index += batch.block_count
            batch_index += 1
            continue
        block = blocks[local_index]
        if block.linear_stiffness is not None:
            pattern.fill_block(flat_values, block_index, block.linear_stiffness)
        else:
            structural = structural_elements[block.structural_index]
            _dofs, _fe, ke = structural_element_force_tangent(
                structural,
                mesh,
                materials,
                axisymmetric=axisymmetric,
                rotation_dofs=rotation_dofs,
            )
            pattern.fill_block(flat_values, block_index, ke)
        block_index += 1
        local_index += 1
    return block_index


def _interface_is_linear(interface: Interface2D) -> bool:
    return bool(interface.active and interface.friction <= 0.0 and interface.cohesion <= 0.0 and not interface.no_tension)


def build_load_vector_assembly_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    loads: Any,
    structural_elements: list[StructuralElement2D] | None = None,
) -> LoadVectorAssemblyCache | None:
    if structural_elements:
        return None
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements)
    node_load_vector = np.zeros(ndof, dtype=float)
    body_blocks: list[BodyLoadBlockCache] = []
    edge_blocks: list[EdgeLoadBlockCache] = []
    load_items = _ensure_list(loads)
    for load in load_items:
        if not isinstance(load, Mapping):
            return None
        ltype = str(load.get("type", "")).lower().strip()
        body_load = ltype in {"gravity", "self_weight", "body"} or bool(load.get("self_weight", False))
        if body_load:
            gx = float(load.get("gx", 0.0))
            gy = float(load.get("gy", -1.0))
            scale = float(load.get("scale", 1.0))
            target = dict(load)
            for element_index, element in enumerate(mesh.elements):
                if not element.active:
                    continue
                if element.material not in materials:
                    raise FEM2DError(f"element {element.id}: material '{element.material}' is not defined")
                if not _body_load_targets_element(mesh, element, target):
                    continue
                conn = np.asarray(_element_node_indices(element.nodes, node_index), dtype=np.int64)
                body_blocks.append(BodyLoadBlockCache(element_index, conn, _dofs_from_node_indices(conn), gx, gy, scale, target))
            continue
        if "node" in load or "nodes" in load or "set" in load:
            for nid in _target_nodes(mesh, load):
                i = node_index[nid]
                node_load_vector[2 * i] += float(load.get("fx", load.get("px", load.get("ux", 0.0))))
                node_load_vector[2 * i + 1] += float(load.get("fy", load.get("py", load.get("uy", 0.0))))
        if "edge" in load or "edges" in load:
            edges = load.get("edges", [load.get("edge")])
            if isinstance(edges, str):
                edges = _edge_set(mesh, edges)
            tx1, ty1, tx2, ty2 = _edge_traction_components(load)
            for edge in edges:
                edge_nodes = tuple(str(n) for n in _require_sequence(edge, "load.edge"))
                indices = np.asarray([node_index[nid] for nid in edge_nodes], dtype=np.int64)
                edge_blocks.append(EdgeLoadBlockCache(indices, _dofs_from_node_indices(indices), tx1, ty1, tx2, ty2))
    if not load_items:
        return LoadVectorAssemblyCache(ndof, node_load_vector, (), (), 0)
    return LoadVectorAssemblyCache(
        ndof=ndof,
        node_load_vector=node_load_vector,
        body_blocks=tuple(body_blocks),
        edge_blocks=tuple(edge_blocks),
        load_count=len(load_items),
    )


def assemble_load_vector_cached(
    cache: LoadVectorAssemblyCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> np.ndarray:
    F = np.asarray(cache.node_load_vector, dtype=float).copy()
    if F.shape != (cache.ndof,):
        raise FEM2DError("cached load vector size mismatch")
    for block in cache.body_blocks:
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        coords = mesh.coords[block.conn]
        fe = np.zeros(block.dofs.size, dtype=float)
        bx, by = _body_force_density(material, block.gx, block.gy, block.scale, block.target)
        for gp in integration_points(element.type, "FULL"):
            _B4, detJ, N = strain_displacement_matrix(element.type, coords, gp)
            dV = detJ * gp[2] * material.thickness
            for a, Na in enumerate(N):
                fe[2 * a] += Na * bx * dV
                fe[2 * a + 1] += Na * by * dV
        F[block.dofs] += fe
    for block in cache.edge_blocks:
        _add_edge_traction_by_indices(F, mesh.coords, block.node_indices, block.tx1, block.ty1, block.tx2, block.ty2)
    return F


def build_mass_matrix_assembly_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    structural_elements: list[StructuralElement2D] | None = None,
) -> MassMatrixAssemblyCache:
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements)
    element_blocks: list[ElementMassBlockCache] = []
    structural_blocks: list[StructuralMassBlockCache] = []
    dof_blocks: list[np.ndarray] = []
    for element_index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        if element.material not in materials:
            raise FEM2DError(f"element {element.id}: material '{element.material}' is not defined")
        if _material_mass_density(materials[element.material]) <= 0.0:
            continue
        conn = _element_node_indices(element.nodes, node_index)
        dofs = _dofs_from_node_indices(conn)
        element_blocks.append(ElementMassBlockCache(element_index, conn, dofs))
        dof_blocks.append(dofs)
    for structural_index, structural in enumerate(structural_elements or []):
        if not structural.active:
            continue
        if _structural_mass_per_length(structural, materials) <= 0.0:
            continue
        i = node_index[structural.nodes[0]]
        j = node_index[structural.nodes[1]]
        dofs = np.asarray([2 * i, 2 * i + 1, 2 * j, 2 * j + 1], dtype=int)
        structural_blocks.append(StructuralMassBlockCache(structural_index, dofs, _structural_mass_matrix_block(structural, mesh, materials)))
        dof_blocks.append(dofs)
    pattern = SparseAssemblyPattern.from_square_blocks(dof_blocks, (ndof, ndof))
    element_tuple = tuple(element_blocks)
    return MassMatrixAssemblyCache(
        shape=(ndof, ndof),
        pattern=pattern,
        element_blocks=element_tuple,
        structural_blocks=tuple(structural_blocks),
        batch_blocks=_build_mass_element_batches(mesh, element_tuple),
    )


def assemble_mass_matrix_cached(
    cache: MassMatrixAssemblyCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    structural_elements: list[StructuralElement2D] | None = None,
    *,
    lumped: bool = False,
) -> csr_matrix:
    values = cache.pattern.empty_flat_values()
    batched = np.zeros(len(cache.element_blocks), dtype=bool)
    for batch in cache.batch_blocks:
        blocks, min_det = _assemble_mass_batch_flat(batch, mesh, materials)
        _fill_mass_batch_flat_values(cache.pattern, values, batch.block_indices, blocks)
        if min_det <= 0.0:
            raise FEM2DError(f"{batch.element_type}: detJ must be positive, got {min_det:.6e}")
        batched[batch.block_indices] = True
    block_index = 0
    for local_index, block in enumerate(cache.element_blocks):
        if not batched[local_index]:
            element = mesh.elements[block.element_index]
            material = materials[element.material]
            density = _material_mass_density(material)
            cache.pattern.fill_block(values, block_index, _element_mass_matrix_block(element.type, mesh.coords[block.conn], material, density))
        block_index += 1
    structural_list = structural_elements or []
    for block in cache.structural_blocks:
        structural = structural_list[block.structural_index]
        cache.pattern.fill_block(values, block_index, block.mass_matrix if block.mass_matrix is not None else _structural_mass_matrix_block(structural, mesh, materials))
        block_index += 1
    cache.pattern.validate_filled_block_count(block_index)
    mass = cache.pattern.assemble_flat_values(values)
    if lumped:
        diag = np.asarray(mass.sum(axis=1)).ravel()
        return diags(diag, format="csr")
    return mass


def _build_mass_element_batches(mesh: Mesh2D, blocks: tuple[ElementMassBlockCache, ...]) -> tuple[MassElementBatchCache, ...]:
    grouped: dict[str, dict[str, list[Any]]] = {
        "TRI3": {"block_indices": [], "element_indices": [], "conn": []},
        "TRI6": {"block_indices": [], "element_indices": [], "conn": []},
        "QUAD4": {"block_indices": [], "element_indices": [], "conn": []},
        "QUAD8": {"block_indices": [], "element_indices": [], "conn": []},
    }
    for block_index, block in enumerate(blocks):
        element_type = str(mesh.elements[block.element_index].type).upper()
        if element_type not in grouped:
            continue
        grouped[element_type]["block_indices"].append(block_index)
        grouped[element_type]["element_indices"].append(block.element_index)
        grouped[element_type]["conn"].append(np.asarray(block.conn, dtype=np.int64))
    batches: list[MassElementBatchCache] = []
    for element_type in ("TRI3", "TRI6", "QUAD4", "QUAD8"):
        raw = grouped[element_type]
        if len(raw["block_indices"]) < _MASS_BATCH_MIN_ELEMENTS:
            continue
        batches.append(
            MassElementBatchCache(
                element_type=element_type,
                block_indices=np.asarray(raw["block_indices"], dtype=np.int64),
                element_indices=np.asarray(raw["element_indices"], dtype=np.int64),
                conn=np.vstack(raw["conn"]).astype(np.int64, copy=False),
            )
        )
    return tuple(batches)


def _mass_batch_group_info(batch_blocks: tuple[MassElementBatchCache, ...]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for batch in batch_blocks:
        indices = np.asarray(batch.block_indices, dtype=np.int64)
        groups.append(
            {
                "element_type": batch.element_type,
                "elements": int(indices.size),
                "contiguous_blocks": bool(indices.size == 0 or np.all(np.diff(indices) == 1)),
            }
        )
    return groups


def _mass_batch_inputs(
    batch: MassElementBatchCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.ascontiguousarray(mesh.coords[batch.conn], dtype=np.float64)
    density = np.empty(batch.element_indices.size, dtype=np.float64)
    thickness = np.empty(batch.element_indices.size, dtype=np.float64)
    for local_index, element_index in enumerate(batch.element_indices):
        element = mesh.elements[int(element_index)]
        material = materials[element.material]
        density[local_index] = float(_material_mass_density(material))
        thickness[local_index] = float(material.thickness)
    return coords, np.ascontiguousarray(density), np.ascontiguousarray(thickness)


def _assemble_mass_batch_flat(
    batch: MassElementBatchCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> tuple[np.ndarray, float]:
    coords, density, thickness = _mass_batch_inputs(batch, mesh, materials)
    if batch.element_type == "TRI3":
        return _tri_mass_batch_flat_numba(coords, density, thickness, 3)
    if batch.element_type == "TRI6":
        return _tri_mass_batch_flat_numba(coords, density, thickness, 6)
    if batch.element_type == "QUAD4":
        return _quad4_mass_batch_flat_numba(coords, density, thickness)
    if batch.element_type == "QUAD8":
        return _quad8_mass_batch_flat_numba(coords, density, thickness)
    return np.zeros((0, 0), dtype=float), 1.0


def _fill_mass_batch_flat_values(pattern: SparseAssemblyPattern, flat_values: np.ndarray, block_indices: np.ndarray, blocks_flat: np.ndarray) -> None:
    indices = np.asarray(block_indices, dtype=np.int64)
    if indices.size == 0:
        return
    values = np.asarray(blocks_flat, dtype=np.float64)
    if bool(np.all(np.diff(indices) == 1)):
        pattern.fill_blocks_flat(flat_values, int(indices[0]), values, int(indices.size))
        return
    for local_index, block_index in enumerate(indices):
        pattern.fill_block(flat_values, int(block_index), values[local_index])


def _element_mass_matrix_block(
    element_type: str,
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    density: float,
) -> np.ndarray:
    nnode = int(coords.shape[0])
    etype = str(element_type).upper()
    if etype == "QUAD4":
        return _quad4_consistent_mass_matrix_fast(coords, material, density)
    if etype == "QUAD8":
        return _quad8_consistent_mass_matrix_fast(coords, material, density)
    if etype in {"TRI3", "TRI6"}:
        return _tri_consistent_mass_matrix_fast(coords, material, density)
    me = np.zeros((2 * nnode, 2 * nnode), dtype=float)
    for xi, eta, weight in integration_points(element_type, "FULL"):
        N, dN_dnatural = shape_functions(element_type, xi, eta)
        jac = dN_dnatural @ coords
        detJ = float(np.linalg.det(jac))
        if detJ <= 0.0:
            raise FEM2DError(f"{element_type}: detJ must be positive, got {detJ:.6e}")
        dV = detJ * weight * material.thickness
        for a, Na in enumerate(N):
            for b, Nb in enumerate(N):
                value = density * Na * Nb * dV
                me[2 * a, 2 * b] += value
                me[2 * a + 1, 2 * b + 1] += value
    return me


def _structural_mass_matrix_block(
    structural: StructuralElement2D,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> np.ndarray:
    node_index = mesh.node_index
    mass_per_length = _structural_mass_per_length(structural, materials)
    i = node_index[structural.nodes[0]]
    j = node_index[structural.nodes[1]]
    length = float(np.linalg.norm(mesh.coords[j] - mesh.coords[i]))
    if length <= 0.0:
        raise FEM2DError(f"structural element {structural.id}: length must be positive")
    return mass_per_length * length / 6.0 * np.array(
        [
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0, 1.0],
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 2.0],
        ],
        dtype=float,
    )


def assemble_mass_matrix(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    structural_elements: list[StructuralElement2D] | None = None,
    *,
    lumped: bool = False,
) -> csr_matrix:
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements)
    builder = SparseAssemblyBuilder()
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        density = _material_mass_density(material)
        if density <= 0.0:
            continue
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        nnode = len(conn)
        etype = element.type.upper()
        if etype == "QUAD4":
            me = _quad4_consistent_mass_matrix_fast(coords, material, density)
        elif etype == "QUAD8":
            me = _quad8_consistent_mass_matrix_fast(coords, material, density)
        else:
            me = np.zeros((2 * nnode, 2 * nnode), dtype=float)
            for xi, eta, weight in integration_points(element.type, "FULL"):
                N, dN_dnatural = shape_functions(element.type, xi, eta)
                jac = dN_dnatural @ coords
                detJ = float(np.linalg.det(jac))
                if detJ <= 0.0:
                    raise FEM2DError(f"{element.type}: detJ must be positive, got {detJ:.6e}")
                dV = detJ * weight * material.thickness
                for a, Na in enumerate(N):
                    for b, Nb in enumerate(N):
                        value = density * Na * Nb * dV
                        me[2 * a, 2 * b] += value
                        me[2 * a + 1, 2 * b + 1] += value
        dofs = _dofs_from_node_indices(conn)
        builder.add_block(dofs, dofs, me)
    for structural in structural_elements or []:
        if not structural.active:
            continue
        mass_per_length = _structural_mass_per_length(structural, materials)
        if mass_per_length <= 0.0:
            continue
        i = node_index[structural.nodes[0]]
        j = node_index[structural.nodes[1]]
        length = float(np.linalg.norm(mesh.coords[j] - mesh.coords[i]))
        if length <= 0.0:
            raise FEM2DError(f"structural element {structural.id}: length must be positive")
        me = mass_per_length * length / 6.0 * np.array(
            [
                [2.0, 0.0, 1.0, 0.0],
                [0.0, 2.0, 0.0, 1.0],
                [1.0, 0.0, 2.0, 0.0],
                [0.0, 1.0, 0.0, 2.0],
            ],
            dtype=float,
        )
        dofs = np.asarray([2 * i, 2 * i + 1, 2 * j, 2 * j + 1], dtype=int)
        builder.add_block(dofs, dofs, me)
    mass = builder.to_csr((ndof, ndof))
    if lumped:
        diag = np.asarray(mass.sum(axis=1)).ravel()
        return diags(diag, format="csr")
    return mass

def _material_mass_density(material: ElasticPlaneStrainMaterial) -> float:
    params = material.advanced_params or {}
    for key in ("density", "rho", "mass_density"):
        value = params.get(key)
        if value not in (None, ""):
            return float(value)
    gravity = float(params.get("gravity", params.get("g", 9.80665)) or 9.80665)
    if gravity <= 0.0:
        gravity = 9.80665
    return max(float(material.gamma), 0.0) / gravity

def _structural_mass_per_length(element: StructuralElement2D, materials: Mapping[str, ElasticPlaneStrainMaterial]) -> float:
    section = element.section if isinstance(element.section, Mapping) else {}
    for key in ("mass_per_length", "line_mass", "rhoA", "rho_a", "mass"):
        value = section.get(key)
        if value not in (None, ""):
            return float(value)
    area = section.get("A", section.get("area"))
    if area in (None, ""):
        return 0.0
    density = section.get("density", section.get("rho"))
    if density in (None, "") and element.material in materials:
        density = _material_mass_density(materials[element.material])
    if density in (None, ""):
        return 0.0
    return max(float(density), 0.0) * max(float(area), 0.0)

def assemble_axisymmetric_stiffness(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
) -> csr_matrix:
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements, axisymmetric=True)
    builder = SparseAssemblyBuilder()
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        if element.type.upper() == "QUAD4":
            ke = _quad4_axisymmetric_element_stiffness_fast(coords, material)
        else:
            ke = axisymmetric_element_stiffness(element.type, coords, material, element.integration)
        dofs = _dofs_from_node_indices(conn)
        builder.add_block(dofs, dofs, ke)
    for interface in interfaces or []:
        if not interface.active:
            continue
        dofs, _fe, ke = interface_force_tangent(interface, mesh, axisymmetric=True)
        builder.add_block(dofs, dofs, ke)
    for structural in structural_elements or []:
        if not structural.active:
            continue
        dofs, _fe, ke = structural_element_force_tangent(structural, mesh, materials, axisymmetric=True)
        builder.add_block(dofs, dofs, ke)
    return builder.to_csr((ndof, ndof))


def build_axisymmetric_stiffness_assembly_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
    *,
    precompute_linear_element_stiffness: bool = False,
) -> AxisymmetricStiffnessAssemblyCache:
    node_index = mesh.node_index
    ndof = structural_total_dofs(mesh, structural_elements, axisymmetric=True)
    element_blocks: list[ElementStiffnessBlockCache] = []
    interface_blocks: list[InterfaceStiffnessBlockCache] = []
    structural_blocks: list[StructuralStiffnessBlockCache] = []
    dof_blocks: list[np.ndarray] = []
    for index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        if element.material not in materials:
            raise FEM2DError(f"element {element.id}: material '{element.material}' is not defined")
        conn = _element_node_indices(element.nodes, node_index)
        dofs = _dofs_from_node_indices(conn)
        material = materials[element.material]
        linear_stiffness = None
        if precompute_linear_element_stiffness and not material.is_plastic:
            if element.type.upper() == "QUAD4":
                ke = _quad4_axisymmetric_element_stiffness_fast(mesh.coords[conn], material)
            else:
                ke = axisymmetric_element_stiffness(element.type, mesh.coords[conn], material, element.integration)
            linear_stiffness = np.asarray(ke, dtype=float).copy()
        element_blocks.append(ElementStiffnessBlockCache(index, conn, dofs, linear_stiffness))
        dof_blocks.append(dofs)
    for index, interface in enumerate(interfaces or []):
        if not interface.active:
            continue
        dofs = _dofs_from_node_indices(_element_node_indices((*interface.minus_nodes, *interface.plus_nodes), node_index))
        linear_stiffness: np.ndarray | None = None
        if _interface_is_linear(interface):
            _dofs, _fe, ke = interface_force_tangent(interface, mesh, axisymmetric=True)
            linear_stiffness = np.asarray(ke, dtype=float).copy()
        interface_blocks.append(InterfaceStiffnessBlockCache(index, dofs, linear_stiffness))
        dof_blocks.append(dofs)
    for index, structural in enumerate(structural_elements or []):
        if not structural.active:
            continue
        dofs = structural_element_dofs(structural, mesh, axisymmetric=True)
        linear_stiffness = None
        if not structural_has_nonlinear([structural]):
            _dofs, _fe, ke = structural_element_force_tangent(structural, mesh, materials, axisymmetric=True)
            linear_stiffness = np.asarray(ke, dtype=float).copy()
        structural_blocks.append(StructuralStiffnessBlockCache(index, dofs, linear_stiffness))
        dof_blocks.append(dofs)
    pattern = SparseAssemblyPattern.from_square_blocks(dof_blocks, (ndof, ndof))
    element_batches = (
        _build_precomputed_stiffness_block_batches(element_blocks, 0, "axisymmetric_element")
        if element_blocks and all(block.linear_stiffness is not None for block in element_blocks)
        else ()
    )
    interface_batches = _build_precomputed_stiffness_block_batches(interface_blocks, len(element_blocks), "axisymmetric_interface")
    structural_batches = _build_precomputed_stiffness_block_batches(structural_blocks, len(element_blocks) + len(interface_blocks), "axisymmetric_structural")
    return AxisymmetricStiffnessAssemblyCache(
        shape=(ndof, ndof),
        pattern=pattern,
        element_blocks=tuple(element_blocks),
        interface_blocks=tuple(interface_blocks),
        structural_blocks=tuple(structural_blocks),
        quad4_elastic_batch=_build_quad4_axisymmetric_elastic_batch_cache(mesh, materials, element_blocks),
        quad8_elastic_batch=_build_quad8_axisymmetric_elastic_batch_cache(mesh, materials, element_blocks),
        tri3_elastic_batch=_build_tri_axisymmetric_elastic_batch_cache(mesh, materials, element_blocks, "TRI3"),
        tri6_elastic_batch=_build_tri_axisymmetric_elastic_batch_cache(mesh, materials, element_blocks, "TRI6"),
        precomputed_element_batches=element_batches,
        precomputed_interface_batches=interface_batches,
        precomputed_structural_batches=structural_batches,
    )


def assemble_axisymmetric_stiffness_cached(
    cache: AxisymmetricStiffnessAssemblyCache,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
) -> csr_matrix:
    flat_values = cache.pattern.empty_flat_values()
    block_index = 0
    if cache.precomputed_element_batches:
        block_index = _fill_precomputed_stiffness_batches(cache.pattern, flat_values, block_index, cache.precomputed_element_batches)
    elif cache.quad4_elastic_batch is not None:
        element_values = _assemble_quad4_axisymmetric_elastic_batch_flat(cache.quad4_elastic_batch, mesh)
        block_index = cache.pattern.fill_blocks_flat(flat_values, block_index, element_values, len(cache.element_blocks))
    elif cache.quad8_elastic_batch is not None:
        element_values = _assemble_quad8_axisymmetric_elastic_batch_flat(cache.quad8_elastic_batch, mesh)
        block_index = cache.pattern.fill_blocks_flat(flat_values, block_index, element_values, len(cache.element_blocks))
    elif cache.tri3_elastic_batch is not None:
        element_values = _assemble_tri_elastic_batch_flat(cache.tri3_elastic_batch, mesh, 3, axisymmetric=True)
        block_index = cache.pattern.fill_blocks_flat(flat_values, block_index, element_values, len(cache.element_blocks))
    elif cache.tri6_elastic_batch is not None:
        element_values = _assemble_tri_elastic_batch_flat(cache.tri6_elastic_batch, mesh, 6, axisymmetric=True)
        block_index = cache.pattern.fill_blocks_flat(flat_values, block_index, element_values, len(cache.element_blocks))
    else:
        for block in cache.element_blocks:
            element = mesh.elements[block.element_index]
            material = materials[element.material]
            coords = mesh.coords[block.conn]
            if element.type.upper() == "QUAD4":
                ke = _quad4_axisymmetric_element_stiffness_fast(coords, material)
            else:
                ke = axisymmetric_element_stiffness(element.type, coords, material, element.integration)
            cache.pattern.fill_block(flat_values, block_index, ke)
            block_index += 1
    block_index = _fill_interface_stiffness_blocks(
        cache.pattern,
        flat_values,
        block_index,
        cache.interface_blocks,
        cache.precomputed_interface_batches,
        interfaces or [],
        mesh,
        axisymmetric=True,
    )
    block_index = _fill_structural_stiffness_blocks(
        cache.pattern,
        flat_values,
        block_index,
        cache.structural_blocks,
        cache.precomputed_structural_batches,
        structural_elements or [],
        mesh,
        materials,
        axisymmetric=True,
    )
    cache.pattern.validate_filled_block_count(block_index)
    return cache.pattern.assemble_flat_values(flat_values)


def _build_quad4_axisymmetric_elastic_batch_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    element_blocks: list[ElementStiffnessBlockCache],
) -> Quad4AxisymmetricElasticBatchAssemblyCache | None:
    if not element_blocks:
        return None
    conn_rows: list[np.ndarray] = []
    d4_rows: list[np.ndarray] = []
    thickness_values: list[float] = []
    for block in element_blocks:
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        if element.type.upper() != "QUAD4" or material.is_plastic:
            return None
        conn_rows.append(np.asarray(block.conn, dtype=np.int64))
        d4_rows.append(np.asarray(material.D4, dtype=np.float64))
        thickness_values.append(float(material.thickness))
    return Quad4AxisymmetricElasticBatchAssemblyCache(
        conn=np.ascontiguousarray(np.vstack(conn_rows), dtype=np.int64),
        d4=np.ascontiguousarray(np.stack(d4_rows), dtype=np.float64),
        thickness=np.ascontiguousarray(np.asarray(thickness_values, dtype=np.float64)),
    )


def _assemble_quad4_axisymmetric_elastic_batch_flat(cache: Quad4AxisymmetricElasticBatchAssemblyCache, mesh: Mesh2D) -> np.ndarray:
    values, min_det, min_radius = _quad4_axisymmetric_elastic_batch_stiffness_flat_numba(
        np.ascontiguousarray(mesh.coords, dtype=np.float64),
        cache.conn,
        cache.d4,
        cache.thickness,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD4 axisymmetric batch: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD4 axisymmetric batch: axisymmetric radius must be positive, got {min_radius:.6e}")
    return values


def _build_quad8_axisymmetric_elastic_batch_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    element_blocks: list[ElementStiffnessBlockCache],
) -> Quad8AxisymmetricElasticBatchAssemblyCache | None:
    if not element_blocks:
        return None
    conn_rows: list[np.ndarray] = []
    d4_rows: list[np.ndarray] = []
    cdev_rows: list[np.ndarray] = []
    cvol_rows: list[np.ndarray] = []
    pvol_rows: list[np.ndarray] = []
    idev_rows: list[np.ndarray] = []
    thickness_values: list[float] = []
    mode_values: list[int] = []
    for block in element_blocks:
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        if element.type.upper() != "QUAD8" or material.is_plastic:
            return None
        conn_rows.append(np.asarray(block.conn, dtype=np.int64))
        pvol = np.asarray(material.volumetric_projector, dtype=np.float64)
        d4_rows.append(np.asarray(material.D4, dtype=np.float64))
        cdev_rows.append(np.asarray(material.C_dev, dtype=np.float64))
        cvol_rows.append(np.asarray(material.C_vol, dtype=np.float64))
        pvol_rows.append(pvol)
        idev_rows.append(np.eye(4, dtype=np.float64) - pvol)
        thickness_values.append(float(material.thickness))
        mode_values.append(int(_quad4_mode_code(normalize_integration(element.integration))))
    return Quad8AxisymmetricElasticBatchAssemblyCache(
        conn=np.ascontiguousarray(np.vstack(conn_rows), dtype=np.int64),
        d4=np.ascontiguousarray(np.stack(d4_rows), dtype=np.float64),
        cdev=np.ascontiguousarray(np.stack(cdev_rows), dtype=np.float64),
        cvol=np.ascontiguousarray(np.stack(cvol_rows), dtype=np.float64),
        pvol=np.ascontiguousarray(np.stack(pvol_rows), dtype=np.float64),
        idev=np.ascontiguousarray(np.stack(idev_rows), dtype=np.float64),
        thickness=np.ascontiguousarray(np.asarray(thickness_values, dtype=np.float64)),
        mode_codes=np.ascontiguousarray(np.asarray(mode_values, dtype=np.int64)),
    )


def _assemble_quad8_axisymmetric_elastic_batch_flat(cache: Quad8AxisymmetricElasticBatchAssemblyCache, mesh: Mesh2D) -> np.ndarray:
    values, min_det, min_radius, min_volume = _quad8_axisymmetric_elastic_batch_stiffness_flat_numba(
        np.ascontiguousarray(mesh.coords, dtype=np.float64),
        cache.conn,
        cache.d4,
        cache.cdev,
        cache.cvol,
        cache.pvol,
        cache.idev,
        cache.thickness,
        cache.mode_codes,
    )
    if min_det <= 0.0:
        raise FEM2DError(f"QUAD8 axisymmetric batch: detJ must be positive, got {min_det:.6e}")
    if min_radius <= 0.0:
        raise FEM2DError(f"QUAD8 axisymmetric batch: axisymmetric radius must be positive, got {min_radius:.6e}")
    if np.any(cache.mode_codes == _QUAD4_MODE_BBAR) and min_volume <= 0.0:
        raise FEM2DError("QUAD8 axisymmetric batch: non-positive element measure")
    return values


def _build_tri_axisymmetric_elastic_batch_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    element_blocks: list[ElementStiffnessBlockCache],
    element_type: str,
) -> TriAxisymmetricElasticBatchAssemblyCache | None:
    if not element_blocks:
        return None
    target_type = element_type.upper()
    conn_rows: list[np.ndarray] = []
    d4_rows: list[np.ndarray] = []
    cdev_rows: list[np.ndarray] = []
    cvol_rows: list[np.ndarray] = []
    pvol_rows: list[np.ndarray] = []
    idev_rows: list[np.ndarray] = []
    thickness_values: list[float] = []
    mode_values: list[int] = []
    for block in element_blocks:
        element = mesh.elements[block.element_index]
        material = materials[element.material]
        if element.type.upper() != target_type or material.is_plastic:
            return None
        conn_rows.append(np.asarray(block.conn, dtype=np.int64))
        pvol = np.asarray(material.volumetric_projector, dtype=np.float64)
        d4_rows.append(np.asarray(material.D4, dtype=np.float64))
        cdev_rows.append(np.asarray(material.C_dev, dtype=np.float64))
        cvol_rows.append(np.asarray(material.C_vol, dtype=np.float64))
        pvol_rows.append(pvol)
        idev_rows.append(np.eye(4, dtype=np.float64) - pvol)
        thickness_values.append(float(material.thickness))
        mode_values.append(int(_quad4_mode_code(normalize_integration(element.integration))))
    return TriAxisymmetricElasticBatchAssemblyCache(
        conn=np.ascontiguousarray(np.vstack(conn_rows), dtype=np.int64),
        d4=np.ascontiguousarray(np.stack(d4_rows), dtype=np.float64),
        cdev=np.ascontiguousarray(np.stack(cdev_rows), dtype=np.float64),
        cvol=np.ascontiguousarray(np.stack(cvol_rows), dtype=np.float64),
        pvol=np.ascontiguousarray(np.stack(pvol_rows), dtype=np.float64),
        idev=np.ascontiguousarray(np.stack(idev_rows), dtype=np.float64),
        thickness=np.ascontiguousarray(np.asarray(thickness_values, dtype=np.float64)),
        mode_codes=np.ascontiguousarray(np.asarray(mode_values, dtype=np.int64)),
    )


def assemble_axisymmetric_load_vector(mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial], loads: Any) -> np.ndarray:
    node_index = mesh.node_index
    F = np.zeros(len(mesh.node_ids) * 2, dtype=float)
    for load in _ensure_list(loads):
        if not isinstance(load, Mapping):
            raise FEM2DError("each load must be a mapping")
        ltype = str(load.get("type", "")).lower().strip()
        if ltype in {"gravity", "self_weight", "body"} or bool(load.get("self_weight", False)):
            _add_axisymmetric_body_weight(F, mesh, materials, float(load.get("gx", 0.0)), float(load.get("gy", -1.0)), float(load.get("scale", 1.0)), load)
            continue
        if "node" in load or "nodes" in load or "set" in load:
            for nid in _target_nodes(mesh, load):
                i = node_index[nid]
                F[2 * i] += float(load.get("fx", load.get("px", load.get("ux", 0.0))))
                F[2 * i + 1] += float(load.get("fy", load.get("py", load.get("uy", 0.0))))
        if "edge" in load or "edges" in load:
            edges = load.get("edges", [load.get("edge")])
            if isinstance(edges, str):
                edges = _edge_set(mesh, edges)
            tx1, ty1, tx2, ty2 = _edge_traction_components(load)
            for edge in edges:
                edge_nodes = tuple(str(n) for n in _require_sequence(edge, "load.edge"))
                _add_axisymmetric_edge_traction(F, mesh, edge_nodes, tx1, ty1, tx2, ty2)
    return F

def _add_axisymmetric_body_weight(F: np.ndarray, mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial], gx: float, gy: float, scale: float, target: Mapping[str, Any] | None = None) -> None:
    node_index = mesh.node_index
    for element in mesh.elements:
        if not element.active:
            continue
        if target is not None and not _body_load_targets_element(mesh, element, target):
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        fe = np.zeros(2 * len(element.nodes), dtype=float)
        bx, by = _body_force_density(material, gx, gy, scale, target)
        for gp in integration_points(element.type, "FULL"):
            _B4, detJ, N, radius = axisymmetric_strain_displacement_matrix(element.type, coords, gp)
            dV = detJ * gp[2] * material.thickness * 2.0 * math.pi * radius
            for a, Na in enumerate(N):
                fe[2 * a] += Na * bx * dV
                fe[2 * a + 1] += Na * by * dV
        F[_dofs_from_node_indices(conn)] += fe

def _add_axisymmetric_edge_traction(F: np.ndarray, mesh: Mesh2D, edge_nodes: tuple[str, ...], tx: float, ty: float, tx2: float | None = None, ty2: float | None = None) -> None:
    node_index = mesh.node_index
    tx_end = tx if tx2 is None else tx2
    ty_end = ty if ty2 is None else ty2
    uniform = math.isclose(tx, tx_end, rel_tol=1.0e-12, abs_tol=1.0e-12) and math.isclose(ty, ty_end, rel_tol=1.0e-12, abs_tol=1.0e-12)
    if uniform and len(edge_nodes) in {2, 3}:
        pts = np.array([mesh.coords[node_index[nid]] for nid in edge_nodes], dtype=float)
        fe = _quad8_axisymmetric_edge_traction_fast(pts, tx, ty)
        for a, nid in enumerate(edge_nodes):
            i = node_index[nid]
            F[2 * i] += float(fe[2 * a])
            F[2 * i + 1] += float(fe[2 * a + 1])
        return
    if len(edge_nodes) == 2:
        pts = np.array([mesh.coords[node_index[nid]] for nid in edge_nodes], dtype=float)
        gauss = [(-1.0 / math.sqrt(3.0), 1.0), (1.0 / math.sqrt(3.0), 1.0)]
        for s, w in gauss:
            N = np.array([(1.0 - s) * 0.5, (1.0 + s) * 0.5], dtype=float)
            dN = np.array([-0.5, 0.5], dtype=float)
            tangent = dN @ pts
            jac = float(np.linalg.norm(tangent))
            radius = max(float(N @ pts[:, 0]), np.finfo(float).eps)
            alpha = (s + 1.0) * 0.5
            qx = tx * (1.0 - alpha) + tx_end * alpha
            qy = ty * (1.0 - alpha) + ty_end * alpha
            dS = 2.0 * math.pi * radius * jac * w
            for a, nid in enumerate(edge_nodes):
                i = node_index[nid]
                F[2 * i] += N[a] * qx * dS
                F[2 * i + 1] += N[a] * qy * dS
        return
    if len(edge_nodes) == 3:
        pts = np.array([mesh.coords[node_index[nid]] for nid in edge_nodes], dtype=float)
        gauss = [(-math.sqrt(3.0 / 5.0), 5.0 / 9.0), (0.0, 8.0 / 9.0), (math.sqrt(3.0 / 5.0), 5.0 / 9.0)]
        for s, w in gauss:
            N = np.array([0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)], dtype=float)
            dN = np.array([s - 0.5, -2.0 * s, s + 0.5], dtype=float)
            tangent = dN @ pts
            jac = float(np.linalg.norm(tangent))
            radius = max(float(N @ pts[:, 0]), np.finfo(float).eps)
            alpha = (s + 1.0) * 0.5
            qx = tx * (1.0 - alpha) + tx_end * alpha
            qy = ty * (1.0 - alpha) + ty_end * alpha
            dS = 2.0 * math.pi * radius * jac * w
            for a, nid in enumerate(edge_nodes):
                i = node_index[nid]
                F[2 * i] += N[a] * qx * dS
                F[2 * i + 1] += N[a] * qy * dS
        return
    raise FEM2DError("axisymmetric edge traction supports 2-node or 3-node edges")

def assemble_load_vector(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    loads: Any,
    structural_elements: list[StructuralElement2D] | None = None,
) -> np.ndarray:
    node_index = mesh.node_index
    F = np.zeros(structural_total_dofs(mesh, structural_elements), dtype=float)
    rotation_dofs = structural_rotation_dof_map(mesh, structural_elements)
    for load in _ensure_list(loads):
        if not isinstance(load, Mapping):
            raise FEM2DError("each load must be a mapping")
        ltype = str(load.get("type", "")).lower().strip()
        if ltype in {"gravity", "self_weight", "body"} or bool(load.get("self_weight", False)):
            gx = float(load.get("gx", 0.0))
            gy = float(load.get("gy", -1.0))
            scale = float(load.get("scale", 1.0))
            _add_body_weight(F, mesh, materials, gx, gy, scale, load)
            continue
        if "node" in load or "nodes" in load or "set" in load:
            for nid in _target_nodes(mesh, load):
                i = node_index[nid]
                F[2 * i] += float(load.get("fx", load.get("px", load.get("ux", 0.0))))
                F[2 * i + 1] += float(load.get("fy", load.get("py", load.get("uy", 0.0))))
                if nid in rotation_dofs:
                    F[rotation_dofs[nid]] += float(load.get("mz", load.get("moment", load.get("rz", 0.0))))
        if "edge" in load or "edges" in load:
            edges = load.get("edges", [load.get("edge")])
            if isinstance(edges, str):
                edges = _edge_set(mesh, edges)
            tx1, ty1, tx2, ty2 = _edge_traction_components(load)
            for edge in edges:
                edge_nodes = tuple(str(n) for n in _require_sequence(edge, "load.edge"))
                _add_edge_traction(F, mesh, edge_nodes, tx1, ty1, tx2, ty2)
        for structural in structural_elements or []:
            equiv = structural_element_equivalent_load(structural, mesh, materials, load, rotation_dofs=rotation_dofs)
            if equiv is None:
                continue
            dofs, fe = equiv
            F[dofs] += fe
    return F

def _add_body_weight(F: np.ndarray, mesh: Mesh2D, materials: Mapping[str, ElasticPlaneStrainMaterial], gx: float, gy: float, scale: float, target: Mapping[str, Any] | None = None) -> None:
    node_index = mesh.node_index
    for element in mesh.elements:
        if not element.active:
            continue
        if target is not None and not _body_load_targets_element(mesh, element, target):
            continue
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        fe = np.zeros(2 * len(element.nodes), dtype=float)
        bx, by = _body_force_density(material, gx, gy, scale, target)
        for gp in integration_points(element.type, "FULL"):
            _B4, detJ, N = strain_displacement_matrix(element.type, coords, gp)
            dV = detJ * gp[2] * material.thickness
            for a, Na in enumerate(N):
                fe[2 * a] += Na * bx * dV
                fe[2 * a + 1] += Na * by * dV
        dofs = _dofs_from_node_indices(conn)
        F[dofs] += fe

def _body_force_density(material: ElasticPlaneStrainMaterial, gx: float, gy: float, scale: float, target: Mapping[str, Any] | None = None) -> tuple[float, float]:
    if target is not None and any(key in target for key in ("bx", "by", "body_fx", "body_fy", "body_x", "body_y")):
        bx = float(target.get("bx", target.get("body_fx", target.get("body_x", 0.0))) or 0.0)
        by = float(target.get("by", target.get("body_fy", target.get("body_y", 0.0))) or 0.0)
        return bx * scale, by * scale
    return material.gamma * gx * scale, material.gamma * gy * scale

def _body_load_targets_element(mesh: Mesh2D, element: Any, target: Mapping[str, Any]) -> bool:
    element_id = str(getattr(element, "id", element))
    element_material = str(getattr(element, "material", ""))
    if bool(target.get("all", False)):
        return True
    if "element" in target:
        if str(target["element"]) != element_id:
            return False
    if "elements" in target:
        raw = target["elements"]
        if isinstance(raw, str):
            if raw in mesh.element_sets:
                if element_id not in set(mesh.element_sets[raw]):
                    return False
            elif raw != element_id:
                return False
        elif element_id not in {str(value) for value in _ensure_list(raw)}:
            return False
    for key_name in ("element_set", "elementSet"):
        if key_name in target:
            key = str(target[key_name])
            if element_id not in set(mesh.element_sets.get(key, [])):
                return False
    if "set" in target:
        key = str(target["set"])
        if key not in mesh.element_sets or element_id not in set(mesh.element_sets.get(key, [])):
            return False
    if "material" in target and str(target["material"]) != element_material:
        return False
    if "materials" in target:
        raw_materials = target["materials"]
        if isinstance(raw_materials, str):
            material_names = {part.strip() for part in raw_materials.replace(";", ",").split(",") if part.strip()}
        else:
            material_names = {str(value) for value in _ensure_list(raw_materials)}
        if element_material not in material_names:
            return False
    return True

def _edge_traction_components(load: Mapping[str, Any]) -> tuple[float, float, float, float]:
    base_tx = float(load.get("tx", load.get("qx", 0.0)) or 0.0)
    base_ty = float(load.get("ty", load.get("qy", 0.0)) or 0.0)
    tx1 = float(load.get("tx1", load.get("qx1", load.get("tx_start", base_tx))) or 0.0)
    ty1 = float(load.get("ty1", load.get("qy1", load.get("ty_start", base_ty))) or 0.0)
    tx2 = float(load.get("tx2", load.get("qx2", load.get("tx_end", base_tx))) or 0.0)
    ty2 = float(load.get("ty2", load.get("qy2", load.get("ty_end", base_ty))) or 0.0)
    return tx1, ty1, tx2, ty2

def _add_edge_traction(F: np.ndarray, mesh: Mesh2D, edge_nodes: tuple[str, ...], tx: float, ty: float, tx2: float | None = None, ty2: float | None = None) -> None:
    node_index = mesh.node_index
    indices = np.asarray([node_index[nid] for nid in edge_nodes], dtype=np.int64)
    _add_edge_traction_by_indices(F, mesh.coords, indices, tx, ty, tx2, ty2)


def _add_edge_traction_by_indices(F: np.ndarray, coords_all: np.ndarray, indices: np.ndarray, tx: float, ty: float, tx2: float | None = None, ty2: float | None = None) -> None:
    tx_end = tx if tx2 is None else tx2
    ty_end = ty if ty2 is None else ty2
    if len(indices) == 2:
        p0 = coords_all[int(indices[0])]
        p1 = coords_all[int(indices[1])]
        length = float(np.linalg.norm(p1 - p0))
        i0 = int(indices[0])
        i1 = int(indices[1])
        F[2 * i0] += length * (2.0 * tx + tx_end) / 6.0
        F[2 * i0 + 1] += length * (2.0 * ty + ty_end) / 6.0
        F[2 * i1] += length * (tx + 2.0 * tx_end) / 6.0
        F[2 * i1 + 1] += length * (ty + 2.0 * ty_end) / 6.0
        return
    if len(indices) == 3:
        pts = np.asarray(coords_all[np.asarray(indices, dtype=np.int64)], dtype=float)
        gauss = [(-math.sqrt(3.0 / 5.0), 5.0 / 9.0), (0.0, 8.0 / 9.0), (math.sqrt(3.0 / 5.0), 5.0 / 9.0)]
        for s, w in gauss:
            N = np.array([0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)], dtype=float)
            dN = np.array([s - 0.5, -2.0 * s, s + 0.5], dtype=float)
            tangent = dN @ pts
            jac = float(np.linalg.norm(tangent))
            alpha = (s + 1.0) * 0.5
            qx = tx * (1.0 - alpha) + tx_end * alpha
            qy = ty * (1.0 - alpha) + ty_end * alpha
            for a, idx in enumerate(indices):
                i = int(idx)
                F[2 * i] += N[a] * qx * jac * w
                F[2 * i + 1] += N[a] * qy * jac * w
        return
    raise FEM2DError("edge traction supports 2-node or 3-node edges")
