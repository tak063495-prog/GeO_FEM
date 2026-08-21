"""Block-oriented plastic tangent/internal-force helpers for 2D elements."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
from threading import RLock
from typing import Any, Mapping

import numpy as np

from .fem2d_elements import (
    _quad8_j2dp_bbar_post_fast,
    _quad8_j2dp_post_fast,
    _quad8_j2dp_tangent_force_fast,
    _quad8_mc_bbar_post_fast,
    _quad8_mc_post_fast,
    _quad8_mc_tangent_force_fast,
    _quad4_biot_matrix_fast,
    _quad4_pressure_matrices_fast,
    _quad8_biot_matrix_fast,
    _quad8_pressure_matrices_fast,
    element_stiffness,
    integration_points,
    strain_displacement_matrix,
)
from .fem2d_element_j2dp_kernels import _quad4_j2dp_post_numba, _quad4_j2dp_tangent_force_numba
from .fem2d_element_j2dp_kernels import _j2dp_post_update_numba, _j2dp_stress_tangent_numba
from .fem2d_element_mohr_coulomb_kernels import (
    _quad4_mc_post_numba,
    _quad4_mc_post_update_numba,
    _quad4_mc_stress_precomputed_regularized_numba,
    _quad4_mc_stress_tangent_state_active_set_numba,
    _quad4_mc_stress_tangent_numba,
    _quad4_mc_tangent_force_numba,
)
from .fem2d_element_numba_primitives import _quad4_add_btcb_numba, _quad4_b_det_numba, _quad4_project_b_numba, _quad4_symmetrize_numba
from .fem2d_materials import (
    _mc_plane_coeffs,
    _mc_python_candidate_matrix_cache,
    _mc_reduced_parameters,
    _plastic_state_key,
    _yield_surface_parameters,
    algorithmic_material_tangent,
    record_mohr_coulomb_active_set_batch,
    record_mohr_coulomb_numba_regularized_batch,
    update_plane_strain_stress,
)
from .fem2d_plastic_state_arrays import PlasticStateArrayCache
from .fem2d_types import FEM2DError, ElasticPlaneStrainMaterial, Mesh2D, PlasticState2D, PlasticStateView2D, njit, normalize_integration
from .fem2d_utils import _dofs_from_node_indices, _element_node_indices


PLASTIC_BATCH_STATUS_OK = 0
PLASTIC_BATCH_STATUS_INVALID_GEOMETRY = 1
PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK = 2
PLASTIC_BATCH_STATUS_UNSUPPORTED = 3

_STRENGTH_PARAMETER_ARRAY_CACHE_MAX = 128
_STRENGTH_PARAMETER_ARRAY_CACHE: OrderedDict[tuple[Any, ...], tuple[np.ndarray, ...]] = OrderedDict()
_STRENGTH_PARAMETER_ARRAY_CACHE_LOCK = RLock()
_STRENGTH_PARAMETER_ARRAY_CACHE_HITS = 0
_STRENGTH_PARAMETER_ARRAY_CACHE_MISSES = 0


def _plastic_block_kernel_name(block: "PlasticElementBlock") -> str:
    if block.element_type == "QUAD4" and block.material_model in {"j2", "drucker_prager"} and not block.tension_cutoff:
        if block.integration == "SRI":
            return "quad4_sri_j2dp_batch_numba"
        if block.integration == "B-BAR":
            return "quad4_bbar_j2dp_batch_numba"
        if block.integration == "FULL":
            return "quad4_full_j2dp_batch_numba"
    if (
        block.element_type == "QUAD4"
        and block.material_model == "mohr_coulomb"
        and not block.tension_cutoff
        and (
            block.mohr_coulomb_apex_policy == "associated_multisurface"
            or (
                block.mohr_coulomb_apex_policy == "legacy_bounded"
                and block.integration in {"SRI", "B-BAR"}
            )
        )
    ):
        if block.integration == "SRI":
            return "quad4_sri_mc_batch_numba"
        if block.integration == "B-BAR":
            return "quad4_bbar_mc_batch_numba"
        if block.integration == "FULL":
            return "quad4_full_mc_batch_numba"
    return "generic"


@dataclass(frozen=True)
class PlasticElementBlock:
    element_indices: np.ndarray
    element_ids: tuple[str, ...]
    element_type: str
    integration: str
    material_model: str
    tension_cutoff: bool
    mohr_coulomb_apex_policy: str = "legacy_bounded"

    @property
    def element_count(self) -> int:
        return int(self.element_indices.size)

    def solver_info(self) -> dict[str, Any]:
        return {
            "element_type": self.element_type,
            "integration": self.integration,
            "material_model": self.material_model,
            "tension_cutoff": self.tension_cutoff,
            "mohr_coulomb_apex_policy": self.mohr_coulomb_apex_policy,
            "batched_elements": self.element_count,
            "batch_kernel": _plastic_block_kernel_name(self),
        }


def _quad4_mc_geometry_block_key(
    block: PlasticElementBlock,
) -> tuple[str, str, str, bool, str]:
    return (
        block.element_type,
        block.integration,
        block.material_model,
        bool(block.tension_cutoff),
        block.mohr_coulomb_apex_policy,
    )


@dataclass(frozen=True)
class Quad4MCGeometryBlockCache:
    conn: np.ndarray
    dofs: np.ndarray
    strain_b: np.ndarray
    force_b: np.ndarray
    volumes: np.ndarray
    min_det_values: np.ndarray
    status_flags: np.ndarray


@dataclass
class Quad4MCGeometryCache:
    """Factor-invariant QUAD4 MC geometry and integration coefficients."""

    node_ids: tuple[str, ...]
    coords_shape: tuple[int, int]
    coordinate_digest: str
    blocks: dict[tuple[str, str, str, bool, str], Quad4MCGeometryBlockCache]
    block_cache_hits: int = 0
    block_cache_misses: int = 0

    def block_for(
        self, block: PlasticElementBlock
    ) -> Quad4MCGeometryBlockCache | None:
        cached = self.blocks.get(_quad4_mc_geometry_block_key(block))
        if cached is None or cached.dofs.shape[0] != block.element_count:
            self.block_cache_misses += 1
            return None
        self.block_cache_hits += 1
        return cached

    def solver_info(self) -> dict[str, Any]:
        element_count = sum(
            int(block.dofs.shape[0]) for block in self.blocks.values()
        )
        return {
            "enabled": bool(self.blocks),
            "scope": "small_deformation_step_and_srm_factor",
            "coordinate_digest": self.coordinate_digest,
            "block_count": int(len(self.blocks)),
            "element_count": int(element_count),
            "block_cache_hits": int(self.block_cache_hits),
            "block_cache_misses": int(self.block_cache_misses),
            "cached_coefficients": [
                "connectivity",
                "dofs",
                "strain_B",
                "force_B",
                "integration_volume",
                "minimum_detJ",
            ],
        }


@dataclass
class MohrCoulombActiveSetBlockCache:
    active_ids: np.ndarray
    active_counts: np.ndarray
    tangents: np.ndarray
    strains: np.ndarray
    stresses: np.ndarray
    tangent_valid: np.ndarray
    tangent_reuse_counts: np.ndarray


@dataclass
class MohrCoulombActiveSetCache:
    """Per-Newton-solve active-set hints for QUAD4 SRI/B-bar MC batches."""

    enabled: bool = True
    tangent_reuse_enabled: bool = True
    tangent_reuse_disabled_reason: str = ""
    direct_consistent_tangent_enabled: bool = True
    regularized_projection_numerical_tangent: bool = True
    strict_unstable_points_only: bool = False
    numerical_tangent_switch_reason: str = ""
    geometry_cache: Quad4MCGeometryCache | None = None
    _blocks: dict[tuple[Any, ...], MohrCoulombActiveSetBlockCache] = field(default_factory=dict)
    block_cache_hits: int = 0
    block_cache_misses: int = 0
    tangent_invalidation_count: int = 0
    tangent_invalidated_point_count: int = 0
    tangent_invalidation_reasons: dict[str, int] = field(default_factory=dict)
    numerical_tangent_switch_count: int = 0

    def hint_arrays(
        self,
        block: PlasticElementBlock,
        strength_factor: float,
    ) -> MohrCoulombActiveSetBlockCache:
        state_points = 5 if block.integration == "SRI" else 4
        key = (
            block.element_type,
            block.integration,
            block.material_model,
            block.mohr_coulomb_apex_policy,
            block.element_count,
            block.element_ids[0] if block.element_ids else "",
            block.element_ids[-1] if block.element_ids else "",
            round(float(strength_factor), 12),
        )
        cached = self._blocks.get(key)
        expected_ids = (block.element_count, state_points, 3)
        expected_counts = (block.element_count, state_points)
        if (
            cached is not None
            and cached.active_ids.shape == expected_ids
            and cached.active_counts.shape == expected_counts
        ):
            self.block_cache_hits += 1
            return cached
        active_ids = np.full(expected_ids, -1, dtype=np.int64)
        active_counts = np.zeros(expected_counts, dtype=np.int64)
        cached = MohrCoulombActiveSetBlockCache(
            active_ids=active_ids,
            active_counts=active_counts,
            tangents=np.zeros((*expected_counts, 4, 4), dtype=np.float64),
            strains=np.zeros((*expected_counts, 4), dtype=np.float64),
            stresses=np.zeros((*expected_counts, 4), dtype=np.float64),
            tangent_valid=np.zeros(expected_counts, dtype=np.bool_),
            tangent_reuse_counts=np.zeros(expected_counts, dtype=np.int64),
        )
        self._blocks[key] = cached
        self.block_cache_misses += 1
        return cached

    def invalidate_tangents(self, reason: str) -> int:
        """Discard approximate tangents while preserving active-set IDs."""

        invalidated = 0
        for block in self._blocks.values():
            invalidated += int(np.count_nonzero(block.tangent_valid))
            block.tangent_valid.fill(False)
            block.tangent_reuse_counts.fill(0)
        normalized_reason = str(reason or "unspecified")
        self.tangent_invalidation_count += 1
        self.tangent_invalidated_point_count += invalidated
        self.tangent_invalidation_reasons[normalized_reason] = (
            self.tangent_invalidation_reasons.get(normalized_reason, 0) + 1
        )
        return invalidated

    def disable_tangent_reuse(self, reason: str) -> None:
        self.tangent_reuse_enabled = False
        self.tangent_reuse_disabled_reason = str(reason or "strict_tangent")
        self.invalidate_tangents(self.tangent_reuse_disabled_reason)

    def force_numerical_tangent(self, reason: str) -> int:
        """Keep active-set hints but stop using approximate/direct tangents."""

        normalized_reason = str(reason or "adaptive_numerical_tangent")
        changed = bool(
            self.tangent_reuse_enabled or self.direct_consistent_tangent_enabled
        )
        if not changed:
            return 0
        self.tangent_reuse_enabled = False
        self.direct_consistent_tangent_enabled = False
        self.tangent_reuse_disabled_reason = normalized_reason
        self.numerical_tangent_switch_reason = normalized_reason
        self.numerical_tangent_switch_count += 1
        return self.invalidate_tangents(normalized_reason)

    def solver_info(self) -> dict[str, Any]:
        point_capacity = sum(int(block.active_counts.size) for block in self._blocks.values())
        return {
            "enabled": bool(self.enabled),
            "policy": (
                "validated_active_set_with_bounded_secant_tangent_reuse"
                if self.tangent_reuse_enabled
                else (
                    "validated_active_set_with_direct_consistent_tangent"
                    if self.direct_consistent_tangent_enabled
                    else "validated_active_set_with_regularized_numerical_tangent"
                )
            ),
            "tangent_reuse_enabled": bool(self.tangent_reuse_enabled),
            "tangent_reuse_disabled_reason": self.tangent_reuse_disabled_reason,
            "direct_consistent_tangent_enabled": bool(
                self.direct_consistent_tangent_enabled
            ),
            "regularized_projection_numerical_tangent": bool(
                self.regularized_projection_numerical_tangent
            ),
            "strict_unstable_points_only": bool(
                self.strict_unstable_points_only
            ),
            "numerical_tangent_switch_count": int(
                self.numerical_tangent_switch_count
            ),
            "numerical_tangent_switch_reason": self.numerical_tangent_switch_reason,
            "refresh_interval": 8,
            "max_relative_strain_change": 0.15,
            "secant_update": "rank_one_broyden",
            "consistent_tangent": (
                (
                    "strict_direct_stable_then_numerical_unstable_points"
                    if self.strict_unstable_points_only
                    else "adaptive_fixed_active_set_then_regularized_numerical"
                )
                if self.direct_consistent_tangent_enabled
                else "numerical_regularized_projection"
            ),
            "block_entries": int(len(self._blocks)),
            "integration_point_capacity": int(point_capacity),
            "block_cache_hits": int(self.block_cache_hits),
            "block_cache_misses": int(self.block_cache_misses),
            "tangent_invalidation_count": int(self.tangent_invalidation_count),
            "tangent_invalidated_point_count": int(
                self.tangent_invalidated_point_count
            ),
            "tangent_invalidation_reasons": dict(
                self.tangent_invalidation_reasons
            ),
            "cache_scope": "single_newton_solve",
            "cutback_reset_policy": "new_cache_per_increment_attempt",
            "geometry_cache": (
                {"enabled": False, "reason": "not_supplied"}
                if self.geometry_cache is None
                else self.geometry_cache.solver_info()
            ),
        }


@dataclass(frozen=True)
class PlasticBatchResult:
    block: PlasticElementBlock
    dofs: np.ndarray
    ke_values: np.ndarray
    internal_force_values: np.ndarray
    updated_state_values: np.ndarray
    status_flags: np.ndarray
    fallback_reasons: tuple[str, ...]
    min_det_values: np.ndarray
    kernel: str = "generic"
    mc_numba_regularized_projection_count: int = 0
    mc_active_set_update_attempt_count: int = 0
    mc_active_set_update_hit_count: int = 0
    mc_active_set_regularized_update_hit_count: int = 0

    def solver_info(self) -> dict[str, Any]:
        reasons = list(self.fallback_reasons)
        return {
            **self.block.solver_info(),
            "fallback_count": int(np.count_nonzero(self.status_flags)),
            "fallback_reasons": reasons,
            "status_flags": [int(value) for value in self.status_flags.tolist()],
            "min_det": float(np.min(self.min_det_values)) if self.min_det_values.size else 0.0,
            "kernel": self.kernel,
            "mc_numba_regularized_projection_count": int(self.mc_numba_regularized_projection_count),
            "mc_active_set_update_attempt_count": int(self.mc_active_set_update_attempt_count),
            "mc_active_set_update_hit_count": int(self.mc_active_set_update_hit_count),
            "mc_active_set_regularized_update_hit_count": int(
                self.mc_active_set_regularized_update_hit_count
            ),
        }


@dataclass(frozen=True)
class PlasticForceCandidateBatchResult:
    block: PlasticElementBlock
    dofs: np.ndarray
    internal_force_values: np.ndarray
    status_flags: np.ndarray


def evaluate_mc_internal_force_candidates(
    block: PlasticElementBlock,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    displacements: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: Any | None = None,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    strength_factor: float = 1.0,
    mohr_coulomb_active_set_cache: MohrCoulombActiveSetCache | None = None,
    quad4_mc_geometry_cache: Quad4MCGeometryCache | None = None,
) -> PlasticForceCandidateBatchResult | None:
    """Evaluate ordered line-search forces without constructing tangents."""

    candidates = np.ascontiguousarray(displacements, dtype=np.float64)
    if (
        candidates.ndim != 2
        or candidates.shape[0] == 0
        or block.element_type != "QUAD4"
        or block.integration not in {"SRI", "B-BAR"}
        or block.material_model != "mohr_coulomb"
        or block.tension_cutoff
        or block.mohr_coulomb_apex_policy
        not in {"associated_multisurface", "legacy_bounded"}
    ):
        return None
    geometry_cache = quad4_mc_geometry_cache
    if geometry_cache is None and mohr_coulomb_active_set_cache is not None:
        geometry_cache = mohr_coulomb_active_set_cache.geometry_cache
    geometry = _quad4_mc_geometry_block_for_evaluation(
        block, mesh, materials, geometry_cache
    )
    _conn, dofs, mats, initial, plastic_strains, kappas = _block_arrays(
        block,
        mesh,
        materials,
        candidates[0],
        initial_stresses,
        plastic_state,
        plastic_state_cache,
        initial_stress_cache,
        geometry,
    )
    (
        d4,
        _s4,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        hardening,
        _thickness,
        operator_indices,
        operator1,
        operator2,
        operator3,
        candidate_h,
    ) = _mc_material_arrays(mats, strength_factor)
    state_points = 5 if block.integration == "SRI" else 4
    if (
        mohr_coulomb_active_set_cache is not None
        and mohr_coulomb_active_set_cache.enabled
    ):
        hints = mohr_coulomb_active_set_cache.hint_arrays(
            block, strength_factor
        )
        active_ids_hint = hints.active_ids
        active_count_hint = hints.active_counts
    else:
        active_ids_hint = np.full(
            (block.element_count, state_points, 3), -1, dtype=np.int64
        )
        active_count_hint = np.zeros(
            (block.element_count, state_points), dtype=np.int64
        )
    force_values, status_flags = _quad4_mc_mode_force_candidates_numba(
        dofs,
        candidates,
        d4,
        initial,
        plastic_strains,
        kappas,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        hardening,
        operator_indices,
        operator1,
        operator2,
        operator3,
        candidate_h,
        1 if block.integration == "SRI" else 2,
        active_ids_hint,
        active_count_hint,
        geometry.strain_b,
        geometry.force_b,
        geometry.volumes,
        geometry.status_flags,
        1 if block.mohr_coulomb_apex_policy == "legacy_bounded" else 0,
    )
    return PlasticForceCandidateBatchResult(
        block=block,
        dofs=dofs,
        internal_force_values=force_values,
        status_flags=status_flags,
    )


@dataclass(frozen=True)
class UPCouplingBlockResult:
    kuu_values: np.ndarray
    kup_values: np.ndarray
    kpu_values: np.ndarray
    kpp_values: np.ndarray
    flow_residual_values: np.ndarray
    updated_pressure_state_values: np.ndarray
    status_flags: np.ndarray


def plastic_batch_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.plastic_batch.v1",
        "block_keys": ["element_type", "material_model", "integration", "tension_cutoff"],
        "outputs": ["ke_values", "internal_force_values", "updated_state_values", "status_flags"],
        "up_outputs": ["kuu_values", "kup_values", "kpu_values", "kpp_values", "flow_residual_values", "updated_pressure_state_values"],
        "supported_blocks": [
            "QUAD4:FULL:j2",
            "QUAD4:FULL:drucker_prager",
            "QUAD4:FULL:mohr_coulomb",
            "QUAD4:SRI:j2",
            "QUAD4:SRI:drucker_prager",
            "QUAD4:SRI:mohr_coulomb",
            "QUAD4:B-BAR:j2",
            "QUAD4:B-BAR:drucker_prager",
            "QUAD4:B-BAR:mohr_coulomb",
            "QUAD8:FULL:j2",
            "QUAD8:FULL:drucker_prager",
            "QUAD8:FULL:mohr_coulomb",
            "QUAD8:SRI:j2",
            "QUAD8:SRI:drucker_prager",
            "QUAD8:SRI:mohr_coulomb",
            "QUAD8:B-BAR:j2",
            "QUAD8:B-BAR:drucker_prager",
            "QUAD8:B-BAR:mohr_coulomb",
        ],
        "up_supported_blocks": ["QUAD4:FULL", "QUAD4:SRI", "QUAD4:B-BAR", "QUAD8:FULL", "QUAD8:SRI", "QUAD8:B-BAR"],
        "fallback_policy": "unsupported or singular elements set status_flags and can be routed to the existing safe element path",
    }


def build_plastic_element_blocks(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> list[PlasticElementBlock]:
    grouped: dict[tuple[str, str, str, bool, str], list[tuple[int, str]]] = {}
    for index, element in enumerate(mesh.elements):
        if not element.active:
            continue
        material = materials.get(element.material)
        if material is None or not material.is_plastic:
            continue
        key = (
            element.type.upper(),
            normalize_integration(element.integration),
            _canonical_material_model(material),
            bool(material.tension_cutoff),
            str(material.mohr_coulomb_apex_policy),
        )
        grouped.setdefault(key, []).append((index, element.id))
    blocks: list[PlasticElementBlock] = []
    for (
        element_type,
        integration,
        material_model,
        tension_cutoff,
        apex_policy,
    ), rows in sorted(grouped.items()):
        blocks.append(
            PlasticElementBlock(
                element_indices=np.asarray([row[0] for row in rows], dtype=np.int64),
                element_ids=tuple(row[1] for row in rows),
                element_type=element_type,
                integration=integration,
                material_model=material_model,
                tension_cutoff=tension_cutoff,
                mohr_coulomb_apex_policy=apex_policy,
            )
        )
    return blocks


def evaluate_plastic_tangent_block(
    block: PlasticElementBlock,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    displacement: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
    initial_stress_cache: Any | None = None,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    strength_factor: float = 1.0,
    mohr_coulomb_active_set_cache: MohrCoulombActiveSetCache | None = None,
    quad4_mc_geometry_cache: Quad4MCGeometryCache | None = None,
) -> PlasticBatchResult:
    if (
        block.material_model == "mohr_coulomb"
        and not (
            block.mohr_coulomb_apex_policy == "associated_multisurface"
            or (
                block.mohr_coulomb_apex_policy == "legacy_bounded"
                and block.element_type == "QUAD4"
                and block.integration in {"SRI", "B-BAR"}
                and not block.tension_cutoff
            )
        )
    ):
        if block.element_type == "QUAD4":
            return _evaluate_quad4_generic_plastic_tangent_block(
                block,
                mesh,
                materials,
                displacement,
                initial_stresses=initial_stresses,
                initial_stress_cache=initial_stress_cache,
                plastic_state=plastic_state,
                plastic_state_cache=plastic_state_cache,
                strength_factor=strength_factor,
            )
        return _unsupported_result(block, mesh, "nondefault_mohr_coulomb_apex_policy")
    if block.element_type == "QUAD8" and block.material_model in {"j2", "drucker_prager", "mohr_coulomb"}:
        return _evaluate_quad8_plastic_tangent_block(
            block,
            mesh,
            materials,
            displacement,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            strength_factor=strength_factor,
        )
    if block.element_type == "QUAD4" and block.integration in {"SRI", "B-BAR"} and block.material_model in {"j2", "drucker_prager"} and not block.tension_cutoff:
        return _evaluate_quad4_j2dp_mode_plastic_tangent_block(
            block,
            mesh,
            materials,
            displacement,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            strength_factor=strength_factor,
        )
    if block.element_type == "QUAD4" and block.integration in {"SRI", "B-BAR"} and block.material_model == "mohr_coulomb" and not block.tension_cutoff:
        return _evaluate_quad4_mc_mode_plastic_tangent_block(
            block,
            mesh,
            materials,
            displacement,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            strength_factor=strength_factor,
            mohr_coulomb_active_set_cache=mohr_coulomb_active_set_cache,
            quad4_mc_geometry_cache=quad4_mc_geometry_cache,
        )
    if block.element_type == "QUAD4" and (block.integration != "FULL" or block.tension_cutoff):
        return _evaluate_quad4_generic_plastic_tangent_block(
            block,
            mesh,
            materials,
            displacement,
            initial_stresses=initial_stresses,
            initial_stress_cache=initial_stress_cache,
            plastic_state=plastic_state,
            plastic_state_cache=plastic_state_cache,
            strength_factor=strength_factor,
        )
    if block.element_type != "QUAD4" or block.integration != "FULL":
        return _unsupported_result(block, mesh, "unsupported_block_shape_or_tension_cutoff")
    if block.material_model not in {"j2", "drucker_prager", "mohr_coulomb"}:
        return _unsupported_result(block, mesh, "unsupported_material_model")

    coords_all = np.ascontiguousarray(mesh.coords, dtype=np.float64)
    conn, dofs, mats, initial, plastic_strains, kappas = _block_arrays(block, mesh, materials, displacement, initial_stresses, plastic_state, plastic_state_cache, initial_stress_cache)
    u = np.ascontiguousarray(np.asarray(displacement, dtype=np.float64).reshape(-1), dtype=np.float64)

    if block.material_model in {"j2", "drucker_prager"}:
        d4, s4, alpha, cohesion, hardening, shear_mu, thickness = _j2dp_material_arrays(mats, strength_factor)
        ke, fe, state, status, min_det = _quad4_j2dp_full_batch_numba(
            coords_all,
            conn,
            dofs,
            u,
            d4,
            s4,
            initial,
            plastic_strains,
            kappas,
            alpha,
            cohesion,
            hardening,
            shear_mu,
            thickness,
        )
    else:
        (
            d4,
            s4,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
            thickness,
            _operator_indices,
            _operator1,
            _operator2,
            _operator3,
            _candidate_h,
        ) = _mc_material_arrays(mats, strength_factor)
        ke, fe, state, status, min_det = _quad4_mc_full_batch_numba(
            coords_all,
            conn,
            dofs,
            u,
            d4,
            s4,
            initial,
            plastic_strains,
            kappas,
            yield_coeffs,
            flow_coeffs,
            cohesion_term,
            hardening,
            thickness,
        )
    return PlasticBatchResult(
        block=block,
        dofs=dofs,
        ke_values=ke,
        internal_force_values=fe,
        updated_state_values=state,
        status_flags=status,
        fallback_reasons=_fallback_reasons_from_status(status),
        min_det_values=min_det,
        kernel="quad4_full_j2dp_batch_numba" if block.material_model in {"j2", "drucker_prager"} else "quad4_full_mc_batch_numba",
    )


def _evaluate_quad4_j2dp_mode_plastic_tangent_block(
    block: PlasticElementBlock,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    displacement: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None,
    initial_stress_cache: Any | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    strength_factor: float,
) -> PlasticBatchResult:
    coords_all = np.ascontiguousarray(mesh.coords, dtype=np.float64)
    conn, dofs, mats, initial, plastic_strains, kappas = _block_arrays(block, mesh, materials, displacement, initial_stresses, plastic_state, plastic_state_cache, initial_stress_cache)
    u = np.ascontiguousarray(np.asarray(displacement, dtype=np.float64).reshape(-1), dtype=np.float64)
    d4, s4, alpha, cohesion, hardening, shear_mu, thickness = _j2dp_material_arrays(mats, strength_factor)
    pvol, idev = _projector_arrays(mats)
    mode_code = 1 if block.integration == "SRI" else 2
    ke, fe, state, status, min_det = _quad4_j2dp_mode_batch_numba(
        coords_all,
        conn,
        dofs,
        u,
        d4,
        s4,
        pvol,
        idev,
        initial,
        plastic_strains,
        kappas,
        alpha,
        cohesion,
        hardening,
        shear_mu,
        thickness,
        mode_code,
    )
    return PlasticBatchResult(
        block=block,
        dofs=dofs,
        ke_values=ke,
        internal_force_values=fe,
        updated_state_values=state,
        status_flags=status,
        fallback_reasons=_fallback_reasons_from_status(status),
        min_det_values=min_det,
        kernel="quad4_sri_j2dp_batch_numba" if block.integration == "SRI" else "quad4_bbar_j2dp_batch_numba",
    )


def _evaluate_quad4_mc_mode_plastic_tangent_block(
    block: PlasticElementBlock,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    displacement: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None,
    initial_stress_cache: Any | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    strength_factor: float,
    mohr_coulomb_active_set_cache: MohrCoulombActiveSetCache | None,
    quad4_mc_geometry_cache: Quad4MCGeometryCache | None,
) -> PlasticBatchResult:
    geometry_cache = quad4_mc_geometry_cache
    if geometry_cache is None and mohr_coulomb_active_set_cache is not None:
        geometry_cache = mohr_coulomb_active_set_cache.geometry_cache
    geometry = _quad4_mc_geometry_block_for_evaluation(
        block, mesh, materials, geometry_cache
    )
    _conn, dofs, mats, initial, plastic_strains, kappas = _block_arrays(
        block,
        mesh,
        materials,
        displacement,
        initial_stresses,
        plastic_state,
        plastic_state_cache,
        initial_stress_cache,
        geometry,
    )
    u = np.ascontiguousarray(np.asarray(displacement, dtype=np.float64).reshape(-1), dtype=np.float64)
    (
        d4,
        s4,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        hardening,
        _thickness,
        operator_indices,
        operator1,
        operator2,
        operator3,
        candidate_h,
    ) = _mc_material_arrays(mats, strength_factor)
    mode_code = 1 if block.integration == "SRI" else 2
    if mohr_coulomb_active_set_cache is not None and mohr_coulomb_active_set_cache.enabled:
        active_set_block_cache = mohr_coulomb_active_set_cache.hint_arrays(
            block,
            strength_factor,
        )
        active_ids_hint = active_set_block_cache.active_ids
        active_count_hint = active_set_block_cache.active_counts
        tangent_cache = active_set_block_cache.tangents
        tangent_strain_cache = active_set_block_cache.strains
        tangent_stress_cache = active_set_block_cache.stresses
        tangent_cache_valid = active_set_block_cache.tangent_valid
        tangent_cache_reuse_counts = active_set_block_cache.tangent_reuse_counts
        active_set_update_enabled = True
        tangent_reuse_enabled = bool(
            mohr_coulomb_active_set_cache.tangent_reuse_enabled
        )
        direct_consistent_tangent_enabled = bool(
            mohr_coulomb_active_set_cache.direct_consistent_tangent_enabled
        )
    else:
        state_points = 5 if block.integration == "SRI" else 4
        active_ids_hint = np.full((block.element_count, state_points, 3), -1, dtype=np.int64)
        active_count_hint = np.zeros((block.element_count, state_points), dtype=np.int64)
        tangent_cache = np.zeros((block.element_count, state_points, 4, 4), dtype=np.float64)
        tangent_strain_cache = np.zeros((block.element_count, state_points, 4), dtype=np.float64)
        tangent_stress_cache = np.zeros((block.element_count, state_points, 4), dtype=np.float64)
        tangent_cache_valid = np.zeros((block.element_count, state_points), dtype=np.bool_)
        tangent_cache_reuse_counts = np.zeros((block.element_count, state_points), dtype=np.int64)
        active_set_update_enabled = False
        tangent_reuse_enabled = False
        # Preserve the established consistent tangent for ordinary return
        # mapping. The kernel itself routes regularized projections through
        # their numerical derivative.
        direct_consistent_tangent_enabled = True
    (
        ke,
        fe,
        state,
        status,
        min_det,
        regularized_counts,
        yield_violations,
        relative_yield_violations,
        relaxed_tolerances,
        active_set_attempt_counts,
        active_set_hit_counts,
        regularized_active_set_hit_counts,
    ) = _quad4_mc_mode_batch_numba(
        dofs,
        u,
        d4,
        s4,
        initial,
        plastic_strains,
        kappas,
        yield_coeffs,
        flow_coeffs,
        cohesion_term,
        hardening,
        operator_indices,
        operator1,
        operator2,
        operator3,
        candidate_h,
        mode_code,
        active_ids_hint,
        active_count_hint,
        tangent_cache,
        tangent_strain_cache,
        tangent_stress_cache,
        tangent_cache_valid,
        tangent_cache_reuse_counts,
        active_set_update_enabled,
        tangent_reuse_enabled,
        direct_consistent_tangent_enabled,
        geometry.strain_b,
        geometry.force_b,
        geometry.volumes,
        geometry.min_det_values,
        geometry.status_flags,
        1 if block.mohr_coulomb_apex_policy == "legacy_bounded" else 0,
    )
    regularized_count = record_mohr_coulomb_numba_regularized_batch(
        block.element_ids,
        regularized_counts,
        yield_violations,
        relative_yield_violations,
        relaxed_tolerances,
        status,
        apex_policy=block.mohr_coulomb_apex_policy,
    )
    active_set_counts = record_mohr_coulomb_active_set_batch(
        active_set_attempt_counts,
        active_set_hit_counts,
        regularized_active_set_hit_counts,
        status,
    )
    return PlasticBatchResult(
        block=block,
        dofs=dofs,
        ke_values=ke,
        internal_force_values=fe,
        updated_state_values=state,
        status_flags=status,
        fallback_reasons=_fallback_reasons_from_status(status),
        min_det_values=min_det,
        kernel="quad4_sri_mc_batch_numba" if block.integration == "SRI" else "quad4_bbar_mc_batch_numba",
        mc_numba_regularized_projection_count=regularized_count,
        mc_active_set_update_attempt_count=active_set_counts["attempt_count"],
        mc_active_set_update_hit_count=active_set_counts["hit_count"],
        mc_active_set_regularized_update_hit_count=active_set_counts[
            "regularized_hit_count"
        ],
    )


def _evaluate_quad4_generic_plastic_tangent_block(
    block: PlasticElementBlock,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    displacement: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None,
    initial_stress_cache: Any | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    strength_factor: float,
) -> PlasticBatchResult:
    if block.material_model not in {"j2", "drucker_prager", "mohr_coulomb"}:
        return _unsupported_result(block, mesh, "unsupported_material_model")
    _conn, dofs, mats, initial, _plastic_strains, _kappas = _block_arrays(block, mesh, materials, displacement, initial_stresses, plastic_state, plastic_state_cache, initial_stress_cache)
    u = np.asarray(displacement, dtype=float).reshape(-1)
    n = block.element_count
    state_points = _state_point_count(block.element_type, block.integration)
    ke_values = np.zeros((n, 8 * 8), dtype=float)
    internal_force_values = np.zeros((n, 8), dtype=float)
    updated_state_values = np.zeros((n, state_points, 5), dtype=float)
    status_flags = np.zeros(n, dtype=np.int64)
    min_det_values = np.zeros(n, dtype=float)
    for row, element_index in enumerate(block.element_indices):
        element = mesh.elements[int(element_index)]
        conn = _element_node_indices(element.nodes, mesh.node_index)
        coords = mesh.coords[conn]
        ue = u[dofs[row]]
        material = mats[row]
        min_det_values[row] = _minimum_block_det(element.type, coords)
        try:
            ke, fe, state_values = _quad4_generic_tangent_force_state(
                element,
                coords,
                ue,
                material,
                initial[row],
                strength_factor=strength_factor,
                plastic_state=plastic_state,
                plastic_state_cache=plastic_state_cache,
            )
            ke_values[row, :] = ke.reshape(-1)
            internal_force_values[row, :] = fe
            updated_state_values[row, : state_values.shape[0], :] = state_values
        except FEM2DError:
            status_flags[row] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY if min_det_values[row] <= 0.0 else PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
    return PlasticBatchResult(
        block=block,
        dofs=dofs,
        ke_values=ke_values,
        internal_force_values=internal_force_values,
        updated_state_values=updated_state_values,
        status_flags=status_flags,
        fallback_reasons=_fallback_reasons_from_status(status_flags),
        min_det_values=min_det_values,
    )


def _quad4_generic_tangent_force_state(
    element: Any,
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial: np.ndarray,
    *,
    strength_factor: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mode = normalize_integration(element.integration)
    full_points = integration_points(element.type, "FULL")
    Pvol = material.volumetric_projector
    Pdev = np.eye(4) - Pvol
    ke = np.zeros((8, 8), dtype=float)
    fe = np.zeros(8, dtype=float)
    state_values = np.zeros((_state_point_count("QUAD4", mode), 5), dtype=float)

    if mode == "B-BAR":
        volume = 0.0
        Bv_acc = np.zeros((4, 8), dtype=float)
        cached: list[tuple[int, np.ndarray, float]] = []
        for gp_index, gp in enumerate(full_points):
            B4, detJ, _N = strain_displacement_matrix(element.type, coords, gp)
            if detJ <= 0.0:
                raise FEM2DError(f"{element.type}: detJ must be positive, got {detJ:.6e}")
            dV = detJ * gp[2] * material.thickness
            volume += dV
            Bv_acc += (Pvol @ B4) * dV
            cached.append((gp_index, B4, dV))
        if volume <= 0.0:
            raise FEM2DError(f"{element.type}: non-positive element measure")
        Bv_bar = Bv_acc / volume
        for gp_index, B4, dV in cached:
            B_eff = (Pdev @ B4) + Bv_bar
            strain = B_eff @ ue
            state = _plastic_state_for_point(plastic_state, element.id, gp_index, plastic_state_cache)
            update = update_plane_strain_stress(
                material, strain, state=state, initial_stress=initial, strength_factor=strength_factor,
                diagnostic_context=(element.id, gp_index),
            )
            tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
            ke += B_eff.T @ tangent @ B_eff * dV
            fe += B_eff.T @ update.stress * dV
            state_values[gp_index, :] = _state_values_from_update(update)
        return ke, fe, state_values

    if mode == "SRI":
        for gp_index, gp in enumerate(full_points):
            B4, detJ, _N = strain_displacement_matrix(element.type, coords, gp)
            if detJ <= 0.0:
                raise FEM2DError(f"{element.type}: detJ must be positive, got {detJ:.6e}")
            dV = detJ * gp[2] * material.thickness
            strain = B4 @ ue
            state = _plastic_state_for_point(plastic_state, element.id, gp_index, plastic_state_cache)
            update = update_plane_strain_stress(
                material, strain, state=state, initial_stress=initial, strength_factor=strength_factor,
                diagnostic_context=(element.id, gp_index),
            )
            tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
            Bdev = Pdev @ B4
            ke += Bdev.T @ tangent @ B4 * dV
            fe += Bdev.T @ update.stress * dV
            state_values[gp_index, :] = _state_values_from_update(update)
        offset = len(full_points)
        for red_index, gp in enumerate(integration_points(element.type, "REDUCED")):
            gp_index = offset + red_index
            B4, detJ, _N = strain_displacement_matrix(element.type, coords, gp)
            if detJ <= 0.0:
                raise FEM2DError(f"{element.type}: detJ must be positive, got {detJ:.6e}")
            dV = detJ * gp[2] * material.thickness
            strain = B4 @ ue
            state = _plastic_state_for_point(plastic_state, element.id, gp_index, plastic_state_cache)
            update = update_plane_strain_stress(
                material, strain, state=state, initial_stress=initial, strength_factor=strength_factor,
                diagnostic_context=(element.id, gp_index),
            )
            tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
            Bv = Pvol @ B4
            ke += Bv.T @ tangent @ B4 * dV
            fe += Bv.T @ update.stress * dV
            state_values[gp_index, :] = _state_values_from_update(update)
        return ke, fe, state_values

    for gp_index, gp in enumerate(full_points):
        B4, detJ, _N = strain_displacement_matrix(element.type, coords, gp)
        if detJ <= 0.0:
            raise FEM2DError(f"{element.type}: detJ must be positive, got {detJ:.6e}")
        dV = detJ * gp[2] * material.thickness
        strain = B4 @ ue
        state = _plastic_state_for_point(plastic_state, element.id, gp_index, plastic_state_cache)
        update = update_plane_strain_stress(
            material, strain, state=state, initial_stress=initial, strength_factor=strength_factor,
            diagnostic_context=(element.id, gp_index),
        )
        tangent = algorithmic_material_tangent(material, strain, state=state, initial_stress=initial, strength_factor=strength_factor)
        ke += B4.T @ tangent @ B4 * dV
        fe += B4.T @ update.stress * dV
        state_values[gp_index, :] = _state_values_from_update(update)
    return ke, fe, state_values


def _projector_arrays(mats: list[ElasticPlaneStrainMaterial]) -> tuple[np.ndarray, np.ndarray]:
    pvol_rows: list[np.ndarray] = []
    idev_rows: list[np.ndarray] = []
    for material in mats:
        pvol = np.asarray(material.volumetric_projector, dtype=np.float64)
        pvol_rows.append(pvol)
        idev_rows.append(np.eye(4, dtype=np.float64) - pvol)
    return np.ascontiguousarray(np.stack(pvol_rows), dtype=np.float64), np.ascontiguousarray(np.stack(idev_rows), dtype=np.float64)


@njit(cache=True)
def _quad4_mc_geometry_coefficients_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    pvol: np.ndarray,
    idev: np.ndarray,
    thickness: np.ndarray,
    mode_code: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    element_count = conn.shape[0]
    point_count = 5 if mode_code == 1 else 4
    strain_b = np.zeros(
        (element_count, point_count, 4, 8), dtype=np.float64
    )
    force_b = np.zeros_like(strain_b)
    volumes = np.zeros((element_count, point_count), dtype=np.float64)
    min_det_values = np.zeros(element_count, dtype=np.float64)
    status_flags = np.zeros(element_count, dtype=np.int64)
    for element_index in range(element_count):
        coords = np.empty((4, 2), dtype=np.float64)
        for local_node in range(4):
            node_index = conn[element_index, local_node]
            coords[local_node, 0] = coords_all[node_index, 0]
            coords[local_node, 1] = coords_all[node_index, 1]
        min_det = 1.0e300
        if mode_code == 1:
            for gp_index in range(4):
                xi, eta, weight = _quad4_gp_full_numba(gp_index)
                bmat, det = _quad4_b_det_numba(coords, xi, eta)
                min_det = min(min_det, det)
                if det <= 0.0:
                    status_flags[element_index] = (
                        PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                    )
                    break
                strain_b[element_index, gp_index, :, :] = bmat
                force_b[element_index, gp_index, :, :] = (
                    _quad4_project_b_numba(idev[element_index], bmat)
                )
                volumes[element_index, gp_index] = (
                    det * weight * thickness[element_index]
                )
            if status_flags[element_index] == PLASTIC_BATCH_STATUS_OK:
                bmat, det = _quad4_b_det_numba(coords, 0.0, 0.0)
                min_det = min(min_det, det)
                if det <= 0.0:
                    status_flags[element_index] = (
                        PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                    )
                else:
                    strain_b[element_index, 4, :, :] = bmat
                    force_b[element_index, 4, :, :] = (
                        _quad4_project_b_numba(pvol[element_index], bmat)
                    )
                    volumes[element_index, 4] = (
                        det * 4.0 * thickness[element_index]
                    )
        else:
            raw_b = np.zeros((4, 4, 8), dtype=np.float64)
            bv_acc = np.zeros((4, 8), dtype=np.float64)
            volume = 0.0
            for gp_index in range(4):
                xi, eta, weight = _quad4_gp_full_numba(gp_index)
                bmat, det = _quad4_b_det_numba(coords, xi, eta)
                min_det = min(min_det, det)
                if det <= 0.0:
                    status_flags[element_index] = (
                        PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                    )
                    break
                dvol = det * weight * thickness[element_index]
                raw_b[gp_index, :, :] = bmat
                volumes[element_index, gp_index] = dvol
                volume += dvol
                bv_acc += (
                    _quad4_project_b_numba(pvol[element_index], bmat) * dvol
                )
            if (
                status_flags[element_index] == PLASTIC_BATCH_STATUS_OK
                and volume <= np.finfo(np.float64).eps
            ):
                status_flags[element_index] = (
                    PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                )
            if status_flags[element_index] == PLASTIC_BATCH_STATUS_OK:
                bv_bar = bv_acc / volume
                for gp_index in range(4):
                    beff = (
                        _quad4_project_b_numba(
                            idev[element_index], raw_b[gp_index]
                        )
                        + bv_bar
                    )
                    strain_b[element_index, gp_index, :, :] = beff
                    force_b[element_index, gp_index, :, :] = beff
        min_det_values[element_index] = min_det
    return strain_b, force_b, volumes, min_det_values, status_flags


def _build_quad4_mc_geometry_block_cache(
    block: PlasticElementBlock,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> Quad4MCGeometryBlockCache:
    element_count = block.element_count
    conn = np.empty((element_count, 4), dtype=np.int64)
    dofs = np.empty((element_count, 8), dtype=np.int64)
    block_materials: list[ElasticPlaneStrainMaterial] = []
    thickness = np.empty(element_count, dtype=np.float64)
    for row, raw_element_index in enumerate(block.element_indices):
        element = mesh.elements[int(raw_element_index)]
        node_indices = _element_node_indices(element.nodes, mesh.node_index)
        conn[row, :] = node_indices
        dofs[row, :] = _dofs_from_node_indices(node_indices)
        material = materials[element.material]
        block_materials.append(material)
        thickness[row] = float(material.thickness)
    pvol, idev = _projector_arrays(block_materials)
    strain_b, force_b, volumes, min_det, status = (
        _quad4_mc_geometry_coefficients_numba(
            np.ascontiguousarray(mesh.coords, dtype=np.float64),
            np.ascontiguousarray(conn, dtype=np.int64),
            pvol,
            idev,
            np.ascontiguousarray(thickness, dtype=np.float64),
            1 if block.integration == "SRI" else 2,
        )
    )
    return Quad4MCGeometryBlockCache(
        conn=np.ascontiguousarray(conn, dtype=np.int64),
        dofs=np.ascontiguousarray(dofs, dtype=np.int64),
        strain_b=np.ascontiguousarray(strain_b, dtype=np.float64),
        force_b=np.ascontiguousarray(force_b, dtype=np.float64),
        volumes=np.ascontiguousarray(volumes, dtype=np.float64),
        min_det_values=np.ascontiguousarray(min_det, dtype=np.float64),
        status_flags=np.ascontiguousarray(status, dtype=np.int64),
    )


def build_quad4_mc_geometry_cache(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
) -> Quad4MCGeometryCache:
    coords = np.ascontiguousarray(mesh.coords, dtype=np.float64)
    digest = hashlib.blake2b(
        coords.view(np.uint8), digest_size=8
    ).hexdigest()
    blocks: dict[
        tuple[str, str, str, bool, str], Quad4MCGeometryBlockCache
    ] = {}
    for block in build_plastic_element_blocks(mesh, materials):
        if (
            block.element_type == "QUAD4"
            and block.integration in {"SRI", "B-BAR"}
            and block.material_model == "mohr_coulomb"
            and not block.tension_cutoff
        ):
            blocks[_quad4_mc_geometry_block_key(block)] = (
                _build_quad4_mc_geometry_block_cache(block, mesh, materials)
            )
    return Quad4MCGeometryCache(
        node_ids=tuple(mesh.node_ids),
        coords_shape=(int(coords.shape[0]), int(coords.shape[1])),
        coordinate_digest=digest,
        blocks=blocks,
    )


def _quad4_mc_geometry_block_for_evaluation(
    block: PlasticElementBlock,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    geometry_cache: Quad4MCGeometryCache | None,
) -> Quad4MCGeometryBlockCache:
    if geometry_cache is not None:
        cached = geometry_cache.block_for(block)
        if cached is not None:
            return cached
    return _build_quad4_mc_geometry_block_cache(block, mesh, materials)


def _plastic_state_for_point(
    plastic_state: Mapping[str, PlasticState2D] | None,
    element_id: str,
    gp_index: int,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> PlasticState2D | PlasticStateView2D | None:
    if plastic_state_cache is not None:
        return plastic_state_cache.state_view_for_gp(element_id, gp_index)
    if not plastic_state:
        return None
    return plastic_state.get(_plastic_state_key(element_id, gp_index))


def _state_values_from_update(update: Any) -> np.ndarray:
    values = np.zeros(5, dtype=float)
    strain = np.asarray(update.plastic_strain, dtype=float)
    if strain.shape == (4,):
        values[:4] = strain
    values[4] = float(update.kappa)
    return values


def _evaluate_quad8_plastic_tangent_block(
    block: PlasticElementBlock,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    displacement: np.ndarray,
    *,
    initial_stresses: Mapping[str, np.ndarray] | None,
    initial_stress_cache: Any | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None,
    strength_factor: float,
) -> PlasticBatchResult:
    _conn, dofs, mats, initial, plastic_strains, kappas = _block_arrays(block, mesh, materials, displacement, initial_stresses, plastic_state, plastic_state_cache, initial_stress_cache)
    u = np.asarray(displacement, dtype=float).reshape(-1)
    n = block.element_count
    state_points = _state_point_count(block.element_type, block.integration)
    ke_values = np.zeros((n, 16 * 16), dtype=float)
    internal_force_values = np.zeros((n, 16), dtype=float)
    updated_state_values = np.zeros((n, state_points, 5), dtype=float)
    status_flags = np.zeros(n, dtype=np.int64)
    min_det_values = np.zeros(n, dtype=float)
    for row, element_index in enumerate(block.element_indices):
        element = mesh.elements[int(element_index)]
        conn = _element_node_indices(element.nodes, mesh.node_index)
        coords = mesh.coords[conn]
        ue = u[dofs[row]]
        material = mats[row]
        min_det_values[row] = _minimum_block_det(element.type, coords)
        try:
            if block.material_model in {"j2", "drucker_prager"}:
                alpha, cohesion_term = _yield_surface_parameters(material, strength_factor)
                ke, fe = _quad8_j2dp_tangent_force_fast(
                    coords,
                    ue,
                    material,
                    block.integration,
                    initial_stress=initial[row],
                    plastic_strains=plastic_strains[row],
                    kappas=kappas[row],
                    alpha=alpha,
                    cohesion_term=cohesion_term,
                )
                post = _quad8_j2dp_post_for_mode(block.integration, coords, ue, material, initial[row], plastic_strains[row], kappas[row], alpha, cohesion_term)
            else:
                fast = _quad8_mc_tangent_force_fast(
                    coords,
                    ue,
                    material,
                    block.integration,
                    initial_stress=initial[row],
                    plastic_strains=plastic_strains[row],
                    kappas=kappas[row],
                    strength_factor=strength_factor,
                )
                if fast is None:
                    status_flags[row] = PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
                    continue
                ke, fe = fast
                post = _quad8_mc_post_for_mode(block.integration, coords, ue, material, initial[row], plastic_strains[row], kappas[row], strength_factor)
            if post is None:
                post = _state_values_from_existing(plastic_strains[row], kappas[row])
            ke_values[row, :] = np.asarray(ke, dtype=float).reshape(-1)
            internal_force_values[row, :] = np.asarray(fe, dtype=float).reshape(-1)
            limit = min(state_points, post.shape[0])
            updated_state_values[row, :limit, :] = post[:limit, :]
            if limit < state_points:
                updated_state_values[row, limit:, :] = _state_values_from_existing(plastic_strains[row, limit:], kappas[row, limit:])
        except FEM2DError:
            status_flags[row] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY if min_det_values[row] <= 0.0 else PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
    return PlasticBatchResult(
        block=block,
        dofs=dofs,
        ke_values=ke_values,
        internal_force_values=internal_force_values,
        updated_state_values=updated_state_values,
        status_flags=status_flags,
        fallback_reasons=_fallback_reasons_from_status(status_flags),
        min_det_values=min_det_values,
    )


def _quad8_j2dp_post_for_mode(
    mode: str,
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: float,
    cohesion_term: float,
) -> np.ndarray | None:
    normalized = normalize_integration(mode)
    if normalized == "SRI":
        return None
    post_raw = (
        _quad8_j2dp_bbar_post_fast(coords, ue, material, initial_stress=initial, plastic_strains=plastic_strains[:9], kappas=kappas[:9], alpha=alpha, cohesion_term=cohesion_term)
        if normalized == "B-BAR"
        else _quad8_j2dp_post_fast(coords, ue, material, initial_stress=initial, plastic_strains=plastic_strains[:9], kappas=kappas[:9], alpha=alpha, cohesion_term=cohesion_term)
    )
    return _state_values_from_post(post_raw)


def _quad8_mc_post_for_mode(
    mode: str,
    coords: np.ndarray,
    ue: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    initial: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    strength_factor: float,
) -> np.ndarray | None:
    normalized = normalize_integration(mode)
    if normalized == "SRI":
        return None
    post_raw = (
        _quad8_mc_bbar_post_fast(coords, ue, material, initial_stress=initial, plastic_strains=plastic_strains[:9], kappas=kappas[:9], strength_factor=strength_factor)
        if normalized == "B-BAR"
        else _quad8_mc_post_fast(coords, ue, material, initial_stress=initial, plastic_strains=plastic_strains[:9], kappas=kappas[:9], strength_factor=strength_factor)
    )
    return None if post_raw is None else _state_values_from_post(post_raw)


def _state_values_from_post(post: np.ndarray) -> np.ndarray:
    values = np.zeros((post.shape[0], 5), dtype=float)
    values[:, :4] = post[:, 23:27]
    values[:, 4] = post[:, 22]
    return values


def _state_values_from_existing(plastic_strains: np.ndarray, kappas: np.ndarray) -> np.ndarray:
    values = np.zeros((plastic_strains.shape[0], 5), dtype=float)
    values[:, :4] = plastic_strains[:, :4]
    values[:, 4] = kappas[: plastic_strains.shape[0]]
    return values


def _minimum_block_det(element_type: str, coords: np.ndarray) -> float:
    min_det = np.inf
    for gp in integration_points(element_type, "FULL"):
        _b, det, *_rest = strain_displacement_matrix(element_type, coords, gp)
        min_det = min(min_det, float(det))
    return 0.0 if not np.isfinite(min_det) else float(min_det)


def empty_up_coupling_block_result(displacement_dofs: int, pressure_dofs: int, element_count: int = 1) -> UPCouplingBlockResult:
    u_size = int(displacement_dofs)
    p_size = int(pressure_dofs)
    n = int(element_count)
    return UPCouplingBlockResult(
        kuu_values=np.zeros((n, u_size, u_size), dtype=float),
        kup_values=np.zeros((n, u_size, p_size), dtype=float),
        kpu_values=np.zeros((n, p_size, u_size), dtype=float),
        kpp_values=np.zeros((n, p_size, p_size), dtype=float),
        flow_residual_values=np.zeros((n, p_size), dtype=float),
        updated_pressure_state_values=np.zeros((n, p_size), dtype=float),
        status_flags=np.zeros(n, dtype=np.int64),
    )


def evaluate_up_coupling_block(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    *,
    element_indices: np.ndarray | list[int] | None = None,
    displacement: np.ndarray | None = None,
    pressure: np.ndarray | None = None,
    storage: float = 1.0,
    permeability: float = 1.0,
    biot_alpha: float = 1.0,
    dt: float = 1.0,
) -> UPCouplingBlockResult:
    indices = [int(i) for i in (element_indices if element_indices is not None else range(len(mesh.elements))) if mesh.elements[int(i)].active]
    if not indices:
        return empty_up_coupling_block_result(0, 0, 0)
    max_nodes = max(len(mesh.elements[i].nodes) for i in indices)
    u_width = 2 * max_nodes
    p_width = max_nodes
    n = len(indices)
    out = empty_up_coupling_block_result(u_width, p_width, n)
    p_global = np.zeros(len(mesh.node_ids), dtype=float) if pressure is None else np.asarray(pressure, dtype=float).reshape(-1)
    if p_global.size != len(mesh.node_ids):
        raise FEM2DError("up coupling pressure vector must match node count")
    if displacement is not None:
        u_global = np.asarray(displacement, dtype=float).reshape(-1)
        if u_global.size < 2 * len(mesh.node_ids):
            raise FEM2DError("up coupling displacement vector is smaller than displacement dofs")
    dt_value = max(float(dt), np.finfo(float).eps)
    for row, element_index in enumerate(indices):
        element = mesh.elements[element_index]
        material = materials[element.material]
        conn = _element_node_indices(element.nodes, mesh.node_index)
        coords = mesh.coords[conn]
        dof_count = 2 * len(conn)
        pressure_count = len(conn)
        try:
            kuu = element_stiffness(element.type, coords, material, element.integration)
            mass, conductivity = _pressure_blocks(element.type, coords, material, storage=storage, permeability=permeability)
            biot = _biot_block(element.type, coords, material, element.integration, biot_alpha)
            pressure_lhs = mass / dt_value + conductivity
            local_pressure = p_global[conn]
            out.kuu_values[row, :dof_count, :dof_count] = kuu
            out.kup_values[row, :dof_count, :pressure_count] = -biot
            out.kpu_values[row, :pressure_count, :dof_count] = biot.T / dt_value
            out.kpp_values[row, :pressure_count, :pressure_count] = pressure_lhs
            out.flow_residual_values[row, :pressure_count] = pressure_lhs @ local_pressure
            out.updated_pressure_state_values[row, :pressure_count] = local_pressure
        except FEM2DError:
            out.status_flags[row] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
    return out


def _pressure_blocks(
    element_type: str,
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    *,
    storage: float,
    permeability: float,
) -> tuple[np.ndarray, np.ndarray]:
    etype = str(element_type).upper().strip()
    if etype == "QUAD4":
        return _quad4_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability)
    if etype == "QUAD8":
        return _quad8_pressure_matrices_fast(coords, material, storage=storage, permeability=permeability)
    raise FEM2DError(f"unsupported U-P block element '{element_type}'")


def _biot_block(
    element_type: str,
    coords: np.ndarray,
    material: ElasticPlaneStrainMaterial,
    integration: str,
    alpha: float,
) -> np.ndarray:
    etype = str(element_type).upper().strip()
    if etype == "QUAD4":
        return _quad4_biot_matrix_fast(coords, material, alpha)
    if etype == "QUAD8":
        return _quad8_biot_matrix_fast(coords, material, alpha, normalize_integration(integration))
    raise FEM2DError(f"unsupported U-P block element '{element_type}'")


@njit(cache=True)
def _quad4_gp_full_numba(gp: int) -> tuple[float, float, float]:
    a = 1.0 / np.sqrt(3.0)
    if gp == 0:
        return -a, -a, 1.0
    if gp == 1:
        return a, -a, 1.0
    if gp == 2:
        return a, a, 1.0
    return -a, a, 1.0


@njit(cache=True)
def _quad4_add_btlcbr_numba(ke: np.ndarray, b_left: np.ndarray, cmat: np.ndarray, b_right: np.ndarray, scale: float) -> None:
    for i in range(8):
        for j in range(8):
            value = 0.0
            for a in range(4):
                bia = b_left[a, i]
                if bia == 0.0:
                    continue
                for b in range(4):
                    value += bia * cmat[a, b] * b_right[b, j]
            ke[i, j] += value * scale


@njit(cache=True)
def _quad4_add_btstress_numba(fe: np.ndarray, bmat: np.ndarray, stress: np.ndarray, scale: float) -> None:
    for i in range(8):
        value = 0.0
        for j in range(4):
            value += bmat[j, i] * stress[j]
        fe[i] += value * scale


@njit(cache=True)
def _quad4_store_j2dp_state(updated_state_values: np.ndarray, element_index: int, gp_index: int, plastic_strain: np.ndarray, kappa: float) -> None:
    for i in range(4):
        updated_state_values[element_index, gp_index, i] = plastic_strain[i]
    updated_state_values[element_index, gp_index, 4] = kappa


@njit(cache=True)
def _quad4_j2dp_mode_batch_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    dofs: np.ndarray,
    u: np.ndarray,
    d4: np.ndarray,
    s4: np.ndarray,
    pvol: np.ndarray,
    idev: np.ndarray,
    initial: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: np.ndarray,
    cohesion_term: np.ndarray,
    hardening: np.ndarray,
    shear_mu: np.ndarray,
    thickness: np.ndarray,
    mode_code: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = conn.shape[0]
    state_points = 5 if mode_code == 1 else 4
    ke_values = np.zeros((n, 64), dtype=np.float64)
    internal_force_values = np.zeros((n, 8), dtype=np.float64)
    updated_state_values = np.zeros((n, state_points, 5), dtype=np.float64)
    status_flags = np.zeros(n, dtype=np.int64)
    min_det_values = np.zeros(n, dtype=np.float64)
    for e in range(n):
        coords = np.empty((4, 2), dtype=np.float64)
        ue = np.empty(8, dtype=np.float64)
        for a in range(4):
            node = conn[e, a]
            coords[a, 0] = coords_all[node, 0]
            coords[a, 1] = coords_all[node, 1]
        for i in range(8):
            ue[i] = u[dofs[e, i]]
        ke = np.zeros((8, 8), dtype=np.float64)
        fe = np.zeros(8, dtype=np.float64)
        min_det = 1.0e300

        if mode_code == 1:
            for gp in range(4):
                xi, eta, weight = _quad4_gp_full_numba(gp)
                bmat, det = _quad4_b_det_numba(coords, xi, eta)
                if det < min_det:
                    min_det = det
                if det <= 0.0:
                    status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                    break
                bdev = _quad4_project_b_numba(idev[e], bmat)
                strain = np.zeros(4, dtype=np.float64)
                for i in range(4):
                    value = 0.0
                    for j in range(8):
                        value += bmat[i, j] * ue[j]
                    strain[i] = value
                stress, tangent = _j2dp_stress_tangent_numba(
                    strain,
                    plastic_strains[e, gp],
                    kappas[e, gp],
                    d4[e],
                    initial[e],
                    alpha[e],
                    cohesion_term[e],
                    hardening[e],
                    shear_mu[e],
                )
                _post_stress, _plastic, _yield_value, _p, _q, plastic_strain_new, kappa_new = _j2dp_post_update_numba(
                    strain,
                    plastic_strains[e, gp],
                    kappas[e, gp],
                    d4[e],
                    s4[e],
                    initial[e],
                    alpha[e],
                    cohesion_term[e],
                    hardening[e],
                    shear_mu[e],
                )
                dV = det * weight * thickness[e]
                _quad4_add_btlcbr_numba(ke, bdev, tangent, bmat, dV)
                _quad4_add_btstress_numba(fe, bdev, stress, dV)
                _quad4_store_j2dp_state(updated_state_values, e, gp, plastic_strain_new, kappa_new)
            if status_flags[e] == 0:
                bmat, det = _quad4_b_det_numba(coords, 0.0, 0.0)
                if det < min_det:
                    min_det = det
                if det <= 0.0:
                    status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                else:
                    bv = _quad4_project_b_numba(pvol[e], bmat)
                    strain = np.zeros(4, dtype=np.float64)
                    for i in range(4):
                        value = 0.0
                        for j in range(8):
                            value += bmat[i, j] * ue[j]
                        strain[i] = value
                    stress, tangent = _j2dp_stress_tangent_numba(
                        strain,
                        plastic_strains[e, 4],
                        kappas[e, 4],
                        d4[e],
                        initial[e],
                        alpha[e],
                        cohesion_term[e],
                        hardening[e],
                        shear_mu[e],
                    )
                    _post_stress, _plastic, _yield_value, _p, _q, plastic_strain_new, kappa_new = _j2dp_post_update_numba(
                        strain,
                        plastic_strains[e, 4],
                        kappas[e, 4],
                        d4[e],
                        s4[e],
                        initial[e],
                        alpha[e],
                        cohesion_term[e],
                        hardening[e],
                        shear_mu[e],
                    )
                    dV = det * 4.0 * thickness[e]
                    _quad4_add_btlcbr_numba(ke, bv, tangent, bmat, dV)
                    _quad4_add_btstress_numba(fe, bv, stress, dV)
                    _quad4_store_j2dp_state(updated_state_values, e, 4, plastic_strain_new, kappa_new)
        else:
            b_cache = np.zeros((4, 4, 8), dtype=np.float64)
            dV_cache = np.zeros(4, dtype=np.float64)
            bv_acc = np.zeros((4, 8), dtype=np.float64)
            volume = 0.0
            for gp in range(4):
                xi, eta, weight = _quad4_gp_full_numba(gp)
                bmat, det = _quad4_b_det_numba(coords, xi, eta)
                if det < min_det:
                    min_det = det
                if det <= 0.0:
                    status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                    break
                dV = det * weight * thickness[e]
                dV_cache[gp] = dV
                volume += dV
                bv = _quad4_project_b_numba(pvol[e], bmat)
                for r in range(4):
                    for c in range(8):
                        b_cache[gp, r, c] = bmat[r, c]
                        bv_acc[r, c] += bv[r, c] * dV
            if status_flags[e] == 0 and volume <= np.finfo(np.float64).eps:
                status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
            if status_flags[e] == 0:
                bv_bar = bv_acc / volume
                for gp in range(4):
                    bmat = b_cache[gp]
                    bdev = _quad4_project_b_numba(idev[e], bmat)
                    beff = bdev + bv_bar
                    strain = np.zeros(4, dtype=np.float64)
                    for i in range(4):
                        value = 0.0
                        for j in range(8):
                            value += beff[i, j] * ue[j]
                        strain[i] = value
                    stress, tangent = _j2dp_stress_tangent_numba(
                        strain,
                        plastic_strains[e, gp],
                        kappas[e, gp],
                        d4[e],
                        initial[e],
                        alpha[e],
                        cohesion_term[e],
                        hardening[e],
                        shear_mu[e],
                    )
                    _post_stress, _plastic, _yield_value, _p, _q, plastic_strain_new, kappa_new = _j2dp_post_update_numba(
                        strain,
                        plastic_strains[e, gp],
                        kappas[e, gp],
                        d4[e],
                        s4[e],
                        initial[e],
                        alpha[e],
                        cohesion_term[e],
                        hardening[e],
                        shear_mu[e],
                    )
                    dV = dV_cache[gp]
                    _quad4_add_btcb_numba(ke, beff, tangent, dV)
                    _quad4_add_btstress_numba(fe, beff, stress, dV)
                    _quad4_store_j2dp_state(updated_state_values, e, gp, plastic_strain_new, kappa_new)
            if status_flags[e] == 0 and abs(alpha[e]) <= 1.0e-14:
                _quad4_symmetrize_numba(ke)

        min_det_values[e] = min_det
        if status_flags[e] != 0:
            continue
        for i in range(8):
            internal_force_values[e, i] = fe[i]
            for j in range(8):
                ke_values[e, i * 8 + j] = ke[i, j]
    return ke_values, internal_force_values, updated_state_values, status_flags, min_det_values


@njit(cache=True)
def _quad4_mc_mode_force_candidates_numba(
    dofs: np.ndarray,
    u_candidates: np.ndarray,
    d4: np.ndarray,
    initial: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: np.ndarray,
    hardening: np.ndarray,
    operator_indices: np.ndarray,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
    mode_code: int,
    active_ids_hint: np.ndarray,
    active_count_hint: np.ndarray,
    strain_b: np.ndarray,
    force_b: np.ndarray,
    volumes: np.ndarray,
    geometry_status_flags: np.ndarray,
    apex_policy_code: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_count = u_candidates.shape[0]
    element_count = dofs.shape[0]
    force_values = np.zeros(
        (candidate_count, element_count, 8), dtype=np.float64
    )
    status_flags = np.zeros(
        (candidate_count, element_count), dtype=np.int64
    )
    point_count = 5 if mode_code == 1 else 4
    for e in range(element_count):
        if geometry_status_flags[e] != PLASTIC_BATCH_STATUS_OK:
            for candidate_index in range(candidate_count):
                status_flags[candidate_index, e] = geometry_status_flags[e]
            continue

        operator_index = operator_indices[e]
        for candidate_index in range(candidate_count):
            ue = np.empty(8, dtype=np.float64)
            for local_dof in range(8):
                ue[local_dof] = u_candidates[
                    candidate_index, dofs[e, local_dof]
                ]
            fe = np.zeros(8, dtype=np.float64)
            for gp in range(point_count):
                strain = strain_b[e, gp] @ ue
                (
                    ok,
                    stress,
                    _regularized_count,
                    _yield_violation,
                    _relative_yield_violation,
                    _relaxed_tolerance,
                    _active_set_attempts,
                    _active_set_hits,
                    _regularized_active_set_hits,
                ) = _quad4_mc_stress_precomputed_regularized_numba(
                    strain,
                    plastic_strains[e, gp],
                    kappas[e, gp],
                    d4[e],
                    initial[e],
                    yield_coeffs[e],
                    flow_coeffs[e],
                    cohesion_term[e],
                    hardening[e],
                    operator1[operator_index],
                    operator2[operator_index],
                    operator3[operator_index],
                    candidate_h[operator_index],
                    active_ids_hint[e, gp],
                    active_count_hint[e, gp],
                    False,
                    apex_policy_code,
                )
                if not ok:
                    status_flags[candidate_index, e] = (
                        PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
                    )
                    break
                _quad4_add_btstress_numba(
                    fe, force_b[e, gp], stress, volumes[e, gp]
                )
            if status_flags[candidate_index, e] == PLASTIC_BATCH_STATUS_OK:
                force_values[candidate_index, e, :] = fe
    return force_values, status_flags


@njit(cache=True)
def _quad4_mc_mode_batch_uncached_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    dofs: np.ndarray,
    u: np.ndarray,
    d4: np.ndarray,
    s4: np.ndarray,
    pvol: np.ndarray,
    idev: np.ndarray,
    initial: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: np.ndarray,
    hardening: np.ndarray,
    thickness: np.ndarray,
    operator_indices: np.ndarray,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
    mode_code: int,
    active_ids_hint: np.ndarray,
    active_count_hint: np.ndarray,
    tangent_cache: np.ndarray,
    tangent_strain_cache: np.ndarray,
    tangent_stress_cache: np.ndarray,
    tangent_cache_valid: np.ndarray,
    tangent_cache_reuse_counts: np.ndarray,
    active_set_update_enabled: bool,
    tangent_reuse_enabled: bool,
    direct_consistent_tangent_enabled: bool,
    apex_policy_code: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Assemble QUAD4 MC blocks using bounded cone-tip fallback regularization."""

    n = conn.shape[0]
    state_points = 5 if mode_code == 1 else 4
    ke_values = np.zeros((n, 64), dtype=np.float64)
    internal_force_values = np.zeros((n, 8), dtype=np.float64)
    updated_state_values = np.zeros((n, state_points, 5), dtype=np.float64)
    status_flags = np.zeros(n, dtype=np.int64)
    min_det_values = np.zeros(n, dtype=np.float64)
    regularized_counts = np.zeros((n, state_points), dtype=np.int64)
    yield_violations = np.zeros((n, state_points), dtype=np.float64)
    relative_yield_violations = np.zeros((n, state_points), dtype=np.float64)
    relaxed_tolerances = np.zeros((n, state_points), dtype=np.float64)
    active_set_attempt_counts = np.zeros((n, state_points), dtype=np.int64)
    active_set_hit_counts = np.zeros((n, state_points), dtype=np.int64)
    regularized_active_set_hit_counts = np.zeros((n, state_points), dtype=np.int64)
    for e in range(n):
        operator_index = operator_indices[e]
        coords = np.empty((4, 2), dtype=np.float64)
        ue = np.empty(8, dtype=np.float64)
        for a in range(4):
            node = conn[e, a]
            coords[a, 0] = coords_all[node, 0]
            coords[a, 1] = coords_all[node, 1]
        for i in range(8):
            ue[i] = u[dofs[e, i]]
        ke = np.zeros((8, 8), dtype=np.float64)
        fe = np.zeros(8, dtype=np.float64)
        min_det = 1.0e300

        if mode_code == 1:
            for gp in range(4):
                xi, eta, weight = _quad4_gp_full_numba(gp)
                bmat, det = _quad4_b_det_numba(coords, xi, eta)
                if det < min_det:
                    min_det = det
                if det <= 0.0:
                    status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                    break
                bdev = _quad4_project_b_numba(idev[e], bmat)
                strain = bmat @ ue
                (
                    ok,
                    stress,
                    tangent,
                    plastic_strain_new,
                    kappa_new,
                    regularized_count,
                    yield_violation,
                    relative_yield_violation,
                    relaxed_tolerance,
                    active_ids_new,
                    active_count_new,
                    active_set_attempts,
                    active_set_hits,
                    regularized_active_set_hits,
                    base_regularized,
                    tangent_cache_reused,
                    tangent_cache_reuse_count_next,
                ) = _quad4_mc_stress_tangent_state_active_set_numba(
                    strain,
                    plastic_strains[e, gp],
                    kappas[e, gp],
                    d4[e],
                    s4[e],
                    initial[e],
                    yield_coeffs[e],
                    flow_coeffs[e],
                    cohesion_term[e],
                    hardening[e],
                    operator1[operator_index],
                    operator2[operator_index],
                    operator3[operator_index],
                    candidate_h[operator_index],
                    active_ids_hint[e, gp],
                    active_count_hint[e, gp],
                    tangent_cache[e, gp],
                    tangent_strain_cache[e, gp],
                    tangent_stress_cache[e, gp],
                    tangent_cache_valid[e, gp],
                    tangent_cache_reuse_counts[e, gp],
                    active_set_update_enabled,
                    tangent_reuse_enabled,
                    direct_consistent_tangent_enabled,
                    apex_policy_code,
                )
                if not ok:
                    status_flags[e] = PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
                    break
                regularized_counts[e, gp] += regularized_count
                yield_violations[e, gp] = max(yield_violations[e, gp], yield_violation)
                relative_yield_violations[e, gp] = max(
                    relative_yield_violations[e, gp],
                    relative_yield_violation,
                )
                relaxed_tolerances[e, gp] = max(relaxed_tolerances[e, gp], relaxed_tolerance)
                active_set_attempt_counts[e, gp] += active_set_attempts
                active_set_hit_counts[e, gp] += active_set_hits
                regularized_active_set_hit_counts[e, gp] += regularized_active_set_hits
                if active_count_new > 0:
                    active_count_hint[e, gp] = active_count_new
                    for slot in range(3):
                        active_ids_hint[e, gp, slot] = active_ids_new[slot]
                if base_regularized and active_count_new > 0:
                    tangent_cache[e, gp, :, :] = tangent
                    tangent_strain_cache[e, gp, :] = strain
                    tangent_stress_cache[e, gp, :] = stress
                    tangent_cache_valid[e, gp] = True
                    tangent_cache_reuse_counts[e, gp] = tangent_cache_reuse_count_next
                elif active_count_new > 0:
                    tangent_cache_valid[e, gp] = False
                    tangent_cache_reuse_counts[e, gp] = 0
                dV = det * weight * thickness[e]
                _quad4_add_btlcbr_numba(ke, bdev, tangent, bmat, dV)
                _quad4_add_btstress_numba(fe, bdev, stress, dV)
                _quad4_store_j2dp_state(updated_state_values, e, gp, plastic_strain_new, kappa_new)
            if status_flags[e] == 0:
                bmat, det = _quad4_b_det_numba(coords, 0.0, 0.0)
                if det < min_det:
                    min_det = det
                if det <= 0.0:
                    status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                else:
                    bv = _quad4_project_b_numba(pvol[e], bmat)
                    strain = bmat @ ue
                    (
                        ok,
                        stress,
                        tangent,
                        plastic_strain_new,
                        kappa_new,
                        regularized_count,
                        yield_violation,
                        relative_yield_violation,
                        relaxed_tolerance,
                        active_ids_new,
                        active_count_new,
                        active_set_attempts,
                        active_set_hits,
                        regularized_active_set_hits,
                        base_regularized,
                        tangent_cache_reused,
                        tangent_cache_reuse_count_next,
                    ) = _quad4_mc_stress_tangent_state_active_set_numba(
                        strain,
                        plastic_strains[e, 4],
                        kappas[e, 4],
                        d4[e],
                        s4[e],
                        initial[e],
                        yield_coeffs[e],
                        flow_coeffs[e],
                        cohesion_term[e],
                        hardening[e],
                        operator1[operator_index],
                        operator2[operator_index],
                        operator3[operator_index],
                        candidate_h[operator_index],
                        active_ids_hint[e, 4],
                        active_count_hint[e, 4],
                        tangent_cache[e, 4],
                        tangent_strain_cache[e, 4],
                        tangent_stress_cache[e, 4],
                        tangent_cache_valid[e, 4],
                        tangent_cache_reuse_counts[e, 4],
                        active_set_update_enabled,
                        tangent_reuse_enabled,
                        direct_consistent_tangent_enabled,
                        apex_policy_code,
                    )
                    if not ok:
                        status_flags[e] = PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
                    else:
                        regularized_counts[e, 4] += regularized_count
                        yield_violations[e, 4] = max(yield_violations[e, 4], yield_violation)
                        relative_yield_violations[e, 4] = max(
                            relative_yield_violations[e, 4],
                            relative_yield_violation,
                        )
                        relaxed_tolerances[e, 4] = max(relaxed_tolerances[e, 4], relaxed_tolerance)
                        active_set_attempt_counts[e, 4] += active_set_attempts
                        active_set_hit_counts[e, 4] += active_set_hits
                        regularized_active_set_hit_counts[e, 4] += regularized_active_set_hits
                        if active_count_new > 0:
                            active_count_hint[e, 4] = active_count_new
                            for slot in range(3):
                                active_ids_hint[e, 4, slot] = active_ids_new[slot]
                        if base_regularized and active_count_new > 0:
                            tangent_cache[e, 4, :, :] = tangent
                            tangent_strain_cache[e, 4, :] = strain
                            tangent_stress_cache[e, 4, :] = stress
                            tangent_cache_valid[e, 4] = True
                            tangent_cache_reuse_counts[e, 4] = tangent_cache_reuse_count_next
                        elif active_count_new > 0:
                            tangent_cache_valid[e, 4] = False
                            tangent_cache_reuse_counts[e, 4] = 0
                        dV = det * 4.0 * thickness[e]
                        _quad4_add_btlcbr_numba(ke, bv, tangent, bmat, dV)
                        _quad4_add_btstress_numba(fe, bv, stress, dV)
                        _quad4_store_j2dp_state(updated_state_values, e, 4, plastic_strain_new, kappa_new)
        else:
            b_cache = np.zeros((4, 4, 8), dtype=np.float64)
            dV_cache = np.zeros(4, dtype=np.float64)
            bv_acc = np.zeros((4, 8), dtype=np.float64)
            volume = 0.0
            for gp in range(4):
                xi, eta, weight = _quad4_gp_full_numba(gp)
                bmat, det = _quad4_b_det_numba(coords, xi, eta)
                if det < min_det:
                    min_det = det
                if det <= 0.0:
                    status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
                    break
                dV = det * weight * thickness[e]
                dV_cache[gp] = dV
                volume += dV
                bv = _quad4_project_b_numba(pvol[e], bmat)
                for r in range(4):
                    for c in range(8):
                        b_cache[gp, r, c] = bmat[r, c]
                        bv_acc[r, c] += bv[r, c] * dV
            if status_flags[e] == 0 and volume <= np.finfo(np.float64).eps:
                status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
            if status_flags[e] == 0:
                bv_bar = bv_acc / volume
                for gp in range(4):
                    bmat = b_cache[gp]
                    bdev = _quad4_project_b_numba(idev[e], bmat)
                    beff = bdev + bv_bar
                    strain = beff @ ue
                    (
                        ok,
                        stress,
                        tangent,
                        plastic_strain_new,
                        kappa_new,
                        regularized_count,
                        yield_violation,
                        relative_yield_violation,
                        relaxed_tolerance,
                        active_ids_new,
                        active_count_new,
                        active_set_attempts,
                        active_set_hits,
                        regularized_active_set_hits,
                        base_regularized,
                        tangent_cache_reused,
                        tangent_cache_reuse_count_next,
                    ) = _quad4_mc_stress_tangent_state_active_set_numba(
                        strain,
                        plastic_strains[e, gp],
                        kappas[e, gp],
                        d4[e],
                        s4[e],
                        initial[e],
                        yield_coeffs[e],
                        flow_coeffs[e],
                        cohesion_term[e],
                        hardening[e],
                        operator1[operator_index],
                        operator2[operator_index],
                        operator3[operator_index],
                        candidate_h[operator_index],
                        active_ids_hint[e, gp],
                        active_count_hint[e, gp],
                        tangent_cache[e, gp],
                        tangent_strain_cache[e, gp],
                        tangent_stress_cache[e, gp],
                        tangent_cache_valid[e, gp],
                        tangent_cache_reuse_counts[e, gp],
                        active_set_update_enabled,
                        tangent_reuse_enabled,
                        direct_consistent_tangent_enabled,
                        apex_policy_code,
                    )
                    if not ok:
                        status_flags[e] = PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
                        break
                    regularized_counts[e, gp] += regularized_count
                    yield_violations[e, gp] = max(yield_violations[e, gp], yield_violation)
                    relative_yield_violations[e, gp] = max(
                        relative_yield_violations[e, gp],
                        relative_yield_violation,
                    )
                    relaxed_tolerances[e, gp] = max(relaxed_tolerances[e, gp], relaxed_tolerance)
                    active_set_attempt_counts[e, gp] += active_set_attempts
                    active_set_hit_counts[e, gp] += active_set_hits
                    regularized_active_set_hit_counts[e, gp] += regularized_active_set_hits
                    if active_count_new > 0:
                        active_count_hint[e, gp] = active_count_new
                        for slot in range(3):
                            active_ids_hint[e, gp, slot] = active_ids_new[slot]
                    if base_regularized and active_count_new > 0:
                        tangent_cache[e, gp, :, :] = tangent
                        tangent_strain_cache[e, gp, :] = strain
                        tangent_stress_cache[e, gp, :] = stress
                        tangent_cache_valid[e, gp] = True
                        tangent_cache_reuse_counts[e, gp] = tangent_cache_reuse_count_next
                    elif active_count_new > 0:
                        tangent_cache_valid[e, gp] = False
                        tangent_cache_reuse_counts[e, gp] = 0
                    dV = dV_cache[gp]
                    _quad4_add_btcb_numba(ke, beff, tangent, dV)
                    _quad4_add_btstress_numba(fe, beff, stress, dV)
                    _quad4_store_j2dp_state(updated_state_values, e, gp, plastic_strain_new, kappa_new)

        min_det_values[e] = min_det
        if status_flags[e] != 0:
            continue
        for i in range(8):
            internal_force_values[e, i] = fe[i]
            for j in range(8):
                ke_values[e, i * 8 + j] = ke[i, j]
    return (
        ke_values,
        internal_force_values,
        updated_state_values,
        status_flags,
        min_det_values,
        regularized_counts,
        yield_violations,
        relative_yield_violations,
        relaxed_tolerances,
        active_set_attempt_counts,
        active_set_hit_counts,
        regularized_active_set_hit_counts,
    )


@njit(cache=True)
def _quad4_mc_mode_batch_numba(
    dofs: np.ndarray,
    u: np.ndarray,
    d4: np.ndarray,
    s4: np.ndarray,
    initial: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: np.ndarray,
    hardening: np.ndarray,
    operator_indices: np.ndarray,
    operator1: np.ndarray,
    operator2: np.ndarray,
    operator3: np.ndarray,
    candidate_h: np.ndarray,
    mode_code: int,
    active_ids_hint: np.ndarray,
    active_count_hint: np.ndarray,
    tangent_cache: np.ndarray,
    tangent_strain_cache: np.ndarray,
    tangent_stress_cache: np.ndarray,
    tangent_cache_valid: np.ndarray,
    tangent_cache_reuse_counts: np.ndarray,
    active_set_update_enabled: bool,
    tangent_reuse_enabled: bool,
    direct_consistent_tangent_enabled: bool,
    strain_b: np.ndarray,
    force_b: np.ndarray,
    volumes: np.ndarray,
    geometry_min_det_values: np.ndarray,
    geometry_status_flags: np.ndarray,
    apex_policy_code: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Assemble MC tangent/force from factor-invariant geometry arrays."""

    associated_apex_policy_revision = 3
    element_count = dofs.shape[0]
    point_count = 5 if mode_code == 1 and associated_apex_policy_revision >= 1 else 4
    ke_values = np.zeros((element_count, 64), dtype=np.float64)
    internal_force_values = np.zeros((element_count, 8), dtype=np.float64)
    updated_state_values = np.zeros(
        (element_count, point_count, 5), dtype=np.float64
    )
    status_flags = np.zeros(element_count, dtype=np.int64)
    min_det_values = geometry_min_det_values.copy()
    regularized_counts = np.zeros(
        (element_count, point_count), dtype=np.int64
    )
    yield_violations = np.zeros(
        (element_count, point_count), dtype=np.float64
    )
    relative_yield_violations = np.zeros(
        (element_count, point_count), dtype=np.float64
    )
    relaxed_tolerances = np.zeros(
        (element_count, point_count), dtype=np.float64
    )
    active_set_attempt_counts = np.zeros(
        (element_count, point_count), dtype=np.int64
    )
    active_set_hit_counts = np.zeros(
        (element_count, point_count), dtype=np.int64
    )
    regularized_active_set_hit_counts = np.zeros(
        (element_count, point_count), dtype=np.int64
    )

    for element_index in range(element_count):
        if geometry_status_flags[element_index] != PLASTIC_BATCH_STATUS_OK:
            status_flags[element_index] = geometry_status_flags[element_index]
            continue
        operator_index = operator_indices[element_index]
        ue = np.empty(8, dtype=np.float64)
        for local_dof in range(8):
            ue[local_dof] = u[dofs[element_index, local_dof]]
        ke = np.zeros((8, 8), dtype=np.float64)
        fe = np.zeros(8, dtype=np.float64)

        for gp_index in range(point_count):
            strain_matrix = strain_b[element_index, gp_index]
            force_matrix = force_b[element_index, gp_index]
            strain = strain_matrix @ ue
            (
                ok,
                stress,
                tangent,
                plastic_strain_new,
                kappa_new,
                regularized_count,
                yield_violation,
                relative_yield_violation,
                relaxed_tolerance,
                active_ids_new,
                active_count_new,
                active_set_attempts,
                active_set_hits,
                regularized_active_set_hits,
                base_regularized,
                tangent_cache_reused,
                tangent_cache_reuse_count_next,
            ) = _quad4_mc_stress_tangent_state_active_set_numba(
                strain,
                plastic_strains[element_index, gp_index],
                kappas[element_index, gp_index],
                d4[element_index],
                s4[element_index],
                initial[element_index],
                yield_coeffs[element_index],
                flow_coeffs[element_index],
                cohesion_term[element_index],
                hardening[element_index],
                operator1[operator_index],
                operator2[operator_index],
                operator3[operator_index],
                candidate_h[operator_index],
                active_ids_hint[element_index, gp_index],
                active_count_hint[element_index, gp_index],
                tangent_cache[element_index, gp_index],
                tangent_strain_cache[element_index, gp_index],
                tangent_stress_cache[element_index, gp_index],
                tangent_cache_valid[element_index, gp_index],
                tangent_cache_reuse_counts[element_index, gp_index],
                active_set_update_enabled,
                tangent_reuse_enabled,
                direct_consistent_tangent_enabled,
                apex_policy_code,
            )
            if not ok:
                status_flags[element_index] = (
                    PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
                )
                break

            regularized_counts[element_index, gp_index] += regularized_count
            yield_violations[element_index, gp_index] = max(
                yield_violations[element_index, gp_index], yield_violation
            )
            relative_yield_violations[element_index, gp_index] = max(
                relative_yield_violations[element_index, gp_index],
                relative_yield_violation,
            )
            relaxed_tolerances[element_index, gp_index] = max(
                relaxed_tolerances[element_index, gp_index], relaxed_tolerance
            )
            active_set_attempt_counts[element_index, gp_index] += (
                active_set_attempts
            )
            active_set_hit_counts[element_index, gp_index] += active_set_hits
            regularized_active_set_hit_counts[element_index, gp_index] += (
                regularized_active_set_hits
            )
            if active_count_new > 0:
                active_count_hint[element_index, gp_index] = active_count_new
                for slot in range(3):
                    active_ids_hint[element_index, gp_index, slot] = (
                        active_ids_new[slot]
                    )
            if base_regularized and active_count_new > 0:
                tangent_cache[element_index, gp_index, :, :] = tangent
                tangent_strain_cache[element_index, gp_index, :] = strain
                tangent_stress_cache[element_index, gp_index, :] = stress
                tangent_cache_valid[element_index, gp_index] = True
                tangent_cache_reuse_counts[element_index, gp_index] = (
                    tangent_cache_reuse_count_next
                )
            elif active_count_new > 0:
                tangent_cache_valid[element_index, gp_index] = False
                tangent_cache_reuse_counts[element_index, gp_index] = 0

            dvol = volumes[element_index, gp_index]
            _quad4_add_btlcbr_numba(
                ke, force_matrix, tangent, strain_matrix, dvol
            )
            _quad4_add_btstress_numba(fe, force_matrix, stress, dvol)
            _quad4_store_j2dp_state(
                updated_state_values,
                element_index,
                gp_index,
                plastic_strain_new,
                kappa_new,
            )

        if status_flags[element_index] != PLASTIC_BATCH_STATUS_OK:
            continue
        for row in range(8):
            internal_force_values[element_index, row] = fe[row]
            for column in range(8):
                ke_values[element_index, row * 8 + column] = ke[row, column]

    return (
        ke_values,
        internal_force_values,
        updated_state_values,
        status_flags,
        min_det_values,
        regularized_counts,
        yield_violations,
        relative_yield_violations,
        relaxed_tolerances,
        active_set_attempt_counts,
        active_set_hit_counts,
        regularized_active_set_hit_counts,
    )


@njit(cache=True)
def _quad4_j2dp_full_batch_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    dofs: np.ndarray,
    u: np.ndarray,
    d4: np.ndarray,
    s4: np.ndarray,
    initial: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    alpha: np.ndarray,
    cohesion_term: np.ndarray,
    hardening: np.ndarray,
    shear_mu: np.ndarray,
    thickness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = conn.shape[0]
    ke_values = np.zeros((n, 64), dtype=np.float64)
    internal_force_values = np.zeros((n, 8), dtype=np.float64)
    updated_state_values = np.zeros((n, 4, 5), dtype=np.float64)
    status_flags = np.zeros(n, dtype=np.int64)
    min_det_values = np.zeros(n, dtype=np.float64)
    for e in range(n):
        coords = np.empty((4, 2), dtype=np.float64)
        ue = np.empty(8, dtype=np.float64)
        for a in range(4):
            node = conn[e, a]
            coords[a, 0] = coords_all[node, 0]
            coords[a, 1] = coords_all[node, 1]
        for i in range(8):
            ue[i] = u[dofs[e, i]]
        ke, fe, min_det = _quad4_j2dp_tangent_force_numba(
            coords,
            ue,
            d4[e],
            initial[e],
            plastic_strains[e],
            kappas[e],
            alpha[e],
            cohesion_term[e],
            hardening[e],
            shear_mu[e],
            thickness[e],
        )
        min_det_values[e] = min_det
        if min_det <= 0.0:
            status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
            continue
        post, post_min_det = _quad4_j2dp_post_numba(
            coords,
            ue,
            d4[e],
            s4[e],
            initial[e],
            plastic_strains[e],
            kappas[e],
            alpha[e],
            cohesion_term[e],
            hardening[e],
            shear_mu[e],
            thickness[e],
        )
        if post_min_det <= 0.0:
            status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
            continue
        for i in range(8):
            internal_force_values[e, i] = fe[i]
            for j in range(8):
                ke_values[e, i * 8 + j] = ke[i, j]
        for gp in range(4):
            for k in range(4):
                updated_state_values[e, gp, k] = post[gp, 23 + k]
            updated_state_values[e, gp, 4] = post[gp, 22]
    return ke_values, internal_force_values, updated_state_values, status_flags, min_det_values


@njit(cache=True)
def _quad4_mc_full_batch_numba(
    coords_all: np.ndarray,
    conn: np.ndarray,
    dofs: np.ndarray,
    u: np.ndarray,
    d4: np.ndarray,
    s4: np.ndarray,
    initial: np.ndarray,
    plastic_strains: np.ndarray,
    kappas: np.ndarray,
    yield_coeffs: np.ndarray,
    flow_coeffs: np.ndarray,
    cohesion_term: np.ndarray,
    hardening: np.ndarray,
    thickness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = conn.shape[0]
    ke_values = np.zeros((n, 64), dtype=np.float64)
    internal_force_values = np.zeros((n, 8), dtype=np.float64)
    updated_state_values = np.zeros((n, 4, 5), dtype=np.float64)
    status_flags = np.zeros(n, dtype=np.int64)
    min_det_values = np.zeros(n, dtype=np.float64)
    for e in range(n):
        coords = np.empty((4, 2), dtype=np.float64)
        ue = np.empty(8, dtype=np.float64)
        for a in range(4):
            node = conn[e, a]
            coords[a, 0] = coords_all[node, 0]
            coords[a, 1] = coords_all[node, 1]
        for i in range(8):
            ue[i] = u[dofs[e, i]]
        ke, fe, min_det, ok = _quad4_mc_tangent_force_numba(
            coords,
            ue,
            d4[e],
            initial[e],
            plastic_strains[e],
            kappas[e],
            yield_coeffs[e],
            flow_coeffs[e],
            cohesion_term[e],
            hardening[e],
            thickness[e],
        )
        min_det_values[e] = min_det
        if min_det <= 0.0:
            status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
            continue
        if not ok:
            status_flags[e] = PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
            continue
        post, post_min_det, post_ok = _quad4_mc_post_numba(
            coords,
            ue,
            d4[e],
            s4[e],
            initial[e],
            plastic_strains[e],
            kappas[e],
            yield_coeffs[e],
            flow_coeffs[e],
            cohesion_term[e],
            hardening[e],
            thickness[e],
        )
        if post_min_det <= 0.0:
            status_flags[e] = PLASTIC_BATCH_STATUS_INVALID_GEOMETRY
            continue
        if not post_ok:
            status_flags[e] = PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK
            continue
        for i in range(8):
            internal_force_values[e, i] = fe[i]
            for j in range(8):
                ke_values[e, i * 8 + j] = ke[i, j]
        for gp in range(4):
            for k in range(4):
                updated_state_values[e, gp, k] = post[gp, 23 + k]
            updated_state_values[e, gp, 4] = post[gp, 22]
    return ke_values, internal_force_values, updated_state_values, status_flags, min_det_values


def _block_arrays(
    block: PlasticElementBlock,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    displacement: np.ndarray,
    initial_stresses: Mapping[str, np.ndarray] | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
    plastic_state_cache: PlasticStateArrayCache | None = None,
    initial_stress_cache: Any | None = None,
    geometry_block: Quad4MCGeometryBlockCache | None = None,
) -> tuple[np.ndarray, np.ndarray, list[ElasticPlaneStrainMaterial], np.ndarray, np.ndarray, np.ndarray]:
    u = np.asarray(displacement, dtype=float).reshape(-1)
    conn_rows: list[np.ndarray] = []
    dof_rows: list[np.ndarray] = []
    mats: list[ElasticPlaneStrainMaterial] = []
    initial_rows: list[np.ndarray] = []
    strain_rows: list[np.ndarray] = []
    kappa_rows: list[np.ndarray] = []
    for row, element_index in enumerate(block.element_indices):
        element = mesh.elements[int(element_index)]
        if geometry_block is None:
            conn = _element_node_indices(element.nodes, mesh.node_index)
            dofs = _dofs_from_node_indices(conn)
        else:
            conn = geometry_block.conn[row]
            dofs = geometry_block.dofs[row]
        if np.max(dofs) >= u.size:
            raise FEM2DError("plastic batch displacement vector is smaller than element dofs")
        material = materials[element.material]
        if geometry_block is None:
            conn_rows.append(np.asarray(conn, dtype=np.int64))
            dof_rows.append(np.asarray(dofs, dtype=np.int64))
        mats.append(material)
        initial = _initial_stress_for_block_element(initial_stresses, initial_stress_cache, int(element_index), element.id)
        if initial.shape != (4,):
            raise FEM2DError(f"element {element.id}: initial stress must have 4 components")
        initial_rows.append(initial)
        strains, kappas = _plastic_state_arrays(
            element.id,
            plastic_state,
            _state_point_count(block.element_type, block.integration),
            plastic_state_cache,
        )
        strain_rows.append(strains)
        kappa_rows.append(kappas)
    return (
        (
            np.ascontiguousarray(np.vstack(conn_rows), dtype=np.int64)
            if geometry_block is None
            else geometry_block.conn
        ),
        (
            np.ascontiguousarray(np.vstack(dof_rows), dtype=np.int64)
            if geometry_block is None
            else geometry_block.dofs
        ),
        mats,
        np.ascontiguousarray(np.stack(initial_rows), dtype=np.float64),
        np.ascontiguousarray(np.stack(strain_rows), dtype=np.float64),
        np.ascontiguousarray(np.stack(kappa_rows), dtype=np.float64),
    )


def _initial_stress_for_block_element(
    initial_stresses: Mapping[str, np.ndarray] | None,
    initial_stress_cache: Any | None,
    element_index: int,
    element_id: str,
) -> np.ndarray:
    if initial_stress_cache is not None:
        index_map = getattr(initial_stress_cache, "element_index_to_row", None)
        values = getattr(initial_stress_cache, "values", None)
        if index_map is not None and values is not None and 0 <= element_index < len(index_map):
            row = int(index_map[element_index])
            if row >= 0:
                return np.asarray(values[row], dtype=float)
        return np.zeros(4, dtype=float)
    return np.asarray((initial_stresses or {}).get(element_id, np.zeros(4, dtype=float)), dtype=float)


def _plastic_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
    point_count: int,
    plastic_state_cache: PlasticStateArrayCache | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if plastic_state_cache is not None:
        try:
            return plastic_state_cache.state_arrays(element_id, point_count, require_no_state_vars=True)
        except ValueError as exc:
            raise FEM2DError(str(exc)) from exc
    strains = np.zeros((point_count, 4), dtype=float)
    kappas = np.zeros(point_count, dtype=float)
    if not plastic_state:
        return strains, kappas
    for gp_index in range(point_count):
        state = plastic_state.get(_plastic_state_key(element_id, gp_index))
        if state is None:
            continue
        if state.state_vars:
            raise FEM2DError(f"element {element_id}: state_vars require the safe non-batch plastic path")
        strain = np.asarray(state.plastic_strain, dtype=float)
        if strain.shape == (4,):
            strains[gp_index, :] = strain
        kappas[gp_index] = float(state.kappa)
    return strains, kappas


def _j2dp_material_arrays(
    materials: list[ElasticPlaneStrainMaterial],
    strength_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _cached_strength_parameter_arrays(
        "j2dp",
        materials,
        strength_factor,
        lambda: _build_j2dp_material_arrays(materials, strength_factor),
    )


def _build_j2dp_material_arrays(
    materials: list[ElasticPlaneStrainMaterial],
    strength_factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d4 = []
    s4 = []
    alpha = []
    cohesion = []
    hardening = []
    shear_mu = []
    thickness = []
    for material in materials:
        a, c = _yield_surface_parameters(material, strength_factor)
        d4_arr = np.asarray(material.D4, dtype=float)
        d4.append(d4_arr)
        s4.append(np.linalg.inv(d4_arr))
        alpha.append(float(a))
        cohesion.append(float(c))
        hardening.append(float(material.hardening))
        shear_mu.append(float(material.shear_mu))
        thickness.append(float(material.thickness))
    return (
        np.ascontiguousarray(np.stack(d4), dtype=np.float64),
        np.ascontiguousarray(np.stack(s4), dtype=np.float64),
        np.ascontiguousarray(np.asarray(alpha), dtype=np.float64),
        np.ascontiguousarray(np.asarray(cohesion), dtype=np.float64),
        np.ascontiguousarray(np.asarray(hardening), dtype=np.float64),
        np.ascontiguousarray(np.asarray(shear_mu), dtype=np.float64),
        np.ascontiguousarray(np.asarray(thickness), dtype=np.float64),
    )


def _mc_material_arrays(
    materials: list[ElasticPlaneStrainMaterial],
    strength_factor: float,
) -> tuple[np.ndarray, ...]:
    return _cached_strength_parameter_arrays(
        "mohr_coulomb",
        materials,
        strength_factor,
        lambda: _build_mc_material_arrays(materials, strength_factor),
    )


def _build_mc_material_arrays(
    materials: list[ElasticPlaneStrainMaterial],
    strength_factor: float,
) -> tuple[np.ndarray, ...]:
    d4 = []
    s4 = []
    yield_coeffs = []
    flow_coeffs = []
    cohesion_term = []
    hardening = []
    thickness = []
    operator_indices = []
    operator1_rows: list[np.ndarray] = []
    operator2_rows: list[np.ndarray] = []
    operator3_rows: list[np.ndarray] = []
    candidate_h_rows: list[np.ndarray] = []
    operator_lookup: dict[tuple[bytes, bytes, bytes, float], int] = {}
    for material in materials:
        c, phi, psi = _mc_reduced_parameters(material, strength_factor)
        d4_arr = np.asarray(material.D4, dtype=float)
        yield_coeff_arr = np.ascontiguousarray(_mc_plane_coeffs(phi), dtype=np.float64)
        flow_coeff_arr = np.ascontiguousarray(_mc_plane_coeffs(psi), dtype=np.float64)
        hardening_value = float(material.hardening)
        d4.append(d4_arr)
        s4.append(np.linalg.inv(d4_arr))
        yield_coeffs.append(yield_coeff_arr)
        flow_coeffs.append(flow_coeff_arr)
        cohesion_term.append(float(2.0 * c * np.cos(phi)))
        hardening.append(hardening_value)
        thickness.append(float(material.thickness))
        operator_key = (
            yield_coeff_arr.tobytes(),
            flow_coeff_arr.tobytes(),
            np.ascontiguousarray(d4_arr[:3, :3], dtype=np.float64).tobytes(),
            hardening_value,
        )
        operator_index = operator_lookup.get(operator_key)
        if operator_index is None:
            operator_index = len(operator1_rows)
            operator_lookup[operator_key] = operator_index
            operator1 = np.zeros(6, dtype=np.float64)
            operator2 = np.zeros((6, 6, 2, 2), dtype=np.float64)
            operator3 = np.zeros((6, 6, 6, 3, 3), dtype=np.float64)
            candidate_h = np.zeros((258, 3, 3), dtype=np.float64)
            candidate_cache = _mc_python_candidate_matrix_cache(
                operator_key[0],
                operator_key[1],
                operator_key[2],
                hardening_value,
            )
            for subset, cached in candidate_cache.items():
                solution_operator = np.asarray(cached[6], dtype=np.float64)
                h_matrix = np.asarray(cached[3], dtype=np.float64)
                if len(subset) == 1:
                    operator1[subset[0]] = solution_operator[0, 0]
                    candidate_index = subset[0]
                elif len(subset) == 2:
                    operator2[subset[0], subset[1], :, :] = solution_operator
                    candidate_index = 6 + subset[0] * 6 + subset[1]
                else:
                    operator3[subset[0], subset[1], subset[2], :, :] = solution_operator
                    candidate_index = 42 + subset[0] * 36 + subset[1] * 6 + subset[2]
                candidate_h[candidate_index, : len(subset), : len(subset)] = h_matrix
            operator1_rows.append(operator1)
            operator2_rows.append(operator2)
            operator3_rows.append(operator3)
            candidate_h_rows.append(candidate_h)
        operator_indices.append(operator_index)
    return (
        np.ascontiguousarray(np.stack(d4), dtype=np.float64),
        np.ascontiguousarray(np.stack(s4), dtype=np.float64),
        np.ascontiguousarray(np.stack(yield_coeffs), dtype=np.float64),
        np.ascontiguousarray(np.stack(flow_coeffs), dtype=np.float64),
        np.ascontiguousarray(np.asarray(cohesion_term), dtype=np.float64),
        np.ascontiguousarray(np.asarray(hardening), dtype=np.float64),
        np.ascontiguousarray(np.asarray(thickness), dtype=np.float64),
        np.ascontiguousarray(np.asarray(operator_indices), dtype=np.int64),
        np.ascontiguousarray(np.stack(operator1_rows), dtype=np.float64),
        np.ascontiguousarray(np.stack(operator2_rows), dtype=np.float64),
        np.ascontiguousarray(np.stack(operator3_rows), dtype=np.float64),
        np.ascontiguousarray(np.stack(candidate_h_rows), dtype=np.float64),
    )


def _cached_strength_parameter_arrays(
    kind: str,
    materials: list[ElasticPlaneStrainMaterial],
    strength_factor: float,
    builder: Any,
) -> tuple[np.ndarray, ...]:
    global _STRENGTH_PARAMETER_ARRAY_CACHE_HITS, _STRENGTH_PARAMETER_ARRAY_CACHE_MISSES
    key = _strength_parameter_array_cache_key(kind, materials, strength_factor)
    with _STRENGTH_PARAMETER_ARRAY_CACHE_LOCK:
        cached = _STRENGTH_PARAMETER_ARRAY_CACHE.get(key)
        if cached is not None:
            _STRENGTH_PARAMETER_ARRAY_CACHE.move_to_end(key)
            _STRENGTH_PARAMETER_ARRAY_CACHE_HITS += 1
            return cached
        _STRENGTH_PARAMETER_ARRAY_CACHE_MISSES += 1
    value = builder()
    with _STRENGTH_PARAMETER_ARRAY_CACHE_LOCK:
        existing = _STRENGTH_PARAMETER_ARRAY_CACHE.get(key)
        if existing is not None:
            _STRENGTH_PARAMETER_ARRAY_CACHE.move_to_end(key)
            return existing
        _STRENGTH_PARAMETER_ARRAY_CACHE[key] = value
        while len(_STRENGTH_PARAMETER_ARRAY_CACHE) > _STRENGTH_PARAMETER_ARRAY_CACHE_MAX:
            _STRENGTH_PARAMETER_ARRAY_CACHE.popitem(last=False)
    return value


def _strength_parameter_array_cache_key(
    kind: str,
    materials: list[ElasticPlaneStrainMaterial],
    strength_factor: float,
) -> tuple[Any, ...]:
    return (
        str(kind),
        round(float(strength_factor), 12),
        tuple(_strength_material_array_signature(material) for material in materials),
    )


def _strength_material_array_signature(material: ElasticPlaneStrainMaterial) -> tuple[Any, ...]:
    return (
        str(material.model).lower().strip(),
        float(material.E),
        float(material.nu),
        float(material.thickness),
        float(material.cohesion),
        float(material.friction_angle),
        float(material.dilation_angle),
        float(material.yield_stress),
        float(material.hardening),
        bool(material.tension_cutoff),
        float(material.tensile_strength),
        str(material.mohr_coulomb_apex_policy),
    )


def clear_material_strength_parameter_array_cache() -> None:
    global _STRENGTH_PARAMETER_ARRAY_CACHE_HITS, _STRENGTH_PARAMETER_ARRAY_CACHE_MISSES
    with _STRENGTH_PARAMETER_ARRAY_CACHE_LOCK:
        _STRENGTH_PARAMETER_ARRAY_CACHE.clear()
        _STRENGTH_PARAMETER_ARRAY_CACHE_HITS = 0
        _STRENGTH_PARAMETER_ARRAY_CACHE_MISSES = 0


def material_strength_parameter_array_cache_info() -> dict[str, Any]:
    with _STRENGTH_PARAMETER_ARRAY_CACHE_LOCK:
        return {
            "enabled": True,
            "scope": "material_strength_factor_arrays",
            "entries": len(_STRENGTH_PARAMETER_ARRAY_CACHE),
            "max_entries": _STRENGTH_PARAMETER_ARRAY_CACHE_MAX,
            "hits": _STRENGTH_PARAMETER_ARRAY_CACHE_HITS,
            "misses": _STRENGTH_PARAMETER_ARRAY_CACHE_MISSES,
        }


def _unsupported_result(block: PlasticElementBlock, mesh: Mesh2D, reason: str) -> PlasticBatchResult:
    n = block.element_count
    dof_count = _dof_count(block.element_type)
    state_points = _state_point_count(block.element_type, block.integration)
    dofs = np.zeros((n, dof_count), dtype=np.int64)
    for row, element_index in enumerate(block.element_indices):
        element = mesh.elements[int(element_index)]
        conn = _element_node_indices(element.nodes, mesh.node_index)
        local_dofs = _dofs_from_node_indices(conn)
        limit = min(local_dofs.size, dof_count)
        dofs[row, :limit] = local_dofs[:limit]
    return PlasticBatchResult(
        block=block,
        dofs=dofs,
        ke_values=np.zeros((n, dof_count * dof_count), dtype=float),
        internal_force_values=np.zeros((n, dof_count), dtype=float),
        updated_state_values=np.zeros((n, state_points, 5), dtype=float),
        status_flags=np.full(n, PLASTIC_BATCH_STATUS_UNSUPPORTED, dtype=np.int64),
        fallback_reasons=(reason,),
        min_det_values=np.zeros(n, dtype=float),
    )


def _fallback_reasons_from_status(status: np.ndarray) -> tuple[str, ...]:
    reasons: list[str] = []
    values = {int(value) for value in np.asarray(status, dtype=np.int64).tolist()}
    if PLASTIC_BATCH_STATUS_INVALID_GEOMETRY in values:
        reasons.append("detJ_nonpositive")
    if PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK in values:
        reasons.append("constitutive_update_fallback")
    if PLASTIC_BATCH_STATUS_UNSUPPORTED in values:
        reasons.append("unsupported_block")
    return tuple(reasons)


def _canonical_material_model(material: ElasticPlaneStrainMaterial) -> str:
    model = str(material.advanced_model or material.model).lower().strip()
    if model in {"von_mises", "j2"}:
        return "j2"
    if model in {"drucker_prager", "dp"}:
        return "drucker_prager"
    if model in {"mohr_coulomb", "mc"}:
        return "mohr_coulomb"
    return model


def _dof_count(element_type: str) -> int:
    return 16 if str(element_type).upper() == "QUAD8" else 8


def _state_point_count(element_type: str, integration: str) -> int:
    if str(element_type).upper() == "QUAD8":
        return 13 if normalize_integration(integration) == "SRI" else 9
    if str(element_type).upper() == "QUAD4" and normalize_integration(integration) == "SRI":
        return 5
    return 4


__all__ = [
    "PLASTIC_BATCH_STATUS_CONSTITUTIVE_FALLBACK",
    "PLASTIC_BATCH_STATUS_INVALID_GEOMETRY",
    "PLASTIC_BATCH_STATUS_OK",
    "PLASTIC_BATCH_STATUS_UNSUPPORTED",
    "PlasticBatchResult",
    "PlasticElementBlock",
    "UPCouplingBlockResult",
    "build_plastic_element_blocks",
    "clear_material_strength_parameter_array_cache",
    "empty_up_coupling_block_result",
    "evaluate_up_coupling_block",
    "evaluate_plastic_tangent_block",
    "material_strength_parameter_array_cache_info",
    "plastic_batch_contract",
]
