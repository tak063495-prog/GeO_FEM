"""Shared 2D FEM data types and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

try:
    from numba import njit  # type: ignore
except ImportError as exc:  # pragma: no cover - exercised only in an invalid installation
    raise ImportError("GeoFEM 2D core requires numba. Install requirements.txt before running 2D analyses.") from exc

_NUMBA_AVAILABLE = True

DOF_NAMES = ("ux", "uy")
SUPPORTED_ELEMENTS = {"TRI3", "TRI6", "QUAD4", "QUAD8"}
SUPPORTED_INTEGRATION = {"FULL", "SRI", "B-BAR", "BBAR", "B_BAR"}
DEACTIVATION_2D_STAGE_TYPES = {"death", "excavation", "deactivate"}
GEOSTATIC_2D_STAGE_TYPES = {"geostatic", "initial", "k0"}
SRM_2D_STAGE_TYPES = {"srm", "safety_factor"}
CONSOLIDATION_2D_STAGE_TYPES = {"consolidation", "up", "u-p", "u_p", "coupled_consolidation"}
RIKS_2D_STAGE_TYPES = {"riks", "arc_length", "arclength"}
LARGE_DEFORMATION_2D_STAGE_TYPES = {"large_deformation", "large_displacement", "finite_deformation", "updated_lagrangian"}
AXISYMMETRIC_2D_STAGE_TYPES = {"axisymmetric", "axisymmetric_static", "static_axisymmetric"}
DYNAMIC_2D_STAGE_TYPES = {"dynamic", "time_history", "dynamic_time_history", "newmark", "seismic_time_history"}
VGFLOW_2D_STAGE_TYPES = {"vgflow", "vgflow2d", "richards", "seepage_flow", "steady_seepage", "transient_seepage"}
SUPPORTED_2D_CORE_STAGE_TYPES = (
    {"", "static", "linear_static", "static_plane_strain", "plane_strain_static"}
    | AXISYMMETRIC_2D_STAGE_TYPES
    | DEACTIVATION_2D_STAGE_TYPES
    | GEOSTATIC_2D_STAGE_TYPES
    | SRM_2D_STAGE_TYPES
    | CONSOLIDATION_2D_STAGE_TYPES
    | RIKS_2D_STAGE_TYPES
    | LARGE_DEFORMATION_2D_STAGE_TYPES
    | DYNAMIC_2D_STAGE_TYPES
    | VGFLOW_2D_STAGE_TYPES
)
PLANNED_2D_CORE_STAGE_TYPES: set[str] = set()
PLANNED_2D_CORE_BLOCKS = {"auto_drain", "auto-drain"}


class FEM2DError(ValueError):
    """Raised when a 2D FEM input is invalid or numerically unsafe."""

    def __init__(self, message: str = "", *, diagnostics: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class Element2D:
    id: str
    type: str
    nodes: tuple[str, ...]
    material: str
    integration: str = "FULL"
    active: bool = True


@dataclass
class Mesh2D:
    node_ids: list[str]
    coords: np.ndarray
    elements: list[Element2D]
    node_sets: dict[str, list[str]] = field(default_factory=dict)
    element_sets: dict[str, list[str]] = field(default_factory=dict)
    _node_index_cache: dict[str, int] = field(default_factory=dict, init=False, repr=False, compare=False)
    _node_index_ids: tuple[str, ...] = field(default=(), init=False, repr=False, compare=False)

    @property
    def node_index(self) -> dict[str, int]:
        ids = tuple(self.node_ids)
        if self._node_index_ids != ids:
            self._node_index_cache = {nid: i for i, nid in enumerate(self.node_ids)}
            self._node_index_ids = ids
        return self._node_index_cache


@dataclass(frozen=True)
class Interface2D:
    id: str
    minus_nodes: tuple[str, str]
    plus_nodes: tuple[str, str]
    kn: float
    kt: float
    thickness: float = 1.0
    friction: float = 0.0
    cohesion: float = 0.0
    no_tension: bool = False
    material_model: str = ""
    roughness: float = 0.0
    dilatancy_angle: float = 0.0
    roughness_degradation: float = 0.0
    residual_roughness_ratio: float = 0.2
    hydraulic_transfer: float = 0.0
    active: bool = True
    history: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.kn < 0.0 or self.kt < 0.0:
            raise FEM2DError(f"interface {self.id}: kn and kt must be non-negative")
        if self.thickness <= 0.0:
            raise FEM2DError(f"interface {self.id}: thickness must be positive")
        if self.friction < 0.0 or self.cohesion < 0.0:
            raise FEM2DError(f"interface {self.id}: friction and cohesion must be non-negative")
        if self.roughness_degradation < 0.0:
            raise FEM2DError(f"interface {self.id}: roughness_degradation must be non-negative")
        if not 0.0 <= self.residual_roughness_ratio <= 1.0:
            raise FEM2DError(f"interface {self.id}: residual_roughness_ratio must be between 0 and 1")
        if self.hydraulic_transfer < 0.0:
            raise FEM2DError(f"interface {self.id}: hydraulic_transfer must be non-negative")


@dataclass(frozen=True)
class StructuralElement2D:
    id: str
    type: str
    nodes: tuple[str, str]
    material: str = ""
    section: dict[str, Any] = field(default_factory=dict)
    behavior: str = ""
    active: bool = True

    def __post_init__(self) -> None:
        if len(self.nodes) != 2:
            raise FEM2DError(f"structural element {self.id}: two nodes are required")


@dataclass(frozen=True)
class StressUpdate2D:
    stress: np.ndarray
    plastic: bool
    yield_value: float
    p: float
    q: float
    plastic_strain: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    kappa: float = 0.0
    active_set: tuple[int, ...] = field(default_factory=tuple)
    state_vars: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlasticState2D:
    plastic_strain: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    kappa: float = 0.0
    state_vars: dict[str, Any] = field(default_factory=dict)


_EMPTY_STATE_VARS = MappingProxyType({})


class PlasticStateView2D:
    """Array-backed plastic state used inside assembly loops."""

    __slots__ = ("plastic_strain", "kappa")

    state_vars = _EMPTY_STATE_VARS

    def __init__(self, plastic_strain: np.ndarray, kappa: float = 0.0) -> None:
        self.plastic_strain = plastic_strain
        self.kappa = float(kappa)


@dataclass(frozen=True)
class ElasticPlaneStrainMaterial:
    name: str
    E: float
    nu: float
    gamma: float = 0.0
    thickness: float = 1.0
    k0: float | None = None
    model: str = "elastic"
    cohesion: float = 0.0
    friction_angle: float = 0.0
    dilation_angle: float = 0.0
    yield_stress: float = 0.0
    hardening: float = 0.0
    tension_cutoff: bool = False
    tensile_strength: float = math.inf
    tension_cutoff_stage: str = "corrector"
    advanced_model: str = ""
    advanced_params: dict[str, Any] = field(default_factory=dict)
    mohr_coulomb_apex_policy: str = "legacy_bounded"
    mohr_coulomb_apex_policy_explicit: bool = False

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise FEM2DError(f"material {self.name}: E must be positive")
        if not (0.0 <= self.nu < 0.5):
            raise FEM2DError(f"material {self.name}: nu must satisfy 0 <= nu < 0.5")
        if self.thickness <= 0.0:
            raise FEM2DError(f"material {self.name}: thickness must be positive")
        if self.k0 is not None and self.k0 < 0.0:
            raise FEM2DError(f"material {self.name}: k0 must be non-negative")
        if self.cohesion < 0.0:
            raise FEM2DError(f"material {self.name}: cohesion must be non-negative")
        if self.yield_stress < 0.0:
            raise FEM2DError(f"material {self.name}: yield_stress must be non-negative")
        if self.hardening < 0.0:
            raise FEM2DError(f"material {self.name}: hardening must be non-negative")
        if self.tension_cutoff and self.tensile_strength < 0.0:
            raise FEM2DError(f"material {self.name}: tensile_strength must be non-negative")
        apex_policy = str(self.mohr_coulomb_apex_policy).lower().strip().replace("-", "_")
        if apex_policy not in {
            "associated_multisurface",
            "strict_nonassociated",
            "legacy_bounded",
            "rankine_cap",
        }:
            raise FEM2DError(
                f"material {self.name}: unsupported Mohr-Coulomb apex policy "
                f"'{self.mohr_coulomb_apex_policy}'"
            )
        object.__setattr__(self, "mohr_coulomb_apex_policy", apex_policy)

    @property
    def lame_lambda(self) -> float:
        return self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def shear_mu(self) -> float:
        return self.E / (2.0 * (1.0 + self.nu))

    @property
    def bulk_K(self) -> float:
        return self.E / (3.0 * (1.0 - 2.0 * self.nu))

    @cached_property
    def D4(self) -> np.ndarray:
        lam = self.lame_lambda
        mu = self.shear_mu
        return np.array(
            [
                [lam + 2.0 * mu, lam, lam, 0.0],
                [lam, lam + 2.0 * mu, lam, 0.0],
                [lam, lam, lam + 2.0 * mu, 0.0],
                [0.0, 0.0, 0.0, mu],
            ],
            dtype=float,
        )

    @cached_property
    def volumetric_projector(self) -> np.ndarray:
        v = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
        return np.outer(v, v) / 3.0

    @cached_property
    def C_vol(self) -> np.ndarray:
        v = np.array([1.0, 1.0, 1.0, 0.0], dtype=float)
        return self.bulk_K * np.outer(v, v)

    @cached_property
    def C_dev(self) -> np.ndarray:
        return self.D4 - self.C_vol

    @property
    def is_plastic(self) -> bool:
        return self.model in {
            "drucker_prager",
            "dp",
            "von_mises",
            "j2",
            "mohr_coulomb",
            "mc",
            "nonlinear_elastic",
            "hardin_drnevich",
            "duncan_chang",
            "ramberg_osgood",
            "uw_clay",
            "pastor_zienkiewicz_sand",
            "pastor_zienkiewicz_clay",
            "liquefaction",
            "bilinear_liquefaction",
        } or bool(self.advanced_model) or self.tension_cutoff


@dataclass
class StageResult2D:
    name: str
    displacements: np.ndarray
    reactions: np.ndarray
    element_results: list[dict[str, Any]]
    constrained_dofs: dict[int, float]
    active_elements: list[str] = field(default_factory=list)
    solver_info: dict[str, Any] = field(default_factory=dict)
    pore_pressure: np.ndarray | None = None
    time: float = 0.0
    plastic_state: dict[str, PlasticState2D] = field(default_factory=dict)
    plastic_state_array_cache: Any | None = None
    interface_results: list[dict[str, Any]] = field(default_factory=list)
    structural_results: list[dict[str, Any]] = field(default_factory=list)
    integration_point_results: list[dict[str, Any]] = field(default_factory=list)
    output_dir: Path | None = None


@dataclass
class SolveResult2D:
    mesh: Mesh2D
    materials: dict[str, ElasticPlaneStrainMaterial]
    stages: list[StageResult2D]
    output_dir: Path
    interfaces: list[Interface2D] = field(default_factory=list)
    structural_elements: list[StructuralElement2D] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    input_config: dict[str, Any] | None = None


def normalize_integration(value: Any) -> str:
    text = str(value or "FULL").strip().upper().replace("_", "-")
    if text == "BBAR":
        text = "B-BAR"
    if text not in SUPPORTED_INTEGRATION:
        raise FEM2DError(f"unsupported 2D integration '{value}'")
    return text


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)

_TYPE_EXPORTS = [
    "njit",
    "_NUMBA_AVAILABLE",
    "DOF_NAMES",
    "SUPPORTED_ELEMENTS",
    "SUPPORTED_INTEGRATION",
    "DEACTIVATION_2D_STAGE_TYPES",
    "GEOSTATIC_2D_STAGE_TYPES",
    "SRM_2D_STAGE_TYPES",
    "CONSOLIDATION_2D_STAGE_TYPES",
    "RIKS_2D_STAGE_TYPES",
    "LARGE_DEFORMATION_2D_STAGE_TYPES",
    "AXISYMMETRIC_2D_STAGE_TYPES",
    "DYNAMIC_2D_STAGE_TYPES",
    "SUPPORTED_2D_CORE_STAGE_TYPES",
    "PLANNED_2D_CORE_STAGE_TYPES",
    "PLANNED_2D_CORE_BLOCKS",
]


__all__ = _TYPE_EXPORTS + [
    "FEM2DError",
    "Element2D",
    "Mesh2D",
    "Interface2D",
    "StructuralElement2D",
    "StressUpdate2D",
    "PlasticState2D",
    "PlasticStateView2D",
    "ElasticPlaneStrainMaterial",
    "StageResult2D",
    "SolveResult2D",
    "normalize_integration",
    "_symmetrize",
]

