"""Shared performance metadata contracts for 2D FEM stages."""

from __future__ import annotations

from typing import Any, Mapping

from .fem2d_types import ElasticPlaneStrainMaterial, Mesh2D, normalize_integration


DEFORMATION_MODE_SMALL = "small_deformation"
DEFORMATION_MODE_LARGE = "large_deformation"

_SMALL_MODE_ALIASES = {
    "",
    "small",
    "small_deformation",
    "small-displacement",
    "small_displacement",
    "linear_geometry",
    "reference",
    "reference_geometry",
}
_LARGE_MODE_ALIASES = {
    "large",
    "large_deformation",
    "large-displacement",
    "large_displacement",
    "finite_deformation",
    "updated_lagrangian",
    "updated-lagrangian",
}


def normalize_deformation_mode(value: Any, *, default: str = DEFORMATION_MODE_SMALL) -> str:
    raw = str(value if value is not None else default).lower().strip()
    if raw in _LARGE_MODE_ALIASES:
        return DEFORMATION_MODE_LARGE
    if raw in _SMALL_MODE_ALIASES:
        return DEFORMATION_MODE_SMALL
    return default


def deformation_mode_from_config(cfg: Mapping[str, Any], stage_cfg: Mapping[str, Any] | None = None) -> str:
    stage = stage_cfg if isinstance(stage_cfg, Mapping) else {}
    for key in ("deformation_mode", "geometry_mode", "kinematics"):
        if key in stage:
            return normalize_deformation_mode(stage.get(key))
    analysis = cfg.get("analysis", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(analysis, Mapping):
        for key in ("deformation_mode", "geometry_mode", "kinematics"):
            if key in analysis:
                return normalize_deformation_mode(analysis.get(key))
    solver = cfg.get("solver", {}) if isinstance(cfg, Mapping) else {}
    if isinstance(solver, Mapping):
        large = solver.get("large_deformation", solver.get("finite_deformation"))
        if isinstance(large, bool) and large:
            return DEFORMATION_MODE_LARGE
        if isinstance(large, Mapping) and bool(large.get("enabled", False)):
            return DEFORMATION_MODE_LARGE
    return DEFORMATION_MODE_SMALL


def common_solver_info_fields(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    *,
    geometry_mode: str,
    batched_elements: int = 0,
    fallback_reasons: list[str] | tuple[str, ...] | None = None,
    hydro_coupled: bool = False,
) -> dict[str, Any]:
    active = [element for element in mesh.elements if element.active]
    reasons = [str(reason) for reason in (fallback_reasons or []) if str(reason)]
    return {
        "geometry_mode": str(geometry_mode),
        "element_type": _unique_or_mixed(element.type.upper() for element in active),
        "integration": _unique_or_mixed(normalize_integration(element.integration) for element in active),
        "material_model": _unique_or_mixed(_material_model(materials.get(element.material)) for element in active),
        "batched_elements": int(batched_elements),
        "fallback_count": len(reasons),
        "fallback_reasons": reasons,
        "hydro_coupled": bool(hydro_coupled),
    }


def large_deformation_comparison_tolerances() -> dict[str, Any]:
    return {
        "schema": "geofem.large_deformation_comparison_tolerances.v1",
        "displacement": {"atol": 1.0e-8, "rtol": 1.0e-6},
        "reaction": {"atol": 1.0e-7, "rtol": 1.0e-5},
        "factor_of_safety": {"atol": 1.0e-6, "rtol": 1.0e-5},
        "plastic_point_count": {"atol": 0.0, "rtol": 0.0},
        "stress": {"atol": 1.0e-6, "rtol": 1.0e-5},
        "convergence_history": {"iteration_delta": 2, "residual_rtol": 1.0e-4},
        "pore_pressure": {"atol": 1.0e-8, "rtol": 1.0e-6},
        "drainage_flow": {"atol": 1.0e-8, "rtol": 1.0e-6},
        "volumetric_strain": {"atol": 1.0e-9, "rtol": 1.0e-6},
        "excess_pore_pressure_history": {"atol": 1.0e-8, "rtol": 1.0e-6},
    }


def large_deformation_fast_path_matrix() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for deformation_mode in (DEFORMATION_MODE_SMALL, DEFORMATION_MODE_LARGE):
        for element_type in ("QUAD4", "QUAD8"):
            for integration in ("FULL", "SRI", "B-BAR"):
                for material_model in ("drucker_prager", "mohr_coulomb", "srm", "tension_cutoff", "u-p"):
                    rows.append(_matrix_row(deformation_mode, element_type, integration, material_model))
    return {
        "schema": "geofem.large_deformation_fast_path_matrix.v1",
        "dimensions": ["deformation_mode", "element_type", "integration", "material_model", "tension_cutoff", "hydro_coupled"],
        "rows": rows,
    }


def large_deformation_performance_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.large_deformation_plastic_performance_contract.v1",
        "deformation_modes": [DEFORMATION_MODE_SMALL, DEFORMATION_MODE_LARGE],
        "common_solver_info_fields": [
            "geometry_mode",
            "element_type",
            "integration",
            "material_model",
            "batched_elements",
            "fallback_count",
            "fallback_reasons",
        ],
        "comparison_tolerances": large_deformation_comparison_tolerances(),
        "fast_path_matrix": large_deformation_fast_path_matrix(),
    }


def _matrix_row(deformation_mode: str, element_type: str, integration: str, material_model: str) -> dict[str, Any]:
    tension = material_model == "tension_cutoff"
    hydro = material_model == "u-p"
    if material_model == "u-p":
        status = "supported"
        reason = "large-deformation U-P cache, pressure history contract, and element coupling block outputs are available"
    elif element_type == "QUAD4" and material_model in {"drucker_prager", "mohr_coulomb"}:
        status = "supported"
        reason = "QUAD4 FULL/SRI/B-BAR plastic tangent/internal-force block contracts are available"
    elif element_type == "QUAD8" and material_model in {"drucker_prager", "mohr_coulomb"}:
        status = "supported"
        reason = "QUAD8 per-element Numba kernels cover FULL/SRI/B-BAR and share solver_info metadata"
    elif material_model == "srm":
        status = "supported"
        reason = "strength_factor is part of solver_info and plastic material parameter inputs"
    elif material_model == "tension_cutoff" and element_type in {"QUAD4", "QUAD8"}:
        status = "supported"
        reason = "tension-cutoff plastic paths report batch metadata or a tracked constitutive fallback"
    else:
        status = "fallback_safe"
        reason = "tracked fallback with explicit reason until a dedicated block kernel is selected"
    return {
        "deformation_mode": deformation_mode,
        "element_type": element_type,
        "integration": integration,
        "material_model": material_model,
        "tension_cutoff": tension,
        "hydro_coupled": hydro,
        "status": status,
        "reason": reason,
    }


def _material_model(material: ElasticPlaneStrainMaterial | None) -> str:
    if material is None:
        return "undefined"
    return str(material.advanced_model or material.model or "elastic").lower().strip()


def _unique_or_mixed(values: Any) -> str:
    unique = sorted({str(value) for value in values if str(value)})
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    return "mixed:" + ",".join(unique)


__all__ = [
    "DEFORMATION_MODE_LARGE",
    "DEFORMATION_MODE_SMALL",
    "common_solver_info_fields",
    "deformation_mode_from_config",
    "large_deformation_comparison_tolerances",
    "large_deformation_fast_path_matrix",
    "large_deformation_performance_contract",
    "normalize_deformation_mode",
]
