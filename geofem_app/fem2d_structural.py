"""2-node structural line elements for the 2D displacement core."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, Mapping

import numpy as np

from .fem2d_types import ElasticPlaneStrainMaterial, FEM2DError, Mesh2D, StructuralElement2D, _symmetrize
from .fem2d_utils import _element_dofs, _ensure_list, _require_sequence


_PIPE_D = 0.500
_PIPE_T = 0.012
_PIPE_DI = _PIPE_D - 2.0 * _PIPE_T

STRUCTURAL_SECTION_LIBRARY: dict[str, dict[str, Any]] = {
    "RC_RECT_1M": {"section_name": "RC_RECT_1M", "A": 1.0, "I": 1.0 / 12.0, "As": 5.0 / 6.0, "kappa": 5.0 / 6.0, "section_type": "rc_rectangle"},
    "RC_RECT_1000X1000": {"section_name": "RC_RECT_1000X1000", "A": 1.0, "I": 1.0 / 12.0, "As": 5.0 / 6.0, "kappa": 5.0 / 6.0, "section_type": "rc_rectangle"},
    "RC_RECT_500X1000": {"section_name": "RC_RECT_500X1000", "A": 0.5, "I": 0.5 / 12.0, "As": 0.5 * 5.0 / 6.0, "kappa": 5.0 / 6.0, "section_type": "rc_rectangle"},
    "STEEL_H_300X300": {"section_name": "STEEL_H_300X300", "A": 0.01198, "I": 2.04e-4, "As": 0.0065, "kappa": 1.0, "section_type": "steel_h"},
    "STEEL_PIPE_D500_T12": {
        "section_name": "STEEL_PIPE_D500_T12",
        "A": math.pi * (_PIPE_D**2 - _PIPE_DI**2) / 4.0,
        "I": math.pi * (_PIPE_D**4 - _PIPE_DI**4) / 64.0,
        "As": 0.5 * math.pi * (_PIPE_D**2 - _PIPE_DI**2) / 4.0,
        "kappa": 0.9,
        "section_type": "steel_pipe",
    },
}

STRUCTURAL_SPRING_HYSTERESIS_MODELS: dict[str, dict[str, Any]] = {
    "BILINEAR_STANDARD": {"law": "bilinear", "post_yield_ratio": 0.05, "residual_ratio": 1.0},
    "DEGRADING_TAKEDA_LIKE": {
        "law": "degrading",
        "post_yield_ratio": 0.03,
        "degradation": 0.08,
        "pinching": 0.20,
        "unloading_stiffness_ratio": 0.60,
        "residual_ratio": 0.25,
    },
    "PINCHING_CLOUGH_LIKE": {
        "law": "pinching",
        "post_yield_ratio": 0.02,
        "degradation": 0.05,
        "pinching": 0.45,
        "unloading_stiffness_ratio": 0.45,
        "residual_ratio": 0.20,
    },
    "GAP_BILINEAR": {"law": "bilinear", "post_yield_ratio": 0.02, "gap": 0.0, "residual_ratio": 1.0},
    "GEOFEAS_LIKE_DEGRADING": {
        "law": "degrading",
        "post_yield_ratio": 0.03,
        "degradation": 0.06,
        "pinching": 0.20,
        "unloading_stiffness_ratio": 0.60,
        "residual_ratio": 0.25,
        "parameter_system": "geo_feas_like",
    },
    "GEOFEAS_LIKE_PINCHING": {
        "law": "pinching",
        "post_yield_ratio": 0.02,
        "degradation": 0.05,
        "pinching": 0.40,
        "unloading_stiffness_ratio": 0.45,
        "residual_ratio": 0.20,
        "parameter_system": "geo_feas_like",
    },
}

SPRING_PARAMETER_ALIASES: dict[str, tuple[str, ...]] = {
    "yield_force": ("yield_load", "yield_strength", "yield_resistance", "fy", "Fy", "Py", "p_y", "q_y", "yield"),
    "post_yield_stiffness": ("k2", "secondary_stiffness", "post_stiffness", "plastic_stiffness", "kp"),
    "post_yield_ratio": ("hardening_ratio", "secondary_stiffness_ratio", "alpha_post", "alpha", "k2_over_k1"),
    "degradation": ("strength_loss_per_cycle", "strength_degradation", "degradation_ratio", "beta_degrade", "damage_factor"),
    "pinching": ("pinch_ratio", "pinching_ratio", "pinch_factor", "slip_pinching"),
    "unloading_stiffness_ratio": ("unloading_ratio", "unload_ratio", "unloading_degradation", "eta_unload", "ku_over_k1"),
    "residual_ratio": ("residual_strength_ratio", "residual_ratio", "lower_bound_ratio"),
    "damping": ("damping_coefficient", "viscous_c", "dashpot", "c_damp"),
    "gap": ("clearance", "initial_gap", "opening_gap", "gap_open"),
}


def structural_elements_from_config(cfg: Mapping[str, Any], mesh: Mesh2D) -> list[StructuralElement2D]:
    raw_elements = cfg.get("structural_elements", cfg.get("line_elements", cfg.get("frame_elements", [])))
    section_library = _structural_section_library(cfg)
    spring_models = _spring_hysteresis_model_library(cfg)
    elements: list[StructuralElement2D] = []
    for index, raw in enumerate(_ensure_list(raw_elements), start=1):
        if not isinstance(raw, Mapping):
            raise FEM2DError("each structural element must be a mapping")
        nodes = tuple(str(value) for value in _require_sequence(raw.get("nodes"), f"structural_elements[{index}].nodes"))
        if len(nodes) != 2:
            raise FEM2DError(f"structural element {raw.get('id', index)}: nodes must contain 2 node ids")
        missing = [nid for nid in nodes if nid not in mesh.node_index]
        if missing:
            raise FEM2DError(f"structural element {raw.get('id', index)}: unknown nodes {missing}")
        etype = str(raw.get("type", raw.get("element_type", raw.get("kind", "BAR2")))).upper().replace("-", "_")
        section = raw.get("section", raw.get("stiffness", raw.get("properties", {})))
        if not isinstance(section, Mapping):
            section = {}
        behavior = raw.get("behavior", raw.get("law", ""))
        if isinstance(behavior, Mapping):
            merged = dict(section)
            merged.update({str(k): v for k, v in behavior.items()})
            section = merged
            behavior = str(behavior.get("type", behavior.get("model", "")) or "")
        section = _apply_structural_section_library(section, raw, section_library)
        section = _normalize_spring_parameter_aliases(section)
        section = _apply_spring_hysteresis_model(section, raw, spring_models)
        elements.append(
            StructuralElement2D(
                id=str(raw.get("id", index)),
                type=etype,
                nodes=(nodes[0], nodes[1]),
                material=str(raw.get("material", raw.get("property", "")) or ""),
                section=dict(section),
                behavior=str(behavior or ""),
                active=bool(raw.get("active", True)),
            )
        )
    return elements


def _structural_section_library(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    library = {name: dict(values) for name, values in STRUCTURAL_SECTION_LIBRARY.items()}
    raw_library = cfg.get("structural_section_library", cfg.get("section_library", cfg.get("sections", {})))
    if isinstance(raw_library, Mapping):
        for name, values in raw_library.items():
            if isinstance(values, Mapping):
                entry = {str(key): value for key, value in values.items()}
                entry.setdefault("section_name", str(name))
                library[str(name).upper()] = entry
    return library


def _spring_hysteresis_model_library(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    library = {name: dict(values) for name, values in STRUCTURAL_SPRING_HYSTERESIS_MODELS.items()}
    raw_library = cfg.get("spring_hysteresis_models", cfg.get("structural_spring_models", {}))
    if isinstance(raw_library, Mapping):
        for name, values in raw_library.items():
            if isinstance(values, Mapping):
                library[str(name).upper()] = {str(key): value for key, value in values.items()}
    return library


def _apply_structural_section_library(
    section: Mapping[str, Any],
    raw: Mapping[str, Any],
    library: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result = {str(key): value for key, value in section.items()}
    name = _first_named_value(
        result,
        raw,
        ("section_name", "section_library", "section_profile", "profile", "catalog_section"),
    )
    if not name:
        return result
    defaults = library.get(str(name).upper())
    if defaults is None:
        raise FEM2DError(f"structural section library entry '{name}' is not defined")
    merged = {str(key): value for key, value in defaults.items()}
    merged.update(result)
    merged.setdefault("section_name", str(name))
    return merged


def _apply_spring_hysteresis_model(
    section: Mapping[str, Any],
    raw: Mapping[str, Any],
    library: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    result = {str(key): value for key, value in section.items()}
    name = _first_named_value(
        result,
        raw,
        ("hysteresis_model", "spring_model", "commercial_model", "model_preset", "hysteresis_preset"),
    )
    if not name:
        return result
    defaults = library.get(str(name).upper())
    if defaults is None:
        raise FEM2DError(f"structural spring hysteresis model '{name}' is not defined")
    merged = {str(key): value for key, value in defaults.items()}
    merged.update(result)
    merged["hysteresis_model"] = str(name)
    return merged


def _normalize_spring_parameter_aliases(section: Mapping[str, Any]) -> dict[str, Any]:
    result = {str(key): value for key, value in section.items()}
    external = _spring_external_parameter_mapping(result)
    for canonical, aliases in SPRING_PARAMETER_ALIASES.items():
        if canonical in result:
            continue
        value = _first_numeric_alias(result, external, aliases)
        if value is not None:
            result[canonical] = value
    pos_yield = _first_numeric_alias(result, external, ("yield_force_positive", "fy_positive", "positive_yield", "Py_plus"))
    neg_yield = _first_numeric_alias(result, external, ("yield_force_negative", "fy_negative", "negative_yield", "Py_minus"))
    if "yield_force" not in result:
        candidates = [abs(value) for value in (pos_yield, neg_yield) if value is not None]
        if candidates:
            result["yield_force"] = min(candidates)
    system = _first_text_alias(result, external, ("parameter_system", "commercial_parameter_system", "vendor_parameter_system", "hysteresis_parameter_system"))
    if system:
        result.setdefault("parameter_system", system)
    return result


def _spring_external_parameter_mapping(section: Mapping[str, Any]) -> dict[str, Any]:
    external: dict[str, Any] = {}
    for container_key in ("commercial_parameters", "vendor_parameters", "hysteresis_parameters", "parameters", "model_parameters"):
        raw = section.get(container_key)
        if isinstance(raw, Mapping):
            external.update({str(key): value for key, value in raw.items()})
    return external


def _first_numeric_alias(section: Mapping[str, Any], external: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    lowered_section = {str(key).lower(): value for key, value in section.items()}
    lowered_external = {str(key).lower(): value for key, value in external.items()}
    for name in names:
        for source, lowered in ((section, lowered_section), (external, lowered_external)):
            value = source.get(name, lowered.get(name.lower(), None))
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def _first_text_alias(section: Mapping[str, Any], external: Mapping[str, Any], names: tuple[str, ...]) -> str:
    lowered_section = {str(key).lower(): value for key, value in section.items()}
    lowered_external = {str(key).lower(): value for key, value in external.items()}
    for name in names:
        for source, lowered in ((section, lowered_section), (external, lowered_external)):
            value = source.get(name, lowered.get(name.lower(), None))
            if value not in (None, ""):
                return str(value)
    return ""


def _first_named_value(section: Mapping[str, Any], raw: Mapping[str, Any], names: tuple[str, ...]) -> str:
    lowered_section = {str(key).lower(): value for key, value in section.items()}
    lowered_raw = {str(key).lower(): value for key, value in raw.items()}
    for name in names:
        lname = name.lower()
        value = section.get(name, lowered_section.get(lname, raw.get(name, lowered_raw.get(lname, ""))))
        if value not in (None, ""):
            return str(value)
    return ""


def structural_elements_with_active(elements: list[StructuralElement2D], active_ids: set[str]) -> list[StructuralElement2D]:
    return [replace(element, active=element.id in active_ids) for element in elements]


def structural_rotation_dof_map(mesh: Mesh2D, elements: list[StructuralElement2D] | None, *, axisymmetric: bool = False) -> dict[str, int]:
    if axisymmetric:
        return {}
    nodes: set[str] = set()
    for element in elements or []:
        if not element.active or not _is_frame_kind(element):
            continue
        released = _released_local_indices(element)
        if 2 not in released:
            nodes.add(element.nodes[0])
        if 5 not in released:
            nodes.add(element.nodes[1])
    start = len(mesh.node_ids) * 2
    return {nid: start + i for i, nid in enumerate(nid for nid in mesh.node_ids if nid in nodes)}


def structural_total_dofs(mesh: Mesh2D, elements: list[StructuralElement2D] | None, *, axisymmetric: bool = False) -> int:
    return len(mesh.node_ids) * 2 + len(structural_rotation_dof_map(mesh, elements, axisymmetric=axisymmetric))


def structural_extra_dof_labels(mesh: Mesh2D, elements: list[StructuralElement2D] | None, *, axisymmetric: bool = False) -> dict[int, str]:
    return {dof: f"{nid}:rz" for nid, dof in structural_rotation_dof_map(mesh, elements, axisymmetric=axisymmetric).items()}


def structural_has_nonlinear(elements: list[StructuralElement2D] | None) -> bool:
    return any(element.active and _spring_law(element) != "linear" for element in elements or [])


def update_structural_element_histories(elements: list[StructuralElement2D] | None, rows: list[dict[str, Any]]) -> None:
    by_id = {element.id: element for element in elements or []}
    for row in rows:
        element = by_id.get(str(row.get("element_id", "")))
        if element is None:
            continue
        history = dict(element.section.get("history", {})) if isinstance(element.section.get("history"), Mapping) else {}
        for component in ("axial", "shear"):
            force_key = f"{component}_force" if component == "axial" else "shear_force"
            deformation_key = f"{component}_deformation"
            history[component] = {
                "deformation": float(row.get(deformation_key, 0.0) or 0.0),
                "force": float(row.get(force_key, 0.0) or 0.0),
                "plastic_deformation": float(row.get(f"plastic_deformation_{component}", 0.0) or 0.0),
                "loading_direction": float(row.get(f"loading_direction_{component}", 0.0) or 0.0),
                "reversal_count": float(row.get(f"reversal_count_{component}", 0.0) or 0.0),
                "cycle_count": float(row.get(f"cycle_count_{component}", 0.0) or 0.0),
                "max_abs_deformation": float(row.get(f"max_abs_deformation_{component}", abs(float(row.get(deformation_key, 0.0) or 0.0))) or 0.0),
                "cumulative_energy": float(row.get(f"cumulative_energy_{component}", 0.0) or 0.0),
            }
        element.section["history"] = history


def structural_element_dofs(
    element: StructuralElement2D,
    mesh: Mesh2D,
    rotation_dofs: Mapping[str, int] | None = None,
    *,
    axisymmetric: bool = False,
) -> np.ndarray:
    if not _is_frame_kind(element) or axisymmetric:
        return _element_dofs(element.nodes, mesh.node_index)
    rotation_dofs = rotation_dofs or structural_rotation_dof_map(mesh, [element])
    base = _frame_full_dofs(element, mesh, rotation_dofs)
    retained = [idx for idx in range(6) if idx not in _released_local_indices(element)]
    return np.asarray([base[idx] for idx in retained], dtype=int)


def structural_element_force_tangent(
    element: StructuralElement2D,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    ue: np.ndarray | None = None,
    *,
    axisymmetric: bool = False,
    rotation_dofs: Mapping[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if _is_frame_kind(element) and not axisymmetric:
        return _frame_force_tangent(element, mesh, materials, ue, rotation_dofs=rotation_dofs)
    return _spring_or_bar_force_tangent(element, mesh, materials, ue, axisymmetric=axisymmetric)


def structural_element_equivalent_load(
    element: StructuralElement2D,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    load: Mapping[str, Any],
    *,
    rotation_dofs: Mapping[str, int] | None = None,
    axisymmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not element.active or not _load_targets_structural_element(load, element.id):
        return None
    if axisymmetric or not _is_frame_kind(element):
        return _translational_line_equivalent_load(element, mesh, load, axisymmetric=axisymmetric)
    dofs, transform, _k_local, retained, released, length, scale = _frame_operators(element, mesh, materials, rotation_dofs)
    k_full = _frame_local_stiffness(element, materials, length)
    q_local = _line_load_local_components(element, mesh, load)
    f_full = np.zeros(6, dtype=float)
    f_full[0] = q_local[0] * length / 2.0
    f_full[3] = q_local[0] * length / 2.0
    f_full[1] = q_local[1] * length / 2.0
    f_full[4] = q_local[1] * length / 2.0
    f_full[2] = q_local[1] * length**2 / 12.0
    f_full[5] = -q_local[1] * length**2 / 12.0
    f_eff = _condense_load(k_full, f_full, retained, released)
    return dofs, transform.T @ (f_eff * scale)


def compute_structural_results(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    elements: list[StructuralElement2D] | None,
    u: np.ndarray,
    *,
    axisymmetric: bool = False,
    loads: Any = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rotation_dofs = structural_rotation_dof_map(mesh, elements, axisymmetric=axisymmetric)
    for element in elements or []:
        if not element.active:
            rows.append(_inactive_structural_result(element))
            continue
        if _is_frame_kind(element) and not axisymmetric:
            rows.append(_frame_result_row(element, mesh, materials, u, rotation_dofs, loads))
        else:
            rows.append(_spring_or_bar_result_row(element, mesh, materials, u, axisymmetric=axisymmetric))
    return rows


def _spring_or_bar_force_tangent(
    element: StructuralElement2D,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    ue: np.ndarray | None = None,
    *,
    axisymmetric: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dofs = _element_dofs(element.nodes, mesh.node_index)
    if ue is None:
        ue = np.zeros(4, dtype=float)
    else:
        ue = np.asarray(ue, dtype=float)
        if ue.shape != (4,):
            raise FEM2DError(f"structural element {element.id}: displacement vector must have 4 entries")
    transform, _length, scale = _structural_transform(element, mesh, axisymmetric=axisymmetric)
    u_local = transform @ ue
    k_axial, k_shear = _structural_stiffness_values(element, materials, _length)
    etype = _structural_kind(element)
    if etype in {"BAR2", "TRUSS2", "AXIAL_SPRING2"}:
        k_shear = 0.0
    elif etype in {"SHEAR_SPRING2"}:
        k_axial = 0.0
    axial_deformation = float(u_local[2] - u_local[0])
    shear_deformation = float(u_local[3] - u_local[1])
    axial = _spring_component_response(element, "axial", axial_deformation, k_axial)
    shear = _spring_component_response(element, "shear", shear_deformation, k_shear)
    k_local = np.zeros((4, 4), dtype=float)
    b_axial = np.array([-1.0, 0.0, 1.0, 0.0], dtype=float)
    b_shear = np.array([0.0, -1.0, 0.0, 1.0], dtype=float)
    if axial["tangent"] > 0.0:
        k_local += float(axial["tangent"]) * np.outer(b_axial, b_axial)
    if shear["tangent"] > 0.0:
        k_local += float(shear["tangent"]) * np.outer(b_shear, b_shear)
    f_local = float(axial["force"]) * b_axial + float(shear["force"]) * b_shear
    return dofs, transform.T @ (f_local * scale), _symmetrize(transform.T @ (k_local * scale) @ transform)


def _frame_force_tangent(
    element: StructuralElement2D,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    ue: np.ndarray | None,
    *,
    rotation_dofs: Mapping[str, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dofs, transform, k_local, _retained, _released, _length, scale = _frame_operators(element, mesh, materials, rotation_dofs)
    if ue is None:
        ue = np.zeros(len(dofs), dtype=float)
    else:
        ue = np.asarray(ue, dtype=float)
        if ue.shape != (len(dofs),):
            raise FEM2DError(f"structural element {element.id}: displacement vector must have {len(dofs)} entries")
    u_local = transform @ ue
    f_local = k_local @ u_local
    return dofs, transform.T @ (f_local * scale), _symmetrize(transform.T @ (k_local * scale) @ transform)


def _frame_operators(
    element: StructuralElement2D,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    rotation_dofs: Mapping[str, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], list[int], float, float]:
    rotation_dofs = rotation_dofs or structural_rotation_dof_map(mesh, [element])
    full_dofs = _frame_full_dofs(element, mesh, rotation_dofs)
    retained = [idx for idx in range(6) if idx not in _released_local_indices(element)]
    released = [idx for idx in range(6) if idx not in retained]
    dofs = np.asarray([full_dofs[idx] for idx in retained], dtype=int)
    if np.any(dofs < 0):
        raise FEM2DError(f"structural element {element.id}: missing rotational dof for an unreleased beam end")
    transform_full, length, scale = _frame_transform(element, mesh)
    transform = transform_full[np.ix_(retained, retained)]
    k_full = _frame_local_stiffness(element, materials, length)
    k_local = _condense_stiffness(k_full, retained, released)
    return dofs, transform, k_local, retained, released, length, scale


def _frame_full_dofs(element: StructuralElement2D, mesh: Mesh2D, rotation_dofs: Mapping[str, int]) -> list[int]:
    node_index = mesh.node_index
    ri = rotation_dofs.get(element.nodes[0], -1)
    rj = rotation_dofs.get(element.nodes[1], -1)
    i = node_index[element.nodes[0]]
    j = node_index[element.nodes[1]]
    return [2 * i, 2 * i + 1, ri, 2 * j, 2 * j + 1, rj]


def _frame_transform(element: StructuralElement2D, mesh: Mesh2D) -> tuple[np.ndarray, float, float]:
    p0 = mesh.coords[mesh.node_index[element.nodes[0]]]
    p1 = mesh.coords[mesh.node_index[element.nodes[1]]]
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise FEM2DError(f"structural element {element.id}: length must be positive")
    c = float(axis[0] / length)
    s = float(axis[1] / length)
    transform = np.zeros((6, 6), dtype=float)
    transform[0, 0] = c
    transform[0, 1] = s
    transform[1, 0] = -s
    transform[1, 1] = c
    transform[2, 2] = 1.0
    transform[3, 3] = c
    transform[3, 4] = s
    transform[4, 3] = -s
    transform[4, 4] = c
    transform[5, 5] = 1.0
    return transform, length, 1.0


def _frame_local_stiffness(
    element: StructuralElement2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    length: float,
) -> np.ndarray:
    section = element.section
    material = materials.get(element.material) if element.material else None
    e_mod = _section_float(section, ("E", "young", "young_modulus"), material.E if material else 0.0)
    nu = material.nu if material else _section_float(section, ("nu", "poisson"), 0.3)
    g_mod = _section_float(section, ("G", "shear_modulus"), e_mod / max(2.0 * (1.0 + nu), np.finfo(float).eps) if e_mod > 0.0 else 0.0)
    area = _section_float(section, ("A", "area"), 1.0)
    inertia = _section_float(section, ("I", "Iz", "inertia"), 0.0)
    shear_area = _section_float(section, ("As", "shear_area"), area)
    shear_correction = _section_float(section, ("kappa", "shear_correction"), 5.0 / 6.0)
    ea = _section_float(section, ("EA", "axial_rigidity"), e_mod * area)
    ei = _section_float(section, ("EI", "bending_rigidity"), e_mod * inertia)
    if ea <= 0.0 and "kx" in {str(k).lower() for k in section}:
        ea = _section_float(section, ("kx", "ka", "axial_stiffness"), 0.0) * length
    if ea <= 0.0 or ei <= 0.0:
        raise FEM2DError(f"structural element {element.id}: frame/beam elements require positive EA and EI")
    k = np.zeros((6, 6), dtype=float)
    axial = ea / length
    k[0, 0] = axial
    k[0, 3] = -axial
    k[3, 0] = -axial
    k[3, 3] = axial
    theory = str(element.section.get("theory", element.section.get("beam_theory", element.type))).lower()
    use_timoshenko = "timoshenko" in theory or _structural_kind(element) in {"TIMOSHENKO_BEAM2", "TIMOSHENKO_FRAME2"}
    phi = 0.0
    if use_timoshenko and g_mod > 0.0 and shear_area > 0.0 and shear_correction > 0.0:
        phi = 12.0 * ei / max(shear_correction * g_mod * shear_area * length**2, np.finfo(float).eps)
    factor = ei / (length**3 * (1.0 + phi))
    bending = np.array(
        [
            [12.0, 6.0 * length, -12.0, 6.0 * length],
            [6.0 * length, (4.0 + phi) * length**2, -6.0 * length, (2.0 - phi) * length**2],
            [-12.0, -6.0 * length, 12.0, -6.0 * length],
            [6.0 * length, (2.0 - phi) * length**2, -6.0 * length, (4.0 + phi) * length**2],
        ],
        dtype=float,
    ) * factor
    idx = [1, 2, 4, 5]
    for a, ia in enumerate(idx):
        for b, ib in enumerate(idx):
            k[ia, ib] = bending[a, b]
    k = _apply_frame_end_rotational_springs(k, element)
    return _symmetrize(k)


def _apply_frame_end_rotational_springs(k_beam: np.ndarray, element: StructuralElement2D) -> np.ndarray:
    springs = {
        2: _end_rotational_spring(element, "i"),
        5: _end_rotational_spring(element, "j"),
    }
    explicit_releases = set(_explicit_released_local_indices(element))
    active = {idx: value for idx, value in springs.items() if value is not None and value > 0.0 and idx not in explicit_releases}
    if not active:
        return k_beam
    size = 6 + len(active)
    expanded = np.zeros((size, size), dtype=float)
    beam_map = list(range(6))
    internal_by_end: dict[int, int] = {}
    for offset, idx in enumerate(sorted(active), start=6):
        beam_map[idx] = offset
        internal_by_end[idx] = offset
    for a in range(6):
        ia = beam_map[a]
        for b in range(6):
            expanded[ia, beam_map[b]] += k_beam[a, b]
    for idx, value in active.items():
        internal = internal_by_end[idx]
        ktheta = float(value)
        expanded[idx, idx] += ktheta
        expanded[internal, internal] += ktheta
        expanded[idx, internal] -= ktheta
        expanded[internal, idx] -= ktheta
    external = list(range(6))
    internal = [internal_by_end[idx] for idx in sorted(active)]
    k_ee = expanded[np.ix_(external, external)]
    k_ei = expanded[np.ix_(external, internal)]
    k_ie = expanded[np.ix_(internal, external)]
    k_ii = expanded[np.ix_(internal, internal)]
    return _symmetrize(k_ee - k_ei @ np.linalg.solve(k_ii, k_ie))


def _condense_stiffness(k_full: np.ndarray, retained: list[int], released: list[int]) -> np.ndarray:
    if not released:
        return k_full[np.ix_(retained, retained)].copy()
    k_rr = k_full[np.ix_(retained, retained)]
    k_rc = k_full[np.ix_(retained, released)]
    k_cr = k_full[np.ix_(released, retained)]
    k_cc = k_full[np.ix_(released, released)]
    return _symmetrize(k_rr - k_rc @ np.linalg.solve(k_cc, k_cr))


def _condense_load(k_full: np.ndarray, f_full: np.ndarray, retained: list[int], released: list[int]) -> np.ndarray:
    f_r = f_full[retained]
    if not released:
        return f_r
    k_rc = k_full[np.ix_(retained, released)]
    k_cc = k_full[np.ix_(released, released)]
    f_c = f_full[released]
    return f_r - k_rc @ np.linalg.solve(k_cc, f_c)


def _translational_line_equivalent_load(
    element: StructuralElement2D,
    mesh: Mesh2D,
    load: Mapping[str, Any],
    *,
    axisymmetric: bool,
) -> tuple[np.ndarray, np.ndarray] | None:
    dofs = _element_dofs(element.nodes, mesh.node_index)
    transform, length, scale = _structural_transform(element, mesh, axisymmetric=axisymmetric)
    q_local = _line_load_local_components(element, mesh, load)
    f_local = np.array([q_local[0] * length / 2.0, q_local[1] * length / 2.0, q_local[0] * length / 2.0, q_local[1] * length / 2.0], dtype=float)
    return dofs, transform.T @ (f_local * scale)


def _line_load_local_components(element: StructuralElement2D, mesh: Mesh2D, load: Mapping[str, Any]) -> np.ndarray:
    qx_local = _mapping_float(load, ("qx_local", "qa", "q_axial", "axial"), None)
    qy_local = _mapping_float(load, ("qy_local", "qn", "q_transverse", "q_shear", "q"), None)
    if qx_local is not None or qy_local is not None:
        return np.array([float(qx_local or 0.0), float(qy_local or 0.0)], dtype=float)
    qx = float(load.get("qx", load.get("tx", 0.0)))
    qy = float(load.get("qy", load.get("ty", 0.0)))
    p0 = mesh.coords[mesh.node_index[element.nodes[0]]]
    p1 = mesh.coords[mesh.node_index[element.nodes[1]]]
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise FEM2DError(f"structural element {element.id}: length must be positive")
    c = float(axis[0] / length)
    s = float(axis[1] / length)
    return np.array([c * qx + s * qy, -s * qx + c * qy], dtype=float)


def _load_targets_structural_element(load: Mapping[str, Any], element_id: str) -> bool:
    target = load.get("structural_element", load.get("line_element", load.get("element")))
    targets = load.get("structural_elements", load.get("line_elements", load.get("elements")))
    ltype = str(load.get("type", "")).lower().strip()
    if target is None and targets is None and ltype not in {"beam_uniform", "frame_uniform", "structural_line", "line_load"}:
        return False
    if isinstance(target, str):
        return target == element_id or target.lower() == "all"
    if targets is not None:
        if isinstance(targets, str):
            return targets == element_id or targets.lower() == "all"
        return element_id in {str(value) for value in _ensure_list(targets)}
    return bool(load.get("all", False))


def _spring_or_bar_result_row(
    element: StructuralElement2D,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    axisymmetric: bool = False,
) -> dict[str, Any]:
    dofs = _element_dofs(element.nodes, mesh.node_index)
    transform, length, scale = _structural_transform(element, mesh, axisymmetric=axisymmetric)
    u_local = transform @ u[dofs]
    k_axial, k_shear = _structural_stiffness_values(element, materials, length)
    etype = _structural_kind(element)
    if etype in {"BAR2", "TRUSS2", "AXIAL_SPRING2"}:
        k_shear = 0.0
    elif etype in {"SHEAR_SPRING2"}:
        k_axial = 0.0
    axial_deformation = float(u_local[2] - u_local[0])
    shear_deformation = float(u_local[3] - u_local[1])
    axial = _spring_component_response(element, "axial", axial_deformation, k_axial)
    shear = _spring_component_response(element, "shear", shear_deformation, k_shear)
    axial_force = float(axial["force"]) * scale
    shear_force = float(shear["force"]) * scale
    row = {
        "element_id": element.id,
        "type": element.type,
        "behavior": element.behavior,
        "node_i": element.nodes[0],
        "node_j": element.nodes[1],
        "active": 1.0,
        "length": length,
        "axisymmetric_scale": scale,
        "axial_deformation": axial_deformation,
        "shear_deformation": shear_deformation,
        "axial_strain": axial_deformation / max(length, np.finfo(float).eps),
        "axial_force": axial_force,
        "shear_force": shear_force,
        "spring_reaction": math.hypot(axial_force, shear_force),
        "end_moment_i": -0.5 * shear_force * length,
        "end_moment_j": 0.5 * shear_force * length,
        "kx": float(axial["tangent"]),
        "ky": float(shear["tangent"]),
        "spring_state_axial": axial["state"],
        "spring_state_shear": shear["state"],
        "plastic_deformation_axial": float(axial["plastic_deformation"]),
        "plastic_deformation_shear": float(shear["plastic_deformation"]),
        "loading_direction_axial": float(axial.get("loading_direction", 0.0)),
        "loading_direction_shear": float(shear.get("loading_direction", 0.0)),
        "reversal_count_axial": float(axial.get("reversal_count", 0.0)),
        "reversal_count_shear": float(shear.get("reversal_count", 0.0)),
        "cycle_count_axial": float(axial.get("cycle_count", 0.0)),
        "cycle_count_shear": float(shear.get("cycle_count", 0.0)),
        "cumulative_energy_axial": float(axial.get("cumulative_energy", 0.0)),
        "cumulative_energy_shear": float(shear.get("cumulative_energy", 0.0)),
        "max_abs_deformation_axial": float(axial.get("max_abs_deformation", abs(axial_deformation))),
        "max_abs_deformation_shear": float(shear.get("max_abs_deformation", abs(shear_deformation))),
        "force_degradation_axial": float(axial.get("degradation_factor", 1.0)),
        "force_degradation_shear": float(shear.get("degradation_factor", 1.0)),
        "pinching_factor_axial": float(axial.get("pinching_factor", 1.0)),
        "pinching_factor_shear": float(shear.get("pinching_factor", 1.0)),
        "damping_force_axial": float(axial.get("damping_force", 0.0)),
        "damping_force_shear": float(shear.get("damping_force", 0.0)),
        "spring_law": _spring_law(element),
        "hysteresis_model": str(element.section.get("hysteresis_model", element.section.get("spring_model", "")) or ""),
        "spring_parameter_system": str(element.section.get("parameter_system", element.section.get("commercial_parameter_system", "")) or ""),
        "unloading_stiffness_ratio": _section_float(element.section, ("unloading_stiffness_ratio", "unloading_ratio"), 1.0),
        "hysteretic_damping_ratio": _section_float(element.section, ("damping_ratio", "hysteretic_damping", "equivalent_damping"), 0.0),
        "state": "active",
    }
    if axisymmetric:
        row["geometry"] = "axisymmetric"
    return row


def _frame_result_row(
    element: StructuralElement2D,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    rotation_dofs: Mapping[str, int],
    loads: Any,
) -> dict[str, Any]:
    dofs, transform, k_local, retained, released, length, scale = _frame_operators(element, mesh, materials, rotation_dofs)
    ue = u[dofs]
    u_local_retained = transform @ ue
    f_local_retained = k_local @ u_local_retained
    f_full_load = _frame_equivalent_load_full(element, mesh, loads, length)
    full_u = np.zeros(6, dtype=float)
    full_u[retained] = u_local_retained
    if released:
        k_full = _frame_local_stiffness(element, materials, length)
        k_cc = k_full[np.ix_(released, released)]
        k_cr = k_full[np.ix_(released, retained)]
        full_u[released] = np.linalg.solve(k_cc, f_full_load[released] - k_cr @ u_local_retained)
    full_force = _frame_local_stiffness(element, materials, length) @ full_u - f_full_load
    released_names = ",".join("i" if idx == 2 else "j" for idx in released)
    section_forces = _frame_section_force_samples(element, mesh, full_force, loads, length)
    return {
        "element_id": element.id,
        "type": element.type,
        "behavior": element.behavior,
        "node_i": element.nodes[0],
        "node_j": element.nodes[1],
        "active": 1.0,
        "length": length,
        "axisymmetric_scale": scale,
        "axial_deformation": float(full_u[3] - full_u[0]),
        "shear_deformation": float(full_u[4] - full_u[1]),
        "rotation_i": float(full_u[2]),
        "rotation_j": float(full_u[5]),
        "axial_strain": float((full_u[3] - full_u[0]) / max(length, np.finfo(float).eps)),
        "axial_force": float(full_force[3] * scale),
        "shear_force": float(full_force[4] * scale),
        "shear_force_i": float(full_force[1] * scale),
        "shear_force_j": float(full_force[4] * scale),
        "spring_reaction": float(np.linalg.norm(f_local_retained) * scale),
        "end_moment_i": float(full_force[2] * scale),
        "end_moment_j": float(full_force[5] * scale),
        "kx": float(k_local[0, 0] if k_local.size else 0.0),
        "ky": float(np.max(np.abs(k_local)) if k_local.size else 0.0),
        "beam_theory": "timoshenko" if "timoshenko" in str(element.section.get("theory", element.type)).lower() or _structural_kind(element) in {"TIMOSHENKO_BEAM2", "TIMOSHENKO_FRAME2"} else "euler",
        "release_i": int(2 in released),
        "release_j": int(5 in released),
        "released_ends": released_names,
        "connection_i": _frame_connection_label(element, "i"),
        "connection_j": _frame_connection_label(element, "j"),
        "rotational_spring_i": float(_end_rotational_spring(element, "i") or 0.0),
        "rotational_spring_j": float(_end_rotational_spring(element, "j") or 0.0),
        "section_name": str(element.section.get("section_name", element.section.get("profile", "")) or ""),
        "section_type": str(element.section.get("section_type", "")),
        "section_area": _section_float(element.section, ("A", "area"), 0.0),
        "section_inertia": _section_float(element.section, ("I", "Iz", "inertia"), 0.0),
        "section_forces": section_forces,
        "state": "active",
    }


def _frame_equivalent_load_full(element: StructuralElement2D, mesh: Mesh2D, loads: Any, length: float) -> np.ndarray:
    f_full = np.zeros(6, dtype=float)
    for load in _ensure_list(loads):
        if not isinstance(load, Mapping) or not _load_targets_structural_element(load, element.id):
            continue
        q_local = _line_load_local_components(element, mesh, load)
        f_full[0] += q_local[0] * length / 2.0
        f_full[3] += q_local[0] * length / 2.0
        f_full[1] += q_local[1] * length / 2.0
        f_full[4] += q_local[1] * length / 2.0
        f_full[2] += q_local[1] * length**2 / 12.0
        f_full[5] += -q_local[1] * length**2 / 12.0
    return f_full


def _frame_section_force_samples(
    element: StructuralElement2D,
    mesh: Mesh2D,
    full_force: np.ndarray,
    loads: Any,
    length: float,
) -> list[dict[str, float]]:
    q_local = np.zeros(2, dtype=float)
    for load in _ensure_list(loads):
        if isinstance(load, Mapping) and _load_targets_structural_element(load, element.id):
            q_local += _line_load_local_components(element, mesh, load)
    samples: list[dict[str, float]] = []
    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = ratio * length
        axial = full_force[0] + q_local[0] * x
        shear = full_force[1] + q_local[1] * x
        moment = full_force[2] + full_force[1] * x + 0.5 * q_local[1] * x**2
        samples.append({"x": float(x), "ratio": float(ratio), "axial_force": float(axial), "shear_force": float(shear), "bending_moment": float(moment)})
    return samples


def _inactive_structural_result(element: StructuralElement2D) -> dict[str, Any]:
    return {
        "element_id": element.id,
        "type": element.type,
        "behavior": element.behavior,
        "node_i": element.nodes[0],
        "node_j": element.nodes[1],
        "active": 0.0,
        "length": 0.0,
        "axisymmetric_scale": 0.0,
        "axial_deformation": 0.0,
        "shear_deformation": 0.0,
        "rotation_i": 0.0,
        "rotation_j": 0.0,
        "axial_strain": 0.0,
        "axial_force": 0.0,
        "shear_force": 0.0,
        "spring_reaction": 0.0,
        "end_moment_i": 0.0,
        "end_moment_j": 0.0,
        "kx": 0.0,
        "ky": 0.0,
        "state": "inactive",
    }


def _spring_component_response(element: StructuralElement2D, component: str, deformation: float, stiffness: float) -> dict[str, Any]:
    if stiffness <= 0.0:
        return {"force": 0.0, "tangent": 0.0, "state": "inactive", "plastic_deformation": 0.0}
    law = _spring_law(element)
    history_all = element.section.get("history", {})
    history = dict(history_all.get(component, {})) if isinstance(history_all, Mapping) and isinstance(history_all.get(component, {}), Mapping) else {}
    gap = _component_float(element.section, component, ("gap",), 0.0)
    compression_only = bool(element.section.get("compression_only", False))
    tension_only = bool(element.section.get("tension_only", False))
    if gap > 0.0:
        effective = math.copysign(max(abs(deformation) - gap, 0.0), deformation)
        if effective == 0.0:
            return {"force": 0.0, "tangent": 0.0, "state": "open_gap", "plastic_deformation": 0.0}
        deformation = effective
    if compression_only and deformation > 0.0:
        return {"force": 0.0, "tangent": 0.0, "state": "open_gap", "plastic_deformation": 0.0}
    if tension_only and deformation < 0.0:
        return {"force": 0.0, "tangent": 0.0, "state": "open_gap", "plastic_deformation": 0.0}
    if law in {"bilinear", "plastic", "elastoplastic", "hysteretic", "degrading", "pinching"}:
        yield_force = _component_float(element.section, component, ("yield_force", "Fy", "fy"), math.inf)
        post_k = _component_float(element.section, component, ("post_yield_stiffness", "k_post", "kp"), _section_float(element.section, ("alpha", "post_yield_ratio"), 0.0) * stiffness)
        unloading_ratio = max(_component_float(element.section, component, ("unloading_stiffness_ratio", "unloading_ratio"), _section_float(element.section, ("unloading_stiffness_ratio", "unloading_ratio"), 1.0)), 0.0)
        degradation_rate = max(_component_float(element.section, component, ("degradation", "strength_degradation"), _section_float(element.section, ("degradation", "strength_degradation"), 0.0)), 0.0)
        residual_ratio = min(max(_component_float(element.section, component, ("residual_ratio",), _section_float(element.section, ("residual_ratio",), 0.2)), 0.0), 1.0)
        last_deformation = float(history.get("deformation", 0.0) or 0.0)
        last_force = float(history.get("force", 0.0) or 0.0)
        last_direction = float(history.get("loading_direction", 0.0) or 0.0)
        delta = deformation - last_deformation
        direction = math.copysign(1.0, delta) if abs(delta) > 1.0e-14 else last_direction
        reversal = bool(direction and last_direction and direction != last_direction)
        reversal_count = float(history.get("reversal_count", 0.0) or 0.0) + (1.0 if reversal else 0.0)
        cycle_count = reversal_count / 2.0
        degradation_factor = max(1.0 - degradation_rate * cycle_count, residual_ratio)
        yield_force_eff = yield_force * degradation_factor if math.isfinite(yield_force) else yield_force
        yield_deformation = yield_force_eff / stiffness if math.isfinite(yield_force_eff) else math.inf
        abs_def = abs(deformation)
        sign = math.copysign(1.0, deformation) if deformation != 0.0 else 1.0
        max_abs_deformation = max(float(history.get("max_abs_deformation", 0.0) or 0.0), abs_def)
        pinching = min(max(_component_float(element.section, component, ("pinching", "pinching_ratio"), _section_float(element.section, ("pinching", "pinching_ratio"), 0.0)), 0.0), 0.95)
        pinching_factor = 1.0
        if law in {"hysteretic", "degrading", "pinching"} and reversal_count > 0.0 and max_abs_deformation > 0.0:
            pinching_factor = max(1.0 - pinching * (1.0 - abs_def / max_abs_deformation), 1.0 - pinching)
        damping = max(_component_float(element.section, component, ("damping", "damping_coefficient", "c"), _section_float(element.section, ("damping", "damping_coefficient", "c"), 0.0)), 0.0)
        dt = max(_section_float(element.section, ("dt", "time_step"), 1.0), np.finfo(float).eps)
        damping_force = damping * delta / dt
        damping_tangent = damping / dt if law in {"hysteretic", "degrading", "pinching"} else 0.0
        elastic_stiffness = stiffness
        if law in {"hysteretic", "degrading", "pinching"} and reversal:
            elastic_stiffness = stiffness * unloading_ratio
        if abs_def <= yield_deformation:
            force = elastic_stiffness * deformation * pinching_factor + damping_force
            tangent = elastic_stiffness * pinching_factor + damping_tangent
            state = "hysteretic_reversal" if reversal else "elastic"
            plastic = 0.0
        else:
            force = sign * (yield_force_eff + post_k * (abs_def - yield_deformation)) * pinching_factor + damping_force
            tangent = max(post_k, 0.0) * pinching_factor + damping_tangent
            state = "hysteretic_yielded" if law in {"hysteretic", "degrading", "pinching"} else "yielded"
            elastic_def = (force - damping_force) / stiffness if stiffness > 0.0 else 0.0
            plastic = deformation - elastic_def
        energy = float(history.get("cumulative_energy", 0.0) or 0.0) + abs(0.5 * (force + last_force) * delta)
        return {
            "force": force,
            "tangent": max(tangent, 0.0),
            "state": state,
            "plastic_deformation": plastic,
            "loading_direction": direction,
            "reversal_count": reversal_count,
            "cycle_count": cycle_count,
            "cumulative_energy": energy,
            "max_abs_deformation": max_abs_deformation,
            "degradation_factor": degradation_factor,
            "pinching_factor": pinching_factor,
            "damping_force": damping_force,
        }
    return {
        "force": stiffness * deformation,
        "tangent": stiffness,
        "state": "closed_gap" if gap > 0.0 else "linear",
        "plastic_deformation": 0.0,
        "loading_direction": math.copysign(1.0, deformation) if deformation else 0.0,
        "reversal_count": float(history.get("reversal_count", 0.0) or 0.0),
        "cycle_count": float(history.get("cycle_count", 0.0) or 0.0),
        "cumulative_energy": float(history.get("cumulative_energy", 0.0) or 0.0),
        "max_abs_deformation": max(float(history.get("max_abs_deformation", 0.0) or 0.0), abs(deformation)),
        "degradation_factor": 1.0,
        "pinching_factor": 1.0,
        "damping_force": 0.0,
    }


def _spring_law(element: StructuralElement2D) -> str:
    text = str(element.section.get("law", element.section.get("model", element.behavior or "linear"))).lower().replace("-", "_")
    if any(key in text for key in ("bilinear", "plastic", "hysteretic", "degrading", "pinching", "gap")):
        if "bilinear" in text:
            return "bilinear"
        if "degrading" in text:
            return "degrading"
        if "pinching" in text:
            return "pinching"
        if "plastic" in text:
            return "plastic"
        if "hysteretic" in text:
            return "hysteretic"
        return "gap"
    if "yield_force" in {str(key).lower() for key in element.section}:
        return "bilinear"
    if "gap" in {str(key).lower() for key in element.section}:
        return "gap"
    return "linear"


def _structural_transform(element: StructuralElement2D, mesh: Mesh2D, *, axisymmetric: bool) -> tuple[np.ndarray, float, float]:
    p0 = mesh.coords[mesh.node_index[element.nodes[0]]]
    p1 = mesh.coords[mesh.node_index[element.nodes[1]]]
    axis = p1 - p0
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise FEM2DError(f"structural element {element.id}: length must be positive")
    tangent = axis / length
    normal = np.array([-tangent[1], tangent[0]], dtype=float)
    transform = np.zeros((4, 4), dtype=float)
    transform[0, 0:2] = tangent
    transform[1, 0:2] = normal
    transform[2, 2:4] = tangent
    transform[3, 2:4] = normal
    if not axisymmetric:
        return transform, length, 1.0
    radius = max(float(0.5 * (p0[0] + p1[0])), np.finfo(float).eps)
    return transform, length, 2.0 * math.pi * radius


def _structural_stiffness_values(
    element: StructuralElement2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    length: float,
) -> tuple[float, float]:
    section = element.section
    material = materials.get(element.material) if element.material else None
    e_mod = _section_float(section, ("E", "young", "young_modulus"), material.E if material else 0.0)
    nu = material.nu if material else _section_float(section, ("nu", "poisson"), 0.3)
    g_mod = _section_float(section, ("G", "shear_modulus"), e_mod / max(2.0 * (1.0 + nu), np.finfo(float).eps) if e_mod > 0.0 else 0.0)
    area = _section_float(section, ("A", "area"), 1.0)
    inertia = _section_float(section, ("I", "Iz", "inertia"), 0.0)
    shear_area = _section_float(section, ("As", "shear_area"), area)
    k_axial = _section_float(section, ("kx", "ka", "EA_over_L", "axial_stiffness"), e_mod * area / length if e_mod > 0.0 else 0.0)
    beam_shear = 12.0 * e_mod * inertia / max(length**3, np.finfo(float).eps) if e_mod > 0.0 and inertia > 0.0 else 0.0
    shear_default = max(g_mod * shear_area / length if g_mod > 0.0 else 0.0, beam_shear)
    k_shear = _section_float(section, ("ky", "ks", "shear_stiffness", "transverse_stiffness"), shear_default)
    return max(float(k_axial), 0.0), max(float(k_shear), 0.0)


def _released_local_indices(element: StructuralElement2D) -> list[int]:
    released = _explicit_released_local_indices(element)
    spring_i = _end_rotational_spring(element, "i")
    spring_j = _end_rotational_spring(element, "j")
    if spring_i is not None and spring_i <= 0.0:
        released.append(2)
    if spring_j is not None and spring_j <= 0.0:
        released.append(5)
    return sorted(set(released))


def _explicit_released_local_indices(element: StructuralElement2D) -> list[int]:
    section = {str(key).lower(): value for key, value in element.section.items()}
    behavior = element.behavior.lower().replace("-", "_")
    released: list[int] = []
    if bool(section.get("release_i", section.get("moment_release_i", section.get("hinge_i", False)))) or "release_i" in behavior or "hinge_i" in behavior:
        released.append(2)
    if bool(section.get("release_j", section.get("moment_release_j", section.get("hinge_j", False)))) or "release_j" in behavior or "hinge_j" in behavior:
        released.append(5)
    if str(section.get("end_release", "")).lower() in {"i", "both"}:
        released.append(2)
    if str(section.get("end_release", "")).lower() in {"j", "both"}:
        released.append(5)
    return sorted(set(released))


def _end_rotational_spring(element: StructuralElement2D, end: str) -> float | None:
    suffix = "i" if end.lower().startswith("i") else "j"
    names = (
        f"rotational_spring_{suffix}",
        f"rotation_spring_{suffix}",
        f"k_theta_{suffix}",
        f"ktheta_{suffix}",
        f"end_spring_{suffix}",
        f"semi_rigid_{suffix}",
        f"semi_rigid_stiffness_{suffix}",
        f"connection_stiffness_{suffix}",
        f"joint_rotational_stiffness_{suffix}",
    )
    return _section_optional_float(element.section, names)


def _frame_connection_label(element: StructuralElement2D, end: str) -> str:
    idx = 2 if end == "i" else 5
    spring = _end_rotational_spring(element, end)
    if idx in _explicit_released_local_indices(element):
        return "released"
    if spring is not None and spring <= 0.0:
        return "released"
    if spring is not None:
        return "semi_rigid"
    return "rigid"


def _component_float(section: Mapping[str, Any], component: str, names: tuple[str, ...], default: float) -> float:
    prefixes = (component, component[0])
    keys: list[str] = []
    for name in names:
        keys.append(name)
        for prefix in prefixes:
            keys.extend([f"{prefix}_{name}", f"{name}_{prefix}"])
    return _section_float(section, tuple(keys), default)


def _section_float(section: Mapping[str, Any], names: tuple[str, ...], default: float | None) -> float:
    lowered = {str(key).lower(): value for key, value in section.items()}
    for name in names:
        if name in section:
            return float(section[name])
        lname = name.lower()
        if lname in lowered:
            return float(lowered[lname])
    if default is None:
        raise KeyError(names[0])
    return float(default)


def _section_optional_float(section: Mapping[str, Any], names: tuple[str, ...]) -> float | None:
    lowered = {str(key).lower(): value for key, value in section.items()}
    for name in names:
        if name in section:
            return float(section[name])
        lname = name.lower()
        if lname in lowered:
            return float(lowered[lname])
    return None


def _mapping_float(mapping: Mapping[str, Any], names: tuple[str, ...], default: float | None) -> float | None:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        if name in mapping:
            return float(mapping[name])
        lname = name.lower()
        if lname in lowered:
            return float(lowered[lname])
    return default


def _structural_kind(element: StructuralElement2D) -> str:
    text = (element.type or element.behavior or "BAR2").upper().replace("-", "_")
    aliases = {
        "BEAM": "BEAM2",
        "EULER_BEAM": "EULER_BEAM2",
        "TIMOSHENKO_BEAM": "TIMOSHENKO_BEAM2",
        "FRAME": "FRAME2",
        "BAR": "BAR2",
        "TRUSS": "TRUSS2",
        "SPRING": "SPRING2",
        "AXIAL_SPRING": "AXIAL_SPRING2",
        "SHEAR_SPRING": "SHEAR_SPRING2",
    }
    return aliases.get(text, text)


def _is_frame_kind(element: StructuralElement2D) -> bool:
    return _structural_kind(element) in {"BEAM2", "FRAME2", "EULER_BEAM2", "EULER_FRAME2", "TIMOSHENKO_BEAM2", "TIMOSHENKO_FRAME2"}


__all__ = [
    "STRUCTURAL_SECTION_LIBRARY",
    "STRUCTURAL_SPRING_HYSTERESIS_MODELS",
    "SPRING_PARAMETER_ALIASES",
    "structural_elements_from_config",
    "structural_elements_with_active",
    "structural_rotation_dof_map",
    "structural_total_dofs",
    "structural_extra_dof_labels",
    "structural_has_nonlinear",
    "update_structural_element_histories",
    "structural_element_dofs",
    "structural_element_force_tangent",
    "structural_element_equivalent_load",
    "compute_structural_results",
]
