"""Fast-path selection and state-array helpers for element post processing."""

from __future__ import annotations

from typing import Any, Mapping
import math

import numpy as np

from .fem2d_materials import (
    _ADV_PARAM_RU_INITIAL,
    _ADV_STATE_FIELDS,
    _advanced_state_array_from_vars,
    _advanced_strength_model_name,
    _is_advanced_material,
    _plastic_state_key,
    _uses_plastic_strength_model,
)
from .fem2d_types import ElasticPlaneStrainMaterial, Element2D, PlasticState2D, normalize_integration


ELEMENT_FAST_PATH_FUNCTIONS = (
    "element_fast_path_contract",
    "_quad4_elastic_post_fast_path",
    "_quad4_elastic_bbar_post_fast_path",
    "_quad8_elastic_post_fast_path",
    "_quad8_elastic_bbar_post_fast_path",
    "_quad8_elastic_tension_post_supported",
    "_quad8_elastic_tension_post_fast_path",
    "_quad8_elastic_tension_bbar_post_fast_path",
    "_quad4_advanced_elastic_post_supported",
    "_quad4_advanced_elastic_post_fast_path",
    "_quad4_advanced_elastic_bbar_post_fast_path",
    "_quad8_advanced_elastic_post_fast_path",
    "_quad8_advanced_elastic_bbar_post_fast_path",
    "_quad4_advanced_elastic_tension_post_supported",
    "_quad4_advanced_elastic_tension_post_fast_path",
    "_quad4_advanced_elastic_tension_bbar_post_fast_path",
    "_quad8_advanced_elastic_tension_post_fast_path",
    "_quad8_advanced_elastic_tension_bbar_post_fast_path",
    "_quad4_advanced_strength_j2dp_post_supported",
    "_quad4_advanced_strength_mc_post_supported",
    "_quad4_advanced_strength_j2dp_post_fast_path",
    "_quad4_advanced_strength_j2dp_bbar_post_fast_path",
    "_quad8_advanced_strength_mc_post_fast_path",
    "_quad8_advanced_strength_mc_bbar_post_fast_path",
    "_quad8_advanced_strength_j2dp_post_fast_path",
    "_quad8_advanced_strength_j2dp_bbar_post_fast_path",
    "_quad4_j2dp_post_fast_path",
    "_quad4_j2dp_bbar_post_fast_path",
    "_quad8_j2dp_post_fast_path",
    "_quad8_j2dp_bbar_post_fast_path",
    "_quad8_mc_post_fast_path",
    "_quad8_mc_bbar_post_fast_path",
    "_quad4_mc_post_fast_path",
    "_quad4_mc_bbar_post_fast_path",
    "_quad4_post_state_arrays",
    "_quad8_post_state_arrays",
    "_quad4_advanced_state_arrays",
    "_quad8_advanced_state_arrays",
)


def element_fast_path_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_fast_paths.v1",
        "module": "geofem_app.fem2d_element_fast_paths",
        "function_count": len(ELEMENT_FAST_PATH_FUNCTIONS),
        "functions": list(ELEMENT_FAST_PATH_FUNCTIONS),
        "covered_surfaces": [
            "elastic_post_fast_path_selection",
            "advanced_material_post_fast_path_selection",
            "j2dp_post_fast_path_selection",
            "mohr_coulomb_post_fast_path_selection",
            "plastic_state_arrays",
            "advanced_material_state_arrays",
        ],
    }


def _quad4_elastic_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    return (
        element.type.upper() == "QUAD4"
        and normalize_integration(element.integration) == "FULL"
        and not plastic_state
        and not _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and not bool(material.tension_cutoff)
    )


def _quad4_elastic_bbar_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    return (
        element.type.upper() == "QUAD4"
        and normalize_integration(element.integration) == "B-BAR"
        and not plastic_state
        and not _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and not bool(material.tension_cutoff)
    )


def _quad8_elastic_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) in {"FULL", "SRI"}
        and not plastic_state
        and not _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and not bool(material.tension_cutoff)
    )


def _quad8_elastic_bbar_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) == "B-BAR"
        and not plastic_state
        and not _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and not bool(material.tension_cutoff)
    )


def _quad8_elastic_tension_post_supported(material: ElasticPlaneStrainMaterial) -> bool:
    return (
        not _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and bool(material.tension_cutoff)
        and math.isfinite(float(material.tensile_strength))
    )


def _quad8_elastic_tension_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) in {"FULL", "SRI"}
        and not plastic_state
        and _quad8_elastic_tension_post_supported(material)
    )


def _quad8_elastic_tension_bbar_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) == "B-BAR"
        and not plastic_state
        and _quad8_elastic_tension_post_supported(material)
    )


def _quad4_advanced_elastic_post_supported(material: ElasticPlaneStrainMaterial) -> bool:
    source_model = str(material.advanced_model or material.model).lower().strip()
    return (
        source_model in {"nonlinear_elastic", "hardin_drnevich", "duncan_chang", "ramberg_osgood"}
        and _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and not bool(material.tension_cutoff)
    )


def _quad4_advanced_elastic_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD4"
        and normalize_integration(element.integration) == "FULL"
        and _quad4_advanced_elastic_post_supported(material)
    )


def _quad4_advanced_elastic_bbar_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD4"
        and normalize_integration(element.integration) == "B-BAR"
        and _quad4_advanced_elastic_post_supported(material)
    )


def _quad8_advanced_elastic_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) in {"FULL", "SRI"}
        and _quad4_advanced_elastic_post_supported(material)
    )


def _quad8_advanced_elastic_bbar_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) == "B-BAR"
        and _quad4_advanced_elastic_post_supported(material)
    )


def _quad4_advanced_elastic_tension_post_supported(material: ElasticPlaneStrainMaterial) -> bool:
    source_model = str(material.advanced_model or material.model).lower().strip()
    return (
        source_model in {"nonlinear_elastic", "hardin_drnevich", "duncan_chang", "ramberg_osgood"}
        and _is_advanced_material(material)
        and not _uses_plastic_strength_model(material)
        and bool(material.tension_cutoff)
        and math.isfinite(float(material.tensile_strength))
    )


def _quad4_advanced_elastic_tension_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD4"
        and normalize_integration(element.integration) == "FULL"
        and _quad4_advanced_elastic_tension_post_supported(material)
    )


def _quad4_advanced_elastic_tension_bbar_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD4"
        and normalize_integration(element.integration) == "B-BAR"
        and _quad4_advanced_elastic_tension_post_supported(material)
    )


def _quad8_advanced_elastic_tension_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) in {"FULL", "SRI"}
        and _quad4_advanced_elastic_tension_post_supported(material)
    )


def _quad8_advanced_elastic_tension_bbar_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) == "B-BAR"
        and _quad4_advanced_elastic_tension_post_supported(material)
    )


def _quad4_advanced_strength_j2dp_post_supported(material: ElasticPlaneStrainMaterial) -> bool:
    source_model = str(material.advanced_model or material.model).lower().strip()
    return (
        source_model in {"uw_clay", "pastor_zienkiewicz_sand", "pastor_zienkiewicz_clay", "liquefaction", "bilinear_liquefaction"}
        and _is_advanced_material(material)
        and _advanced_strength_model_name(material.advanced_params or {}) not in {"mohr_coulomb", "mc"}
    )


def _quad4_advanced_strength_mc_post_supported(material: ElasticPlaneStrainMaterial) -> bool:
    source_model = str(material.advanced_model or material.model).lower().strip()
    return (
        source_model in {"uw_clay", "pastor_zienkiewicz_sand", "pastor_zienkiewicz_clay", "liquefaction", "bilinear_liquefaction"}
        and _is_advanced_material(material)
        and (not bool(material.tension_cutoff) or math.isfinite(float(material.tensile_strength)))
        and _advanced_strength_model_name(material.advanced_params or {}) in {"mohr_coulomb", "mc"}
    )


def _quad4_advanced_strength_j2dp_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD4"
        and normalize_integration(element.integration) == "FULL"
        and _quad4_advanced_strength_j2dp_post_supported(material)
    )


def _quad4_advanced_strength_j2dp_bbar_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD4"
        and normalize_integration(element.integration) == "B-BAR"
        and _quad4_advanced_strength_j2dp_post_supported(material)
    )


def _quad8_advanced_strength_mc_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) in {"FULL", "SRI"}
        and _quad4_advanced_strength_mc_post_supported(material)
    )


def _quad8_advanced_strength_mc_bbar_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) == "B-BAR"
        and _quad4_advanced_strength_mc_post_supported(material)
    )


def _quad8_advanced_strength_j2dp_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) in {"FULL", "SRI"}
        and _quad4_advanced_strength_j2dp_post_supported(material)
    )


def _quad8_advanced_strength_j2dp_bbar_post_fast_path(element: Element2D, material: ElasticPlaneStrainMaterial) -> bool:
    return (
        element.type.upper() == "QUAD8"
        and normalize_integration(element.integration) == "B-BAR"
        and _quad4_advanced_strength_j2dp_post_supported(material)
    )


def _quad4_j2dp_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    model = str(material.model).lower().strip()
    if (
        element.type.upper() != "QUAD4"
        or normalize_integration(element.integration) != "FULL"
        or _is_advanced_material(material)
        or bool(material.tension_cutoff)
        or model not in {"von_mises", "j2", "drucker_prager", "dp"}
    ):
        return False
    if plastic_state:
        for gp_index in range(4):
            state = plastic_state.get(_plastic_state_key(element.id, gp_index))
            if state is not None and state.state_vars:
                return False
    return True


def _quad4_j2dp_bbar_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    model = str(material.model).lower().strip()
    if (
        element.type.upper() != "QUAD4"
        or normalize_integration(element.integration) != "B-BAR"
        or _is_advanced_material(material)
        or bool(material.tension_cutoff)
        or model not in {"von_mises", "j2", "drucker_prager", "dp"}
    ):
        return False
    if plastic_state:
        for gp_index in range(4):
            state = plastic_state.get(_plastic_state_key(element.id, gp_index))
            if state is not None and state.state_vars:
                return False
    return True


def _quad8_j2dp_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    model = str(material.model).lower().strip()
    mode = normalize_integration(element.integration)
    tension_ok = (not bool(material.tension_cutoff)) or math.isfinite(float(material.tensile_strength))
    if (
        element.type.upper() != "QUAD8"
        or mode not in {"FULL", "SRI"}
        or _is_advanced_material(material)
        or not tension_ok
        or model not in {"von_mises", "j2", "drucker_prager", "dp"}
    ):
        return False
    if plastic_state:
        count = 13 if mode == "SRI" else 9
        for gp_index in range(count):
            state = plastic_state.get(_plastic_state_key(element.id, gp_index))
            if state is not None and state.state_vars:
                return False
    return True


def _quad8_j2dp_bbar_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    model = str(material.model).lower().strip()
    tension_ok = (not bool(material.tension_cutoff)) or math.isfinite(float(material.tensile_strength))
    if (
        element.type.upper() != "QUAD8"
        or normalize_integration(element.integration) != "B-BAR"
        or _is_advanced_material(material)
        or not tension_ok
        or model not in {"von_mises", "j2", "drucker_prager", "dp"}
    ):
        return False
    if plastic_state:
        for gp_index in range(9):
            state = plastic_state.get(_plastic_state_key(element.id, gp_index))
            if state is not None and state.state_vars:
                return False
    return True


def _quad8_mc_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    model = str(material.model).lower().strip()
    mode = normalize_integration(element.integration)
    tension_ok = (not bool(material.tension_cutoff)) or math.isfinite(float(material.tensile_strength))
    if (
        element.type.upper() != "QUAD8"
        or mode not in {"FULL", "SRI"}
        or _is_advanced_material(material)
        or not tension_ok
        or model not in {"mohr_coulomb", "mc"}
    ):
        return False
    if plastic_state:
        count = 13 if mode == "SRI" else 9
        for gp_index in range(count):
            state = plastic_state.get(_plastic_state_key(element.id, gp_index))
            if state is not None and state.state_vars:
                return False
    return True


def _quad8_mc_bbar_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    model = str(material.model).lower().strip()
    tension_ok = (not bool(material.tension_cutoff)) or math.isfinite(float(material.tensile_strength))
    if (
        element.type.upper() != "QUAD8"
        or normalize_integration(element.integration) != "B-BAR"
        or _is_advanced_material(material)
        or not tension_ok
        or model not in {"mohr_coulomb", "mc"}
    ):
        return False
    if plastic_state:
        for gp_index in range(9):
            state = plastic_state.get(_plastic_state_key(element.id, gp_index))
            if state is not None and state.state_vars:
                return False
    return True


def _quad4_mc_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    model = str(material.model).lower().strip()
    if (
        element.type.upper() != "QUAD4"
        or normalize_integration(element.integration) != "FULL"
        or _is_advanced_material(material)
        or bool(material.tension_cutoff)
        or model not in {"mohr_coulomb", "mc"}
    ):
        return False
    if plastic_state:
        for gp_index in range(4):
            state = plastic_state.get(_plastic_state_key(element.id, gp_index))
            if state is not None and state.state_vars:
                return False
    return True


def _quad4_mc_bbar_post_fast_path(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> bool:
    model = str(material.model).lower().strip()
    if (
        element.type.upper() != "QUAD4"
        or normalize_integration(element.integration) != "B-BAR"
        or _is_advanced_material(material)
        or bool(material.tension_cutoff)
        or model not in {"mohr_coulomb", "mc"}
    ):
        return False
    if plastic_state:
        for gp_index in range(4):
            state = plastic_state.get(_plastic_state_key(element.id, gp_index))
            if state is not None and state.state_vars:
                return False
    return True


def _quad4_post_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> tuple[np.ndarray, np.ndarray]:
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


def _quad8_post_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
    count: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
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


def _quad4_advanced_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((4, len(_ADV_STATE_FIELDS)), dtype=float)
    kappas = np.zeros(4, dtype=float)
    ru_default = float(params[_ADV_PARAM_RU_INITIAL])
    for gp_index in range(4):
        state_vars = None
        if plastic_state:
            state = plastic_state.get(_plastic_state_key(element_id, gp_index))
            if state is not None:
                state_vars = state.state_vars
                kappas[gp_index] = float(state.kappa)
        states[gp_index, :] = _advanced_state_array_from_vars(state_vars, ru_default=ru_default)
    return states, kappas


def _quad8_advanced_state_arrays(
    element_id: str,
    plastic_state: Mapping[str, PlasticState2D] | None,
    params: np.ndarray,
    count: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.zeros((count, len(_ADV_STATE_FIELDS)), dtype=float)
    kappas = np.zeros(count, dtype=float)
    ru_default = float(params[_ADV_PARAM_RU_INITIAL])
    for gp_index in range(count):
        state_vars = None
        if plastic_state:
            state = plastic_state.get(_plastic_state_key(element_id, gp_index))
            if state is not None:
                state_vars = state.state_vars
                kappas[gp_index] = float(state.kappa)
        states[gp_index, :] = _advanced_state_array_from_vars(state_vars, ru_default=ru_default)
    return states, kappas




__all__ = [
    "ELEMENT_FAST_PATH_FUNCTIONS",
    "element_fast_path_contract",
    "_quad4_elastic_post_fast_path",
    "_quad4_elastic_bbar_post_fast_path",
    "_quad8_elastic_post_fast_path",
    "_quad8_elastic_bbar_post_fast_path",
    "_quad8_elastic_tension_post_supported",
    "_quad8_elastic_tension_post_fast_path",
    "_quad8_elastic_tension_bbar_post_fast_path",
    "_quad4_advanced_elastic_post_supported",
    "_quad4_advanced_elastic_post_fast_path",
    "_quad4_advanced_elastic_bbar_post_fast_path",
    "_quad8_advanced_elastic_post_fast_path",
    "_quad8_advanced_elastic_bbar_post_fast_path",
    "_quad4_advanced_elastic_tension_post_supported",
    "_quad4_advanced_elastic_tension_post_fast_path",
    "_quad4_advanced_elastic_tension_bbar_post_fast_path",
    "_quad8_advanced_elastic_tension_post_fast_path",
    "_quad8_advanced_elastic_tension_bbar_post_fast_path",
    "_quad4_advanced_strength_j2dp_post_supported",
    "_quad4_advanced_strength_mc_post_supported",
    "_quad4_advanced_strength_j2dp_post_fast_path",
    "_quad4_advanced_strength_j2dp_bbar_post_fast_path",
    "_quad8_advanced_strength_mc_post_fast_path",
    "_quad8_advanced_strength_mc_bbar_post_fast_path",
    "_quad8_advanced_strength_j2dp_post_fast_path",
    "_quad8_advanced_strength_j2dp_bbar_post_fast_path",
    "_quad4_j2dp_post_fast_path",
    "_quad4_j2dp_bbar_post_fast_path",
    "_quad8_j2dp_post_fast_path",
    "_quad8_j2dp_bbar_post_fast_path",
    "_quad8_mc_post_fast_path",
    "_quad8_mc_bbar_post_fast_path",
    "_quad4_mc_post_fast_path",
    "_quad4_mc_bbar_post_fast_path",
    "_quad4_post_state_arrays",
    "_quad8_post_state_arrays",
    "_quad4_advanced_state_arrays",
    "_quad8_advanced_state_arrays",
]
