"""Element post-processing row builders for stress and state output."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .fem2d_element_state_output import (
    _default_material_state_output,
    _inactive_material_state_output,
    _material_state_output,
)
from .fem2d_materials import (
    _ADV_STATE_FIELDS,
    _plastic_state_for_gp,
    _plastic_state_key,
    principal_stresses,
    update_plane_strain_stress,
)
from .fem2d_types import ElasticPlaneStrainMaterial, Element2D, PlasticState2D, normalize_integration


ELEMENT_RESULT_ROW_FUNCTIONS = (
    "element_result_row_contract",
    "_quad4_elastic_post_result_rows",
    "_quad8_elastic_tension_post_result_rows",
    "_quad4_advanced_elastic_post_result_rows",
    "_quad4_advanced_elastic_tension_post_result_rows",
    "_quad4_advanced_strength_j2dp_post_result_rows",
    "_quad4_j2dp_post_result_rows",
    "_quad4_mc_post_result_rows",
    "_integration_point_result_row",
    "_inactive_integration_point_result",
    "_inactive_element_result",
)


def element_result_row_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_result_rows.v1",
        "module": "geofem_app.fem2d_element_result_rows",
        "function_count": len(ELEMENT_RESULT_ROW_FUNCTIONS),
        "functions": list(ELEMENT_RESULT_ROW_FUNCTIONS),
        "covered_surfaces": [
            "elastic_post_result_rows",
            "advanced_material_result_rows",
            "j2dp_result_rows",
            "mohr_coulomb_result_rows",
            "generic_integration_point_rows",
            "inactive_result_rows",
        ],
    }


def _quad4_elastic_post_result_rows(
    element: Element2D,
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
            "plastic": 0.0,
            "yield_value": 0.0,
            "p": float(row_data[18]),
            "q": float(row_data[19]),
            "active_set": "",
            "kappa": 0.0,
            "plastic_strain_x": 0.0,
            "plastic_strain_y": 0.0,
            "plastic_strain_z": 0.0,
            "plastic_strain_gamma_xy": 0.0,
        }
        row.update(material_state)
        rows.append(row)
    return rows


def _quad8_elastic_tension_post_result_rows(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    data: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    integration = normalize_integration(element.integration)
    material_state = _default_material_state_output(material)
    for gp_index, row_data in enumerate(np.asarray(data, dtype=float)):
        clipped = float(row_data[20])
        row = {
            "element_id": element.id,
            "ip": gp_index + 1,
            "state_key": _plastic_state_key(element.id, gp_index),
            "xi": float(row_data[0]),
            "eta": float(row_data[1]),
            "weight": float(row_data[2]),
            "x": float(row_data[3]),
            "y": float(row_data[4]),
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
            "plastic": clipped,
            "yield_value": float(row_data[21]),
            "p": float(row_data[18]),
            "q": float(row_data[19]),
            "active_set": "tension_cutoff" if clipped else "",
            "kappa": 0.0,
            "plastic_strain_x": 0.0,
            "plastic_strain_y": 0.0,
            "plastic_strain_z": 0.0,
            "plastic_strain_gamma_xy": 0.0,
        }
        row.update(material_state)
        rows.append(row)
    return rows


def _quad4_advanced_elastic_post_result_rows(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    data: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    integration = normalize_integration(element.integration)
    source_model = material.advanced_model or material.model
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
            "plastic": 0.0,
            "yield_value": 0.0,
            "p": float(row_data[18]),
            "q": float(row_data[19]),
            "active_set": "",
            "kappa": float(row_data[35]),
            "plastic_strain_x": 0.0,
            "plastic_strain_y": 0.0,
            "plastic_strain_z": 0.0,
            "plastic_strain_gamma_xy": 0.0,
            "material_model": material.model,
            "advanced_model": source_model,
        }
        for index, key in enumerate(_ADV_STATE_FIELDS):
            row[key] = float(row_data[20 + index])
        row["plastic_multiplier"] = 0.0
        rows.append(row)
    return rows


def _quad4_advanced_elastic_tension_post_result_rows(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    data: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    integration = normalize_integration(element.integration)
    source_model = material.advanced_model or material.model
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
            "plastic": float(row_data[36]),
            "yield_value": float(row_data[37]),
            "p": float(row_data[18]),
            "q": float(row_data[19]),
            "active_set": "",
            "kappa": float(row_data[35]),
            "plastic_strain_x": 0.0,
            "plastic_strain_y": 0.0,
            "plastic_strain_z": 0.0,
            "plastic_strain_gamma_xy": 0.0,
            "material_model": material.model,
            "advanced_model": source_model,
        }
        for index, key in enumerate(_ADV_STATE_FIELDS):
            row[key] = float(row_data[20 + index])
        row["plastic_multiplier"] = 0.0
        rows.append(row)
    return rows


def _quad4_advanced_strength_j2dp_post_result_rows(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    data: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    integration = normalize_integration(element.integration)
    source_model = material.advanced_model or material.model
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
            "active_set": "",
            "kappa": float(row_data[22]),
            "plastic_strain_x": float(row_data[23]),
            "plastic_strain_y": float(row_data[24]),
            "plastic_strain_z": float(row_data[25]),
            "plastic_strain_gamma_xy": float(row_data[26]),
            "material_model": material.model,
            "advanced_model": source_model,
        }
        for index, key in enumerate(_ADV_STATE_FIELDS):
            row[key] = float(row_data[27 + index])
        row["plastic_multiplier"] = float(row_data[42])
        rows.append(row)
    return rows


def _quad4_j2dp_post_result_rows(
    element: Element2D,
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
            "active_set": "",
            "kappa": float(row_data[22]),
            "plastic_strain_x": float(row_data[23]),
            "plastic_strain_y": float(row_data[24]),
            "plastic_strain_z": float(row_data[25]),
            "plastic_strain_gamma_xy": float(row_data[26]),
        }
        row.update(material_state)
        rows.append(row)
    return rows


def _quad4_mc_post_result_rows(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    data: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    integration = normalize_integration(element.integration)
    material_state = _default_material_state_output(material)
    for gp_index, row_data in enumerate(np.asarray(data, dtype=float)):
        active_count = max(0, min(3, int(round(float(row_data[27])))))
        active_ids = [str(int(round(float(row_data[28 + i])))) for i in range(active_count)]
        row = {
            "element_id": element.id,
            "ip": gp_index + 1,
            "state_key": _plastic_state_key(element.id, gp_index),
            "xi": float(row_data[0]),
            "eta": float(row_data[1]),
            "weight": float(row_data[2]),
            "x": float(row_data[3]),
            "y": float(row_data[4]),
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
            "active_set": "/".join(active_ids),
            "kappa": float(row_data[22]),
            "plastic_strain_x": float(row_data[23]),
            "plastic_strain_y": float(row_data[24]),
            "plastic_strain_z": float(row_data[25]),
            "plastic_strain_gamma_xy": float(row_data[26]),
        }
        row.update(material_state)
        rows.append(row)
    return rows


def _integration_point_result_row(
    element: Element2D,
    material: ElasticPlaneStrainMaterial,
    gp_index: int,
    gp: tuple[float, float, float],
    x: float,
    y: float,
    dV: float,
    strain: np.ndarray,
    initial_stress: np.ndarray,
    strength_factor: float,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> dict[str, Any]:
    old_state = _plastic_state_for_gp(plastic_state, element.id, gp_index)
    update = update_plane_strain_stress(material, strain, state=old_state, initial_stress=initial_stress, strength_factor=strength_factor)
    principal = principal_stresses(update.stress)
    active_set = "/".join(str(i) for i in update.active_set)
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
        "active_set": active_set,
        "kappa": float(update.kappa),
        "plastic_strain_x": float(plastic_strain[0]),
        "plastic_strain_y": float(plastic_strain[1]),
        "plastic_strain_z": float(plastic_strain[2]),
        "plastic_strain_gamma_xy": float(plastic_strain[3]),
    }
    row.update(_material_state_output(material, update))
    return row


def _inactive_integration_point_result(
    element: Element2D,
    gp_index: int,
    gp: tuple[float, float, float],
    x: float,
    y: float,
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
        "active_set": "",
        "kappa": 0.0,
        "plastic_strain_x": 0.0,
        "plastic_strain_y": 0.0,
        "plastic_strain_z": 0.0,
        "plastic_strain_gamma_xy": 0.0,
    }
    row.update(_inactive_material_state_output(element.material))
    return row


def _inactive_element_result(element: Element2D) -> dict[str, Any]:
    row = {
        "element_id": element.id,
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
        "active_set": "",
    }
    row.update(_inactive_material_state_output(element.material))
    return row





__all__ = [
    "ELEMENT_RESULT_ROW_FUNCTIONS",
    "element_result_row_contract",
    "_quad4_elastic_post_result_rows",
    "_quad8_elastic_tension_post_result_rows",
    "_quad4_advanced_elastic_post_result_rows",
    "_quad4_advanced_elastic_tension_post_result_rows",
    "_quad4_advanced_strength_j2dp_post_result_rows",
    "_quad4_j2dp_post_result_rows",
    "_quad4_mc_post_result_rows",
    "_integration_point_result_row",
    "_inactive_integration_point_result",
    "_inactive_element_result",
]
