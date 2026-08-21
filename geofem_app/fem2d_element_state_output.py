"""Element material-state output helpers.

The element kernel module keeps interpolation, stress recovery, and element
matrices. This module owns the scalar material-state rows that are merged into
post-processing element and integration-point results.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fem2d_types import ElasticPlaneStrainMaterial


ELEMENT_STATE_OUTPUT_FUNCTIONS = (
    '_material_state_output',
    '_default_material_state_output',
    '_average_material_state_outputs',
    '_inactive_material_state_output',
)


def element_state_output_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.element_state_output.v1",
        "function_count": len(ELEMENT_STATE_OUTPUT_FUNCTIONS),
        "functions": list(ELEMENT_STATE_OUTPUT_FUNCTIONS),
        "numeric_field_count": len(_MATERIAL_STATE_NUMERIC_FIELDS),
        "owner_boundary": "element_state_output builds material-state rows for post-processing; fem2d_elements keeps element kinematics, kernels, and stress recovery",
        "covered_surfaces": ["post_material_state", "inactive_result_state", "integration_point_state_average"],
    }


_MATERIAL_STATE_NUMERIC_FIELDS = (
    "gamma_eq",
    "delta_gamma",
    "cyclic_strain",
    "cycle_increment",
    "cycles",
    "ru",
    "ru_generation_increment",
    "ru_dissipation_increment",
    "ru_dissipation_rate",
    "liquefaction_FL",
    "modulus_ratio",
    "effective_G",
    "effective_E",
    "hardening_variable",
    "dilatancy",
    "plastic_multiplier",
)
def _material_state_output(material: ElasticPlaneStrainMaterial, update: Any) -> dict[str, Any]:
    state = dict(getattr(update, "state_vars", {}) or {})
    out: dict[str, Any] = {
        "material_model": material.model,
        "advanced_model": str(state.get("advanced_model", material.advanced_model or "")),
    }
    for key in _MATERIAL_STATE_NUMERIC_FIELDS:
        value = state.get(key, 0.0)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = math.inf if key == "liquefaction_FL" else 0.0
        out[key] = parsed if math.isfinite(parsed) or key == "liquefaction_FL" else 0.0
    return out
def _default_material_state_output(material: ElasticPlaneStrainMaterial) -> dict[str, Any]:
    row: dict[str, Any] = {"material_model": material.model, "advanced_model": material.advanced_model or ""}
    for key in _MATERIAL_STATE_NUMERIC_FIELDS:
        row[key] = 0.0
    return row
def _average_material_state_outputs(rows: list[dict[str, Any]], weights: np.ndarray, material: ElasticPlaneStrainMaterial) -> dict[str, Any]:
    out: dict[str, Any] = {
        "material_model": material.model,
        "advanced_model": ";".join(sorted({str(row.get("advanced_model", "")) for row in rows if row.get("advanced_model")})),
    }
    if not out["advanced_model"]:
        out["advanced_model"] = material.advanced_model or ""
    for key in _MATERIAL_STATE_NUMERIC_FIELDS:
        values = np.asarray([float(row.get(key, math.inf if key == "liquefaction_FL" else 0.0)) for row in rows], dtype=float)
        finite = np.isfinite(values)
        if key == "liquefaction_FL" and not np.all(finite):
            out[key] = math.inf
        elif values.size:
            safe_weights = weights[finite] if finite.size == weights.size else weights
            safe_values = values[finite] if finite.any() else np.zeros(1, dtype=float)
            if safe_values.size == safe_weights.size and safe_values.size:
                out[key] = float(np.average(safe_values, weights=safe_weights))
            else:
                out[key] = float(np.mean(values[finite])) if finite.any() else 0.0
        else:
            out[key] = 0.0
    return out
def _inactive_material_state_output(material_name: str) -> dict[str, Any]:
    row: dict[str, Any] = {"material_model": "", "advanced_model": ""}
    for key in _MATERIAL_STATE_NUMERIC_FIELDS:
        row[key] = math.inf if key == "liquefaction_FL" else 0.0
    return row

