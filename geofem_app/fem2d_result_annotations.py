"""Result annotation and liquefaction state helpers for the 2D solver.

Solver stage functions keep orchestration and nonlinear solves. This module owns
small post-solve annotations: matrix/runtime metadata, integration-point result
attachment, and liquefaction state summaries derived from pore pressure.
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import numpy as np
from scipy.sparse import csr_matrix

from .fem2d_elements import compute_integration_point_results, integration_points, shape_functions
from .fem2d_materials import _param_float, _plastic_state_key
from .fem2d_types import ElasticPlaneStrainMaterial, Mesh2D, PlasticState2D, StageResult2D
from .fem2d_utils import _element_node_indices


RESULT_ANNOTATION_FUNCTIONS = (
    '_attach_matrix_profile',
    '_attach_stage_runtime',
    '_attach_integration_point_results',
    '_update_liquefaction_state_from_pore_pressure',
    '_material_has_liquefaction',
    '_liquefaction_state_summary',
    '_liquefaction_effective_stress_reference',
)


def result_annotation_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.result_annotations.v1",
        "function_count": len(RESULT_ANNOTATION_FUNCTIONS),
        "functions": list(RESULT_ANNOTATION_FUNCTIONS),
        "owner_boundary": "result_annotations attaches runtime/matrix metadata, integration-point post results, and liquefaction state summaries; fem2d_solver keeps stage progression and equation solves",
        "covered_surfaces": [
            "matrix_profile_metadata",
            "stage_runtime_metadata",
            "integration_point_results",
            "liquefaction_state_update",
            "liquefaction_summary",
        ],
    }


def _attach_matrix_profile(
    solver_info: dict[str, Any],
    matrix: csr_matrix,
    rhs: np.ndarray,
    constrained: Mapping[int, float],
    *,
    label: str = "global_stiffness",
) -> None:
    n = int(matrix.shape[0])
    fixed = len(constrained)
    nnz = int(matrix.nnz)
    solver_info["matrix"] = {
        "label": label,
        "size": n,
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": nnz,
        "density": 0.0 if n <= 0 else nnz / float(max(n * n, 1)),
        "constrained_dofs": fixed,
        "free_dofs": max(n - fixed, 0),
        "rhs_norm": float(np.linalg.norm(rhs)) if rhs.size else 0.0,
        "estimated_sparse_bytes": int(nnz * (8 + 4) + (n + 1) * 4),
    }
def _attach_stage_runtime(result: StageResult2D, started_at: float, mesh: Mesh2D) -> None:
    elapsed = max(time.perf_counter() - started_at, 0.0)
    perf = result.solver_info.setdefault("performance", {})
    if not isinstance(perf, dict):
        perf = {}
        result.solver_info["performance"] = perf
    perf.update(
        {
            "elapsed_seconds": elapsed,
            "node_count": len(mesh.node_ids),
            "element_count": len(mesh.elements),
            "active_element_count": len(result.active_elements),
            "dof_count": int(result.displacements.size),
        }
    )
def _attach_integration_point_results(
    result: StageResult2D,
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    u: np.ndarray,
    *,
    axisymmetric: bool = False,
    strength_factor: float = 1.0,
    plastic_state: Mapping[str, PlasticState2D] | None = None,
    initial_stresses: Mapping[str, np.ndarray] | None = None,
) -> None:
    if axisymmetric:
        from .fem2d_solver import compute_axisymmetric_integration_point_results

        result.integration_point_results = compute_axisymmetric_integration_point_results(
            mesh,
            materials,
            u,
            strength_factor=strength_factor,
            plastic_state=plastic_state,
            initial_stresses=initial_stresses,
        )
        return
    result.integration_point_results = compute_integration_point_results(
        mesh,
        materials,
        u,
        initial_stresses=initial_stresses,
        strength_factor=strength_factor,
        plastic_state=plastic_state,
    )
def _update_liquefaction_state_from_pore_pressure(
    mesh: Mesh2D,
    materials: Mapping[str, ElasticPlaneStrainMaterial],
    pore_pressure: np.ndarray | None,
    plastic_state: Mapping[str, PlasticState2D] | None,
) -> dict[str, PlasticState2D]:
    updated: dict[str, PlasticState2D] = dict(plastic_state or {})
    if pore_pressure is None or pore_pressure.size != len(mesh.node_ids):
        return updated
    node_index = mesh.node_index
    y_top = float(np.max(mesh.coords[:, 1])) if mesh.coords.size else 0.0
    for element in mesh.elements:
        if not element.active:
            continue
        material = materials[element.material]
        if not _material_has_liquefaction(material):
            continue
        conn = _element_node_indices(element.nodes, node_index)
        coords = mesh.coords[conn]
        p_nodes = pore_pressure[conn]
        params = material.advanced_params or {}
        liq = params.get("liquefaction")
        liq_map = liq if isinstance(liq, Mapping) else params
        for gp_index, gp in enumerate(integration_points(element.type, "FULL")):
            _Nshape, _dN = shape_functions(element.type, gp[0], gp[1])
            pore_gp = float(_Nshape @ p_nodes)
            xy = _Nshape @ coords
            effective_ref = _liquefaction_effective_stress_reference(material, liq_map if isinstance(liq_map, Mapping) else {}, xy, y_top)
            ru = max(0.0, min(abs(pore_gp) / max(effective_ref, 1.0e-12), 0.99))
            key = _plastic_state_key(element.id, gp_index)
            old = updated.get(key, PlasticState2D())
            state_vars = dict(old.state_vars or {})
            state_vars["ru"] = max(float(state_vars.get("ru", 0.0) or 0.0), ru)
            state_vars["pore_pressure"] = pore_gp
            state_vars["initial_effective_stress"] = effective_ref
            state_vars["up_liquefaction_update"] = 1.0
            if isinstance(liq_map, Mapping):
                state_vars["ru_dissipation_rate"] = _param_float(liq_map, ("dissipation_rate", "ru_dissipation_rate"), float(state_vars.get("ru_dissipation_rate", 0.0) or 0.0))
            state_vars.setdefault("advanced_model", material.advanced_model or material.model)
            updated[key] = PlasticState2D(old.plastic_strain.copy(), old.kappa, state_vars)
    return updated
def _material_has_liquefaction(material: ElasticPlaneStrainMaterial) -> bool:
    model = material.advanced_model or material.model
    params = material.advanced_params or {}
    return model in {"liquefaction", "bilinear_liquefaction"} or isinstance(params.get("liquefaction"), Mapping)
def _liquefaction_state_summary(plastic_state: Mapping[str, PlasticState2D]) -> dict[str, Any]:
    rows = [state.state_vars for state in plastic_state.values() if state.state_vars and ("ru" in state.state_vars or "liquefaction_FL" in state.state_vars)]
    if not rows:
        return {"count": 0}
    ru_values = [float(row.get("ru", 0.0) or 0.0) for row in rows]
    fl_values = [float(row.get("liquefaction_FL", math.inf) or math.inf) for row in rows]
    finite_fl = [value for value in fl_values if math.isfinite(value)]
    return {
        "count": len(rows),
        "max_ru": max(ru_values) if ru_values else 0.0,
        "mean_ru": sum(ru_values) / len(ru_values) if ru_values else 0.0,
        "min_FL": min(finite_fl) if finite_fl else math.inf,
        "liquefied_points": sum(1 for value in ru_values if value >= 0.95),
        "up_updated_points": sum(1 for row in rows if float(row.get("up_liquefaction_update", 0.0) or 0.0) > 0.0),
    }
def _liquefaction_effective_stress_reference(
    material: ElasticPlaneStrainMaterial,
    liq_map: Mapping[str, Any],
    xy: np.ndarray,
    y_top: float,
) -> float:
    effective_ref = _param_float(liq_map, ("initial_effective_stress", "sigma_v_eff", "effective_stress"), 0.0)
    if effective_ref <= 0.0:
        effective_ref = abs(material.gamma) * max(y_top - float(xy[1]), 0.0)
    return max(float(effective_ref), 1.0)

