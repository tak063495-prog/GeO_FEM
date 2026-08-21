"""Boundary constraint and MPC penalty helpers for 2D FEM solvers."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .fem2d_mesh import _target_nodes
from .fem2d_structural import structural_rotation_dof_map
from .fem2d_types import DOF_NAMES, FEM2DError, Interface2D, Mesh2D, StructuralElement2D
from .fem2d_utils import _ensure_list, _require_sequence

CONSTRAINT_HELPER_FUNCTIONS = (
    "collect_constraints",
    "assemble_mpc_penalty",
    "mpc_violation",
    "_add_inactive_node_constraints",
)


def constraint_helper_contract() -> dict[str, Any]:
    return {
        "schema": "geofem.fem2d.constraint_helpers.v1",
        "module": "geofem_app.fem2d_constraints",
        "function_count": len(CONSTRAINT_HELPER_FUNCTIONS),
        "functions": list(CONSTRAINT_HELPER_FUNCTIONS),
        "covered_surfaces": [
            "boundary_condition_dof_mapping",
            "inactive_node_constraints",
            "mpc_penalty_matrix",
            "mpc_violation_postcheck",
        ],
    }


def collect_constraints(
    mesh: Mesh2D,
    boundary_conditions: Any,
    structural_elements: list[StructuralElement2D] | None = None,
) -> dict[int, float]:
    node_index = mesh.node_index
    rotation_dofs = structural_rotation_dof_map(mesh, structural_elements)
    constrained: dict[int, float] = {}
    for bc in _ensure_list(boundary_conditions):
        if not isinstance(bc, Mapping):
            raise FEM2DError("each boundary condition must be a mapping")
        targets = _target_nodes(mesh, bc)
        if "dof" in bc:
            dofs = ["rz" if str(bc["dof"]).lower() in {"theta", "rotation"} else str(bc["dof"]).lower()]
            value = float(bc.get("value", 0.0))
            values = {dof: value for dof in dofs}
        else:
            values = {}
            for dof in DOF_NAMES:
                if dof in bc and bc[dof] is not None:
                    values[dof] = float(bc[dof])
            for dof in ("rz", "theta", "rotation"):
                if dof in bc and bc[dof] is not None:
                    values["rz"] = float(bc[dof])
            if "dofs" in bc:
                value = float(bc.get("value", 0.0))
                for dof in _require_sequence(bc["dofs"], "bc.dofs"):
                    name = str(dof).lower()
                    values["rz" if name in {"theta", "rotation"} else name] = value
            if bool(bc.get("fixed", False)):
                values.setdefault("ux", 0.0)
                values.setdefault("uy", 0.0)
            if bool(bc.get("fix_rotation", bc.get("fixed_rotation", False))):
                values.setdefault("rz", 0.0)
        for dof in values:
            if dof != "rz" and dof not in DOF_NAMES:
                raise FEM2DError(f"unsupported 2D dof '{dof}'")
        for nid in targets:
            base = 2 * node_index[nid]
            for dof, value in values.items():
                if dof == "rz":
                    if nid not in rotation_dofs:
                        continue
                    constrained[rotation_dofs[nid]] = float(value)
                else:
                    constrained[base + DOF_NAMES.index(dof)] = float(value)
    return constrained

def assemble_mpc_penalty(mesh: Mesh2D, reference_stiffness: csr_matrix, mpc_constraints: Any) -> tuple[csr_matrix, np.ndarray, dict[str, Any]]:
    ndof = reference_stiffness.shape[0]
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    load = np.zeros(ndof, dtype=float)
    specs = _ensure_list(mpc_constraints)
    if not specs:
        return csr_matrix((ndof, ndof), dtype=float), load, {"count": 0, "penalty": 0.0, "max_violation": 0.0}
    diag = reference_stiffness.diagonal()
    base = float(np.max(np.abs(diag))) if diag.size else 1.0
    if not np.isfinite(base) or base <= 0.0:
        base = 1.0
    default_penalty = base * 1.0e10
    equations: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        if not isinstance(spec, Mapping):
            raise FEM2DError("each MPC constraint must be a mapping")
        dof_name = str(spec.get("dof", spec.get("component", "ux"))).lower().strip()
        if dof_name not in DOF_NAMES:
            raise FEM2DError(f"MPC[{idx}]: unsupported dof '{dof_name}'")
        master = str(spec.get("master", spec.get("master_node", spec.get("node_master", "")))).strip()
        slave = str(spec.get("slave", spec.get("slave_node", spec.get("node_slave", "")))).strip()
        if not master or not slave:
            raise FEM2DError(f"MPC[{idx}]: master and slave nodes are required")
        if master not in mesh.node_index or slave not in mesh.node_index:
            raise FEM2DError(f"MPC[{idx}]: unknown master/slave node '{master}'/'{slave}'")
        coefficient = float(spec.get("coefficient", spec.get("coef", spec.get("scale", 1.0))))
        value = float(spec.get("value", spec.get("offset", 0.0)))
        penalty = float(spec.get("penalty", default_penalty))
        if penalty <= 0.0:
            raise FEM2DError(f"MPC[{idx}]: penalty must be positive")
        dof_offset = DOF_NAMES.index(dof_name)
        master_dof = 2 * mesh.node_index[master] + dof_offset
        slave_dof = 2 * mesh.node_index[slave] + dof_offset
        coeffs = [(slave_dof, 1.0), (master_dof, -coefficient)]
        for row_dof, row_coeff in coeffs:
            load[row_dof] += penalty * value * row_coeff
            for col_dof, col_coeff in coeffs:
                rows.append(row_dof)
                cols.append(col_dof)
                data.append(penalty * row_coeff * col_coeff)
        equations.append(
            {
                "master": master,
                "slave": slave,
                "dof": dof_name,
                "master_dof": master_dof,
                "slave_dof": slave_dof,
                "coefficient": coefficient,
                "value": value,
                "penalty": penalty,
                "method": str(spec.get("method", "penalty")),
            }
        )
    matrix = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()
    return matrix, load, {"count": len(equations), "penalty": default_penalty, "equations": equations}

def mpc_violation(mesh: Mesh2D, u: np.ndarray, mpc_info: Mapping[str, Any]) -> float:
    equations = mpc_info.get("equations", [])
    if not isinstance(equations, list) or not equations:
        return 0.0
    max_violation = 0.0
    for eq in equations:
        if not isinstance(eq, Mapping):
            continue
        dof_name = str(eq.get("dof", "ux"))
        if dof_name not in DOF_NAMES:
            continue
        master = str(eq.get("master", ""))
        slave = str(eq.get("slave", ""))
        if master not in mesh.node_index or slave not in mesh.node_index:
            continue
        offset = DOF_NAMES.index(dof_name)
        master_dof = 2 * mesh.node_index[master] + offset
        slave_dof = 2 * mesh.node_index[slave] + offset
        value = float(eq.get("value", 0.0))
        coefficient = float(eq.get("coefficient", 1.0))
        max_violation = max(max_violation, abs(float(u[slave_dof] - coefficient * u[master_dof] - value)))
    return max_violation

def _add_inactive_node_constraints(
    mesh: Mesh2D,
    constrained: dict[int, float],
    interfaces: list[Interface2D] | None = None,
    structural_elements: list[StructuralElement2D] | None = None,
) -> None:
    active_nodes: set[str] = set()
    for element in mesh.elements:
        if element.active:
            active_nodes.update(element.nodes)
    for interface in interfaces or []:
        if interface.active:
            active_nodes.update(interface.minus_nodes)
            active_nodes.update(interface.plus_nodes)
    for element in structural_elements or []:
        if element.active:
            active_nodes.update(element.nodes)
    node_index = mesh.node_index
    for nid in mesh.node_ids:
        if nid in active_nodes:
            continue
        base = 2 * node_index[nid]
        constrained.setdefault(base, 0.0)
        constrained.setdefault(base + 1, 0.0)
