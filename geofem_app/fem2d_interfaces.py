"""Mechanical and hydraulic interface element operations."""

from __future__ import annotations

import html
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from .fem2d_types import FEM2DError, Interface2D, Mesh2D, _symmetrize
from .fem2d_utils import _element_dofs


def assemble_interface_hydraulic_transfer(mesh: Mesh2D, interfaces: list[Interface2D] | None, *, axisymmetric: bool = False) -> tuple[csr_matrix, dict[str, Any]]:
    nnode_total = len(mesh.node_ids)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    total = 0.0
    count = 0
    node_index = mesh.node_index
    for interface in interfaces or []:
        if not interface.active or interface.hydraulic_transfer <= 0.0:
            continue
        minus = np.array([mesh.coords[node_index[nid]] for nid in interface.minus_nodes], dtype=float)
        plus = np.array([mesh.coords[node_index[nid]] for nid in interface.plus_nodes], dtype=float)
        midline = 0.5 * (minus + plus)
        axis = midline[1] - midline[0]
        length = float(np.linalg.norm(axis))
        if length <= 0.0:
            raise FEM2DError(f"interface {interface.id}: length must be positive")
        gauss = [(-1.0 / math.sqrt(3.0), 1.0), (1.0 / math.sqrt(3.0), 1.0)]
        interface_measure = 0.0
        for s, weight in gauss:
            shape = np.array([(1.0 - s) / 2.0, (1.0 + s) / 2.0], dtype=float)
            dL = _interface_measure(interface, midline, length, shape, weight, axisymmetric=axisymmetric)
            interface_measure += dL
            for a, Na in enumerate(shape):
                im = node_index[interface.minus_nodes[a]]
                ip = node_index[interface.plus_nodes[a]]
                for b, Nb in enumerate(shape):
                    jm = node_index[interface.minus_nodes[b]]
                    jp = node_index[interface.plus_nodes[b]]
                    value = interface.hydraulic_transfer * Na * Nb * dL
                    rows.extend([im, im, ip, ip])
                    cols.extend([jm, jp, jp, jm])
                    data.extend([value, -value, value, -value])
        total += interface.hydraulic_transfer * interface_measure
        count += 1
    matrix = coo_matrix((data, (rows, cols)), shape=(nnode_total, nnode_total)).tocsr()
    return matrix, {"count": count, "conductance_total": total}


def interface_stiffness(interface: Interface2D, mesh: Mesh2D) -> tuple[np.ndarray, np.ndarray]:
    dofs, _force, tangent = interface_force_tangent(interface, mesh)
    return dofs, tangent


def interface_force_tangent(interface: Interface2D, mesh: Mesh2D, ue: np.ndarray | None = None, *, axisymmetric: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    node_index = mesh.node_index
    ordered_nodes = (*interface.minus_nodes, *interface.plus_nodes)
    dofs = _element_dofs(ordered_nodes, node_index)
    if ue is None:
        ue = np.zeros(8, dtype=float)
    else:
        ue = np.asarray(ue, dtype=float)
        if ue.shape != (8,):
            raise FEM2DError(f"interface {interface.id}: displacement vector must have 8 entries")
    minus = np.array([mesh.coords[node_index[nid]] for nid in interface.minus_nodes], dtype=float)
    plus = np.array([mesh.coords[node_index[nid]] for nid in interface.plus_nodes], dtype=float)
    midline = 0.5 * (minus + plus)
    axis = midline[1] - midline[0]
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise FEM2DError(f"interface {interface.id}: length must be positive")
    tangent = axis / length
    normal = np.array([tangent[1], -tangent[0]], dtype=float)
    transform = np.vstack([tangent, normal])
    linear_local = np.diag([interface.kt, interface.kn])
    force = np.zeros(8, dtype=float)
    ke = np.zeros((8, 8), dtype=float)
    gauss = [(-1.0 / math.sqrt(3.0), 1.0), (1.0 / math.sqrt(3.0), 1.0)]
    for s, weight in gauss:
        shape = np.array([(1.0 - s) / 2.0, (1.0 + s) / 2.0], dtype=float)
        jump = np.zeros((2, 8), dtype=float)
        for a, Na in enumerate(shape):
            jump[:, 2 * a : 2 * a + 2] -= Na * np.eye(2)
            plus_col = 4 + 2 * a
            jump[:, plus_col : plus_col + 2] += Na * np.eye(2)
        local_jump = transform @ jump
        local_gap = local_jump @ ue
        if interface.friction <= 0.0 and interface.cohesion <= 0.0 and not interface.no_tension:
            traction = linear_local @ local_gap
            tangent_local = linear_local
        else:
            shear_trial = interface.kt * local_gap[0]
            normal_trial = interface.kn * local_gap[1]
            if interface.no_tension and normal_trial > 0.0:
                normal_traction = 0.0
                kn_eff = 0.0
            else:
                normal_traction = normal_trial
                kn_eff = interface.kn
            compression = max(-normal_traction, 0.0)
            shear_limit = interface.cohesion + interface.friction * compression
            if abs(shear_trial) > shear_limit:
                shear_traction = math.copysign(shear_limit, shear_trial) if shear_limit > 0.0 else 0.0
                kt_eff = 0.0
            else:
                shear_traction = shear_trial
                kt_eff = interface.kt
            traction = np.array([shear_traction, normal_traction], dtype=float)
            tangent_local = np.diag([kt_eff, kn_eff])
        dL = _interface_measure(interface, midline, length, shape, weight, axisymmetric=axisymmetric)
        force += local_jump.T @ traction * dL
        ke += local_jump.T @ tangent_local @ local_jump * dL
    return dofs, force, _symmetrize(ke)


def compute_interface_results(mesh: Mesh2D, interfaces: list[Interface2D] | None, u: np.ndarray, *, axisymmetric: bool = False) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for interface in interfaces or []:
        if not interface.active:
            continue
        dofs = _element_dofs((*interface.minus_nodes, *interface.plus_nodes), mesh.node_index)
        rows.extend(_interface_state_rows(interface, mesh, u[dofs], axisymmetric=axisymmetric))
    return rows


def update_interface_histories(interfaces: list[Interface2D] | None, rows: list[dict[str, Any]]) -> None:
    by_id = {interface.id: interface for interface in interfaces or []}
    for row in rows:
        interface = by_id.get(str(row.get("interface_id", "")))
        if interface is None:
            continue
        gp = str(row.get("gp", "0"))
        interface.history[gp] = {
            "gap_t": float(row.get("gap_t", 0.0) or 0.0),
            "gap_n": float(row.get("gap_n", 0.0) or 0.0),
            "slip": float(row.get("slip", 0.0) or 0.0),
            "slip_abs": float(row.get("slip_abs", 0.0) or 0.0),
            "cumulative_slip": float(row.get("cumulative_slip", row.get("slip_abs", 0.0)) or 0.0),
            "opening": float(row.get("opening", 0.0) or 0.0),
            "closure": float(row.get("closure", 0.0) or 0.0),
            "normal_state": str(row.get("normal_state", "")),
            "contact_state": str(row.get("contact_state", "")),
            "open_close_cycles": float(row.get("open_close_cycles", 0.0) or 0.0),
            "effective_roughness": float(row.get("effective_roughness", row.get("roughness", 0.0)) or 0.0),
        }


def _interface_state_rows(interface: Interface2D, mesh: Mesh2D, ue: np.ndarray, *, axisymmetric: bool = False) -> list[dict[str, Any]]:
    node_index = mesh.node_index
    minus = np.array([mesh.coords[node_index[nid]] for nid in interface.minus_nodes], dtype=float)
    plus = np.array([mesh.coords[node_index[nid]] for nid in interface.plus_nodes], dtype=float)
    midline = 0.5 * (minus + plus)
    axis = midline[1] - midline[0]
    length = float(np.linalg.norm(axis))
    if length <= 0.0:
        raise FEM2DError(f"interface {interface.id}: length must be positive")
    tangent = axis / length
    normal = np.array([tangent[1], -tangent[0]], dtype=float)
    transform = np.vstack([tangent, normal])
    rows: list[dict[str, Any]] = []
    gauss = [(-1.0 / math.sqrt(3.0), 1.0), (1.0 / math.sqrt(3.0), 1.0)]
    for gp_index, (s, weight) in enumerate(gauss):
        shape = np.array([(1.0 - s) / 2.0, (1.0 + s) / 2.0], dtype=float)
        measure = _interface_measure(interface, midline, length, shape, weight, axisymmetric=axisymmetric)
        jump = np.zeros((2, 8), dtype=float)
        for a, Na in enumerate(shape):
            jump[:, 2 * a : 2 * a + 2] -= Na * np.eye(2)
            plus_col = 4 + 2 * a
            jump[:, plus_col : plus_col + 2] += Na * np.eye(2)
        local_gap = transform @ jump @ ue
        shear_trial = interface.kt * local_gap[0]
        normal_trial = interface.kn * local_gap[1]
        open_contact = bool(interface.no_tension and normal_trial > 0.0)
        if open_contact:
            normal_traction = 0.0
        else:
            normal_traction = normal_trial
        compression = max(-normal_traction, 0.0)
        shear_limit = interface.cohesion + interface.friction * compression
        slip = 0.0
        state = "stick"
        if open_contact:
            shear_traction = 0.0
            state = "open"
        elif (interface.friction > 0.0 or interface.cohesion > 0.0) and abs(shear_trial) > shear_limit:
            shear_traction = math.copysign(shear_limit, shear_trial) if shear_limit > 0.0 else 0.0
            slip = local_gap[0] - (shear_traction / interface.kt if interface.kt > 0.0 else 0.0)
            state = "slip"
        else:
            shear_traction = shear_trial
        opening = float(max(local_gap[1], 0.0))
        closure = float(max(-local_gap[1], 0.0))
        gp_history = _interface_gp_history(interface, gp_index)
        previous_normal = str(gp_history.get("normal_state", ""))
        normal_state = "open" if open_contact else "closed"
        open_close_cycles = float(gp_history.get("open_close_cycles", 0.0) or 0.0)
        if previous_normal in {"open", "closed"} and previous_normal != normal_state:
            open_close_cycles += 1.0
        previous_slip = float(gp_history.get("slip", 0.0) or 0.0)
        previous_opening = float(gp_history.get("opening", 0.0) or 0.0)
        previous_closure = float(gp_history.get("closure", 0.0) or 0.0)
        slip_increment = abs(float(slip) - previous_slip)
        opening_increment = opening - previous_opening
        closure_increment = closure - previous_closure
        cumulative_slip = float(gp_history.get("cumulative_slip", 0.0) or 0.0) + slip_increment
        residual = min(max(float(interface.residual_roughness_ratio), 0.0), 1.0)
        degradation = max(float(interface.roughness_degradation), 0.0)
        roughness_loss = min(degradation * (cumulative_slip + open_close_cycles), 1.0 - residual)
        roughness_factor = max(1.0 - roughness_loss, residual)
        effective_roughness = float(interface.roughness) * roughness_factor
        effective_dilatancy_angle = float(interface.dilatancy_angle) * roughness_factor if interface.roughness else float(interface.dilatancy_angle)
        dilatancy = math.tan(math.radians(effective_dilatancy_angle)) * abs(local_gap[0]) if effective_dilatancy_angle else 0.0
        row = {
            "interface_id": interface.id,
            "material_model": interface.material_model,
            "gp": gp_index,
            "s": float(s),
            "weight": float(weight),
            "gap_t": float(local_gap[0]),
            "gap_n": float(local_gap[1]),
            "traction_t": float(shear_traction),
            "traction_n": float(normal_traction),
            "slip": float(slip),
            "slip_abs": float(abs(slip)),
            "state": state,
            "contact_state": "open" if state == "open" else ("closed_slip" if state == "slip" else "closed_stick"),
            "normal_state": "open" if open_contact else "closed",
            "friction_limit": float(shear_limit),
            "friction": float(interface.friction),
            "cohesion": float(interface.cohesion),
            "no_tension": int(interface.no_tension),
            "roughness": float(interface.roughness),
            "effective_roughness": float(effective_roughness),
            "roughness_loss": float(roughness_loss),
            "roughness_degradation_rate": float(interface.roughness_degradation),
            "residual_roughness_ratio": float(interface.residual_roughness_ratio),
            "dilatancy_angle": float(interface.dilatancy_angle),
            "effective_dilatancy_angle": float(effective_dilatancy_angle),
            "dilation_gap": float(dilatancy if state == "slip" else 0.0),
            "closure": closure,
            "opening": opening,
            "closure_increment": float(closure_increment),
            "opening_increment": float(opening_increment),
            "slip_increment": float(slip_increment),
            "cumulative_slip": float(cumulative_slip),
            "open_close_cycles": float(open_close_cycles),
            "compression": float(compression),
        }
        if axisymmetric:
            row["geometry"] = "axisymmetric"
            row["radius"] = float(shape @ midline[:, 0])
            row["measure"] = float(measure)
        rows.append(row)
    return rows


def _interface_gp_history(interface: Interface2D, gp_index: int) -> Mapping[str, Any]:
    raw = interface.history.get(str(gp_index), interface.history.get(gp_index, {}))
    return raw if isinstance(raw, Mapping) else {}


def estimate_joint_mohr_coulomb_parameters(rows: list[dict[str, Any]] | list[tuple[float, float]]) -> dict[str, float]:
    """Estimate tau = c + mu * compression from direct-shear style points."""

    if not rows:
        raise FEM2DError("joint parameter estimation requires at least one test point")
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        if isinstance(row, dict):
            normal = float(row.get("compression", row.get("normal_stress", row.get("sigma_n", 0.0))))
            shear = float(row.get("shear_peak", row.get("tau_peak", row.get("shear", row.get("tau", 0.0)))))
        else:
            normal, shear = float(row[0]), float(row[1])
        x_values.append(max(normal, 0.0))
        y_values.append(abs(shear))
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    A = np.column_stack([np.ones_like(x), x])
    coeff, *_ = np.linalg.lstsq(A, y, rcond=None)
    cohesion = max(float(coeff[0]), 0.0)
    friction = max(float(coeff[1]), 0.0)
    residual = y - A @ coeff
    rmse = float(math.sqrt(float(np.mean(residual**2)))) if residual.size else 0.0
    return {
        "cohesion": cohesion,
        "friction": friction,
        "friction_angle_deg": math.degrees(math.atan(friction)),
        "rmse": rmse,
        "point_count": float(len(y_values)),
    }


def write_joint_standard_report(
    rows: list[dict[str, Any]],
    path: str | Path,
    *,
    title: str = "Joint element standard report",
    metadata: Mapping[str, Any] | None = None,
) -> None:
    slip_rows = [row for row in rows if str(row.get("state", "")) == "slip"]
    open_rows = [row for row in rows if str(row.get("normal_state", "")) == "open"]
    roughness_values = [float(row.get("effective_roughness", row.get("roughness", 0.0)) or 0.0) for row in rows]
    roughness_base = [float(row.get("roughness", 0.0) or 0.0) for row in rows]
    min_retention = 1.0
    if roughness_values and roughness_base and max(roughness_base) > 0.0:
        ratios = [eff / base for eff, base in zip(roughness_values, roughness_base) if base > 0.0]
        min_retention = min(ratios) if ratios else 1.0
    summary = [
        ("行数", len(rows)),
        ("すべり点数", len(slip_rows)),
        ("開口点数", len(open_rows)),
        ("最大すべり量", _rows_max(rows, "slip_abs")),
        ("最大開口量", _rows_max(rows, "opening")),
        ("最大閉口量", _rows_max(rows, "closure")),
        ("最大開閉サイクル", _rows_max(rows, "open_close_cycles")),
        ("最小粗度保持率", min_retention),
    ]
    meta_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in (metadata or {}).items())
    summary_rows = "".join(f"<tr><th>{html.escape(str(k))}</th><td>{_format_value(v)}</td></tr>" for k, v in summary)
    detail_rows = []
    for row in rows:
        detail_rows.append(
            "<tr><td>{interface}</td><td>{gp}</td><td>{state}</td><td>{normal}</td>"
            "<td>{slip}</td><td>{opening}</td><td>{closure}</td><td>{cycles}</td>"
            "<td>{traction_t}</td><td>{traction_n}</td><td>{limit}</td><td>{roughness}</td></tr>".format(
                interface=html.escape(str(row.get("interface_id", ""))),
                gp=html.escape(str(row.get("gp", ""))),
                state=html.escape(str(row.get("state", ""))),
                normal=html.escape(str(row.get("normal_state", ""))),
                slip=_format_value(row.get("slip_abs", 0.0)),
                opening=_format_value(row.get("opening", 0.0)),
                closure=_format_value(row.get("closure", 0.0)),
                cycles=_format_value(row.get("open_close_cycles", 0.0)),
                traction_t=_format_value(row.get("traction_t", 0.0)),
                traction_n=_format_value(row.get("traction_n", 0.0)),
                limit=_format_value(row.get("friction_limit", 0.0)),
                roughness=_format_value(row.get("effective_roughness", row.get("roughness", 0.0))),
            )
        )
    document = f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #111827; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 12px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3), th:nth-child(4), td:nth-child(4) {{ text-align: left; }}
th {{ background: #e5e7eb; }}
</style>
</head>
<body>
<h1>{html.escape(title)}</h1>
<h2>Metadata</h2>
<table>{meta_rows}</table>
<h2>Summary</h2>
<table>{summary_rows}</table>
<h2>Integration point states</h2>
<table>
<thead><tr><th>interface</th><th>gp</th><th>state</th><th>normal</th><th>slip_abs</th><th>opening</th><th>closure</th><th>cycles</th><th>traction_t</th><th>traction_n</th><th>friction_limit</th><th>effective_roughness</th></tr></thead>
<tbody>{''.join(detail_rows)}</tbody>
</table>
</body>
</html>
"""
    Path(path).write_text(document, encoding="utf-8")


def _rows_max(rows: list[dict[str, Any]], key: str) -> float:
    return max((float(row.get(key, 0.0) or 0.0) for row in rows), default=0.0)


def _format_value(value: Any) -> str:
    try:
        return f"{float(value):.6g}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def _interface_measure(interface: Interface2D, midline: np.ndarray, length: float, shape: np.ndarray, weight: float, *, axisymmetric: bool) -> float:
    measure = (length * weight / 2.0) * interface.thickness
    if not axisymmetric:
        return float(measure)
    radius = float(shape @ midline[:, 0])
    if radius <= 0.0:
        raise FEM2DError(f"interface {interface.id}: axisymmetric radius must be positive, got {radius:.6e}")
    return float(measure * 2.0 * math.pi * radius)

__all__ = [
    "assemble_interface_hydraulic_transfer",
    "interface_stiffness",
    "interface_force_tangent",
    "compute_interface_results",
    "update_interface_histories",
    "estimate_joint_mohr_coulomb_parameters",
    "write_joint_standard_report",
    "_interface_state_rows",
    "_interface_measure",
]

