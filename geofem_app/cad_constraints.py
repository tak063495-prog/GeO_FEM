"""Small analytic-geometry constraint solver used by the CAD GUI.

The solver intentionally operates on 2D point groups, not on Qt objects.
GUI code is responsible for clustering editable CAD handles and applying the
solved point positions back to lines, curves, and dimensions.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares


def solve_cad_constraints(
    points: Mapping[str, Any],
    constraints: list[Mapping[str, Any]],
    *,
    tolerance: float = 1.0e-8,
    max_iterations: int = 100,
    regularization_weight: float = 1.0e-8,
) -> dict[str, Any]:
    """Solve common 2D CAD constraints by nonlinear least squares.

    Supported constraint types include length/distance, horizontal, vertical,
    fixed, coincident, angle, parallel, perpendicular, equal_length, tangent,
    concentric, and curvature_continuity. The result includes rank-based
    redundant/underconstrained diagnostics and hard residuals so the GUI can
    report overconstraints and contradictions.
    """

    point_map = _normalize_points(points)
    normalized_constraints = [_normalize_constraint(raw) for raw in constraints if isinstance(raw, Mapping)]
    auto_fixed = _auto_fixed_points(point_map, normalized_constraints)
    fixed_ids = {pid for pid, point in point_map.items() if point["locked"]}
    fixed_ids.update(auto_fixed)
    variable_ids = [pid for pid in point_map if pid not in fixed_ids]
    initial = {pid: (float(point["x"]), float(point["y"])) for pid, point in point_map.items()}
    variable_index = {pid: i for i, pid in enumerate(variable_ids)}
    x0 = _pack_variables(variable_ids, initial)

    if variable_ids:
        result = least_squares(
            lambda values: _residual_vector(
                values,
                point_map,
                normalized_constraints,
                variable_ids,
                variable_index,
                initial,
                regularization_weight=regularization_weight,
                include_regularization=True,
            )[0],
            x0,
            max_nfev=max_iterations,
            xtol=max(tolerance * 0.1, 1.0e-12),
            ftol=max(tolerance * 0.1, 1.0e-12),
            gtol=max(tolerance * 0.1, 1.0e-12),
        )
        solved = _coords_from_values(result.x, point_map, variable_ids, variable_index, initial)
        hard_residuals, labels = _residual_vector(
            result.x,
            point_map,
            normalized_constraints,
            variable_ids,
            variable_index,
            initial,
            regularization_weight=regularization_weight,
            include_regularization=False,
        )
        hard_count = int(hard_residuals.size)
        jac = np.asarray(result.jac[:hard_count, :], dtype=float) if hard_count else np.zeros((0, len(variable_ids) * 2))
        iterations = int(result.nfev)
        scipy_status = int(result.status)
        scipy_message = str(result.message)
    else:
        solved = dict(initial)
        hard_residuals, labels = _residual_vector(
            np.zeros(0, dtype=float),
            point_map,
            normalized_constraints,
            variable_ids,
            variable_index,
            initial,
            regularization_weight=regularization_weight,
            include_regularization=False,
        )
        jac = np.zeros((int(hard_residuals.size), 0), dtype=float)
        iterations = 0
        scipy_status = 0
        scipy_message = "no free CAD point groups"

    rank = int(np.linalg.matrix_rank(jac, tol=max(tolerance * 10.0, 1.0e-10))) if jac.size else 0
    hard_values = [float(value) for value in hard_residuals.tolist()]
    max_abs = max((abs(value) for value in hard_values), default=0.0)
    residual_tolerance = max(tolerance * 100.0, 1.0e-8)
    hard_count = len(hard_values)
    dof = len(variable_ids) * 2
    redundant_count = max(0, hard_count - rank)
    underconstrained_dof = max(0, dof - rank)
    inconsistent = max_abs > residual_tolerance
    diagnostics = {
        "engine": "cad_constraint_least_squares",
        "status": "inconsistent" if inconsistent else "solved",
        "iterations": iterations,
        "scipy_status": scipy_status,
        "scipy_message": scipy_message,
        "point_count": len(point_map),
        "variable_point_count": len(variable_ids),
        "active_dof": dof,
        "constraint_scalar_count": hard_count,
        "constraint_rank": rank,
        "redundant_count": redundant_count,
        "underconstrained_dof": underconstrained_dof,
        "overconstrained": hard_count > dof,
        "inconsistent": inconsistent,
        "max_abs_residual": max_abs,
        "residual_tolerance": residual_tolerance,
        "auto_fixed_points": sorted(auto_fixed),
        "locked_points": sorted(pid for pid, point in point_map.items() if point["locked"]),
        "residuals": [
            {"label": label, "value": value, "abs": abs(value)}
            for label, value in zip(labels, hard_values)
        ],
    }
    return {
        "points": {pid: [float(x), float(y)] for pid, (x, y) in solved.items()},
        "diagnostics": diagnostics,
    }


def _normalize_points(points: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for pid, raw in points.items():
        key = str(pid)
        if isinstance(raw, Mapping):
            x = float(raw.get("x", raw.get("X", raw.get("coord", [0.0, 0.0])[0] if isinstance(raw.get("coord"), list) else 0.0)))
            y = float(raw.get("y", raw.get("Y", raw.get("coord", [0.0, 0.0])[1] if isinstance(raw.get("coord"), list) and len(raw.get("coord", [])) > 1 else 0.0)))
            locked = bool(raw.get("locked", raw.get("lock", False)))
        else:
            values = list(raw)
            x = float(values[0])
            y = float(values[1])
            locked = False
        out[key] = {"x": x, "y": y, "locked": locked}
    return out


def _normalize_constraint(raw: Mapping[str, Any]) -> dict[str, Any]:
    constraint = dict(raw)
    ctype = str(constraint.get("type", constraint.get("kind", "length"))).strip().lower()
    aliases = {
        "distance": "length",
        "dim": "length",
        "dimension": "length",
        "locked": "length",
        "equal": "equal_length",
        "equal-length": "equal_length",
        "same_length": "equal_length",
        "h": "horizontal",
        "v": "vertical",
        "perp": "perpendicular",
        "orthogonal": "perpendicular",
        "same_center": "concentric",
        "co_center": "concentric",
        "g1": "tangent",
        "c1": "tangent",
        "tangency": "tangent",
        "g2": "curvature_continuity",
        "c2": "curvature_continuity",
        "curvature": "curvature_continuity",
        "curvature-continuity": "curvature_continuity",
        "nurbs_curvature": "curvature_continuity",
    }
    constraint["type"] = aliases.get(ctype, ctype)
    if "p1" not in constraint:
        constraint["p1"] = constraint.get("start_id", constraint.get("center_id", constraint.get("center", constraint.get("start"))))
    if "p2" not in constraint:
        constraint["p2"] = constraint.get("end_id", constraint.get("reference_center_id", constraint.get("reference_center", constraint.get("end"))))
    if "p3" not in constraint:
        constraint["p3"] = constraint.get("third_id", constraint.get("third", constraint.get("next")))
    if "reference_p1" not in constraint:
        constraint["reference_p1"] = constraint.get("ref_p1", constraint.get("reference_start"))
    if "reference_p2" not in constraint:
        constraint["reference_p2"] = constraint.get("ref_p2", constraint.get("reference_end"))
    if "reference_p3" not in constraint:
        constraint["reference_p3"] = constraint.get("ref_p3", constraint.get("reference_third", constraint.get("reference_next")))
    return constraint


def _auto_fixed_points(points: Mapping[str, Mapping[str, Any]], constraints: list[Mapping[str, Any]]) -> set[str]:
    graph: dict[str, set[str]] = defaultdict(set)
    constrained_points: set[str] = set()
    for constraint in constraints:
        ids = [
            str(value)
            for value in (
                constraint.get("p1"),
                constraint.get("p2"),
                constraint.get("p3"),
                constraint.get("reference_p1"),
                constraint.get("reference_p2"),
                constraint.get("reference_p3"),
            )
            if value is not None
        ]
        ids = [pid for pid in ids if pid in points]
        constrained_points.update(ids)
        for a in ids:
            for b in ids:
                if a != b:
                    graph[a].add(b)
    fixed: set[str] = set()
    seen: set[str] = set()
    for start in sorted(constrained_points):
        if start in seen:
            continue
        queue: deque[str] = deque([start])
        component: list[str] = []
        seen.add(start)
        while queue:
            pid = queue.popleft()
            component.append(pid)
            for nxt in graph.get(pid, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        if component and not any(bool(points[pid]["locked"]) for pid in component):
            fixed.add(sorted(component)[0])
    return fixed


def _pack_variables(variable_ids: list[str], coords: Mapping[str, tuple[float, float]]) -> np.ndarray:
    values: list[float] = []
    for pid in variable_ids:
        x, y = coords[pid]
        values.extend([x, y])
    return np.asarray(values, dtype=float)


def _coords_from_values(
    values: np.ndarray,
    points: Mapping[str, Mapping[str, Any]],
    variable_ids: list[str],
    variable_index: Mapping[str, int],
    initial: Mapping[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    coords = dict(initial)
    for pid in variable_ids:
        offset = variable_index[pid] * 2
        coords[pid] = (float(values[offset]), float(values[offset + 1]))
    for pid, point in points.items():
        if bool(point.get("locked", False)):
            coords[pid] = initial[pid]
    return coords


def _residual_vector(
    values: np.ndarray,
    points: Mapping[str, Mapping[str, Any]],
    constraints: list[Mapping[str, Any]],
    variable_ids: list[str],
    variable_index: Mapping[str, int],
    initial: Mapping[str, tuple[float, float]],
    *,
    regularization_weight: float,
    include_regularization: bool,
) -> tuple[np.ndarray, list[str]]:
    coords = _coords_from_values(values, points, variable_ids, variable_index, initial)
    residuals: list[float] = []
    labels: list[str] = []
    for index, constraint in enumerate(constraints, start=1):
        ctype = str(constraint.get("type", "length")).lower()
        cid = str(constraint.get("id", index))
        p1 = str(constraint.get("p1", ""))
        p2 = str(constraint.get("p2", ""))
        if p1 not in coords:
            continue
        if ctype == "fixed":
            target = _target_point(constraint, initial[p1])
            x1, y1 = coords[p1]
            residuals.extend([x1 - target[0], y1 - target[1]])
            labels.extend([f"{cid}:fixed:x", f"{cid}:fixed:y"])
            continue
        if p2 not in coords:
            continue
        x1, y1 = coords[p1]
        x2, y2 = coords[p2]
        dx = x2 - x1
        dy = y2 - y1
        if ctype in {"length", "distance"}:
            value = _target_value(constraint, math.hypot(initial[p2][0] - initial[p1][0], initial[p2][1] - initial[p1][1]))
            residuals.append(math.hypot(dx, dy) - value)
            labels.append(f"{cid}:length")
        elif ctype == "horizontal":
            residuals.append(dy)
            labels.append(f"{cid}:horizontal")
            value = _optional_target_value(constraint)
            if value is not None:
                sign = _direction_sign(initial[p2][0] - initial[p1][0])
                residuals.append(dx - sign * abs(value))
                labels.append(f"{cid}:horizontal_length")
        elif ctype == "vertical":
            residuals.append(dx)
            labels.append(f"{cid}:vertical")
            value = _optional_target_value(constraint)
            if value is not None:
                sign = _direction_sign(initial[p2][1] - initial[p1][1])
                residuals.append(dy - sign * abs(value))
                labels.append(f"{cid}:vertical_length")
        elif ctype in {"coincident", "concentric"}:
            residuals.extend([dx, dy])
            labels.extend([f"{cid}:{ctype}:x", f"{cid}:{ctype}:y"])
        elif ctype == "angle":
            angle = math.radians(float(constraint.get("angle_degrees", constraint.get("value", 0.0))))
            residuals.append(_angle_difference(math.atan2(dy, dx), angle))
            labels.append(f"{cid}:angle")
        elif ctype in {"parallel", "perpendicular", "equal_length", "tangent"}:
            ref = _reference_vector(constraint, coords)
            if ref is None:
                continue
            rdx, rdy = ref
            scale = max(math.hypot(rdx, rdy), 1.0)
            if ctype in {"parallel", "tangent"}:
                residuals.append((dx * rdy - dy * rdx) / scale)
                labels.append(f"{cid}:{ctype}")
            elif ctype == "perpendicular":
                residuals.append((dx * rdx + dy * rdy) / scale)
                labels.append(f"{cid}:perpendicular")
            else:
                residuals.append(math.hypot(dx, dy) - math.hypot(rdx, rdy))
                labels.append(f"{cid}:equal_length")
        elif ctype == "curvature_continuity":
            curvature = _curvature_vectors(constraint, coords)
            if curvature is None:
                continue
            tangent_a, tangent_b, curve_a, curve_b = curvature
            residuals.append(_cross_unit(tangent_a, tangent_b))
            labels.append(f"{cid}:curvature_continuity:tangent")
            residuals.append(curve_a[0] - curve_b[0])
            residuals.append(curve_a[1] - curve_b[1])
            labels.extend([f"{cid}:curvature_continuity:kx", f"{cid}:curvature_continuity:ky"])
    if include_regularization and regularization_weight > 0.0:
        for pid in variable_ids:
            x, y = coords[pid]
            x0, y0 = initial[pid]
            residuals.extend([(x - x0) * regularization_weight, (y - y0) * regularization_weight])
            labels.extend([f"{pid}:anchor:x", f"{pid}:anchor:y"])
    return np.asarray(residuals, dtype=float), labels


def _target_value(constraint: Mapping[str, Any], fallback: float) -> float:
    value = _optional_target_value(constraint)
    return float(fallback) if value is None else float(value)


def _optional_target_value(constraint: Mapping[str, Any]) -> float | None:
    for key in ("value", "target", "length"):
        if key in constraint and constraint.get(key) is not None:
            try:
                return float(constraint[key])
            except (TypeError, ValueError):
                return None
    return None


def _target_point(constraint: Mapping[str, Any], fallback: tuple[float, float]) -> tuple[float, float]:
    target = constraint.get("target_point", constraint.get("point"))
    if isinstance(target, (list, tuple)) and len(target) >= 2:
        return float(target[0]), float(target[1])
    return fallback


def _reference_vector(constraint: Mapping[str, Any], coords: Mapping[str, tuple[float, float]]) -> tuple[float, float] | None:
    p1 = constraint.get("reference_p1")
    p2 = constraint.get("reference_p2")
    if p1 is None or p2 is None:
        return None
    a = str(p1)
    b = str(p2)
    if a not in coords or b not in coords:
        return None
    x1, y1 = coords[a]
    x2, y2 = coords[b]
    return x2 - x1, y2 - y1


def _curvature_vectors(
    constraint: Mapping[str, Any],
    coords: Mapping[str, tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]] | None:
    ids = [
        constraint.get("p1"),
        constraint.get("p2"),
        constraint.get("p3"),
        constraint.get("reference_p1"),
        constraint.get("reference_p2"),
        constraint.get("reference_p3"),
    ]
    if any(value is None or str(value) not in coords for value in ids):
        return None
    p1, p2, p3, q1, q2, q3 = [coords[str(value)] for value in ids]
    tangent_a = (p2[0] - p1[0], p2[1] - p1[1])
    tangent_b = (q2[0] - q1[0], q2[1] - q1[1])
    second_a = (p3[0] - 2.0 * p2[0] + p1[0], p3[1] - 2.0 * p2[1] + p1[1])
    second_b = (q3[0] - 2.0 * q2[0] + q1[0], q3[1] - 2.0 * q2[1] + q1[1])
    scale_a = max(tangent_a[0] * tangent_a[0] + tangent_a[1] * tangent_a[1], 1.0e-14)
    scale_b = max(tangent_b[0] * tangent_b[0] + tangent_b[1] * tangent_b[1], 1.0e-14)
    curve_a = (second_a[0] / scale_a, second_a[1] / scale_a)
    curve_b = (second_b[0] / scale_b, second_b[1] / scale_b)
    return tangent_a, tangent_b, curve_a, curve_b


def _cross_unit(a: tuple[float, float], b: tuple[float, float]) -> float:
    an = math.hypot(a[0], a[1])
    bn = math.hypot(b[0], b[1])
    if an <= 1.0e-14 or bn <= 1.0e-14:
        return 0.0
    return (a[0] * b[1] - a[1] * b[0]) / (an * bn)


def _direction_sign(value: float) -> float:
    return -1.0 if value < 0.0 else 1.0


def _angle_difference(value: float, target: float) -> float:
    delta = value - target
    while delta > math.pi:
        delta -= 2.0 * math.pi
    while delta < -math.pi:
        delta += 2.0 * math.pi
    return delta


__all__ = ["solve_cad_constraints"]
