"""Detached CAD geometry worker helpers for GUI operations."""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping


def split_lines_at_intersections_snapshot(geometry: Mapping[str, Any]) -> dict[str, Any]:
    """Split CAD line records at pairwise intersections."""

    out_geometry = dict(geometry)
    lines = out_geometry.get("lines", [])
    if not isinstance(lines, list) or len(lines) < 2:
        return {"geometry": out_geometry, "split_count": 0}
    parsed: list[tuple[dict[str, Any], tuple[float, float], tuple[float, float]]] = []
    for line in lines:
        if not isinstance(line, Mapping):
            continue
        try:
            p0 = _xy_pair(line.get("start", [0.0, 0.0]))
            p1 = _xy_pair(line.get("end", [0.0, 0.0]))
        except ValueError:
            continue
        if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) > 1.0e-12:
            parsed.append((dict(line), p0, p1))
    split_params: dict[int, list[float]] = {i: [0.0, 1.0] for i in range(len(parsed))}
    for i in range(len(parsed)):
        for j in range(i + 1, len(parsed)):
            hit = _segment_intersection(parsed[i][1], parsed[i][2], parsed[j][1], parsed[j][2])
            if hit is None:
                continue
            ti, tj, _x, _y = hit
            if 1.0e-9 < ti < 1.0 - 1.0e-9:
                split_params[i].append(ti)
            if 1.0e-9 < tj < 1.0 - 1.0e-9:
                split_params[j].append(tj)
    new_lines: list[dict[str, Any]] = []
    split_count = 0
    for idx, (line, p0, p1) in enumerate(parsed):
        params = sorted(set(round(t, 12) for t in split_params[idx]))
        if len(params) <= 2:
            new_lines.append(line)
            continue
        split_count += len(params) - 2
        for a, b in zip(params, params[1:]):
            sx = p0[0] + (p1[0] - p0[0]) * a
            sy = p0[1] + (p1[1] - p0[1]) * a
            ex = p0[0] + (p1[0] - p0[0]) * b
            ey = p0[1] + (p1[1] - p0[1]) * b
            segment = dict(line)
            segment["id"] = _next_id("line", new_lines)
            segment["start"] = [sx, sy]
            segment["end"] = [ex, ey]
            new_lines.append(segment)
    if split_count:
        out_geometry["lines"] = new_lines
    return {"geometry": out_geometry, "split_count": split_count}


def solve_dimension_constraints_snapshot(
    window_cls: type[Any],
    geometry: Mapping[str, Any],
    constraints: list[dict[str, Any]],
) -> dict[str, Any]:
    """Solve CAD dimension constraints against a detached geometry snapshot."""

    from geofem_app.gui.model_check_worker import DetachedGuiContext

    geometry_copy = copy.deepcopy(dict(geometry))
    constraints_copy = copy.deepcopy(list(constraints))
    context = DetachedGuiContext(window_cls, {"geometry": geometry_copy})
    if constraints_copy:
        geometry_copy["dimension_constraints"] = constraints_copy
        window_cls._solve_dimension_constraints(context, geometry_copy, constraints_copy)
    else:
        geometry_copy.pop("dimension_constraints", None)
        geometry_copy.pop("dimension_solver", None)
    diagnostics = geometry_copy.get("dimension_solver", {})
    if not isinstance(diagnostics, Mapping):
        diagnostics = {}
    return {
        "geometry": geometry_copy,
        "constraint_count": len(constraints_copy),
        "diagnostics": dict(diagnostics),
    }


def _xy_pair(value: Any) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError("point must have two coordinates")
    return float(value[0]), float(value[1])


def _segment_intersection(
    a0: tuple[float, float],
    a1: tuple[float, float],
    b0: tuple[float, float],
    b1: tuple[float, float],
) -> tuple[float, float, float, float] | None:
    ax, ay = a0
    bx, by = a1
    cx, cy = b0
    dx, dy = b1
    rx = bx - ax
    ry = by - ay
    sx = dx - cx
    sy = dy - cy
    denom = rx * sy - ry * sx
    if abs(denom) <= 1.0e-12:
        return None
    qpx = cx - ax
    qpy = cy - ay
    t = (qpx * sy - qpy * sx) / denom
    u = (qpx * ry - qpy * rx) / denom
    if -1.0e-9 <= t <= 1.0 + 1.0e-9 and -1.0e-9 <= u <= 1.0 + 1.0e-9:
        return t, u, ax + t * rx, ay + t * ry
    return None


def _next_id(prefix: str, values: list[Mapping[str, Any]]) -> str:
    used = {str(item.get("id")) for item in values if item.get("id") is not None}
    index = len(values) + 1
    while f"{prefix}_{index}" in used:
        index += 1
    return f"{prefix}_{index}"


__all__ = ["solve_dimension_constraints_snapshot", "split_lines_at_intersections_snapshot"]
