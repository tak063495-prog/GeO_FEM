"""Analytic curve intersection and trim graph helpers for CAD-style Boolean work."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

try:
    from scipy.optimize import least_squares
except ImportError:  # pragma: no cover - scipy is a required runtime dependency.
    least_squares = None


Point2D = tuple[float, float]

_CURVE_SAMPLE_CACHE_MAX = 512
_CURVE_SAMPLE_CACHE: dict[tuple[Any, float], list[tuple[float, Point2D]]] = {}
_CURVE_LENGTH_CACHE: dict[tuple[Any, ...], float] = {}
_GRAPH_REGION_CACHE_MAX = 64
_GRAPH_REGION_CURVE_CACHE: dict[tuple[Any, ...], dict[int, list[Mapping[str, Any]]]] = {}
_GRAPH_REGION_POLYGON_CACHE: dict[
    tuple[Any, ...],
    dict[int, tuple[list[Point2D], tuple[float, float, float, float]]],
] = {}


def build_analytic_curve_boolean_graph(
    curve_regions: Sequence[Sequence[Mapping[str, Any]]],
    *,
    target: float,
    tol: float | None = None,
) -> dict[str, Any]:
    """Build a parameter-space split graph from analytic CAD boundary curves."""

    tolerance = tol if tol is not None else max(target * 1.0e-8, 1.0e-9)
    snap_tolerance = max(tolerance * 1000.0, target * 1.0e-6, 1.0e-8)
    curves: list[dict[str, Any]] = []
    for region_index, specs in enumerate(curve_regions, start=1):
        for curve_index, spec in enumerate(specs, start=1):
            source = public_curve_spec(spec)
            curves.append(
                {
                    "id": len(curves),
                    "region": region_index,
                    "curve": curve_index,
                    "type": str(source.get("type", "")).lower(),
                    "source": source,
                }
            )

    curve_params: dict[int, list[float]] = {int(curve["id"]): [0.0, 1.0] for curve in curves}
    intersections: list[dict[str, Any]] = []
    overlap_spans: list[dict[str, Any]] = []
    overlap_tol = max(snap_tolerance * 5.0, target * 1.0e-6, tolerance * 10000.0, 1.0e-8)
    for left_index, right_index in _curve_pair_candidates(curves, target, overlap_tol):
        left = curves[left_index]
        right = curves[right_index]
        pair_overlap_spans = [
            refined
            for span in _curve_overlap_spans(left["source"], right["source"], target, overlap_tol)
            if (refined := _refine_overlap_span(left["source"], right["source"], span, overlap_tol)) is not None
        ]
        hits = intersect_curves(left["source"], right["source"], target=target, tol=tolerance)
        for hit in hits:
            if _hit_inside_overlap_spans(hit, pair_overlap_spans, tolerance):
                continue
            point = hit["point"]
            if _append_unique_intersection(
                intersections,
                {
                    "id": len(intersections) + 1,
                    "curve_a": int(left["id"]),
                    "curve_b": int(right["id"]),
                    "region_a": int(left["region"]),
                    "region_b": int(right["region"]),
                    "t_a": float(hit["t_a"]),
                    "t_b": float(hit["t_b"]),
                    "point": [float(point[0]), float(point[1])],
                    "method": hit["method"],
                    "residual": float(hit.get("residual", 0.0)),
                },
                tolerance,
            ):
                curve_params[int(left["id"])].append(float(hit["t_a"]))
                curve_params[int(right["id"])].append(float(hit["t_b"]))
        for refined in pair_overlap_spans:
            ta0 = float(refined["t_a0"])
            ta1 = float(refined["t_a1"])
            tb0 = float(refined["t_b0"])
            tb1 = float(refined["t_b1"])
            if abs(ta1 - ta0) <= max(tolerance * 10.0, 1.0e-9) or abs(tb1 - tb0) <= max(tolerance * 10.0, 1.0e-9):
                continue
            start_point = eval_curve(left["source"], ta0)
            end_point = eval_curve(left["source"], ta1)
            overlap_spans.append(
                {
                    "id": str(len(overlap_spans) + 1),
                    "curve_a": int(left["id"]),
                    "curve_b": int(right["id"]),
                    "region_a": int(left["region"]),
                    "region_b": int(right["region"]),
                    "t_a0": ta0,
                    "t_a1": ta1,
                    "t_b0": tb0,
                    "t_b1": tb1,
                    "point_start": [float(start_point[0]), float(start_point[1])],
                    "point_end": [float(end_point[0]), float(end_point[1])],
                    "length": float(_trimmed_curve_length(left["source"], ta0, ta1)),
                    "method": str(refined.get("method", "sampled_overlap_span")),
                    "max_residual": float(refined.get("max_residual", 0.0)),
                }
            )
            curve_params[int(left["id"])].extend([ta0, ta1])
            curve_params[int(right["id"])].extend([tb0, tb1])

    vertices: dict[str, dict[str, Any]] = {}
    vertex_keys: dict[tuple[int, int], str] = {}
    edges: list[dict[str, Any]] = []
    quant = max(snap_tolerance, 1.0e-10)

    def vertex_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        found = vertex_keys.get(key)
        if found is not None:
            return found
        vid = str(len(vertices) + 1)
        vertex_keys[key] = vid
        vertices[vid] = {"id": vid, "point": [float(point[0]), float(point[1])]}
        return vid

    for curve in curves:
        curve_id = int(curve["id"])
        params = _unique_sorted_params(curve_params[curve_id], tolerance)
        for t0, t1 in zip(params, params[1:]):
            if t1 - t0 <= max(tolerance, 1.0e-10):
                continue
            start = eval_curve(curve["source"], t0)
            end = eval_curve(curve["source"], t1)
            mid = eval_curve(curve["source"], 0.5 * (t0 + t1))
            edges.append(
                {
                    "id": str(len(edges) + 1),
                    "curve_id": curve_id,
                    "region": int(curve["region"]),
                    "curve": int(curve["curve"]),
                    "type": curve["type"],
                    "t0": float(t0),
                    "t1": float(t1),
                    "start_vertex": vertex_id(start),
                    "end_vertex": vertex_id(end),
                    "midpoint": [float(mid[0]), float(mid[1])],
                    "source": curve["source"],
                }
            )

    overlap_edge_pairs = _annotate_overlap_edges(edges, overlap_spans, snap_tolerance)
    return {
        "engine": "analytic_curve_intersection_graph",
        "selection_engine": "unclassified",
        "region_count": len(curve_regions),
        "target": float(target),
        "curve_count": len(curves),
        "intersection_count": len(intersections),
        "overlap_span_count": len(overlap_spans),
        "overlap_edge_pair_count": len(overlap_edge_pairs),
        "split_edge_count": len(edges),
        "vertex_count": len(vertices),
        "curves": curves,
        "intersections": intersections,
        "overlap_spans": overlap_spans,
        "overlap_edge_pairs": overlap_edge_pairs,
        "overlap_edge_count": sum(1 for edge in edges if edge.get("overlap_span_ids")),
        "vertices": vertices,
        "edges": edges,
        "tolerance": tolerance,
        "snap_tolerance": snap_tolerance,
        "tolerance_diagnostics": _graph_tolerance_diagnostics(curves, intersections, target, tolerance, snap_tolerance),
    }


def classify_graph_boolean_operations(graph: dict[str, Any], expression: str | None = None) -> dict[str, Any]:
    """Classify split edges into Boolean operation boundaries using winding tests only."""

    region_count = int(graph.get("region_count", 0))
    target = float(graph.get("target", 1.0))
    tol = float(graph.get("tolerance", max(target * 1.0e-8, 1.0e-9)))
    snap_tol = float(graph.get("snap_tolerance", max(tol * 1000.0, target * 1.0e-6, 1.0e-8)))
    region_curves = _region_curve_sources(graph)
    region_polygons = _sampled_region_polygons(region_curves, target, snap_tol)
    offset = _graph_offset(graph, target, tol)
    compiled_expression = parse_boolean_expression(expression, region_count) if expression and expression.strip() else None
    for edge in graph.get("edges", []):
        midpoint = _point(edge.get("midpoint", [0.0, 0.0]))
        tangent = _edge_tangent(edge)
        normal = _left_normal(tangent)
        left_point = (midpoint[0] + normal[0] * offset, midpoint[1] + normal[1] * offset)
        right_point = (midpoint[0] - normal[0] * offset, midpoint[1] - normal[1] * offset)
        left_regions = _regions_containing_sampled_point(left_point, region_polygons, snap_tol)
        right_regions = _regions_containing_sampled_point(right_point, region_polygons, snap_tol)
        mid_regions = _regions_containing_sampled_point(midpoint, region_polygons, snap_tol)
        edge["left_regions"] = left_regions
        edge["right_regions"] = right_regions
        edge["inside_regions"] = mid_regions
        _classify_edge_operation(edge, "union", _boolean_value("union", left_regions, region_count), _boolean_value("union", right_regions, region_count))
        _classify_edge_operation(edge, "intersection", _boolean_value("intersection", left_regions, region_count), _boolean_value("intersection", right_regions, region_count))
        if region_count >= 1:
            for region in range(1, region_count + 1):
                name = f"difference_{region}"
                _classify_edge_operation(edge, name, _boolean_value(name, left_regions, region_count), _boolean_value(name, right_regions, region_count))
        if compiled_expression is not None:
            _classify_edge_operation(
                edge,
                "expression",
                _eval_boolean_ast(compiled_expression["ast"], left_regions),
                _eval_boolean_ast(compiled_expression["ast"], right_regions),
            )
        edge["boolean_role"] = "union_boundary" if bool(edge.get("graph_union_boundary", False)) else ("overlap_internal" if len(mid_regions) > 1 else "region_internal")
        edge["on_union_boundary"] = bool(edge.get("graph_union_boundary", False))
        if len(mid_regions) == 1:
            edge["difference_region"] = mid_regions[0]

    operations: dict[str, dict[str, Any]] = {}
    operation_names = ["union", "intersection", *[f"difference_{region}" for region in range(1, region_count + 1)]]
    if compiled_expression is not None:
        operation_names.append("expression")
    for name in operation_names:
        loops = trace_oriented_boolean_loops(graph, operation=name)
        edge_count = sum(1 for edge in graph.get("edges", []) if bool(edge.get(f"graph_{name}_boundary", False)))
        operations[name] = {
            "edge_count": edge_count,
            "loop_count": len(loops),
            "loops": loops,
        }
    graph["selection_engine"] = "analytic_winding_containment"
    graph["boolean_operations"] = operations
    graph["union_loops"] = operations.get("union", {}).get("loops", [])
    graph["intersection_loops"] = operations.get("intersection", {}).get("loops", [])
    graph["union_boundary_edge_count"] = operations.get("union", {}).get("edge_count", 0)
    graph["intersection_boundary_edge_count"] = operations.get("intersection", {}).get("edge_count", 0)
    graph["difference_edge_count"] = sum(operations.get(f"difference_{region}", {}).get("edge_count", 0) for region in range(1, region_count + 1))
    graph["expression_edge_count"] = operations.get("expression", {}).get("edge_count", 0)
    graph["overlap_internal_edge_count"] = sum(1 for edge in graph.get("edges", []) if edge.get("boolean_role") == "overlap_internal")
    if compiled_expression is not None:
        graph["boolean_expression"] = {
            "text": str(expression).strip(),
            "normalized": compiled_expression["normalized"],
            "regions": compiled_expression["regions"],
        }
    return graph


def parse_boolean_expression(expression: str | None, region_count: int) -> dict[str, Any]:
    """Parse a compact region Boolean expression such as ``A-B-C`` or ``(1|2)&3``."""

    text = "" if expression is None else str(expression).strip()
    if not text:
        raise ValueError("Boolean expression is empty")
    tokens = _tokenize_boolean_expression(text)
    parser = _BooleanExpressionParser(tokens, region_count)
    ast = parser.parse_expression()
    parser.expect_end()
    regions = sorted(_ast_region_ids(ast))
    return {
        "text": text,
        "normalized": _format_boolean_ast(ast),
        "ast": ast,
        "regions": regions,
    }


def operation_loop_polygons(
    graph: Mapping[str, Any],
    *,
    operation: str = "union",
    target: float | None = None,
    min_area: float | None = None,
) -> list[dict[str, Any]]:
    """Sample graph Boolean loops directly into mesh boundary rings."""

    requested_target = float(target if target is not None else graph.get("target", 1.0))
    area_tol = float(min_area if min_area is not None else max(requested_target * requested_target * 1.0e-8, 1.0e-14))
    point_tol = max(float(graph.get("snap_tolerance", requested_target * 1.0e-6)), requested_target * 1.0e-8, 1.0e-10)
    edges = {str(edge["id"]): edge for edge in graph.get("edges", [])}
    operation_data = graph.get("boolean_operations", {}).get(operation, {})
    rings: list[dict[str, Any]] = []
    for loop in operation_data.get("loops", []):
        if not bool(loop.get("closed", False)):
            continue
        points: list[Point2D] = []
        for step in loop.get("edges", []):
            edge = edges.get(str(step.get("edge", "")))
            if edge is None:
                continue
            samples = _sample_oriented_edge(edge, requested_target, bool(step.get("reversed", False)))
            for point in samples:
                if points and math.hypot(points[-1][0] - point[0], points[-1][1] - point[1]) <= point_tol:
                    continue
                points.append(point)
        if len(points) > 1 and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= point_tol:
            points.pop()
        points = _remove_near_collinear_points(points, point_tol)
        if len(points) < 3:
            continue
        area = polygon_area(points)
        if abs(area) <= area_tol:
            continue
        rings.append(
            {
                "operation": operation,
                "loop": str(loop.get("id", len(rings) + 1)),
                "points": points,
                "point_count": len(points),
                "area": float(area),
                "orientation": "ccw" if area > 0.0 else "cw",
                "source": "analytic_curve_graph_loop",
            }
        )
    rings.sort(key=lambda item: abs(float(item.get("area", 0.0))), reverse=True)
    return rings


def operation_loop_area(graph: Mapping[str, Any], *, operation: str = "intersection", target: float | None = None) -> float:
    """Return the absolute area enclosed by graph Boolean operation loops."""

    return sum(abs(float(ring.get("area", 0.0))) for ring in operation_loop_polygons(graph, operation=operation, target=target))


def regions_containing_point(graph: Mapping[str, Any], point: Point2D) -> list[int]:
    """Classify a point against graph source regions without Shapely/GEOS."""

    target = float(graph.get("target", 1.0))
    tol = float(graph.get("snap_tolerance", max(target * 1.0e-6, 1.0e-8)))
    signature = _graph_curve_signature(graph)
    region_curves = _cached_region_curve_sources(graph, signature)
    sampled_regions = _cached_sampled_graph_region_polygons(graph, signature, region_curves, target, tol)
    return _regions_containing_sampled_point(point, sampled_regions, tol)


def trace_oriented_boolean_loops(graph: Mapping[str, Any], *, operation: str) -> list[dict[str, Any]]:
    vertices: Mapping[str, Mapping[str, Any]] = graph.get("vertices", {})
    oriented: list[dict[str, Any]] = []
    for edge in graph.get("edges", []):
        if not bool(edge.get(f"graph_{operation}_boundary", False)):
            continue
        interior_side = str(edge.get(f"graph_{operation}_interior_side", "left"))
        reversed_edge = interior_side == "right"
        start = str(edge["end_vertex"] if reversed_edge else edge["start_vertex"])
        end = str(edge["start_vertex"] if reversed_edge else edge["end_vertex"])
        start_point = _point(vertices[start]["point"])
        end_point = _point(vertices[end]["point"])
        tangent = (end_point[0] - start_point[0], end_point[1] - start_point[1])
        oriented.append(
            {
                "edge": str(edge["id"]),
                "reversed": reversed_edge,
                "start": start,
                "end": end,
                "angle": math.atan2(tangent[1], tangent[0]),
            }
        )
    outgoing: dict[str, list[dict[str, Any]]] = {}
    for item in oriented:
        outgoing.setdefault(item["start"], []).append(item)
    for items in outgoing.values():
        items.sort(key=lambda item: item["angle"])

    visited: set[tuple[str, bool]] = set()
    loops: list[dict[str, Any]] = []
    for item in oriented:
        key = (item["edge"], bool(item["reversed"]))
        if key in visited:
            continue
        start_vertex = item["start"]
        current = item
        path: list[dict[str, Any]] = []
        guard = 0
        while guard <= len(oriented) + 2:
            guard += 1
            current_key = (current["edge"], bool(current["reversed"]))
            if current_key in visited:
                break
            visited.add(current_key)
            path.append({"edge": current["edge"], "reversed": bool(current["reversed"])})
            if current["end"] == start_vertex:
                break
            candidates = [candidate for candidate in outgoing.get(current["end"], []) if (candidate["edge"], bool(candidate["reversed"])) not in visited]
            if not candidates:
                break
            current = _choose_next_oriented_edge(current, candidates)
        closed = bool(path) and _path_end_vertex(path[-1], graph) == start_vertex
        points = _loop_points(path, {str(edge["id"]): edge for edge in graph.get("edges", [])}, vertices)
        area = polygon_area(points) if len(points) >= 3 else 0.0
        loops.append(
            {
                "id": str(len(loops) + 1),
                "operation": operation,
                "edge_count": len(path),
                "closed": closed,
                "area": float(area),
                "orientation": "ccw" if area > 0.0 else "cw",
                "edges": path,
            }
        )
    return loops


def trace_curve_boolean_loops(graph: Mapping[str, Any], *, edge_flag: str = "on_union_boundary") -> list[dict[str, Any]]:
    vertices: Mapping[str, Mapping[str, Any]] = graph.get("vertices", {})
    selected = [edge for edge in graph.get("edges", []) if bool(edge.get(edge_flag, False))]
    by_id = {str(edge["id"]): edge for edge in selected}
    adjacency: dict[str, list[str]] = {}
    for edge in selected:
        start = str(edge["start_vertex"])
        end = str(edge["end_vertex"])
        adjacency.setdefault(start, []).append(str(edge["id"]))
        adjacency.setdefault(end, []).append(str(edge["id"]))

    visited: set[str] = set()
    loops: list[dict[str, Any]] = []
    for edge in selected:
        edge_id = str(edge["id"])
        if edge_id in visited:
            continue
        start_vertex = str(edge["start_vertex"])
        current_vertex = str(edge["end_vertex"])
        path = [{"edge": edge_id, "reversed": False}]
        visited.add(edge_id)
        guard = 0
        while current_vertex != start_vertex and guard <= len(selected) + 2:
            guard += 1
            candidates = [candidate for candidate in adjacency.get(current_vertex, []) if candidate not in visited]
            if not candidates:
                break
            next_id = candidates[0]
            next_edge = by_id[next_id]
            reversed_edge = str(next_edge["end_vertex"]) == current_vertex
            current_vertex = str(next_edge["start_vertex"] if reversed_edge else next_edge["end_vertex"])
            path.append({"edge": next_id, "reversed": reversed_edge})
            visited.add(next_id)
        closed = current_vertex == start_vertex
        points = _loop_points(path, by_id, vertices)
        area = polygon_area(points) if len(points) >= 3 else 0.0
        loops.append(
            {
                "id": str(len(loops) + 1),
                "edge_count": len(path),
                "closed": closed,
                "area": float(area),
                "orientation": "ccw" if area > 0.0 else "cw",
                "edges": path,
            }
        )
    return loops


def intersect_curves(a: Mapping[str, Any], b: Mapping[str, Any], *, target: float, tol: float) -> list[dict[str, Any]]:
    if _is_line(a) and _is_line(b):
        return _line_line_intersections(a, b, tol)
    if _is_line(a) and _is_circular(b):
        return _line_circle_intersections(a, b, line_first=True, tol=tol)
    if _is_circular(a) and _is_line(b):
        hits = _line_circle_intersections(b, a, line_first=False, tol=tol)
        return hits
    if _is_circular(a) and _is_circular(b):
        return _circle_circle_intersections(a, b, tol)
    return _numeric_curve_intersections(a, b, target=target, tol=tol)


def eval_curve(spec: Mapping[str, Any], t: float) -> Point2D:
    kind = str(spec.get("type", "")).lower()
    t = min(max(float(t), 0.0), 1.0)
    if kind == "line":
        start = _point(spec.get("start"))
        end = _point(spec.get("end"))
        return (start[0] * (1.0 - t) + end[0] * t, start[1] * (1.0 - t) + end[1] * t)
    if kind == "bezier":
        return _de_casteljau([_point(point) for point in spec.get("control_points", [])], t)
    if kind == "nurbs":
        return _eval_nurbs(spec, t)
    center = _point(spec.get("center", [0.0, 0.0]))
    start_deg = float(spec.get("start_angle", 0.0))
    end_deg = float(spec.get("end_angle", 360.0))
    if bool(spec.get("closed", False)) or kind in {"circle", "ellipse"}:
        end_deg = start_deg + 360.0
    while end_deg <= start_deg:
        end_deg += 360.0
    angle = math.radians(start_deg + (end_deg - start_deg) * t)
    if kind in {"ellipse", "elliptic_arc"}:
        rx = float(spec.get("rx", spec.get("radius", 0.0)))
        ry = float(spec.get("ry", rx))
        theta = math.radians(float(spec.get("rotation", 0.0)))
        ct = math.cos(theta)
        st = math.sin(theta)
        x = rx * math.cos(angle)
        y = ry * math.sin(angle)
        return (center[0] + x * ct - y * st, center[1] + x * st + y * ct)
    radius = float(spec.get("radius", 0.0))
    return (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))


def public_curve_spec(spec: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"type": str(spec.get("type", "")).lower()}
    for key, value in spec.items():
        if key == "type":
            continue
        if isinstance(value, tuple):
            out[key] = [float(value[0]), float(value[1])]
        elif isinstance(value, list):
            converted: list[Any] = []
            for item in value:
                if isinstance(item, tuple):
                    converted.append([float(item[0]), float(item[1])])
                else:
                    converted.append(item)
            out[key] = converted
        else:
            out[key] = value
    return out


def polygon_area(points: list[Point2D]) -> float:
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))


def _line_line_intersections(a: Mapping[str, Any], b: Mapping[str, Any], tol: float) -> list[dict[str, Any]]:
    p = _point(a["start"])
    p2 = _point(a["end"])
    q = _point(b["start"])
    q2 = _point(b["end"])
    rx = p2[0] - p[0]
    ry = p2[1] - p[1]
    sx = q2[0] - q[0]
    sy = q2[1] - q[1]
    denom = rx * sy - ry * sx
    if abs(denom) <= tol:
        return _collinear_line_overlap_intersections(p, p2, q, q2, rx, ry, sx, sy, tol)
    qpx = q[0] - p[0]
    qpy = q[1] - p[1]
    t = (qpx * sy - qpy * sx) / denom
    u = (qpx * ry - qpy * rx) / denom
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        tt = min(max(t, 0.0), 1.0)
        uu = min(max(u, 0.0), 1.0)
        point = eval_curve(a, tt)
        return [{"t_a": tt, "t_b": uu, "point": point, "method": "line_line", "residual": 0.0}]
    return []


def _collinear_line_overlap_intersections(
    p: Point2D,
    p2: Point2D,
    q: Point2D,
    q2: Point2D,
    rx: float,
    ry: float,
    sx: float,
    sy: float,
    tol: float,
) -> list[dict[str, Any]]:
    if abs((q[0] - p[0]) * ry - (q[1] - p[1]) * rx) > max(tol, 1.0e-12):
        return []
    rr = rx * rx + ry * ry
    ss = sx * sx + sy * sy
    if rr <= max(tol, 1.0e-30) or ss <= max(tol, 1.0e-30):
        return []
    ta0 = ((q[0] - p[0]) * rx + (q[1] - p[1]) * ry) / rr
    ta1 = ((q2[0] - p[0]) * rx + (q2[1] - p[1]) * ry) / rr
    lo = max(0.0, min(ta0, ta1))
    hi = min(1.0, max(ta0, ta1))
    if hi < lo - max(tol, 1.0e-12):
        return []
    hits: list[dict[str, Any]] = []
    for ta in (lo, hi):
        point = (p[0] + rx * ta, p[1] + ry * ta)
        tb = ((point[0] - q[0]) * sx + (point[1] - q[1]) * sy) / ss
        if -tol <= tb <= 1.0 + tol:
            _append_unique_hit(
                hits,
                {
                    "t_a": min(max(ta, 0.0), 1.0),
                    "t_b": min(max(tb, 0.0), 1.0),
                    "point": point,
                    "method": "line_line_overlap",
                    "residual": 0.0,
                },
                tol,
            )
    return hits


def _line_circle_intersections(
    line: Mapping[str, Any],
    circle: Mapping[str, Any],
    *,
    line_first: bool,
    tol: float,
) -> list[dict[str, Any]]:
    p0 = _point(line["start"])
    p1 = _point(line["end"])
    center = _point(circle["center"])
    radius = float(circle["radius"])
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    fx = p0[0] - center[0]
    fy = p0[1] - center[1]
    aa = dx * dx + dy * dy
    bb = 2.0 * (fx * dx + fy * dy)
    cc = fx * fx + fy * fy - radius * radius
    disc = bb * bb - 4.0 * aa * cc
    if aa <= tol or disc < -tol:
        return []
    roots = [-bb / (2.0 * aa)] if abs(disc) <= tol else [(-bb - math.sqrt(max(disc, 0.0))) / (2.0 * aa), (-bb + math.sqrt(max(disc, 0.0))) / (2.0 * aa)]
    hits: list[dict[str, Any]] = []
    for t_line in roots:
        if not -tol <= t_line <= 1.0 + tol:
            continue
        point = eval_curve(line, min(max(t_line, 0.0), 1.0))
        t_circle = _circular_t_from_point(circle, point)
        if t_circle is None:
            continue
        ta, tb = (t_line, t_circle) if line_first else (t_circle, t_line)
        _append_unique_hit(hits, {"t_a": min(max(ta, 0.0), 1.0), "t_b": min(max(tb, 0.0), 1.0), "point": point, "method": "line_circle", "residual": 0.0}, tol)
    return hits


def _circle_circle_intersections(a: Mapping[str, Any], b: Mapping[str, Any], tol: float) -> list[dict[str, Any]]:
    c0 = _point(a["center"])
    c1 = _point(b["center"])
    r0 = float(a["radius"])
    r1 = float(b["radius"])
    dx = c1[0] - c0[0]
    dy = c1[1] - c0[1]
    distance = math.hypot(dx, dy)
    if distance <= tol or distance > r0 + r1 + tol or distance < abs(r0 - r1) - tol:
        return []
    along = (r0 * r0 - r1 * r1 + distance * distance) / (2.0 * distance)
    height2 = r0 * r0 - along * along
    if height2 < -tol:
        return []
    height = math.sqrt(max(height2, 0.0))
    ux = dx / distance
    uy = dy / distance
    base = (c0[0] + along * ux, c0[1] + along * uy)
    points = [base] if height <= tol else [(base[0] - uy * height, base[1] + ux * height), (base[0] + uy * height, base[1] - ux * height)]
    hits: list[dict[str, Any]] = []
    for point in points:
        ta = _circular_t_from_point(a, point)
        tb = _circular_t_from_point(b, point)
        if ta is None or tb is None:
            continue
        _append_unique_hit(hits, {"t_a": ta, "t_b": tb, "point": point, "method": "circle_circle", "residual": 0.0}, tol)
    return hits


def _numeric_curve_intersections(a: Mapping[str, Any], b: Mapping[str, Any], *, target: float, tol: float) -> list[dict[str, Any]]:
    samples_a = _sample_curve(a, target)
    samples_b = _sample_curve(b, target)
    seeds: list[tuple[float, float]] = []
    for (ta0, pa0), (ta1, pa1) in zip(samples_a, samples_a[1:]):
        for (tb0, pb0), (tb1, pb1) in zip(samples_b, samples_b[1:]):
            hit = _segment_intersection(pa0, pa1, pb0, pb1, tol)
            if hit is not None:
                ua, ub, _x, _y = hit
                seeds.append((ta0 * (1.0 - ua) + ta1 * ua, tb0 * (1.0 - ub) + tb1 * ub))
                continue
            ua, ub, distance = _closest_segment_parameters(pa0, pa1, pb0, pb1)
            if distance <= max(target * 0.50, tol * 10.0, 1.0e-8):
                seeds.append((ta0 * (1.0 - ua) + ta1 * ua, tb0 * (1.0 - ub) + tb1 * ub))
    for ta, pa in samples_a:
        for tb, pb in samples_b:
            if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= max(tol * 10.0, 1.0e-8):
                seeds.append((ta, tb))
    seeds.extend(_near_curve_minimum_seeds(samples_a, samples_b, target, tol))

    hits: list[dict[str, Any]] = []
    for seed in seeds:
        refined = _refine_curve_intersection(a, b, seed, tol)
        if refined is not None:
            _append_unique_hit(hits, refined, tol)
    return hits


def _near_curve_minimum_seeds(
    samples_a: list[tuple[float, Point2D]],
    samples_b: list[tuple[float, Point2D]],
    target: float,
    tol: float,
) -> list[tuple[float, float]]:
    threshold = max(target * 0.08, tol * 10000.0, 1.0e-7)
    seeds: list[tuple[float, float]] = []
    best_by_a: list[tuple[float, float, float]] = []
    for ta, pa in samples_a:
        tb, distance = min(((tb, math.hypot(pa[0] - pb[0], pa[1] - pb[1])) for tb, pb in samples_b), key=lambda item: item[1])
        best_by_a.append((ta, tb, distance))
    for index, (ta, tb, distance) in enumerate(best_by_a):
        prev_distance = best_by_a[index - 1][2] if index > 0 else math.inf
        next_distance = best_by_a[index + 1][2] if index + 1 < len(best_by_a) else math.inf
        if distance <= threshold and distance <= prev_distance and distance <= next_distance:
            seeds.append((ta, tb))
    best_by_b: list[tuple[float, float, float]] = []
    for tb, pb in samples_b:
        ta, distance = min(((ta, math.hypot(pa[0] - pb[0], pa[1] - pb[1])) for ta, pa in samples_a), key=lambda item: item[1])
        best_by_b.append((ta, tb, distance))
    for index, (ta, tb, distance) in enumerate(best_by_b):
        prev_distance = best_by_b[index - 1][2] if index > 0 else math.inf
        next_distance = best_by_b[index + 1][2] if index + 1 < len(best_by_b) else math.inf
        if distance <= threshold and distance <= prev_distance and distance <= next_distance:
            seeds.append((ta, tb))
    return seeds


def _refine_curve_intersection(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    seed: tuple[float, float],
    tol: float,
) -> dict[str, Any] | None:
    if least_squares is None:
        pa = eval_curve(a, seed[0])
        pb = eval_curve(b, seed[1])
        residual = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        if residual <= max(tol * 10.0, 1.0e-8):
            return {"t_a": seed[0], "t_b": seed[1], "point": ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5), "method": "sampled_seed", "residual": residual}
        return None

    def residual(values: Sequence[float]) -> list[float]:
        pa = eval_curve(a, float(values[0]))
        pb = eval_curve(b, float(values[1]))
        return [pa[0] - pb[0], pa[1] - pb[1]]

    result = least_squares(
        residual,
        [min(max(seed[0], 0.0), 1.0), min(max(seed[1], 0.0), 1.0)],
        bounds=([0.0, 0.0], [1.0, 1.0]),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=80,
    )
    ta = float(result.x[0])
    tb = float(result.x[1])
    pa = eval_curve(a, ta)
    pb = eval_curve(b, tb)
    error = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
    if error > max(tol * 50.0, 1.0e-7):
        return None
    return {"t_a": ta, "t_b": tb, "point": ((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5), "method": "parametric_least_squares", "residual": error}


def _sample_curve(spec: Mapping[str, Any], target: float) -> list[tuple[float, Point2D]]:
    key = (_freeze_curve_value(spec), float(target))
    cached = _CURVE_SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached
    length = _curve_length(spec)
    count = max(12, min(512, int(math.ceil(length / max(target * 0.18, 1.0e-9)))))
    samples = [(index / count, eval_curve(spec, index / count)) for index in range(count + 1)]
    _cache_curve_samples(key, samples)
    return samples


def _curve_length(spec: Mapping[str, Any]) -> float:
    key = _freeze_curve_value(spec)
    cached = _CURVE_LENGTH_CACHE.get(key)
    if cached is not None:
        return cached
    if _is_line(spec):
        start = _point(spec["start"])
        end = _point(spec["end"])
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        _CURVE_LENGTH_CACHE[key] = length
        return length
    points = [eval_curve(spec, index / 32.0) for index in range(33)]
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    _CURVE_LENGTH_CACHE[key] = length
    return length


def _cache_curve_samples(key: tuple[Any, float], samples: list[tuple[float, Point2D]]) -> None:
    if key in _CURVE_SAMPLE_CACHE:
        _CURVE_SAMPLE_CACHE[key] = samples
        return
    if len(_CURVE_SAMPLE_CACHE) >= _CURVE_SAMPLE_CACHE_MAX:
        _CURVE_SAMPLE_CACHE.pop(next(iter(_CURVE_SAMPLE_CACHE)))
    _CURVE_SAMPLE_CACHE[key] = samples


def _freeze_curve_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze_curve_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_curve_value(item) for item in value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


def _segment_intersection(a: Point2D, b: Point2D, c: Point2D, d: Point2D, tol: float) -> tuple[float, float, float, float] | None:
    rx = b[0] - a[0]
    ry = b[1] - a[1]
    sx = d[0] - c[0]
    sy = d[1] - c[1]
    denom = rx * sy - ry * sx
    if abs(denom) <= tol:
        return None
    qpx = c[0] - a[0]
    qpy = c[1] - a[1]
    ta = (qpx * sy - qpy * sx) / denom
    tb = (qpx * ry - qpy * rx) / denom
    if -tol <= ta <= 1.0 + tol and -tol <= tb <= 1.0 + tol:
        return min(max(ta, 0.0), 1.0), min(max(tb, 0.0), 1.0), a[0] + ta * rx, a[1] + ta * ry
    return None


def _closest_segment_parameters(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> tuple[float, float, float]:
    candidates: list[tuple[float, float, float]] = []
    for ta in (0.0, 1.0):
        point = (a[0] * (1.0 - ta) + b[0] * ta, a[1] * (1.0 - ta) + b[1] * ta)
        tb, projected = _point_segment_parameter(point, c, d)
        candidates.append((ta, tb, math.hypot(point[0] - projected[0], point[1] - projected[1])))
    for tb in (0.0, 1.0):
        point = (c[0] * (1.0 - tb) + d[0] * tb, c[1] * (1.0 - tb) + d[1] * tb)
        ta, projected = _point_segment_parameter(point, a, b)
        candidates.append((ta, tb, math.hypot(point[0] - projected[0], point[1] - projected[1])))
    return min(candidates, key=lambda item: item[2])


def _point_segment_parameter(point: Point2D, a: Point2D, b: Point2D) -> tuple[float, Point2D]:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 <= 1.0e-30:
        return 0.0, a
    t = min(max(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2, 0.0), 1.0)
    return t, (a[0] + t * dx, a[1] + t * dy)


def _circular_t_from_point(spec: Mapping[str, Any], point: Point2D) -> float | None:
    center = _point(spec["center"])
    angle = math.degrees(math.atan2(point[1] - center[1], point[0] - center[0]))
    start = float(spec.get("start_angle", 0.0))
    end = float(spec.get("end_angle", 360.0))
    if bool(spec.get("closed", False)) or str(spec.get("type", "")).lower() == "circle":
        while angle < start:
            angle += 360.0
        return ((angle - start) % 360.0) / 360.0
    while end <= start:
        end += 360.0
    while angle < start:
        angle += 360.0
    if angle > end + 1.0e-8:
        return None
    return min(max((angle - start) / (end - start), 0.0), 1.0)


def _append_unique_intersection(items: list[dict[str, Any]], item: dict[str, Any], tol: float) -> bool:
    point = item["point"]
    for existing in items:
        if {existing["curve_a"], existing["curve_b"]} != {item["curve_a"], item["curve_b"]}:
            continue
        other = existing["point"]
        if math.hypot(point[0] - other[0], point[1] - other[1]) <= max(tol * 10.0, 1.0e-8):
            return False
    items.append(item)
    return True


def _append_unique_hit(items: list[dict[str, Any]], item: dict[str, Any], tol: float) -> None:
    point = item["point"]
    merge_tol = max(tol * 10000.0, 1.0e-4)
    for index, existing in enumerate(items):
        other = existing["point"]
        same_parameter = abs(float(existing["t_a"]) - float(item["t_a"])) <= merge_tol and abs(float(existing["t_b"]) - float(item["t_b"])) <= merge_tol
        same_point = math.hypot(point[0] - other[0], point[1] - other[1]) <= merge_tol
        if same_parameter or same_point:
            if float(item.get("residual", 0.0)) < float(existing.get("residual", 0.0)):
                items[index] = item
            return
    items.append(item)


def _hit_inside_overlap_spans(hit: Mapping[str, Any], spans: Sequence[Mapping[str, Any]], tol: float) -> bool:
    ta = float(hit.get("t_a", 0.0))
    tb = float(hit.get("t_b", 0.0))
    margin = max(tol * 100.0, 1.0e-8)
    for span in spans:
        a0 = float(span.get("t_a0", 0.0))
        a1 = float(span.get("t_a1", 0.0))
        b0 = float(span.get("t_b0", 0.0))
        b1 = float(span.get("t_b1", 0.0))
        alo, ahi = min(a0, a1), max(a0, a1)
        blo, bhi = min(b0, b1), max(b0, b1)
        if alo - margin <= ta <= ahi + margin and blo - margin <= tb <= bhi + margin:
            return True
    return False


def _unique_sorted_params(params: list[float], tol: float) -> list[float]:
    out: list[float] = []
    for value in sorted(min(max(float(param), 0.0), 1.0) for param in params):
        if not out or abs(value - out[-1]) > max(tol * 10.0, 1.0e-9):
            out.append(value)
    if out[0] > 0.0:
        out.insert(0, 0.0)
    if out[-1] < 1.0:
        out.append(1.0)
    return out


def _region_curve_sources(graph: Mapping[str, Any]) -> dict[int, list[Mapping[str, Any]]]:
    regions: dict[int, list[Mapping[str, Any]]] = {}
    for curve in sorted(graph.get("curves", []), key=lambda item: (int(item.get("region", 0)), int(item.get("curve", 0)))):
        regions.setdefault(int(curve.get("region", 0)), []).append(curve.get("source", {}))
    return regions


def _graph_curve_signature(graph: Mapping[str, Any]) -> tuple[Any, ...]:
    curves = graph.get("curves", [])
    if not isinstance(curves, Sequence):
        return (id(graph), 0)
    return (
        id(graph),
        tuple(
            (
                id(curve),
                int(curve.get("region", 0)) if isinstance(curve, Mapping) else 0,
                int(curve.get("curve", 0)) if isinstance(curve, Mapping) else 0,
                id(curve.get("source", None)) if isinstance(curve, Mapping) else 0,
            )
            for curve in curves
        ),
    )


def _cached_region_curve_sources(graph: Mapping[str, Any], signature: tuple[Any, ...]) -> dict[int, list[Mapping[str, Any]]]:
    cached = _GRAPH_REGION_CURVE_CACHE.get(signature)
    if cached is not None:
        return cached
    regions = _region_curve_sources(graph)
    _cache_bounded(_GRAPH_REGION_CURVE_CACHE, signature, regions, _GRAPH_REGION_CACHE_MAX)
    return regions


def _cached_sampled_graph_region_polygons(
    graph: Mapping[str, Any],
    signature: tuple[Any, ...],
    region_curves: Mapping[int, list[Mapping[str, Any]]],
    target: float,
    tol: float,
) -> dict[int, tuple[list[Point2D], tuple[float, float, float, float]]]:
    _ = graph
    key = (*signature, float(target), float(tol))
    cached = _GRAPH_REGION_POLYGON_CACHE.get(key)
    if cached is not None:
        return cached
    sampled = _sampled_region_polygons(region_curves, target, tol)
    _cache_bounded(_GRAPH_REGION_POLYGON_CACHE, key, sampled, _GRAPH_REGION_CACHE_MAX)
    return sampled


def _cache_bounded(cache: dict[Any, Any], key: Any, value: Any, max_entries: int) -> None:
    if key in cache:
        cache[key] = value
        return
    if len(cache) >= max_entries:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _regions_containing_point(point: Point2D, region_curves: Mapping[int, list[Mapping[str, Any]]], target: float, tol: float) -> list[int]:
    inside: list[int] = []
    for region, curves in region_curves.items():
        polygon = _sample_region_polygon(curves, target, tol)
        if len(polygon) >= 3 and _point_inside_bounds(point, _polygon_bounds(polygon), tol) and _point_in_polygon_winding(point, polygon, tol) != 0:
            inside.append(int(region))
    return inside


def _sampled_region_polygons(
    region_curves: Mapping[int, list[Mapping[str, Any]]],
    target: float,
    tol: float,
) -> dict[int, tuple[list[Point2D], tuple[float, float, float, float]]]:
    sampled: dict[int, tuple[list[Point2D], tuple[float, float, float, float]]] = {}
    for region, curves in region_curves.items():
        polygon = _sample_region_polygon(curves, target, tol)
        sampled[int(region)] = (polygon, _polygon_bounds(polygon))
    return sampled


def _regions_containing_sampled_point(
    point: Point2D,
    sampled_regions: Mapping[int, tuple[list[Point2D], tuple[float, float, float, float]]],
    tol: float,
) -> list[int]:
    inside: list[int] = []
    for region, (polygon, bounds) in sampled_regions.items():
        if len(polygon) >= 3 and _point_inside_bounds(point, bounds, tol) and _point_in_polygon_winding(point, polygon, tol) != 0:
            inside.append(int(region))
    return inside


def _sample_region_polygon(curves: Sequence[Mapping[str, Any]], target: float, tol: float) -> list[Point2D]:
    points: list[Point2D] = []
    for spec in curves:
        samples = _sample_curve(spec, target)
        for index, (_t, point) in enumerate(samples):
            if points and index == 0 and math.hypot(points[-1][0] - point[0], points[-1][1] - point[1]) <= tol:
                continue
            if points and math.hypot(points[-1][0] - point[0], points[-1][1] - point[1]) <= tol:
                continue
            points.append(point)
    if len(points) > 1 and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= tol:
        points.pop()
    return points


def _polygon_bounds(points: Sequence[Point2D]) -> tuple[float, float, float, float]:
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), max(xs), min(ys), max(ys)


def _point_inside_bounds(point: Point2D, bounds: tuple[float, float, float, float], tol: float) -> bool:
    x0, x1, y0, y1 = bounds
    return x0 - tol <= point[0] <= x1 + tol and y0 - tol <= point[1] <= y1 + tol


def _point_in_polygon_winding(point: Point2D, polygon: list[Point2D], tol: float) -> int:
    px, py = point
    winding = 0
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if _point_segment_distance(point, a, b) <= tol:
            return 1
        if a[1] <= py:
            if b[1] > py and _is_left(a, b, point) > 0.0:
                winding += 1
        else:
            if b[1] <= py and _is_left(a, b, point) < 0.0:
                winding -= 1
    return winding


def _is_left(a: Point2D, b: Point2D, point: Point2D) -> float:
    return (b[0] - a[0]) * (point[1] - a[1]) - (point[0] - a[0]) * (b[1] - a[1])


def _graph_offset(graph: Mapping[str, Any], target: float, tol: float) -> float:
    vertices = graph.get("vertices", {})
    coords = [_point(vertex.get("point", [0.0, 0.0])) for vertex in vertices.values()]
    if not coords:
        return max(target * 2.0e-2, tol * 1000.0, 1.0e-8)
    span = max(max(x for x, _y in coords) - min(x for x, _y in coords), max(y for _x, y in coords) - min(y for _x, y in coords), target, 1.0)
    return max(target * 2.0e-2, span * 1.0e-5, tol * 1000.0, 1.0e-8)


def _edge_tangent(edge: Mapping[str, Any]) -> Point2D:
    source = edge.get("source", {})
    t0 = float(edge.get("t0", 0.0))
    t1 = float(edge.get("t1", 1.0))
    span = max(t1 - t0, 1.0e-9)
    tm = 0.5 * (t0 + t1)
    dt = min(span * 0.20, 1.0e-4)
    pa = eval_curve(source, max(t0, tm - dt))
    pb = eval_curve(source, min(t1, tm + dt))
    tangent = (pb[0] - pa[0], pb[1] - pa[1])
    length = math.hypot(tangent[0], tangent[1])
    if length <= 1.0e-30:
        return (1.0, 0.0)
    return (tangent[0] / length, tangent[1] / length)


def _left_normal(tangent: Point2D) -> Point2D:
    return (-tangent[1], tangent[0])


def _boolean_value(operation: str, regions: list[int], region_count: int) -> bool:
    unique = set(regions)
    if operation == "union":
        return bool(unique)
    if operation == "intersection":
        return region_count > 0 and len(unique) == region_count
    if operation.startswith("difference_"):
        try:
            region = int(operation.split("_", 1)[1])
        except ValueError:
            return False
        return region in unique and not any(other != region for other in unique)
    return False


def _eval_boolean_ast(ast: tuple[Any, ...], regions: list[int]) -> bool:
    op = str(ast[0])
    if op == "region":
        return int(ast[1]) in set(regions)
    if op == "union":
        return _eval_boolean_ast(ast[1], regions) or _eval_boolean_ast(ast[2], regions)
    if op == "intersection":
        return _eval_boolean_ast(ast[1], regions) and _eval_boolean_ast(ast[2], regions)
    if op == "difference":
        return _eval_boolean_ast(ast[1], regions) and not _eval_boolean_ast(ast[2], regions)
    raise ValueError(f"Unsupported Boolean AST node: {op}")


def _tokenize_boolean_expression(text: str) -> list[str]:
    tokens: list[str] = []
    index = 0
    while index < len(text):
        ch = text[index]
        if ch.isspace():
            index += 1
            continue
        if ch in "()|+&*^-\\":
            tokens.append(ch)
            index += 1
            continue
        if ch.isalnum() or ch == "_":
            start = index
            index += 1
            while index < len(text) and (text[index].isalnum() or text[index] == "_"):
                index += 1
            tokens.append(text[start:index])
            continue
        raise ValueError(f"Unsupported Boolean expression character: {ch}")
    return tokens


class _BooleanExpressionParser:
    def __init__(self, tokens: list[str], region_count: int) -> None:
        self.tokens = tokens
        self.region_count = region_count
        self.index = 0

    def parse_expression(self) -> tuple[Any, ...]:
        return self.parse_union()

    def parse_union(self) -> tuple[Any, ...]:
        node = self.parse_difference()
        while self._peek_lower() in {"|", "+", "union", "or", "u"}:
            self.index += 1
            node = ("union", node, self.parse_difference())
        return node

    def parse_difference(self) -> tuple[Any, ...]:
        node = self.parse_intersection()
        while self._peek_lower() in {"-", "\\", "difference", "minus", "except"}:
            self.index += 1
            node = ("difference", node, self.parse_intersection())
        return node

    def parse_intersection(self) -> tuple[Any, ...]:
        node = self.parse_primary()
        while self._peek_lower() in {"&", "*", "^", "intersection", "intersect", "and"}:
            self.index += 1
            node = ("intersection", node, self.parse_primary())
        return node

    def parse_primary(self) -> tuple[Any, ...]:
        token = self._peek()
        if token is None:
            raise ValueError("Boolean expression ended unexpectedly")
        if token == "(":
            self.index += 1
            node = self.parse_expression()
            if self._peek() != ")":
                raise ValueError("Boolean expression is missing ')'")
            self.index += 1
            return node
        if token == ")":
            raise ValueError("Boolean expression has an unexpected ')'")
        self.index += 1
        return ("region", self._region_id(token))

    def expect_end(self) -> None:
        if self._peek() is not None:
            raise ValueError(f"Unexpected Boolean expression token: {self._peek()}")

    def _peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _peek_lower(self) -> str | None:
        token = self._peek()
        return token.lower() if token is not None else None

    def _region_id(self, token: str) -> int:
        lower = token.lower()
        if lower in {"union", "or", "intersection", "intersect", "and", "difference", "minus", "except"}:
            raise ValueError(f"Boolean operator '{token}' cannot be used as a region")
        if token.isdigit():
            region = int(token)
        elif lower.startswith("region_") and lower[7:].isdigit():
            region = int(lower[7:])
        elif lower.startswith("region") and lower[6:].isdigit():
            region = int(lower[6:])
        elif lower.startswith("r") and lower[1:].isdigit():
            region = int(lower[1:])
        elif len(token) == 1 and token.isalpha():
            region = ord(token.upper()) - ord("A") + 1
        else:
            raise ValueError(f"Unknown Boolean region token: {token}")
        if region < 1 or region > self.region_count:
            raise ValueError(f"Boolean region {region} is outside 1..{self.region_count}")
        return region


def _ast_region_ids(ast: tuple[Any, ...]) -> set[int]:
    if ast[0] == "region":
        return {int(ast[1])}
    return _ast_region_ids(ast[1]) | _ast_region_ids(ast[2])


def _format_boolean_ast(ast: tuple[Any, ...]) -> str:
    op = str(ast[0])
    if op == "region":
        return f"R{int(ast[1])}"
    symbol = {"union": "|", "intersection": "&", "difference": "-"}[op]
    return f"({_format_boolean_ast(ast[1])}{symbol}{_format_boolean_ast(ast[2])})"


def _classify_edge_operation(edge: dict[str, Any], operation: str, left_inside: bool, right_inside: bool) -> None:
    boundary = left_inside != right_inside
    edge[f"graph_{operation}_boundary"] = boundary
    if boundary:
        edge[f"graph_{operation}_interior_side"] = "left" if left_inside else "right"


def _choose_next_oriented_edge(current: Mapping[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    incoming = float(current["angle"]) + math.pi
    def turn(candidate: Mapping[str, Any]) -> float:
        value = float(candidate["angle"]) - incoming
        while value <= 0.0:
            value += 2.0 * math.pi
        return value
    return min(candidates, key=turn)


def _path_end_vertex(step: Mapping[str, Any], graph: Mapping[str, Any]) -> str:
    by_id = {str(edge["id"]): edge for edge in graph.get("edges", [])}
    edge = by_id[str(step["edge"])]
    return str(edge["start_vertex"] if bool(step.get("reversed", False)) else edge["end_vertex"])


def _loop_points(path: list[dict[str, Any]], edges: Mapping[str, Mapping[str, Any]], vertices: Mapping[str, Mapping[str, Any]]) -> list[Point2D]:
    points: list[Point2D] = []
    for index, step in enumerate(path):
        edge = edges[str(step["edge"])]
        if bool(step.get("reversed", False)):
            start = str(edge["end_vertex"])
            end = str(edge["start_vertex"])
        else:
            start = str(edge["start_vertex"])
            end = str(edge["end_vertex"])
        if index == 0:
            points.append(_point(vertices[start]["point"]))
        points.append(_point(vertices[end]["point"]))
    if len(points) > 1 and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= 1.0e-10:
        points.pop()
    return points


def _sample_oriented_edge(edge: Mapping[str, Any], target: float, reversed_edge: bool) -> list[Point2D]:
    source = edge.get("source", {})
    t0 = float(edge.get("t0", 0.0))
    t1 = float(edge.get("t1", 1.0))
    length = _trimmed_curve_length(source, t0, t1)
    count = max(1, min(512, int(math.ceil(length / max(target * 0.30, 1.0e-9)))))
    points: list[Point2D] = []
    for index in range(count + 1):
        alpha = index / count
        t = t1 * alpha + t0 * (1.0 - alpha) if not reversed_edge else t0 * alpha + t1 * (1.0 - alpha)
        points.append(eval_curve(source, t))
    return points


def _trimmed_curve_length(source: Mapping[str, Any], t0: float, t1: float) -> float:
    span = abs(t1 - t0)
    if span <= 1.0e-12:
        return 0.0
    count = max(4, min(64, int(math.ceil(_curve_length(source) * span / 0.05))))
    params = [t0 + (t1 - t0) * index / count for index in range(count + 1)]
    points = [eval_curve(source, param) for param in params]
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))


def _remove_near_collinear_points(points: list[Point2D], tol: float) -> list[Point2D]:
    cleaned = [point for point in points]
    changed = True
    while changed and len(cleaned) >= 4:
        changed = False
        next_points: list[Point2D] = []
        for index, point in enumerate(cleaned):
            prev_point = cleaned[index - 1]
            next_point = cleaned[(index + 1) % len(cleaned)]
            if math.hypot(prev_point[0] - next_point[0], prev_point[1] - next_point[1]) > tol and _point_segment_distance(point, prev_point, next_point) <= tol:
                changed = True
                continue
            if next_points and math.hypot(next_points[-1][0] - point[0], next_points[-1][1] - point[1]) <= tol:
                changed = True
                continue
            next_points.append(point)
        cleaned = next_points
    return cleaned


def _point_segment_distance(point: Point2D, a: Point2D, b: Point2D) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    length2 = dx * dx + dy * dy
    if length2 <= 1.0e-30:
        return math.hypot(point[0] - a[0], point[1] - a[1])
    t = min(max(((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / length2, 0.0), 1.0)
    px = a[0] + t * dx
    py = a[1] + t * dy
    return math.hypot(point[0] - px, point[1] - py)


def _annotate_overlap_edges(edges: list[dict[str, Any]], spans: list[dict[str, Any]], tol: float) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for span in spans:
        span_id = str(span.get("id", ""))
        a_edges = [edge for edge in edges if _edge_inside_overlap_span(edge, span, "a", tol)]
        b_edges = [edge for edge in edges if _edge_inside_overlap_span(edge, span, "b", tol)]
        span_pairs: list[dict[str, str]] = []
        for a_edge in a_edges:
            candidates = [edge for edge in b_edges if _edges_geometrically_coincident(a_edge, edge, max(tol * 10.0, 1.0e-8))]
            if not candidates:
                continue
            b_edge = min(candidates, key=lambda edge: _edge_midpoint_distance(a_edge, edge))
            pair_key = tuple(sorted((str(a_edge["id"]), str(b_edge["id"]))) + [span_id])
            if pair_key in seen:
                continue
            seen.add(pair_key)
            _mark_overlap_edge(a_edge, b_edge, span_id)
            _mark_overlap_edge(b_edge, a_edge, span_id)
            pair = {"span": span_id, "edge_a": str(a_edge["id"]), "edge_b": str(b_edge["id"])}
            span_pairs.append(pair)
            pairs.append(pair)
        span["edge_pairs"] = span_pairs
        span["edge_pair_count"] = len(span_pairs)
    return pairs


def _edge_inside_overlap_span(edge: Mapping[str, Any], span: Mapping[str, Any], suffix: str, tol: float) -> bool:
    curve_key = f"curve_{suffix}"
    if int(edge.get("curve_id", -1)) != int(span.get(curve_key, -2)):
        return False
    s0 = float(span.get(f"t_{suffix}0", 0.0))
    s1 = float(span.get(f"t_{suffix}1", 0.0))
    lo, hi = (min(s0, s1), max(s0, s1))
    e0 = float(edge.get("t0", 0.0))
    e1 = float(edge.get("t1", 0.0))
    elo, ehi = min(e0, e1), max(e0, e1)
    margin = max(tol * 100.0, 1.0e-8)
    return elo >= lo - margin and ehi <= hi + margin and ehi - elo > margin * 0.1


def _edges_geometrically_coincident(a: Mapping[str, Any], b: Mapping[str, Any], tol: float) -> bool:
    points_a = _edge_sample_points(a, 4)
    points_b = _edge_sample_points(b, 4)
    forward = max(math.hypot(pa[0] - pb[0], pa[1] - pb[1]) for pa, pb in zip(points_a, points_b))
    reverse = max(math.hypot(pa[0] - pb[0], pa[1] - pb[1]) for pa, pb in zip(points_a, reversed(points_b)))
    return min(forward, reverse) <= tol


def _edge_sample_points(edge: Mapping[str, Any], count: int) -> list[Point2D]:
    source = edge.get("source", {})
    t0 = float(edge.get("t0", 0.0))
    t1 = float(edge.get("t1", 1.0))
    return [eval_curve(source, t0 + (t1 - t0) * index / count) for index in range(count + 1)]


def _edge_midpoint_distance(a: Mapping[str, Any], b: Mapping[str, Any]) -> float:
    pa = _point(a.get("midpoint", eval_curve(a.get("source", {}), 0.5)))
    pb = _point(b.get("midpoint", eval_curve(b.get("source", {}), 0.5)))
    return math.hypot(pa[0] - pb[0], pa[1] - pb[1])


def _curve_pair_candidates(
    curves: Sequence[Mapping[str, Any]],
    target: float,
    margin: float,
) -> list[tuple[int, int]]:
    if len(curves) < 2:
        return []
    entries: list[tuple[float, float, float, float, int]] = []
    pad = max(float(margin), 0.0)
    for index, curve in enumerate(curves):
        x0, x1, y0, y1 = _curve_bounds(curve.get("source", {}), target)
        entries.append((x0 - pad, x1 + pad, y0 - pad, y1 + pad, index))
    entries.sort(key=lambda item: (item[0], item[4]))
    active: list[tuple[float, float, float, int]] = []
    pairs: set[tuple[int, int]] = set()
    for min_x, max_x, min_y, max_y, index in entries:
        active = [item for item in active if item[0] >= min_x]
        for _active_max_x, active_min_y, active_max_y, other_index in active:
            if active_max_y < min_y or max_y < active_min_y:
                continue
            a = min(other_index, index)
            b = max(other_index, index)
            if a != b:
                pairs.add((a, b))
        active.append((max_x, min_y, max_y, index))
    return sorted(pairs)


def _curve_bounds(spec: Mapping[str, Any], target: float) -> tuple[float, float, float, float]:
    kind = str(spec.get("type", "")).lower()
    if _is_line(spec):
        points = [_point(spec["start"]), _point(spec["end"])]
    elif kind == "bezier" and spec.get("control_points"):
        points = [_point(point) for point in spec.get("control_points", [])]
    elif kind == "nurbs" and spec.get("control_points"):
        points = [_point(point) for point in spec.get("control_points", [])]
    elif kind in {"arc", "circle", "ellipse", "elliptic_arc"} and "center" in spec:
        center = _point(spec.get("center"))
        rx = abs(float(spec.get("rx", spec.get("radius", 0.0))))
        ry = abs(float(spec.get("ry", rx)))
        radius = max(rx, ry)
        points = [(center[0] - radius, center[1] - radius), (center[0] + radius, center[1] + radius)]
    else:
        points = [point for _t, point in _sample_curve(spec, target)]
    if not points:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (min(xs), max(xs), min(ys), max(ys))


def _point_pair_candidates(points: Sequence[Point2D], radius: float) -> list[tuple[int, int]]:
    if len(points) < 2 or radius <= 0.0:
        return []
    cell = max(float(radius), 1.0e-30)
    bins: dict[tuple[int, int], list[int]] = {}
    pairs: list[tuple[int, int]] = []
    for index, point in enumerate(points):
        ix = math.floor(float(point[0]) / cell)
        iy = math.floor(float(point[1]) / cell)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other_index in bins.get((ix + dx, iy + dy), []):
                    pairs.append((other_index, index))
        bins.setdefault((ix, iy), []).append(index)
    return pairs


def _mark_overlap_edge(edge: dict[str, Any], other: Mapping[str, Any], span_id: str) -> None:
    edge.setdefault("overlap_span_ids", [])
    if span_id not in edge["overlap_span_ids"]:
        edge["overlap_span_ids"].append(span_id)
    edge.setdefault("coincident_edge_ids", [])
    other_id = str(other.get("id", ""))
    if other_id and other_id not in edge["coincident_edge_ids"]:
        edge["coincident_edge_ids"].append(other_id)
    edge.setdefault("coincident_regions", [])
    other_region = int(other.get("region", 0))
    if other_region and other_region not in edge["coincident_regions"]:
        edge["coincident_regions"].append(other_region)
    edge["overlap_role"] = "coincident_trim_edge"


def _graph_tolerance_diagnostics(
    curves: list[dict[str, Any]],
    intersections: list[dict[str, Any]],
    target: float,
    tolerance: float,
    snap_tolerance: float,
) -> dict[str, Any]:
    duplicate_tol = max(snap_tolerance * 5.0, target * 1.0e-6, tolerance * 1000.0, 1.0e-8)
    gap_tol = max(snap_tolerance * 10.0, target * 1.0e-5, tolerance * 10000.0, 1.0e-7)
    cluster_tol = max(snap_tolerance * 5.0, target * 1.0e-6, tolerance * 1000.0, 1.0e-8)
    duplicate_pairs: list[dict[str, Any]] = []
    tiny_gaps: list[dict[str, Any]] = []
    close_clusters: list[dict[str, Any]] = []
    overlapping_pairs: list[dict[str, Any]] = []
    tangent_pairs: list[dict[str, Any]] = []
    curve_candidate_pairs = _curve_pair_candidates(curves, target, max(duplicate_tol, cluster_tol, tolerance * 10000.0))

    for left_index, right_index in curve_candidate_pairs:
        left = curves[left_index]
        right = curves[right_index]
        if _curves_nearly_duplicate(left["source"], right["source"], duplicate_tol):
            duplicate_pairs.append(
                {
                    "curve_a": int(left["id"]),
                    "curve_b": int(right["id"]),
                    "region_a": int(left["region"]),
                    "region_b": int(right["region"]),
                }
            )
        overlap_spans = _curve_overlap_spans(left["source"], right["source"], target, max(duplicate_tol, tolerance * 10000.0))
        if overlap_spans:
            overlapping_pairs.append(
                {
                    "curve_a": int(left["id"]),
                    "curve_b": int(right["id"]),
                    "region_a": int(left["region"]),
                    "region_b": int(right["region"]),
                    "spans": overlap_spans[:4],
                }
            )
        closest = _closest_curve_pair(left["source"], right["source"], target, tolerance)
        if closest is not None and closest["distance"] <= cluster_tol and not overlap_spans:
            tangent_pairs.append(
                {
                    "curve_a": int(left["id"]),
                    "curve_b": int(right["id"]),
                    "region_a": int(left["region"]),
                    "region_b": int(right["region"]),
                    "distance": float(closest["distance"]),
                    "t_a": float(closest["t_a"]),
                    "t_b": float(closest["t_b"]),
                }
            )

    endpoints: list[dict[str, Any]] = []
    for curve in curves:
        for name, t in (("start", 0.0), ("end", 1.0)):
            point = eval_curve(curve["source"], t)
            endpoints.append({"curve": int(curve["id"]), "region": int(curve["region"]), "end": name, "point": point})
    endpoint_points = [_point(endpoint["point"]) for endpoint in endpoints]
    for left_index, right_index in _point_pair_candidates(endpoint_points, gap_tol):
        left = endpoints[left_index]
        right = endpoints[right_index]
        if left["curve"] == right["curve"]:
            continue
        distance = math.hypot(left["point"][0] - right["point"][0], left["point"][1] - right["point"][1])
        if snap_tolerance < distance <= gap_tol:
            tiny_gaps.append(
                {
                    "curve_a": left["curve"],
                    "curve_b": right["curve"],
                    "end_a": left["end"],
                    "end_b": right["end"],
                    "distance": float(distance),
                }
            )

    used: set[int] = set()
    intersection_points = [_point(hit["point"]) for hit in intersections]
    intersection_neighbors: dict[int, list[int]] = {}
    for left_index, right_index in _point_pair_candidates(intersection_points, cluster_tol):
        intersection_neighbors.setdefault(left_index, []).append(right_index)
    for index, hit in enumerate(intersections):
        if index in used:
            continue
        point = _point(hit["point"])
        members = [index]
        for other_index in intersection_neighbors.get(index, []):
            other = intersections[other_index]
            other_point = _point(other["point"])
            if math.hypot(point[0] - other_point[0], point[1] - other_point[1]) <= cluster_tol:
                members.append(other_index)
        if len(members) > 1:
            used.update(members)
            close_clusters.append(
                {
                    "point": [float(point[0]), float(point[1])],
                    "intersection_ids": [int(intersections[item]["id"]) for item in members],
                }
            )

    return {
        "snap_tolerance": float(snap_tolerance),
        "gap_tolerance": float(gap_tol),
        "duplicate_tolerance": float(duplicate_tol),
        "close_intersection_tolerance": float(cluster_tol),
        "curve_pair_candidate_count": len(curve_candidate_pairs),
        "duplicate_curve_pair_count": len(duplicate_pairs),
        "overlapping_curve_pair_count": len(overlapping_pairs),
        "tangent_close_pair_count": len(tangent_pairs),
        "tiny_gap_count": len(tiny_gaps),
        "close_intersection_cluster_count": len(close_clusters),
        "duplicate_curve_pairs": duplicate_pairs[:20],
        "overlapping_curve_pairs": overlapping_pairs[:20],
        "tangent_close_pairs": tangent_pairs[:20],
        "tiny_gaps": tiny_gaps[:20],
        "close_intersection_clusters": close_clusters[:20],
        "diagnostic_limit": 20,
    }


def _curves_nearly_duplicate(a: Mapping[str, Any], b: Mapping[str, Any], tol: float) -> bool:
    samples_a = [eval_curve(a, index / 32.0) for index in range(33)]
    samples_b = [eval_curve(b, index / 32.0) for index in range(33)]
    forward = max(math.hypot(pa[0] - pb[0], pa[1] - pb[1]) for pa, pb in zip(samples_a, samples_b))
    reverse = max(math.hypot(pa[0] - pb[0], pa[1] - pb[1]) for pa, pb in zip(samples_a, reversed(samples_b)))
    return min(forward, reverse) <= tol


def _curve_overlap_spans(a: Mapping[str, Any], b: Mapping[str, Any], target: float, tol: float) -> list[dict[str, float]]:
    samples_a = _sample_curve(a, target)
    samples_b = _sample_curve(b, target)
    close: list[tuple[float, float]] = []
    for ta, pa in samples_a:
        tb, distance = min(((tb, math.hypot(pa[0] - pb[0], pa[1] - pb[1])) for tb, pb in samples_b), key=lambda item: item[1])
        if distance <= tol:
            close.append((ta, tb))
    if len(close) < max(3, len(samples_a) // 8):
        return []
    spans: list[dict[str, float]] = []
    start_a, start_b = close[0]
    last_a, last_b = close[0]
    gap_tol = max(2.0 / max(len(samples_a) - 1, 1), 1.0e-9)
    for ta, tb in close[1:]:
        if ta - last_a <= gap_tol * 1.5:
            last_a, last_b = ta, tb
            continue
        if last_a - start_a > gap_tol:
            spans.append({"t_a0": float(start_a), "t_a1": float(last_a), "t_b0": float(start_b), "t_b1": float(last_b)})
        start_a, start_b = ta, tb
        last_a, last_b = ta, tb
    if last_a - start_a > gap_tol:
        spans.append({"t_a0": float(start_a), "t_a1": float(last_a), "t_b0": float(start_b), "t_b1": float(last_b)})
    return spans


def _refine_overlap_span(a: Mapping[str, Any], b: Mapping[str, Any], span: Mapping[str, float], tol: float) -> dict[str, float | str] | None:
    ta0 = min(max(float(span.get("t_a0", 0.0)), 0.0), 1.0)
    ta1 = min(max(float(span.get("t_a1", 0.0)), 0.0), 1.0)
    tb0 = min(max(float(span.get("t_b0", 0.0)), 0.0), 1.0)
    tb1 = min(max(float(span.get("t_b1", 0.0)), 0.0), 1.0)
    if abs(ta1 - ta0) <= 1.0e-12 or abs(tb1 - tb0) <= 1.0e-12:
        return None
    forward = math.hypot(eval_curve(a, ta0)[0] - eval_curve(b, tb0)[0], eval_curve(a, ta0)[1] - eval_curve(b, tb0)[1]) + math.hypot(
        eval_curve(a, ta1)[0] - eval_curve(b, tb1)[0], eval_curve(a, ta1)[1] - eval_curve(b, tb1)[1]
    )
    reverse = math.hypot(eval_curve(a, ta0)[0] - eval_curve(b, tb1)[0], eval_curve(a, ta0)[1] - eval_curve(b, tb1)[1]) + math.hypot(
        eval_curve(a, ta1)[0] - eval_curve(b, tb0)[0], eval_curve(a, ta1)[1] - eval_curve(b, tb0)[1]
    )
    if reverse < forward:
        tb0, tb1 = tb1, tb0
    tb0_refined, residual0 = _project_point_to_curve_parameter(b, eval_curve(a, ta0), tb0)
    tb1_refined, residual1 = _project_point_to_curve_parameter(b, eval_curve(a, ta1), tb1)
    ta0_refined, residual2 = _project_point_to_curve_parameter(a, eval_curve(b, tb0_refined), ta0)
    ta1_refined, residual3 = _project_point_to_curve_parameter(a, eval_curve(b, tb1_refined), ta1)
    max_residual = max(residual0, residual1, residual2, residual3)
    if max_residual > max(tol * 20.0, 1.0e-6):
        return None
    return {
        "t_a0": min(max(ta0_refined, 0.0), 1.0),
        "t_a1": min(max(ta1_refined, 0.0), 1.0),
        "t_b0": min(max(tb0_refined, 0.0), 1.0),
        "t_b1": min(max(tb1_refined, 0.0), 1.0),
        "max_residual": float(max_residual),
        "method": "projected_overlap_span",
    }


def _project_point_to_curve_parameter(spec: Mapping[str, Any], point: Point2D, seed: float) -> tuple[float, float]:
    seed = min(max(float(seed), 0.0), 1.0)
    if least_squares is None:
        candidates = [(index / 128.0, eval_curve(spec, index / 128.0)) for index in range(129)]
        t, candidate = min(candidates, key=lambda item: math.hypot(item[1][0] - point[0], item[1][1] - point[1]))
        return t, math.hypot(candidate[0] - point[0], candidate[1] - point[1])

    def residual(values: Sequence[float]) -> list[float]:
        candidate = eval_curve(spec, float(values[0]))
        return [candidate[0] - point[0], candidate[1] - point[1]]

    result = least_squares(
        residual,
        [seed],
        bounds=([0.0], [1.0]),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=60,
    )
    t = float(result.x[0])
    candidate = eval_curve(spec, t)
    return t, math.hypot(candidate[0] - point[0], candidate[1] - point[1])


def _closest_curve_pair(a: Mapping[str, Any], b: Mapping[str, Any], target: float, tol: float) -> dict[str, float] | None:
    samples_a = _sample_curve(a, target)
    samples_b = _sample_curve(b, target)
    seed_ta, seed_tb, seed_distance = min(
        ((ta, tb, math.hypot(pa[0] - pb[0], pa[1] - pb[1])) for ta, pa in samples_a for tb, pb in samples_b),
        key=lambda item: item[2],
    )
    if least_squares is None:
        return {"t_a": float(seed_ta), "t_b": float(seed_tb), "distance": float(seed_distance)}

    def residual(values: Sequence[float]) -> list[float]:
        pa = eval_curve(a, float(values[0]))
        pb = eval_curve(b, float(values[1]))
        return [pa[0] - pb[0], pa[1] - pb[1]]

    result = least_squares(
        residual,
        [seed_ta, seed_tb],
        bounds=([0.0, 0.0], [1.0, 1.0]),
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
        max_nfev=80,
    )
    ta = float(result.x[0])
    tb = float(result.x[1])
    pa = eval_curve(a, ta)
    pb = eval_curve(b, tb)
    return {"t_a": ta, "t_b": tb, "distance": float(math.hypot(pa[0] - pb[0], pa[1] - pb[1]))}


def _is_line(spec: Mapping[str, Any]) -> bool:
    return str(spec.get("type", "")).lower() == "line" and "start" in spec and "end" in spec


def _is_circular(spec: Mapping[str, Any]) -> bool:
    return str(spec.get("type", "")).lower() in {"arc", "circle"} and "center" in spec and "radius" in spec


def _point(raw: Any) -> Point2D:
    return (float(raw[0]), float(raw[1]))


def _de_casteljau(points: list[Point2D], t: float) -> Point2D:
    if not points:
        return (0.0, 0.0)
    work = [(float(x), float(y)) for x, y in points]
    while len(work) > 1:
        work = [
            (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)
            for a, b in zip(work, work[1:])
        ]
    return work[0]


def _eval_nurbs(spec: Mapping[str, Any], t: float) -> Point2D:
    controls = [_point(point) for point in spec.get("control_points", [])]
    if not controls:
        return (0.0, 0.0)
    degree = int(spec.get("degree", min(3, len(controls) - 1)))
    degree = max(1, min(degree, len(controls) - 1))
    weights = [float(value) for value in spec.get("weights", [])]
    if len(weights) != len(controls):
        weights = [1.0] * len(controls)
    knots = [float(value) for value in spec.get("knots", [])]
    if len(knots) != len(controls) + degree + 1:
        knots = _open_uniform_knot_vector(len(controls), degree)
    n = len(controls) - 1
    u_min = knots[degree]
    u_max = knots[n + 1]
    if u_max <= u_min:
        return controls[-1] if t >= 1.0 else controls[0]
    u = u_min + min(max(float(t), 0.0), 1.0) * (u_max - u_min)
    if u >= u_max:
        u = u_max - max((u_max - u_min) * 1.0e-12, 1.0e-14)
    numerator_x = 0.0
    numerator_y = 0.0
    denominator = 0.0
    basis_values = _nurbs_basis_values(degree, u, knots, len(controls))
    for index, (x, y) in enumerate(controls):
        basis = basis_values[index]
        weighted = basis * weights[index]
        numerator_x += weighted * x
        numerator_y += weighted * y
        denominator += weighted
    if abs(denominator) <= 1.0e-30:
        return controls[-1] if t >= 0.5 else controls[0]
    return numerator_x / denominator, numerator_y / denominator


def _nurbs_basis_values(degree: int, u: float, knots: list[float], count: int) -> list[float]:
    basis = [1.0 if knots[index] <= u < knots[index + 1] else 0.0 for index in range(count)]
    for p in range(1, degree + 1):
        previous = basis
        basis = [0.0] * count
        for index in range(count):
            left_den = knots[index + p] - knots[index]
            right_den = knots[index + p + 1] - knots[index + 1]
            left = 0.0
            right = 0.0
            if abs(left_den) > 1.0e-30:
                left = (u - knots[index]) / left_den * previous[index]
            if index + 1 < count and abs(right_den) > 1.0e-30:
                right = (knots[index + p + 1] - u) / right_den * previous[index + 1]
            basis[index] = left + right
    return basis


def _open_uniform_knot_vector(control_count: int, degree: int) -> list[float]:
    if control_count <= 1:
        return [0.0, 1.0]
    degree = max(1, min(degree, control_count - 1))
    interior = control_count - degree - 1
    knots = [0.0] * (degree + 1)
    if interior > 0:
        knots.extend(index / (interior + 1) for index in range(1, interior + 1))
    knots.extend([1.0] * (degree + 1))
    return knots


__all__ = [
    "build_analytic_curve_boolean_graph",
    "classify_graph_boolean_operations",
    "eval_curve",
    "intersect_curves",
    "operation_loop_polygons",
    "operation_loop_area",
    "parse_boolean_expression",
    "public_curve_spec",
    "regions_containing_point",
    "trace_oriented_boolean_loops",
    "trace_curve_boolean_loops",
]
