"""Small 2D mesh generation helpers used by the GUI."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .fem2d_types import njit


Point2D = tuple[float, float]

_CURVE_SPEC_LENGTH_CACHE_MAX = 1024
_CURVE_SPEC_SAMPLE_CACHE_MAX = 512
_CURVE_SPEC_LENGTH_CACHE: dict[tuple[Any, float], float] = {}
_CURVE_SPEC_SAMPLE_CACHE: dict[tuple[Any, float, float, int, int], list[Point2D]] = {}


def generate_quad_dominant_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    material: str,
    integration: str,
    requested_type: str = "QUAD4",
    refinements: list[Any] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
    curved_regions: list[Any] | None = None,
    boolean_expression: str | None = None,
) -> dict[str, Any]:
    x0, x1, y0, y1 = bbox
    if target <= 0.0:
        raise ValueError("target mesh size must be positive")
    holes = normalize_polygon_holes(polygon_holes or [])
    requested = requested_type.upper()
    prefer_quads = not requested.startswith("TRI")
    if prefer_quads:
        curved_paved = generate_parametric_curved_paving_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            tunnels=tunnels,
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=refinements or [],
            polygon_holes=holes,
            curved_regions=curved_regions or [],
            boolean_expression=boolean_expression,
        )
        if curved_paved is not None:
            return with_requested_element_order(curved_paved, requested_type)
        mapped = generate_mapped_quadrilateral_paving_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            tunnels=tunnels,
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=refinements or [],
            polygon_holes=holes,
        )
        if mapped is not None:
            return with_requested_element_order(mapped, requested_type)
        rectilinear = generate_rectilinear_polygon_paving_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            tunnels=tunnels,
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=refinements or [],
            polygon_holes=holes,
        )
        if rectilinear is not None:
            return with_requested_element_order(rectilinear, requested_type)
        ring_paved = generate_quadrilateral_ring_paving_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            tunnels=tunnels,
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=refinements or [],
            polygon_holes=holes,
        )
        if ring_paved is not None:
            return with_requested_element_order(ring_paved, requested_type)
        polygon_hole_paved = generate_quadrilateral_polygon_hole_paving_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            tunnels=tunnels,
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=refinements or [],
            polygon_holes=holes,
        )
        if polygon_hole_paved is not None:
            return with_requested_element_order(polygon_hole_paved, requested_type)
        circular_paved = generate_quadrilateral_circular_tunnel_paving_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            tunnels=tunnels,
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=refinements or [],
            polygon_holes=holes,
        )
        if circular_paved is not None:
            return with_requested_element_order(circular_paved, requested_type)
        polygon_paved = generate_simple_polygon_quad_paving_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            tunnels=tunnels,
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=refinements or [],
            polygon_holes=holes,
        )
        if polygon_paved is not None:
            return with_requested_element_order(polygon_paved, requested_type)
    x_coords, y_coords, layer_info = build_boundary_layer_coordinates(bbox, target, regions, tunnels, refinements or [], polygon_holes=holes)
    nx = len(x_coords) - 1
    ny = len(y_coords) - 1
    boundary_segments = geometry_boundary_segments(bbox, regions, tunnels, target, polygon_holes=holes)
    nodes: dict[str, list[float]] = {}
    node_ids: dict[tuple[int, int], str] = {}
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            nid = str(len(nodes) + 1)
            nodes[nid] = [x, y]
            node_ids[(i, j)] = nid
    nodes, projection_info = project_nodes_to_boundaries(nodes, bbox, regions, tunnels, target, polygon_holes=holes)

    elements: list[dict[str, Any]] = []
    region_sets: dict[str, list[str]] = {f"region_{i + 1}": [] for i in range(len(regions))}
    tunnel_excluded: list[str] = []
    quad_count = 0
    fallback_tri_count = 0

    for j in range(ny):
        for i in range(nx):
            n00 = node_ids[(i, j)]
            n10 = node_ids[(i + 1, j)]
            n11 = node_ids[(i + 1, j + 1)]
            n01 = node_ids[(i, j + 1)]
            corners = [
                tuple(nodes[n00]),
                tuple(nodes[n10]),
                tuple(nodes[n11]),
                tuple(nodes[n01]),
            ]
            if abs(polygon_area(corners)) <= 1.0e-18:
                tunnel_excluded.append(f"cell_{i}_{j}")
                continue
            cx = sum(p[0] for p in corners) * 0.25
            cy = sum(p[1] for p in corners) * 0.25
            if not inside_domain(cx, cy, regions, tunnels, polygon_holes=holes):
                tunnel_excluded.append(f"cell_{i}_{j}")
                continue
            cell_region = first_region_index(cx, cy, regions)
            all_corners_inside = all(inside_domain(x, y, regions, tunnels, polygon_holes=holes) or point_on_boundary((x, y), boundary_segments) for x, y in corners)
            quad_crosses = polygon_crosses_boundary(corners, boundary_segments)
            quad_shape = quad_quality(corners)
            if prefer_quads and all_corners_inside and not quad_crosses and quad_shape["min_angle_deg"] >= 20.0 and quad_shape["aspect_ratio"] <= 10.0:
                eid = str(len(elements) + 1)
                elements.append({"id": eid, "type": "QUAD4", "nodes": [n00, n10, n11, n01], "material": material, "integration": integration})
                quad_count += 1
                if cell_region is not None:
                    region_sets[f"region_{cell_region + 1}"].append(eid)
                continue
            tri_specs = [([corners[0], corners[1], corners[2]], [n00, n10, n11]), ([corners[0], corners[2], corners[3]], [n00, n11, n01])]
            for tri_points, tri_nodes in tri_specs:
                if abs(polygon_area(tri_points)) <= 1.0e-18:
                    continue
                tx = sum(p[0] for p in tri_points) / 3.0
                ty = sum(p[1] for p in tri_points) / 3.0
                if not inside_domain(tx, ty, regions, tunnels, polygon_holes=holes):
                    continue
                if polygon_crosses_boundary(tri_points, boundary_segments):
                    continue
                eid = str(len(elements) + 1)
                elements.append({"id": eid, "type": "TRI3", "nodes": tri_nodes, "material": material, "integration": integration})
                fallback_tri_count += 1
                tri_region = first_region_index(tx, ty, regions)
                if tri_region is not None:
                    region_sets[f"region_{tri_region + 1}"].append(eid)

    if not elements:
        raise ValueError("no valid mesh elements were generated")
    recombine_info = {"recombined_quad_count": 0, "remaining_tri_count": fallback_tri_count}
    if prefer_quads:
        elements, recombine_info = recombine_triangles_to_quads(
            nodes,
            elements,
            regions=regions,
            tunnels=tunnels,
            polygon_holes=holes,
            boundary_segments=boundary_segments,
        )
        nodes, elements, subdivision_info = subdivide_remaining_triangles_to_quads(
            nodes,
            elements,
            target=target,
            regions=regions,
            tunnels=tunnels,
            polygon_holes=holes,
            boundary_segments=boundary_segments,
        )
        region_sets = classify_element_regions(nodes, elements, regions)
        quad_count = sum(1 for element in elements if str(element.get("type", "")).upper() == "QUAD4")
        fallback_tri_count = sum(1 for element in elements if str(element.get("type", "")).upper() == "TRI3")
    else:
        subdivision_info = {"subdivided_triangle_count": 0, "subdivided_triangle_quad_count": 0}
    boundary_tol = max(target * 1.0e-8, 1.0e-10)
    raw_node_sets = {
        "left": [nid for nid, point in nodes.items() if abs(point[0] - x0) <= boundary_tol],
        "right": [nid for nid, point in nodes.items() if abs(point[0] - x1) <= boundary_tol],
        "bottom": [nid for nid, point in nodes.items() if abs(point[1] - y0) <= boundary_tol],
        "top": [nid for nid, point in nodes.items() if abs(point[1] - y1) <= boundary_tol],
        "all": list(nodes),
    }
    raw_node_sets.update(circular_tunnel_node_sets(nodes, tunnels, target))
    hole_boundary_segments = [segment for hole in holes for segment in zip(hole, hole[1:] + hole[:1])]
    if hole_boundary_segments:
        raw_node_sets["hole_boundary"] = node_ids_on_boundary(nodes, hole_boundary_segments)
    nodes, elements, node_sets = prune_unused_nodes(nodes, elements, raw_node_sets)
    element_sets = {"all": [element["id"] for element in elements]}
    element_sets.update({key: value for key, value in region_sets.items() if value})
    quality = mesh_quality_summary(nodes, elements)
    axis_aspect = axis_coordinate_aspect_ratio(x_coords, y_coords)
    active_tunnel_count = sum(1 for _cx, _cy, radius in tunnels if radius > 0.0)
    hole_shape_info = polygon_hole_shape_summary(holes)
    mode = "auto_quad_dominant" if prefer_quads else "auto_tri_grid"
    if prefer_quads and active_tunnel_count > 1:
        mode = "multi_circular_tunnel_boundary_grid"
    elif prefer_quads and hole_shape_info["non_star_concave_hole_count"] > 0:
        mode = "non_star_polygon_hole_boundary_grid"
    mesh = {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": element_sets,
        "mode": mode,
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": tunnel_excluded,
        "mesh_quality": {
            "quad_count": quad_count,
            "fallback_tri_count": fallback_tri_count,
            "quad_ratio": quad_count / max(quad_count + fallback_tri_count, 1),
            "aspect_ratio": axis_aspect,
            "recombined_quad_count": int(recombine_info.get("recombined_quad_count", 0)),
            "subdivided_triangle_count": int(subdivision_info.get("subdivided_triangle_count", 0)),
            "subdivided_triangle_quad_count": int(subdivision_info.get("subdivided_triangle_quad_count", 0)),
            "remaining_tri_count": fallback_tri_count,
            "min_quad_angle_deg": quality["min_quad_angle_deg"],
            "max_quad_aspect_ratio": quality["max_quad_aspect_ratio"],
            "boundary_projected_node_count": int(projection_info.get("projected_node_count", 0)),
            "boundary_snap_tolerance": float(projection_info.get("snap_tolerance", 0.0)),
            "boundary_layer_added_x_count": int(layer_info.get("boundary_layer_added_x_count", 0)),
            "boundary_layer_added_y_count": int(layer_info.get("boundary_layer_added_y_count", 0)),
            "oblique_boundary_layer_added_line_count": int(layer_info.get("oblique_boundary_layer_added_line_count", 0)),
            "curved_boundary_layer_added_line_count": int(layer_info.get("curved_boundary_layer_added_line_count", 0)),
            "curved_boundary_segment_count": int(layer_info.get("curved_boundary_segment_count", 0)),
            "local_refinement_added_line_count": int(layer_info.get("local_refinement_added_line_count", 0)),
            "split_line_constraint_count": int(layer_info.get("split_line_constraint_count", 0)),
            "split_line_added_line_count": int(layer_info.get("split_line_added_line_count", 0)),
            "split_line_local_size_added_line_count": int(layer_info.get("split_line_local_size_added_line_count", 0)),
            "x_line_count": len(x_coords),
            "y_line_count": len(y_coords),
            "circular_tunnel_count": active_tunnel_count,
            "circular_tunnel_boundary_node_count": len(node_sets.get("tunnel_boundary", [])),
            "circular_tunnel_boundary_node_counts": [len(node_sets.get(f"tunnel_boundary_{index + 1}", [])) for index, (_cx, _cy, radius) in enumerate(tunnels) if radius > 0.0],
            "circular_tunnel_min_clearance": minimum_circular_tunnel_clearance(tunnels),
            "polygon_hole_count": len(holes),
            "polygon_hole_boundary_segment_count": sum(len(hole) for hole in holes),
            "polygon_hole_boundary_node_count": len(node_sets.get("hole_boundary", [])),
            "polygon_hole_concave_count": int(hole_shape_info["concave_hole_count"]),
            "polygon_hole_non_star_count": int(hole_shape_info["non_star_concave_hole_count"]),
            "polygon_hole_star_shaped_count": int(hole_shape_info["star_shaped_hole_count"]),
            "polygon_hole_total_concave_vertex_count": int(hole_shape_info["total_concave_vertex_count"]),
        },
    }
    return with_requested_element_order(mesh, requested_type)


def generate_parametric_curved_paving_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    material: str,
    integration: str,
    requested_type: str = "QUAD4",
    refinements: list[Any] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
    curved_regions: list[Any] | None = None,
    boolean_expression: str | None = None,
) -> dict[str, Any] | None:
    if target <= 0.0 or requested_type.upper().startswith("TRI") or tunnels or refinements or not curved_regions:
        return None
    normalized = [raw for raw in curved_regions if isinstance(raw, Mapping)]
    if len(normalized) > 1:
        return _generate_multi_parametric_curved_outer_block_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            curved_regions=normalized,
            material=material,
            integration=integration,
            requested_type=requested_type,
            boolean_expression=boolean_expression,
        )
    if len(normalized) != 1:
        return None
    region = normalized[0]
    boundary_specs = _curve_chain_specs_from_region(region)
    curve_holes = _curve_hole_chains_from_region(region)
    raw_outer = _raw_curved_outer_points_from_region(region, boundary_specs, regions[0] if regions else [], target)
    if raw_outer and polygon_has_crossing_edges(raw_outer):
        return _generate_boolean_repaired_curved_outer_mesh(
            bbox=bbox,
            target=target,
            regions=regions,
            curved_regions=normalized,
            material=material,
            integration=integration,
            requested_type=requested_type,
            boolean_expression=boolean_expression,
        )
    if len(boundary_specs) == 4 and not curve_holes and not polygon_holes:
        return _generate_parametric_curved_quad_paving_mesh(
            bbox=bbox,
            target=target,
            boundary_specs=boundary_specs,
            material=material,
            integration=integration,
            requested_type=requested_type,
        )
    if len(curve_holes) == 1:
        outer = normalize_quadrilateral_region(regions[0]) if len(regions) == 1 else None
        if outer is None:
            outer = normalize_quadrilateral_region(_points_from_region_mapping(region))
        if outer is None and boundary_specs:
            sampled_outer = _curve_chain_sample_points(boundary_specs, target, closed=True, min_count=16)
            if len(sampled_outer) >= 4 and polygon_area(sampled_outer) > 1.0e-18 and not polygon_has_crossing_edges(sampled_outer):
                outer = sampled_outer
        if outer is None:
            return None
        return _generate_parametric_curve_hole_paving_mesh(
            bbox=bbox,
            target=target,
            outer=outer,
            hole_specs=curve_holes[0],
            material=material,
            integration=integration,
            requested_type=requested_type,
        )
    if len(curve_holes) > 1:
        outer = normalize_simple_polygon_region(regions[0]) if len(regions) == 1 else None
        if outer is None:
            outer = normalize_simple_polygon_region(_points_from_region_mapping(region))
        if outer is None and boundary_specs:
            sampled_outer = _curve_chain_sample_points(boundary_specs, target, closed=True, min_count=16)
            if len(sampled_outer) >= 4 and polygon_area(sampled_outer) > 1.0e-18 and not polygon_has_crossing_edges(sampled_outer):
                outer = sampled_outer
        if outer is None:
            return None
        return _generate_multi_parametric_curve_hole_boundary_grid_mesh(
            bbox=bbox,
            target=target,
            outer=outer,
            hole_specs_list=curve_holes,
            material=material,
            integration=integration,
            requested_type=requested_type,
        )
    return None


def _generate_parametric_curved_quad_paving_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    boundary_specs: list[dict[str, Any]],
    material: str,
    integration: str,
    requested_type: str,
) -> dict[str, Any] | None:
    bottom_spec, right_spec, top_spec_reversed, left_spec_reversed = boundary_specs
    sampled_boundary = _curve_chain_sample_points(boundary_specs, target, closed=True, min_count=16)
    if len(sampled_boundary) < 4 or polygon_area(sampled_boundary) <= 1.0e-18:
        return None

    bottom = lambda u: _eval_curve_spec(bottom_spec, u)
    right = lambda v: _eval_curve_spec(right_spec, v)
    top = lambda u: _eval_curve_spec(top_spec_reversed, 1.0 - u)
    left = lambda v: _eval_curve_spec(left_spec_reversed, 1.0 - v)
    p0 = bottom(0.0)
    p1 = bottom(1.0)
    p2 = top(1.0)
    p3 = top(0.0)
    if max(
        math.hypot(right(0.0)[0] - p1[0], right(0.0)[1] - p1[1]),
        math.hypot(right(1.0)[0] - p2[0], right(1.0)[1] - p2[1]),
        math.hypot(left(0.0)[0] - p0[0], left(0.0)[1] - p0[1]),
        math.hypot(left(1.0)[0] - p3[0], left(1.0)[1] - p3[1]),
    ) > max(target * 0.05, 1.0e-8):
        return None
    bottom_length = _curve_spec_length(bottom_spec, target)
    top_length = _curve_spec_length(top_spec_reversed, target)
    right_length = _curve_spec_length(right_spec, target)
    left_length = _curve_spec_length(left_spec_reversed, target)
    nu = max(1, int(math.ceil(max(bottom_length, top_length) / target)))
    nv = max(1, int(math.ceil(max(right_length, left_length) / target)))
    quant = max(target * 1.0e-10, 1.0e-12)
    nodes: dict[str, list[float]] = {}
    node_ids: dict[tuple[int, int], str] = {}

    def node_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        if key not in node_ids:
            nid = str(len(nodes) + 1)
            node_ids[key] = nid
            nodes[nid] = [float(point[0]), float(point[1])]
        return node_ids[key]

    def coons_point(u: float, v: float) -> Point2D:
        cb = bottom(u)
        ct = top(u)
        cl = left(v)
        cr = right(v)
        bilinear = bilinear_quad_point(p0, p1, p2, p3, u, v)
        return (
            (1.0 - v) * cb[0] + v * ct[0] + (1.0 - u) * cl[0] + u * cr[0] - bilinear[0],
            (1.0 - v) * cb[1] + v * ct[1] + (1.0 - u) * cl[1] + u * cr[1] - bilinear[1],
        )

    grid: dict[tuple[int, int], str] = {}
    for j in range(nv + 1):
        v = j / nv
        for i in range(nu + 1):
            u = i / nu
            grid[(i, j)] = node_id(coons_point(u, v))

    elements: list[dict[str, Any]] = []
    min_angle = math.inf
    max_aspect = 0.0
    for j in range(nv):
        for i in range(nu):
            quad_nodes = [grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]]
            pts = [tuple(nodes[nid]) for nid in quad_nodes]
            if polygon_area(pts) < 0.0:
                quad_nodes = list(reversed(quad_nodes))
                pts = [tuple(nodes[nid]) for nid in quad_nodes]
            quality = quad_quality(pts)
            if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 5.0 or quality["aspect_ratio"] > 80.0:
                return None
            min_angle = min(min_angle, quality["min_angle_deg"])
            max_aspect = max(max_aspect, quality["aspect_ratio"])
            eid = str(len(elements) + 1)
            elements.append({"id": eid, "type": "QUAD4", "nodes": quad_nodes, "material": material, "integration": integration, "source": "parametric_curved_quad_paving"})
    if not elements:
        return None

    x0, x1, y0, y1 = bbox
    node_sets = {
        "left": [grid[(0, j)] for j in range(nv + 1)],
        "right": [grid[(nu, j)] for j in range(nv + 1)],
        "bottom": [grid[(i, 0)] for i in range(nu + 1)],
        "top": [grid[(i, nv)] for i in range(nu + 1)],
        "boundary": list(dict.fromkeys([*[grid[(i, 0)] for i in range(nu + 1)], *[grid[(nu, j)] for j in range(1, nv + 1)], *[grid[(i, nv)] for i in range(nu - 1, -1, -1)], *[grid[(0, j)] for j in range(nv - 1, 0, -1)]])),
        "all": list(nodes),
    }
    node_sets.update(
        {
            "bbox_left": [nid for nid, point in nodes.items() if abs(point[0] - x0) <= quant],
            "bbox_right": [nid for nid, point in nodes.items() if abs(point[0] - x1) <= quant],
            "bbox_bottom": [nid for nid, point in nodes.items() if abs(point[1] - y0) <= quant],
            "bbox_top": [nid for nid, point in nodes.items() if abs(point[1] - y1) <= quant],
        }
    )
    node_sets = {name: ids for name, ids in node_sets.items() if ids}
    element_ids = [element["id"] for element in elements]
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": {"all": element_ids, "region_1": element_ids.copy()},
        "mode": "parametric_curved_quad_paving",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": len(elements),
            "fallback_tri_count": 0,
            "quad_ratio": 1.0,
            "aspect_ratio": max_aspect,
            "recombined_quad_count": 0,
            "remaining_tri_count": 0,
            "min_quad_angle_deg": min_angle,
            "max_quad_aspect_ratio": max_aspect,
            "boundary_projected_node_count": 0,
            "boundary_snap_tolerance": 0.0,
            "boundary_layer_added_x_count": 0,
            "boundary_layer_added_y_count": 0,
            "oblique_boundary_layer_added_line_count": 0,
            "curved_boundary_layer_added_line_count": nu + nv,
            "curved_boundary_segment_count": 2 * (nu + nv),
            "local_refinement_added_line_count": 0,
            "x_line_count": nu + 1,
            "y_line_count": nv + 1,
            "parametric_curved_quad_paving_quad_count": len(elements),
            "parametric_curved_quad_paving_curve_count": len(boundary_specs),
            "parametric_curved_quad_paving_u_divisions": nu,
            "parametric_curved_quad_paving_v_divisions": nv,
            "curve_parameter_retained": True,
            "polygon_hole_count": 0,
            "polygon_hole_boundary_segment_count": 0,
        },
    }


def _generate_parametric_curve_hole_paving_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    outer: list[Point2D],
    hole_specs: list[dict[str, Any]],
    material: str,
    integration: str,
    requested_type: str,
) -> dict[str, Any] | None:
    length = _curve_chain_length(hole_specs, target)
    count = max(8, int(math.ceil(length / max(target, 1.0e-9))))
    inner = _curve_chain_sample_points(hole_specs, target, closed=True, min_count=count)
    if len(inner) < 8 or abs(polygon_area(inner)) <= 1.0e-18:
        return None
    if polygon_area(inner) < 0.0:
        inner.reverse()
    if not all(point_in_polygon(x, y, outer) for x, y in inner):
        return None
    center = _curve_chain_center(hole_specs) or polygon_centroid(inner)
    if not point_in_polygon(center[0], center[1], inner):
        center = polygon_centroid(inner)
    outer_points: list[Point2D] = []
    for point in inner:
        direction = (point[0] - center[0], point[1] - center[1])
        outer_point = ray_polygon_intersection(center, direction, outer)
        if outer_point is None:
            return None
        outer_points.append(outer_point)
    radial_divisions = max(1, int(math.ceil(max(math.hypot(outer_points[i][0] - inner[i][0], outer_points[i][1] - inner[i][1]) for i in range(len(inner))) / target)))
    quant = max(target * 1.0e-10, 1.0e-12)
    nodes: dict[str, list[float]] = {}
    node_ids: dict[tuple[int, int], str] = {}

    def node_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        if key not in node_ids:
            nid = str(len(nodes) + 1)
            node_ids[key] = nid
            nodes[nid] = [float(point[0]), float(point[1])]
        return node_ids[key]

    elements: list[dict[str, Any]] = []
    patch_sets: dict[str, list[str]] = {}
    min_angle = math.inf
    max_aspect = 0.0
    outer_segments = list(zip(outer, outer[1:] + outer[:1]))
    inner_segments = list(zip(inner, inner[1:] + inner[:1]))
    patch_count = len(inner)
    for side in range(patch_count):
        p0 = outer_points[side]
        p1 = outer_points[(side + 1) % patch_count]
        p2 = inner[(side + 1) % patch_count]
        p3 = inner[side]
        patch_key = f"parametric_curve_hole_patch_{side + 1}"
        patch_sets[patch_key] = []
        patch_node_ids: dict[tuple[int, int], str] = {}
        for j in range(radial_divisions + 1):
            v = j / radial_divisions
            for i in range(2):
                u = float(i)
                patch_node_ids[(i, j)] = node_id(bilinear_quad_point(p0, p1, p2, p3, u, v))
        for j in range(radial_divisions):
            quad_nodes = [patch_node_ids[(0, j)], patch_node_ids[(1, j)], patch_node_ids[(1, j + 1)], patch_node_ids[(0, j + 1)]]
            pts = [tuple(nodes[nid]) for nid in quad_nodes]
            if polygon_area(pts) < 0.0:
                quad_nodes = list(reversed(quad_nodes))
                pts = [tuple(nodes[nid]) for nid in quad_nodes]
            quality = quad_quality(pts)
            if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 5.0 or quality["aspect_ratio"] > 100.0:
                return None
            if not quad_samples_respect_domain(pts, outer, inner, outer_segments, inner_segments):
                return None
            min_angle = min(min_angle, quality["min_angle_deg"])
            max_aspect = max(max_aspect, quality["aspect_ratio"])
            eid = str(len(elements) + 1)
            elements.append({"id": eid, "type": "QUAD4", "nodes": quad_nodes, "material": material, "integration": integration, "source": "parametric_curve_hole_paving", "parametric_curve_hole_patch": side + 1})
            patch_sets[patch_key].append(eid)
    if not elements:
        return None

    x0, x1, y0, y1 = bbox
    element_ids = [element["id"] for element in elements]
    outer_nodes = node_ids_on_boundary(nodes, outer_segments)
    hole_nodes = node_ids_on_boundary(nodes, inner_segments)
    node_sets = {
        "left": [nid for nid, point in nodes.items() if abs(point[0] - x0) <= quant],
        "right": [nid for nid, point in nodes.items() if abs(point[0] - x1) <= quant],
        "bottom": [nid for nid, point in nodes.items() if abs(point[1] - y0) <= quant],
        "top": [nid for nid, point in nodes.items() if abs(point[1] - y1) <= quant],
        "outer_boundary": outer_nodes,
        "hole_boundary": hole_nodes,
        "boundary": list(dict.fromkeys([*outer_nodes, *hole_nodes])),
        "all": list(nodes),
    }
    node_sets = {name: ids for name, ids in node_sets.items() if ids}
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": {"all": element_ids, "region_1": element_ids.copy(), **{key: value for key, value in patch_sets.items() if value}},
        "mode": "parametric_curve_hole_paving",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": len(elements),
            "fallback_tri_count": 0,
            "quad_ratio": 1.0,
            "aspect_ratio": max_aspect,
            "recombined_quad_count": 0,
            "remaining_tri_count": 0,
            "min_quad_angle_deg": min_angle,
            "max_quad_aspect_ratio": max_aspect,
            "boundary_projected_node_count": 0,
            "boundary_snap_tolerance": 0.0,
            "boundary_layer_added_x_count": 0,
            "boundary_layer_added_y_count": 0,
            "oblique_boundary_layer_added_line_count": 0,
            "curved_boundary_layer_added_line_count": patch_count,
            "curved_boundary_segment_count": patch_count,
            "local_refinement_added_line_count": 0,
            "x_line_count": len({point[0] for point in [*outer, *inner]}),
            "y_line_count": len({point[1] for point in [*outer, *inner]}),
            "parametric_curve_hole_paving_quad_count": len(elements),
            "parametric_curve_hole_paving_patch_count": patch_count,
            "parametric_curve_hole_paving_radial_divisions": radial_divisions,
            "parametric_curve_hole_paving_curve_count": len(hole_specs),
            "curve_parameter_retained": True,
            "polygon_hole_count": 1,
            "polygon_hole_boundary_segment_count": patch_count,
        },
    }


def _generate_multi_parametric_curve_hole_boundary_grid_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    outer: list[Point2D],
    hole_specs_list: list[list[dict[str, Any]]],
    material: str,
    integration: str,
    requested_type: str,
) -> dict[str, Any] | None:
    if len(hole_specs_list) < 2:
        return None
    sampled_holes: list[list[Point2D]] = []
    segment_counts: list[int] = []
    for specs in hole_specs_list:
        length = _curve_chain_length(specs, target)
        count = max(8, int(math.ceil(length / max(target, 1.0e-9))))
        hole = _curve_chain_sample_points(specs, target, closed=True, min_count=count)
        if len(hole) < 8 or abs(polygon_area(hole)) <= 1.0e-18:
            return None
        if polygon_area(hole) < 0.0:
            hole.reverse()
        if not all(point_in_polygon(x, y, outer) for x, y in hole):
            return None
        sampled_holes.append(hole)
        segment_counts.append(len(hole))
    if polygons_any_overlap(sampled_holes):
        return None
    try:
        mesh = generate_quad_dominant_mesh(
            bbox=bbox,
            target=target,
            regions=[outer],
            tunnels=[],
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=[],
            polygon_holes=sampled_holes,
            curved_regions=[],
        )
    except ValueError:
        return None
    nodes: dict[str, list[float]] = mesh["nodes"]
    elements: list[dict[str, Any]] = mesh["elements"]
    node_sets: dict[str, list[str]] = dict(mesh.get("node_sets", {}))
    boundary_candidates = list(dict.fromkeys(node_sets.get("hole_boundary", list(nodes))))
    combined: list[str] = []
    counts: list[int] = []
    projected_count = 0
    for index, (specs, sampled) in enumerate(zip(hole_specs_list, sampled_holes), start=1):
        ids = _curve_hole_boundary_node_ids(nodes, boundary_candidates, sampled, target)
        if not ids:
            ids = _curve_hole_boundary_node_ids(nodes, list(nodes), sampled, target)
        projected_count += _snap_node_ids_to_curve_chain(nodes, ids, specs, target)
        key = f"parametric_hole_boundary_{index}"
        node_sets[key] = ids
        combined.extend(ids)
        counts.append(len(ids))
    if combined:
        node_sets["hole_boundary"] = list(dict.fromkeys(combined))
    mesh["node_sets"] = {name: ids for name, ids in node_sets.items() if ids}
    mesh["mode"] = "multi_parametric_curve_hole_boundary_grid"
    quality = mesh_quality_summary(nodes, elements)
    mesh_quality = mesh.setdefault("mesh_quality", {})
    mesh_quality.update(
        {
            "min_quad_angle_deg": quality["min_quad_angle_deg"],
            "max_quad_aspect_ratio": quality["max_quad_aspect_ratio"],
            "curve_parameter_retained": True,
            "parametric_curve_hole_count": len(hole_specs_list),
            "parametric_curve_hole_boundary_grid_quad_count": sum(1 for element in elements if str(element.get("type", "")).upper() == "QUAD4"),
            "parametric_curve_hole_boundary_node_count": len(mesh["node_sets"].get("hole_boundary", [])),
            "parametric_curve_hole_boundary_node_counts": counts,
            "parametric_curve_hole_segment_counts": segment_counts,
            "parametric_curve_hole_projected_node_count": projected_count,
            "polygon_hole_count": len(sampled_holes),
            "polygon_hole_boundary_segment_count": sum(segment_counts),
        }
    )
    return mesh


def _generate_multi_parametric_curved_outer_block_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    curved_regions: list[Mapping[str, Any]],
    material: str,
    integration: str,
    requested_type: str,
    boolean_expression: str | None = None,
) -> dict[str, Any] | None:
    block_meshes: list[dict[str, Any]] = []
    block_outers: list[list[Point2D]] = []
    for index, region in enumerate(curved_regions):
        boundary_specs = _curve_chain_specs_from_region(region)
        curve_holes = _curve_hole_chains_from_region(region)
        fallback = regions[index] if index < len(regions) else []
        raw_outer = _raw_curved_outer_points_from_region(region, boundary_specs, fallback, target)
        if raw_outer and polygon_has_crossing_edges(raw_outer):
            return _generate_boolean_repaired_curved_outer_mesh(
                bbox=bbox,
                target=target,
                regions=regions,
                curved_regions=curved_regions,
                material=material,
                integration=integration,
                requested_type=requested_type,
                boolean_expression=boolean_expression,
            )
        outer = _curved_outer_polygon_from_region(region, boundary_specs, fallback, target)
        if outer is None:
            return _generate_boolean_repaired_curved_outer_mesh(
                bbox=bbox,
                target=target,
                regions=regions,
                curved_regions=curved_regions,
                material=material,
                integration=integration,
                requested_type=requested_type,
                boolean_expression=boolean_expression,
            )
        if any(polygons_have_positive_area_overlap(outer, previous) for previous in block_outers):
            return _generate_boolean_repaired_curved_outer_mesh(
                bbox=bbox,
                target=target,
                regions=regions,
                curved_regions=curved_regions,
                material=material,
                integration=integration,
                requested_type=requested_type,
                boolean_expression=boolean_expression,
            )
        block = _generate_single_parametric_curved_outer_block_mesh(
            bbox=bbox,
            target=target,
            outer=outer,
            boundary_specs=boundary_specs,
            curve_holes=curve_holes,
            material=material,
            integration=integration,
            requested_type=requested_type,
        )
        if block is None:
            return None
        block_meshes.append(block)
        block_outers.append(outer)
    if not block_meshes:
        return None
    return _merge_parametric_curved_outer_blocks(block_meshes, target=target)


def _generate_single_parametric_curved_outer_block_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    outer: list[Point2D],
    boundary_specs: list[dict[str, Any]],
    curve_holes: list[list[dict[str, Any]]],
    material: str,
    integration: str,
    requested_type: str,
) -> dict[str, Any] | None:
    block: dict[str, Any] | None = None
    if len(boundary_specs) == 4 and not curve_holes:
        block = _generate_parametric_curved_quad_paving_mesh(
            bbox=bbox,
            target=target,
            boundary_specs=boundary_specs,
            material=material,
            integration=integration,
            requested_type=requested_type,
        )
    elif len(curve_holes) == 1:
        block = _generate_parametric_curve_hole_paving_mesh(
            bbox=bbox,
            target=target,
            outer=outer,
            hole_specs=curve_holes[0],
            material=material,
            integration=integration,
            requested_type=requested_type,
        )
    elif len(curve_holes) > 1:
        block = _generate_multi_parametric_curve_hole_boundary_grid_mesh(
            bbox=bbox,
            target=target,
            outer=outer,
            hole_specs_list=curve_holes,
            material=material,
            integration=integration,
            requested_type=requested_type,
        )
    if block is None:
        block = _generate_parametric_curved_outer_boundary_grid_mesh(
            bbox=bbox,
            target=target,
            outer=outer,
            boundary_specs=boundary_specs,
            curve_holes=curve_holes,
            material=material,
            integration=integration,
            requested_type=requested_type,
        )
    if block is None:
        return None
    block.setdefault("mesh_quality", {})["parametric_curved_outer_curve_count"] = len(boundary_specs)
    return block


def _generate_parametric_curved_outer_boundary_grid_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    outer: list[Point2D],
    boundary_specs: list[dict[str, Any]],
    curve_holes: list[list[dict[str, Any]]],
    material: str,
    integration: str,
    requested_type: str,
) -> dict[str, Any] | None:
    sampled_holes: list[list[Point2D]] = []
    for specs in curve_holes:
        length = _curve_chain_length(specs, target)
        count = max(8, int(math.ceil(length / max(target, 1.0e-9))))
        hole = _curve_chain_sample_points(specs, target, closed=True, min_count=count)
        if len(hole) < 8 or abs(polygon_area(hole)) <= 1.0e-18:
            return None
        if polygon_area(hole) < 0.0:
            hole.reverse()
        if not all(point_in_polygon(x, y, outer) for x, y in hole):
            return None
        sampled_holes.append(hole)
    if polygons_any_overlap(sampled_holes):
        return None
    try:
        mesh = generate_quad_dominant_mesh(
            bbox=bbox,
            target=target,
            regions=[outer],
            tunnels=[],
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=[],
            polygon_holes=sampled_holes,
            curved_regions=[],
        )
    except ValueError:
        return None
    nodes: dict[str, list[float]] = mesh["nodes"]
    node_sets: dict[str, list[str]] = dict(mesh.get("node_sets", {}))
    projected_count = 0
    if boundary_specs:
        candidates = list(dict.fromkeys(node_sets.get("boundary", list(nodes))))
        outer_ids = _curve_hole_boundary_node_ids(nodes, candidates, outer, target)
        if not outer_ids:
            outer_ids = _curve_hole_boundary_node_ids(nodes, list(nodes), outer, target)
        projected_count += _snap_node_ids_to_curve_chain(nodes, outer_ids, boundary_specs, target)
        node_sets["outer_boundary"] = outer_ids
    combined_holes: list[str] = []
    for index, (specs, sampled) in enumerate(zip(curve_holes, sampled_holes), start=1):
        candidates = list(dict.fromkeys(node_sets.get("hole_boundary", list(nodes))))
        ids = _curve_hole_boundary_node_ids(nodes, candidates, sampled, target)
        if not ids:
            ids = _curve_hole_boundary_node_ids(nodes, list(nodes), sampled, target)
        projected_count += _snap_node_ids_to_curve_chain(nodes, ids, specs, target)
        node_sets[f"parametric_hole_boundary_{index}"] = ids
        combined_holes.extend(ids)
    if combined_holes:
        node_sets["hole_boundary"] = list(dict.fromkeys(combined_holes))
    if node_sets.get("outer_boundary") and node_sets.get("hole_boundary"):
        node_sets["boundary"] = list(dict.fromkeys([*node_sets["outer_boundary"], *node_sets["hole_boundary"]]))
    elif node_sets.get("outer_boundary"):
        node_sets["boundary"] = list(dict.fromkeys(node_sets["outer_boundary"]))
    mesh["node_sets"] = {name: ids for name, ids in node_sets.items() if ids}
    mesh["mode"] = "parametric_curved_outer_boundary_grid"
    quality = mesh_quality_summary(nodes, mesh["elements"])
    mesh_quality = mesh.setdefault("mesh_quality", {})
    mesh_quality.update(
        {
            "min_quad_angle_deg": quality["min_quad_angle_deg"],
            "max_quad_aspect_ratio": quality["max_quad_aspect_ratio"],
            "curve_parameter_retained": True,
            "parametric_curved_outer_boundary_grid_quad_count": sum(1 for element in mesh["elements"] if str(element.get("type", "")).upper() == "QUAD4"),
            "parametric_curved_outer_boundary_node_count": len(mesh["node_sets"].get("outer_boundary", [])),
            "parametric_curved_outer_curve_count": len(boundary_specs),
            "parametric_curved_outer_projected_node_count": projected_count,
        }
    )
    return mesh


def _merge_parametric_curved_outer_blocks(block_meshes: list[dict[str, Any]], *, target: float) -> dict[str, Any]:
    nodes: dict[str, list[float]] = {}
    coordinate_ids: dict[tuple[int, int], str] = {}
    elements: list[dict[str, Any]] = []
    node_sets: dict[str, list[str]] = {"boundary": [], "all": []}
    element_sets: dict[str, list[str]] = {"all": []}
    boundary_counts: list[int] = []
    duplicate_node_count = 0
    projected_total = 0
    source_modes: list[str] = []
    curve_count = 0
    hole_count = 0
    quant = max(target * 1.0e-8, 1.0e-10)

    def merged_node_id(point: Any) -> tuple[str, bool]:
        x = float(point[0])
        y = float(point[1])
        key = (int(round(x / quant)), int(round(y / quant)))
        existing = coordinate_ids.get(key)
        if existing is not None:
            return existing, True
        nid = str(len(nodes) + 1)
        coordinate_ids[key] = nid
        nodes[nid] = [x, y]
        return nid, False

    for block_index, block in enumerate(block_meshes, start=1):
        node_map: dict[str, str] = {}
        for old_id, point in block.get("nodes", {}).items():
            new_id, reused = merged_node_id(point)
            if reused:
                duplicate_node_count += 1
            node_map[str(old_id)] = new_id
        block_element_ids: list[str] = []
        for element in block.get("elements", []):
            new_element = dict(element)
            new_element["id"] = str(len(elements) + 1)
            new_element["nodes"] = [node_map[str(nid)] for nid in element.get("nodes", [])]
            new_element["curved_outer_block"] = block_index
            elements.append(new_element)
            block_element_ids.append(new_element["id"])
        block_sets = block.get("node_sets", {})
        outer_ids = [node_map[str(nid)] for nid in block_sets.get("outer_boundary", block_sets.get("boundary", [])) if str(nid) in node_map]
        if not outer_ids:
            outer_ids = [node_map[str(nid)] for nid in block_sets.get("boundary", []) if str(nid) in node_map]
        block_boundary_ids = [node_map[str(nid)] for nid in block_sets.get("boundary", []) if str(nid) in node_map]
        node_sets[f"curved_outer_boundary_{block_index}"] = list(dict.fromkeys(outer_ids))
        node_sets["boundary"].extend(block_boundary_ids or outer_ids)
        for name, ids in block_sets.items():
            if name in {"all", "boundary", "outer_boundary"}:
                continue
            mapped = [node_map[str(nid)] for nid in ids if str(nid) in node_map]
            if mapped:
                node_sets[f"block_{block_index}_{name}"] = list(dict.fromkeys(mapped))
        element_sets[f"region_{block_index}"] = block_element_ids
        element_sets["all"].extend(block_element_ids)
        node_sets["all"].extend(node_map.values())
        quality = block.get("mesh_quality", {})
        boundary_counts.append(len(node_sets[f"curved_outer_boundary_{block_index}"]))
        projected_total += int(quality.get("parametric_curved_outer_projected_node_count", 0)) + int(quality.get("parametric_curve_hole_projected_node_count", 0))
        curve_count += int(quality.get("parametric_curved_outer_curve_count", 0))
        hole_count += int(quality.get("polygon_hole_count", 0))
        source_modes.append(str(block.get("mode", "")))
    node_sets = {name: list(dict.fromkeys(ids)) for name, ids in node_sets.items() if ids}
    outer_set_names = [f"curved_outer_boundary_{index}" for index in range(1, len(block_meshes) + 1)]
    shared_nodes = _shared_node_ids_between_sets(node_sets, outer_set_names)
    if shared_nodes:
        node_sets["shared_curved_outer_boundary"] = shared_nodes
    quality = mesh_quality_summary(nodes, elements)
    quad_count = sum(1 for element in elements if str(element.get("type", "")).upper() == "QUAD4")
    tri_count = sum(1 for element in elements if str(element.get("type", "")).upper() == "TRI3")
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": element_sets,
        "mode": "multi_parametric_curved_outer_block_paving",
        "target_size": target,
        "division_width": target,
        "requested_element_type": block_meshes[0].get("requested_element_type", "QUAD4"),
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": quad_count,
            "fallback_tri_count": tri_count,
            "quad_ratio": quad_count / max(quad_count + tri_count, 1),
            "aspect_ratio": quality["max_quad_aspect_ratio"],
            "recombined_quad_count": sum(int(block.get("mesh_quality", {}).get("recombined_quad_count", 0)) for block in block_meshes),
            "remaining_tri_count": tri_count,
            "min_quad_angle_deg": quality["min_quad_angle_deg"],
            "max_quad_aspect_ratio": quality["max_quad_aspect_ratio"],
            "boundary_projected_node_count": projected_total,
            "boundary_snap_tolerance": max(target * 0.25, 1.0e-12),
            "boundary_layer_added_x_count": 0,
            "boundary_layer_added_y_count": 0,
            "oblique_boundary_layer_added_line_count": 0,
            "curved_boundary_layer_added_line_count": sum(int(block.get("mesh_quality", {}).get("curved_boundary_layer_added_line_count", 0)) for block in block_meshes),
            "curved_boundary_segment_count": sum(int(block.get("mesh_quality", {}).get("curved_boundary_segment_count", 0)) for block in block_meshes),
            "local_refinement_added_line_count": 0,
            "curve_parameter_retained": True,
            "parametric_curved_outer_block_count": len(block_meshes),
            "parametric_curved_outer_boundary_count": len(block_meshes),
            "parametric_curved_outer_boundary_node_counts": boundary_counts,
            "parametric_curved_outer_block_quad_count": quad_count,
            "parametric_curved_outer_curve_count": curve_count,
            "parametric_curved_outer_source_modes": source_modes,
            "parametric_curved_outer_shared_boundary_node_count": len(shared_nodes),
            "parametric_curved_outer_merged_node_count": duplicate_node_count,
            "parametric_curved_outer_boolean_repaired": bool(shared_nodes),
            "polygon_hole_count": hole_count,
            "polygon_hole_boundary_segment_count": sum(int(block.get("mesh_quality", {}).get("polygon_hole_boundary_segment_count", 0)) for block in block_meshes),
        },
    }


def generate_rectilinear_polygon_paving_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    material: str,
    integration: str,
    requested_type: str = "QUAD4",
    refinements: list[Any] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
) -> dict[str, Any] | None:
    if target <= 0.0 or requested_type.upper().startswith("TRI") or not regions or tunnels or refinements:
        return None
    polygons: list[list[Point2D]] = []
    for region in regions:
        polygon = normalize_rectilinear_region(region)
        if polygon is None:
            return None
        polygons.append(polygon)
    holes = normalize_polygon_holes(polygon_holes or [])
    rect_holes: list[list[Point2D]] = []
    for hole in holes:
        rect_hole = normalize_rectilinear_region(hole)
        if rect_hole is None:
            return None
        rect_holes.append(rect_hole)
    if len(polygons) == 1 and len(polygons[0]) <= 4 and not rect_holes:
        return None
    xs = sorted({point[0] for polygon in polygons for point in polygon}.union(point[0] for hole in rect_holes for point in hole))
    ys = sorted({point[1] for polygon in polygons for point in polygon}.union(point[1] for hole in rect_holes for point in hole))
    if len(xs) < 2 or len(ys) < 2:
        return None
    coarse_blocks: list[tuple[float, float, float, float, int]] = []
    for ix, (xa, xb) in enumerate(zip(xs, xs[1:])):
        if xb - xa <= 1.0e-12:
            continue
        for iy, (ya, yb) in enumerate(zip(ys, ys[1:])):
            if yb - ya <= 1.0e-12:
                continue
            cx = 0.5 * (xa + xb)
            cy = 0.5 * (ya + yb)
            matched_regions = [index for index, polygon in enumerate(polygons) if point_in_polygon(cx, cy, polygon)]
            if len(matched_regions) > 1:
                return None
            if matched_regions and not any(point_in_polygon(cx, cy, hole) for hole in rect_holes):
                coarse_blocks.append((xa, xb, ya, yb, matched_regions[0]))
    if not coarse_blocks:
        return None

    nodes: dict[str, list[float]] = {}
    node_ids: dict[tuple[int, int], str] = {}
    quant = max(target * 1.0e-10, 1.0e-12)

    def node_id(x: float, y: float) -> str:
        key = (int(round(x / quant)), int(round(y / quant)))
        if key not in node_ids:
            nid = str(len(nodes) + 1)
            node_ids[key] = nid
            nodes[nid] = [float(x), float(y)]
        return node_ids[key]

    elements: list[dict[str, Any]] = []
    region_sets: dict[str, list[str]] = {f"region_{i + 1}": [] for i in range(len(polygons))}
    for block_index, (xa, xb, ya, yb, region_index) in enumerate(coarse_blocks, start=1):
        nu = max(1, int(math.ceil((xb - xa) / target)))
        nv = max(1, int(math.ceil((yb - ya) / target)))
        for j in range(nv):
            y_low = ya + (yb - ya) * j / nv
            y_high = ya + (yb - ya) * (j + 1) / nv
            for i in range(nu):
                x_low = xa + (xb - xa) * i / nu
                x_high = xa + (xb - xa) * (i + 1) / nu
                quad_nodes = [
                    node_id(x_low, y_low),
                    node_id(x_high, y_low),
                    node_id(x_high, y_high),
                    node_id(x_low, y_high),
                ]
                pts = [tuple(nodes[nid]) for nid in quad_nodes]
                quality = quad_quality(pts)
                if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 45.0 or quality["aspect_ratio"] > 20.0:
                    return None
                cx = sum(point[0] for point in pts) * 0.25
                cy = sum(point[1] for point in pts) * 0.25
                if not point_in_polygon(cx, cy, polygons[region_index]) or any(point_in_polygon(cx, cy, hole) for hole in rect_holes):
                    return None
                eid = str(len(elements) + 1)
                elements.append(
                    {
                        "id": eid,
                        "type": "QUAD4",
                        "nodes": quad_nodes,
                        "material": material,
                        "integration": integration,
                        "source": "rectilinear_paving",
                        "paving_block": block_index,
                        "region": region_index + 1,
                    }
                )
                region_sets[f"region_{region_index + 1}"].append(eid)
    if not elements:
        return None

    quality = mesh_quality_summary(nodes, elements)
    element_ids = [element["id"] for element in elements]
    hole_boundary_segments = [segment for hole in rect_holes for segment in zip(hole, hole[1:] + hole[:1])]
    boundary_segments = [segment for polygon in polygons for segment in zip(polygon, polygon[1:] + polygon[:1])]
    boundary_segments.extend(hole_boundary_segments)
    boundary_nodes = node_ids_on_boundary(nodes, boundary_segments)
    hole_boundary_nodes = node_ids_on_boundary(nodes, hole_boundary_segments)
    node_use_count = {nid: 0 for nid in nodes}
    for element in elements:
        for nid in element.get("nodes", []):
            node_use_count[str(nid)] = node_use_count.get(str(nid), 0) + 1
    all_region_points = [point for polygon in polygons for point in polygon]
    x_min = min(point[0] for point in all_region_points)
    x_max = max(point[0] for point in all_region_points)
    y_min = min(point[1] for point in all_region_points)
    y_max = max(point[1] for point in all_region_points)
    node_sets = {
        "left": [nid for nid, point in nodes.items() if abs(point[0] - x_min) <= quant],
        "right": [nid for nid, point in nodes.items() if abs(point[0] - x_max) <= quant],
        "bottom": [nid for nid, point in nodes.items() if abs(point[1] - y_min) <= quant],
        "top": [nid for nid, point in nodes.items() if abs(point[1] - y_max) <= quant],
        "boundary": boundary_nodes,
        "hole_boundary": hole_boundary_nodes,
        "all": list(nodes),
    }
    node_sets = {name: ids for name, ids in node_sets.items() if ids}
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": {"all": element_ids, **{key: value for key, value in region_sets.items() if value}},
        "mode": "rectilinear_polygon_paving",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": len(elements),
            "fallback_tri_count": 0,
            "quad_ratio": 1.0,
            "aspect_ratio": quality["max_quad_aspect_ratio"],
            "recombined_quad_count": 0,
            "remaining_tri_count": 0,
            "min_quad_angle_deg": quality["min_quad_angle_deg"],
            "max_quad_aspect_ratio": quality["max_quad_aspect_ratio"],
            "boundary_projected_node_count": 0,
            "boundary_snap_tolerance": 0.0,
            "boundary_layer_added_x_count": 0,
            "boundary_layer_added_y_count": 0,
            "oblique_boundary_layer_added_line_count": 0,
            "curved_boundary_layer_added_line_count": 0,
            "curved_boundary_segment_count": 0,
            "local_refinement_added_line_count": 0,
            "x_line_count": len(xs),
            "y_line_count": len(ys),
            "rectilinear_paving_quad_count": len(elements),
            "rectilinear_paving_block_count": len(coarse_blocks),
            "rectilinear_paving_region_count": len(polygons),
            "rectilinear_paving_shared_node_count": sum(1 for count in node_use_count.values() if count > 1),
            "polygon_hole_count": len(rect_holes),
            "polygon_hole_boundary_segment_count": sum(len(hole) for hole in rect_holes),
        },
    }


def generate_quadrilateral_polygon_hole_paving_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    material: str,
    integration: str,
    requested_type: str = "QUAD4",
    refinements: list[Any] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
) -> dict[str, Any] | None:
    if target <= 0.0 or requested_type.upper().startswith("TRI") or len(regions) != 1 or tunnels or refinements:
        return None
    holes = normalize_polygon_holes(polygon_holes or [])
    if len(holes) != 1:
        return None
    outer = normalize_quadrilateral_region(regions[0])
    hole = normalize_simple_polygon_region(holes[0])
    if outer is None or hole is None or len(hole) <= 4:
        return None
    if not all(point_in_polygon(x, y, outer) for x, y in hole):
        return None
    center = star_shaped_polygon_center(hole)
    if center is None:
        return None
    outer_segments = list(zip(outer, outer[1:] + outer[:1]))
    hole_segments = list(zip(hole, hole[1:] + hole[:1]))
    outer_points: list[Point2D] = []
    for hx, hy in hole:
        dx = hx - center[0]
        dy = hy - center[1]
        if math.hypot(dx, dy) <= 1.0e-12:
            return None
        point = ray_polygon_intersection(center, (dx, dy), outer)
        if point is None or math.hypot(point[0] - center[0], point[1] - center[1]) <= math.hypot(dx, dy) + 1.0e-9:
            return None
        outer_points.append(point)

    side_divisions = [
        max(1, int(math.ceil(max(math.hypot(outer_points[(i + 1) % len(hole)][0] - outer_points[i][0], outer_points[(i + 1) % len(hole)][1] - outer_points[i][1]), math.hypot(hole[(i + 1) % len(hole)][0] - hole[i][0], hole[(i + 1) % len(hole)][1] - hole[i][1])) / target)))
        for i in range(len(hole))
    ]
    radial_divisions = max(1, int(math.ceil(max(math.hypot(outer_points[i][0] - hole[i][0], outer_points[i][1] - hole[i][1]) for i in range(len(hole))) / target)))
    quant = max(target * 1.0e-10, 1.0e-12)
    nodes: dict[str, list[float]] = {}
    node_ids: dict[tuple[int, int], str] = {}

    def node_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        if key not in node_ids:
            nid = str(len(nodes) + 1)
            node_ids[key] = nid
            nodes[nid] = [float(point[0]), float(point[1])]
        return node_ids[key]

    elements: list[dict[str, Any]] = []
    patch_sets: dict[str, list[str]] = {}
    min_angle = math.inf
    max_aspect = 0.0
    patch_count = len(hole)
    for side in range(patch_count):
        p0 = outer_points[side]
        p1 = outer_points[(side + 1) % patch_count]
        p2 = hole[(side + 1) % patch_count]
        p3 = hole[side]
        divisions = side_divisions[side]
        patch_key = f"polygon_hole_patch_{side + 1}"
        patch_sets[patch_key] = []
        patch_node_ids: dict[tuple[int, int], str] = {}
        for j in range(radial_divisions + 1):
            v = j / radial_divisions
            for i in range(divisions + 1):
                u = i / divisions
                patch_node_ids[(i, j)] = node_id(bilinear_quad_point(p0, p1, p2, p3, u, v))
        for j in range(radial_divisions):
            for i in range(divisions):
                quad_nodes = [
                    patch_node_ids[(i, j)],
                    patch_node_ids[(i + 1, j)],
                    patch_node_ids[(i + 1, j + 1)],
                    patch_node_ids[(i, j + 1)],
                ]
                pts = [tuple(nodes[nid]) for nid in quad_nodes]
                if polygon_area(pts) < 0.0:
                    quad_nodes = list(reversed(quad_nodes))
                    pts = [tuple(nodes[nid]) for nid in quad_nodes]
                quality = quad_quality(pts)
                if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 8.0 or quality["aspect_ratio"] > 60.0:
                    return None
                qx = sum(point[0] for point in pts) * 0.25
                qy = sum(point[1] for point in pts) * 0.25
                if not quad_samples_respect_domain(pts, outer, hole, outer_segments, hole_segments):
                    return None
                min_angle = min(min_angle, quality["min_angle_deg"])
                max_aspect = max(max_aspect, quality["aspect_ratio"])
                eid = str(len(elements) + 1)
                elements.append({"id": eid, "type": "QUAD4", "nodes": quad_nodes, "material": material, "integration": integration, "source": "quadrilateral_polygon_hole_paving", "polygon_hole_patch": side + 1})
                patch_sets[patch_key].append(eid)
    if not elements:
        return None

    x0, x1, y0, y1 = bbox
    element_ids = [element["id"] for element in elements]
    outer_nodes = node_ids_on_boundary(nodes, outer_segments)
    hole_nodes = node_ids_on_boundary(nodes, hole_segments)
    node_sets = {
        "left": [nid for nid, point in nodes.items() if abs(point[0] - x0) <= quant],
        "right": [nid for nid, point in nodes.items() if abs(point[0] - x1) <= quant],
        "bottom": [nid for nid, point in nodes.items() if abs(point[1] - y0) <= quant],
        "top": [nid for nid, point in nodes.items() if abs(point[1] - y1) <= quant],
        "outer_boundary": outer_nodes,
        "hole_boundary": hole_nodes,
        "boundary": list(dict.fromkeys([*outer_nodes, *hole_nodes])),
        "all": list(nodes),
    }
    node_sets = {name: ids for name, ids in node_sets.items() if ids}
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": {"all": element_ids, "region_1": element_ids.copy(), **{key: value for key, value in patch_sets.items() if value}},
        "mode": "quadrilateral_polygon_hole_paving",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": len(elements),
            "fallback_tri_count": 0,
            "quad_ratio": 1.0,
            "aspect_ratio": max_aspect,
            "recombined_quad_count": 0,
            "remaining_tri_count": 0,
            "min_quad_angle_deg": min_angle,
            "max_quad_aspect_ratio": max_aspect,
            "boundary_projected_node_count": 0,
            "boundary_snap_tolerance": 0.0,
            "boundary_layer_added_x_count": 0,
            "boundary_layer_added_y_count": 0,
            "oblique_boundary_layer_added_line_count": 0,
            "curved_boundary_layer_added_line_count": 0,
            "curved_boundary_segment_count": 0,
            "local_refinement_added_line_count": 0,
            "x_line_count": len({point[0] for point in [*outer, *hole]}),
            "y_line_count": len({point[1] for point in [*outer, *hole]}),
            "polygon_hole_paving_quad_count": len(elements),
            "polygon_hole_paving_patch_count": patch_count,
            "polygon_hole_paving_radial_divisions": radial_divisions,
            "polygon_hole_paving_side_divisions": side_divisions,
            "polygon_hole_paving_hole_vertex_count": len(hole),
            "polygon_hole_paving_concave_vertex_count": count_concave_vertices(hole),
            "polygon_hole_paving_star_shaped": True,
            "polygon_hole_count": 1,
            "polygon_hole_boundary_segment_count": len(hole),
        },
    }


def generate_quadrilateral_circular_tunnel_paving_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    material: str,
    integration: str,
    requested_type: str = "QUAD4",
    refinements: list[Any] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
) -> dict[str, Any] | None:
    if target <= 0.0 or requested_type.upper().startswith("TRI") or len(regions) != 1 or len(tunnels) != 1 or refinements or polygon_holes:
        return None
    outer = normalize_quadrilateral_region(regions[0])
    if outer is None:
        return None
    cx, cy, radius = tunnels[0]
    if radius <= 0.0 or not point_in_polygon(cx, cy, outer):
        return None
    if not circle_inside_polygon((cx, cy), radius, outer, samples=max(32, curve_segment_count(radius, target))):
        return None

    angles = unwrap_angles([math.atan2(y - cy, x - cx) for x, y in outer])
    side_lengths = [math.hypot(outer[(i + 1) % 4][0] - outer[i][0], outer[(i + 1) % 4][1] - outer[i][1]) for i in range(4)]
    radial_lengths = [max(0.0, math.hypot(x - cx, y - cy) - radius) for x, y in outer]
    side_divisions = [
        max(2, int(math.ceil(max(side_lengths[i], radius * abs(angles[i + 1] - angles[i])) / target)))
        for i in range(4)
    ]
    radial_divisions = max(1, int(math.ceil(max(radial_lengths) / target)))
    quant = max(target * 1.0e-10, 1.0e-12)
    nodes: dict[str, list[float]] = {}
    node_ids: dict[tuple[int, int], str] = {}

    def node_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        if key not in node_ids:
            nid = str(len(nodes) + 1)
            node_ids[key] = nid
            nodes[nid] = [float(point[0]), float(point[1])]
        return node_ids[key]

    elements: list[dict[str, Any]] = []
    patch_sets: dict[str, list[str]] = {}
    min_angle = math.inf
    max_aspect = 0.0
    for side in range(4):
        p0 = outer[side]
        p1 = outer[(side + 1) % 4]
        a0 = angles[side]
        a1 = angles[side + 1]
        divisions = side_divisions[side]
        patch_key = f"circular_patch_{side + 1}"
        patch_sets[patch_key] = []
        patch_node_ids: dict[tuple[int, int], str] = {}
        for j in range(radial_divisions + 1):
            v = j / radial_divisions
            for i in range(divisions + 1):
                u = i / divisions
                outer_point = (p0[0] * (1.0 - u) + p1[0] * u, p0[1] * (1.0 - u) + p1[1] * u)
                angle = a0 * (1.0 - u) + a1 * u
                tunnel_point = (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
                patch_node_ids[(i, j)] = node_id((outer_point[0] * (1.0 - v) + tunnel_point[0] * v, outer_point[1] * (1.0 - v) + tunnel_point[1] * v))
        for j in range(radial_divisions):
            for i in range(divisions):
                quad_nodes = [
                    patch_node_ids[(i, j)],
                    patch_node_ids[(i + 1, j)],
                    patch_node_ids[(i + 1, j + 1)],
                    patch_node_ids[(i, j + 1)],
                ]
                pts = [tuple(nodes[nid]) for nid in quad_nodes]
                if polygon_area(pts) < 0.0:
                    quad_nodes = list(reversed(quad_nodes))
                    pts = [tuple(nodes[nid]) for nid in quad_nodes]
                quality = quad_quality(pts)
                if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 8.0 or quality["aspect_ratio"] > 50.0:
                    return None
                qx = sum(point[0] for point in pts) * 0.25
                qy = sum(point[1] for point in pts) * 0.25
                if not point_in_polygon(qx, qy, outer) or math.hypot(qx - cx, qy - cy) <= radius:
                    return None
                min_angle = min(min_angle, quality["min_angle_deg"])
                max_aspect = max(max_aspect, quality["aspect_ratio"])
                eid = str(len(elements) + 1)
                elements.append({"id": eid, "type": "QUAD4", "nodes": quad_nodes, "material": material, "integration": integration, "source": "quadrilateral_circular_tunnel_paving", "circular_patch": side + 1})
                patch_sets[patch_key].append(eid)
    if not elements:
        return None

    x0, x1, y0, y1 = bbox
    element_ids = [element["id"] for element in elements]
    outer_segments = list(zip(outer, outer[1:] + outer[:1]))
    outer_nodes = node_ids_on_boundary(nodes, outer_segments)
    tunnel_nodes = [nid for nid, point in nodes.items() if abs(math.hypot(point[0] - cx, point[1] - cy) - radius) <= max(quant, 1.0e-10)]
    node_sets = {
        "left": [nid for nid, point in nodes.items() if abs(point[0] - x0) <= quant],
        "right": [nid for nid, point in nodes.items() if abs(point[0] - x1) <= quant],
        "bottom": [nid for nid, point in nodes.items() if abs(point[1] - y0) <= quant],
        "top": [nid for nid, point in nodes.items() if abs(point[1] - y1) <= quant],
        "outer_boundary": outer_nodes,
        "tunnel_boundary": tunnel_nodes,
        "boundary": list(dict.fromkeys([*outer_nodes, *tunnel_nodes])),
        "all": list(nodes),
    }
    node_sets = {name: ids for name, ids in node_sets.items() if ids}
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": {"all": element_ids, "region_1": element_ids.copy(), **{key: value for key, value in patch_sets.items() if value}},
        "mode": "quadrilateral_circular_tunnel_paving",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": len(elements),
            "fallback_tri_count": 0,
            "quad_ratio": 1.0,
            "aspect_ratio": max_aspect,
            "recombined_quad_count": 0,
            "remaining_tri_count": 0,
            "min_quad_angle_deg": min_angle,
            "max_quad_aspect_ratio": max_aspect,
            "boundary_projected_node_count": 0,
            "boundary_snap_tolerance": 0.0,
            "boundary_layer_added_x_count": 0,
            "boundary_layer_added_y_count": 0,
            "oblique_boundary_layer_added_line_count": 0,
            "curved_boundary_layer_added_line_count": 0,
            "curved_boundary_segment_count": sum(side_divisions),
            "local_refinement_added_line_count": 0,
            "x_line_count": len({point[0] for point in outer}),
            "y_line_count": len({point[1] for point in outer}),
            "circular_tunnel_paving_quad_count": len(elements),
            "circular_tunnel_paving_patch_count": 4,
            "circular_tunnel_paving_radial_divisions": radial_divisions,
            "circular_tunnel_paving_side_divisions": side_divisions,
            "circular_tunnel_paving_radius": radius,
            "polygon_hole_count": 0,
            "polygon_hole_boundary_segment_count": 0,
        },
    }


def generate_quadrilateral_ring_paving_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    material: str,
    integration: str,
    requested_type: str = "QUAD4",
    refinements: list[Any] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
) -> dict[str, Any] | None:
    if target <= 0.0 or requested_type.upper().startswith("TRI") or len(regions) != 1 or tunnels or refinements:
        return None
    holes = normalize_polygon_holes(polygon_holes or [])
    if len(holes) != 1:
        return None
    outer = normalize_quadrilateral_region(regions[0])
    hole = normalize_quadrilateral_region(holes[0])
    if outer is None or hole is None:
        return None
    if not all(point_in_polygon(x, y, outer) for x, y in hole):
        return None
    if polygon_area(hole) >= polygon_area(outer):
        return None

    side_lengths = [math.hypot(outer[(i + 1) % 4][0] - outer[i][0], outer[(i + 1) % 4][1] - outer[i][1]) for i in range(4)]
    hole_side_lengths = [math.hypot(hole[(i + 1) % 4][0] - hole[i][0], hole[(i + 1) % 4][1] - hole[i][1]) for i in range(4)]
    radial_lengths = [math.hypot(outer[i][0] - hole[i][0], outer[i][1] - hole[i][1]) for i in range(4)]
    side_divisions = [max(1, int(math.ceil(max(side_lengths[i], hole_side_lengths[i]) / target))) for i in range(4)]
    radial_divisions = max(1, int(math.ceil(max(radial_lengths) / target)))
    quant = max(target * 1.0e-10, 1.0e-12)
    nodes: dict[str, list[float]] = {}
    node_ids: dict[tuple[int, int], str] = {}

    def node_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        if key not in node_ids:
            nid = str(len(nodes) + 1)
            node_ids[key] = nid
            nodes[nid] = [float(point[0]), float(point[1])]
        return node_ids[key]

    elements: list[dict[str, Any]] = []
    patch_sets: dict[str, list[str]] = {}
    min_angle = math.inf
    max_aspect = 0.0
    for side in range(4):
        p0 = outer[side]
        p1 = outer[(side + 1) % 4]
        p2 = hole[(side + 1) % 4]
        p3 = hole[side]
        divisions = side_divisions[side]
        patch_key = f"ring_patch_{side + 1}"
        patch_sets[patch_key] = []
        patch_node_ids: dict[tuple[int, int], str] = {}
        for j in range(radial_divisions + 1):
            v = j / radial_divisions
            for i in range(divisions + 1):
                u = i / divisions
                patch_node_ids[(i, j)] = node_id(bilinear_quad_point(p0, p1, p2, p3, u, v))
        for j in range(radial_divisions):
            for i in range(divisions):
                n00 = patch_node_ids[(i, j)]
                n10 = patch_node_ids[(i + 1, j)]
                n11 = patch_node_ids[(i + 1, j + 1)]
                n01 = patch_node_ids[(i, j + 1)]
                quad_nodes = [n00, n10, n11, n01]
                pts = [tuple(nodes[nid]) for nid in quad_nodes]
                if polygon_area(pts) < 0.0:
                    quad_nodes = list(reversed(quad_nodes))
                    pts = [tuple(nodes[nid]) for nid in quad_nodes]
                quality = quad_quality(pts)
                if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 10.0 or quality["aspect_ratio"] > 40.0:
                    return None
                cx = sum(point[0] for point in pts) * 0.25
                cy = sum(point[1] for point in pts) * 0.25
                if not point_in_polygon(cx, cy, outer) or point_in_polygon(cx, cy, hole):
                    return None
                min_angle = min(min_angle, quality["min_angle_deg"])
                max_aspect = max(max_aspect, quality["aspect_ratio"])
                eid = str(len(elements) + 1)
                elements.append({"id": eid, "type": "QUAD4", "nodes": quad_nodes, "material": material, "integration": integration, "source": "quadrilateral_ring_paving", "ring_patch": side + 1})
                patch_sets[patch_key].append(eid)
    if not elements:
        return None

    outer_segments = list(zip(outer, outer[1:] + outer[:1]))
    hole_segments = list(zip(hole, hole[1:] + hole[:1]))
    outer_nodes = node_ids_on_boundary(nodes, outer_segments)
    hole_nodes = node_ids_on_boundary(nodes, hole_segments)
    x0, x1, y0, y1 = bbox
    element_ids = [element["id"] for element in elements]
    node_sets = {
        "left": [nid for nid, point in nodes.items() if abs(point[0] - x0) <= quant],
        "right": [nid for nid, point in nodes.items() if abs(point[0] - x1) <= quant],
        "bottom": [nid for nid, point in nodes.items() if abs(point[1] - y0) <= quant],
        "top": [nid for nid, point in nodes.items() if abs(point[1] - y1) <= quant],
        "outer_boundary": outer_nodes,
        "hole_boundary": hole_nodes,
        "boundary": list(dict.fromkeys([*outer_nodes, *hole_nodes])),
        "all": list(nodes),
    }
    node_sets = {name: ids for name, ids in node_sets.items() if ids}
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": {"all": element_ids, "region_1": element_ids.copy(), **{key: value for key, value in patch_sets.items() if value}},
        "mode": "quadrilateral_ring_paving",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": len(elements),
            "fallback_tri_count": 0,
            "quad_ratio": 1.0,
            "aspect_ratio": max_aspect,
            "recombined_quad_count": 0,
            "remaining_tri_count": 0,
            "min_quad_angle_deg": min_angle,
            "max_quad_aspect_ratio": max_aspect,
            "boundary_projected_node_count": 0,
            "boundary_snap_tolerance": 0.0,
            "boundary_layer_added_x_count": 0,
            "boundary_layer_added_y_count": 0,
            "oblique_boundary_layer_added_line_count": 0,
            "curved_boundary_layer_added_line_count": 0,
            "curved_boundary_segment_count": 0,
            "local_refinement_added_line_count": 0,
            "x_line_count": len({point[0] for point in [*outer, *hole]}),
            "y_line_count": len({point[1] for point in [*outer, *hole]}),
            "quadrilateral_ring_paving_quad_count": len(elements),
            "quadrilateral_ring_paving_patch_count": 4,
            "quadrilateral_ring_paving_radial_divisions": radial_divisions,
            "quadrilateral_ring_paving_side_divisions": side_divisions,
            "polygon_hole_count": 1,
            "polygon_hole_boundary_segment_count": 4,
        },
    }


def generate_simple_polygon_quad_paving_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    material: str,
    integration: str,
    requested_type: str = "QUAD4",
    refinements: list[Any] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
) -> dict[str, Any] | None:
    if target <= 0.0 or requested_type.upper().startswith("TRI") or not regions or tunnels or refinements or polygon_holes:
        return None
    polygons: list[list[Point2D]] = []
    for region in regions:
        polygon = normalize_simple_polygon_region(region)
        if polygon is None or len(polygon) < 4:
            return None
        polygons.append(polygon)
    if polygons_any_overlap(polygons):
        return None

    nodes: dict[str, list[float]] = {}
    coordinate_ids: dict[tuple[int, int], str] = {}
    midpoint_ids: dict[tuple[str, str], str] = {}
    quant = max(target * 1.0e-10, 1.0e-12)

    def node_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        if key not in coordinate_ids:
            nid = str(len(nodes) + 1)
            coordinate_ids[key] = nid
            nodes[nid] = [float(point[0]), float(point[1])]
        return coordinate_ids[key]

    def midpoint_id(a: str, b: str) -> str:
        key = tuple(sorted((a, b)))
        if key not in midpoint_ids:
            pa = nodes[a]
            pb = nodes[b]
            midpoint_ids[key] = node_id(((pa[0] + pb[0]) * 0.5, (pa[1] + pb[1]) * 0.5))
        return midpoint_ids[key]

    elements: list[dict[str, Any]] = []
    region_sets: dict[str, list[str]] = {f"region_{i + 1}": [] for i in range(len(polygons))}
    boundary_segments: list[tuple[Point2D, Point2D]] = []
    min_angle = math.inf
    max_aspect = 0.0
    triangle_count = 0
    concave_count = 0
    for region_index, polygon in enumerate(polygons):
        triangles = triangulate_simple_polygon(polygon)
        if not triangles:
            return None
        triangle_count += len(triangles)
        concave_count += count_concave_vertices(polygon)
        boundary_segments.extend(zip(polygon, polygon[1:] + polygon[:1]))
        for tri_index, tri in enumerate(triangles, start=1):
            if polygon_area(tri) < 0.0:
                tri = [tri[0], tri[2], tri[1]]
            a_id = node_id(tri[0])
            b_id = node_id(tri[1])
            c_id = node_id(tri[2])
            ab_id = midpoint_id(a_id, b_id)
            bc_id = midpoint_id(b_id, c_id)
            ca_id = midpoint_id(c_id, a_id)
            centroid_id = node_id(((tri[0][0] + tri[1][0] + tri[2][0]) / 3.0, (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0))
            quad_specs = [
                [a_id, ab_id, centroid_id, ca_id],
                [b_id, bc_id, centroid_id, ab_id],
                [c_id, ca_id, centroid_id, bc_id],
            ]
            for quad_nodes in quad_specs:
                pts = [tuple(nodes[nid]) for nid in quad_nodes]
                if polygon_area(pts) < 0.0:
                    quad_nodes = list(reversed(quad_nodes))
                    pts = [tuple(nodes[nid]) for nid in quad_nodes]
                quality = quad_quality(pts)
                if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 5.0 or quality["aspect_ratio"] > 80.0:
                    return None
                cx = sum(point[0] for point in pts) * 0.25
                cy = sum(point[1] for point in pts) * 0.25
                if not point_in_polygon(cx, cy, polygon):
                    return None
                min_angle = min(min_angle, quality["min_angle_deg"])
                max_aspect = max(max_aspect, quality["aspect_ratio"])
                eid = str(len(elements) + 1)
                elements.append(
                    {
                        "id": eid,
                        "type": "QUAD4",
                        "nodes": quad_nodes,
                        "material": material,
                        "integration": integration,
                        "source": "simple_polygon_quad_paving",
                        "paving_triangle": tri_index,
                        "region": region_index + 1,
                    }
                )
                region_sets[f"region_{region_index + 1}"].append(eid)
    if not elements:
        return None

    boundary_nodes = node_ids_on_boundary(nodes, boundary_segments)
    x0, x1, y0, y1 = bbox
    element_ids = [element["id"] for element in elements]
    node_sets = {
        "left": [nid for nid, point in nodes.items() if abs(point[0] - x0) <= quant],
        "right": [nid for nid, point in nodes.items() if abs(point[0] - x1) <= quant],
        "bottom": [nid for nid, point in nodes.items() if abs(point[1] - y0) <= quant],
        "top": [nid for nid, point in nodes.items() if abs(point[1] - y1) <= quant],
        "boundary": boundary_nodes,
        "all": list(nodes),
    }
    node_sets = {name: ids for name, ids in node_sets.items() if ids}
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": {"all": element_ids, **{key: value for key, value in region_sets.items() if value}},
        "mode": "simple_polygon_quad_paving",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": len(elements),
            "fallback_tri_count": 0,
            "quad_ratio": 1.0,
            "aspect_ratio": max_aspect,
            "recombined_quad_count": 0,
            "remaining_tri_count": 0,
            "min_quad_angle_deg": min_angle,
            "max_quad_aspect_ratio": max_aspect,
            "boundary_projected_node_count": 0,
            "boundary_snap_tolerance": 0.0,
            "boundary_layer_added_x_count": 0,
            "boundary_layer_added_y_count": 0,
            "oblique_boundary_layer_added_line_count": 0,
            "curved_boundary_layer_added_line_count": 0,
            "curved_boundary_segment_count": 0,
            "local_refinement_added_line_count": 0,
            "x_line_count": len({point[0] for polygon in polygons for point in polygon}),
            "y_line_count": len({point[1] for polygon in polygons for point in polygon}),
            "polygon_quad_paving_quad_count": len(elements),
            "polygon_quad_paving_triangle_count": triangle_count,
            "polygon_quad_paving_boundary_node_count": len(boundary_nodes),
            "polygon_quad_paving_concave_vertex_count": concave_count,
            "polygon_quad_paving_region_count": len(polygons),
            "polygon_hole_count": 0,
            "polygon_hole_boundary_segment_count": 0,
        },
    }


def generate_mapped_quadrilateral_paving_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    material: str,
    integration: str,
    requested_type: str = "QUAD4",
    refinements: list[Any] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
) -> dict[str, Any] | None:
    if target <= 0.0 or requested_type.upper().startswith("TRI") or not regions or tunnels or refinements or polygon_holes:
        return None
    polygons: list[list[Point2D]] = []
    for region in regions:
        polygon = normalize_quadrilateral_region(region)
        if polygon is None:
            return None
        polygons.append(polygon)
    if len(polygons) > 1:
        if any(polygons_have_positive_area_overlap(polygons[left], polygons[right]) for left, right in _polygon_pair_candidates(polygons, tol=target * 1.0e-8)):
            return None
        if not quadrilateral_regions_share_edges(polygons, target=target):
            return None

    edge_counts = mapped_quadrilateral_edge_counts(polygons, target)
    quant = max(target * 1.0e-10, 1.0e-12)
    nodes: dict[str, list[float]] = {}
    coordinate_ids: dict[tuple[int, int], str] = {}

    def node_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        if key not in coordinate_ids:
            nid = str(len(nodes) + 1)
            coordinate_ids[key] = nid
            nodes[nid] = [float(point[0]), float(point[1])]
        return coordinate_ids[key]

    elements: list[dict[str, Any]] = []
    region_sets: dict[str, list[str]] = {f"region_{i + 1}": [] for i in range(len(polygons))}
    single_edge_sets = {"left": [], "right": [], "bottom": [], "top": []}
    min_angle = math.inf
    max_aspect = 0.0
    u_divisions: list[int] = []
    v_divisions: list[int] = []
    for region_index, polygon in enumerate(polygons):
        p0, p1, p2, p3 = polygon
        edge_keys = [_canonical_segment_key(a, b, quant) for a, b in zip(polygon, polygon[1:] + polygon[:1])]
        nu = max(edge_counts[edge_keys[0]], edge_counts[edge_keys[2]], 1)
        nv = max(edge_counts[edge_keys[1]], edge_counts[edge_keys[3]], 1)
        u_divisions.append(nu)
        v_divisions.append(nv)
        grid: dict[tuple[int, int], str] = {}
        for j in range(nv + 1):
            v = j / nv
            for i in range(nu + 1):
                u = i / nu
                grid[(i, j)] = node_id(bilinear_quad_point(p0, p1, p2, p3, u, v))
        if len(polygons) == 1:
            single_edge_sets["bottom"] = [grid[(i, 0)] for i in range(nu + 1)]
            single_edge_sets["top"] = [grid[(i, nv)] for i in range(nu + 1)]
            single_edge_sets["left"] = [grid[(0, j)] for j in range(nv + 1)]
            single_edge_sets["right"] = [grid[(nu, j)] for j in range(nv + 1)]
        for j in range(nv):
            for i in range(nu):
                quad_nodes = [grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]]
                pts = [tuple(nodes[nid]) for nid in quad_nodes]
                quality = quad_quality(pts)
                if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 10.0 or quality["aspect_ratio"] > 20.0:
                    return None
                min_angle = min(min_angle, quality["min_angle_deg"])
                max_aspect = max(max_aspect, quality["aspect_ratio"])
                eid = str(len(elements) + 1)
                elements.append(
                    {
                        "id": eid,
                        "type": "QUAD4",
                        "nodes": quad_nodes,
                        "material": material,
                        "integration": integration,
                        "source": "mapped_paving",
                        "region": region_index + 1,
                    }
                )
                region_sets[f"region_{region_index + 1}"].append(eid)
    if not elements:
        return None
    _x_coords, _y_coords, layer_info = build_boundary_layer_coordinates(bbox, target, polygons, [], refinements or [])
    quality = mesh_quality_summary(nodes, elements)
    x0, x1, y0, y1 = bbox
    boundary_tol = max(target * 1.0e-8, 1.0e-10)
    if len(polygons) == 1:
        node_sets = dict(single_edge_sets)
    else:
        boundary_segments = [segment for polygon in polygons for segment in zip(polygon, polygon[1:] + polygon[:1])]
        node_sets = {
            "left": [nid for nid, point in nodes.items() if abs(point[0] - x0) <= boundary_tol],
            "right": [nid for nid, point in nodes.items() if abs(point[0] - x1) <= boundary_tol],
            "bottom": [nid for nid, point in nodes.items() if abs(point[1] - y0) <= boundary_tol],
            "top": [nid for nid, point in nodes.items() if abs(point[1] - y1) <= boundary_tol],
            "boundary": node_ids_on_boundary(nodes, boundary_segments, tol=boundary_tol),
        }
    node_sets["all"] = list(nodes)
    node_sets = {name: list(dict.fromkeys(ids)) for name, ids in node_sets.items() if ids}
    node_region_use: dict[str, set[int]] = {nid: set() for nid in nodes}
    for element in elements:
        region_number = int(element.get("region", 0) or 0)
        for nid in element.get("nodes", []):
            node_region_use.setdefault(str(nid), set()).add(region_number)
    shared_node_count = sum(1 for regions_used in node_region_use.values() if len(regions_used) > 1)
    element_ids = [element["id"] for element in elements]
    element_sets = {"all": element_ids}
    element_sets.update({key: value for key, value in region_sets.items() if value})
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": element_sets,
        "mode": "mapped_quad_paving",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": len(elements),
            "fallback_tri_count": 0,
            "quad_ratio": 1.0,
            "aspect_ratio": max(axis_coordinate_aspect_ratio([point[0] for polygon in polygons for point in polygon], [point[1] for polygon in polygons for point in polygon]), quality["max_quad_aspect_ratio"]),
            "recombined_quad_count": 0,
            "remaining_tri_count": 0,
            "min_quad_angle_deg": min(min_angle, quality["min_quad_angle_deg"]),
            "max_quad_aspect_ratio": max(max_aspect, quality["max_quad_aspect_ratio"]),
            "boundary_projected_node_count": 0,
            "boundary_snap_tolerance": 0.0,
            "boundary_layer_added_x_count": int(layer_info.get("boundary_layer_added_x_count", 0)),
            "boundary_layer_added_y_count": int(layer_info.get("boundary_layer_added_y_count", 0)),
            "oblique_boundary_layer_added_line_count": int(layer_info.get("oblique_boundary_layer_added_line_count", 0)),
            "curved_boundary_layer_added_line_count": 0,
            "curved_boundary_segment_count": 0,
            "local_refinement_added_line_count": 0,
            "x_line_count": len({point[0] for polygon in polygons for point in polygon}),
            "y_line_count": len({point[1] for polygon in polygons for point in polygon}),
            "mapped_paving_quad_count": len(elements),
            "mapped_paving_region_count": len(polygons),
            "mapped_paving_shared_node_count": shared_node_count,
            "mapped_paving_u_divisions": max(u_divisions),
            "mapped_paving_v_divisions": max(v_divisions),
            "polygon_hole_count": 0,
            "polygon_hole_boundary_segment_count": 0,
        },
    }


def generate_per_region_mapped_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    material: str,
    integration: str,
    requested_type: str = "QUAD4",
    refinements: list[Any] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
    region_ids: list[str] | None = None,
    region_settings: Mapping[str, Any] | list[Any] | None = None,
) -> dict[str, Any] | None:
    if target <= 0.0 or not regions or tunnels or refinements or polygon_holes:
        return None
    polygons: list[list[Point2D]] = []
    for region in regions:
        polygon = normalize_quadrilateral_region(region)
        if polygon is None:
            return None
        polygons.append(polygon)
    if any(polygons_have_positive_area_overlap(polygons[left], polygons[right]) for left, right in _polygon_pair_candidates(polygons, tol=target * 1.0e-8)):
        return None

    ids = [str(value or f"region_{index + 1}") for index, value in enumerate(region_ids or [], start=1)]
    while len(ids) < len(polygons):
        ids.append(f"region_{len(ids) + 1}")
    settings = _region_settings_by_index(ids, region_settings)
    quant = max(min(_region_target(settings[index], target) for index in range(len(polygons))) * 1.0e-10, 1.0e-12)

    edge_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    edge_owner_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    polygon_edge_keys: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []
    desired_divisions: list[tuple[int, int]] = []
    for index, polygon in enumerate(polygons):
        local = settings[index]
        local_target = _region_target(local, target)
        edge_lengths = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(polygon, polygon[1:] + polygon[:1])]
        nx = _positive_int(local.get("nx", local.get("u_divisions")), max(int(math.ceil(max(edge_lengths[0], edge_lengths[2], local_target) / local_target)), 1))
        ny = _positive_int(local.get("ny", local.get("v_divisions")), max(int(math.ceil(max(edge_lengths[1], edge_lengths[3], local_target) / local_target)), 1))
        desired_divisions.append((nx, ny))
        keys = [_canonical_segment_key(a, b, quant) for a, b in zip(polygon, polygon[1:] + polygon[:1])]
        polygon_edge_keys.append(keys)
        for key, count in zip(keys, (nx, ny, nx, ny)):
            edge_counts[key] = max(edge_counts.get(key, 1), int(count))
            edge_owner_counts[key] = edge_owner_counts.get(key, 0) + 1

    changed = True
    while changed:
        changed = False
        for keys in polygon_edge_keys:
            nu = max(edge_counts[keys[0]], edge_counts[keys[2]], 1)
            nv = max(edge_counts[keys[1]], edge_counts[keys[3]], 1)
            for key, value in ((keys[0], nu), (keys[2], nu), (keys[1], nv), (keys[3], nv)):
                if edge_counts.get(key, 1) < value:
                    edge_counts[key] = value
                    changed = True

    nodes: dict[str, list[float]] = {}
    coordinate_ids: dict[tuple[int, int], str] = {}
    edge_mid_nodes: dict[tuple[str, str], str] = {}

    def node_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        if key not in coordinate_ids:
            nid = str(len(nodes) + 1)
            coordinate_ids[key] = nid
            nodes[nid] = [float(point[0]), float(point[1])]
        return coordinate_ids[key]

    def midpoint_node(a: str, b: str) -> str:
        key = tuple(sorted((str(a), str(b))))
        if key not in edge_mid_nodes:
            pa = nodes[str(a)]
            pb = nodes[str(b)]
            nid = node_id(((float(pa[0]) + float(pb[0])) * 0.5, (float(pa[1]) + float(pb[1])) * 0.5))
            edge_mid_nodes[key] = nid
        return edge_mid_nodes[key]

    elements: list[dict[str, Any]] = []
    element_sets: dict[str, list[str]] = {"all": []}
    shape_meshes: list[dict[str, Any]] = []
    min_angle = math.inf
    max_aspect = 0.0
    type_counts: dict[str, int] = {}
    region_node_sets: dict[str, set[str]] = {}
    u_divisions: list[int] = []
    v_divisions: list[int] = []

    def add_element(region_index: int, etype: str, conn: list[str]) -> str:
        local = settings[region_index]
        region_number = region_index + 1
        region_id = ids[region_index]
        eid = str(len(elements) + 1)
        element = {
            "id": eid,
            "type": etype,
            "nodes": conn,
            "material": str(local.get("material", material) or material),
            "integration": str(local.get("integration", integration) or integration),
            "source": "per_region_mapped",
            "region": region_number,
            "region_id": region_id,
        }
        elements.append(element)
        for set_name in ("all", f"region_{region_number}", f"region_{region_id}"):
            element_sets.setdefault(set_name, []).append(eid)
        region_node_sets.setdefault(f"region_{region_number}_nodes", set()).update(conn)
        region_node_sets.setdefault(f"region_{region_id}_nodes", set()).update(conn)
        type_counts[etype] = type_counts.get(etype, 0) + 1
        return eid

    for region_index, polygon in enumerate(polygons):
        p0, p1, p2, p3 = polygon
        keys = polygon_edge_keys[region_index]
        nu = max(edge_counts[keys[0]], edge_counts[keys[2]], 1)
        nv = max(edge_counts[keys[1]], edge_counts[keys[3]], 1)
        desired_nx, desired_ny = desired_divisions[region_index]
        bottom_ratio = _shared_edge_ratio(edge_counts, edge_owner_counts, keys[0], desired_nx)
        right_ratio = _shared_edge_ratio(edge_counts, edge_owner_counts, keys[1], desired_ny)
        top_ratio = _shared_edge_ratio(edge_counts, edge_owner_counts, keys[2], desired_nx)
        left_ratio = _shared_edge_ratio(edge_counts, edge_owner_counts, keys[3], desired_ny)
        u_values = _graded_axis_values(nu, start_ratio=left_ratio, end_ratio=right_ratio)
        v_values = _graded_axis_values(nv, start_ratio=bottom_ratio, end_ratio=top_ratio)
        u_divisions.append(nu)
        v_divisions.append(nv)
        grid: dict[tuple[int, int], str] = {}
        for j in range(nv + 1):
            v = v_values[j]
            for i in range(nu + 1):
                u = u_values[i]
                grid[(i, j)] = node_id(bilinear_quad_point(p0, p1, p2, p3, u, v))
        before_count = len(elements)
        etype = _region_element_type(settings[region_index], requested_type)
        for j in range(nv):
            for i in range(nu):
                quad_nodes = [grid[(i, j)], grid[(i + 1, j)], grid[(i + 1, j + 1)], grid[(i, j + 1)]]
                pts = [tuple(nodes[nid]) for nid in quad_nodes]
                quality = quad_quality(pts)
                if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 10.0 or quality["aspect_ratio"] > 20.0:
                    return None
                min_angle = min(min_angle, quality["min_angle_deg"])
                max_aspect = max(max_aspect, quality["aspect_ratio"])
                if etype == "QUAD8":
                    mids = [
                        midpoint_node(quad_nodes[0], quad_nodes[1]),
                        midpoint_node(quad_nodes[1], quad_nodes[2]),
                        midpoint_node(quad_nodes[2], quad_nodes[3]),
                        midpoint_node(quad_nodes[3], quad_nodes[0]),
                    ]
                    add_element(region_index, "QUAD8", [*quad_nodes, *mids])
                elif etype.startswith("TRI"):
                    tri_specs = ([quad_nodes[0], quad_nodes[1], quad_nodes[2]], [quad_nodes[0], quad_nodes[2], quad_nodes[3]])
                    for tri in tri_specs:
                        if etype == "TRI6":
                            mids = [midpoint_node(tri[0], tri[1]), midpoint_node(tri[1], tri[2]), midpoint_node(tri[2], tri[0])]
                            add_element(region_index, "TRI6", [*tri, *mids])
                        else:
                            add_element(region_index, "TRI3", list(tri))
                else:
                    add_element(region_index, "QUAD4", quad_nodes)
        region_set = f"region_{region_index + 1}"
        shape_meshes.append(
            {
                "region": region_index + 1,
                "region_id": ids[region_index],
                "element_type": etype,
                "target_size": _region_target(settings[region_index], target),
                "u_divisions": nu,
                "v_divisions": nv,
                "desired_u_divisions": desired_nx,
                "desired_v_divisions": desired_ny,
                "transition_grading": {
                    "left": left_ratio,
                    "right": right_ratio,
                    "bottom": bottom_ratio,
                    "top": top_ratio,
                },
                "element_ids": element_sets.get(region_set, [])[before_count - before_count :],
                "element_count": len(elements) - before_count,
            }
        )

    if not elements:
        return None
    for shape in shape_meshes:
        key = f"region_{shape['region']}"
        shape["element_ids"] = list(element_sets.get(key, []))
        node_set = f"region_{shape['region']}_nodes"
        shape["node_ids"] = sorted(region_node_sets.get(node_set, set()), key=_natural_node_key)
    x0, x1, y0, y1 = bbox
    boundary_tol = max(target * 1.0e-8, 1.0e-10)
    boundary_segments = [segment for polygon in polygons for segment in zip(polygon, polygon[1:] + polygon[:1])]
    node_sets = {
        "left": [nid for nid, point in nodes.items() if abs(point[0] - x0) <= boundary_tol],
        "right": [nid for nid, point in nodes.items() if abs(point[0] - x1) <= boundary_tol],
        "bottom": [nid for nid, point in nodes.items() if abs(point[1] - y0) <= boundary_tol],
        "top": [nid for nid, point in nodes.items() if abs(point[1] - y1) <= boundary_tol],
        "boundary": node_ids_on_boundary(nodes, boundary_segments, tol=boundary_tol),
        "all": list(nodes),
    }
    for name, values in region_node_sets.items():
        node_sets[name] = sorted(values, key=_natural_node_key)
    node_sets = {name: list(dict.fromkeys(ids_)) for name, ids_ in node_sets.items() if ids_}
    node_region_use: dict[str, set[int]] = {nid: set() for nid in nodes}
    for element in elements:
        region_number = int(element.get("region", 0) or 0)
        for nid in element.get("nodes", []):
            node_region_use.setdefault(str(nid), set()).add(region_number)
    shared_node_count = sum(1 for regions_used in node_region_use.values() if len(regions_used) > 1)
    quality = mesh_quality_summary(nodes, elements)
    _x_coords, _y_coords, layer_info = build_boundary_layer_coordinates(bbox, target, polygons, [], [])
    return {
        "nodes": nodes,
        "elements": elements,
        "node_sets": node_sets,
        "element_sets": {key: value for key, value in element_sets.items() if value},
        "shape_meshes": shape_meshes,
        "mode": "per_region_mapped",
        "target_size": target,
        "division_width": target,
        "requested_element_type": requested_type,
        "element_type": requested_type,
        "tunnel_excluded_cells": [],
        "mesh_quality": {
            "quad_count": type_counts.get("QUAD4", 0) + type_counts.get("QUAD8", 0),
            "fallback_tri_count": type_counts.get("TRI3", 0) + type_counts.get("TRI6", 0),
            "quad_ratio": (type_counts.get("QUAD4", 0) + type_counts.get("QUAD8", 0)) / max(len(elements), 1),
            "aspect_ratio": max(axis_coordinate_aspect_ratio([point[0] for polygon in polygons for point in polygon], [point[1] for polygon in polygons for point in polygon]), quality["max_quad_aspect_ratio"]),
            "recombined_quad_count": 0,
            "remaining_tri_count": type_counts.get("TRI3", 0) + type_counts.get("TRI6", 0),
            "min_quad_angle_deg": min(min_angle, quality["min_quad_angle_deg"]),
            "max_quad_aspect_ratio": max(max_aspect, quality["max_quad_aspect_ratio"]),
            "boundary_projected_node_count": 0,
            "boundary_snap_tolerance": 0.0,
            "boundary_layer_added_x_count": int(layer_info.get("boundary_layer_added_x_count", 0)),
            "boundary_layer_added_y_count": int(layer_info.get("boundary_layer_added_y_count", 0)),
            "oblique_boundary_layer_added_line_count": int(layer_info.get("oblique_boundary_layer_added_line_count", 0)),
            "curved_boundary_layer_added_line_count": 0,
            "curved_boundary_segment_count": 0,
            "local_refinement_added_line_count": 0,
            "x_line_count": len({point[0] for polygon in polygons for point in polygon}),
            "y_line_count": len({point[1] for polygon in polygons for point in polygon}),
            "per_region_mapped_element_count": len(elements),
            "per_region_mapped_region_count": len(polygons),
            "per_region_mapped_shared_node_count": shared_node_count,
            "per_region_mapped_u_divisions": max(u_divisions),
            "per_region_mapped_v_divisions": max(v_divisions),
            "per_region_element_type_counts": dict(type_counts),
            "per_region_transition_grading": True,
            "polygon_hole_count": 0,
            "polygon_hole_boundary_segment_count": 0,
        },
    }


def _shared_edge_ratio(
    edge_counts: Mapping[tuple[tuple[int, int], tuple[int, int]], int],
    edge_owner_counts: Mapping[tuple[tuple[int, int], tuple[int, int]], int],
    key: tuple[tuple[int, int], tuple[int, int]],
    desired: int,
) -> float:
    if edge_owner_counts.get(key, 0) < 2:
        return 1.0
    return max(float(edge_counts.get(key, desired)) / max(int(desired), 1), 1.0)


def _graded_axis_values(count: int, *, start_ratio: float = 1.0, end_ratio: float = 1.0) -> list[float]:
    count = max(int(count), 1)
    start = max(float(start_ratio), 1.0)
    end = max(float(end_ratio), 1.0)
    if max(start, end) <= 1.05 or count <= 1:
        return [index / count for index in range(count + 1)]
    strength = min(max(max(start, end) - 1.0, 0.0), 3.0)
    weights: list[float] = []
    for index in range(count):
        t = (index + 0.5) / count
        if start > 1.05 and end > 1.05:
            shape = 1.0 - abs(2.0 * t - 1.0)
        elif start > 1.05:
            shape = t
        else:
            shape = 1.0 - t
        weights.append(max(0.05, 1.0 + strength * shape))
    total = sum(weights) or 1.0
    values = [0.0]
    accum = 0.0
    for weight in weights:
        accum += weight / total
        values.append(accum)
    values[-1] = 1.0
    return values


def _region_settings_by_index(region_ids: list[str], raw_settings: Mapping[str, Any] | list[Any] | None) -> list[dict[str, Any]]:
    settings = [dict() for _ in region_ids]
    if isinstance(raw_settings, Mapping):
        for index, region_id in enumerate(region_ids):
            candidates = (region_id, f"region_{index + 1}", str(index + 1), index + 1)
            for key in candidates:
                raw = raw_settings.get(key)  # type: ignore[arg-type]
                if isinstance(raw, Mapping):
                    settings[index].update(dict(raw))
    elif isinstance(raw_settings, list):
        for index, raw in enumerate(raw_settings[: len(settings)]):
            if isinstance(raw, Mapping):
                settings[index].update(dict(raw))
    return settings


def _region_target(settings: Mapping[str, Any], default: float) -> float:
    for key in ("target_size", "division_width", "size", "target"):
        try:
            value = float(settings.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value > 0.0:
            return value
    return float(default)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = int(default)
    return max(parsed, 1)


def _region_element_type(settings: Mapping[str, Any], default: str) -> str:
    etype = str(settings.get("element_type", settings.get("requested_element_type", settings.get("type", default))) or default).upper()
    return etype if etype in {"QUAD4", "QUAD8", "TRI3", "TRI6"} else str(default or "QUAD4").upper()


def _natural_node_key(value: str) -> tuple[int, str]:
    text = str(value)
    try:
        return (int(text), text)
    except ValueError:
        return (10**9, text)


def _canonical_point_key(point: Point2D, quant: float) -> tuple[int, int]:
    return (int(round(point[0] / quant)), int(round(point[1] / quant)))


def _canonical_segment_key(a: Point2D, b: Point2D, quant: float) -> tuple[tuple[int, int], tuple[int, int]]:
    left = _canonical_point_key(a, quant)
    right = _canonical_point_key(b, quant)
    return (left, right) if left <= right else (right, left)


def quadrilateral_regions_share_edges(polygons: list[list[Point2D]], *, target: float) -> bool:
    if len(polygons) < 2:
        return False
    quant = max(target * 1.0e-8, 1.0e-10)
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for polygon in polygons:
        for a, b in zip(polygon, polygon[1:] + polygon[:1]):
            key = _canonical_segment_key(a, b, quant)
            if key in seen:
                return True
            seen.add(key)
    return False


def mapped_quadrilateral_edge_counts(polygons: list[list[Point2D]], target: float) -> dict[tuple[tuple[int, int], tuple[int, int]], int]:
    quant = max(target * 1.0e-10, 1.0e-12)
    parent: dict[tuple[tuple[int, int], tuple[int, int]], tuple[tuple[int, int], tuple[int, int]]] = {}
    base_counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

    def find(key: tuple[tuple[int, int], tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
        parent.setdefault(key, key)
        if parent[key] != key:
            parent[key] = find(parent[key])
        return parent[key]

    def union(a: tuple[tuple[int, int], tuple[int, int]], b: tuple[tuple[int, int], tuple[int, int]]) -> None:
        root_a = find(a)
        root_b = find(b)
        if root_a != root_b:
            parent[root_b] = root_a

    for polygon in polygons:
        keys = [_canonical_segment_key(a, b, quant) for a, b in zip(polygon, polygon[1:] + polygon[:1])]
        for key, (a, b) in zip(keys, zip(polygon, polygon[1:] + polygon[:1])):
            parent.setdefault(key, key)
            base_counts[key] = max(base_counts.get(key, 1), int(math.ceil(max(math.hypot(b[0] - a[0], b[1] - a[1]), target) / target)))
        union(keys[0], keys[2])
        union(keys[1], keys[3])

    grouped: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for key, count in base_counts.items():
        root = find(key)
        grouped[root] = max(grouped.get(root, 1), count)
    return {key: grouped[find(key)] for key in base_counts}


def with_requested_element_order(mesh: dict[str, Any], requested_type: str) -> dict[str, Any]:
    requested = str(requested_type or mesh.get("requested_element_type") or mesh.get("element_type") or "").upper()
    if not requested:
        return mesh
    mesh["requested_element_type"] = requested
    mesh["element_type"] = requested
    if requested not in {"QUAD8", "TRI6"}:
        return mesh

    nodes = mesh.get("nodes", {})
    elements = mesh.get("elements", [])
    if not isinstance(nodes, dict) or not isinstance(elements, list):
        return mesh

    edge_mid_nodes: dict[tuple[str, str], str] = {}
    new_node_count = 0

    def midpoint_node(a: str, b: str) -> str:
        nonlocal new_node_count
        key = tuple(sorted((str(a), str(b))))
        if key not in edge_mid_nodes:
            pa = nodes[str(a)]
            pb = nodes[str(b)]
            nid = str(len(nodes) + 1)
            nodes[nid] = [(float(pa[0]) + float(pb[0])) * 0.5, (float(pa[1]) + float(pb[1])) * 0.5]
            edge_mid_nodes[key] = nid
            new_node_count += 1
        return edge_mid_nodes[key]

    promoted = 0
    for element in elements:
        etype = str(element.get("type", "")).upper()
        conn = [str(nid) for nid in element.get("nodes", [])]
        if requested == "QUAD8" and etype == "QUAD4" and len(conn) == 4:
            mids = [midpoint_node(conn[0], conn[1]), midpoint_node(conn[1], conn[2]), midpoint_node(conn[2], conn[3]), midpoint_node(conn[3], conn[0])]
            element["nodes"] = [conn[0], conn[1], conn[2], conn[3], *mids]
            element["type"] = "QUAD8"
            promoted += 1
        elif requested == "TRI6" and etype == "TRI3" and len(conn) == 3:
            mids = [midpoint_node(conn[0], conn[1]), midpoint_node(conn[1], conn[2]), midpoint_node(conn[2], conn[0])]
            element["nodes"] = [conn[0], conn[1], conn[2], *mids]
            element["type"] = "TRI6"
            promoted += 1

    if promoted:
        node_sets = mesh.get("node_sets", {})
        if isinstance(node_sets, dict):
            for name, ids in list(node_sets.items()):
                id_list = [str(nid) for nid in ids] if isinstance(ids, list) else []
                id_set = set(id_list)
                extra = [mid for (a, b), mid in edge_mid_nodes.items() if a in id_set and b in id_set]
                if name == "all":
                    node_sets[name] = list(nodes)
                elif extra:
                    node_sets[name] = list(dict.fromkeys([*id_list, *extra]))
            node_sets["all"] = list(nodes)
        quality = mesh.setdefault("mesh_quality", {})
        if isinstance(quality, dict):
            quality["promoted_element_type"] = requested
            quality["promoted_element_count"] = promoted
            quality["promoted_midside_node_count"] = new_node_count
    return mesh


def normalize_quadrilateral_region(region: list[Point2D]) -> list[Point2D] | None:
    points = normalize_region_points(region)
    if len(points) != 4:
        return None
    if polygon_area(points) < 0.0:
        points.reverse()
    if not is_convex_polygon(points):
        return None
    return points


def normalize_rectilinear_region(region: list[Point2D]) -> list[Point2D] | None:
    points = normalize_region_points(region)
    if len(points) < 4 or abs(polygon_area(points)) <= 1.0e-18:
        return None
    if polygon_area(points) < 0.0:
        points.reverse()
    span_x = max(point[0] for point in points) - min(point[0] for point in points)
    span_y = max(point[1] for point in points) - min(point[1] for point in points)
    axis_tol = max(span_x, span_y, 1.0) * 1.0e-10
    for a, b in zip(points, points[1:] + points[:1]):
        dx = abs(b[0] - a[0])
        dy = abs(b[1] - a[1])
        if dx <= axis_tol and dy <= axis_tol:
            return None
        if dx > axis_tol and dy > axis_tol:
            return None
    if polygon_has_crossing_edges(points):
        return None
    return points


def normalize_simple_polygon_region(region: list[Point2D]) -> list[Point2D] | None:
    points = normalize_region_points(region)
    if len(points) < 3 or abs(polygon_area(points)) <= 1.0e-18:
        return None
    if polygon_area(points) < 0.0:
        points.reverse()
    points = remove_collinear_polygon_vertices(points)
    if len(points) < 3 or abs(polygon_area(points)) <= 1.0e-18:
        return None
    if polygon_area(points) < 0.0:
        points.reverse()
    if polygon_has_crossing_edges(points):
        return None
    return points


def normalize_polygon_holes(holes: list[list[Point2D]]) -> list[list[Point2D]]:
    normalized: list[list[Point2D]] = []
    for raw in holes:
        points = normalize_region_points(raw)
        if len(points) >= 3 and abs(polygon_area(points)) > 1.0e-18 and not polygon_has_crossing_edges(points):
            if polygon_area(points) < 0.0:
                points.reverse()
            normalized.append(points)
    return normalized


def _points_from_region_mapping(region: Mapping[str, Any]) -> list[Point2D]:
    points_raw = region.get("points", [])
    if not isinstance(points_raw, list):
        return []
    points: list[Point2D] = []
    for raw in points_raw:
        point = _mesh_xy_pair(raw)
        if point is None:
            return []
        points.append(point)
    return points


def _curved_outer_polygon_from_region(
    region: Mapping[str, Any],
    boundary_specs: list[dict[str, Any]],
    fallback: list[Point2D],
    target: float,
) -> list[Point2D] | None:
    candidates: list[list[Point2D]] = []
    if boundary_specs:
        candidates.append(_curve_chain_sample_points(boundary_specs, target, closed=True, min_count=max(16, len(boundary_specs) * 4)))
    mapped = _points_from_region_mapping(region)
    if mapped:
        candidates.append(mapped)
    if fallback:
        candidates.append(fallback)
    for candidate in candidates:
        polygon = normalize_simple_polygon_region(candidate)
        if polygon is not None:
            return polygon
    return None


def _raw_curved_outer_points_from_region(
    region: Mapping[str, Any],
    boundary_specs: list[dict[str, Any]],
    fallback: list[Point2D],
    target: float,
) -> list[Point2D]:
    if boundary_specs:
        sampled = _curve_chain_sample_points(boundary_specs, target, closed=True, min_count=max(32, len(boundary_specs) * 8))
        if len(sampled) >= 3:
            return sampled
    mapped = normalize_region_points(_points_from_region_mapping(region))
    if len(mapped) >= 3:
        return mapped
    fallback_points = normalize_region_points(fallback)
    return fallback_points if len(fallback_points) >= 3 else []


def _generate_boolean_repaired_curved_outer_mesh(
    *,
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    curved_regions: list[Mapping[str, Any]],
    material: str,
    integration: str,
    requested_type: str,
    boolean_expression: str | None = None,
) -> dict[str, Any] | None:
    try:
        from geofem_app.analytic_boolean import (
            build_analytic_curve_boolean_graph,
            classify_graph_boolean_operations,
            operation_loop_area,
            operation_loop_polygons,
            regions_containing_point,
        )
    except ImportError:
        return None

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

    def classify_boolean_sets(mesh: dict[str, Any], graph: Mapping[str, Any], region_count: int) -> dict[str, list[str]]:
        mesh_nodes: dict[str, list[float]] = mesh["nodes"]
        sets: dict[str, list[str]] = {"all": []}
        for index in range(region_count):
            sets[f"region_{index + 1}"] = []
            sets[f"boolean_region_{index + 1}_union"] = []
            sets[f"boolean_region_{index + 1}_difference"] = []
        sets["boolean_overlap"] = []
        for element in mesh.get("elements", []):
            eid = str(element.get("id", ""))
            if not eid:
                continue
            sets["all"].append(eid)
            pts = [mesh_nodes[str(nid)] for nid in element.get("nodes", []) if str(nid) in mesh_nodes]
            if not pts:
                continue
            centroid = (sum(point[0] for point in pts) / len(pts), sum(point[1] for point in pts) / len(pts))
            matches = [index - 1 for index in regions_containing_point(graph, centroid)]
            if not matches:
                continue
            owner = matches[-1]
            sets[f"region_{owner + 1}"].append(eid)
            element["boolean_owner_region"] = owner + 1
            for index in matches:
                sets[f"boolean_region_{index + 1}_union"].append(eid)
            if len(matches) == 1:
                sets[f"boolean_region_{matches[0] + 1}_difference"].append(eid)
            else:
                sets["boolean_overlap"].append(eid)
                element["boolean_overlap_regions"] = [index + 1 for index in matches]
        return {name: list(dict.fromkeys(ids)) for name, ids in sets.items() if ids}

    def classify_analytic_curve_graph(graph: dict[str, Any], expression: str | None) -> dict[str, Any]:
        graph = classify_graph_boolean_operations(graph, expression=expression)
        for edge in graph.get("edges", []):
            edge["on_union_boundary"] = bool(edge.get("graph_union_boundary", False))
        return graph

    def graph_boundary_segments(graph: Mapping[str, Any], operation: str) -> list[dict[str, Any]]:
        segments: list[dict[str, Any]] = []
        for edge in graph.get("edges", []):
            if not bool(edge.get(f"graph_{operation}_boundary", False)):
                continue
            segments.append(
                {
                    "edge": str(edge.get("id", "")),
                    "region": int(edge.get("region", 0)),
                    "curve": int(edge.get("curve", 0)),
                    "type": str(edge.get("type", "")).lower(),
                    "t0": float(edge.get("t0", 0.0)),
                    "t1": float(edge.get("t1", 1.0)),
                    "start_vertex": str(edge.get("start_vertex", "")),
                    "end_vertex": str(edge.get("end_vertex", "")),
                    "source": edge.get("source", {}),
                }
            )
        return segments

    def graph_loop_regions_and_holes(graph: Mapping[str, Any], operation: str) -> tuple[list[list[Point2D]], list[list[Point2D]], list[dict[str, Any]]]:
        ring_records = operation_loop_polygons(graph, operation=operation, target=target)
        output_regions: list[list[Point2D]] = []
        output_holes: list[list[Point2D]] = []
        for ring in ring_records:
            points = normalize_region_points(ring.get("points", []))
            if len(points) < 3 or abs(polygon_area(points)) <= 1.0e-18 or polygon_has_crossing_edges(points):
                continue
            if polygon_area(points) > 0.0:
                output_regions.append(points)
            else:
                points.reverse()
                output_holes.append(points)
        return output_regions, output_holes, ring_records

    def polygon_boundary_node_ids(nodes: Mapping[str, list[float]], mesh_regions: list[list[Point2D]], mesh_holes: list[list[Point2D]]) -> list[str]:
        segments = [segment for region in mesh_regions for segment in zip(region, region[1:] + region[:1])]
        segments.extend(segment for hole in mesh_holes for segment in zip(hole, hole[1:] + hole[:1]))
        return node_ids_on_boundary(nodes, segments, boundary_tol)

    def curve_boundary_node_ids(nodes: Mapping[str, list[float]], specs: list[dict[str, Any]]) -> list[str]:
        if not specs:
            return []
        ids: list[str] = []
        for nid, point in nodes.items():
            projected = _nearest_point_on_curve_chain((float(point[0]), float(point[1])), specs, target)
            if projected is not None and math.hypot(projected[0] - float(point[0]), projected[1] - float(point[1])) <= boundary_tol:
                ids.append(str(nid))
        return list(dict.fromkeys(ids))

    sampled_point_count = 0
    curve_count = 0
    repaired_count = 0
    curve_spec_regions: list[list[dict[str, Any]]] = []
    for index, region in enumerate(curved_regions):
        boundary_specs = _curve_chain_specs_from_region(region)
        curve_spec_regions.append(boundary_specs)
        fallback = regions[index] if index < len(regions) else []
        raw_outer = _raw_curved_outer_points_from_region(region, boundary_specs, fallback, target)
        if len(raw_outer) < 3:
            continue
        sampled_point_count += len(raw_outer)
        curve_count += len(boundary_specs)
        if polygon_has_crossing_edges(raw_outer):
            repaired_count += 1
    if not curve_spec_regions:
        return None

    expression_text = str(boolean_expression).strip() if boolean_expression else ""
    selected_operation = "expression" if expression_text else "union"
    analytic_curve_graph = classify_analytic_curve_graph(
        build_analytic_curve_boolean_graph(curve_spec_regions, target=target, tol=max(target * 1.0e-8, 1.0e-10)),
        expression_text or None,
    )
    mesh_regions, mesh_holes, graph_ring_records = graph_loop_regions_and_holes(analytic_curve_graph, selected_operation)
    mesh_boundary_source = "analytic_curve_graph_loops" if mesh_regions else "analytic_curve_graph_failed"
    union_regions, union_holes = mesh_regions, mesh_holes
    if not union_regions:
        return None
    pairwise_overlap_area = operation_loop_area(analytic_curve_graph, operation="intersection", target=target)
    analytic_segments = graph_boundary_segments(analytic_curve_graph, selected_operation)
    analytic_kind_counts: dict[str, int] = {}
    for segment in analytic_segments:
        kind = str(segment.get("type", ""))
        analytic_kind_counts[kind] = analytic_kind_counts.get(kind, 0) + 1

    try:
        mesh = generate_quad_dominant_mesh(
            bbox=bbox,
            target=target,
            regions=union_regions,
            tunnels=[],
            material=material,
            integration=integration,
            requested_type=requested_type,
            refinements=[],
            polygon_holes=union_holes,
            curved_regions=[],
        )
    except ValueError:
        return None

    nodes: dict[str, list[float]] = mesh["nodes"]
    elements: list[dict[str, Any]] = mesh["elements"]
    mesh["element_sets"] = classify_boolean_sets(mesh, analytic_curve_graph, len(curve_spec_regions))
    node_sets: dict[str, list[str]] = dict(mesh.get("node_sets", {}))
    boundary_tol = max(target * 0.35, 1.0e-8)
    boundary_nodes = list(dict.fromkeys(node_sets.get("boundary", [])))
    if not boundary_nodes:
        boundary_nodes = polygon_boundary_node_ids(nodes, union_regions, union_holes)
    if boundary_nodes:
        node_sets["boolean_boundary"] = boundary_nodes
    for index, specs in enumerate(curve_spec_regions, start=1):
        ids = curve_boundary_node_ids(nodes, specs)
        if ids:
            node_sets[f"curved_outer_boundary_{index}"] = list(dict.fromkeys(ids))
    mesh["node_sets"] = {name: ids for name, ids in node_sets.items() if ids}
    quality = mesh_quality_summary(nodes, elements)
    quad_count = sum(1 for element in elements if str(element.get("type", "")).upper() == "QUAD4")
    tri_count = sum(1 for element in elements if str(element.get("type", "")).upper() == "TRI3")
    mesh["mode"] = "boolean_repaired_curved_outer_expression_paving" if expression_text else "boolean_repaired_curved_outer_union_paving"
    mesh["cad_boolean"] = {
        "engine": "analytic_curve_graph_winding_containment",
        "area_selection_engine": "analytic_winding_containment",
        "selected_operation": selected_operation,
        "boolean_expression": expression_text,
        "mesh_boundary_source": mesh_boundary_source,
        "operation_loop_rings": graph_ring_records,
        "analytic_boundary_segments": analytic_segments,
        "analytic_curve_kind_counts": analytic_kind_counts,
        "analytic_curve_graph": analytic_curve_graph,
        "topology_curve_representation": "analytic_split_graph",
        "trim_representation": "direct_curve_intersection_parameters",
    }
    mesh_quality = mesh.setdefault("mesh_quality", {})
    mesh_quality.update(
        {
            "quad_count": quad_count,
            "fallback_tri_count": tri_count,
            "remaining_tri_count": tri_count,
            "quad_ratio": quad_count / max(quad_count + tri_count, 1),
            "min_quad_angle_deg": quality["min_quad_angle_deg"],
            "max_quad_aspect_ratio": quality["max_quad_aspect_ratio"],
            "curve_parameter_retained": True,
            "parametric_curved_outer_boolean_repaired": True,
            "parametric_curved_outer_curve_count": curve_count,
            "parametric_curved_outer_boundary_count": len(curve_spec_regions),
            "parametric_curved_outer_union_boundary_node_count": len(mesh["node_sets"].get("boolean_boundary", [])),
            "boolean_engine": "analytic_curve_graph",
            "boolean_mesh_boundary_source": mesh_boundary_source,
            "boolean_element_set_classifier": "analytic_curve_graph_winding",
            "boolean_overlap_area_source": "analytic_intersection_loop_area",
            "boolean_self_intersection_repair": "analytic_curve_graph_loops",
            "boolean_selected_operation": selected_operation,
            "boolean_expression": expression_text,
            "boolean_input_region_count": len(curve_spec_regions),
            "boolean_union_region_count": len(union_regions),
            "boolean_polygon_hole_count": len(union_holes),
            "boolean_overlap_area": pairwise_overlap_area,
            "boolean_overlap_element_count": len(mesh["element_sets"].get("boolean_overlap", [])),
            "boolean_difference_region_count": sum(1 for index in range(len(curve_spec_regions)) if mesh["element_sets"].get(f"boolean_region_{index + 1}_difference")),
            "boolean_self_intersection_repaired_count": repaired_count,
            "boolean_sampled_outer_point_count": sampled_point_count,
            "boolean_analytic_curve_retained_count": len(analytic_segments),
            "boolean_analytic_trimmed_curve_count": sum(1 for segment in analytic_segments if float(segment.get("t0", 0.0)) > 1.0e-9 or float(segment.get("t1", 1.0)) < 1.0 - 1.0e-9),
            "boolean_analytic_intersection_count": int(analytic_curve_graph.get("intersection_count", 0)),
            "boolean_analytic_overlap_span_count": int(analytic_curve_graph.get("overlap_span_count", 0)),
            "boolean_analytic_overlap_edge_pair_count": int(analytic_curve_graph.get("overlap_edge_pair_count", 0)),
            "boolean_analytic_split_edge_count": int(analytic_curve_graph.get("split_edge_count", 0)),
            "boolean_analytic_union_loop_count": len(analytic_curve_graph.get("union_loops", [])),
            "boolean_analytic_intersection_loop_count": len(analytic_curve_graph.get("intersection_loops", [])),
            "boolean_analytic_difference_edge_count": int(analytic_curve_graph.get("difference_edge_count", 0)),
            "boolean_analytic_expression_edge_count": int(analytic_curve_graph.get("expression_edge_count", 0)),
            "boolean_analytic_operation_loop_ring_count": len(graph_ring_records),
            "boolean_tiny_gap_count": int(analytic_curve_graph.get("tolerance_diagnostics", {}).get("tiny_gap_count", 0)),
            "boolean_duplicate_curve_pair_count": int(analytic_curve_graph.get("tolerance_diagnostics", {}).get("duplicate_curve_pair_count", 0)),
            "boolean_overlapping_curve_pair_count": int(analytic_curve_graph.get("tolerance_diagnostics", {}).get("overlapping_curve_pair_count", 0)),
            "boolean_tangent_close_pair_count": int(analytic_curve_graph.get("tolerance_diagnostics", {}).get("tangent_close_pair_count", 0)),
            "boolean_close_intersection_cluster_count": int(analytic_curve_graph.get("tolerance_diagnostics", {}).get("close_intersection_cluster_count", 0)),
        }
    )
    return mesh


def _curve_chain_specs_from_region(region: Mapping[str, Any]) -> list[dict[str, Any]]:
    for key in ("curve_boundary", "boundary", "segments", "curves", "edges"):
        specs = _curve_chain_specs_from_value(region.get(key))
        if specs:
            return specs
    return []


def _curve_hole_chains_from_region(region: Mapping[str, Any]) -> list[list[dict[str, Any]]]:
    raw_holes = region.get("curve_holes")
    chains: list[list[dict[str, Any]]] = []
    if isinstance(raw_holes, list):
        for raw in raw_holes:
            specs = _curve_chain_specs_from_value(raw)
            if specs:
                chains.append(specs)
    if chains:
        return chains
    raw_holes = region.get("holes", region.get("islands", []))
    if isinstance(raw_holes, list):
        for raw in raw_holes:
            specs = _curve_chain_specs_from_value(raw)
            if specs:
                chains.append(specs)
    return chains


def _curve_chain_specs_from_value(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        spec = _normalize_curve_spec(value)
        if spec is not None:
            return [spec]
        for key in ("curve_boundary", "boundary", "segments", "curves", "edges"):
            specs = _curve_chain_specs_from_value(value.get(key))
            if specs:
                return specs
        return []
    if not isinstance(value, list):
        return []
    specs: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        spec = _normalize_curve_spec(item)
        if spec is not None:
            specs.append(spec)
    return specs


def _normalize_curve_spec(value: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = str(value.get("type", value.get("kind", value.get("curve_type", "")))).strip().lower()
    if not kind and "start" in value and "end" in value:
        kind = "line"
    if kind in {"line", "segment"}:
        start = _mesh_xy_pair(value.get("start", value.get("p1")))
        end = _mesh_xy_pair(value.get("end", value.get("p2")))
        if start is None or end is None:
            return None
        return {"type": "line", "start": start, "end": end}
    if kind in {"nurbs", "nurbs_curve", "rational_bspline", "rational_b_spline"} or (kind in {"spline", "bspline", "b_spline"} and any(key in value for key in ("knots", "weights", "degree", "order"))):
        raw_points = value.get("control_points", value.get("points", []))
        if not isinstance(raw_points, list):
            return None
        points = [_mesh_xy_pair(point) for point in raw_points]
        controls = [point for point in points if point is not None]
        if len(controls) < 2:
            return None
        degree_raw = value.get("degree", value.get("order", min(3, len(controls) - 1)))
        try:
            degree = int(degree_raw)
        except (TypeError, ValueError):
            degree = min(3, len(controls) - 1)
        if "order" in value and "degree" not in value:
            degree -= 1
        degree = max(1, min(degree, len(controls) - 1))
        weights = _mesh_float_list(value.get("weights"))
        if len(weights) != len(controls):
            weights = [1.0] * len(controls)
        knots = _mesh_float_list(value.get("knots", value.get("knot_vector")))
        if len(knots) != len(controls) + degree + 1:
            knots = _open_uniform_knot_vector(len(controls), degree)
        return {
            "type": "nurbs",
            "control_points": controls,
            "weights": weights,
            "knots": knots,
            "degree": degree,
            "closed": bool(value.get("closed", value.get("periodic", False))),
        }
    if kind in {"bezier", "spline", "bspline", "b_spline"}:
        raw_points = value.get("control_points", value.get("points", []))
        if not isinstance(raw_points, list):
            return None
        points = [_mesh_xy_pair(point) for point in raw_points]
        controls = [point for point in points if point is not None]
        if len(controls) < 2:
            return None
        return {"type": "bezier", "control_points": controls}
    center = _mesh_xy_pair(value.get("center", value.get("origin")))
    if center is None:
        return None
    radius = _mesh_float(value, "radius", "r")
    rx = _mesh_float(value, "rx", "radius_x", "major_radius", "a")
    ry = _mesh_float(value, "ry", "radius_y", "minor_radius", "b")
    if kind in {"ellipse", "elliptic_arc"} or rx is not None or ry is not None:
        rx = rx if rx is not None else radius
        ry = ry if ry is not None else rx
        if rx is None or ry is None or rx <= 0.0 or ry <= 0.0:
            return None
        return {
            "type": "ellipse" if kind == "ellipse" or bool(value.get("closed", value.get("full", False))) else "elliptic_arc",
            "center": center,
            "rx": float(rx),
            "ry": float(ry),
            "start_angle": _mesh_angle(value, "start_angle", "start_deg", "angle_start", default=0.0),
            "end_angle": _mesh_angle(value, "end_angle", "end_deg", "angle_end", default=360.0),
            "rotation": _mesh_angle(value, "rotation", "rotation_deg", default=0.0),
            "closed": kind == "ellipse" or bool(value.get("closed", value.get("full", False))),
        }
    if radius is None or radius <= 0.0:
        return None
    closed = kind == "circle" or bool(value.get("closed", value.get("full", False)))
    return {
        "type": "circle" if closed else "arc",
        "center": center,
        "radius": float(radius),
        "start_angle": _mesh_angle(value, "start_angle", "start_deg", "angle_start", default=0.0),
        "end_angle": _mesh_angle(value, "end_angle", "end_deg", "angle_end", default=360.0),
        "closed": closed,
    }


def _eval_curve_spec(spec: Mapping[str, Any], t: float) -> Point2D:
    kind = str(spec.get("type", "")).lower()
    t = min(max(float(t), 0.0), 1.0)
    if kind == "line":
        start = spec["start"]
        end = spec["end"]
        return (float(start[0]) * (1.0 - t) + float(end[0]) * t, float(start[1]) * (1.0 - t) + float(end[1]) * t)
    if kind == "bezier":
        return _mesh_de_casteljau([(float(x), float(y)) for x, y in spec.get("control_points", [])], t)
    if kind == "nurbs":
        return _mesh_eval_nurbs(spec, t)
    center = spec.get("center")
    if not isinstance(center, tuple):
        center = tuple(center) if isinstance(center, list) and len(center) >= 2 else (0.0, 0.0)
    cx, cy = float(center[0]), float(center[1])
    start_deg = float(spec.get("start_angle", 0.0))
    end_deg = float(spec.get("end_angle", 360.0))
    if bool(spec.get("closed", False)) or kind in {"circle", "ellipse"}:
        start_deg = 0.0 if kind in {"circle", "ellipse"} else start_deg
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
        return (cx + x * ct - y * st, cy + x * st + y * ct)
    radius = float(spec.get("radius", 0.0))
    return (cx + radius * math.cos(angle), cy + radius * math.sin(angle))


def _curve_spec_length(spec: Mapping[str, Any], target: float) -> float:
    key = (_mesh_freeze_curve_value(spec), float(target))
    cached = _CURVE_SPEC_LENGTH_CACHE.get(key)
    if cached is not None:
        return cached
    if str(spec.get("type", "")).lower() == "line":
        start = spec["start"]
        end = spec["end"]
        length = math.hypot(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
        _cache_curve_spec_length(key, length)
        return length
    samples = max(12, min(96, int(math.ceil(2.0 / max(target, 1.0e-9)))))
    points = [_eval_curve_spec(spec, i / samples) for i in range(samples + 1)]
    length = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:]))
    _cache_curve_spec_length(key, length)
    return length


def _cache_curve_spec_length(key: tuple[Any, float], length: float) -> None:
    if key in _CURVE_SPEC_LENGTH_CACHE:
        _CURVE_SPEC_LENGTH_CACHE[key] = length
        return
    if len(_CURVE_SPEC_LENGTH_CACHE) >= _CURVE_SPEC_LENGTH_CACHE_MAX:
        _CURVE_SPEC_LENGTH_CACHE.pop(next(iter(_CURVE_SPEC_LENGTH_CACHE)))
    _CURVE_SPEC_LENGTH_CACHE[key] = length


def _curve_chain_length(specs: list[dict[str, Any]], target: float) -> float:
    return sum(_curve_spec_length(spec, target) for spec in specs)


def _curve_chain_sample_points(specs: list[dict[str, Any]], target: float, *, closed: bool, min_count: int = 0) -> list[Point2D]:
    if not specs:
        return []
    total_length = max(_curve_chain_length(specs, target), target)
    points: list[Point2D] = []
    for spec in specs:
        length = _curve_spec_length(spec, target)
        count = max(1, int(math.ceil(length / max(target, 1.0e-9))))
        if len(specs) == 1 and closed:
            count = max(count, min_count)
            for i in range(count):
                _append_unique_point(points, _eval_curve_spec(spec, i / count))
            continue
        for i in range(count + 1):
            _append_unique_point(points, _eval_curve_spec(spec, i / count))
    if closed and len(points) > 1 and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= max(target * 1.0e-8, 1.0e-10):
        points.pop()
    if closed and len(points) < min_count and specs:
        points = []
        for i in range(min_count):
            distance = total_length * i / min_count
            _append_unique_point(points, _curve_chain_point_at_distance(specs, target, distance))
    return points


def _curve_chain_point_at_distance(specs: list[dict[str, Any]], target: float, distance: float) -> Point2D:
    remaining = distance
    for spec in specs:
        length = _curve_spec_length(spec, target)
        if remaining <= length or spec is specs[-1]:
            return _eval_curve_spec(spec, 0.0 if length <= 1.0e-12 else remaining / length)
        remaining -= length
    return _eval_curve_spec(specs[-1], 1.0)


def _curve_spec_sample_points(
    spec: Mapping[str, Any],
    target: float,
    *,
    spacing_factor: float,
    min_count: int,
    max_count: int,
) -> list[Point2D]:
    key = (_mesh_freeze_curve_value(spec), float(target), float(spacing_factor), int(min_count), int(max_count))
    cached = _CURVE_SPEC_SAMPLE_CACHE.get(key)
    if cached is not None:
        return cached
    length = max(_curve_spec_length(spec, target), target)
    count = max(min_count, min(max_count, int(math.ceil(length / max(spacing_factor * target, 1.0e-9)))))
    samples = [_eval_curve_spec(spec, index / count) for index in range(count + 1)]
    if key in _CURVE_SPEC_SAMPLE_CACHE:
        _CURVE_SPEC_SAMPLE_CACHE[key] = samples
        return samples
    if len(_CURVE_SPEC_SAMPLE_CACHE) >= _CURVE_SPEC_SAMPLE_CACHE_MAX:
        _CURVE_SPEC_SAMPLE_CACHE.pop(next(iter(_CURVE_SPEC_SAMPLE_CACHE)))
    _CURVE_SPEC_SAMPLE_CACHE[key] = samples
    return samples


def _curve_chain_center(specs: list[dict[str, Any]]) -> Point2D | None:
    if len(specs) != 1:
        return None
    center = specs[0].get("center")
    return _mesh_xy_pair(center)


def _curve_hole_boundary_node_ids(nodes: dict[str, list[float]], candidates: list[str], hole: list[Point2D], target: float) -> list[str]:
    segments = list(zip(hole, hole[1:] + hole[:1]))
    tol = max(0.35 * target, 1.0e-8)
    ids: list[str] = []
    for nid in candidates:
        point = nodes.get(str(nid))
        if point is None:
            continue
        if min(point_to_segment_distance((point[0], point[1]), a, b) for a, b in segments) <= tol:
            ids.append(str(nid))
    return list(dict.fromkeys(ids))


def _snap_node_ids_to_curve_chain(nodes: dict[str, list[float]], ids: list[str], specs: list[dict[str, Any]], target: float) -> int:
    changed = 0
    for nid in ids:
        point = nodes.get(str(nid))
        if point is None:
            continue
        projected = _nearest_point_on_curve_chain((point[0], point[1]), specs, target)
        if projected is None:
            continue
        if math.hypot(projected[0] - point[0], projected[1] - point[1]) <= 1.0e-12:
            continue
        nodes[str(nid)] = [float(projected[0]), float(projected[1])]
        changed += 1
    return changed


def _nearest_point_on_curve_chain(point: Point2D, specs: list[dict[str, Any]], target: float) -> Point2D | None:
    if len(specs) == 1:
        spec = specs[0]
        kind = str(spec.get("type", "")).lower()
        center = _mesh_xy_pair(spec.get("center"))
        if kind == "circle" and center is not None:
            radius = float(spec.get("radius", 0.0))
            dx = point[0] - center[0]
            dy = point[1] - center[1]
            length = math.hypot(dx, dy)
            if radius > 0.0 and length > 1.0e-30:
                return (center[0] + radius * dx / length, center[1] + radius * dy / length)
    best_point: Point2D | None = None
    best_distance = math.inf
    for spec in specs:
        samples = _curve_spec_sample_points(spec, target, spacing_factor=0.25, min_count=16, max_count=256)
        if len(samples) < 2:
            continue
        prev = samples[0]
        for current in samples[1:]:
            candidate, distance = nearest_point_on_segment(point, prev, current)
            if distance < best_distance:
                best_distance = distance
                best_point = candidate
            prev = current
    return best_point


def _append_unique_point(points: list[Point2D], point: Point2D) -> None:
    if points and math.hypot(points[-1][0] - point[0], points[-1][1] - point[1]) <= 1.0e-10:
        return
    points.append(point)


def _mesh_xy_pair(value: Any) -> Point2D | None:
    if isinstance(value, Mapping):
        if "x" not in value or "y" not in value:
            return None
        try:
            return float(value["x"]), float(value["y"])
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return float(value[0]), float(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _mesh_float(value: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        if key in value:
            try:
                return float(value[key])
            except (TypeError, ValueError):
                return None
    return None


def _mesh_angle(value: Mapping[str, Any], *keys: str, default: float) -> float:
    parsed = _mesh_float(value, *keys)
    return default if parsed is None else parsed


def _mesh_float_list(raw: Any) -> list[float]:
    if not isinstance(raw, list):
        return []
    values: list[float] = []
    for item in raw:
        try:
            values.append(float(item))
        except (TypeError, ValueError):
            return []
    return values


def _mesh_freeze_curve_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _mesh_freeze_curve_value(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_mesh_freeze_curve_value(item) for item in value)
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return str(value)


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


def _mesh_de_casteljau(points: list[Point2D], t: float) -> Point2D:
    if not points:
        return (0.0, 0.0)
    work = [(float(x), float(y)) for x, y in points]
    while len(work) > 1:
        work = [
            (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)
            for a, b in zip(work, work[1:])
        ]
    return work[0]


def _mesh_eval_nurbs(spec: Mapping[str, Any], t: float) -> Point2D:
    controls = [(float(x), float(y)) for x, y in spec.get("control_points", [])]
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
    basis_values = _mesh_nurbs_basis_values(degree, u, knots, len(controls))
    for index, (x, y) in enumerate(controls):
        basis = basis_values[index]
        weighted = basis * weights[index]
        numerator_x += weighted * x
        numerator_y += weighted * y
        denominator += weighted
    if abs(denominator) <= 1.0e-30:
        return controls[-1] if t >= 0.5 else controls[0]
    return numerator_x / denominator, numerator_y / denominator


def _mesh_nurbs_basis_values(degree: int, u: float, knots: list[float], count: int) -> list[float]:
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


def remove_collinear_polygon_vertices(points: list[Point2D], *, tol: float = 1.0e-12) -> list[Point2D]:
    cleaned = list(points)
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        out: list[Point2D] = []
        for i, point in enumerate(cleaned):
            prev = cleaned[i - 1]
            nxt = cleaned[(i + 1) % len(cleaned)]
            cross = (point[0] - prev[0]) * (nxt[1] - point[1]) - (point[1] - prev[1]) * (nxt[0] - point[0])
            edge_scale = max(math.hypot(point[0] - prev[0], point[1] - prev[1]), math.hypot(nxt[0] - point[0], nxt[1] - point[1]), 1.0)
            if abs(cross) <= tol * edge_scale * edge_scale:
                changed = True
                continue
            out.append(point)
        if out:
            cleaned = out
    return cleaned


def count_concave_vertices(points: list[Point2D]) -> int:
    if polygon_area(points) < 0.0:
        points = list(reversed(points))
    count = 0
    for i, point in enumerate(points):
        prev = points[i - 1]
        nxt = points[(i + 1) % len(points)]
        cross = (point[0] - prev[0]) * (nxt[1] - point[1]) - (point[1] - prev[1]) * (nxt[0] - point[0])
        if cross < -1.0e-12:
            count += 1
    return count


def polygon_hole_shape_summary(holes: list[list[Point2D]]) -> dict[str, int]:
    concave_holes = 0
    non_star_concave_holes = 0
    star_shaped_holes = 0
    total_concave_vertices = 0
    for hole in holes:
        concave_vertices = count_concave_vertices(hole)
        total_concave_vertices += concave_vertices
        if concave_vertices <= 0:
            continue
        concave_holes += 1
        if star_shaped_polygon_center(hole) is None:
            non_star_concave_holes += 1
        else:
            star_shaped_holes += 1
    return {
        "concave_hole_count": concave_holes,
        "non_star_concave_hole_count": non_star_concave_holes,
        "star_shaped_hole_count": star_shaped_holes,
        "total_concave_vertex_count": total_concave_vertices,
    }


def polygon_centroid(points: list[Point2D]) -> Point2D:
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for a, b in zip(points, points[1:] + points[:1]):
        cross = a[0] * b[1] - b[0] * a[1]
        area2 += cross
        cx += (a[0] + b[0]) * cross
        cy += (a[1] + b[1]) * cross
    if abs(area2) <= 1.0e-18:
        return (sum(point[0] for point in points) / len(points), sum(point[1] for point in points) / len(points))
    return (cx / (3.0 * area2), cy / (3.0 * area2))


def star_shaped_polygon_center(points: list[Point2D]) -> Point2D | None:
    candidates = [polygon_centroid(points)]
    triangles = sorted(triangulate_simple_polygon(points), key=lambda tri: abs(polygon_area(tri)), reverse=True)
    candidates.extend(((tri[0][0] + tri[1][0] + tri[2][0]) / 3.0, (tri[0][1] + tri[1][1] + tri[2][1]) / 3.0) for tri in triangles)
    for candidate in candidates:
        if not point_in_polygon(candidate[0], candidate[1], points):
            continue
        if all(segment_stays_in_polygon(candidate, vertex, points) for vertex in points):
            return candidate
    return None


def segment_stays_in_polygon(a: Point2D, b: Point2D, polygon: list[Point2D], *, samples: int = 9) -> bool:
    segments = list(zip(polygon, polygon[1:] + polygon[:1]))
    for i in range(1, samples):
        t = i / samples
        point = (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)
        if not (point_in_polygon(point[0], point[1], polygon) or point_on_boundary(point, segments)):
            return False
    return True


def quad_samples_respect_domain(
    points: list[Point2D],
    outer: list[Point2D],
    hole: list[Point2D],
    outer_segments: list[tuple[Point2D, Point2D]],
    hole_segments: list[tuple[Point2D, Point2D]],
) -> bool:
    samples = [((sum(point[0] for point in points) / len(points)), (sum(point[1] for point in points) / len(points)))]
    samples.extend(edge_midpoints(points))
    for point in samples:
        if not (point_in_polygon(point[0], point[1], outer) or point_on_boundary(point, outer_segments)):
            return False
        if point_in_polygon(point[0], point[1], hole) and not point_on_boundary(point, hole_segments):
            return False
    return True


def ray_polygon_intersection(origin: Point2D, direction: Point2D, polygon: list[Point2D]) -> Point2D | None:
    ox, oy = origin
    dx, dy = direction
    length = math.hypot(dx, dy)
    if length <= 1.0e-30:
        return None
    rx = dx / length
    ry = dy / length
    best_t = math.inf
    best: Point2D | None = None
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        sx = b[0] - a[0]
        sy = b[1] - a[1]
        denom = rx * sy - ry * sx
        if abs(denom) <= 1.0e-14:
            continue
        qx = a[0] - ox
        qy = a[1] - oy
        t = (qx * sy - qy * sx) / denom
        u = (qx * ry - qy * rx) / denom
        if t > 1.0e-10 and -1.0e-10 <= u <= 1.0 + 1.0e-10 and t < best_t:
            best_t = t
            best = (ox + t * rx, oy + t * ry)
    return best


def unwrap_angles(angles: list[float]) -> list[float]:
    if not angles:
        return []
    out = [float(angles[0])]
    for angle in angles[1:]:
        value = float(angle)
        while value <= out[-1]:
            value += 2.0 * math.pi
        out.append(value)
    out.append(out[0] + 2.0 * math.pi)
    return out


def circle_inside_polygon(center: Point2D, radius: float, polygon: list[Point2D], *, samples: int = 32) -> bool:
    if radius <= 0.0:
        return False
    cx, cy = center
    count = max(8, samples)
    return all(point_in_polygon(cx + radius * math.cos(2.0 * math.pi * i / count), cy + radius * math.sin(2.0 * math.pi * i / count), polygon) for i in range(count))


def triangulate_simple_polygon(points: list[Point2D]) -> list[list[Point2D]]:
    polygon = normalize_simple_polygon_region(points)
    if polygon is None or len(polygon) < 3:
        return []
    remaining = list(range(len(polygon)))
    triangles: list[list[Point2D]] = []
    guard = 0
    while len(remaining) > 3 and guard < len(polygon) * len(polygon):
        guard += 1
        clipped = False
        for local_index, current in enumerate(list(remaining)):
            prev_index = remaining[(local_index - 1) % len(remaining)]
            next_index = remaining[(local_index + 1) % len(remaining)]
            prev = polygon[prev_index]
            point = polygon[current]
            nxt = polygon[next_index]
            cross = (point[0] - prev[0]) * (nxt[1] - point[1]) - (point[1] - prev[1]) * (nxt[0] - point[0])
            if cross <= 1.0e-12:
                continue
            tri = [prev, point, nxt]
            if any(index not in {prev_index, current, next_index} and point_in_triangle(polygon[index], tri) for index in remaining):
                continue
            triangles.append(tri)
            remaining.pop(local_index)
            clipped = True
            break
        if not clipped:
            return []
    if len(remaining) == 3:
        tri = [polygon[index] for index in remaining]
        if abs(polygon_area(tri)) > 1.0e-18:
            triangles.append(tri)
    return triangles


def point_in_triangle(point: Point2D, triangle: list[Point2D], *, tol: float = 1.0e-12) -> bool:
    a, b, c = triangle
    area = abs(polygon_area(triangle))
    if area <= tol:
        return False
    a1 = abs(polygon_area([point, b, c]))
    a2 = abs(polygon_area([a, point, c]))
    a3 = abs(polygon_area([a, b, point]))
    return abs((a1 + a2 + a3) - area) <= tol * max(area, 1.0)


def normalize_region_points(region: list[Point2D]) -> list[Point2D]:
    points: list[Point2D] = []
    for point in region:
        x, y = float(point[0]), float(point[1])
        if not points or math.hypot(x - points[-1][0], y - points[-1][1]) > 1.0e-12:
            points.append((x, y))
    if len(points) > 1 and math.hypot(points[0][0] - points[-1][0], points[0][1] - points[-1][1]) <= 1.0e-12:
        points.pop()
    return points


def bilinear_quad_point(p0: Point2D, p1: Point2D, p2: Point2D, p3: Point2D, u: float, v: float) -> Point2D:
    w0 = (1.0 - u) * (1.0 - v)
    w1 = u * (1.0 - v)
    w2 = u * v
    w3 = (1.0 - u) * v
    return (
        w0 * p0[0] + w1 * p1[0] + w2 * p2[0] + w3 * p3[0],
        w0 * p0[1] + w1 * p1[1] + w2 * p2[1] + w3 * p3[1],
    )


def build_boundary_layer_coordinates(
    bbox: tuple[float, float, float, float],
    target: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    refinements: list[Any],
    *,
    polygon_holes: list[list[Point2D]] | None = None,
    layer_count: int = 2,
) -> tuple[list[float], list[float], dict[str, Any]]:
    x0, x1, y0, y1 = bbox
    if target <= 0.0:
        raise ValueError("target mesh size must be positive")
    nx = max(1, int(math.ceil((x1 - x0) / target)))
    ny = max(1, int(math.ceil((y1 - y0) / target)))
    base_x = {x0 + (x1 - x0) * i / nx for i in range(nx + 1)}
    base_y = {y0 + (y1 - y0) * j / ny for j in range(ny + 1)}
    x_values = set(base_x)
    y_values = set(base_y)
    boundary_x: set[float] = set()
    boundary_y: set[float] = set()
    oblique_x: set[float] = set()
    oblique_y: set[float] = set()
    curved_x: set[float] = set()
    curved_y: set[float] = set()
    refinement_x: set[float] = set()
    refinement_y: set[float] = set()
    layer_step = 0.5 * target
    axis_tol = max(target * 1.0e-8, 1.0e-12)

    def add_x(value: float, bucket: set[float]) -> None:
        if x0 - axis_tol <= value <= x1 + axis_tol:
            bucket.add(min(max(value, x0), x1))

    def add_y(value: float, bucket: set[float]) -> None:
        if y0 - axis_tol <= value <= y1 + axis_tol:
            bucket.add(min(max(value, y0), y1))

    def add_segment(a: Point2D, b: Point2D) -> None:
        ax, ay = a
        bx, by = b
        length = math.hypot(bx - ax, by - ay)
        if length <= 1.0e-30:
            add_x(ax, boundary_x)
            add_y(ay, boundary_y)
            return
        horizontal = abs(by - ay) <= axis_tol * max(length / target, 1.0)
        vertical = abs(bx - ax) <= axis_tol * max(length / target, 1.0)
        if not (horizontal or vertical):
            nxn = -(by - ay) / length
            nyn = (bx - ax) / length
            samples = max(1, min(64, int(math.ceil(length / max(0.5 * target, 1.0e-9)))))
            for idx in range(samples + 1):
                t = idx / samples
                px = ax * (1.0 - t) + bx * t
                py = ay * (1.0 - t) + by * t
                for layer in range(-layer_count, layer_count + 1):
                    offset = layer * layer_step
                    add_x(px + nxn * offset, oblique_x)
                    add_y(py + nyn * offset, oblique_y)
            return
        samples = max(1, min(32, int(math.ceil(length / target))))
        for idx in range(samples + 1):
            t = idx / samples
            px = ax * (1.0 - t) + bx * t
            py = ay * (1.0 - t) + by * t
            add_x(px, boundary_x)
            add_y(py, boundary_y)
        if horizontal:
            add_y(ay, boundary_y)
            for layer in range(1, layer_count + 1):
                offset = layer * layer_step
                add_y(ay - offset, boundary_y)
                add_y(ay + offset, boundary_y)
        if vertical:
            add_x(ax, boundary_x)
            for layer in range(1, layer_count + 1):
                offset = layer * layer_step
                add_x(ax - offset, boundary_x)
                add_x(ax + offset, boundary_x)

    if regions:
        for region in regions:
            for a, b in zip(region, region[1:] + region[:1]):
                add_segment(a, b)
    for hole in polygon_holes or []:
        for a, b in zip(hole, hole[1:] + hole[:1]):
            add_segment(a, b)
    for cx, cy, radius in tunnels:
        if radius <= 0.0:
            continue
        for layer in range(-layer_count, layer_count + 1):
            expanded = radius + layer * layer_step
            if expanded <= 0.0:
                continue
            segments = min(64, curve_segment_count(expanded, target))
            for idx in range(segments):
                angle = 2.0 * math.pi * idx / segments
                add_x(cx + expanded * math.cos(angle), curved_x)
                add_y(cy + expanded * math.sin(angle), curved_y)

    split_before_x = set(boundary_x) | set(oblique_x)
    split_before_y = set(boundary_y) | set(oblique_y)
    split_size_x: set[float] = set()
    split_size_y: set[float] = set()
    split_lines = normalize_mesh_split_lines(refinements)
    for start, end, local_target in split_lines:
        add_segment(start, end)
        if local_target is None or local_target <= 0.0:
            continue
        length = math.hypot(end[0] - start[0], end[1] - start[1])
        if length <= 1.0e-30:
            continue
        samples = max(1, int(math.ceil(length / max(local_target, 1.0e-9))))
        for idx in range(samples + 1):
            t = idx / samples
            add_x(start[0] * (1.0 - t) + end[0] * t, split_size_x)
            add_y(start[1] * (1.0 - t) + end[1] * t, split_size_y)

    for cx, cy, radius, factor in normalize_mesh_refinements(refinements):
        local = max(target / factor, target * 0.05)
        rx0 = max(x0, cx - radius)
        rx1 = min(x1, cx + radius)
        ry0 = max(y0, cy - radius)
        ry1 = min(y1, cy + radius)
        if rx1 <= rx0 or ry1 <= ry0:
            continue
        rnx = max(1, int(math.ceil((rx1 - rx0) / local)))
        rny = max(1, int(math.ceil((ry1 - ry0) / local)))
        for i in range(rnx + 1):
            add_x(rx0 + (rx1 - rx0) * i / rnx, refinement_x)
        for j in range(rny + 1):
            add_y(ry0 + (ry1 - ry0) * j / rny, refinement_y)

    x_values.update(boundary_x)
    x_values.update(oblique_x)
    x_values.update(curved_x)
    x_values.update(refinement_x)
    x_values.update(split_size_x)
    y_values.update(boundary_y)
    y_values.update(oblique_y)
    y_values.update(curved_y)
    y_values.update(refinement_y)
    y_values.update(split_size_y)
    x_coords = merge_axis_coordinates(x_values, x0, x1, target)
    y_coords = merge_axis_coordinates(y_values, y0, y1, target)
    info = {
        "base_x_count": len(base_x),
        "base_y_count": len(base_y),
        "boundary_layer_added_x_count": count_new_axis_values(boundary_x | oblique_x | curved_x, base_x, target),
        "boundary_layer_added_y_count": count_new_axis_values(boundary_y | oblique_y | curved_y, base_y, target),
        "oblique_boundary_layer_added_line_count": count_new_axis_values(oblique_x, base_x, target) + count_new_axis_values(oblique_y, base_y, target),
        "curved_boundary_layer_added_line_count": count_new_axis_values(curved_x, base_x, target) + count_new_axis_values(curved_y, base_y, target),
        "curved_boundary_segment_count": sum(curve_segment_count(radius, target) for _cx, _cy, radius in tunnels if radius > 0.0),
        "local_refinement_added_line_count": count_new_axis_values(refinement_x, base_x, target) + count_new_axis_values(refinement_y, base_y, target),
        "split_line_constraint_count": len(split_lines),
        "split_line_added_line_count": count_new_axis_values((boundary_x | oblique_x) - split_before_x, base_x, target)
        + count_new_axis_values((boundary_y | oblique_y) - split_before_y, base_y, target),
        "split_line_local_size_added_line_count": count_new_axis_values(split_size_x, base_x, target) + count_new_axis_values(split_size_y, base_y, target),
    }
    return x_coords, y_coords, info


def curve_segment_count(radius: float, target: float) -> int:
    if radius <= 0.0 or target <= 0.0:
        return 0
    by_size = int(math.ceil(2.0 * math.pi * radius / max(0.5 * target, 1.0e-9)))
    by_angle = 24
    return max(16, min(128, max(by_size, by_angle)))


def normalize_mesh_refinements(refinements: list[Any]) -> list[tuple[float, float, float, float]]:
    normalized: list[tuple[float, float, float, float]] = []
    for raw in refinements:
        try:
            if isinstance(raw, dict):
                if str(raw.get("type", raw.get("kind", ""))).lower() in {"split_line", "constraint_line", "division_line", "block_split"}:
                    continue
                center = raw.get("center", [raw.get("cx", 0.0), raw.get("cy", 0.0)])
                cx = float(center[0])
                cy = float(center[1])
                radius = float(raw.get("radius", 0.0))
                factor = float(raw.get("factor", raw.get("refinement", 2.0)))
            else:
                cx, cy, radius, factor = (float(value) for value in raw)
        except (TypeError, ValueError, IndexError):
            continue
        if radius > 0.0 and factor > 1.0:
            normalized.append((cx, cy, radius, factor))
    return normalized


def normalize_mesh_split_lines(refinements: list[Any]) -> list[tuple[Point2D, Point2D, float | None]]:
    normalized: list[tuple[Point2D, Point2D, float | None]] = []
    for raw in refinements:
        if not isinstance(raw, Mapping):
            continue
        kind = str(raw.get("type", raw.get("kind", ""))).lower()
        has_endpoints = any(key in raw for key in ("start", "p1", "x1")) and any(key in raw for key in ("end", "p2", "x2"))
        if kind not in {"split_line", "constraint_line", "division_line", "block_split"} and not has_endpoints:
            continue
        try:
            start = raw.get("start", raw.get("p1", [raw.get("x1", 0.0), raw.get("y1", 0.0)]))
            end = raw.get("end", raw.get("p2", [raw.get("x2", 0.0), raw.get("y2", 0.0)]))
            x1, y1 = float(start[0]), float(start[1])
            x2, y2 = float(end[0]), float(end[1])
        except (TypeError, ValueError, IndexError):
            continue
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            continue
        if math.hypot(x2 - x1, y2 - y1) <= 1.0e-12:
            continue
        target_size: float | None = None
        raw_target = raw.get("target_size", raw.get("size", None))
        if raw_target not in (None, ""):
            try:
                parsed = float(raw_target)
            except (TypeError, ValueError):
                parsed = 0.0
            if parsed > 0.0 and math.isfinite(parsed):
                target_size = parsed
        normalized.append(((x1, y1), (x2, y2), target_size))
    return normalized


def merge_axis_coordinates(values: set[float], lower: float, upper: float, target: float) -> list[float]:
    tol = max(target * 0.2, 1.0e-12)
    coords: list[float] = []
    for value in sorted(values):
        clipped = min(max(float(value), lower), upper)
        if coords and abs(clipped - coords[-1]) <= tol:
            continue
        coords.append(clipped)
    if not coords or abs(coords[0] - lower) > tol:
        coords.insert(0, lower)
    if abs(coords[-1] - upper) > tol:
        coords.append(upper)
    return coords


def count_new_axis_values(values: set[float], base_values: set[float], target: float) -> int:
    tol = max(target * 0.2, 1.0e-12)
    return sum(1 for value in values if all(abs(value - base) > tol for base in base_values))


def axis_coordinate_aspect_ratio(x_coords: list[float], y_coords: list[float]) -> float:
    widths = [b - a for a, b in zip(x_coords, x_coords[1:]) if b > a]
    heights = [b - a for a, b in zip(y_coords, y_coords[1:]) if b > a]
    if not widths or not heights:
        return 0.0
    largest = max(max(widths), max(heights))
    smallest = max(min(min(widths), min(heights)), 1.0e-30)
    return largest / smallest


def project_nodes_to_boundaries(
    nodes: dict[str, list[float]],
    bbox: tuple[float, float, float, float],
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    target: float,
    *,
    polygon_holes: list[list[Point2D]] | None = None,
    snap_tolerance: float | None = None,
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Snap near-boundary grid nodes to exact region/tunnel boundaries."""
    if target <= 0.0:
        raise ValueError("target mesh size must be positive")
    tolerance = float(snap_tolerance) if snap_tolerance is not None else 0.25 * target
    tolerance = max(tolerance, 1.0e-12)
    projected = {str(nid): [float(value[0]), float(value[1])] for nid, value in nodes.items()}
    if not regions and not tunnels and not polygon_holes:
        return projected, {"projected_node_count": 0, "snap_tolerance": tolerance}

    x0, x1, y0, y1 = bbox
    segments = [segment for region in regions for segment in zip(region, region[1:] + region[:1])]
    segments.extend(segment for hole in polygon_holes or [] for segment in zip(hole, hole[1:] + hole[:1]))
    ids = list(projected)
    points = np.ascontiguousarray([projected[nid] for nid in ids], dtype=np.float64)
    projected_points, changed_mask = _project_points_to_boundaries_numba(
        points,
        _boundary_segments_array(segments),
        _tunnels_array(tunnels),
        np.ascontiguousarray([x0, x1, y0, y1], dtype=np.float64),
        tolerance,
    )
    changed = 0
    for index, nid in enumerate(ids):
        if bool(changed_mask[index]):
            projected[nid] = [float(projected_points[index, 0]), float(projected_points[index, 1])]
            changed += 1
    return projected, {"projected_node_count": changed, "snap_tolerance": tolerance}


def near_point_pairs(
    points: Mapping[str, Any] | list[tuple[str, float, float]] | list[Mapping[str, Any]],
    tolerance: float,
    *,
    max_pairs: int = 100,
) -> dict[str, Any]:
    """Return near/duplicate point pairs using sorted grid candidates."""
    tol = float(tolerance)
    if tol <= 0.0:
        raise ValueError("tolerance must be positive")
    labels, coords = _near_point_labels_and_coords(points)
    if coords.shape[0] < 2:
        return {"tolerance": tol, "count": 0, "pairs": []}
    cell_x = np.floor(coords[:, 0] / tol).astype(np.int64)
    cell_y = np.floor(coords[:, 1] / tol).astype(np.int64)
    order = np.lexsort((cell_y, cell_x)).astype(np.int64)
    pair_i, pair_j, distances, total_count = _near_point_pairs_from_sorted_cells_numba(
        np.ascontiguousarray(coords, dtype=np.float64),
        np.ascontiguousarray(order, dtype=np.int64),
        np.ascontiguousarray(cell_x[order], dtype=np.int64),
        np.ascontiguousarray(cell_y[order], dtype=np.int64),
        tol,
        max(0, int(max_pairs)),
    )
    stored = min(int(total_count), int(pair_i.shape[0]))
    pairs = [
        {
            "a": labels[int(pair_i[index])],
            "b": labels[int(pair_j[index])],
            "distance": float(distances[index]),
        }
        for index in range(stored)
    ]
    return {"tolerance": tol, "count": int(total_count), "pairs": pairs}


def _near_point_labels_and_coords(
    points: Mapping[str, Any] | list[tuple[str, float, float]] | list[Mapping[str, Any]],
) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    coords: list[tuple[float, float]] = []
    if isinstance(points, Mapping):
        iterator = ((str(label), value) for label, value in points.items())
        for label, value in iterator:
            point = _mesh_xy_pair(value)
            if point is None:
                continue
            labels.append(label)
            coords.append((float(point[0]), float(point[1])))
    else:
        for index, item in enumerate(points):
            if isinstance(item, Mapping):
                point = _mesh_xy_pair(item.get("point", item.get("xy", item)))
                if point is None:
                    continue
                label = str(item.get("label", item.get("id", index)))
                labels.append(label)
                coords.append((float(point[0]), float(point[1])))
                continue
            if len(item) < 3:
                continue
            labels.append(str(item[0]))
            coords.append((float(item[1]), float(item[2])))
    return labels, np.asarray(coords, dtype=float).reshape((-1, 2))


def recombine_triangles_to_quads(
    nodes: dict[str, list[float]],
    elements: list[dict[str, Any]],
    *,
    regions: list[list[Point2D]] | None = None,
    tunnels: list[tuple[float, float, float]] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
    boundary_segments: list[tuple[Point2D, Point2D]] | None = None,
    min_angle_deg: float = 25.0,
    max_aspect_ratio: float = 8.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    regions = regions or []
    tunnels = tunnels or []
    polygon_holes = polygon_holes or []
    boundary_segments = boundary_segments or []
    edge_to_tris: dict[tuple[str, str], list[int]] = {}
    for idx, element in enumerate(elements):
        if str(element.get("type", "")).upper() != "TRI3":
            continue
        tri_nodes = [str(nid) for nid in element.get("nodes", [])]
        if len(tri_nodes) != 3:
            continue
        for a, b in zip(tri_nodes, tri_nodes[1:] + tri_nodes[:1]):
            key = tuple(sorted((a, b)))
            edge_to_tris.setdefault(key, []).append(idx)

    candidates: list[tuple[float, int, int, list[str], dict[str, float]]] = []
    for tri_indices in edge_to_tris.values():
        if len(tri_indices) != 2:
            continue
        i, j = tri_indices
        ordered = ordered_quad_nodes(nodes, [*elements[i].get("nodes", []), *elements[j].get("nodes", [])])
        if ordered is None:
            continue
        pts = [tuple(nodes[nid]) for nid in ordered]
        quality = quad_quality(pts)
        if quality["area"] <= 1.0e-18:
            continue
        if quality["min_angle_deg"] < min_angle_deg or quality["aspect_ratio"] > max_aspect_ratio:
            continue
        cx = sum(point[0] for point in pts) * 0.25
        cy = sum(point[1] for point in pts) * 0.25
        if not inside_domain(cx, cy, regions, tunnels, polygon_holes=polygon_holes):
            continue
        if any(not (inside_domain(mx, my, regions, tunnels, polygon_holes=polygon_holes) or point_on_boundary((mx, my), boundary_segments)) for mx, my in edge_midpoints(pts)):
            continue
        if boundary_segments and polygon_crosses_boundary(pts, boundary_segments):
            continue
        score = quality["min_angle_deg"] - 2.0 * max(quality["aspect_ratio"] - 1.0, 0.0)
        candidates.append((score, i, j, ordered, quality))

    used: set[int] = set()
    selected: dict[int, tuple[int, list[str]]] = {}
    for _score, i, j, ordered, _quality in sorted(candidates, reverse=True):
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        selected[i] = (j, ordered)

    recombined = 0
    out: list[dict[str, Any]] = []
    for idx, element in enumerate(elements):
        if idx in selected:
            _mate, ordered = selected[idx]
            merged = dict(element)
            merged["type"] = "QUAD4"
            merged["nodes"] = ordered
            merged["source"] = "tri_recombined"
            out.append(merged)
            recombined += 1
        elif idx in used:
            continue
        else:
            out.append(dict(element))

    for eid, element in enumerate(out, start=1):
        element["id"] = str(eid)
    remaining_tri = sum(1 for element in out if str(element.get("type", "")).upper() == "TRI3")
    return out, {"recombined_quad_count": recombined, "remaining_tri_count": remaining_tri}


def subdivide_remaining_triangles_to_quads(
    nodes: dict[str, list[float]],
    elements: list[dict[str, Any]],
    *,
    target: float,
    regions: list[list[Point2D]] | None = None,
    tunnels: list[tuple[float, float, float]] | None = None,
    polygon_holes: list[list[Point2D]] | None = None,
    boundary_segments: list[tuple[Point2D, Point2D]] | None = None,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], dict[str, int]]:
    if target <= 0.0:
        raise ValueError("target mesh size must be positive")
    regions = regions or []
    tunnels = tunnels or []
    polygon_holes = polygon_holes or []
    boundary_segments = boundary_segments or []
    quant = max(target * 1.0e-10, 1.0e-12)
    out_nodes = {str(nid): [float(point[0]), float(point[1])] for nid, point in nodes.items()}
    node_ids: dict[tuple[int, int], str] = {
        (int(round(point[0] / quant)), int(round(point[1] / quant))): str(nid)
        for nid, point in out_nodes.items()
    }
    tunnel_tol = max(target * 1.0e-7, 1.0e-10)

    def node_id(point: Point2D) -> str:
        key = (int(round(point[0] / quant)), int(round(point[1] / quant)))
        if key not in node_ids:
            nid = str(len(out_nodes) + 1)
            node_ids[key] = nid
            out_nodes[nid] = [float(point[0]), float(point[1])]
        return node_ids[key]

    def midpoint(a: Point2D, b: Point2D) -> Point2D:
        mx = 0.5 * (a[0] + b[0])
        my = 0.5 * (a[1] + b[1])
        for cx, cy, radius in tunnels:
            if radius <= 0.0:
                continue
            da = abs(math.hypot(a[0] - cx, a[1] - cy) - radius)
            db = abs(math.hypot(b[0] - cx, b[1] - cy) - radius)
            if da > tunnel_tol or db > tunnel_tol:
                continue
            distance = math.hypot(mx - cx, my - cy)
            if distance <= 1.0e-30:
                continue
            scale = radius / distance
            return (cx + (mx - cx) * scale, cy + (my - cy) * scale)
        return (mx, my)

    out: list[dict[str, Any]] = []
    split_triangles = 0
    split_quads = 0
    for element in elements:
        if str(element.get("type", "")).upper() != "TRI3":
            out.append(dict(element))
            continue
        tri_nodes = [str(nid) for nid in element.get("nodes", [])]
        if len(tri_nodes) != 3 or any(nid not in out_nodes for nid in tri_nodes):
            out.append(dict(element))
            continue
        p0, p1, p2 = (tuple(out_nodes[nid]) for nid in tri_nodes)
        if abs(polygon_area([p0, p1, p2])) <= 1.0e-18:
            out.append(dict(element))
            continue
        m01 = node_id(midpoint(p0, p1))
        m12 = node_id(midpoint(p1, p2))
        m20 = node_id(midpoint(p2, p0))
        center = node_id(((p0[0] + p1[0] + p2[0]) / 3.0, (p0[1] + p1[1] + p2[1]) / 3.0))
        quad_specs = [
            [tri_nodes[0], m01, center, m20],
            [tri_nodes[1], m12, center, m01],
            [tri_nodes[2], m20, center, m12],
        ]
        parent_id = str(element.get("id", ""))
        for quad_nodes in quad_specs:
            pts = [tuple(out_nodes[nid]) for nid in quad_nodes]
            if polygon_area(pts) < 0.0:
                quad_nodes = list(reversed(quad_nodes))
                pts = [tuple(out_nodes[nid]) for nid in quad_nodes]
            quality = quad_quality(pts)
            if quality["area"] <= 1.0e-18 or quality["min_angle_deg"] < 5.0 or quality["aspect_ratio"] > 80.0:
                continue
            samples = [((sum(point[0] for point in pts) / 4.0), (sum(point[1] for point in pts) / 4.0))]
            samples.extend(edge_midpoints(pts))
            if any(not (inside_domain(x, y, regions, tunnels, polygon_holes=polygon_holes) or point_on_boundary((x, y), boundary_segments)) for x, y in samples):
                continue
            if boundary_segments and polygon_crosses_boundary(pts, boundary_segments):
                continue
            quad = dict(element)
            quad["type"] = "QUAD4"
            quad["nodes"] = quad_nodes
            quad["source"] = "tri_subdivided_quad"
            if parent_id:
                quad["parent_triangle"] = parent_id
            out.append(quad)
            split_quads += 1
        split_triangles += 1

    for eid, element in enumerate(out, start=1):
        element["id"] = str(eid)
    return out_nodes, out, {"subdivided_triangle_count": split_triangles, "subdivided_triangle_quad_count": split_quads}


def improve_mesh_quality(
    nodes: dict[str, list[float]],
    elements: list[dict[str, Any]],
    *,
    method: str,
    min_area: float = 1.0e-12,
    min_angle_deg: float = 10.0,
    max_aspect_ratio: float = 20.0,
    max_skew: float = 0.85,
    iterations: int = 5,
    selected_elements: list[str] | None = None,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], dict[str, Any]]:
    """Apply one mesh-quality improvement pass and return a comparison report.

    The helpers intentionally avoid CAD/GUI dependencies. Boundary nodes are
    kept fixed for node-moving passes so existing geometry constraints are not
    silently relaxed.
    """
    method_key = str(method or "laplace").lower().strip()
    before_nodes = _copy_mesh_nodes(nodes)
    before_elements = [dict(element) for element in elements]
    before = mesh_quality_diagnostic(before_nodes, before_elements, min_area=min_area, min_angle_deg=min_angle_deg, max_aspect_ratio=max_aspect_ratio, max_skew=max_skew)
    target_elements = set(str(item) for item in selected_elements or [])
    if method_key in {"laplace", "laplacian", "laplace_smoothing"}:
        after_nodes, after_elements, info = laplacian_smooth_mesh(before_nodes, before_elements, iterations=iterations, relaxation=0.55, selected_elements=target_elements)
    elif method_key in {"node_optimize", "node_optimization", "optimization", "optimize"}:
        after_nodes, after_elements, info = optimize_mesh_node_positions(before_nodes, before_elements, iterations=iterations, selected_elements=target_elements)
    elif method_key in {"local_remesh", "remesh", "subdivide"}:
        bad = target_elements or set(str(item["element"]) for item in before["violations"])
        after_nodes, after_elements, info = local_remesh_bad_elements(before_nodes, before_elements, bad_elements=bad, min_area=min_area, min_angle_deg=min_angle_deg, max_aspect_ratio=max_aspect_ratio, max_skew=max_skew)
    elif method_key in {"quad_topology", "topology", "quad_recombine"}:
        after_nodes, after_elements, info = improve_quad_topology(before_nodes, before_elements, min_angle_deg=min_angle_deg, max_aspect_ratio=max_aspect_ratio, selected_elements=target_elements)
    else:
        after_nodes, after_elements, info = before_nodes, before_elements, {"status": f"unknown method: {method_key}", "changed_nodes": 0, "changed_elements": 0}
    after = mesh_quality_diagnostic(after_nodes, after_elements, min_area=min_area, min_angle_deg=min_angle_deg, max_aspect_ratio=max_aspect_ratio, max_skew=max_skew)
    report = {
        "method": method_key,
        "status": str(info.get("status", "ok")),
        "before": before["summary"],
        "after": after["summary"],
        "before_violation_count": len(before["violations"]),
        "after_violation_count": len(after["violations"]),
        "changed_nodes": int(info.get("changed_nodes", 0)),
        "changed_elements": int(info.get("changed_elements", 0)),
        "details": info,
    }
    report["score_delta"] = _quality_score(after["summary"], len(after["violations"])) - _quality_score(before["summary"], len(before["violations"]))
    return after_nodes, after_elements, report


def mesh_quality_diagnostic(
    nodes: dict[str, list[float]],
    elements: list[dict[str, Any]],
    *,
    min_area: float = 1.0e-12,
    min_angle_deg: float = 10.0,
    max_aspect_ratio: float = 20.0,
    max_skew: float = 0.85,
) -> dict[str, Any]:
    metrics: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    for element in elements:
        pts = _element_points(nodes, element)
        if len(pts) < 3:
            continue
        item = _element_quality_metrics(pts)
        item["element"] = str(element.get("id", ""))
        item["type"] = str(element.get("type", ""))
        metrics.append(item)
        reasons: list[str] = []
        if item["area"] < min_area:
            reasons.append(f"area<{min_area:g}")
        if item["min_angle_deg"] < min_angle_deg:
            reasons.append(f"min_angle<{min_angle_deg:g}")
        if item["aspect_ratio"] > max_aspect_ratio:
            reasons.append(f"aspect>{max_aspect_ratio:g}")
        if len(pts) == 4 and item["skew"] > max_skew:
            reasons.append(f"skew>{max_skew:g}")
        if reasons:
            bad = dict(item)
            bad["reason"] = ", ".join(reasons)
            violations.append(bad)
    if metrics:
        min_angle = min(float(item["min_angle_deg"]) for item in metrics)
        max_aspect = max(float(item["aspect_ratio"]) for item in metrics)
        max_skew_value = max(float(item["skew"]) for item in metrics)
        min_area_value = min(float(item["area"]) for item in metrics)
    else:
        min_angle = 0.0
        max_aspect = 0.0
        max_skew_value = 0.0
        min_area_value = 0.0
    return {
        "summary": {
            "element_count": len(elements),
            "node_count": len(nodes),
            "min_area": min_area_value,
            "min_angle_deg": min_angle,
            "max_aspect_ratio": max_aspect,
            "max_skew": max_skew_value,
        },
        "metrics": metrics,
        "violations": violations,
    }


def laplacian_smooth_mesh(
    nodes: dict[str, list[float]],
    elements: list[dict[str, Any]],
    *,
    iterations: int = 5,
    relaxation: float = 0.55,
    selected_elements: set[str] | None = None,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], dict[str, Any]]:
    out_nodes = _copy_mesh_nodes(nodes)
    out_elements = [dict(element) for element in elements]
    neighbors, adjacent = _node_graph(out_elements)
    fixed = _boundary_nodes(out_elements)
    movable = _selected_nodes(out_elements, selected_elements) if selected_elements else set(out_nodes)
    changed = 0
    for _ in range(max(1, int(iterations))):
        trial = _copy_mesh_nodes(out_nodes)
        for nid in sorted(movable, key=_natural_key):
            if nid in fixed or nid not in neighbors or nid not in out_nodes:
                continue
            near = [other for other in neighbors[nid] if other in out_nodes]
            if not near:
                continue
            avg_x = sum(out_nodes[other][0] for other in near) / len(near)
            avg_y = sum(out_nodes[other][1] for other in near) / len(near)
            old = out_nodes[nid]
            candidate = [old[0] * (1.0 - relaxation) + avg_x * relaxation, old[1] * (1.0 - relaxation) + avg_y * relaxation]
            if _node_move_improves(out_nodes, out_elements, adjacent, nid, candidate):
                trial[nid] = candidate
                changed += 1
        out_nodes = trial
    return out_nodes, out_elements, {"status": "ok", "changed_nodes": changed, "changed_elements": 0, "fixed_nodes": len(fixed)}


def optimize_mesh_node_positions(
    nodes: dict[str, list[float]],
    elements: list[dict[str, Any]],
    *,
    iterations: int = 5,
    selected_elements: set[str] | None = None,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], dict[str, Any]]:
    out_nodes = _copy_mesh_nodes(nodes)
    out_elements = [dict(element) for element in elements]
    neighbors, adjacent = _node_graph(out_elements)
    fixed = _boundary_nodes(out_elements)
    movable = _selected_nodes(out_elements, selected_elements) if selected_elements else set(out_nodes)
    edge_lengths = [
        math.hypot(nodes[a][0] - nodes[b][0], nodes[a][1] - nodes[b][1])
        for a, values in neighbors.items()
        for b in values
        if a in nodes and b in nodes
    ]
    base_step = (sum(edge_lengths) / len(edge_lengths) if edge_lengths else 1.0) * 0.18
    changed = 0
    for iteration in range(max(1, int(iterations))):
        step = base_step * (0.55**iteration)
        for nid in sorted(movable, key=_natural_key):
            if nid in fixed or nid not in out_nodes or nid not in adjacent:
                continue
            old = out_nodes[nid]
            best = list(old)
            best_score = _adjacent_quality_score(out_nodes, out_elements, adjacent[nid])
            for dx, dy in ((step, 0.0), (-step, 0.0), (0.0, step), (0.0, -step), (step, step), (step, -step), (-step, step), (-step, -step)):
                candidate = [old[0] + dx, old[1] + dy]
                if not _node_move_valid(out_nodes, out_elements, adjacent[nid], nid, candidate):
                    continue
                trial = _copy_mesh_nodes(out_nodes)
                trial[nid] = candidate
                score = _adjacent_quality_score(trial, out_elements, adjacent[nid])
                if score > best_score + 1.0e-12:
                    best_score = score
                    best = candidate
            if math.hypot(best[0] - old[0], best[1] - old[1]) > 1.0e-12:
                out_nodes[nid] = best
                changed += 1
    return out_nodes, out_elements, {"status": "ok", "changed_nodes": changed, "changed_elements": 0, "fixed_nodes": len(fixed)}


def local_remesh_bad_elements(
    nodes: dict[str, list[float]],
    elements: list[dict[str, Any]],
    *,
    bad_elements: set[str],
    min_area: float = 1.0e-12,
    min_angle_deg: float = 10.0,
    max_aspect_ratio: float = 20.0,
    max_skew: float = 0.85,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], dict[str, Any]]:
    out_nodes = _copy_mesh_nodes(nodes)
    node_lookup = {(round(point[0], 12), round(point[1], 12)): nid for nid, point in out_nodes.items()}

    def get_node(point: Point2D) -> str:
        key = (round(point[0], 12), round(point[1], 12))
        if key in node_lookup:
            return node_lookup[key]
        nid = str(max([int(raw) for raw in out_nodes if str(raw).isdigit()] or [0]) + 1)
        while nid in out_nodes:
            nid = str(int(nid) + 1)
        out_nodes[nid] = [float(point[0]), float(point[1])]
        node_lookup[key] = nid
        return nid

    out_elements: list[dict[str, Any]] = []
    changed = 0
    new_eid = 1
    diagnostics = mesh_quality_diagnostic(nodes, elements, min_area=min_area, min_angle_deg=min_angle_deg, max_aspect_ratio=max_aspect_ratio, max_skew=max_skew)
    auto_bad = {str(item["element"]) for item in diagnostics["violations"]}
    targets = set(bad_elements) | auto_bad
    for element in elements:
        eid = str(element.get("id", ""))
        pts = _element_points(out_nodes, element)
        if eid not in targets or len(pts) not in {3, 4}:
            item = dict(element)
            item["id"] = str(new_eid)
            out_elements.append(item)
            new_eid += 1
            continue
        if len(pts) == 4:
            p0, p1, p2, p3 = pts
            m01 = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
            m12 = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            m23 = ((p2[0] + p3[0]) * 0.5, (p2[1] + p3[1]) * 0.5)
            m30 = ((p3[0] + p0[0]) * 0.5, (p3[1] + p0[1]) * 0.5)
            center = ((p0[0] + p1[0] + p2[0] + p3[0]) * 0.25, (p0[1] + p1[1] + p2[1] + p3[1]) * 0.25)
            original = [str(nid) for nid in element.get("nodes", [])[:4]]
            quads = [
                [original[0], get_node(m01), get_node(center), get_node(m30)],
                [get_node(m01), original[1], get_node(m12), get_node(center)],
                [get_node(center), get_node(m12), original[2], get_node(m23)],
                [get_node(m30), get_node(center), get_node(m23), original[3]],
            ]
        else:
            p0, p1, p2 = pts
            m01 = ((p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5)
            m12 = ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)
            m20 = ((p2[0] + p0[0]) * 0.5, (p2[1] + p0[1]) * 0.5)
            original = [str(nid) for nid in element.get("nodes", [])[:3]]
            quads = [[original[0], get_node(m01), get_node(m20)], [get_node(m01), original[1], get_node(m12)], [get_node(m20), get_node(m12), original[2]], [get_node(m01), get_node(m12), get_node(m20)]]
        for conn in quads:
            item = dict(element)
            item["id"] = str(new_eid)
            item["type"] = "QUAD4" if len(conn) == 4 else "TRI3"
            item["nodes"] = conn
            item["source"] = "quality_local_remesh"
            item["parent_element"] = eid
            out_elements.append(item)
            new_eid += 1
        changed += 1
    return out_nodes, out_elements, {"status": "ok", "changed_nodes": max(0, len(out_nodes) - len(nodes)), "changed_elements": changed}


def improve_quad_topology(
    nodes: dict[str, list[float]],
    elements: list[dict[str, Any]],
    *,
    min_angle_deg: float = 10.0,
    max_aspect_ratio: float = 20.0,
    selected_elements: set[str] | None = None,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], dict[str, Any]]:
    out_nodes = _copy_mesh_nodes(nodes)
    out_elements = [dict(element) for element in elements]
    recombined, recombine_info = recombine_triangles_to_quads(out_nodes, out_elements, min_angle_deg=min_angle_deg, max_aspect_ratio=max_aspect_ratio)
    if int(recombine_info.get("recombined_quad_count", 0)) > 0:
        return out_nodes, recombined, {"status": "ok", "changed_nodes": 0, "changed_elements": int(recombine_info.get("recombined_quad_count", 0)), **recombine_info}
    edge_to_elements: dict[tuple[str, str], list[int]] = {}
    for idx, element in enumerate(out_elements):
        if str(element.get("type", "")).upper() != "QUAD4":
            continue
        conn = [str(nid) for nid in element.get("nodes", [])[:4]]
        for a, b in zip(conn, conn[1:] + conn[:1]):
            edge_to_elements.setdefault(tuple(sorted((a, b), key=_natural_key)), []).append(idx)
    used: set[int] = set()
    changed = 0
    for pair in edge_to_elements.values():
        if len(pair) != 2 or pair[0] in used or pair[1] in used:
            continue
        i, j = pair
        if selected_elements and str(out_elements[i].get("id", "")) not in selected_elements and str(out_elements[j].get("id", "")) not in selected_elements:
            continue
        union = list(dict.fromkeys(str(nid) for nid in out_elements[i].get("nodes", [])[:4] + out_elements[j].get("nodes", [])[:4]))
        if len(union) != 6 or any(nid not in out_nodes for nid in union):
            continue
        cx = sum(out_nodes[nid][0] for nid in union) / 6.0
        cy = sum(out_nodes[nid][1] for nid in union) / 6.0
        ring = sorted(union, key=lambda nid: math.atan2(out_nodes[nid][1] - cy, out_nodes[nid][0] - cx))
        current = _adjacent_quality_score(out_nodes, out_elements, [i, j])
        best_pair: tuple[list[str], list[str]] | None = None
        best_score = current
        for start in range(3):
            q1 = [ring[start % 6], ring[(start + 1) % 6], ring[(start + 2) % 6], ring[(start + 3) % 6]]
            q2 = [ring[(start + 3) % 6], ring[(start + 4) % 6], ring[(start + 5) % 6], ring[start % 6]]
            trial_elements = [dict(out_elements[i]), dict(out_elements[j])]
            trial_elements[0]["nodes"] = q1
            trial_elements[1]["nodes"] = q2
            score = _adjacent_quality_score(out_nodes, trial_elements, [0, 1])
            if score > best_score + 1.0e-12:
                best_score = score
                best_pair = (q1, q2)
        if best_pair is None:
            continue
        out_elements[i]["nodes"] = best_pair[0]
        out_elements[j]["nodes"] = best_pair[1]
        out_elements[i]["source"] = "quad_topology_improved"
        out_elements[j]["source"] = "quad_topology_improved"
        used.update(pair)
        changed += 2
    return out_nodes, out_elements, {"status": "ok", "changed_nodes": 0, "changed_elements": changed}


def circular_tunnel_node_sets(nodes: dict[str, list[float]], tunnels: list[tuple[float, float, float]], target: float) -> dict[str, list[str]]:
    if not tunnels:
        return {}
    tol = max(target * 1.0e-7, 1.0e-10)
    sets: dict[str, list[str]] = {}
    combined: list[str] = []
    for index, (cx, cy, radius) in enumerate(tunnels, start=1):
        if radius <= 0.0:
            continue
        ids = [
            nid
            for nid, point in nodes.items()
            if abs(math.hypot(point[0] - cx, point[1] - cy) - radius) <= tol
        ]
        if not ids:
            continue
        key = f"tunnel_boundary_{index}"
        sets[key] = ids
        combined.extend(ids)
    if combined:
        sets["tunnel_boundary"] = list(dict.fromkeys(combined))
    return sets


def minimum_circular_tunnel_clearance(tunnels: list[tuple[float, float, float]]) -> float:
    active = [(cx, cy, radius) for cx, cy, radius in tunnels if radius > 0.0]
    if len(active) < 2:
        return 0.0
    return min(
        math.hypot(ax - bx, ay - by) - ar - br
        for index, (ax, ay, ar) in enumerate(active)
        for bx, by, br in active[index + 1 :]
    )


def ordered_quad_nodes(nodes: dict[str, list[float]], raw_nodes: list[Any]) -> list[str] | None:
    unique = list(dict.fromkeys(str(nid) for nid in raw_nodes))
    if len(unique) != 4 or any(nid not in nodes for nid in unique):
        return None
    cx = sum(nodes[nid][0] for nid in unique) * 0.25
    cy = sum(nodes[nid][1] for nid in unique) * 0.25
    ordered = sorted(unique, key=lambda nid: math.atan2(nodes[nid][1] - cy, nodes[nid][0] - cx))
    pts = [tuple(nodes[nid]) for nid in ordered]
    if polygon_area(pts) < 0.0:
        ordered.reverse()
        pts = [tuple(nodes[nid]) for nid in ordered]
    if not is_convex_polygon(pts):
        return None
    return ordered


def _copy_mesh_nodes(nodes: Mapping[str, Any]) -> dict[str, list[float]]:
    return {str(nid): [float(point[0]), float(point[1])] for nid, point in nodes.items()}


def _natural_key(value: Any) -> tuple[int, Any]:
    raw = str(value)
    return (0, int(raw)) if raw.isdigit() else (1, raw)


def _element_points(nodes: Mapping[str, list[float]], element: Mapping[str, Any]) -> list[Point2D]:
    etype = str(element.get("type", "")).upper()
    corner_count = 3 if etype.startswith("TRI") else 4 if etype.startswith("QUAD") else len(element.get("nodes", []))
    points: list[Point2D] = []
    for nid in list(element.get("nodes", []))[:corner_count]:
        key = str(nid)
        if key in nodes:
            points.append((float(nodes[key][0]), float(nodes[key][1])))
    return points


def _element_quality_metrics(points: list[Point2D]) -> dict[str, float]:
    if len(points) < 3:
        return {"area": 0.0, "min_angle_deg": 0.0, "aspect_ratio": math.inf, "skew": math.inf}
    area = abs(polygon_area(points))
    edges = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:] + points[:1])]
    positive = [length for length in edges if length > 1.0e-30]
    aspect = max(positive) / min(positive) if positive else math.inf
    angles: list[float] = []
    for index, point in enumerate(points):
        prev = points[index - 1]
        nxt = points[(index + 1) % len(points)]
        v1 = (prev[0] - point[0], prev[1] - point[1])
        v2 = (nxt[0] - point[0], nxt[1] - point[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 <= 1.0e-30 or n2 <= 1.0e-30:
            angles.append(0.0)
            continue
        cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        angles.append(math.degrees(math.acos(cosv)))
    skew = max((abs(angle - 90.0) / 90.0 for angle in angles), default=0.0) if len(points) == 4 else 0.0
    return {"area": area, "min_angle_deg": min(angles) if angles else 0.0, "aspect_ratio": aspect, "skew": skew}


def _node_graph(elements: list[dict[str, Any]]) -> tuple[dict[str, set[str]], dict[str, list[int]]]:
    neighbors: dict[str, set[str]] = {}
    adjacent: dict[str, list[int]] = {}
    for index, element in enumerate(elements):
        conn = [str(nid) for nid in element.get("nodes", [])]
        corner_count = 3 if str(element.get("type", "")).upper().startswith("TRI") else 4 if str(element.get("type", "")).upper().startswith("QUAD") else len(conn)
        conn = conn[:corner_count]
        for nid in conn:
            adjacent.setdefault(nid, []).append(index)
            neighbors.setdefault(nid, set()).update(other for other in conn if other != nid)
    return neighbors, adjacent


def _boundary_nodes(elements: list[dict[str, Any]]) -> set[str]:
    edge_count: dict[tuple[str, str], int] = {}
    for element in elements:
        conn = [str(nid) for nid in element.get("nodes", [])]
        corner_count = 3 if str(element.get("type", "")).upper().startswith("TRI") else 4 if str(element.get("type", "")).upper().startswith("QUAD") else len(conn)
        conn = conn[:corner_count]
        for a, b in zip(conn, conn[1:] + conn[:1]):
            edge = tuple(sorted((a, b), key=_natural_key))
            edge_count[edge] = edge_count.get(edge, 0) + 1
    fixed: set[str] = set()
    for edge, count in edge_count.items():
        if count == 1:
            fixed.update(edge)
    return fixed


def _selected_nodes(elements: list[dict[str, Any]], selected_elements: set[str] | None) -> set[str]:
    if not selected_elements:
        return {str(nid) for element in elements for nid in element.get("nodes", [])}
    return {str(nid) for element in elements if str(element.get("id", "")) in selected_elements for nid in element.get("nodes", [])}


def _node_move_valid(nodes: dict[str, list[float]], elements: list[dict[str, Any]], indices: list[int], nid: str, candidate: list[float]) -> bool:
    trial = _copy_mesh_nodes(nodes)
    trial[nid] = [float(candidate[0]), float(candidate[1])]
    for index in indices:
        pts = _element_points(trial, elements[index])
        metrics = _element_quality_metrics(pts)
        if metrics["area"] <= 1.0e-18 or metrics["min_angle_deg"] <= 1.0e-6:
            return False
        if len(pts) == 4 and not is_convex_polygon(pts):
            return False
    return True


def _node_move_improves(nodes: dict[str, list[float]], elements: list[dict[str, Any]], adjacent: dict[str, list[int]], nid: str, candidate: list[float]) -> bool:
    indices = adjacent.get(nid, [])
    if not indices or not _node_move_valid(nodes, elements, indices, nid, candidate):
        return False
    before = _adjacent_quality_score(nodes, elements, indices)
    trial = _copy_mesh_nodes(nodes)
    trial[nid] = [float(candidate[0]), float(candidate[1])]
    after = _adjacent_quality_score(trial, elements, indices)
    return after > before + 1.0e-12


def _adjacent_quality_score(nodes: dict[str, list[float]], elements: list[dict[str, Any]], indices: list[int]) -> float:
    if not indices:
        return -math.inf
    scores = []
    for index in indices:
        pts = _element_points(nodes, elements[index])
        metrics = _element_quality_metrics(pts)
        if metrics["area"] <= 1.0e-18:
            scores.append(-1.0e9)
            continue
        aspect_penalty = math.log(max(metrics["aspect_ratio"], 1.0))
        skew_penalty = metrics["skew"] * 20.0 if len(pts) == 4 else 0.0
        scores.append(metrics["min_angle_deg"] - aspect_penalty * 12.0 - skew_penalty)
    return min(scores) + 0.05 * sum(scores)


def _quality_score(summary: Mapping[str, Any], violation_count: int) -> float:
    try:
        min_angle = float(summary.get("min_angle_deg", 0.0))
        aspect = float(summary.get("max_aspect_ratio", 0.0))
        skew = float(summary.get("max_skew", 0.0))
    except (TypeError, ValueError):
        return -math.inf
    return min_angle - 10.0 * math.log(max(aspect, 1.0)) - 15.0 * skew - 100.0 * violation_count


def classify_element_regions(nodes: dict[str, list[float]], elements: list[dict[str, Any]], regions: list[list[Point2D]]) -> dict[str, list[str]]:
    region_sets: dict[str, list[str]] = {f"region_{i + 1}": [] for i in range(len(regions))}
    if not regions:
        return region_sets
    centroids: list[list[float]] = []
    centroid_elements: list[dict[str, Any]] = []
    for element in elements:
        pts = [nodes[str(nid)] for nid in element.get("nodes", []) if str(nid) in nodes]
        if not pts:
            continue
        cx = sum(point[0] for point in pts) / len(pts)
        cy = sum(point[1] for point in pts) / len(pts)
        centroids.append([float(cx), float(cy)])
        centroid_elements.append(element)
    if not centroids:
        return region_sets
    region_indices = first_region_indices(np.ascontiguousarray(centroids, dtype=np.float64), regions)
    for element, region in zip(centroid_elements, region_indices):
        region = int(region)
        if region >= 0:
            region_sets[f"region_{region + 1}"].append(str(element.get("id", "")))
    return region_sets


def mesh_quality_summary(nodes: dict[str, list[float]], elements: list[dict[str, Any]]) -> dict[str, float]:
    quad_qualities = [quad_quality([tuple(nodes[str(nid)]) for nid in element.get("nodes", [])]) for element in elements if str(element.get("type", "")).upper() == "QUAD4" and len(element.get("nodes", [])) == 4]
    if not quad_qualities:
        return {"min_quad_angle_deg": 0.0, "max_quad_aspect_ratio": 0.0}
    return {
        "min_quad_angle_deg": min(item["min_angle_deg"] for item in quad_qualities),
        "max_quad_aspect_ratio": max(item["aspect_ratio"] for item in quad_qualities),
    }


def quad_quality(points: list[Point2D]) -> dict[str, float]:
    if len(points) != 4:
        return {"area": 0.0, "min_angle_deg": 0.0, "aspect_ratio": math.inf}
    edges = [math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(points, points[1:] + points[:1])]
    min_edge = max(min(edges), 1.0e-30)
    angles = [corner_angle(points[i - 1], points[i], points[(i + 1) % 4]) for i in range(4)]
    return {
        "area": abs(polygon_area(points)),
        "min_angle_deg": min(angles),
        "aspect_ratio": max(edges) / min_edge,
    }


def edge_midpoints(points: list[Point2D]) -> list[Point2D]:
    return [((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5) for a, b in zip(points, points[1:] + points[:1])]


def corner_angle(prev: Point2D, point: Point2D, nxt: Point2D) -> float:
    ax = prev[0] - point[0]
    ay = prev[1] - point[1]
    bx = nxt[0] - point[0]
    by = nxt[1] - point[1]
    denom = max(math.hypot(ax, ay) * math.hypot(bx, by), 1.0e-30)
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / denom))
    return math.degrees(math.acos(cosine))


def polygon_area(points: list[Point2D]) -> float:
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))


def is_convex_polygon(points: list[Point2D]) -> bool:
    if len(points) < 4:
        return False
    signs: list[float] = []
    for i, point in enumerate(points):
        prev = points[i - 1]
        nxt = points[(i + 1) % len(points)]
        cross = (point[0] - prev[0]) * (nxt[1] - point[1]) - (point[1] - prev[1]) * (nxt[0] - point[0])
        if abs(cross) > 1.0e-12:
            signs.append(cross)
    return bool(signs) and (all(value > 0.0 for value in signs) or all(value < 0.0 for value in signs))


def prune_unused_nodes(
    nodes: dict[str, list[float]],
    elements: list[dict[str, Any]],
    node_sets: dict[str, list[str]],
) -> tuple[dict[str, list[float]], list[dict[str, Any]], dict[str, list[str]]]:
    used = {str(nid) for element in elements for nid in element.get("nodes", [])}
    ordered = [nid for nid in nodes if nid in used]
    mapping = {old: str(index + 1) for index, old in enumerate(ordered)}
    new_nodes = {mapping[old]: nodes[old] for old in ordered}
    new_elements: list[dict[str, Any]] = []
    for element in elements:
        updated = dict(element)
        updated["nodes"] = [mapping[str(nid)] for nid in element.get("nodes", [])]
        new_elements.append(updated)
    new_sets: dict[str, list[str]] = {}
    for name, ids in node_sets.items():
        mapped = [mapping[str(nid)] for nid in ids if str(nid) in mapping]
        if mapped:
            new_sets[name] = mapped
    new_sets["all"] = list(new_nodes)
    return new_nodes, new_elements, new_sets


def geometry_boundary_segments(
    bbox: tuple[float, float, float, float],
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    target: float,
    *,
    polygon_holes: list[list[Point2D]] | None = None,
) -> list[tuple[Point2D, Point2D]]:
    x0, x1, y0, y1 = bbox
    segments: list[tuple[Point2D, Point2D]] = []
    if regions:
        for region in regions:
            segments.extend(zip(region, region[1:] + region[:1]))
    else:
        rect = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        segments.extend(zip(rect, rect[1:] + rect[:1]))
    for hole in polygon_holes or []:
        segments.extend(zip(hole, hole[1:] + hole[:1]))
    for cx, cy, radius in tunnels:
        count = curve_segment_count(radius, target)
        ring = [(cx + radius * math.cos(2.0 * math.pi * i / count), cy + radius * math.sin(2.0 * math.pi * i / count)) for i in range(count)]
        segments.extend(zip(ring, ring[1:] + ring[:1]))
    return segments


def inside_domain(
    x: float,
    y: float,
    regions: list[list[Point2D]],
    tunnels: list[tuple[float, float, float]],
    *,
    polygon_holes: list[list[Point2D]] | None = None,
) -> bool:
    inside_region = True if not regions else any(point_in_polygon(x, y, region) for region in regions)
    inside_tunnel = any(math.hypot(x - cx, y - cy) <= radius for cx, cy, radius in tunnels)
    inside_hole = any(point_in_polygon(x, y, hole) for hole in polygon_holes or [])
    return inside_region and not inside_tunnel and not inside_hole


def first_region_index(x: float, y: float, regions: list[list[Point2D]]) -> int | None:
    for index, region in enumerate(regions):
        if point_in_polygon(x, y, region):
            return index
    return None


def _flatten_regions(regions: list[list[Point2D]]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    flat: list[list[float]] = []
    for region in regions:
        flat.extend([[float(x), float(y)] for x, y in region])
        offsets.append(len(flat))
    return np.ascontiguousarray(offsets, dtype=np.int64), np.ascontiguousarray(flat, dtype=np.float64)


@njit(cache=True)
def _point_in_polygon_array_numba(x: float, y: float, polygon: np.ndarray, start: int, stop: int) -> bool:
    inside = False
    j = stop - 1
    for i in range(start, stop):
        xi = polygon[i, 0]
        yi = polygon[i, 1]
        xj = polygon[j, 0]
        yj = polygon[j, 1]
        denom = yj - yi
        if abs(denom) <= 1.0e-30:
            denom = 1.0e-30
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / denom + xi
        if intersects:
            inside = not inside
        j = i
    return inside


@njit(cache=True)
def _first_region_indices_numba(points: np.ndarray, offsets: np.ndarray, region_points: np.ndarray) -> np.ndarray:
    out = np.full(points.shape[0], -1, dtype=np.int64)
    for p in range(points.shape[0]):
        x = points[p, 0]
        y = points[p, 1]
        for region in range(offsets.shape[0] - 1):
            if _point_in_polygon_array_numba(x, y, region_points, offsets[region], offsets[region + 1]):
                out[p] = region
                break
    return out


def first_region_indices(points: np.ndarray, regions: list[list[Point2D]]) -> np.ndarray:
    if points.size == 0 or not regions:
        return np.full(points.shape[0], -1, dtype=np.int64)
    offsets, region_points = _flatten_regions(regions)
    return _first_region_indices_numba(np.ascontiguousarray(points, dtype=np.float64), offsets, region_points)


def polygon_crosses_boundary(points: list[Point2D], boundary_segments: list[tuple[Point2D, Point2D]]) -> bool:
    for a, b in zip(points, points[1:] + points[:1]):
        for c, d in boundary_segments:
            hit = segment_intersection(a, b, c, d)
            if hit is None:
                continue
            _ta, _tb, x, y = hit
            p = (x, y)
            if point_matches_any(p, (a, b, c, d)):
                continue
            return True
    return False


def _polygon_points_array(points: list[Point2D]) -> np.ndarray:
    if not points:
        return np.empty((0, 2), dtype=np.float64)
    return np.ascontiguousarray([[float(x), float(y)] for x, y in points], dtype=np.float64)


@njit(cache=True)
def _point_matches_endpoint_numba(px: float, py: float, points: np.ndarray, tol: float) -> bool:
    for i in range(points.shape[0]):
        dx = px - points[i, 0]
        dy = py - points[i, 1]
        if math.sqrt(dx * dx + dy * dy) <= tol:
            return True
    return False


@njit(cache=True)
def _segment_intersection_numba(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float) -> tuple[bool, float, float]:
    rx = bx - ax
    ry = by - ay
    sx = dx - cx
    sy = dy - cy
    denom = rx * sy - ry * sx
    if abs(denom) <= 1.0e-12:
        return False, 0.0, 0.0
    qpx = cx - ax
    qpy = cy - ay
    t = (qpx * sy - qpy * sx) / denom
    u = (qpx * ry - qpy * rx) / denom
    if -1.0e-10 <= t <= 1.0 + 1.0e-10 and -1.0e-10 <= u <= 1.0 + 1.0e-10:
        return True, ax + t * rx, ay + t * ry
    return False, 0.0, 0.0


@njit(cache=True)
def _segment_bbox_overlap_numba(ax: float, ay: float, bx: float, by: float, cx: float, cy: float, dx: float, dy: float, tol: float) -> bool:
    amin_x = min(ax, bx) - tol
    amax_x = max(ax, bx) + tol
    amin_y = min(ay, by) - tol
    amax_y = max(ay, by) + tol
    bmin_x = min(cx, dx) - tol
    bmax_x = max(cx, dx) + tol
    bmin_y = min(cy, dy) - tol
    bmax_y = max(cy, dy) + tol
    return not (amax_x < bmin_x or bmax_x < amin_x or amax_y < bmin_y or bmax_y < amin_y)


@njit(cache=True)
def _polygon_bbox_overlap_numba(a: np.ndarray, b: np.ndarray, tol: float) -> bool:
    amin_x = 1.0e300
    amax_x = -1.0e300
    amin_y = 1.0e300
    amax_y = -1.0e300
    bmin_x = 1.0e300
    bmax_x = -1.0e300
    bmin_y = 1.0e300
    bmax_y = -1.0e300
    for i in range(a.shape[0]):
        x = a[i, 0]
        y = a[i, 1]
        if x < amin_x:
            amin_x = x
        if x > amax_x:
            amax_x = x
        if y < amin_y:
            amin_y = y
        if y > amax_y:
            amax_y = y
    for i in range(b.shape[0]):
        x = b[i, 0]
        y = b[i, 1]
        if x < bmin_x:
            bmin_x = x
        if x > bmax_x:
            bmax_x = x
        if y < bmin_y:
            bmin_y = y
        if y > bmax_y:
            bmax_y = y
    return not (amax_x + tol < bmin_x - tol or bmax_x + tol < amin_x - tol or amax_y + tol < bmin_y - tol or bmax_y + tol < amin_y - tol)


@njit(cache=True)
def _polygon_has_crossing_edges_numba(points: np.ndarray) -> bool:
    n = points.shape[0]
    if n < 4:
        return False
    last = n - 1
    for i in range(n):
        ai = i
        bi = 0 if i == last else i + 1
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == last):
                continue
            cj = j
            dj = 0 if j == last else j + 1
            if not _segment_bbox_overlap_numba(
                points[ai, 0],
                points[ai, 1],
                points[bi, 0],
                points[bi, 1],
                points[cj, 0],
                points[cj, 1],
                points[dj, 0],
                points[dj, 1],
                1.0e-10,
            ):
                continue
            hit, _x, _y = _segment_intersection_numba(
                points[ai, 0],
                points[ai, 1],
                points[bi, 0],
                points[bi, 1],
                points[cj, 0],
                points[cj, 1],
                points[dj, 0],
                points[dj, 1],
            )
            if hit:
                return True
    return False


@njit(cache=True)
def _polygons_overlap_numba(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape[0] < 3 or b.shape[0] < 3:
        return False
    if not _polygon_bbox_overlap_numba(a, b, 1.0e-10):
        return False
    endpoints = np.empty((4, 2), dtype=np.float64)
    for i in range(a.shape[0]):
        ai1 = 0 if i == a.shape[0] - 1 else i + 1
        for j in range(b.shape[0]):
            bj1 = 0 if j == b.shape[0] - 1 else j + 1
            if not _segment_bbox_overlap_numba(a[i, 0], a[i, 1], a[ai1, 0], a[ai1, 1], b[j, 0], b[j, 1], b[bj1, 0], b[bj1, 1], 1.0e-10):
                continue
            hit, hx, hy = _segment_intersection_numba(a[i, 0], a[i, 1], a[ai1, 0], a[ai1, 1], b[j, 0], b[j, 1], b[bj1, 0], b[bj1, 1])
            if hit:
                endpoints[0, 0] = a[i, 0]
                endpoints[0, 1] = a[i, 1]
                endpoints[1, 0] = a[ai1, 0]
                endpoints[1, 1] = a[ai1, 1]
                endpoints[2, 0] = b[j, 0]
                endpoints[2, 1] = b[j, 1]
                endpoints[3, 0] = b[bj1, 0]
                endpoints[3, 1] = b[bj1, 1]
                if not _point_matches_endpoint_numba(hx, hy, endpoints, 1.0e-8):
                    return True
    for i in range(a.shape[0]):
        if _point_in_polygon_array_numba(a[i, 0], a[i, 1], b, 0, b.shape[0]):
            return True
    for i in range(b.shape[0]):
        if _point_in_polygon_array_numba(b[i, 0], b[i, 1], a, 0, a.shape[0]):
            return True
    return False


def polygon_has_crossing_edges(points: list[Point2D]) -> bool:
    return bool(_polygon_has_crossing_edges_numba(_polygon_points_array(points)))


def polygons_overlap(a: list[Point2D], b: list[Point2D]) -> bool:
    return bool(_polygons_overlap_numba(_polygon_points_array(a), _polygon_points_array(b)))


def polygons_any_overlap(polygons: list[list[Point2D]], *, tol: float = 0.0) -> bool:
    for left_index, right_index in _polygon_pair_candidates(polygons, tol=tol):
        if polygons_overlap(polygons[left_index], polygons[right_index]):
            return True
    return False


def _polygon_pair_candidates(polygons: list[list[Point2D]], *, tol: float = 0.0) -> list[tuple[int, int]]:
    if len(polygons) < 2:
        return []
    pad = max(float(tol), 0.0)
    entries: list[tuple[float, float, float, float, int]] = []
    for index, polygon in enumerate(polygons):
        if not polygon:
            continue
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
        entries.append((min(xs) - pad, max(xs) + pad, min(ys) - pad, max(ys) + pad, index))
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


def polygons_have_positive_area_overlap(a: list[Point2D], b: list[Point2D], *, tol: float = 1.0e-8) -> bool:
    a_segments = list(zip(a, a[1:] + a[:1]))
    b_segments = list(zip(b, b[1:] + b[:1]))
    for point in [*a, *edge_midpoints(a), polygon_centroid(a)]:
        if point_in_polygon(point[0], point[1], b) and not point_on_boundary(point, b_segments, tol=tol):
            return True
    for point in [*b, *edge_midpoints(b), polygon_centroid(b)]:
        if point_in_polygon(point[0], point[1], a) and not point_on_boundary(point, a_segments, tol=tol):
            return True
    for p0, p1 in a_segments:
        for q0, q1 in b_segments:
            hit = segment_intersection(p0, p1, q0, q1)
            if hit is None:
                continue
            hx, hy = hit[2], hit[3]
            if point_matches_any((hx, hy), (p0, p1, q0, q1), tol=tol):
                continue
            return True
    return False


def _shared_node_ids_between_sets(node_sets: Mapping[str, list[str]], names: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    for name in names:
        for nid in set(str(value) for value in node_sets.get(name, [])):
            counts[nid] = counts.get(nid, 0) + 1
    return [nid for nid, count in counts.items() if count > 1]


def point_matches_any(point: Point2D, candidates: tuple[Point2D, ...], tol: float = 1.0e-8) -> bool:
    return any(math.hypot(point[0] - candidate[0], point[1] - candidate[1]) <= tol for candidate in candidates)


@njit(cache=True)
def _point_segment_distance_numba(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1.0e-30:
        nx = ax
        ny = ay
    else:
        t = ((px - ax) * dx + (py - ay) * dy) / length2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nx = ax + t * dx
        ny = ay + t * dy
    return math.sqrt((px - nx) * (px - nx) + (py - ny) * (py - ny))


@njit(cache=True)
def _nearest_point_on_segment_numba(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> tuple[float, float, float]:
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1.0e-30:
        nx = ax
        ny = ay
    else:
        t = ((px - ax) * dx + (py - ay) * dy) / length2
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        nx = ax + t * dx
        ny = ay + t * dy
    return nx, ny, math.sqrt((px - nx) * (px - nx) + (py - ny) * (py - ny))


@njit(cache=True)
def _points_on_boundary_mask_numba(points: np.ndarray, segments: np.ndarray, tol: float) -> np.ndarray:
    mask = np.zeros(points.shape[0], dtype=np.bool_)
    for i in range(points.shape[0]):
        px = points[i, 0]
        py = points[i, 1]
        for j in range(segments.shape[0]):
            if _point_segment_distance_numba(px, py, segments[j, 0], segments[j, 1], segments[j, 2], segments[j, 3]) <= tol:
                mask[i] = True
                break
    return mask


def _boundary_segments_array(boundary_segments: list[tuple[Point2D, Point2D]]) -> np.ndarray:
    if not boundary_segments:
        return np.empty((0, 4), dtype=np.float64)
    return np.ascontiguousarray(
        [[float(a[0]), float(a[1]), float(b[0]), float(b[1])] for a, b in boundary_segments],
        dtype=np.float64,
    )


def _tunnels_array(tunnels: list[tuple[float, float, float]]) -> np.ndarray:
    if not tunnels:
        return np.empty((0, 3), dtype=np.float64)
    return np.ascontiguousarray([[float(cx), float(cy), float(radius)] for cx, cy, radius in tunnels], dtype=np.float64)


@njit(cache=True)
def _project_points_to_boundaries_numba(
    points: np.ndarray,
    segments: np.ndarray,
    tunnels: np.ndarray,
    bbox: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    projected = points.copy()
    changed = np.zeros(points.shape[0], dtype=np.bool_)
    x0 = bbox[0]
    x1 = bbox[1]
    y0 = bbox[2]
    y1 = bbox[3]
    for i in range(points.shape[0]):
        px = points[i, 0]
        py = points[i, 1]
        best_distance = tolerance
        best_x = 0.0
        best_y = 0.0
        has_best = False
        for j in range(segments.shape[0]):
            nx, ny, distance = _nearest_point_on_segment_numba(px, py, segments[j, 0], segments[j, 1], segments[j, 2], segments[j, 3])
            if distance <= best_distance:
                best_distance = distance
                best_x = nx
                best_y = ny
                has_best = True
        for j in range(tunnels.shape[0]):
            cx = tunnels[j, 0]
            cy = tunnels[j, 1]
            radius = tunnels[j, 2]
            if radius <= 0.0:
                continue
            dx = px - cx
            dy = py - cy
            distance_to_center = math.sqrt(dx * dx + dy * dy)
            if distance_to_center <= 1.0e-30:
                continue
            gap = abs(distance_to_center - radius)
            if gap <= best_distance:
                scale = radius / distance_to_center
                best_distance = gap
                best_x = cx + dx * scale
                best_y = cy + dy * scale
                has_best = True
        if not has_best:
            continue
        if not (x0 - tolerance <= best_x <= x1 + tolerance and y0 - tolerance <= best_y <= y1 + tolerance):
            continue
        move_x = best_x - px
        move_y = best_y - py
        if math.sqrt(move_x * move_x + move_y * move_y) <= 1.0e-12:
            continue
        projected[i, 0] = best_x
        projected[i, 1] = best_y
        changed[i] = True
    return projected, changed


@njit(cache=True)
def _cell_lower_bound_numba(sorted_x: np.ndarray, sorted_y: np.ndarray, cell_x: int, cell_y: int) -> int:
    lo = 0
    hi = sorted_x.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_x[mid] < cell_x or (sorted_x[mid] == cell_x and sorted_y[mid] < cell_y):
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True)
def _near_point_pairs_from_sorted_cells_numba(
    coords: np.ndarray,
    order: np.ndarray,
    sorted_cell_x: np.ndarray,
    sorted_cell_y: np.ndarray,
    tolerance: float,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    pair_i = np.zeros(max_pairs, dtype=np.int64)
    pair_j = np.zeros(max_pairs, dtype=np.int64)
    distances = np.zeros(max_pairs, dtype=np.float64)
    total = 0
    tol2 = tolerance * tolerance
    n = order.shape[0]
    for sorted_pos in range(n):
        i = order[sorted_pos]
        base_x = sorted_cell_x[sorted_pos]
        base_y = sorted_cell_y[sorted_pos]
        xi = coords[i, 0]
        yi = coords[i, 1]
        for dx in range(-1, 2):
            target_x = base_x + dx
            for dy in range(-1, 2):
                target_y = base_y + dy
                pos = _cell_lower_bound_numba(sorted_cell_x, sorted_cell_y, target_x, target_y)
                while pos < n and sorted_cell_x[pos] == target_x and sorted_cell_y[pos] == target_y:
                    j = order[pos]
                    if j > i:
                        ddx = coords[j, 0] - xi
                        ddy = coords[j, 1] - yi
                        dist2 = ddx * ddx + ddy * ddy
                        if dist2 <= tol2:
                            if total < max_pairs:
                                pair_i[total] = i
                                pair_j[total] = j
                                distances[total] = math.sqrt(dist2)
                            total += 1
                    pos += 1
    return pair_i, pair_j, distances, total


def node_ids_on_boundary(
    nodes: Mapping[str, list[float]],
    boundary_segments: list[tuple[Point2D, Point2D]],
    tol: float = 1.0e-8,
) -> list[str]:
    if not nodes or not boundary_segments:
        return []
    ids = [str(nid) for nid in nodes]
    points = np.ascontiguousarray([[float(point[0]), float(point[1])] for point in nodes.values()], dtype=np.float64)
    mask = _points_on_boundary_mask_numba(points, _boundary_segments_array(boundary_segments), float(tol))
    return [nid for nid, keep in zip(ids, mask) if bool(keep)]


def point_on_boundary(point: Point2D, boundary_segments: list[tuple[Point2D, Point2D]], tol: float = 1.0e-8) -> bool:
    return any(point_to_segment_distance(point, a, b) <= tol for a, b in boundary_segments)


def nearest_point_on_segment(point: Point2D, a: Point2D, b: Point2D) -> tuple[Point2D, float]:
    ax, ay = a
    bx, by = b
    px, py = point
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    if length2 <= 1.0e-30:
        nearest = (ax, ay)
    else:
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length2))
        nearest = (ax + t * dx, ay + t * dy)
    return nearest, math.hypot(px - nearest[0], py - nearest[1])


def point_to_segment_distance(point: Point2D, a: Point2D, b: Point2D) -> float:
    return float(_point_segment_distance_numba(float(point[0]), float(point[1]), float(a[0]), float(a[1]), float(b[0]), float(b[1])))


def segment_intersection(a: Point2D, b: Point2D, c: Point2D, d: Point2D) -> tuple[float, float, float, float] | None:
    ax, ay = a
    bx, by = b
    cx, cy = c
    dx, dy = d
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
    if -1.0e-10 <= t <= 1.0 + 1.0e-10 and -1.0e-10 <= u <= 1.0 + 1.0e-10:
        return t, u, ax + t * rx, ay + t * ry
    return None


def point_in_polygon(x: float, y: float, polygon: list[Point2D]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        intersects = (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / ((yj - yi) or 1.0e-30) + xi
        if intersects:
            inside = not inside
        j = i
    return inside


__all__ = [
    "Point2D",
    "axis_coordinate_aspect_ratio",
    "build_boundary_layer_coordinates",
    "circle_inside_polygon",
    "circular_tunnel_node_sets",
    "classify_element_regions",
    "corner_angle",
    "count_concave_vertices",
    "count_new_axis_values",
    "curve_segment_count",
    "bilinear_quad_point",
    "edge_midpoints",
    "first_region_index",
    "first_region_indices",
    "generate_quad_dominant_mesh",
    "generate_mapped_quadrilateral_paving_mesh",
    "generate_parametric_curved_paving_mesh",
    "generate_quadrilateral_circular_tunnel_paving_mesh",
    "generate_quadrilateral_polygon_hole_paving_mesh",
    "generate_quadrilateral_ring_paving_mesh",
    "generate_rectilinear_polygon_paving_mesh",
    "generate_simple_polygon_quad_paving_mesh",
    "geometry_boundary_segments",
    "inside_domain",
    "is_convex_polygon",
    "improve_mesh_quality",
    "laplacian_smooth_mesh",
    "local_remesh_bad_elements",
    "mesh_quality_summary",
    "mesh_quality_diagnostic",
    "merge_axis_coordinates",
    "minimum_circular_tunnel_clearance",
    "nearest_point_on_segment",
    "near_point_pairs",
    "node_ids_on_boundary",
    "normalize_mesh_refinements",
    "normalize_mesh_split_lines",
    "normalize_polygon_holes",
    "normalize_quadrilateral_region",
    "normalize_rectilinear_region",
    "normalize_region_points",
    "normalize_simple_polygon_region",
    "point_in_polygon",
    "point_in_triangle",
    "point_on_boundary",
    "point_to_segment_distance",
    "polygon_area",
    "polygon_centroid",
    "polygon_crosses_boundary",
    "polygon_has_crossing_edges",
    "polygons_have_positive_area_overlap",
    "polygon_hole_shape_summary",
    "polygons_any_overlap",
    "polygons_overlap",
    "quadrilateral_regions_share_edges",
    "project_nodes_to_boundaries",
    "optimize_mesh_node_positions",
    "prune_unused_nodes",
    "quad_quality",
    "improve_quad_topology",
    "quad_samples_respect_domain",
    "ray_polygon_intersection",
    "recombine_triangles_to_quads",
    "remove_collinear_polygon_vertices",
    "segment_intersection",
    "segment_stays_in_polygon",
    "star_shaped_polygon_center",
    "subdivide_remaining_triangles_to_quads",
    "triangulate_simple_polygon",
    "unwrap_angles",
    "with_requested_element_order",
]
