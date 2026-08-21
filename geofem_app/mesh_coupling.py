"""Mesh coupling helpers for VGFlow2D QUAD4 and GeoFEAS QUAD8 workflows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numba import prange

from .fem2d_elements import integration_points, shape_functions
from .fem2d_types import Element2D, FEM2DError, Mesh2D, njit
from .public_artifacts import html_table_document, write_dict_rows_csv, write_html_artifact, write_json_artifact


QUAD4_EDGE_PAIRS = ((0, 1), (1, 2), (2, 3), (3, 0))
PROJECTION_FIELDS = [
    "location_id",
    "location_type",
    "target_node",
    "target_element",
    "gp_index",
    "x",
    "y",
    "value",
    "status",
    "method",
    "source_element",
    "source_nodes",
    "weights",
    "natural_xi",
    "natural_eta",
    "residual",
    "nearest_distance",
]
EDGE_DIAGNOSTIC_FIELDS = [
    "element_id",
    "edge_index",
    "boundary",
    "start_node",
    "mid_node",
    "end_node",
    "chord_length",
    "quadratic_length",
    "length_difference",
    "sagitta",
    "relative_sagitta",
    "status",
]
PRESSURE_LOAD_FIELDS = [
    "element_id",
    "edge_index",
    "start_node",
    "mid_node",
    "end_node",
    "pressure_start",
    "pressure_mid",
    "pressure_end",
    "edge_length",
    "normal_x",
    "normal_y",
    "start_force_x",
    "start_force_y",
    "mid_force_x",
    "mid_force_y",
    "end_force_x",
    "end_force_y",
    "resultant_x",
    "resultant_y",
    "resultant_normal",
    "expected_resultant_normal",
    "conservation_error",
    "status",
]
QUAD8_EDGE_TRIPLETS = ((0, 4, 1), (1, 5, 2), (2, 6, 3), (3, 7, 0))
QUADRATIC_EDGE_CONSISTENT = np.array(
    [[2.0 / 15.0, 1.0 / 15.0, -1.0 / 30.0], [1.0 / 15.0, 8.0 / 15.0, 1.0 / 15.0], [-1.0 / 30.0, 1.0 / 15.0, 2.0 / 15.0]],
    dtype=float,
)


@dataclass(frozen=True)
class ScalarProjectionPlan:
    """Reusable source-to-target scalar projection geometry and weights."""

    method: str
    fallback: str
    source_node_ids: tuple[str, ...]
    source_summary: Mapping[str, Any]
    target_summary: Mapping[str, Any]
    row_templates: tuple[dict[str, Any], ...]
    source_indices: np.ndarray
    weights: np.ndarray


def upgrade_quad4_mesh_to_quad8(mesh: Mesh2D, *, midpoint_prefix: str = "m") -> tuple[Mesh2D, dict[str, Any]]:
    """Create a same-topology QUAD8 mesh from a QUAD4 mesh."""

    if any(element.type != "QUAD4" for element in mesh.elements):
        raise FEM2DError("QUAD4 to QUAD8 coupling upgrade requires all elements to be QUAD4")
    node_ids = list(mesh.node_ids)
    coords = [tuple(map(float, xy)) for xy in mesh.coords]
    node_index = {nid: i for i, nid in enumerate(node_ids)}
    midpoint_nodes: dict[tuple[str, str], str] = {}
    node_rows: list[dict[str, Any]] = [{"source_node": nid, "target_node": nid, "kind": "corner", "weight": "1.0"} for nid in node_ids]

    def midpoint(a: str, b: str) -> str:
        key = tuple(sorted((a, b)))
        if key in midpoint_nodes:
            return midpoint_nodes[key]
        base = f"{midpoint_prefix}_{key[0]}_{key[1]}"
        nid = base
        suffix = 1
        while nid in node_index:
            suffix += 1
            nid = f"{base}_{suffix}"
        xy = 0.5 * (mesh.coords[node_index[a]] + mesh.coords[node_index[b]])
        node_index[nid] = len(node_ids)
        node_ids.append(nid)
        coords.append((float(xy[0]), float(xy[1])))
        midpoint_nodes[key] = nid
        node_rows.append({"source_node": f"{a};{b}", "target_node": nid, "kind": "midside", "weight": "0.5;0.5"})
        return nid

    elements: list[Element2D] = []
    element_rows: list[dict[str, Any]] = []
    for element in mesh.elements:
        corners = element.nodes[:4]
        mids = tuple(midpoint(corners[a], corners[b]) for a, b in QUAD4_EDGE_PAIRS)
        elements.append(Element2D(element.id, "QUAD8", (*corners, *mids), element.material, element.integration, element.active))
        element_rows.append({"source_element": element.id, "target_element": element.id, "source_type": "QUAD4", "target_type": "QUAD8", "material": element.material, "integration": element.integration})

    upgraded = Mesh2D(
        node_ids=node_ids,
        coords=np.asarray(coords, dtype=float),
        elements=elements,
        node_sets=_upgrade_node_sets(mesh, midpoint_nodes),
        element_sets={key: list(value) for key, value in mesh.element_sets.items()},
    )
    manifest = _manifest(mesh, upgraded, "quad4_to_quad8", node_rows, element_rows)
    return upgraded, manifest


def downgrade_quad8_mesh_to_quad4(mesh: Mesh2D) -> tuple[Mesh2D, dict[str, Any]]:
    """Extract a corner-node QUAD4 hydraulic mesh from a QUAD8 mechanical mesh."""

    if any(element.type != "QUAD8" for element in mesh.elements):
        raise FEM2DError("QUAD8 to QUAD4 coupling downgrade requires all elements to be QUAD8")
    keep_order: list[str] = []
    for element in mesh.elements:
        for nid in element.nodes[:4]:
            if nid not in keep_order:
                keep_order.append(nid)
    old_index = mesh.node_index
    new_index = {nid: i for i, nid in enumerate(keep_order)}
    downgraded = Mesh2D(
        node_ids=keep_order,
        coords=np.asarray([mesh.coords[old_index[nid]] for nid in keep_order], dtype=float),
        elements=[
            Element2D(element.id, "QUAD4", tuple(element.nodes[:4]), element.material, element.integration, element.active)
            for element in mesh.elements
        ],
        node_sets={key: [nid for nid in value if nid in new_index] for key, value in mesh.node_sets.items()},
        element_sets={key: list(value) for key, value in mesh.element_sets.items()},
    )
    node_rows = [{"source_node": nid, "target_node": nid, "kind": "corner", "weight": "1.0"} for nid in keep_order]
    element_rows = [
        {"source_element": element.id, "target_element": element.id, "source_type": "QUAD8", "target_type": "QUAD4", "material": element.material, "integration": element.integration}
        for element in mesh.elements
    ]
    manifest = _manifest(mesh, downgraded, "quad8_to_quad4", node_rows, element_rows)
    downgrade_diagnostics = diagnose_quad8_to_quad4_downgrade(mesh)
    manifest["features"] = list(dict.fromkeys([*manifest["features"], "quad8_to_quad4_curved_edge_diagnostics"]))
    manifest["diagnostics"]["curved_edge_count"] = downgrade_diagnostics["diagnostics"]["curved_edge_count"]
    manifest["diagnostics"]["max_relative_sagitta"] = downgrade_diagnostics["diagnostics"]["max_relative_sagitta"]
    manifest["diagnostics"]["max_boundary_length_difference"] = downgrade_diagnostics["diagnostics"]["max_length_difference"]
    manifest["edge_diagnostics"] = downgrade_diagnostics["edge_diagnostics"]
    return downgraded, manifest


def diagnose_quad8_to_quad4_downgrade(mesh: Mesh2D, *, tolerance: float = 1.0e-6) -> dict[str, Any]:
    """Diagnose geometric loss when QUAD8 midside nodes are dropped."""

    if any(element.type != "QUAD8" for element in mesh.elements):
        raise FEM2DError("QUAD8 downgrade diagnostics require all elements to be QUAD8")
    boundary_pairs = _boundary_corner_pairs(mesh)
    rows: list[dict[str, Any]] = []
    element_area_rows: list[dict[str, Any]] = []
    for element in mesh.elements:
        curved_points = []
        for edge_index, (left_idx, mid_idx, right_idx) in enumerate(QUAD8_EDGE_TRIPLETS):
            start = element.nodes[left_idx]
            mid = element.nodes[mid_idx]
            end = element.nodes[right_idx]
            p0 = mesh.coords[mesh.node_index[start]]
            pm = mesh.coords[mesh.node_index[mid]]
            p1 = mesh.coords[mesh.node_index[end]]
            chord = float(np.linalg.norm(p1 - p0))
            quadratic_length = float(np.linalg.norm(pm - p0) + np.linalg.norm(p1 - pm))
            sagitta = _point_segment_distance(pm, p0, p1)
            relative = 0.0 if chord <= 1.0e-30 else sagitta / chord
            status = "warning" if relative > tolerance else "pass"
            rows.append(
                {
                    "element_id": element.id,
                    "edge_index": edge_index,
                    "boundary": tuple(sorted((start, end))) in boundary_pairs,
                    "start_node": start,
                    "mid_node": mid,
                    "end_node": end,
                    "chord_length": chord,
                    "quadratic_length": quadratic_length,
                    "length_difference": quadratic_length - chord,
                    "sagitta": sagitta,
                    "relative_sagitta": relative,
                    "status": status,
                }
            )
        for node_idx in (0, 4, 1, 5, 2, 6, 3, 7):
            curved_points.append(mesh.coords[mesh.node_index[element.nodes[node_idx]]])
        curved_area = abs(_polygon_area(np.asarray(curved_points, dtype=float)))
        corner_area = abs(_polygon_area(np.asarray([mesh.coords[mesh.node_index[nid]] for nid in element.nodes[:4]], dtype=float)))
        element_area_rows.append(
            {
                "element_id": element.id,
                "corner_area": corner_area,
                "curved_boundary_area": curved_area,
                "area_difference": curved_area - corner_area,
            }
        )
    max_sagitta = max((float(row["sagitta"]) for row in rows), default=0.0)
    max_relative = max((float(row["relative_sagitta"]) for row in rows), default=0.0)
    max_length_diff = max((abs(float(row["length_difference"])) for row in rows), default=0.0)
    return {
        "schema": "geofem.mesh_coupling.quad8_downgrade_diagnostics.v1",
        "features": ["curved_quad8_edge_diagnostics", "boundary_edge_grouping", "corner_area_vs_curved_boundary_area"],
        "source": _mesh_summary(mesh),
        "tolerance": tolerance,
        "diagnostics": {
            "edge_count": len(rows),
            "boundary_edge_count": sum(1 for row in rows if row["boundary"]),
            "curved_edge_count": sum(1 for row in rows if row["status"] != "pass"),
            "max_sagitta": max_sagitta,
            "max_relative_sagitta": max_relative,
            "max_length_difference": max_length_diff,
            "max_area_difference": max((abs(float(row["area_difference"])) for row in element_area_rows), default=0.0),
        },
        "edge_diagnostics": rows,
        "element_area_diagnostics": element_area_rows,
    }


def build_quad8_pressure_load_report(
    mesh: Mesh2D,
    pressure_values: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    thickness: float = 1.0,
    tolerance: float = 1.0e-8,
) -> dict[str, Any]:
    """Build an open substitute report for QUAD8 boundary pressure loads."""

    if any(element.type != "QUAD8" for element in mesh.elements):
        raise FEM2DError("QUAD8 pressure load reports require all elements to be QUAD8")
    pressures = _value_map(mesh, pressure_values)
    missing = [nid for nid in mesh.node_ids if nid not in pressures]
    if missing:
        raise FEM2DError(f"missing pressure values for QUAD8 pressure load report: {missing[:5]}")
    boundary_pairs = _boundary_corner_pairs(mesh)
    rows: list[dict[str, Any]] = []
    total_x = 0.0
    total_y = 0.0
    total_normal = 0.0
    max_error = 0.0
    for element in mesh.elements:
        for edge_index, (left_idx, mid_idx, right_idx) in enumerate(QUAD8_EDGE_TRIPLETS):
            start = element.nodes[left_idx]
            mid = element.nodes[mid_idx]
            end = element.nodes[right_idx]
            if tuple(sorted((start, end))) not in boundary_pairs:
                continue
            p0 = mesh.coords[mesh.node_index[start]]
            pm = mesh.coords[mesh.node_index[mid]]
            p1 = mesh.coords[mesh.node_index[end]]
            length = float(np.linalg.norm(pm - p0) + np.linalg.norm(p1 - pm))
            if length <= 1.0e-30:
                raise FEM2DError(f"element {element.id} edge {edge_index}: boundary length must be positive")
            tangent = p1 - p0
            tangent_length = float(np.linalg.norm(tangent))
            if tangent_length <= 1.0e-30:
                raise FEM2DError(f"element {element.id} edge {edge_index}: chord length must be positive")
            normal = np.array([tangent[1], -tangent[0]], dtype=float) / tangent_length
            pressure_vec = np.array([pressures[start], pressures[mid], pressures[end]], dtype=float)
            nodal_normal = (QUADRATIC_EDGE_CONSISTENT * length * float(thickness)) @ pressure_vec
            forces = np.outer(nodal_normal, normal)
            resultant = np.sum(forces, axis=0)
            resultant_normal = float(np.sum(nodal_normal))
            expected = float(length * float(thickness) * (pressure_vec[0] + 4.0 * pressure_vec[1] + pressure_vec[2]) / 6.0)
            error = resultant_normal - expected
            max_error = max(max_error, abs(error))
            status = "pass" if abs(error) <= tolerance * max(1.0, abs(expected)) else "warning"
            row = {
                "element_id": element.id,
                "edge_index": edge_index,
                "start_node": start,
                "mid_node": mid,
                "end_node": end,
                "pressure_start": float(pressure_vec[0]),
                "pressure_mid": float(pressure_vec[1]),
                "pressure_end": float(pressure_vec[2]),
                "edge_length": length,
                "normal_x": float(normal[0]),
                "normal_y": float(normal[1]),
                "start_force_x": float(forces[0, 0]),
                "start_force_y": float(forces[0, 1]),
                "mid_force_x": float(forces[1, 0]),
                "mid_force_y": float(forces[1, 1]),
                "end_force_x": float(forces[2, 0]),
                "end_force_y": float(forces[2, 1]),
                "resultant_x": float(resultant[0]),
                "resultant_y": float(resultant[1]),
                "resultant_normal": resultant_normal,
                "expected_resultant_normal": expected,
                "conservation_error": error,
                "status": status,
            }
            rows.append(row)
            total_x += float(resultant[0])
            total_y += float(resultant[1])
            total_normal += resultant_normal
    return {
        "schema": "geofem.mesh_coupling.quad8_pressure_load_report.v1",
        "features": ["quad8_quadratic_pressure_load_table", "consistent_edge_pressure_integration", "resultant_conservation_check"],
        "source": _mesh_summary(mesh),
        "sign_convention": "positive pressure acts in the element outward normal direction for each boundary edge",
        "thickness": float(thickness),
        "diagnostics": {
            "edge_count": len(rows),
            "total_resultant_x": total_x,
            "total_resultant_y": total_y,
            "total_resultant_normal": total_normal,
            "max_conservation_error": max_error,
            "warning_count": sum(1 for row in rows if row["status"] != "pass"),
        },
        "pressure_loads": rows,
    }


def interpolate_quad4_nodal_values_to_quad8(
    quad4_mesh: Mesh2D,
    quad8_mesh: Mesh2D,
    values: Mapping[str, float] | Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """Interpolate corner-node QUAD4 scalar values onto same-topology QUAD8 nodes."""

    source = _value_map(quad4_mesh, values)
    target: dict[str, float] = {}
    q4_by_id = {element.id: element for element in quad4_mesh.elements}
    for element in quad8_mesh.elements:
        if element.type != "QUAD8":
            raise FEM2DError("target mesh must be QUAD8 for QUAD4 to QUAD8 value interpolation")
        source_element = q4_by_id.get(element.id)
        if source_element is None:
            raise FEM2DError(f"missing QUAD4 source element for target element {element.id}")
        for sid, tid in zip(source_element.nodes[:4], element.nodes[:4]):
            if sid not in source:
                raise FEM2DError(f"missing source nodal value for {sid}")
            target[tid] = float(source[sid])
        for mid_node, (a, b) in zip(element.nodes[4:8], QUAD4_EDGE_PAIRS):
            na = source_element.nodes[a]
            nb = source_element.nodes[b]
            if na not in source or nb not in source:
                raise FEM2DError(f"missing source nodal value for edge {na}-{nb}")
            target[mid_node] = 0.5 * (float(source[na]) + float(source[nb]))
    return target


def project_scalar_values_between_meshes(
    source_mesh: Mesh2D,
    target_mesh: Mesh2D,
    values: Mapping[str, float] | Sequence[float] | np.ndarray,
    *,
    method: str = "shape_function",
    locations: str = "nodes",
    fallback: str = "nearest",
    tolerance: float = 1.0e-8,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Project scalar nodal results between non-identical 2D meshes.

    The default path searches the source element containing each target
    location and evaluates the source shape functions there. Points outside the
    source mesh are explicitly reported and can fall back to nearest-node
    transfer.
    """

    plan = build_scalar_projection_plan(source_mesh, target_mesh, method=method, locations=locations, fallback=fallback, tolerance=tolerance)
    return apply_scalar_projection_plan(plan, values)


def build_scalar_projection_plan(
    source_mesh: Mesh2D,
    target_mesh: Mesh2D,
    *,
    method: str = "shape_function",
    locations: str = "nodes",
    fallback: str = "nearest",
    tolerance: float = 1.0e-8,
) -> ScalarProjectionPlan:
    """Resolve projection geometry once so multiple scalar fields can reuse it."""

    method_key = _projection_method_key(method)
    fallback_key = _projection_fallback_key(fallback)

    row_templates: list[dict[str, Any]] = []
    index_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    source_element_bboxes = _source_element_bboxes(source_mesh) if method_key == "shape_function" else None
    for location in _projection_locations(target_mesh, locations):
        xy = np.array([location["x"], location["y"]], dtype=float)
        if method_key == "nearest":
            row, node_indices, weights = _nearest_projection_plan_row(source_mesh, xy)
        else:
            resolved = _shape_projection_plan_row(source_mesh, xy, tolerance, source_element_bboxes)
            if resolved is None:
                if fallback_key == "error":
                    raise FEM2DError(f"target location {location['location_id']} is outside source mesh")
                if fallback_key == "nearest":
                    row, node_indices, weights = _nearest_projection_plan_row(source_mesh, xy)
                    row["status"] = "nearest_fallback"
                else:
                    row, node_indices, weights = _empty_projection_plan_row("out_of_domain")
            else:
                row, node_indices, weights = resolved
        row_templates.append({**location, **row})
        index_rows.append(node_indices)
        weight_rows.append(weights)

    max_terms = max((len(indices) for indices in index_rows), default=0)
    source_indices = np.full((len(index_rows), max_terms), -1, dtype=np.int64)
    projection_weights = np.zeros((len(weight_rows), max_terms), dtype=float)
    for i, (indices, weights) in enumerate(zip(index_rows, weight_rows)):
        count = len(indices)
        if count:
            source_indices[i, :count] = indices
            projection_weights[i, :count] = weights
    return ScalarProjectionPlan(
        method=method_key,
        fallback=fallback_key,
        source_node_ids=tuple(source_mesh.node_ids),
        source_summary=_mesh_summary(source_mesh),
        target_summary=_mesh_summary(target_mesh),
        row_templates=tuple(row_templates),
        source_indices=source_indices,
        weights=projection_weights,
    )


def apply_scalar_projection_plan(plan: ScalarProjectionPlan, values: Mapping[str, float] | Sequence[float] | np.ndarray) -> tuple[dict[str, float], dict[str, Any]]:
    """Apply nodal values to a reusable projection plan."""

    source_values = _value_array_from_node_ids(plan.source_node_ids, values)
    projected_values = _apply_projection_weights(source_values, plan.source_indices, plan.weights)
    projected: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for template, value in zip(plan.row_templates, projected_values):
        row = dict(template)
        row["value"] = float(value)
        projected[str(row["location_id"])] = float(value)
        rows.append(row)
    manifest = _projection_manifest_from_summaries(plan.source_summary, plan.target_summary, rows, plan.method, plan.fallback)
    return projected, manifest


def write_mesh_coupling_manifest(output_dir: str | Path, manifest: Mapping[str, Any], *, prefix: str = "vgflow_geofeas_mesh_coupling") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": str(out / f"{prefix}.json"),
        "nodes_csv": str(out / f"{prefix}_nodes.csv"),
        "elements_csv": str(out / f"{prefix}_elements.csv"),
        "html": str(out / f"{prefix}.html"),
    }
    write_json_artifact(paths["json"], manifest)
    write_dict_rows_csv(paths["nodes_csv"], manifest.get("node_map", []), ["source_node", "target_node", "kind", "weight"])
    write_dict_rows_csv(paths["elements_csv"], manifest.get("element_map", []), ["source_element", "target_element", "source_type", "target_type", "material", "integration"])
    if manifest.get("edge_diagnostics"):
        paths["edge_diagnostics_csv"] = str(out / f"{prefix}_edge_diagnostics.csv")
        write_dict_rows_csv(paths["edge_diagnostics_csv"], manifest.get("edge_diagnostics", []), EDGE_DIAGNOSTIC_FIELDS)
    write_html_artifact(paths["html"], _manifest_html(manifest))
    return paths


def write_mesh_projection_manifest(output_dir: str | Path, manifest: Mapping[str, Any], *, prefix: str = "vgflow_geofeas_mesh_projection") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": str(out / f"{prefix}.json"),
        "projection_csv": str(out / f"{prefix}.csv"),
        "html": str(out / f"{prefix}.html"),
    }
    write_json_artifact(paths["json"], manifest)
    write_dict_rows_csv(paths["projection_csv"], manifest.get("projection_map", []), PROJECTION_FIELDS)
    write_html_artifact(paths["html"], _projection_manifest_html(manifest))
    return paths


def write_quad8_pressure_load_report(output_dir: str | Path, report: Mapping[str, Any], *, prefix: str = "vgflow_geofeas_quad8_pressure_loads") -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": str(out / f"{prefix}.json"),
        "csv": str(out / f"{prefix}.csv"),
        "html": str(out / f"{prefix}.html"),
    }
    write_json_artifact(paths["json"], report)
    write_dict_rows_csv(paths["csv"], report.get("pressure_loads", []), PRESSURE_LOAD_FIELDS)
    write_html_artifact(paths["html"], _pressure_load_report_html(report))
    return paths


def _upgrade_node_sets(mesh: Mesh2D, midpoint_nodes: Mapping[tuple[str, str], str]) -> dict[str, list[str]]:
    upgraded: dict[str, list[str]] = {}
    for name, ids in mesh.node_sets.items():
        out: list[str] = []
        for left, right in zip(ids, ids[1:]):
            if left not in out:
                out.append(left)
            mid = midpoint_nodes.get(tuple(sorted((left, right))))
            if mid is not None:
                out.append(mid)
        if ids:
            out.append(ids[-1])
        upgraded[name] = out
    upgraded["all"] = list(dict.fromkeys([nid for ids in upgraded.values() for nid in ids] + list(mesh.node_ids) + [mid for mid in midpoint_nodes.values()]))
    return upgraded


def _boundary_corner_pairs(mesh: Mesh2D) -> set[tuple[str, str]]:
    counts: dict[tuple[str, str], int] = {}
    for element in mesh.elements:
        if element.type == "QUAD8":
            pairs = [(element.nodes[left], element.nodes[right]) for left, _mid, right in QUAD8_EDGE_TRIPLETS]
        else:
            corners = element.nodes[:4] if element.type.startswith("QUAD") else element.nodes[:3]
            pairs = list(zip(corners, [*corners[1:], corners[0]]))
        for left, right in pairs:
            key = tuple(sorted((left, right)))
            counts[key] = counts.get(key, 0) + 1
    return {key for key, count in counts.items() if count == 1}


def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    edge = end - start
    denom = float(np.dot(edge, edge))
    if denom <= 1.0e-30:
        return float(np.linalg.norm(point - start))
    t = max(0.0, min(1.0, float(np.dot(point - start, edge) / denom)))
    closest = start + t * edge
    return float(np.linalg.norm(point - closest))


def _projection_locations(mesh: Mesh2D, locations: str) -> list[dict[str, Any]]:
    key = str(locations).strip().lower().replace("-", "_")
    if key not in {"nodes", "integration_points", "both"}:
        raise FEM2DError("projection locations must be 'nodes', 'integration_points', or 'both'")
    rows: list[dict[str, Any]] = []
    if key in {"nodes", "both"}:
        for nid in mesh.node_ids:
            xy = mesh.coords[mesh.node_index[nid]]
            rows.append(
                {
                    "location_id": nid,
                    "location_type": "node",
                    "target_node": nid,
                    "target_element": "",
                    "gp_index": "",
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                }
            )
    if key in {"integration_points", "both"}:
        for element in mesh.elements:
            if not element.active:
                continue
            coords = np.asarray([mesh.coords[mesh.node_index[nid]] for nid in element.nodes], dtype=float)
            for gp_index, (xi, eta, _weight) in enumerate(integration_points(element.type, "FULL")):
                N, _dN = shape_functions(element.type, xi, eta)
                xy = N @ coords
                rows.append(
                    {
                        "location_id": f"{element.id}:gp{gp_index}",
                        "location_type": "integration_point",
                        "target_node": "",
                        "target_element": element.id,
                        "gp_index": gp_index,
                        "x": float(xy[0]),
                        "y": float(xy[1]),
                    }
                )
    return rows


def _shape_projection_plan_row(
    source_mesh: Mesh2D,
    xy: np.ndarray,
    tolerance: float,
    element_bboxes: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray] | None:
    candidates: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    element_indices = _candidate_source_elements(source_mesh, xy, tolerance, element_bboxes)
    for element_index in element_indices:
        element = source_mesh.elements[int(element_index)]
        if not element.active:
            continue
        coords = np.asarray([source_mesh.coords[source_mesh.node_index[nid]] for nid in element.nodes], dtype=float)
        natural = _natural_coordinates_for_point(element.type, coords, xy)
        if natural is None:
            continue
        xi, eta, residual = natural
        if not _natural_inside(element.type, xi, eta, tolerance):
            continue
        N, _dN = shape_functions(element.type, xi, eta)
        candidates.append(
            (
                {
                    "status": "inside",
                    "method": "shape_function",
                    "source_element": element.id,
                    "source_nodes": ";".join(element.nodes),
                    "weights": ";".join(f"{nid}:{float(weight):.12g}" for nid, weight in zip(element.nodes, N)),
                    "natural_xi": float(xi),
                    "natural_eta": float(eta),
                    "residual": float(residual),
                    "nearest_distance": "",
                },
                np.asarray([source_mesh.node_index[nid] for nid in element.nodes], dtype=np.int64),
                np.asarray(N, dtype=float),
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda item: abs(float(item[0]["residual"])))
    selected, node_indices, weights = candidates[0]
    selected = dict(selected)
    if len(candidates) > 1:
        selected["status"] = "ambiguous_inside"
        selected["candidate_count"] = len(candidates)
    return selected, node_indices, weights


def _source_element_bboxes(mesh: Mesh2D) -> np.ndarray:
    bboxes = np.empty((len(mesh.elements), 4), dtype=float)
    for i, element in enumerate(mesh.elements):
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        bboxes[i, 0] = float(np.min(coords[:, 0]))
        bboxes[i, 1] = float(np.min(coords[:, 1]))
        bboxes[i, 2] = float(np.max(coords[:, 0]))
        bboxes[i, 3] = float(np.max(coords[:, 1]))
    return bboxes


def _candidate_source_elements(mesh: Mesh2D, xy: np.ndarray, tolerance: float, element_bboxes: np.ndarray | None) -> np.ndarray:
    if element_bboxes is None:
        return np.arange(len(mesh.elements), dtype=np.int64)
    tol = max(float(tolerance), 1.0e-12)
    mask = (
        (element_bboxes[:, 0] - tol <= float(xy[0]))
        & (float(xy[0]) <= element_bboxes[:, 2] + tol)
        & (element_bboxes[:, 1] - tol <= float(xy[1]))
        & (float(xy[1]) <= element_bboxes[:, 3] + tol)
    )
    return np.nonzero(mask)[0].astype(np.int64)


def _projection_method_key(method: str) -> str:
    method_key = str(method).strip().lower().replace("-", "_")
    if method_key in {"shape", "shape_functions", "shape_function", "natural", "element_natural"}:
        return "shape_function"
    if method_key in {"nearest", "nearest_node"}:
        return "nearest"
    raise FEM2DError(f"unsupported mesh projection method '{method}'")


def _projection_fallback_key(fallback: str) -> str:
    fallback_key = str(fallback).strip().lower().replace("-", "_")
    if fallback_key not in {"nearest", "none", "error"}:
        raise FEM2DError(f"unsupported mesh projection fallback '{fallback}'")
    return fallback_key


def _nearest_projection_plan_row(source_mesh: Mesh2D, xy: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    delta = source_mesh.coords - xy
    distances = np.einsum("ij,ij->i", delta, delta)
    idx = int(np.argmin(distances))
    nid = source_mesh.node_ids[idx]
    return {
        "status": "nearest",
        "method": "nearest",
        "source_element": "",
        "source_nodes": nid,
        "weights": "1.0",
        "natural_xi": "",
        "natural_eta": "",
        "residual": "",
        "nearest_distance": float(math.sqrt(float(distances[idx]))),
    }, np.asarray([idx], dtype=np.int64), np.asarray([1.0], dtype=float)


def _empty_projection_plan_row(status: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    return {
        "status": status,
        "method": "none",
        "source_element": "",
        "source_nodes": "",
        "weights": "",
        "natural_xi": "",
        "natural_eta": "",
        "residual": "",
        "nearest_distance": "",
    }, np.empty(0, dtype=np.int64), np.empty(0, dtype=float)


def _natural_coordinates_for_point(element_type: str, coords: np.ndarray, xy: np.ndarray) -> tuple[float, float, float] | None:
    etype = element_type.upper()
    natural = np.array([1.0 / 3.0, 1.0 / 3.0], dtype=float) if etype.startswith("TRI") else np.zeros(2, dtype=float)
    residual = math.inf
    for _iteration in range(30):
        N, dN = shape_functions(etype, float(natural[0]), float(natural[1]))
        mapped = N @ coords
        vector = mapped - xy
        residual = float(np.linalg.norm(vector))
        jac = dN @ coords
        try:
            step = np.linalg.solve(jac.T, vector)
        except np.linalg.LinAlgError:
            return None
        natural -= step
        if float(np.linalg.norm(step)) <= 1.0e-12 and residual <= 1.0e-10:
            break
    N, _dN = shape_functions(etype, float(natural[0]), float(natural[1]))
    residual = float(np.linalg.norm(N @ coords - xy))
    return float(natural[0]), float(natural[1]), residual


def _natural_inside(element_type: str, xi: float, eta: float, tolerance: float) -> bool:
    etype = element_type.upper()
    if etype.startswith("TRI"):
        return xi >= -tolerance and eta >= -tolerance and xi + eta <= 1.0 + tolerance
    return -1.0 - tolerance <= xi <= 1.0 + tolerance and -1.0 - tolerance <= eta <= 1.0 + tolerance


def _projection_manifest(source: Mesh2D, target: Mesh2D, rows: list[dict[str, Any]], method: str, fallback: str) -> dict[str, Any]:
    return _projection_manifest_from_summaries(_mesh_summary(source), _mesh_summary(target), rows, method, fallback)


def _projection_manifest_from_summaries(source_summary: Mapping[str, Any], target_summary: Mapping[str, Any], rows: list[dict[str, Any]], method: str, fallback: str) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    max_residual = 0.0
    max_nearest_distance = 0.0
    for row in rows:
        status = str(row.get("status", ""))
        status_counts[status] = status_counts.get(status, 0) + 1
        residual = row.get("residual", "")
        if residual not in ("", None):
            max_residual = max(max_residual, abs(float(residual)))
        nearest = row.get("nearest_distance", "")
        if nearest not in ("", None):
            max_nearest_distance = max(max_nearest_distance, float(nearest))
    return {
        "schema": "geofem.mesh_coupling.projection.v1",
        "method": method,
        "fallback": fallback,
        "features": [
            "nonmatching_mesh_scalar_projection",
            "natural_coordinate_shape_interpolation",
            "bbox_prefiltered_source_element_search",
            "reusable_projection_weight_cache",
            "parallel_projection_weight_application",
            "nearest_fallback_diagnostics",
            "target_nodes_and_integration_points",
        ],
        "source": dict(source_summary),
        "target": dict(target_summary),
        "diagnostics": {
            "location_count": len(rows),
            "status_counts": status_counts,
            "fallback_count": status_counts.get("nearest_fallback", 0),
            "ambiguous_candidate_count": status_counts.get("ambiguous_inside", 0),
            "out_of_domain_count": status_counts.get("out_of_domain", 0),
            "max_residual": max_residual,
            "max_nearest_distance": max_nearest_distance,
        },
        "projection_map": rows,
    }


def _manifest(source: Mesh2D, target: Mesh2D, mode: str, node_rows: list[dict[str, Any]], element_rows: list[dict[str, Any]]) -> dict[str, Any]:
    source_area = _total_quad_area(source)
    target_area = _total_quad_area(target)
    return {
        "schema": "geofem.mesh_coupling.quad4_quad8.v1",
        "mode": mode,
        "features": ["quad4_quad8_same_topology_mapping", "corner_node_identity", "midside_linear_interpolation", "area_orientation_material_checks"],
        "source": _mesh_summary(source),
        "target": _mesh_summary(target),
        "diagnostics": {
            "source_area": source_area,
            "target_area": target_area,
            "area_difference": target_area - source_area,
            "area_match": math.isclose(source_area, target_area, rel_tol=1.0e-12, abs_tol=1.0e-12),
            "element_count_match": len(source.elements) == len(target.elements),
            "material_ids_match": sorted({e.material for e in source.elements}) == sorted({e.material for e in target.elements}),
        },
        "node_map": node_rows,
        "element_map": element_rows,
    }


def _mesh_summary(mesh: Mesh2D) -> dict[str, Any]:
    return {
        "node_count": len(mesh.node_ids),
        "element_count": len(mesh.elements),
        "element_types": sorted({element.type for element in mesh.elements}),
        "materials": sorted({element.material for element in mesh.elements}),
        "node_sets": sorted(mesh.node_sets),
        "element_sets": sorted(mesh.element_sets),
    }


def _total_quad_area(mesh: Mesh2D) -> float:
    total = 0.0
    for element in mesh.elements:
        corners = element.nodes[:4]
        points = np.asarray([mesh.coords[mesh.node_index[nid]] for nid in corners], dtype=float)
        total += abs(_polygon_area(points))
    return float(total)


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _value_map(mesh: Mesh2D, values: Mapping[str, float] | Sequence[float] | np.ndarray) -> dict[str, float]:
    if isinstance(values, Mapping):
        return {str(key): float(value) for key, value in values.items()}
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != len(mesh.node_ids):
        raise FEM2DError("nodal value array length must match source mesh node count")
    return {nid: float(arr[i]) for i, nid in enumerate(mesh.node_ids)}


def _value_array_from_node_ids(node_ids: Sequence[str], values: Mapping[str, float] | Sequence[float] | np.ndarray) -> np.ndarray:
    if isinstance(values, Mapping):
        value_map = {str(key): float(value) for key, value in values.items()}
        missing = [nid for nid in node_ids if nid not in value_map]
        if missing:
            raise FEM2DError(f"missing source nodal values for projection: {missing[:5]}")
        return np.asarray([value_map[nid] for nid in node_ids], dtype=float)
    arr = np.asarray(values, dtype=float)
    if arr.shape[0] != len(node_ids):
        raise FEM2DError("nodal value array length must match source mesh node count")
    return arr.astype(float, copy=False)


def _apply_projection_weights(source_values: np.ndarray, source_indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return _apply_projection_weights_numba(source_values, source_indices, weights)


@njit(cache=True, parallel=True)
def _apply_projection_weights_numba(source_values: np.ndarray, source_indices: np.ndarray, weights: np.ndarray) -> np.ndarray:
    projected = np.empty(source_indices.shape[0], dtype=np.float64)
    term_count = source_indices.shape[1]
    for i in prange(source_indices.shape[0]):
        total = 0.0
        valid = False
        for j in range(term_count):
            idx = source_indices[i, j]
            if idx >= 0:
                total += weights[i, j] * source_values[idx]
                valid = True
        projected[i] = total if valid else math.nan
    return projected


def _manifest_html(manifest: Mapping[str, Any]) -> str:
    diagnostics = manifest.get("diagnostics", {})
    rows = [[key, value] for key, value in diagnostics.items()]
    return html_table_document(
        title="VGFlow2D / GeoFEAS Mesh Coupling Manifest",
        lead="Records same-topology QUAD4 hydraulic and QUAD8 mechanical mesh maps, node maps, element maps, and area diagnostics.",
        headers=["diagnostic", "value"],
        rows=rows,
    )


def _projection_manifest_html(manifest: Mapping[str, Any]) -> str:
    diagnostics = manifest.get("diagnostics", {})
    rows = [[key, value] for key, value in diagnostics.items()]
    return html_table_document(
        title="VGFlow2D / GeoFEAS Mesh Projection Manifest",
        lead="Records scalar field projection between non-identical meshes, including fallback and ambiguity diagnostics.",
        headers=["diagnostic", "value"],
        rows=rows,
    )


def _pressure_load_report_html(report: Mapping[str, Any]) -> str:
    diagnostics = report.get("diagnostics", {})
    rows = [[key, value] for key, value in diagnostics.items()]
    return html_table_document(
        title="VGFlow2D / GeoFEAS QUAD8 Pressure Load Report",
        lead="Records equivalent nodal loads for QUAD8 quadratic boundary pressure transfer and resultant conservation checks.",
        headers=["diagnostic", "value"],
        rows=rows,
    )


__all__ = [
    "ScalarProjectionPlan",
    "apply_scalar_projection_plan",
    "build_scalar_projection_plan",
    "build_quad8_pressure_load_report",
    "diagnose_quad8_to_quad4_downgrade",
    "downgrade_quad8_mesh_to_quad4",
    "interpolate_quad4_nodal_values_to_quad8",
    "project_scalar_values_between_meshes",
    "upgrade_quad4_mesh_to_quad8",
    "write_mesh_coupling_manifest",
    "write_mesh_projection_manifest",
    "write_quad8_pressure_load_report",
]
