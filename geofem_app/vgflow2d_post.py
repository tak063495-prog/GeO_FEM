"""Post-processing exports for the VGFlow 2D public substitute."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fem2d_elements import shape_functions
from .fem2d_types import FEM2DError, Mesh2D
from .fem2d_utils import _ensure_list
from .vgflow2d_kernels import (
    VGFLOW_MATERIAL_SATURATED,
    VGFLOW_MATERIAL_TABLE,
    VGFLOW_MATERIAL_VAN_GENUCHTEN,
    vgflow_contour_segments_numba,
    vgflow_post_element_fields_numba,
    vgflow_table_state_array_numba,
    vgflow_table_state_numba,
    vgflow_water_state_array_numba,
    vgflow_water_state_numba,
)
from .vgflow2d_video import write_vgflow_animation_avi


def write_vgflow_post_outputs(
    out: Path,
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    problem_type: str,
    gamma_w: float,
    seepage: Mapping[str, Any],
) -> dict[str, str]:
    paths = {
        "post_nodal_fields": str(out / "vgflow_post_nodal_fields.csv"),
        "post_contours": str(out / "vgflow_post_contours.csv"),
        "flow_vectors": str(out / "vgflow_flow_vectors.csv"),
        "flowlines": str(out / "vgflow_flowlines.csv"),
        "section_flows": str(out / "vgflow_section_flows.csv"),
        "time_history": str(out / "vgflow_time_history.csv"),
        "post_table_schema": str(out / "vgflow_post_table_schema.json"),
        "post_node_table_tsv": str(out / "vgflow_post_node_table.tsv"),
        "post_element_table_tsv": str(out / "vgflow_post_element_table.tsv"),
        "post_section_flow_units": str(out / "vgflow_post_section_flow_units.csv"),
        "post_animation_manifest": str(out / "vgflow_post_animation_manifest.json"),
        "post_animation_frames": str(out / "vgflow_post_animation_frames.csv"),
        "post_animation_html": str(out / "vgflow_post_animation.html"),
        "post_animation_avi": str(out / "vgflow_post_animation.avi"),
        "post_animation_avi_manifest": str(out / "vgflow_post_animation_avi_manifest.json"),
        "post_manifest": str(out / "vgflow_post_manifest.json"),
    }
    post = _post_options(seepage)
    elements_by_step = {step.index: _element_fields(mesh, materials, step, problem_type) for step in steps}
    fields_by_step = {step.index: _node_fields(mesh, materials, step, problem_type, gamma_w, elements_by_step[step.index]) for step in steps}
    _write_node_field_csv(Path(paths["post_nodal_fields"]), fields_by_step)
    _write_contour_csv(Path(paths["post_contours"]), mesh, steps, fields_by_step, post)
    _write_vector_csv(Path(paths["flow_vectors"]), steps, elements_by_step)
    _write_flowline_csv(Path(paths["flowlines"]), mesh, steps, elements_by_step, post)
    _write_section_flow_csv(Path(paths["section_flows"]), mesh, steps, elements_by_step, post)
    _write_time_history_csv(Path(paths["time_history"]), steps, fields_by_step, elements_by_step, post)
    _write_post_table_schema(Path(paths["post_table_schema"]))
    _write_copy_tables(Path(paths["post_node_table_tsv"]), Path(paths["post_element_table_tsv"]), fields_by_step, elements_by_step)
    _write_section_flow_units(Path(paths["post_section_flow_units"]), Path(paths["section_flows"]), post)
    avi_manifest = write_vgflow_animation_avi(Path(paths["post_animation_avi"]), Path(paths["post_animation_avi_manifest"]), mesh, steps, problem_type, post)
    _write_animation_outputs(Path(paths["post_animation_manifest"]), Path(paths["post_animation_frames"]), Path(paths["post_animation_html"]), steps, paths, avi_manifest)
    manifest = {
        "schema": "geofem.vgflow2d.post.public_substitute.v1",
        "features": [
            "equal_potential_contour_segments",
            "pore_pressure_contour_segments",
            "hydraulic_gradient_contour_segments",
            "saturation_water_content_contour_segments",
            "velocity_vector_table",
            "flowline_polyline_table",
            "section_flow_table",
            "node_element_time_history_table",
            "unit_annotated_post_tables",
            "clipboard_friendly_tsv_tables",
            "flow_sign_convention_per_meter_thickness",
            "transient_animation_manifest",
            "direct_avi_animation_export",
        ],
        "artifacts": paths,
    }
    Path(paths["post_manifest"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def vgflow_node_post_fields(mesh: Mesh2D, materials: Mapping[str, Any], step: Any, problem_type: str, gamma_w: float) -> list[dict[str, Any]]:
    return [dict(row) for row in _node_fields(mesh, materials, step, problem_type, gamma_w)]


def vgflow_element_post_fields(mesh: Mesh2D, materials: Mapping[str, Any], step: Any, problem_type: str) -> list[dict[str, Any]]:
    return [dict(row) for row in _element_fields(mesh, materials, step, problem_type)]


def _post_options(seepage: Mapping[str, Any]) -> Mapping[str, Any]:
    post = seepage.get("post", seepage.get("vgflow_post", {}))
    return post if isinstance(post, Mapping) else {}


def _node_fields(mesh: Mesh2D, materials: Mapping[str, Any], step: Any, problem_type: str, gamma_w: float, element_rows: Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, float | str | int]]:
    element_rows = list(element_rows) if element_rows is not None else _element_fields(mesh, materials, step, problem_type)
    grad_x, grad_y, vel_x, vel_y, material_by_node = _node_post_arrays(mesh, materials, element_rows)
    pressure_heads = _pressure_head_array(mesh, step.total_head, problem_type)
    theta, saturation = _node_water_state_arrays(materials, material_by_node, pressure_heads)
    rows: list[dict[str, float | str | int]] = []
    for i, nid in enumerate(mesh.node_ids):
        total_head = float(step.total_head[i])
        pressure_head = float(pressure_heads[i])
        gx = float(grad_x[i])
        gy = float(grad_y[i])
        vx = float(vel_x[i])
        vy = float(vel_y[i])
        rows.append(
            {
                "step": int(step.index),
                "time": float(step.time),
                "node_id": nid,
                "x": float(mesh.coords[i, 0]),
                "y": float(mesh.coords[i, 1]),
                "total_head_m": total_head,
                "pressure_head_m": pressure_head,
                "pore_pressure_kpa": gamma_w * pressure_head,
                "saturation": float(saturation[i]),
                "water_content": float(theta[i]),
                "hydraulic_gradient_x": gx,
                "hydraulic_gradient_y": gy,
                "hydraulic_gradient_abs": math.hypot(gx, gy),
                "velocity_x_m_s": vx,
                "velocity_y_m_s": vy,
                "velocity_abs_m_s": math.hypot(vx, vy),
            }
        )
    return rows


def _element_fields(mesh: Mesh2D, materials: Mapping[str, Any], step: Any, problem_type: str) -> list[dict[str, Any]]:
    fast = _element_field_arrays(mesh, materials, step, problem_type)
    if fast is not None:
        centers, gradients, velocities, bboxes = fast
        rows: list[dict[str, Any]] = []
        for i, element in enumerate(mesh.elements):
            gx = float(gradients[i, 0])
            gy = float(gradients[i, 1])
            vx = float(velocities[i, 0])
            vy = float(velocities[i, 1])
            rows.append(
                {
                    "_element": element,
                    "step": int(step.index),
                    "time": float(step.time),
                    "element_id": element.id,
                    "x": float(centers[i, 0]),
                    "y": float(centers[i, 1]),
                    "hydraulic_gradient_x": gx,
                    "hydraulic_gradient_y": gy,
                    "hydraulic_gradient_abs": math.hypot(gx, gy),
                    "velocity_x_m_s": vx,
                    "velocity_y_m_s": vy,
                    "velocity_abs_m_s": math.hypot(vx, vy),
                    "bbox": tuple(float(value) for value in bboxes[i]),
                }
            )
        return rows

    rows: list[dict[str, Any]] = []
    for element in mesh.elements:
        conn = [mesh.node_index[nid] for nid in element.nodes]
        coords = mesh.coords[conn]
        xi = eta = 0.0
        N, dN_dnatural = shape_functions(element.type, xi, eta)
        jac = dN_dnatural @ coords
        grad = np.linalg.inv(jac) @ dN_dnatural
        grad_h = grad @ step.total_head[conn]
        material = materials[element.material]
        pressure_head = _pressure_head(mesh, element.nodes[0], float(np.mean(step.total_head[conn])), problem_type)
        tensor = _permeability_tensor(material, _water_state(material, pressure_head)["kr"])
        velocity = -(tensor @ grad_h)
        center = N @ coords
        rows.append(
            {
                "_element": element,
                "step": int(step.index),
                "time": float(step.time),
                "element_id": element.id,
                "x": float(center[0]),
                "y": float(center[1]),
                "hydraulic_gradient_x": float(grad_h[0]),
                "hydraulic_gradient_y": float(grad_h[1]),
                "hydraulic_gradient_abs": float(np.linalg.norm(grad_h)),
                "velocity_x_m_s": float(velocity[0]),
                "velocity_y_m_s": float(velocity[1]),
                "velocity_abs_m_s": float(np.linalg.norm(velocity)),
                "bbox": _bbox(coords),
            }
        )
    return rows


def _element_field_arrays(mesh: Mesh2D, materials: Mapping[str, Any], step: Any, problem_type: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    element_type_codes = {"TRI3": 3, "QUAD4": 4, "TRI6": 6, "QUAD8": 8}
    grouped: dict[str, list[int]] = {}
    for index, element in enumerate(mesh.elements):
        etype = element.type.upper()
        if etype not in element_type_codes:
            return None
        grouped.setdefault(etype, []).append(index)
    material_names, material_index, kx, ky, angle_rad, model_codes, table_offsets, table_values, alpha, n_values, theta_r, theta_s = _material_numeric_arrays(materials)
    centers = np.empty((len(mesh.elements), 2), dtype=float)
    gradients = np.empty((len(mesh.elements), 2), dtype=float)
    velocities = np.empty((len(mesh.elements), 2), dtype=float)
    bboxes = np.empty((len(mesh.elements), 4), dtype=float)
    for etype, indices in grouped.items():
        node_count = element_type_codes[etype]
        connectivity = np.empty((len(indices), node_count), dtype=np.int64)
        material_ids = np.empty(len(indices), dtype=np.int64)
        for row, element_index in enumerate(indices):
            element = mesh.elements[element_index]
            if len(element.nodes) != node_count:
                return None
            connectivity[row, :] = [mesh.node_index[nid] for nid in element.nodes]
            material_ids[row] = material_index[element.material]
        c, g, v, b, invalid, det = vgflow_post_element_fields_numba(
            np.ascontiguousarray(mesh.coords, dtype=np.float64),
            connectivity,
            material_ids,
            np.ascontiguousarray(step.total_head, dtype=np.float64),
            kx,
            ky,
            angle_rad,
            model_codes,
            table_offsets,
            table_values,
            alpha,
            n_values,
            theta_r,
            theta_s,
            _problem_type_is_horizontal(problem_type),
            element_type_codes[etype],
        )
        if invalid >= 0:
            element = mesh.elements[indices[int(invalid)]]
            raise FEM2DError(f"{element.type}: detJ must be positive for VGFlow2D post, got {det:.6e}")
        centers[indices, :] = c
        gradients[indices, :] = g
        velocities[indices, :] = v
        bboxes[indices, :] = b
    return centers, gradients, velocities, bboxes


def _node_post_arrays(mesh: Mesh2D, materials: Mapping[str, Any], element_rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    node_count = len(mesh.node_ids)
    grad_x = np.zeros(node_count, dtype=float)
    grad_y = np.zeros(node_count, dtype=float)
    vel_x = np.zeros(node_count, dtype=float)
    vel_y = np.zeros(node_count, dtype=float)
    weight = np.zeros(node_count, dtype=float)
    material_names = list(materials)
    material_index = {name: i for i, name in enumerate(material_names)}
    material_by_node = np.full(node_count, 0, dtype=np.int64)
    assigned = np.zeros(node_count, dtype=bool)
    indices_parts: list[np.ndarray] = []
    grad_x_parts: list[np.ndarray] = []
    grad_y_parts: list[np.ndarray] = []
    vel_x_parts: list[np.ndarray] = []
    vel_y_parts: list[np.ndarray] = []
    for row in element_rows:
        element = row["_element"]
        conn = np.asarray([mesh.node_index[nid] for nid in element.nodes], dtype=np.int64)
        indices_parts.append(conn)
        count = len(conn)
        grad_x_parts.append(np.full(count, float(row["hydraulic_gradient_x"]), dtype=float))
        grad_y_parts.append(np.full(count, float(row["hydraulic_gradient_y"]), dtype=float))
        vel_x_parts.append(np.full(count, float(row["velocity_x_m_s"]), dtype=float))
        vel_y_parts.append(np.full(count, float(row["velocity_y_m_s"]), dtype=float))
        mat_id = material_index[element.material]
        new_nodes = conn[~assigned[conn]]
        material_by_node[new_nodes] = mat_id
        assigned[new_nodes] = True
    if indices_parts:
        indices = np.concatenate(indices_parts)
        np.add.at(grad_x, indices, np.concatenate(grad_x_parts))
        np.add.at(grad_y, indices, np.concatenate(grad_y_parts))
        np.add.at(vel_x, indices, np.concatenate(vel_x_parts))
        np.add.at(vel_y, indices, np.concatenate(vel_y_parts))
        np.add.at(weight, indices, 1.0)
    weight = np.maximum(weight, 1.0)
    return grad_x / weight, grad_y / weight, vel_x / weight, vel_y / weight, material_by_node


def _node_water_state_arrays(materials: Mapping[str, Any], material_by_node: np.ndarray, pressure_heads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    material_names, _material_index, _kx, _ky, _angle_rad, model_codes, table_offsets, table_values, alpha, n_values, theta_r, theta_s = _material_numeric_arrays(materials)
    theta = np.empty(pressure_heads.shape[0], dtype=float)
    saturation = np.empty(pressure_heads.shape[0], dtype=float)
    for mat_id, _name in enumerate(material_names):
        mask = material_by_node == mat_id
        if not np.any(mask):
            continue
        pressures = np.ascontiguousarray(pressure_heads[mask], dtype=np.float64)
        if model_codes[mat_id] == VGFLOW_MATERIAL_TABLE:
            _psi, theta_values, sat_values, _kr, _capacity = vgflow_table_state_array_numba(
                table_values,
                int(table_offsets[mat_id]),
                int(table_offsets[mat_id + 1]),
                pressures,
                float(theta_r[mat_id]),
                float(theta_s[mat_id]),
            )
        else:
            theta_values, sat_values, _kr, _capacity = vgflow_water_state_array_numba(
                int(model_codes[mat_id]),
                pressures,
                float(alpha[mat_id]),
                float(n_values[mat_id]),
                float(theta_r[mat_id]),
                float(theta_s[mat_id]),
            )
        theta[mask] = theta_values
        saturation[mask] = sat_values
    return theta, saturation


def _pressure_head_array(mesh: Mesh2D, total_head: np.ndarray, problem_type: str) -> np.ndarray:
    head = np.asarray(total_head, dtype=float)
    if _problem_type_is_horizontal(problem_type):
        return head.copy()
    return head - mesh.coords[:, 1]


def _problem_type_is_horizontal(problem_type: str) -> bool:
    return problem_type in {"horizontal", "plane", "plan"}


def _material_numeric_arrays(
    materials: Mapping[str, Any],
) -> tuple[list[str], dict[str, int], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    names = list(materials)
    index = {name: i for i, name in enumerate(names)}
    kx = np.asarray([float(materials[name].kx) for name in names], dtype=np.float64)
    ky = np.asarray([float(materials[name].ky) for name in names], dtype=np.float64)
    angle_rad = np.radians(np.asarray([float(getattr(materials[name], "angle_deg", 0.0)) for name in names], dtype=np.float64))
    model_codes = np.asarray([_material_model_code(materials[name]) for name in names], dtype=np.int64)
    alpha = np.asarray([float(getattr(materials[name], "alpha", 1.0)) for name in names], dtype=np.float64)
    n_values = np.asarray([float(getattr(materials[name], "n", 2.0)) for name in names], dtype=np.float64)
    theta_r = np.asarray([float(getattr(materials[name], "theta_r", 0.0)) for name in names], dtype=np.float64)
    theta_s = np.asarray([float(getattr(materials[name], "theta_s", 1.0)) for name in names], dtype=np.float64)
    table_offsets, table_values = _material_table_arrays([materials[name] for name in names])
    return names, index, kx, ky, angle_rad, model_codes, table_offsets, table_values, alpha, n_values, theta_r, theta_s


def _material_model_code(material: Any) -> int:
    if getattr(material, "unsaturated_model", "saturated") == "table" and getattr(material, "table", ()):
        return VGFLOW_MATERIAL_TABLE
    if getattr(material, "unsaturated_model", "saturated") == "van_genuchten":
        return VGFLOW_MATERIAL_VAN_GENUCHTEN
    return VGFLOW_MATERIAL_SATURATED


def _material_table_arrays(materials: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    rows: list[tuple[float, float, float]] = []
    for material in materials:
        rows.extend(getattr(material, "table", ()))
        offsets.append(len(rows))
    values = np.asarray(rows, dtype=np.float64) if rows else np.zeros((1, 3), dtype=np.float64)
    return np.asarray(offsets, dtype=np.int64), np.ascontiguousarray(values, dtype=np.float64)


def _write_node_field_csv(path: Path, fields_by_step: Mapping[int, list[dict[str, Any]]]) -> None:
    fields = [
        "step",
        "time",
        "node_id",
        "x",
        "y",
        "total_head_m",
        "pressure_head_m",
        "pore_pressure_kpa",
        "saturation",
        "water_content",
        "hydraulic_gradient_x",
        "hydraulic_gradient_y",
        "hydraulic_gradient_abs",
        "velocity_x_m_s",
        "velocity_y_m_s",
        "velocity_abs_m_s",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in sorted(fields_by_step):
            for row in fields_by_step[step]:
                writer.writerow({key: row.get(key, "") for key in fields})


def _write_contour_csv(path: Path, mesh: Mesh2D, steps: Sequence[Any], fields_by_step: Mapping[int, list[dict[str, Any]]], post: Mapping[str, Any]) -> None:
    variables = [
        "total_head_m",
        "pore_pressure_kpa",
        "hydraulic_gradient_x",
        "hydraulic_gradient_y",
        "hydraulic_gradient_abs",
        "saturation",
        "water_content",
    ]
    fields = ["step", "time", "variable", "level", "element_id", "x1", "y1", "x2", "y2"]
    corner_connectivity, corner_counts = _contour_corner_arrays(mesh)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            node_rows = {str(row["node_id"]): row for row in fields_by_step[step.index]}
            for variable in variables:
                levels = _contour_levels(node_rows.values(), variable, post)
                values = _node_value_array(mesh, node_rows, variable)
                for level, segment in _contour_segments_for_levels(mesh, corner_connectivity, corner_counts, values, levels):
                    writer.writerow(
                        {
                            "step": step.index,
                            "time": step.time,
                            "variable": variable,
                            "level": level,
                            **segment,
                        }
                    )


def _write_vector_csv(path: Path, steps: Sequence[Any], elements_by_step: Mapping[int, list[dict[str, Any]]]) -> None:
    fields = ["step", "time", "element_id", "x", "y", "velocity_x_m_s", "velocity_y_m_s", "velocity_abs_m_s", "hydraulic_gradient_x", "hydraulic_gradient_y", "hydraulic_gradient_abs"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            for row in elements_by_step[step.index]:
                writer.writerow({key: row.get(key, "") for key in fields})


def _write_flowline_csv(path: Path, mesh: Mesh2D, steps: Sequence[Any], elements_by_step: Mapping[int, list[dict[str, Any]]], post: Mapping[str, Any]) -> None:
    fields = ["step", "time", "line_id", "point_index", "x", "y", "velocity_x_m_s", "velocity_y_m_s", "velocity_abs_m_s"]
    x_min, y_min, x_max, y_max = _mesh_bbox(mesh)
    seed_count = int(post.get("flowline_seed_count", 7) or 7)
    seeds = _flowline_seeds(post, x_min, y_min, x_max, y_max, seed_count)
    step_length = float(post.get("flowline_step_length", max(x_max - x_min, y_max - y_min) / 40.0) or 1.0)
    max_points = int(post.get("flowline_max_points", 120) or 120)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            elements = elements_by_step[step.index]
            centers, velocities, bboxes = _element_lookup_arrays(elements)
            for line_id, seed in enumerate(seeds, start=1):
                x, y = seed
                for point_index in range(max_points):
                    vx, vy = _velocity_at_arrays(centers, velocities, bboxes, x, y)
                    speed = math.hypot(vx, vy)
                    writer.writerow(
                        {
                            "step": step.index,
                            "time": step.time,
                            "line_id": line_id,
                            "point_index": point_index,
                            "x": x,
                            "y": y,
                            "velocity_x_m_s": vx,
                            "velocity_y_m_s": vy,
                            "velocity_abs_m_s": speed,
                        }
                    )
                    if speed <= 1.0e-30:
                        break
                    x += vx / speed * step_length
                    y += vy / speed * step_length
                    if x < x_min or x > x_max or y < y_min or y > y_max:
                        break


def _write_section_flow_csv(path: Path, mesh: Mesh2D, steps: Sequence[Any], elements_by_step: Mapping[int, list[dict[str, Any]]], post: Mapping[str, Any]) -> None:
    fields = ["step", "time", "section", "orientation", "normal_x", "normal_y", "length_m", "flow_rate_m3_s_per_m", "abs_flow_rate_m3_s_per_m"]
    sections = _flow_sections(post, mesh)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            for section in sections:
                length, flow = _section_flow(elements_by_step[step.index], section)
                writer.writerow(
                    {
                        "step": step.index,
                        "time": step.time,
                        "section": section["name"],
                        "orientation": section["orientation"],
                        "normal_x": section["normal_x"],
                        "normal_y": section["normal_y"],
                        "length_m": length,
                        "flow_rate_m3_s_per_m": flow,
                        "abs_flow_rate_m3_s_per_m": abs(flow),
                    }
                )


def _write_time_history_csv(path: Path, steps: Sequence[Any], fields_by_step: Mapping[int, list[dict[str, Any]]], elements_by_step: Mapping[int, list[dict[str, Any]]], post: Mapping[str, Any]) -> None:
    fields = ["step", "time", "kind", "id", "x", "y", "total_head_m", "pressure_head_m", "pore_pressure_kpa", "saturation", "water_content", "hydraulic_gradient_abs", "velocity_abs_m_s"]
    history_nodes = {str(item) for item in _ensure_list(post.get("history_nodes", []))}
    history_elements = {str(item) for item in _ensure_list(post.get("history_elements", []))}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            for row in fields_by_step[step.index]:
                if history_nodes and str(row["node_id"]) not in history_nodes:
                    continue
                writer.writerow(
                    {
                        "step": step.index,
                        "time": step.time,
                        "kind": "node",
                        "id": row["node_id"],
                        **{key: row.get(key, "") for key in fields if key not in {"step", "time", "kind", "id"}},
                    }
                )
            for row in elements_by_step[step.index]:
                if history_elements and str(row["element_id"]) not in history_elements:
                    continue
                writer.writerow(
                    {
                        "step": step.index,
                        "time": step.time,
                        "kind": "element",
                        "id": row["element_id"],
                        "x": row["x"],
                        "y": row["y"],
                        "hydraulic_gradient_abs": row["hydraulic_gradient_abs"],
                        "velocity_abs_m_s": row["velocity_abs_m_s"],
                    }
                )


def _write_post_table_schema(path: Path) -> None:
    schema = {
        "schema": "geofem.vgflow2d.post_table_schema.public_substitute.v1",
        "copy_format": "tab-separated values",
        "columns": {
            "node": _node_display_columns(),
            "element": _element_display_columns(),
            "section_flow": _section_flow_display_columns(),
        },
        "flow_sign_convention": {
            "vertical_section": "positive when flow crosses the section in +X direction",
            "horizontal_section": "positive when flow crosses the section in +Y direction",
            "thickness_basis": "section flow is reported per 1 m out-of-plane thickness by default",
        },
    }
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_copy_tables(
    node_path: Path,
    element_path: Path,
    fields_by_step: Mapping[int, list[dict[str, Any]]],
    elements_by_step: Mapping[int, list[dict[str, Any]]],
) -> None:
    _write_tsv(node_path, _node_display_columns(), _flatten_steps(fields_by_step))
    element_rows = []
    for row in _flatten_steps(elements_by_step):
        cleaned = {key: value for key, value in row.items() if key not in {"_element", "bbox"}}
        element_rows.append(cleaned)
    _write_tsv(element_path, _element_display_columns(), element_rows)


def _write_tsv(path: Path, columns: Sequence[Mapping[str, str]], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([column["label"] for column in columns])
        for row in rows:
            writer.writerow([_display_value(row.get(column["key"], "")) for column in columns])


def _write_section_flow_units(path: Path, section_flow_path: Path, post: Mapping[str, Any]) -> None:
    thickness = float(post.get("thickness_m", post.get("out_of_plane_thickness_m", 1.0)) or 1.0)
    fields = [
        "step",
        "time",
        "section",
        "orientation",
        "positive_direction",
        "length_m",
        "thickness_m",
        "flow_rate_m3_s_per_m",
        "flow_rate_m3_s",
        "abs_flow_rate_m3_s",
        "unit_note",
    ]
    with section_flow_path.open(encoding="utf-8", newline="") as src, path.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=fields)
        writer.writeheader()
        for row in reader:
            per_m = float(row.get("flow_rate_m3_s_per_m", 0.0) or 0.0)
            orientation = row.get("orientation", "")
            direction = "+X" if orientation == "vertical" else "+Y" if orientation == "horizontal" else "section normal"
            writer.writerow(
                {
                    "step": row.get("step", ""),
                    "time": row.get("time", ""),
                    "section": row.get("section", ""),
                    "orientation": orientation,
                    "positive_direction": direction,
                    "length_m": row.get("length_m", ""),
                    "thickness_m": thickness,
                    "flow_rate_m3_s_per_m": per_m,
                    "flow_rate_m3_s": per_m * thickness,
                    "abs_flow_rate_m3_s": abs(per_m * thickness),
                    "unit_note": "m3/s per m thickness is converted by thickness_m for total flow.",
                }
            )


def _write_animation_outputs(manifest_path: Path, frames_path: Path, html_path: Path, steps: Sequence[Any], paths: Mapping[str, str], avi_manifest: Mapping[str, Any]) -> None:
    frame_rows = _animation_rows(steps, paths)
    fields = ["frame", "step", "time", "view", "variable", "source_artifact", "description"]
    with frames_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in frame_rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    manifest = {
        "schema": "geofem.vgflow2d.animation_manifest.public_substitute.v1",
        "profile": "CSV/HTML animation timeline plus direct lightweight AVI binary export for the public substitute.",
        "frame_count": len(frame_rows),
        "features": [
            "time_varying_contour_frames",
            "flowline_framenet_frames",
            "html_timeline_preview",
            "direct_avi_animation_export",
        ],
        "artifacts": {
            "frames_csv": str(frames_path),
            "html": str(html_path),
            "avi": paths.get("post_animation_avi", ""),
            "avi_manifest": paths.get("post_animation_avi_manifest", ""),
            "contours": paths.get("post_contours", ""),
            "flowlines": paths.get("flowlines", ""),
        },
        "avi": dict(avi_manifest),
        "frames": frame_rows,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    html_path.write_text(_animation_html(frame_rows), encoding="utf-8")


def _animation_rows(steps: Sequence[Any], paths: Mapping[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variables = [
        ("contour", "total_head_m", paths.get("post_contours", ""), "equal potential / total head contour"),
        ("contour", "pore_pressure_kpa", paths.get("post_contours", ""), "pore pressure contour"),
        ("flownet", "flowline", paths.get("flowlines", ""), "flowline / flownet frame"),
    ]
    frame = 0
    for step in steps:
        for view, variable, artifact, description in variables:
            rows.append(
                {
                    "frame": frame,
                    "step": int(step.index),
                    "time": float(step.time),
                    "view": view,
                    "variable": variable,
                    "source_artifact": artifact,
                    "description": description,
                }
            )
            frame += 1
    return rows


def _animation_html(rows: Sequence[Mapping[str, Any]]) -> str:
    body = "\n".join(
        "<tr>"
        f"<td>{row.get('frame', '')}</td>"
        f"<td>{row.get('step', '')}</td>"
        f"<td>{row.get('time', '')}</td>"
        f"<td>{row.get('view', '')}</td>"
        f"<td>{row.get('variable', '')}</td>"
        f"<td>{row.get('description', '')}</td>"
        "</tr>"
        for row in rows
    )
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D Post Animation Timeline</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left}th{background:#f6f8fa}</style></head><body>"
        "<h1>VGFlow 2D Post Animation Timeline</h1>"
        "<p>Public substitute timeline for transient contours and flowlines. Use the CSV sources for external video encoding when needed.</p>"
        "<table><thead><tr><th>frame</th><th>step</th><th>time</th><th>view</th><th>variable</th><th>description</th></tr></thead>"
        f"<tbody>{body}</tbody></table></body></html>"
    )


def _flatten_steps(rows_by_step: Mapping[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in sorted(rows_by_step):
        rows.extend(rows_by_step[step])
    return rows


def _display_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12g}"
    return value


def _node_display_columns() -> list[dict[str, str]]:
    return [
        {"key": "step", "label": "Step", "unit": "-"},
        {"key": "time", "label": "Time", "unit": "s"},
        {"key": "node_id", "label": "Node", "unit": "-"},
        {"key": "x", "label": "X (m)", "unit": "m"},
        {"key": "y", "label": "Y (m)", "unit": "m"},
        {"key": "total_head_m", "label": "Total head (m)", "unit": "m"},
        {"key": "pressure_head_m", "label": "Pressure head (m)", "unit": "m"},
        {"key": "pore_pressure_kpa", "label": "Pore pressure (kPa)", "unit": "kPa"},
        {"key": "saturation", "label": "Saturation (-)", "unit": "-"},
        {"key": "water_content", "label": "Water content (-)", "unit": "-"},
        {"key": "hydraulic_gradient_abs", "label": "Hydraulic gradient (-)", "unit": "-"},
        {"key": "velocity_abs_m_s", "label": "Velocity (m/s)", "unit": "m/s"},
    ]


def _element_display_columns() -> list[dict[str, str]]:
    return [
        {"key": "step", "label": "Step", "unit": "-"},
        {"key": "time", "label": "Time", "unit": "s"},
        {"key": "element_id", "label": "Element", "unit": "-"},
        {"key": "x", "label": "Center X (m)", "unit": "m"},
        {"key": "y", "label": "Center Y (m)", "unit": "m"},
        {"key": "hydraulic_gradient_x", "label": "Hydraulic gradient X (-)", "unit": "-"},
        {"key": "hydraulic_gradient_y", "label": "Hydraulic gradient Y (-)", "unit": "-"},
        {"key": "hydraulic_gradient_abs", "label": "Hydraulic gradient (-)", "unit": "-"},
        {"key": "velocity_x_m_s", "label": "Velocity X (m/s)", "unit": "m/s"},
        {"key": "velocity_y_m_s", "label": "Velocity Y (m/s)", "unit": "m/s"},
        {"key": "velocity_abs_m_s", "label": "Velocity (m/s)", "unit": "m/s"},
    ]


def _section_flow_display_columns() -> list[dict[str, str]]:
    return [
        {"key": "step", "label": "Step", "unit": "-"},
        {"key": "time", "label": "Time", "unit": "s"},
        {"key": "section", "label": "Section", "unit": "-"},
        {"key": "orientation", "label": "Orientation", "unit": "-"},
        {"key": "positive_direction", "label": "Positive direction", "unit": "-"},
        {"key": "flow_rate_m3_s_per_m", "label": "Flow per 1 m thickness (m3/s/m)", "unit": "m3/s/m"},
        {"key": "flow_rate_m3_s", "label": "Flow (m3/s)", "unit": "m3/s"},
    ]


def _contour_levels(rows: Sequence[Mapping[str, Any]], variable: str, post: Mapping[str, Any]) -> list[float]:
    custom = post.get("contour_levels", {})
    if isinstance(custom, Mapping) and variable in custom:
        return [float(value) for value in _ensure_list(custom[variable])]
    values = [float(row[variable]) for row in rows if variable in row]
    if not values:
        return []
    low, high = min(values), max(values)
    if abs(high - low) <= 1.0e-14:
        return []
    count = max(1, int(post.get("contour_level_count", 8) or 8))
    return [float(value) for value in np.linspace(low, high, count + 2)[1:-1]]


def _contour_corner_arrays(mesh: Mesh2D) -> tuple[np.ndarray, np.ndarray]:
    connectivity = np.zeros((len(mesh.elements), 4), dtype=np.int64)
    counts = np.zeros(len(mesh.elements), dtype=np.int64)
    for i, element in enumerate(mesh.elements):
        corners = list(element.nodes[:4] if element.type.upper().startswith("QUAD") else element.nodes[:3])
        counts[i] = len(corners)
        for j, nid in enumerate(corners):
            connectivity[i, j] = mesh.node_index[nid]
    return connectivity, counts


def _node_value_array(mesh: Mesh2D, node_rows: Mapping[str, Mapping[str, Any]], variable: str) -> np.ndarray:
    return np.asarray([float(node_rows[nid][variable]) for nid in mesh.node_ids], dtype=np.float64)


def _contour_segments_for_levels(
    mesh: Mesh2D,
    corner_connectivity: np.ndarray,
    corner_counts: np.ndarray,
    values: np.ndarray,
    levels: Sequence[float],
) -> list[tuple[float, dict[str, Any]]]:
    if not levels:
        return []
    level_array = np.asarray(levels, dtype=np.float64)
    level_indices, element_indices, p0, p1, count = vgflow_contour_segments_numba(
        np.ascontiguousarray(mesh.coords, dtype=np.float64),
        corner_connectivity,
        corner_counts,
        np.ascontiguousarray(values, dtype=np.float64),
        level_array,
    )
    segments: list[tuple[float, dict[str, Any]]] = []
    for i in range(int(count)):
        segments.append(
            (
                float(level_array[int(level_indices[i])]),
                {
                    "element_id": mesh.elements[int(element_indices[i])].id,
                    "x1": float(p0[i, 0]),
                    "y1": float(p0[i, 1]),
                    "x2": float(p1[i, 0]),
                    "y2": float(p1[i, 1]),
                },
            )
        )
    return segments


def _contour_segments(mesh: Mesh2D, node_rows: Mapping[str, Mapping[str, Any]], variable: str, level: float) -> list[dict[str, Any]]:
    connectivity, counts = _contour_corner_arrays(mesh)
    values = _node_value_array(mesh, node_rows, variable)
    return [segment for _level, segment in _contour_segments_for_levels(mesh, connectivity, counts, values, [level])]


def _flowline_seeds(post: Mapping[str, Any], x_min: float, y_min: float, x_max: float, y_max: float, seed_count: int) -> list[tuple[float, float]]:
    raw = post.get("flowline_seeds", post.get("streamline_seeds"))
    if raw:
        seeds: list[tuple[float, float]] = []
        for item in _ensure_list(raw):
            if isinstance(item, Mapping):
                seeds.append((float(item.get("x", 0.0)), float(item.get("y", 0.0))))
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                seeds.append((float(item[0]), float(item[1])))
        if seeds:
            return seeds
    return [(x_min, y) for y in np.linspace(y_min, y_max, max(seed_count, 2))]


def _flow_sections(post: Mapping[str, Any], mesh: Mesh2D) -> list[dict[str, Any]]:
    x_min, y_min, x_max, y_max = _mesh_bbox(mesh)
    raw_sections = _ensure_list(post.get("flow_sections", post.get("sections", post.get("discharge_sections", []))))
    if not raw_sections:
        raw_sections = [{"name": "mid_x", "x": 0.5 * (x_min + x_max), "y_range": [y_min, y_max]}]
    sections: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_sections, start=1):
        if not isinstance(raw, Mapping):
            continue
        name = str(raw.get("name", f"section_{index}"))
        if "x" in raw:
            yr = _ensure_list(raw.get("y_range", [y_min, y_max]))
            sections.append({"name": name, "orientation": "vertical", "x": float(raw["x"]), "a": float(yr[0]), "b": float(yr[-1]), "normal_x": 1.0, "normal_y": 0.0})
        elif "y" in raw:
            xr = _ensure_list(raw.get("x_range", [x_min, x_max]))
            sections.append({"name": name, "orientation": "horizontal", "y": float(raw["y"]), "a": float(xr[0]), "b": float(xr[-1]), "normal_x": 0.0, "normal_y": 1.0})
    return sections


def _section_flow(elements: Sequence[Mapping[str, Any]], section: Mapping[str, Any]) -> tuple[float, float]:
    _centers, velocities, bboxes = _element_lookup_arrays(elements)
    if section["orientation"] == "vertical":
        x = float(section["x"])
        a, b = sorted((float(section["a"]), float(section["b"])))
        mask = (bboxes[:, 0] - 1.0e-12 <= x) & (x <= bboxes[:, 2] + 1.0e-12)
        lengths = np.maximum(0.0, np.minimum(b, bboxes[:, 3]) - np.maximum(a, bboxes[:, 1]))
        lengths = np.where(mask, lengths, 0.0)
        return float(np.sum(lengths)), float(np.dot(velocities[:, 0], lengths))
    elif section["orientation"] == "horizontal":
        y = float(section["y"])
        a, b = sorted((float(section["a"]), float(section["b"])))
        mask = (bboxes[:, 1] - 1.0e-12 <= y) & (y <= bboxes[:, 3] + 1.0e-12)
        lengths = np.maximum(0.0, np.minimum(b, bboxes[:, 2]) - np.maximum(a, bboxes[:, 0]))
        lengths = np.where(mask, lengths, 0.0)
        return float(np.sum(lengths)), float(np.dot(velocities[:, 1], lengths))
    return 0.0, 0.0


def _velocity_at(elements: Sequence[Mapping[str, Any]], x: float, y: float) -> tuple[float, float]:
    centers, velocities, bboxes = _element_lookup_arrays(elements)
    return _velocity_at_arrays(centers, velocities, bboxes, x, y)


def _element_lookup_arrays(elements: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray([[float(row["x"]), float(row["y"])] for row in elements], dtype=float)
    velocities = np.asarray([[float(row["velocity_x_m_s"]), float(row["velocity_y_m_s"])] for row in elements], dtype=float)
    bboxes = np.asarray([row["bbox"] for row in elements], dtype=float)
    return centers, velocities, bboxes


def _velocity_at_arrays(centers: np.ndarray, velocities: np.ndarray, bboxes: np.ndarray, x: float, y: float) -> tuple[float, float]:
    containing = (bboxes[:, 0] - 1.0e-12 <= x) & (x <= bboxes[:, 2] + 1.0e-12) & (bboxes[:, 1] - 1.0e-12 <= y) & (y <= bboxes[:, 3] + 1.0e-12)
    candidates = np.nonzero(containing)[0]
    if candidates.size == 0:
        candidates = np.arange(centers.shape[0])
    delta = centers[candidates] - np.array([x, y], dtype=float)
    nearest = int(candidates[int(np.argmin(np.sum(delta * delta, axis=1)))])
    return float(velocities[nearest, 0]), float(velocities[nearest, 1])


def _inside_bbox(bbox: tuple[float, float, float, float], x: float, y: float) -> bool:
    x0, y0, x1, y1 = bbox
    return x0 - 1.0e-12 <= x <= x1 + 1.0e-12 and y0 - 1.0e-12 <= y <= y1 + 1.0e-12


def _mesh_bbox(mesh: Mesh2D) -> tuple[float, float, float, float]:
    return (float(np.min(mesh.coords[:, 0])), float(np.min(mesh.coords[:, 1])), float(np.max(mesh.coords[:, 0])), float(np.max(mesh.coords[:, 1])))


def _bbox(coords: np.ndarray) -> tuple[float, float, float, float]:
    return (float(np.min(coords[:, 0])), float(np.min(coords[:, 1])), float(np.max(coords[:, 0])), float(np.max(coords[:, 1])))


def _pressure_head(mesh: Mesh2D, nid: str, total_head: float, problem_type: str) -> float:
    if problem_type == "horizontal":
        return total_head
    return total_head - float(mesh.coords[mesh.node_index[nid], 1])


def _permeability_tensor(material: Any, kr: float) -> np.ndarray:
    angle = math.radians(float(getattr(material, "angle_deg", 0.0)))
    c = math.cos(angle)
    s = math.sin(angle)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    local = np.diag([float(material.kx) * kr, float(material.ky) * kr])
    return rot @ local @ rot.T


def _water_state(material: Any, pressure_head: float) -> dict[str, float]:
    if getattr(material, "unsaturated_model", "saturated") == "table" and getattr(material, "table", ()):
        psi, theta, kr = _table_state(material.table, pressure_head)
        saturation = _theta_saturation(material, theta)
        return {"pressure_head": psi, "theta": theta, "saturation": saturation, "kr": kr}
    if getattr(material, "unsaturated_model", "saturated") == "van_genuchten":
        if pressure_head >= 0.0:
            return {"pressure_head": pressure_head, "theta": float(material.theta_s), "saturation": 1.0, "kr": 1.0}
        suction = -pressure_head
        n = float(material.n)
        m = 1.0 - 1.0 / n
        alpha_s = float(material.alpha) * suction
        se = (1.0 + alpha_s**n) ** (-m)
        theta = float(material.theta_r) + se * (float(material.theta_s) - float(material.theta_r))
        inner = max(0.0, 1.0 - (1.0 - se ** (1.0 / m)) ** m)
        kr = math.sqrt(max(se, 0.0)) * inner * inner
        return {"pressure_head": pressure_head, "theta": theta, "saturation": se, "kr": max(0.0, min(1.0, kr))}
    return {"pressure_head": pressure_head, "theta": float(getattr(material, "theta_s", 1.0)), "saturation": 1.0, "kr": 1.0}


def _table_state(table: Sequence[tuple[float, float, float]], pressure_head: float) -> tuple[float, float, float]:
    if pressure_head <= table[0][0]:
        return table[0]
    if pressure_head >= table[-1][0]:
        return table[-1]
    for left, right in zip(table[:-1], table[1:]):
        if left[0] <= pressure_head <= right[0]:
            t = (pressure_head - left[0]) / max(right[0] - left[0], np.finfo(float).eps)
            return pressure_head, left[1] + t * (right[1] - left[1]), left[2] + t * (right[2] - left[2])
    return table[-1]


def _theta_saturation(material: Any, theta: float) -> float:
    theta_r = float(getattr(material, "theta_r", 0.0))
    theta_s = float(getattr(material, "theta_s", 1.0))
    return max(0.0, min(1.0, (theta - theta_r) / max(theta_s - theta_r, np.finfo(float).eps)))


__all__ = ["vgflow_element_post_fields", "vgflow_node_post_fields", "write_vgflow_post_outputs"]
