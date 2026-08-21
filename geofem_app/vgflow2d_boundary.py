"""Boundary-condition diagnostics for the VGFlow 2D public substitute."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .fem2d_mesh import _edge_length, _pressure_edges, _target_nodes
from .fem2d_types import FEM2DError, Mesh2D
from .fem2d_utils import _ensure_list
from .html_report_utils import report_css


CurveReader = Callable[..., list[dict[str, float]]]


def write_vgflow_boundary_diagnostics(
    out: Path,
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    times: Sequence[float],
    seepage: Mapping[str, Any],
    problem_type: str,
    curve_reader: CurveReader,
) -> dict[str, str]:
    paths = {
        "boundary_diagnostics_json": str(out / "vgflow_boundary_diagnostics.json"),
        "boundary_diagnostics_csv": str(out / "vgflow_boundary_diagnostics.csv"),
        "boundary_diagnostics_html": str(out / "vgflow_boundary_diagnostics.html"),
    }
    diagnostics = collect_vgflow_boundary_diagnostics(mesh, materials, times, seepage, problem_type, curve_reader)
    payload = {
        "schema": "geofem.vgflow2d.boundary_diagnostics.public_substitute.v1",
        "features": [
            "rainfall_infiltration_excess_diagnostics",
            "fixed_head_flux_overlap_diagnostics",
            "boundary_curve_time_alignment",
        ],
        "analysis_times": [float(value) for value in times],
        "diagnostics": diagnostics,
    }
    Path(paths["boundary_diagnostics_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_diagnostics_csv(Path(paths["boundary_diagnostics_csv"]), diagnostics)
    Path(paths["boundary_diagnostics_html"]).write_text(_diagnostics_html(payload), encoding="utf-8")
    return paths


def collect_vgflow_boundary_diagnostics(
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    times: Sequence[float],
    seepage: Mapping[str, Any],
    problem_type: str,
    curve_reader: CurveReader,
) -> list[dict[str, Any]]:
    analysis_times = [float(value) for value in times] or [0.0]
    specs = _boundary_specs(seepage)
    diagnostics: list[dict[str, Any]] = []
    diagnostics.extend(_rainfall_capacity_diagnostics(mesh, materials, analysis_times, specs, problem_type))
    diagnostics.extend(_condition_overlap_diagnostics(mesh, analysis_times, specs, problem_type, curve_reader))
    diagnostics.extend(_curve_time_diagnostics(analysis_times, specs, curve_reader))
    if not diagnostics:
        diagnostics.append(_row("boundary_conditions", "pass", message="No VGFlow2D public substitute boundary-condition risks were detected."))
    return diagnostics


def _rainfall_capacity_diagnostics(
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    times: Sequence[float],
    specs: Sequence[Mapping[str, Any]],
    problem_type: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in specs:
        if entry["role"] != "rainfall":
            continue
        spec = entry["spec"]
        nodes = _nodes_from_spec(mesh, spec)
        edges = _edges_from_spec(mesh, spec)
        capacity = _infiltration_capacity(mesh, materials, nodes, problem_type)
        boundary_length = sum(_edge_length(mesh, edge) for edge in edges)
        for time_value in times:
            rainfall_flux = _flux_value(spec, time_value)
            excess = max(0.0, rainfall_flux - capacity)
            status = "warning" if excess > max(1.0e-12, abs(rainfall_flux) * 1.0e-9) else "pass"
            rows.append(
                _row(
                    "rainfall_infiltration_capacity",
                    status,
                    time=time_value,
                    group=entry["group"],
                    boundary_index=entry["index"],
                    target_set=_target_name(spec),
                    value=rainfall_flux,
                    limit=capacity,
                    excess=excess,
                    message=(
                        "Rainfall exceeds the public substitute saturated infiltration capacity; "
                        "route the excess as runoff or review impermeable-boundary assumptions."
                        if status == "warning"
                        else "Rainfall is within the public substitute saturated infiltration capacity."
                    ),
                    details={
                        "boundary_length_m": boundary_length,
                        "target_node_count": len(nodes),
                        "runoff_policy": spec.get("runoff_policy", spec.get("infiltration_policy", "diagnostic_only")),
                    },
                )
            )
    return rows


def _condition_overlap_diagnostics(
    mesh: Mesh2D,
    times: Sequence[float],
    specs: Sequence[Mapping[str, Any]],
    problem_type: str,
    curve_reader: CurveReader,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fixed_specs = [entry for entry in specs if entry["role"] == "fixed_head"]
    flux_specs = [entry for entry in specs if entry["role"] in {"flux", "rainfall"}]
    seepage_specs = [entry for entry in specs if entry["role"] == "seepage_face"]
    for time_value in times:
        fixed_by_node: dict[str, list[tuple[Mapping[str, Any], float]]] = {}
        for entry in fixed_specs:
            for nid in _nodes_from_spec(mesh, entry["spec"]):
                fixed_by_node.setdefault(nid, []).append((entry, _fixed_head_value(mesh, nid, entry["spec"], time_value, problem_type, curve_reader)))

        for nid, values in fixed_by_node.items():
            scalar_values = [value for _, value in values]
            if max(scalar_values) - min(scalar_values) > 1.0e-9:
                rows.append(
                    _row(
                        "fixed_head_value_conflict",
                        "fail",
                        time=time_value,
                        node_id=nid,
                        value=min(scalar_values),
                        limit=max(scalar_values),
                        message="Multiple fixed-head style conditions assign different total heads to the same node.",
                        details={"sources": [_source_label(entry) for entry, _ in values]},
                    )
                )

        fixed_nodes = set(fixed_by_node)
        for entry in flux_specs:
            overlap = sorted(fixed_nodes.intersection(_nodes_from_spec(mesh, entry["spec"])))
            for nid in overlap:
                rows.append(
                    _row(
                        "fixed_head_flux_overlap",
                        "warning",
                        time=time_value,
                        group=entry["group"],
                        boundary_index=entry["index"],
                        node_id=nid,
                        target_set=_target_name(entry["spec"]),
                        value=_flux_value(entry["spec"], time_value),
                        message="A fixed-head style boundary and a flux/rainfall boundary act on the same node.",
                        details={"fixed_sources": [_source_label(item[0]) for item in fixed_by_node[nid]], "flux_source": _source_label(entry)},
                    )
                )

        rainfall_nodes = set()
        for entry in flux_specs:
            if entry["role"] == "rainfall":
                rainfall_nodes.update(_nodes_from_spec(mesh, entry["spec"]))
        for entry in seepage_specs:
            overlap = sorted(rainfall_nodes.intersection(_nodes_from_spec(mesh, entry["spec"])))
            for nid in overlap:
                rows.append(
                    _row(
                        "rainfall_seepage_face_overlap",
                        "warning",
                        time=time_value,
                        group=entry["group"],
                        boundary_index=entry["index"],
                        node_id=nid,
                        target_set=_target_name(entry["spec"]),
                        message="Rainfall and seepage-face switching share a node; inspect drainage/runoff assumptions for this time.",
                    )
                )
    return rows


def _curve_time_diagnostics(times: Sequence[float], specs: Sequence[Mapping[str, Any]], curve_reader: CurveReader) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not times:
        return rows
    analysis_start = min(times)
    analysis_end = max(times)
    curve_ends: list[float] = []
    for entry in specs:
        spec = entry["spec"]
        for key in _curve_keys(spec):
            pairs = _curve_pairs(spec, key, curve_reader)
            if not pairs:
                continue
            start = pairs[0][0]
            end = pairs[-1][0]
            curve_ends.append(end)
            status = "pass"
            message = "Boundary curve covers the analysis time range."
            if start > analysis_start + 1.0e-12:
                status = "warning"
                message = "Boundary curve starts after the first analysis time; the first curve value is held backward."
            elif end < analysis_end - 1.0e-12:
                status = "warning"
                message = "Boundary curve ends before the final analysis time; the last curve value is held forward."
            rows.append(
                _row(
                    "boundary_curve_time_range",
                    status,
                    group=entry["group"],
                    boundary_index=entry["index"],
                    target_set=_target_name(spec),
                    value_key=key,
                    value=end,
                    limit=analysis_end,
                    message=message,
                    details={"curve_start": start, "curve_end": end, "analysis_start": analysis_start, "analysis_end": analysis_end},
                )
            )
    if len(curve_ends) >= 2 and max(curve_ends) - min(curve_ends) > 1.0e-12:
        rows.append(
            _row(
                "boundary_curve_final_time_alignment",
                "warning",
                value=min(curve_ends),
                limit=max(curve_ends),
                message="Boundary curves do not share the same final time; verify VGFlow2D-style transient condition synchronization.",
            )
        )
    return rows


def _boundary_specs(seepage: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    groups = (
        "known_head_bcs",
        "head_bcs",
        "water_level_bcs",
        "pressure_head_bcs",
        "flux_bcs",
        "pore_flux_bcs",
        "flow_bcs",
        "rainfall_bcs",
        "point_sources",
        "point_source_bcs",
        "seepage_faces",
        "seepage_face_bcs",
    )
    for group in groups:
        for index, raw in enumerate(_ensure_list(seepage.get(group, []))):
            if isinstance(raw, Mapping):
                specs.append({"group": group, "index": index, "role": _role(group, raw), "spec": raw})
    for index, raw in enumerate(_ensure_list(seepage.get("boundary_conditions", seepage.get("bc", [])))):
        if isinstance(raw, Mapping):
            specs.append({"group": "boundary_conditions", "index": index, "role": _role("boundary_conditions", raw), "spec": raw})
    if "rainfall" in seepage:
        rainfall = seepage["rainfall"]
        spec = {"set": "top", "rainfall": rainfall} if not isinstance(rainfall, Mapping) else {"set": rainfall.get("set", "top"), **rainfall}
        specs.append({"group": "rainfall", "index": 0, "role": "rainfall", "spec": spec})
    return specs


def _role(group: str, spec: Mapping[str, Any]) -> str:
    kind = str(spec.get("type", spec.get("kind", ""))).lower().strip().replace("-", "_")
    if group in {"known_head_bcs", "head_bcs", "water_level_bcs", "pressure_head_bcs"} or kind in {"head", "known_head", "water_level", "pressure_head"}:
        return "fixed_head"
    if group in {"rainfall_bcs"} or kind == "rainfall" or "rainfall" in spec:
        return "rainfall"
    if group in {"flux_bcs", "pore_flux_bcs", "flow_bcs", "point_sources", "point_source_bcs"} or kind in {"flux", "flow", "q"}:
        return "flux"
    if group in {"seepage_faces", "seepage_face_bcs"} or kind == "seepage_face":
        return "seepage_face"
    return "other"


def _infiltration_capacity(mesh: Mesh2D, materials: Mapping[str, Any], nodes: Sequence[str], problem_type: str) -> float:
    target_nodes = set(nodes)
    candidates: list[float] = []
    axis = "kx" if problem_type == "horizontal" else "ky"
    for element in mesh.elements:
        if target_nodes and target_nodes.isdisjoint(element.nodes):
            continue
        material = materials.get(element.material)
        if material is not None:
            candidates.append(float(getattr(material, axis, getattr(material, "ky", getattr(material, "kx", 0.0))) or 0.0))
    if not candidates:
        candidates.extend(float(getattr(material, axis, getattr(material, "ky", getattr(material, "kx", 0.0))) or 0.0) for material in materials.values())
    return max(0.0, min(candidates) if candidates else 0.0)


def _nodes_from_spec(mesh: Mesh2D, spec: Mapping[str, Any]) -> list[str]:
    try:
        return _target_nodes(mesh, spec)
    except FEM2DError:
        nodes: list[str] = []
        for edge in _edges_from_spec(mesh, spec):
            nodes.extend(edge)
        return list(dict.fromkeys(nodes))


def _edges_from_spec(mesh: Mesh2D, spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    try:
        return list(_pressure_edges(mesh, spec))
    except FEM2DError:
        return []


def _target_name(spec: Mapping[str, Any]) -> str:
    for key in ("set", "node_set", "edge_set", "target", "boundary"):
        value = spec.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _fixed_head_value(mesh: Mesh2D, nid: str, spec: Mapping[str, Any], time: float, problem_type: str, curve_reader: CurveReader) -> float:
    kind = str(spec.get("type", spec.get("kind", ""))).lower().strip().replace("-", "_")
    if "pressure_head" in spec or kind == "pressure_head" or _curve_file_from_spec(spec, "pressure_head") not in (None, ""):
        pressure = _value_at_time(spec, "pressure_head", time, curve_reader, default=_value_at_time(spec, "value", time, curve_reader, default=0.0))
        return pressure if problem_type == "horizontal" else pressure + float(mesh.coords[mesh.node_index[nid], 1])
    for key in ("head", "total_head", "water_head", "water_level", "level", "value"):
        if key in spec or f"{key}_curve" in spec or _curve_file_from_spec(spec, key) not in (None, ""):
            return _value_at_time(spec, key, time, curve_reader)
    return 0.0


def _flux_value(spec: Mapping[str, Any], time: float) -> float:
    if "rainfall" in spec or str(spec.get("type", spec.get("kind", ""))).lower().strip().replace("-", "_") == "rainfall":
        value = _value_at_time_no_file(spec, "rainfall", time, default=_value_at_time_no_file(spec, "value", time, default=0.0))
        unit = str(spec.get("unit", spec.get("rainfall_unit", ""))).lower().replace(" ", "")
        if unit in {"mm/hr", "mm/h", "mmhour"}:
            return value / 1000.0 / 3600.0
        if unit == "mm/s":
            return value / 1000.0
        return value
    return _value_at_time_no_file(spec, "flux", time, default=_value_at_time_no_file(spec, "q", time, default=_value_at_time_no_file(spec, "value", time, default=0.0)))


def _value_at_time(spec: Mapping[str, Any], key: str, time: float, curve_reader: CurveReader, default: float | None = None) -> float:
    pairs = _curve_pairs(spec, key, curve_reader)
    if pairs:
        return _interpolate_pairs(pairs, time)
    return _value_at_time_no_file(spec, key, time, default=default)


def _value_at_time_no_file(spec: Mapping[str, Any], key: str, time: float, default: float | None = None) -> float:
    curve = spec.get(f"{key}_curve", spec.get("curve", spec.get("time_series")))
    pairs = _inline_curve_pairs(curve, key) if curve is not None else []
    if pairs:
        return _interpolate_pairs(pairs, time)
    value = spec.get(key, default)
    return 0.0 if value is None else float(value)


def _interpolate_pairs(pairs: Sequence[tuple[float, float]], time: float) -> float:
    if time <= pairs[0][0]:
        return pairs[0][1]
    if time >= pairs[-1][0]:
        return pairs[-1][1]
    for left, right in zip(pairs[:-1], pairs[1:]):
        if left[0] <= time <= right[0]:
            ratio = (time - left[0]) / max(right[0] - left[0], np.finfo(float).eps)
            return left[1] + ratio * (right[1] - left[1])
    return pairs[-1][1]


def _curve_pairs(spec: Mapping[str, Any], key: str, curve_reader: CurveReader) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    curve_path = _curve_file_from_spec(spec, key)
    if curve_path not in (None, ""):
        for row in curve_reader(curve_path, value_field=key):
            pairs.append((float(row["time"]), float(row.get(key, row.get("value", 0.0)))))
    curve = spec.get(f"{key}_curve", spec.get("curve", spec.get("time_series")))
    if curve is not None:
        pairs.extend(_inline_curve_pairs(curve, key))
    return sorted(pairs)


def _curve_file_from_spec(spec: Mapping[str, Any], key: str) -> Any:
    for field in (f"{key}_curve_file", f"{key}_time_series_file", f"{key}_file", "curve_file", "time_series_file"):
        value = spec.get(field)
        if value not in (None, ""):
            return value
    if "file" in spec and any(marker in spec for marker in ("curve", "time_series", "curve_file", "time_series_file")):
        return spec["file"]
    return None


def _inline_curve_pairs(curve: Any, key: str) -> list[tuple[float, float]]:
    pairs: list[tuple[float, float]] = []
    for row in _ensure_list(curve):
        if isinstance(row, Mapping):
            value = row.get(key, row.get("value", row.get("head", row.get("water_level", row.get("rainfall", 0.0)))))
            pairs.append((float(row.get("time", row.get("t", 0.0)) or 0.0), float(value or 0.0)))
        elif isinstance(row, (list, tuple)) and len(row) >= 2:
            pairs.append((float(row[0]), float(row[1])))
    return sorted(pairs)


def _curve_keys(spec: Mapping[str, Any]) -> list[str]:
    candidates = ("head", "water_level", "pressure_head", "rainfall", "flux", "q", "flow", "value")
    keys = [
        key
        for key in candidates
        if any(spec.get(field) is not None for field in (f"{key}_curve", f"{key}_curve_file", f"{key}_time_series_file", f"{key}_file"))
    ]
    if keys:
        return keys
    if not any(spec.get(field) is not None for field in ("curve", "time_series", "curve_file", "time_series_file")):
        return []
    role = _role("", spec)
    if role == "rainfall":
        return ["rainfall"]
    if role == "fixed_head":
        if "pressure_head" in spec:
            return ["pressure_head"]
        if "water_level" in spec:
            return ["water_level"]
        return ["head"]
    if role == "flux":
        return ["flux"]
    return ["value"]


def _source_label(entry: Mapping[str, Any]) -> str:
    return f"{entry['group']}[{entry['index']}]"


def _row(
    check: str,
    status: str,
    *,
    time: float | None = None,
    group: Any = "",
    boundary_index: Any = "",
    node_id: Any = "",
    target_set: str = "",
    value_key: str = "",
    value: Any = "",
    limit: Any = "",
    excess: Any = "",
    message: str = "",
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "check": check,
        "status": status,
        "time": "" if time is None else float(time),
        "group": group,
        "boundary_index": boundary_index,
        "node_id": node_id,
        "target_set": target_set,
        "value_key": value_key,
        "value": value,
        "limit": limit,
        "excess": excess,
        "message": message,
        "details": dict(details or {}),
    }


def _write_diagnostics_csv(path: Path, diagnostics: Sequence[Mapping[str, Any]]) -> None:
    fields = ["check", "status", "time", "group", "boundary_index", "node_id", "target_set", "value_key", "value", "limit", "excess", "message", "details"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in diagnostics:
            payload = {key: row.get(key, "") for key in fields}
            payload["details"] = json.dumps(payload.get("details", {}), ensure_ascii=False, default=str)
            writer.writerow(payload)


def _diagnostics_html(payload: Mapping[str, Any]) -> str:
    diagnostics = payload.get("diagnostics", [])
    rows = []
    for row in diagnostics:
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('check', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            f"<td>{html.escape(str(row.get('time', '')))}</td>"
            f"<td>{html.escape(str(row.get('group', '')))}</td>"
            f"<td>{html.escape(str(row.get('target_set', '')))}</td>"
            f"<td>{html.escape(str(row.get('node_id', '')))}</td>"
            f"<td>{html.escape(str(row.get('message', '')))}</td>"
            "</tr>"
        )
    body = "\n".join(rows) or "<tr><td colspan=\"7\">No diagnostics.</td></tr>"
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D Boundary Diagnostics</title>"
        f"<style>{report_css()}</style></head><body>"
        "<h1>VGFlow 2D Boundary Diagnostics</h1>"
        "<table><thead><tr><th>check</th><th>status</th><th>time</th><th>group</th><th>target</th><th>node</th><th>message</th></tr></thead>"
        f"<tbody>{body}</tbody></table></body></html>"
    )
