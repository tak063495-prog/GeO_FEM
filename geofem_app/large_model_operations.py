"""Large-model operation indexes and response-time summaries.

The GUI can use these artifacts to avoid full-table or full-scene scans when a
model grows large.  The writer intentionally produces compact JSON summaries
plus complete CSV indexes so the same data can be checked by CLI quality gates.
"""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import numpy as np

from .fem2d_types import Mesh2D, SolveResult2D


DEFAULT_RESULT_PAGE_SIZE = 2000
DEFAULT_DETAIL_LIMIT = 6000
DEFAULT_VECTOR_LIMIT = 2500


def build_large_model_operation_profile(
    result: SolveResult2D,
    *,
    result_page_size: int = DEFAULT_RESULT_PAGE_SIZE,
    detail_limit: int | None = None,
    vector_limit: int | None = None,
) -> dict[str, Any]:
    """Build a compact profile for search, LOD, partial selection, and tables."""

    t0 = perf_counter()
    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    display_cfg = _mapping(cfg.get("display", cfg.get("gui", {})))
    detail = _positive_int(detail_limit, display_cfg.get("detail_limit"), DEFAULT_DETAIL_LIMIT)
    vectors = _positive_int(vector_limit, display_cfg.get("vector_limit"), DEFAULT_VECTOR_LIMIT)

    nodes = _node_index_rows(result.mesh)
    t_nodes = perf_counter()
    elements = _element_index_rows(result.mesh)
    t_elements = perf_counter()
    node_bins = _bin_summary(nodes, "node_id")
    element_bins = _bin_summary(elements, "element_id")
    partial = _partial_selection_summary(result.mesh)
    t_partial = perf_counter()
    table_virtualization = _result_table_virtualization(result, result_page_size=result_page_size)
    t_tables = perf_counter()
    lod = _lod_policy(len(result.mesh.node_ids), len(result.mesh.elements), detail_limit=detail, vector_limit=vectors)

    timings = {
        "node_index_build_ms": _ms(t_nodes - t0),
        "element_index_build_ms": _ms(t_elements - t_nodes),
        "partial_selection_probe_ms": _ms(t_partial - t_elements),
        "result_table_summary_ms": _ms(t_tables - t_partial),
        "total_profile_build_ms": _ms(t_tables - t0),
    }
    budgets = {
        "node_index_build_ms": _operation_budget_ms(len(nodes)),
        "element_index_build_ms": _operation_budget_ms(len(elements)),
        "partial_selection_probe_ms": 120.0,
        "result_table_summary_ms": _operation_budget_ms(sum(int(row["row_count"]) for row in table_virtualization["tables"])),
    }
    timing_rows = [
        {
            "operation": key,
            "elapsed_ms": value,
            "budget_ms": budgets.get(key, ""),
            "status": "passed" if not isinstance(budgets.get(key), float) or value <= float(budgets[key]) else "warning",
        }
        for key, value in timings.items()
    ]
    return {
        "schema": "geofem.large_model_operations.v1",
        "node_count": len(nodes),
        "element_count": len(elements),
        "output_dir": str(result.output_dir),
        "features": [
            "node_search_index",
            "element_search_index",
            "display_lod_policy",
            "partial_selection_plan",
            "result_table_virtualization",
            "response_time_measurements",
        ],
        "node_search": {
            "index_file": "large_model_node_index.csv",
            "id_prefix_index": _prefix_index([row["node_id"] for row in nodes]),
            "spatial_bins": node_bins,
        },
        "element_search": {
            "index_file": "large_model_element_index.csv",
            "id_prefix_index": _prefix_index([row["element_id"] for row in elements]),
            "material_index": _group_count(elements, "material"),
            "type_index": _group_count(elements, "type"),
            "spatial_bins": element_bins,
        },
        "display_lod": lod,
        "partial_selection": partial,
        "result_table_virtualization": table_virtualization,
        "response_time": {
            "timings": timings,
            "budgets": budgets,
            "rows": timing_rows,
            "passed": all(row["status"] == "passed" for row in timing_rows),
        },
    }


def write_large_model_operation_artifacts(
    result: SolveResult2D,
    output_dir: str | Path | None = None,
    *,
    result_page_size: int = DEFAULT_RESULT_PAGE_SIZE,
) -> dict[str, str]:
    """Write large-model operation artifacts beside solver outputs."""

    out = Path(output_dir) if output_dir is not None else result.output_dir
    out.mkdir(parents=True, exist_ok=True)
    profile = build_large_model_operation_profile(result, result_page_size=result_page_size)
    nodes = _node_index_rows(result.mesh)
    elements = _element_index_rows(result.mesh)

    json_path = out / "large_model_operations.json"
    csv_path = out / "large_model_operations.csv"
    html_path = out / "large_model_operations.html"
    node_path = out / "large_model_node_index.csv"
    element_path = out / "large_model_element_index.csv"
    json_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_operation_csv(profile, csv_path)
    _write_rows_csv(nodes, node_path)
    _write_rows_csv(elements, element_path)
    html_path.write_text(_large_model_html(profile), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "html": str(html_path),
        "node_index_csv": str(node_path),
        "element_index_csv": str(element_path),
    }


def query_nodes_by_bbox(mesh: Mesh2D, xmin: float, ymin: float, xmax: float, ymax: float) -> list[str]:
    """Return node ids inside a rectangular selection box using a vector mask."""

    if not mesh.node_ids:
        return []
    x0, x1 = sorted((float(xmin), float(xmax)))
    y0, y1 = sorted((float(ymin), float(ymax)))
    coords = np.asarray(mesh.coords, dtype=float)
    mask = (coords[:, 0] >= x0) & (coords[:, 0] <= x1) & (coords[:, 1] >= y0) & (coords[:, 1] <= y1)
    indices = np.flatnonzero(mask)
    return [mesh.node_ids[int(index)] for index in indices]


def query_elements_by_bbox(mesh: Mesh2D, xmin: float, ymin: float, xmax: float, ymax: float) -> list[str]:
    """Return element ids whose bounding boxes intersect a rectangular box."""

    if not mesh.elements:
        return []
    x0, x1 = sorted((float(xmin), float(xmax)))
    y0, y1 = sorted((float(ymin), float(ymax)))
    rows = _element_index_rows(mesh)
    return [
        str(row["element_id"])
        for row in rows
        if float(row["xmax"]) >= x0 and float(row["xmin"]) <= x1 and float(row["ymax"]) >= y0 and float(row["ymin"]) <= y1
    ]


def _node_index_rows(mesh: Mesh2D) -> list[dict[str, Any]]:
    bounds = _mesh_bounds(mesh)
    rows: list[dict[str, Any]] = []
    set_membership = _membership_index(mesh.node_sets)
    for index, nid in enumerate(mesh.node_ids):
        x = float(mesh.coords[index, 0])
        y = float(mesh.coords[index, 1])
        rows.append(
            {
                "node_id": str(nid),
                "index": index,
                "x": x,
                "y": y,
                "bin": _bin_key(x, y, bounds),
                "sets": "|".join(set_membership.get(str(nid), [])),
            }
        )
    return rows


def _element_index_rows(mesh: Mesh2D) -> list[dict[str, Any]]:
    node_index = mesh.node_index
    bounds = _mesh_bounds(mesh)
    set_membership = _membership_index(mesh.element_sets)
    rows: list[dict[str, Any]] = []
    for index, element in enumerate(mesh.elements):
        coords = np.asarray([mesh.coords[node_index[nid]] for nid in element.nodes if nid in node_index], dtype=float)
        if coords.size == 0:
            xmin = xmax = ymin = ymax = cx = cy = 0.0
        else:
            xmin = float(np.min(coords[:, 0]))
            xmax = float(np.max(coords[:, 0]))
            ymin = float(np.min(coords[:, 1]))
            ymax = float(np.max(coords[:, 1]))
            cx = float(np.mean(coords[:, 0]))
            cy = float(np.mean(coords[:, 1]))
        rows.append(
            {
                "element_id": str(element.id),
                "index": index,
                "type": str(element.type),
                "material": str(element.material),
                "node_count": len(element.nodes),
                "active": bool(element.active),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "centroid_x": cx,
                "centroid_y": cy,
                "bin": _bin_key(cx, cy, bounds),
                "sets": "|".join(set_membership.get(str(element.id), [])),
            }
        )
    return rows


def _partial_selection_summary(mesh: Mesh2D) -> dict[str, Any]:
    bounds = _mesh_bounds(mesh)
    xmid = 0.5 * (bounds["xmin"] + bounds["xmax"])
    ymid = 0.5 * (bounds["ymin"] + bounds["ymax"])
    qx = 0.25 * (bounds["xmax"] - bounds["xmin"])
    qy = 0.25 * (bounds["ymax"] - bounds["ymin"])
    probes = [
        ("left_half", bounds["xmin"], bounds["ymin"], xmid, bounds["ymax"]),
        ("center_window", xmid - qx, ymid - qy, xmid + qx, ymid + qy),
    ]
    rows = []
    for name, xmin, ymin, xmax, ymax in probes:
        t0 = perf_counter()
        nodes = query_nodes_by_bbox(mesh, xmin, ymin, xmax, ymax)
        t1 = perf_counter()
        elements = query_elements_by_bbox(mesh, xmin, ymin, xmax, ymax)
        t2 = perf_counter()
        rows.append(
            {
                "name": name,
                "bbox": [xmin, ymin, xmax, ymax],
                "node_count": len(nodes),
                "element_count": len(elements),
                "node_query_ms": _ms(t1 - t0),
                "element_query_ms": _ms(t2 - t1),
                "node_preview": nodes[:12],
                "element_preview": elements[:12],
            }
        )
    return {
        "strategy": "bbox_vector_node_mask_and_element_bbox_intersection",
        "supports": ["bbox", "node_set", "element_set", "material", "element_type"],
        "probes": rows,
    }


def _result_table_virtualization(result: SolveResult2D, *, result_page_size: int) -> dict[str, Any]:
    tables: list[dict[str, Any]] = []
    for stage in result.stages:
        stage_dir = stage.output_dir or result.output_dir
        for name in (
            "displacements",
            "reactions",
            "element_stress",
            "integration_point_stress",
            "pore_pressure",
            "dynamic_history",
            "riks_path",
            "liquefaction_history",
        ):
            path = stage_dir / f"{name}.csv"
            if not path.exists():
                continue
            summary = _csv_summary(path)
            tables.append(
                {
                    "stage": stage.name,
                    "table": name,
                    "path": _rel(path, result.output_dir),
                    "row_count": summary["row_count"],
                    "column_count": len(summary["headers"]),
                    "page_size": result_page_size,
                    "page_count": max(1, math.ceil(int(summary["row_count"]) / max(1, result_page_size))),
                    "numeric_fields": summary["numeric_fields"],
                    "minimums": summary["minimums"],
                    "maximums": summary["maximums"],
                }
            )
    return {
        "page_size": result_page_size,
        "table_count": len(tables),
        "tables": tables,
        "strategy": "csv_header_scan_plus_page_reads",
    }


def _csv_summary(path: Path) -> dict[str, Any]:
    row_count = 0
    headers: list[str] = []
    numeric_candidates: set[str] = set()
    minimums: dict[str, float] = {}
    maximums: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        numeric_candidates = set(headers)
        for row in reader:
            row_count += 1
            for header in headers:
                try:
                    value = float(row.get(header, ""))
                except (TypeError, ValueError):
                    numeric_candidates.discard(header)
                    continue
                minimums[header] = min(minimums.get(header, value), value)
                maximums[header] = max(maximums.get(header, value), value)
    numeric_fields = [header for header in headers if header in numeric_candidates]
    return {
        "headers": headers,
        "row_count": row_count,
        "numeric_fields": numeric_fields,
        "minimums": {key: minimums[key] for key in numeric_fields if key in minimums},
        "maximums": {key: maximums[key] for key in numeric_fields if key in maximums},
    }


def _lod_policy(node_count: int, element_count: int, *, detail_limit: int, vector_limit: int) -> dict[str, Any]:
    size = max(int(node_count), int(element_count))
    reduced = size > detail_limit
    if reduced:
        draw_ratio = max(0.02, min(1.0, detail_limit / max(size, 1)))
        vector_limit = min(vector_limit, max(100, detail_limit // 2))
        contour_levels = 8
        boundary_mode = "decimated"
    else:
        draw_ratio = 1.0
        contour_levels = 14
        boundary_mode = "full"
    return {
        "mode": "auto-reduced" if reduced else "full",
        "node_count": int(node_count),
        "element_count": int(element_count),
        "detail_limit": int(detail_limit),
        "vector_limit": int(vector_limit),
        "draw_ratio": draw_ratio,
        "draw_node_labels": not reduced,
        "draw_element_labels": not reduced,
        "element_boundary_mode": boundary_mode,
        "contour_level_count": contour_levels,
    }


def _mesh_bounds(mesh: Mesh2D) -> dict[str, float]:
    if len(mesh.coords) == 0:
        return {"xmin": 0.0, "ymin": 0.0, "xmax": 1.0, "ymax": 1.0}
    xmin = float(np.min(mesh.coords[:, 0]))
    xmax = float(np.max(mesh.coords[:, 0]))
    ymin = float(np.min(mesh.coords[:, 1]))
    ymax = float(np.max(mesh.coords[:, 1]))
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def _bin_key(x: float, y: float, bounds: Mapping[str, float], *, divisions: int = 32) -> str:
    ix = int((float(x) - float(bounds["xmin"])) / max(float(bounds["xmax"]) - float(bounds["xmin"]), 1.0e-12) * divisions)
    iy = int((float(y) - float(bounds["ymin"])) / max(float(bounds["ymax"]) - float(bounds["ymin"]), 1.0e-12) * divisions)
    ix = max(0, min(divisions - 1, ix))
    iy = max(0, min(divisions - 1, iy))
    return f"{ix}:{iy}"


def _bin_summary(rows: list[Mapping[str, Any]], id_field: str) -> dict[str, Any]:
    bins: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("bin", ""))
        item = bins.setdefault(key, {"count": 0, "preview": []})
        item["count"] += 1
        if len(item["preview"]) < 8:
            item["preview"].append(str(row.get(id_field, "")))
    return {"bin_count": len(bins), "bins": bins}


def _prefix_index(values: list[str], *, prefix_len: int = 3, preview_limit: int = 8) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for value in values:
        key = str(value)[:prefix_len].lower()
        group = groups.setdefault(key, {"count": 0, "preview": []})
        group["count"] += 1
        if len(group["preview"]) < preview_limit:
            group["preview"].append(str(value))
    return {"prefix_length": prefix_len, "group_count": len(groups), "groups": groups}


def _group_count(rows: list[Mapping[str, Any]], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = str(row.get(field, ""))
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _membership_index(sets: Mapping[str, Any]) -> dict[str, list[str]]:
    membership: dict[str, list[str]] = {}
    for set_name, values in sets.items():
        if not isinstance(values, (list, tuple, set)):
            continue
        for value in values:
            membership.setdefault(str(value), []).append(str(set_name))
    for names in membership.values():
        names.sort()
    return membership


def _write_operation_csv(profile: Mapping[str, Any], path: Path) -> None:
    fields = ["area", "metric", "value", "unit", "status"]
    rows = [
        ("model", "node_count", profile.get("node_count", ""), "count", ""),
        ("model", "element_count", profile.get("element_count", ""), "count", ""),
        ("lod", "mode", _mapping(profile.get("display_lod")).get("mode", ""), "", ""),
        ("lod", "draw_ratio", _mapping(profile.get("display_lod")).get("draw_ratio", ""), "ratio", ""),
        (
            "table",
            "table_count",
            _mapping(profile.get("result_table_virtualization")).get("table_count", ""),
            "count",
            "",
        ),
    ]
    response = _mapping(profile.get("response_time"))
    for row in response.get("rows", []) if isinstance(response.get("rows", []), list) else []:
        if isinstance(row, Mapping):
            rows.append(("response", str(row.get("operation", "")), row.get("elapsed_ms", ""), "ms", row.get("status", "")))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for area, metric, value, unit, status in rows:
            writer.writerow({"area": area, "metric": metric, "value": value, "unit": unit, "status": status})


def _write_rows_csv(rows: list[Mapping[str, Any]], path: Path) -> None:
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _large_model_html(profile: Mapping[str, Any]) -> str:
    response_rows = []
    response = _mapping(profile.get("response_time"))
    for row in response.get("rows", []) if isinstance(response.get("rows", []), list) else []:
        if not isinstance(row, Mapping):
            continue
        response_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('operation', '')))}</td>"
            f"<td>{html.escape(str(row.get('elapsed_ms', '')))}</td>"
            f"<td>{html.escape(str(row.get('budget_ms', '')))}</td>"
            f"<td>{html.escape(str(row.get('status', '')))}</td>"
            "</tr>"
        )
    table_rows = []
    tables = _mapping(profile.get("result_table_virtualization")).get("tables", [])
    for row in tables if isinstance(tables, list) else []:
        if not isinstance(row, Mapping):
            continue
        table_rows.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('stage', '')))}</td>"
            f"<td>{html.escape(str(row.get('table', '')))}</td>"
            f"<td>{html.escape(str(row.get('row_count', '')))}</td>"
            f"<td>{html.escape(str(row.get('page_count', '')))}</td>"
            f"<td>{html.escape(str(row.get('path', '')))}</td>"
            "</tr>"
        )
    lod = _mapping(profile.get("display_lod"))
    features = ", ".join(html.escape(str(item)) for item in profile.get("features", []) if item)
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM large-model operations</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:12px 0}}th,td{{border:1px solid #ccd2d8;padding:6px;text-align:left}}th{{background:#eef2f7}}</style></head>
<body>
<h1>大規模モデル操作プロファイル</h1>
<p>nodes={html.escape(str(profile.get('node_count', '')))}, elements={html.escape(str(profile.get('element_count', '')))}</p>
<p>features: {features}</p>
<h2>表示LOD</h2>
<table><tbody>
<tr><th>mode</th><td>{html.escape(str(lod.get('mode', '')))}</td></tr>
<tr><th>draw_ratio</th><td>{html.escape(str(lod.get('draw_ratio', '')))}</td></tr>
<tr><th>boundary</th><td>{html.escape(str(lod.get('element_boundary_mode', '')))}</td></tr>
</tbody></table>
<h2>応答時間</h2>
<table><thead><tr><th>operation</th><th>elapsed ms</th><th>budget ms</th><th>status</th></tr></thead><tbody>{''.join(response_rows)}</tbody></table>
<h2>結果表仮想化</h2>
<table><thead><tr><th>stage</th><th>table</th><th>rows</th><th>pages</th><th>path</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
</body></html>
"""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(*values: Any) -> int:
    for value in values:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 1


def _operation_budget_ms(size: int) -> float:
    return max(150.0, 15.0 * math.sqrt(max(1, int(size))))


def _ms(seconds: float) -> float:
    return round(max(0.0, float(seconds)) * 1000.0, 3)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


__all__ = [
    "DEFAULT_DETAIL_LIMIT",
    "DEFAULT_RESULT_PAGE_SIZE",
    "DEFAULT_VECTOR_LIMIT",
    "build_large_model_operation_profile",
    "query_elements_by_bbox",
    "query_nodes_by_bbox",
    "write_large_model_operation_artifacts",
]
