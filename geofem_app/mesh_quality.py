"""Mesh-quality evaluation and repair-candidate artifacts for GeoFEM 2D."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fem2d_elements import integration_points, strain_displacement_matrix
from .fem2d_types import Element2D, Mesh2D


@dataclass(frozen=True)
class MeshQualityThresholds:
    min_area: float = 1.0e-12
    min_angle_deg: float = 10.0
    max_aspect_ratio: float = 20.0
    max_skew: float = 0.85
    min_jacobian: float = 1.0e-12
    max_boundary_projection_error: float = 0.25


def thresholds_from_config(cfg: Mapping[str, Any] | None = None) -> MeshQualityThresholds:
    raw: Mapping[str, Any] = {}
    if isinstance(cfg, Mapping):
        checks = cfg.get("checks", {})
        if isinstance(checks, Mapping) and isinstance(checks.get("mesh_quality"), Mapping):
            raw = checks["mesh_quality"]  # type: ignore[assignment]
        mesh = cfg.get("mesh", {})
        if not raw and isinstance(mesh, Mapping) and isinstance(mesh.get("quality"), Mapping):
            raw = mesh["quality"]  # type: ignore[assignment]

    def value(name: str, default: float, *aliases: str) -> float:
        for key in (name, *aliases):
            if key not in raw:
                continue
            try:
                out = float(raw[key])
            except (TypeError, ValueError):
                return default
            return out if math.isfinite(out) else default
        return default

    return MeshQualityThresholds(
        min_area=max(value("min_area", 1.0e-12), 0.0),
        min_angle_deg=max(value("min_angle_deg", 10.0, "min_angle"), 0.0),
        max_aspect_ratio=max(value("max_aspect_ratio", 20.0, "max_aspect"), 1.0e-12),
        max_skew=max(value("max_skew", 0.85), 0.0),
        min_jacobian=max(value("min_jacobian", 1.0e-12, "min_det_j"), 0.0),
        max_boundary_projection_error=max(value("max_boundary_projection_error", 0.25, "max_boundary_fit_error"), 0.0),
    )


def evaluate_mesh_quality(mesh: Mesh2D, cfg: Mapping[str, Any] | None = None) -> dict[str, Any]:
    thresholds = thresholds_from_config(cfg)
    metrics = [element_quality_metrics(mesh, element) for element in mesh.elements]
    violations = [_violation_record(item, thresholds) for item in metrics]
    violations = [item for item in violations if item is not None]
    repair_candidates = build_repair_candidates(mesh, violations)
    summary = _quality_summary(mesh, metrics, violations, thresholds)
    return {
        "schema": "geofem.mesh_quality.v1",
        "passed": not any(item["severity"] == "ERROR" for item in violations),
        "summary": summary,
        "thresholds": thresholds.__dict__,
        "metrics": metrics,
        "violations": violations,
        "repair_candidates": repair_candidates,
    }


def element_quality_metrics(mesh: Mesh2D, element: Element2D) -> dict[str, Any]:
    points = _corner_points(mesh, element)
    signed_area = _polygon_area(points)
    area = abs(signed_area)
    edge_lengths = [_edge_length(a, b) for a, b in zip(points, points[1:] + points[:1])]
    positive = [length for length in edge_lengths if length > 1.0e-30]
    min_edge = min(positive, default=0.0)
    max_edge = max(positive, default=0.0)
    aspect = max_edge / min_edge if min_edge > 0.0 else math.inf
    angles = _polygon_angles(points)
    min_angle = min(angles, default=0.0)
    max_angle = max(angles, default=0.0)
    skew = max((abs(angle - 90.0) / 90.0 for angle in angles), default=0.0) if len(points) == 4 else 0.0
    perimeter = sum(edge_lengths)
    compactness = 0.0 if perimeter <= 0.0 else 4.0 * math.pi * area / (perimeter * perimeter)
    distortion = 0.0 if max_edge <= 0.0 else 1.0 - min_edge / max_edge
    jacobians = _element_jacobians(mesh, element)
    boundary_error = _mid_edge_projection_error(mesh, element)
    centroid = [sum(p[0] for p in points) / max(len(points), 1), sum(p[1] for p in points) / max(len(points), 1)]
    return {
        "element": str(element.id),
        "element_id": str(element.id),
        "type": element.type,
        "material": element.material,
        "active": bool(element.active),
        "area": area,
        "signed_area": signed_area,
        "orientation": "ccw" if signed_area >= 0.0 else "cw",
        "min_angle_deg": min_angle,
        "min_angle": min_angle,
        "max_angle_deg": max_angle,
        "aspect_ratio": aspect,
        "aspect": aspect,
        "skew": skew,
        "distortion": distortion,
        "compactness": compactness,
        "min_edge_length": min_edge,
        "max_edge_length": max_edge,
        "jacobian_min": min(jacobians, default=math.nan),
        "jacobian_max": max(jacobians, default=math.nan),
        "boundary_projection_error": boundary_error,
        "centroid_x": centroid[0],
        "centroid_y": centroid[1],
    }


def build_repair_candidates(mesh: Mesh2D, violations: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    element_lookup = {str(element.id): element for element in mesh.elements}
    candidates: list[dict[str, Any]] = []
    for item in violations:
        element_id = str(item.get("element", item.get("element_id", "")))
        element = element_lookup.get(element_id)
        if element is None:
            continue
        center, radius, target_size = _repair_geometry(mesh, element)
        priority = "high" if item.get("severity") == "ERROR" else "medium"
        reasons = str(item.get("reason", ""))
        local_refinement = {
            "id": f"repair_{element_id}_local_refinement",
            "element": element_id,
            "method": "local_refinement",
            "priority": priority,
            "reason": reasons,
            "expected_effect": "adds local refinement and a size-map entry around the bad element",
            "config_patch": {
                "mesh.refinements[]": {"id": f"repair_{element_id}", "center": center, "radius": radius, "factor": 2.0, "source": "mesh_quality"},
                "mesh.size_map[]": {"id": f"size_repair_{element_id}", "center": center, "radius": radius, "target_size": target_size, "grading": 1.35, "source": "mesh_quality"},
                "mesh.quality_repairs[]": {"id": f"repair_{element_id}", "element": element_id, "action": "local_refinement", "center": center, "radius": radius, "target_size": target_size},
            },
        }
        candidates.append(local_refinement)
        if any(token in reasons for token in ("min_angle", "aspect", "skew")):
            candidates.append(
                {
                    "id": f"repair_{element_id}_smoothing",
                    "element": element_id,
                    "method": "laplacian_smoothing",
                    "priority": "medium",
                    "reason": reasons,
                    "expected_effect": "moves non-boundary neighboring nodes while preserving boundary nodes",
                    "config_patch": {
                        "mesh.quality_repairs[]": {"id": f"smooth_{element_id}", "element": element_id, "action": "laplacian_smoothing", "iterations": 5}
                    },
                }
            )
        if float(item.get("boundary_projection_error", 0.0) or 0.0) > 0.0:
            candidates.append(
                {
                    "id": f"repair_{element_id}_boundary_reprojection",
                    "element": element_id,
                    "method": "boundary_reprojection",
                    "priority": "medium",
                    "reason": reasons,
                    "expected_effect": "projects midside boundary nodes back to the local chord or CAD curve",
                    "config_patch": {
                        "mesh.quality_repairs[]": {"id": f"reproject_{element_id}", "element": element_id, "action": "boundary_reprojection"}
                    },
                }
            )
        if float(item.get("aspect_ratio", item.get("aspect", 0.0)) or 0.0) > 10.0:
            split = _split_line_candidate(mesh, element, target_size)
            if split is not None:
                candidates.append(
                    {
                        "id": f"repair_{element_id}_split_line",
                        "element": element_id,
                        "method": "split_line",
                        "priority": "medium",
                        "reason": reasons,
                        "expected_effect": "adds a division line across the longest direction of the element patch",
                        "config_patch": {"mesh.split_lines[]": split},
                    }
                )
    return candidates


def apply_repair_candidates_to_config(
    cfg: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
    *,
    candidate_ids: set[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    updated = json.loads(json.dumps(dict(cfg), ensure_ascii=False, default=str))
    mesh = updated.setdefault("mesh", {})
    if not isinstance(mesh, dict):
        mesh = {}
        updated["mesh"] = mesh
    applied: list[str] = []
    for candidate in candidates:
        cid = str(candidate.get("id", ""))
        if candidate_ids is not None and cid not in candidate_ids:
            continue
        patch = candidate.get("config_patch", {})
        if not isinstance(patch, Mapping):
            continue
        for target, value in patch.items():
            key = str(target)
            if key == "mesh.refinements[]":
                mesh.setdefault("refinements", []).append(value)
            elif key == "mesh.size_map[]":
                mesh.setdefault("size_map", []).append(value)
            elif key == "mesh.quality_repairs[]":
                mesh.setdefault("quality_repairs", []).append(value)
            elif key == "mesh.split_lines[]":
                mesh.setdefault("split_lines", []).append(value)
        if cid:
            applied.append(cid)
    return updated, applied


def write_mesh_quality_report(mesh: Mesh2D, output_dir: str | Path, cfg: Mapping[str, Any] | None = None) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = evaluate_mesh_quality(mesh, cfg)
    json_path = out / "mesh_quality.json"
    csv_path = out / "mesh_quality.csv"
    repairs_path = out / "mesh_quality_repairs.csv"
    html_path = out / "mesh_quality.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    metric_fields = [
        "element",
        "type",
        "material",
        "area",
        "min_angle_deg",
        "aspect_ratio",
        "skew",
        "jacobian_min",
        "boundary_projection_error",
        "distortion",
        "active",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        for row in report["metrics"]:
            writer.writerow({field: row.get(field, "") for field in metric_fields})
    repair_fields = ["id", "element", "method", "priority", "reason", "expected_effect"]
    with repairs_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=repair_fields)
        writer.writeheader()
        for row in report["repair_candidates"]:
            writer.writerow({field: row.get(field, "") for field in repair_fields})
    html_path.write_text(_mesh_quality_html(report), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "repairs_csv": str(repairs_path), "html": str(html_path)}


def _violation_record(item: Mapping[str, Any], thresholds: MeshQualityThresholds) -> dict[str, Any] | None:
    reasons: list[str] = []
    severity = "WARN"
    if float(item.get("area", 0.0) or 0.0) < thresholds.min_area:
        reasons.append(f"area<{thresholds.min_area:g}")
        severity = "ERROR"
    if float(item.get("min_angle_deg", 0.0) or 0.0) < thresholds.min_angle_deg:
        reasons.append(f"min_angle<{thresholds.min_angle_deg:g}")
    if float(item.get("aspect_ratio", 0.0) or 0.0) > thresholds.max_aspect_ratio:
        reasons.append(f"aspect>{thresholds.max_aspect_ratio:g}")
    if str(item.get("type", "")).startswith("QUAD") and float(item.get("skew", 0.0) or 0.0) > thresholds.max_skew:
        reasons.append(f"skew>{thresholds.max_skew:g}")
    jac_min = float(item.get("jacobian_min", math.nan) or math.nan)
    if math.isfinite(jac_min) and jac_min < thresholds.min_jacobian:
        reasons.append(f"jacobian<{thresholds.min_jacobian:g}")
        severity = "ERROR"
    boundary_error = float(item.get("boundary_projection_error", 0.0) or 0.0)
    if boundary_error > thresholds.max_boundary_projection_error:
        reasons.append(f"boundary_fit>{thresholds.max_boundary_projection_error:g}")
    if not reasons:
        return None
    out = dict(item)
    out["severity"] = severity
    out["reason"] = ", ".join(reasons)
    out["repair"] = _primary_repair(reasons)
    return out


def _quality_summary(
    mesh: Mesh2D,
    metrics: list[Mapping[str, Any]],
    violations: list[Mapping[str, Any]],
    thresholds: MeshQualityThresholds,
) -> dict[str, Any]:
    if metrics:
        min_area = min(float(item.get("area", 0.0) or 0.0) for item in metrics)
        min_angle = min(float(item.get("min_angle_deg", 0.0) or 0.0) for item in metrics)
        max_aspect = max(float(item.get("aspect_ratio", 0.0) or 0.0) for item in metrics)
        max_skew = max(float(item.get("skew", 0.0) or 0.0) for item in metrics)
        min_jac = min(float(item.get("jacobian_min", math.inf) or math.inf) for item in metrics)
        max_boundary_error = max(float(item.get("boundary_projection_error", 0.0) or 0.0) for item in metrics)
    else:
        min_area = min_angle = max_aspect = max_skew = min_jac = max_boundary_error = 0.0
    return {
        "node_count": len(mesh.node_ids),
        "element_count": len(mesh.elements),
        "violation_count": len(violations),
        "error_count": sum(1 for item in violations if item.get("severity") == "ERROR"),
        "warning_count": sum(1 for item in violations if item.get("severity") == "WARN"),
        "min_area": min_area,
        "min_angle_deg": min_angle,
        "max_aspect_ratio": max_aspect,
        "max_skew": max_skew,
        "min_jacobian": min_jac,
        "max_boundary_projection_error": max_boundary_error,
        "thresholds": thresholds.__dict__,
    }


def _primary_repair(reasons: list[str]) -> str:
    if any(reason.startswith("jacobian") or reason.startswith("area") for reason in reasons):
        return "local_remesh"
    if any(reason.startswith("boundary_fit") for reason in reasons):
        return "boundary_reprojection"
    if any(reason.startswith("aspect") for reason in reasons):
        return "split_line_or_local_refinement"
    return "laplacian_smoothing"


def _corner_points(mesh: Mesh2D, element: Element2D) -> list[tuple[float, float]]:
    corner_count = 3 if element.type.startswith("TRI") else 4
    points = []
    for nid in element.nodes[:corner_count]:
        idx = mesh.node_index[nid]
        points.append((float(mesh.coords[idx, 0]), float(mesh.coords[idx, 1])))
    return points


def _polygon_area(points: list[tuple[float, float]]) -> float:
    return 0.5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))


def _edge_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def _polygon_angles(points: list[tuple[float, float]]) -> list[float]:
    angles: list[float] = []
    for i, p in enumerate(points):
        prev = points[i - 1]
        nxt = points[(i + 1) % len(points)]
        v1 = (prev[0] - p[0], prev[1] - p[1])
        v2 = (nxt[0] - p[0], nxt[1] - p[1])
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 <= 1.0e-30 or n2 <= 1.0e-30:
            angles.append(0.0)
            continue
        cosv = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        angles.append(math.degrees(math.acos(cosv)))
    return angles


def _element_jacobians(mesh: Mesh2D, element: Element2D) -> list[float]:
    try:
        coords = np.array([mesh.coords[mesh.node_index[nid]] for nid in element.nodes], dtype=float)
        out = []
        for gp in integration_points(element.type, "FULL"):
            _B4, detj, _N = strain_displacement_matrix(element.type, coords, gp)
            out.append(float(detj))
        return out
    except Exception:
        return [math.nan]


def _mid_edge_projection_error(mesh: Mesh2D, element: Element2D) -> float:
    edge_specs: list[tuple[int, int, int]]
    if element.type == "TRI6":
        edge_specs = [(0, 1, 3), (1, 2, 4), (2, 0, 5)]
    elif element.type == "QUAD8":
        edge_specs = [(0, 1, 4), (1, 2, 5), (2, 3, 6), (3, 0, 7)]
    else:
        return 0.0
    errors = []
    for i0, i1, imid in edge_specs:
        if imid >= len(element.nodes):
            continue
        p0 = mesh.coords[mesh.node_index[element.nodes[i0]]]
        p1 = mesh.coords[mesh.node_index[element.nodes[i1]]]
        pm = mesh.coords[mesh.node_index[element.nodes[imid]]]
        length = float(np.linalg.norm(p1 - p0))
        if length <= 1.0e-30:
            continue
        edge = p1 - p0
        offset = pm - p0
        distance = abs(float(edge[0] * offset[1] - edge[1] * offset[0])) / length
        errors.append(distance / length)
    return max(errors, default=0.0)


def _repair_geometry(mesh: Mesh2D, element: Element2D) -> tuple[list[float], float, float]:
    points = _corner_points(mesh, element)
    cx = sum(x for x, _y in points) / max(len(points), 1)
    cy = sum(y for _x, y in points) / max(len(points), 1)
    edge_lengths = [_edge_length(a, b) for a, b in zip(points, points[1:] + points[:1])]
    max_edge = max(edge_lengths, default=1.0)
    min_edge = min([value for value in edge_lengths if value > 1.0e-30], default=max_edge)
    radius = max(max_edge * 1.25, 1.0e-9)
    target_size = max(min_edge * 0.5, 1.0e-9)
    return [float(cx), float(cy)], float(radius), float(target_size)


def _split_line_candidate(mesh: Mesh2D, element: Element2D, target_size: float) -> dict[str, Any] | None:
    points = _corner_points(mesh, element)
    if len(points) < 3:
        return None
    edges = [(i, _edge_length(points[i], points[(i + 1) % len(points)])) for i in range(len(points))]
    if not edges:
        return None
    idx, _length = max(edges, key=lambda item: item[1])
    a = points[idx]
    b = points[(idx + 1) % len(points)]
    opposite = points[(idx + 2) % len(points)]
    mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)
    return {
        "id": f"split_quality_{element.id}",
        "type": "split_line",
        "start": [float(mid[0]), float(mid[1])],
        "end": [float(opposite[0]), float(opposite[1])],
        "target_size": float(target_size),
        "locked": True,
        "source": "mesh_quality",
    }


def _mesh_quality_html(report: Mapping[str, Any]) -> str:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), Mapping) else {}
    rows = []
    for item in report.get("violations", []):
        if not isinstance(item, Mapping):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('element', '')))}</td>"
            f"<td>{html.escape(str(item.get('severity', '')))}</td>"
            f"<td>{html.escape(str(item.get('reason', '')))}</td>"
            f"<td>{_fmt(item.get('area', ''))}</td>"
            f"<td>{_fmt(item.get('min_angle_deg', ''))}</td>"
            f"<td>{_fmt(item.get('aspect_ratio', ''))}</td>"
            f"<td>{_fmt(item.get('jacobian_min', ''))}</td>"
            f"<td>{html.escape(str(item.get('repair', '')))}</td>"
            "</tr>"
        )
    repairs = []
    for item in report.get("repair_candidates", []):
        if not isinstance(item, Mapping):
            continue
        repairs.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('id', '')))}</td>"
            f"<td>{html.escape(str(item.get('method', '')))}</td>"
            f"<td>{html.escape(str(item.get('priority', '')))}</td>"
            f"<td>{html.escape(str(item.get('expected_effect', '')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM mesh quality</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>Mesh Quality</h1>
<p>elements={int(summary.get('element_count', 0) or 0)}, violations={int(summary.get('violation_count', 0) or 0)}, errors={int(summary.get('error_count', 0) or 0)}</p>
<h2>Violations</h2>
<table><thead><tr><th>element</th><th>severity</th><th>reason</th><th>area</th><th>min angle</th><th>aspect</th><th>jacobian</th><th>repair</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Repair Candidates</h2>
<table><thead><tr><th>id</th><th>method</th><th>priority</th><th>expected effect</th></tr></thead><tbody>{''.join(repairs)}</tbody></table>
</body></html>
"""


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return html.escape(str(value))
    return html.escape(f"{number:.8g}")


__all__ = [
    "MeshQualityThresholds",
    "thresholds_from_config",
    "evaluate_mesh_quality",
    "element_quality_metrics",
    "build_repair_candidates",
    "apply_repair_candidates_to_config",
    "write_mesh_quality_report",
]
