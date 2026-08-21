"""Mesh-planning diagnostics for the VGFlow 2D public substitute."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fem2d_types import Mesh2D
from .fem2d_utils import _ensure_list
from .html_report_utils import html_escape, report_css, table
from .mesh_quality import evaluate_mesh_quality
from .vgflow2d_post import vgflow_element_post_fields
from .vgflow2d_pre import vgflow_pre_template_catalog


VGFLOW_EMBANKMENT_MESH_TEMPLATES: dict[str, dict[str, Any]] = {
    "river_embankment_standard": {
        "label": "River embankment standard semi-auto mesh",
        "mesh_mode": "semi_auto",
        "recommended_blocks": ["embankment_body", "foundation", "seepage_boundary_layer"],
        "segment_divisions": [
            {"name": "levee_crest", "division_count": 6},
            {"name": "upstream_slope", "division_count": 8},
            {"name": "downstream_slope", "division_count": 8},
            {"name": "foundation_base", "division_count": 12},
        ],
        "quality_targets": {"max_aspect_ratio": 8.0, "min_angle_deg": 25.0, "max_skew": 0.55},
    }
}


def vgflow_mesh_template_catalog() -> list[dict[str, Any]]:
    return [{"id": key, **value} for key, value in sorted(VGFLOW_EMBANKMENT_MESH_TEMPLATES.items())]


def write_vgflow_mesh_outputs(
    out: Path,
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    problem_type: str,
    seepage: Mapping[str, Any],
) -> dict[str, str]:
    paths = {
        "mesh_plan_json": str(out / "vgflow_mesh_plan.json"),
        "mesh_plan_csv": str(out / "vgflow_mesh_plan.csv"),
        "mesh_quality_json": str(out / "vgflow_mesh_quality.json"),
        "mesh_quality_csv": str(out / "vgflow_mesh_quality.csv"),
        "mesh_quality_html": str(out / "vgflow_mesh_quality.html"),
        "mesh_templates": str(out / "vgflow_mesh_templates.json"),
    }
    cfg = _mesh_cfg(seepage)
    quality = evaluate_mesh_quality(mesh, {"mesh": {"quality": cfg.get("quality", cfg.get("quality_targets", {}))}})
    plan = _mesh_plan(mesh, seepage, cfg, quality)
    hydraulic_recommendations = _hydraulic_refinement_recommendations(mesh, materials, steps, problem_type, cfg)
    payload = {
        "schema": "geofem.vgflow2d.mesh_generation.public_substitute.v1",
        "features": [
            "mesh_mode_plan",
            "line_division_plan",
            "river_embankment_semi_auto_template",
            "mesh_quality_diagnostics",
            "hydraulic_gradient_refinement_recommendations",
            "condition_reset_guidance",
        ],
        "plan": plan,
        "quality": quality,
        "hydraulic_refinement_recommendations": hydraulic_recommendations,
    }
    Path(paths["mesh_plan_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_plan_csv(Path(paths["mesh_plan_csv"]), plan, hydraulic_recommendations)
    Path(paths["mesh_quality_json"]).write_text(json.dumps(quality, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_quality_csv(Path(paths["mesh_quality_csv"]), quality)
    Path(paths["mesh_quality_html"]).write_text(_mesh_quality_html(payload), encoding="utf-8")
    Path(paths["mesh_templates"]).write_text(json.dumps({"templates": vgflow_mesh_template_catalog()}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return paths


def _mesh_cfg(seepage: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("mesh_generation", "mesh_plan", "vgflow_mesh", "mesh"):
        value = seepage.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _mesh_plan(mesh: Mesh2D, seepage: Mapping[str, Any], cfg: Mapping[str, Any], quality: Mapping[str, Any]) -> dict[str, Any]:
    mode = _mesh_mode(seepage, cfg)
    element_types = _element_type_counts(mesh)
    return {
        "mesh_mode": mode,
        "mesh_mode_definition": _mesh_mode_definition(mode),
        "element_type_counts": element_types,
        "line_divisions": _line_divisions(cfg),
        "selected_templates": _selected_templates(cfg),
        "condition_reset_guidance": bool(cfg.get("condition_reset_guidance", cfg.get("mesh_regeneration_clears_conditions", True))),
        "quality_passed": bool(quality.get("passed", False)),
        "quality_violation_count": int(quality.get("summary", {}).get("violation_count", len(quality.get("violations", []))) or 0),
    }


def _mesh_mode(seepage: Mapping[str, Any], cfg: Mapping[str, Any]) -> str:
    raw = cfg.get("mesh_mode", seepage.get("mesh_mode", seepage.get("vgflow_mesh_mode", "auto_mixed")))
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "auto": "auto_mixed",
        "mixed": "auto_mixed",
        "quad": "quadrilateral_only",
        "quadrilateral": "quadrilateral_only",
        "tri": "triangular_only",
        "triangular": "triangular_only",
        "semi": "semi_auto",
        "semiauto": "semi_auto",
    }
    return aliases.get(key, key if key in {row["id"] for row in vgflow_pre_template_catalog()} else "auto_mixed")


def _mesh_mode_definition(mode: str) -> dict[str, Any]:
    for row in vgflow_pre_template_catalog():
        if row["id"] == mode:
            return row
    return {"id": mode, "allowed_elements": [], "notes": "unknown public substitute mesh mode"}


def _line_divisions(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = cfg.get("line_divisions", cfg.get("segment_divisions", cfg.get("division_lines", [])))
    out: list[dict[str, Any]] = []
    for index, item in enumerate(_ensure_list(raw), start=1):
        if not isinstance(item, Mapping):
            continue
        division_count = item.get("division_count", item.get("divisions", item.get("n")))
        division_width = item.get("division_width", item.get("width", item.get("target_size")))
        out.append(
            {
                "name": str(item.get("name", item.get("id", f"line_{index}"))),
                "division_count": int(division_count) if division_count not in (None, "") else None,
                "division_width": float(division_width) if division_width not in (None, "") else None,
                "grading": float(item.get("grading", 1.0) or 1.0),
            }
        )
    return out


def _selected_templates(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    requested = [str(item) for item in _ensure_list(cfg.get("templates", cfg.get("semi_auto_templates", [])))]
    if bool(cfg.get("river_embankment_template", False)) and "river_embankment_standard" not in requested:
        requested.append("river_embankment_standard")
    return [{"id": key, **VGFLOW_EMBANKMENT_MESH_TEMPLATES[key]} for key in requested if key in VGFLOW_EMBANKMENT_MESH_TEMPLATES]


def _hydraulic_refinement_recommendations(mesh: Mesh2D, materials: Mapping[str, Any], steps: Sequence[Any], problem_type: str, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not steps:
        return []
    threshold = float(cfg.get("gradient_refinement_threshold", cfg.get("local_gradient_threshold", 0.75)) or 0.75)
    max_items = int(cfg.get("max_hydraulic_refinement_items", 20) or 20)
    last_step = steps[-1]
    rows = vgflow_element_post_fields(mesh, materials, last_step, problem_type)
    hot = [row for row in rows if float(row.get("hydraulic_gradient_abs", 0.0) or 0.0) >= threshold]
    hot.sort(key=lambda row: float(row.get("hydraulic_gradient_abs", 0.0) or 0.0), reverse=True)
    recommendations: list[dict[str, Any]] = []
    for row in hot[:max_items]:
        element = next((item for item in mesh.elements if str(item.id) == str(row["element_id"])), None)
        target_size = _element_min_edge(mesh, element) * 0.5 if element is not None else 0.0
        recommendations.append(
            {
                "id": f"hydraulic_refine_{row['element_id']}",
                "element_id": str(row["element_id"]),
                "time": float(last_step.time),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "hydraulic_gradient_abs": float(row["hydraulic_gradient_abs"]),
                "threshold": threshold,
                "recommendation": "add local refinement or reduce segment division width around this element",
                "target_size": target_size,
            }
        )
    return recommendations


def _element_min_edge(mesh: Mesh2D, element: Any) -> float:
    if element is None:
        return 0.0
    corners = list(element.nodes[:4] if element.type.upper().startswith("QUAD") else element.nodes[:3])
    lengths = []
    for a, b in zip(corners, [*corners[1:], corners[0]]):
        pa = mesh.coords[mesh.node_index[a]]
        pb = mesh.coords[mesh.node_index[b]]
        lengths.append(float(np.linalg.norm(pa - pb)))
    return min([value for value in lengths if value > 0.0], default=0.0)


def _element_type_counts(mesh: Mesh2D) -> dict[str, int]:
    counts: dict[str, int] = {}
    for element in mesh.elements:
        counts[element.type] = counts.get(element.type, 0) + 1
    return counts


def _write_plan_csv(path: Path, plan: Mapping[str, Any], recommendations: Sequence[Mapping[str, Any]]) -> None:
    fields = ["kind", "id", "metric", "value", "detail"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"kind": "mesh_mode", "id": plan["mesh_mode"], "metric": "allowed_elements", "value": ",".join(plan["mesh_mode_definition"].get("allowed_elements", [])), "detail": plan["mesh_mode_definition"].get("notes", "")})
        for line in plan["line_divisions"]:
            writer.writerow({"kind": "line_division", "id": line["name"], "metric": "division", "value": line.get("division_count") or line.get("division_width") or "", "detail": json.dumps(line, ensure_ascii=False)})
        for template in plan["selected_templates"]:
            writer.writerow({"kind": "template", "id": template["id"], "metric": "mesh_mode", "value": template.get("mesh_mode", ""), "detail": template.get("label", "")})
        for item in recommendations:
            writer.writerow({"kind": "hydraulic_refinement", "id": item["id"], "metric": "hydraulic_gradient_abs", "value": item["hydraulic_gradient_abs"], "detail": item["recommendation"]})


def _write_quality_csv(path: Path, quality: Mapping[str, Any]) -> None:
    fields = ["element_id", "type", "area", "min_angle_deg", "aspect_ratio", "skew", "jacobian_min"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in quality.get("metrics", []):
            writer.writerow({key: row.get(key, "") for key in fields})


def _mesh_quality_html(payload: Mapping[str, Any]) -> str:
    plan = payload["plan"]
    recs = payload["hydraulic_refinement_recommendations"]
    rows = [[item["id"], item["element_id"], item["hydraulic_gradient_abs"], item["threshold"], item["target_size"], item["recommendation"]] for item in recs]
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D mesh generation diagnostics</title>"
        f"<style>{report_css()}</style></head><body>"
        "<h1>VGFlow 2D mesh generation diagnostics</h1>"
        f"<p>mesh mode: {html_escape(plan['mesh_mode'])}, quality passed: {html_escape(plan['quality_passed'])}</p>"
        + table(["id", "element", "gradient", "threshold", "target_size", "recommendation"], rows)
        + "</body></html>"
    )


__all__ = ["vgflow_mesh_template_catalog", "write_vgflow_mesh_outputs"]
