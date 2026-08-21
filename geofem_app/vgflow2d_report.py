"""Report bundle writer for the VGFlow 2D public substitute."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fem2d_types import Mesh2D
from .fem2d_utils import _ensure_list
from .html_report_utils import html_escape, kv_table, rel_link, report_css, table
from .pdf_writer import write_text_pdf


DEFAULT_REPORT_SECTIONS = (
    "model",
    "mesh",
    "analysis",
    "materials",
    "boundaries",
    "post_outputs",
    "time_history",
    "warnings",
)


def write_vgflow_report_bundle(
    out: Path,
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    problem_type: str,
    seepage: Mapping[str, Any],
    artifacts: Mapping[str, str],
    warnings: Sequence[str],
) -> dict[str, str]:
    paths = {
        "vgflow_report_data": str(out / "vgflow_report_data.json"),
        "vgflow_report_sections": str(out / "vgflow_report_sections.csv"),
        "vgflow_report_html": str(out / "vgflow_report.html"),
        "vgflow_report_pdf": str(out / "vgflow_report.pdf"),
        "vgflow_report_print_profile": str(out / "vgflow_report_public_ppf_profile.json"),
        "vgflow_report_print_profile_csv": str(out / "vgflow_report_public_ppf_profile.csv"),
        "vgflow_report_print_profile_html": str(out / "vgflow_report_public_ppf_profile.html"),
        "vgflow_report_manifest": str(out / "vgflow_report_manifest.json"),
    }
    report_cfg = _report_cfg(seepage)
    sections = _selected_sections(report_cfg)
    data = _report_data(mesh, materials, steps, problem_type, seepage, artifacts, warnings, report_cfg, sections)
    Path(paths["vgflow_report_data"]).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_sections_csv(Path(paths["vgflow_report_sections"]), data)
    Path(paths["vgflow_report_html"]).write_text(_report_html(data, out), encoding="utf-8")
    write_text_pdf(paths["vgflow_report_pdf"], _report_lines(data), title="VGFlow 2D 公開代替帳票")
    print_profile = _print_profile(data, report_cfg, artifacts)
    Path(paths["vgflow_report_print_profile"]).write_text(json.dumps(print_profile, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_print_profile_csv(Path(paths["vgflow_report_print_profile_csv"]), print_profile)
    Path(paths["vgflow_report_print_profile_html"]).write_text(_print_profile_html(print_profile), encoding="utf-8")
    manifest = {
        "schema": "geofem.vgflow2d.report.public_substitute.v1",
        "profile": "VGFlow 2D public-information based report substitute",
        "features": [
            "html_report",
            "direct_pdf_report",
            "manifest_backed_report",
            "pre_post_section_selection",
            "node_element_time_range_filters",
            "post_artifact_links",
            "public_ppf_print_profile_substitute",
        ],
        "selection": data["selection"],
        "artifacts": paths,
        "source_artifacts": dict(artifacts),
    }
    Path(paths["vgflow_report_manifest"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return paths


def _report_cfg(seepage: Mapping[str, Any]) -> Mapping[str, Any]:
    post = seepage.get("post", seepage.get("vgflow_post", {}))
    candidates = [seepage.get("report"), seepage.get("print"), seepage.get("vgflow_report")]
    if isinstance(post, Mapping):
        candidates.append(post.get("report"))
    for candidate in candidates:
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _selected_sections(report_cfg: Mapping[str, Any]) -> list[str]:
    requested = report_cfg.get("sections", report_cfg.get("include_sections"))
    if requested:
        selected = [str(item) for item in _ensure_list(requested) if str(item) in DEFAULT_REPORT_SECTIONS]
    else:
        selected = list(DEFAULT_REPORT_SECTIONS)
    excluded = {str(item) for item in _ensure_list(report_cfg.get("exclude_sections", []))}
    return [section for section in selected if section not in excluded]


def _report_data(
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    problem_type: str,
    seepage: Mapping[str, Any],
    artifacts: Mapping[str, str],
    warnings: Sequence[str],
    report_cfg: Mapping[str, Any],
    sections: Sequence[str],
) -> dict[str, Any]:
    node_ids = [str(item) for item in _ensure_list(report_cfg.get("node_ids", report_cfg.get("nodes", [])))]
    element_ids = [str(item) for item in _ensure_list(report_cfg.get("element_ids", report_cfg.get("elements", [])))]
    time_range = _time_range(report_cfg)
    filtered_steps = [step for step in steps if _time_in_range(float(step.time), time_range)]
    section_rows = {
        "model": _model_rows(mesh, problem_type),
        "mesh": _mesh_rows(mesh),
        "analysis": _analysis_rows(filtered_steps, problem_type, seepage),
        "materials": _material_rows(materials),
        "boundaries": _boundary_rows(seepage),
        "post_outputs": _artifact_rows(artifacts),
        "time_history": _history_selection_rows(node_ids, element_ids, time_range),
        "warnings": [["warning", warning] for warning in warnings],
    }
    return {
        "schema": "geofem.vgflow2d.report_data.v1",
        "selection": {
            "sections": list(sections),
            "node_ids": node_ids,
            "element_ids": element_ids,
            "time_range": time_range,
        },
        "summary": {
            "node_count": len(mesh.node_ids),
            "element_count": len(mesh.elements),
            "material_count": len(materials),
            "step_count": len(steps),
            "reported_step_count": len(filtered_steps),
            "problem_type": problem_type,
        },
        "sections": [
            {
                "id": section_id,
                "title": _section_title(section_id),
                "enabled": section_id in sections,
                "rows": section_rows.get(section_id, []),
            }
            for section_id in DEFAULT_REPORT_SECTIONS
        ],
    }


def _time_range(report_cfg: Mapping[str, Any]) -> list[float] | None:
    raw = report_cfg.get("time_range", report_cfg.get("times"))
    values = [float(item) for item in _ensure_list(raw)] if raw is not None else []
    if len(values) >= 2:
        return [min(values[0], values[-1]), max(values[0], values[-1])]
    return None


def _time_in_range(time_value: float, time_range: Sequence[float] | None) -> bool:
    if time_range is None:
        return True
    return float(time_range[0]) <= time_value <= float(time_range[1])


def _model_rows(mesh: Mesh2D, problem_type: str) -> list[list[Any]]:
    x_min, y_min = np.min(mesh.coords, axis=0)
    x_max, y_max = np.max(mesh.coords, axis=0)
    return [
        ["problem_type", problem_type],
        ["x_range_m", f"{float(x_min):.8g} - {float(x_max):.8g}"],
        ["y_range_m", f"{float(y_min):.8g} - {float(y_max):.8g}"],
    ]


def _mesh_rows(mesh: Mesh2D) -> list[list[Any]]:
    element_types: dict[str, int] = {}
    for element in mesh.elements:
        element_types[element.type] = element_types.get(element.type, 0) + 1
    return [
        ["node_count", len(mesh.node_ids)],
        ["element_count", len(mesh.elements)],
        ["element_types", ", ".join(f"{key}:{value}" for key, value in sorted(element_types.items()))],
        ["node_set_count", len(mesh.node_sets)],
    ]


def _analysis_rows(steps: Sequence[Any], problem_type: str, seepage: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = [["mode", seepage.get("mode", seepage.get("analysis_mode", ""))], ["problem_type", problem_type]]
    for step in steps:
        head = np.asarray(step.total_head, dtype=float)
        rows.append([f"step {step.index}", f"time={step.time:g}, iter={step.iteration_count}, residual={step.residual_norm:.3e}, head=[{float(np.min(head)):.6g}, {float(np.max(head)):.6g}]"])
    return rows


def _material_rows(materials: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for name, material in materials.items():
        rows.append(
            [
                name,
                f"kx={material.kx:.6g}, ky={material.ky:.6g}, Ss={material.specific_storage:.6g}, angle={material.angle_deg:.6g}, unsat={material.unsaturated_model}, alpha={material.alpha:.6g}, n={material.n:.6g}, theta=[{material.theta_r:.6g}, {material.theta_s:.6g}]",
            ]
        )
    return rows


def _boundary_rows(seepage: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key in (
        "known_head_bcs",
        "head_bcs",
        "water_level_bcs",
        "pressure_head_bcs",
        "flux_bcs",
        "pore_flux_bcs",
        "flow_bcs",
        "rainfall_bcs",
        "seepage_faces",
        "point_sources",
        "point_source_bcs",
    ):
        count = len([item for item in _ensure_list(seepage.get(key, [])) if isinstance(item, Mapping)])
        if count:
            rows.append([key, count])
    if "rainfall" in seepage:
        rows.append(["rainfall", "defined"])
    return rows or [["boundary_conditions", "none"]]


def _artifact_rows(artifacts: Mapping[str, str]) -> list[list[Any]]:
    return [[key, value] for key, value in sorted(artifacts.items())]


def _history_selection_rows(node_ids: Sequence[str], element_ids: Sequence[str], time_range: Sequence[float] | None) -> list[list[Any]]:
    return [
        ["node_ids", ", ".join(node_ids) if node_ids else "all"],
        ["element_ids", ", ".join(element_ids) if element_ids else "all"],
        ["time_range", f"{time_range[0]:g} - {time_range[1]:g}" if time_range else "all"],
    ]


def _write_sections_csv(path: Path, data: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "title", "enabled", "row_count"])
        writer.writeheader()
        for section in data["sections"]:
            writer.writerow({"section": section["id"], "title": section["title"], "enabled": section["enabled"], "row_count": len(section["rows"])})


def _print_profile(data: Mapping[str, Any], report_cfg: Mapping[str, Any], artifacts: Mapping[str, str]) -> dict[str, Any]:
    raw_profile = report_cfg.get("print_profile", report_cfg.get("ppf_profile", report_cfg.get("page_profile", {})))
    profile = raw_profile if isinstance(raw_profile, Mapping) else {}
    paper = profile.get("paper", profile.get("paper_size", report_cfg.get("paper_size", "A4")))
    orientation = profile.get("orientation", report_cfg.get("orientation", "portrait"))
    margins = profile.get("margins_mm", report_cfg.get("margins_mm", [15.0, 15.0, 15.0, 15.0]))
    figure_style = profile.get("figure_style", report_cfg.get("figure_style", {}))
    if not isinstance(figure_style, Mapping):
        figure_style = {}
    return {
        "schema": "geofem.vgflow2d.public_ppf_profile.v1",
        "profile": "Open print-profile substitute for VGFlow 2D Pre/Post PPF handoff; not a proprietary PPF binary.",
        "page": {
            "paper_size": str(paper),
            "orientation": str(orientation),
            "margins_mm": _four_margins(margins),
            "dpi": int(profile.get("dpi", report_cfg.get("dpi", 300)) or 300),
        },
        "selection": data["selection"],
        "section_layout": [
            {
                "section": section["id"],
                "title": section["title"],
                "enabled": bool(section["enabled"]),
                "row_count": len(section["rows"]),
                "page_break_before": section["id"] in set(str(item) for item in _ensure_list(profile.get("page_break_before", []))),
            }
            for section in data["sections"]
        ],
        "post_apply": _post_apply_rows(profile, report_cfg, artifacts),
        "figure_style": {
            "contour_palette": str(figure_style.get("contour_palette", "blue_red")),
            "vector_scale": float(figure_style.get("vector_scale", 1.0) or 1.0),
            "line_width_pt": float(figure_style.get("line_width_pt", 0.6) or 0.6),
            "show_legend": bool(figure_style.get("show_legend", True)),
            "show_frame": bool(figure_style.get("show_frame", True)),
        },
    }


def _post_apply_rows(profile: Mapping[str, Any], report_cfg: Mapping[str, Any], artifacts: Mapping[str, str]) -> list[dict[str, Any]]:
    requested = _ensure_list(profile.get("post_apply", report_cfg.get("post_apply", [])))
    if not requested:
        requested = ["post_contours", "flow_vectors", "flowlines", "section_flows", "time_history"]
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(requested, start=1):
        artifact_key = str(item)
        rows.append(
            {
                "order": index,
                "artifact_key": artifact_key,
                "artifact": artifacts.get(artifact_key, ""),
                "operation": "include_in_report",
                "available": artifact_key in artifacts,
            }
        )
    return rows


def _four_margins(value: Any) -> list[float]:
    values = [float(item) for item in _ensure_list(value)]
    if len(values) == 1:
        return values * 4
    if len(values) >= 4:
        return values[:4]
    return [15.0, 15.0, 15.0, 15.0]


def _write_print_profile_csv(path: Path, profile: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "key", "value"])
        writer.writeheader()
        for key, value in profile["page"].items():
            writer.writerow({"group": "page", "key": key, "value": json.dumps(value, ensure_ascii=False)})
        for row in profile["section_layout"]:
            writer.writerow({"group": "section_layout", "key": row["section"], "value": json.dumps(row, ensure_ascii=False)})
        for row in profile["post_apply"]:
            writer.writerow({"group": "post_apply", "key": row["artifact_key"], "value": json.dumps(row, ensure_ascii=False)})
        for key, value in profile["figure_style"].items():
            writer.writerow({"group": "figure_style", "key": key, "value": json.dumps(value, ensure_ascii=False)})


def _print_profile_html(profile: Mapping[str, Any]) -> str:
    page = kv_table([(key, value) for key, value in profile["page"].items()])
    sections = table(["section", "title", "enabled", "page break"], [[row["section"], row["title"], row["enabled"], row["page_break_before"]] for row in profile["section_layout"]])
    post = table(["order", "artifact", "available", "operation"], [[row["order"], row["artifact_key"], row["available"], row["operation"]] for row in profile["post_apply"]])
    style = kv_table([(key, value) for key, value in profile["figure_style"].items()])
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D Public PPF Profile</title>"
        f"<style>{report_css()}</style></head><body>"
        "<h1>VGFlow 2D Public PPF Profile Substitute</h1>"
        "<p>商用PPFファイルではなく、同じ役割を本ツール内で再現する公開印刷プロファイルです。</p>"
        f"<section><h2>Page</h2>{page}</section>"
        f"<section><h2>Section Layout</h2>{sections}</section>"
        f"<section><h2>Post Apply</h2>{post}</section>"
        f"<section><h2>Figure Style</h2>{style}</section>"
        "</body></html>"
    )


def _report_html(data: Mapping[str, Any], root: Path) -> str:
    summary = kv_table([(key, value) for key, value in data["summary"].items()])
    sections_html: list[str] = []
    for section in data["sections"]:
        if not section["enabled"]:
            continue
        rows = section["rows"]
        if section["id"] == "post_outputs":
            body = table(["項目", "成果物"], [[row[0], rel_link(row[1], root)] for row in rows], raw_columns={1})
        else:
            body = table(["項目", "内容"], rows)
        sections_html.append(f"<section><h2>{html_escape(section['title'])}</h2>{body}</section>")
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D 公開代替帳票</title>"
        f"<style>{report_css()}</style></head><body>"
        "<div class=\"cover\"><div><p class=\"eyebrow\">VGFlow 2D public substitute</p><h1>VGFlow 2D 公開代替帳票</h1></div>"
        f"<div class=\"cover-summary\">{summary}</div></div>"
        + "".join(sections_html)
        + "</body></html>"
    )


def _report_lines(data: Mapping[str, Any]) -> list[str]:
    lines = ["VGFlow 2D 公開代替帳票", ""]
    lines.extend(f"{key}: {value}" for key, value in data["summary"].items())
    for section in data["sections"]:
        if not section["enabled"]:
            continue
        lines.extend(["", f"[{section['title']}]"])
        for row in section["rows"][:40]:
            lines.append(f"{row[0]}: {row[1] if len(row) > 1 else ''}")
    return lines


def _section_title(section_id: str) -> str:
    return {
        "model": "モデル",
        "mesh": "メッシュ分割",
        "analysis": "解析条件",
        "materials": "浸透要素定義・不飽和特性",
        "boundaries": "境界条件",
        "post_outputs": "Post成果物",
        "time_history": "出力範囲指定",
        "warnings": "警告",
    }.get(section_id, section_id)


__all__ = ["write_vgflow_report_bundle"]
