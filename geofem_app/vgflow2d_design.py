"""Design-aid diagnostics for the VGFlow 2D public substitute."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fem2d_types import Mesh2D
from .fem2d_utils import _ensure_list
from .html_report_utils import html_escape, report_css, table
from .vgflow2d_post import vgflow_element_post_fields


VGFLOW_DESIGN_TEMPLATES: dict[str, dict[str, Any]] = {
    "cutoff_wall": {
        "label": "Cutoff wall seepage check",
        "inputs": ["wall_x", "embedment_depth", "critical_gradient", "flow_sections"],
        "checks": ["head_difference_gradient", "section_flow", "local_piping_boiling"],
    },
    "drainage_well": {
        "label": "Drainage well seepage check",
        "inputs": ["well_points", "target_head", "drawdown_pairs"],
        "checks": ["two_point_head_difference", "local_gradient"],
    },
    "wellpoint": {
        "label": "Wellpoint drawdown check",
        "inputs": ["wellpoint_line", "pump_stage_times", "target_head"],
        "checks": ["two_point_head_difference", "courant_number"],
    },
    "confined_aquifer": {
        "label": "Confined aquifer pressure check",
        "inputs": ["pressure_head_boundary", "relief_boundary", "critical_gradient"],
        "checks": ["pressure_head_difference", "piping_boiling"],
    },
    "tunnel_seepage_face": {
        "label": "Tunnel seepage face check",
        "inputs": ["seepage_faces", "flow_sections", "critical_gradient"],
        "checks": ["seepage_face_flow", "local_gradient"],
    },
}


def vgflow_design_template_catalog() -> list[dict[str, Any]]:
    return [{"id": key, **value} for key, value in sorted(VGFLOW_DESIGN_TEMPLATES.items())]


def write_vgflow_design_checks(
    out: Path,
    mesh: Mesh2D,
    materials: Mapping[str, Any],
    steps: Sequence[Any],
    problem_type: str,
    seepage: Mapping[str, Any],
) -> dict[str, str]:
    paths = {
        "design_checks_json": str(out / "vgflow_design_checks.json"),
        "design_checks_csv": str(out / "vgflow_design_checks.csv"),
        "design_checks_html": str(out / "vgflow_design_checks.html"),
        "design_templates": str(out / "vgflow_design_templates.json"),
    }
    cfg = _design_cfg(seepage)
    rows = _design_check_rows(mesh, materials, steps, problem_type, seepage, cfg)
    payload = {
        "schema": "geofem.vgflow2d.design_checks.public_substitute.v1",
        "features": [
            "two_point_head_difference",
            "piping_boiling_gradient_ratio",
            "local_hydraulic_gradient_hotspots",
            "courant_number_guidance",
            "purpose_template_catalog",
        ],
        "thresholds": _thresholds(cfg),
        "checks": rows,
    }
    Path(paths["design_checks_json"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_design_csv(Path(paths["design_checks_csv"]), rows)
    Path(paths["design_checks_html"]).write_text(_design_html(payload), encoding="utf-8")
    Path(paths["design_templates"]).write_text(json.dumps({"templates": vgflow_design_template_catalog()}, ensure_ascii=False, indent=2), encoding="utf-8")
    return paths


def _design_cfg(seepage: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("design_checks", "design", "design_aids", "verification_checks"):
        value = seepage.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _thresholds(cfg: Mapping[str, Any]) -> dict[str, float]:
    critical = float(cfg.get("critical_gradient", cfg.get("boiling_critical_gradient", 1.0)) or 1.0)
    warning_ratio = float(cfg.get("warning_ratio", cfg.get("gradient_warning_ratio", 0.8)) or 0.8)
    max_courant = float(cfg.get("max_courant", cfg.get("courant_limit", 1.0)) or 1.0)
    return {"critical_gradient": critical, "warning_ratio": warning_ratio, "max_courant": max_courant}


def _design_check_rows(mesh: Mesh2D, materials: Mapping[str, Any], steps: Sequence[Any], problem_type: str, seepage: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    thresholds = _thresholds(cfg)
    rows: list[dict[str, Any]] = []
    element_fields = {step.index: vgflow_element_post_fields(mesh, materials, step, problem_type) for step in steps}
    rows.extend(_head_difference_rows(mesh, steps, cfg, thresholds["critical_gradient"], thresholds["warning_ratio"]))
    rows.extend(_local_gradient_rows(steps, element_fields, thresholds["critical_gradient"], thresholds["warning_ratio"]))
    rows.extend(_courant_rows(mesh, steps, element_fields, thresholds["max_courant"]))
    rows.extend(_template_rows(seepage, cfg))
    return rows


def _head_difference_rows(mesh: Mesh2D, steps: Sequence[Any], cfg: Mapping[str, Any], critical_gradient: float, warning_ratio: float) -> list[dict[str, Any]]:
    pairs = _head_pairs(mesh, cfg)
    rows: list[dict[str, Any]] = []
    for step in steps:
        for pair in pairs:
            ia = _point_index(mesh, pair["a"])
            ib = _point_index(mesh, pair["b"])
            ha = float(step.total_head[ia])
            hb = float(step.total_head[ib])
            distance = max(float(np.linalg.norm(mesh.coords[ia] - mesh.coords[ib])), np.finfo(float).eps)
            gradient = abs(ha - hb) / distance
            ratio = gradient / max(critical_gradient, np.finfo(float).eps)
            rows.append(
                _row(
                    step,
                    "two_point_head_difference",
                    pair["name"],
                    "gradient",
                    gradient,
                    critical_gradient,
                    ratio,
                    _status(ratio, warning_ratio),
                    f"head_a={ha:.8g}, head_b={hb:.8g}, distance={distance:.8g}",
                )
            )
    return rows


def _local_gradient_rows(steps: Sequence[Any], element_fields: Mapping[int, Sequence[Mapping[str, Any]]], critical_gradient: float, warning_ratio: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in steps:
        fields = list(element_fields[step.index])
        if not fields:
            continue
        worst = max(fields, key=lambda row: float(row["hydraulic_gradient_abs"]))
        gradient = float(worst["hydraulic_gradient_abs"])
        ratio = gradient / max(critical_gradient, np.finfo(float).eps)
        rows.append(
            _row(
                step,
                "local_piping_boiling",
                str(worst["element_id"]),
                "hydraulic_gradient_abs",
                gradient,
                critical_gradient,
                ratio,
                _status(ratio, warning_ratio),
                f"element={worst['element_id']}, x={float(worst['x']):.8g}, y={float(worst['y']):.8g}",
            )
        )
    return rows


def _courant_rows(mesh: Mesh2D, steps: Sequence[Any], element_fields: Mapping[int, Sequence[Mapping[str, Any]]], max_courant: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    length = _minimum_element_length(mesh)
    previous_time: float | None = None
    for step in steps:
        fields = list(element_fields[step.index])
        if previous_time is None:
            previous_time = float(step.time)
            continue
        dt = max(float(step.time) - previous_time, 0.0)
        previous_time = float(step.time)
        if dt <= 0.0 or not fields:
            continue
        worst = max(fields, key=lambda row: float(row["velocity_abs_m_s"]))
        courant = float(worst["velocity_abs_m_s"]) * dt / max(length, np.finfo(float).eps)
        ratio = courant / max(max_courant, np.finfo(float).eps)
        rows.append(
            _row(
                step,
                "courant_number",
                str(worst["element_id"]),
                "C=v*dt/dl",
                courant,
                max_courant,
                ratio,
                "fail" if ratio >= 1.0 else "ok",
                f"dt={dt:.8g}, dl={length:.8g}, vmax={float(worst['velocity_abs_m_s']):.8g}",
            )
        )
    return rows


def _template_rows(seepage: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    requested = [str(item) for item in _ensure_list(cfg.get("templates", seepage.get("purpose_templates", [])))]
    if not requested:
        return []
    rows: list[dict[str, Any]] = []
    dummy_step = type("_DesignTemplateStep", (), {"index": -1, "time": 0.0})()
    for template_id in requested:
        template = VGFLOW_DESIGN_TEMPLATES.get(template_id)
        status = "ok" if template else "warn"
        detail = json.dumps(template or {"missing": template_id}, ensure_ascii=False, sort_keys=True)
        rows.append(_row(dummy_step, "purpose_template", template_id, "template_defined", 1.0 if template else 0.0, 1.0, 1.0 if template else 0.0, status, detail))
    return rows


def _head_pairs(mesh: Mesh2D, cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_pairs = _ensure_list(cfg.get("head_difference_pairs", cfg.get("point_pairs", cfg.get("gw_pairs", []))))
    pairs: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_pairs, start=1):
        if not isinstance(raw, Mapping):
            continue
        a = raw.get("a", raw.get("from", {"node": raw.get("a_node", raw.get("node_a"))}))
        b = raw.get("b", raw.get("to", {"node": raw.get("b_node", raw.get("node_b"))}))
        pairs.append({"name": str(raw.get("name", f"pair_{index}")), "a": a, "b": b})
    if pairs:
        return pairs
    return [_default_head_pair(mesh)]


def _default_head_pair(mesh: Mesh2D) -> dict[str, Any]:
    x_min, y_min = np.min(mesh.coords, axis=0)
    x_max, y_max = np.max(mesh.coords, axis=0)
    y_mid = 0.5 * (float(y_min) + float(y_max))
    return {"name": "default_upstream_downstream", "a": [float(x_min), y_mid], "b": [float(x_max), y_mid]}


def _point_index(mesh: Mesh2D, spec: Any) -> int:
    if isinstance(spec, Mapping):
        node_id = spec.get("node", spec.get("node_id"))
        if node_id is not None:
            return mesh.node_index[str(node_id)]
        xy = (float(spec.get("x", 0.0)), float(spec.get("y", 0.0)))
    elif isinstance(spec, (list, tuple)) and len(spec) >= 2:
        xy = (float(spec[0]), float(spec[1]))
    else:
        return mesh.node_index[str(spec)]
    delta = mesh.coords - np.asarray(xy, dtype=float)
    return int(np.argmin(np.einsum("ij,ij->i", delta, delta)))


def _minimum_element_length(mesh: Mesh2D) -> float:
    lengths: list[float] = []
    for element in mesh.elements:
        corners = list(element.nodes[:4] if element.type.upper().startswith("QUAD") else element.nodes[:3])
        for a, b in zip(corners, [*corners[1:], corners[0]]):
            pa = mesh.coords[mesh.node_index[a]]
            pb = mesh.coords[mesh.node_index[b]]
            length = float(np.linalg.norm(pa - pb))
            if length > 0.0:
                lengths.append(length)
    return min(lengths) if lengths else 1.0


def _row(step: Any, check: str, name: str, metric: str, value: float, limit: float, ratio: float, status: str, detail: str) -> dict[str, Any]:
    return {
        "step": int(step.index),
        "time": float(step.time),
        "check": check,
        "name": name,
        "metric": metric,
        "value": float(value),
        "limit": float(limit),
        "ratio": float(ratio),
        "status": status,
        "detail": detail,
    }


def _status(ratio: float, warning_ratio: float) -> str:
    if ratio >= 1.0:
        return "fail"
    if ratio >= warning_ratio:
        return "warn"
    return "ok"


def _write_design_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["step", "time", "check", "name", "metric", "value", "limit", "ratio", "status", "detail"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _design_html(payload: Mapping[str, Any]) -> str:
    rows = [[row["step"], row["time"], row["check"], row["name"], row["metric"], row["value"], row["limit"], row["ratio"], row["status"], row["detail"]] for row in payload["checks"]]
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D design checks</title>"
        f"<style>{report_css()}</style></head><body>"
        "<h1>VGFlow 2D design checks</h1>"
        f"<p>{html_escape(payload['schema'])}</p>"
        + table(["step", "time", "check", "name", "metric", "value", "limit", "ratio", "status", "detail"], rows)
        + "</body></html>"
    )


__all__ = ["vgflow_design_template_catalog", "write_vgflow_design_checks"]
