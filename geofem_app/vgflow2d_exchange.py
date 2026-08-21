"""Time-step exchange package writer for VGFlow 2D public substitute outputs."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fem2d_types import Mesh2D
from .fem2d_utils import _ensure_list
from .hydro_exchange import element_potential_points, shared_hydro_exchange_engine, waterline_points_from_total_head


def write_vgflow_exchange_outputs(
    out: Path,
    mesh: Mesh2D,
    steps: Sequence[Any],
    problem_type: str,
    seepage: Mapping[str, Any],
) -> dict[str, str]:
    paths = {
        "exchange_manifest": str(out / "vgflow_exchange_manifest.json"),
        "exchange_time_catalog": str(out / "vgflow_exchange_time_catalog.csv"),
        "exchange_operation_log": str(out / "vgflow_exchange_operation_log.json"),
        "exchange_html": str(out / "vgflow_exchange_time_selection.html"),
        "geofeas_selected_prs": str(out / "vgflow_geofeas_selected_waterline.PRS"),
        "geofeas_selected_ptn": str(out / "vgflow_geofeas_selected_potential.PTN"),
        "slope_selected_waterlines": str(out / "vgflow_slope_selected_waterlines.csv"),
        "slope_selected_potentials": str(out / "vgflow_slope_selected_potentials.csv"),
    }
    cfg = _exchange_cfg(seepage)
    selected_steps = _selected_steps(steps, cfg)
    catalog = _time_catalog(steps, selected_steps)
    waterlines = {step.index: waterline_points_from_total_head(mesh, step.total_head, problem_type, sort_points=True) for step in selected_steps}
    potentials = {step.index: element_potential_points(mesh, step.total_head) for step in selected_steps}
    _write_time_catalog(Path(paths["exchange_time_catalog"]), catalog)
    _write_selected_prs(Path(paths["geofeas_selected_prs"]), selected_steps, waterlines)
    _write_selected_ptn(Path(paths["geofeas_selected_ptn"]), selected_steps, potentials)
    _write_slope_waterlines(Path(paths["slope_selected_waterlines"]), selected_steps, waterlines)
    _write_slope_potentials(Path(paths["slope_selected_potentials"]), selected_steps, potentials)
    operation_log = _operation_log(cfg, catalog, selected_steps)
    Path(paths["exchange_operation_log"]).write_text(json.dumps({"operation_log": operation_log}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest = {
        "schema": "geofem.vgflow2d.exchange.public_substitute.v1",
        "profile": "Open time-step selection package for GeoFEAS/slope-stability handoff; not a native VGFlow2D binary format.",
        "features": [
            "transient_time_step_selection_catalog",
            "selected_geofeas_waterline_prs",
            "selected_geofeas_potential_ptn",
            "slope_stability_waterline_csv",
            "selection_operation_log",
            "shared_hydro_exchange_engine",
        ],
        "shared_engine": shared_hydro_exchange_engine(),
        "selection": {
            "requested_steps": _ensure_list(cfg.get("selected_steps", cfg.get("steps", []))),
            "requested_times": _ensure_list(cfg.get("selected_times", cfg.get("times", []))),
            "selected_steps": [int(step.index) for step in selected_steps],
            "selected_times": [float(step.time) for step in selected_steps],
        },
        "artifacts": paths,
    }
    Path(paths["exchange_manifest"]).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    Path(paths["exchange_html"]).write_text(_html(manifest, catalog), encoding="utf-8")
    return paths


def _exchange_cfg(seepage: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("exchange", "vgflow_exchange", "geofeas_exchange", "slope_exchange"):
        value = seepage.get(key)
        if isinstance(value, Mapping):
            return value
    post = seepage.get("post", seepage.get("vgflow_post", {}))
    if isinstance(post, Mapping) and isinstance(post.get("exchange"), Mapping):
        return post["exchange"]
    return {}


def _selected_steps(steps: Sequence[Any], cfg: Mapping[str, Any]) -> list[Any]:
    if not steps:
        return []
    by_index = {int(step.index): step for step in steps}
    selected_indices = {int(value) for value in _ensure_list(cfg.get("selected_steps", cfg.get("steps", []))) if str(value) != ""}
    selected: list[Any] = [by_index[index] for index in sorted(selected_indices) if index in by_index]
    raw_times = [float(value) for value in _ensure_list(cfg.get("selected_times", cfg.get("times", []))) if str(value) != ""]
    for time_value in raw_times:
        nearest = min(steps, key=lambda step: abs(float(step.time) - time_value))
        if nearest not in selected:
            selected.append(nearest)
    if not selected:
        mode = str(cfg.get("default_selection", "latest")).lower()
        selected = list(steps) if mode in {"all", "every", "each"} else [steps[-1]]
    return sorted(selected, key=lambda step: int(step.index))


def _time_catalog(steps: Sequence[Any], selected_steps: Sequence[Any]) -> list[dict[str, Any]]:
    selected = {int(step.index) for step in selected_steps}
    return [
        {
            "step": int(step.index),
            "time": float(step.time),
            "selected": int(step.index) in selected,
            "waterline_label": f"step_{int(step.index):04d}_waterline",
            "potential_label": f"step_{int(step.index):04d}_potential",
        }
        for step in steps
    ]


def _write_time_catalog(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = ["step", "time", "selected", "waterline_label", "potential_label"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _write_selected_prs(path: Path, steps: Sequence[Any], waterlines: Mapping[int, Sequence[tuple[float, float]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# GeoFEM VGFlow2D public substitute selected PRS", "waterline pressure_head=0"])
        writer.writerow(["step", "time", "point_index", "x", "y"])
        for step in steps:
            for point_index, (x, y) in enumerate(waterlines.get(int(step.index), []), start=1):
                writer.writerow([int(step.index), float(step.time), point_index, x, y])


def _write_selected_ptn(path: Path, steps: Sequence[Any], potentials: Mapping[int, Sequence[Mapping[str, Any]]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["# GeoFEM VGFlow2D public substitute selected PTN", "element-center total_head"])
        writer.writerow(["step", "time", "element_id", "x", "y", "total_head_m"])
        for step in steps:
            for row in potentials.get(int(step.index), []):
                writer.writerow([int(step.index), float(step.time), row["element_id"], row["x"], row["y"], row["total_head_m"]])


def _write_slope_waterlines(path: Path, steps: Sequence[Any], waterlines: Mapping[int, Sequence[tuple[float, float]]]) -> None:
    fields = ["step", "time", "polyline_id", "point_index", "x", "y", "purpose"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            polyline_id = f"waterline_step_{int(step.index):04d}"
            for point_index, (x, y) in enumerate(waterlines.get(int(step.index), []), start=1):
                writer.writerow({"step": int(step.index), "time": float(step.time), "polyline_id": polyline_id, "point_index": point_index, "x": x, "y": y, "purpose": "slope_stability_phreatic_line"})


def _write_slope_potentials(path: Path, steps: Sequence[Any], potentials: Mapping[int, Sequence[Mapping[str, Any]]]) -> None:
    fields = ["step", "time", "element_id", "x", "y", "total_head_m", "purpose"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for step in steps:
            for row in potentials.get(int(step.index), []):
                writer.writerow({"step": int(step.index), "time": float(step.time), **row, "purpose": "slope_stability_potential_check"})


def _operation_log(cfg: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]], selected_steps: Sequence[Any]) -> list[dict[str, Any]]:
    return [
        {
            "command": "open_time_selection",
            "target": "VGFlow2D public exchange",
            "message": "List transient result times for GeoFEAS/slope-stability handoff.",
            "step_count": len(catalog),
        },
        {
            "command": "select_result_times",
            "selected_steps": [int(step.index) for step in selected_steps],
            "selected_times": [float(step.time) for step in selected_steps],
            "requested_steps": _ensure_list(cfg.get("selected_steps", cfg.get("steps", []))),
            "requested_times": _ensure_list(cfg.get("selected_times", cfg.get("times", []))),
        },
        {
            "command": "export_selected_waterline_potential",
            "targets": _ensure_list(cfg.get("targets", ["geofeas", "slope_stability"])),
            "message": "Write open PRS/PTN-style and CSV handoff artifacts.",
        },
    ]


def _html(manifest: Mapping[str, Any], catalog: Sequence[Mapping[str, Any]]) -> str:
    rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(row['step']))}</td>"
        f"<td>{html.escape(str(row['time']))}</td>"
        f"<td>{'selected' if row['selected'] else ''}</td>"
        f"<td>{html.escape(str(row['waterline_label']))}</td>"
        f"<td>{html.escape(str(row['potential_label']))}</td>"
        "</tr>"
        for row in catalog
    )
    selection = manifest.get("selection", {})
    return (
        "<!doctype html><html lang=\"ja\"><head><meta charset=\"utf-8\">"
        "<title>VGFlow 2D Exchange Time Selection</title>"
        "<style>body{font-family:Segoe UI,Arial,sans-serif;margin:24px}table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #d0d7de;padding:6px 8px;text-align:left}th{background:#f6f8fa}</style></head><body>"
        "<h1>VGFlow 2D Exchange Time Selection</h1>"
        f"<p>Selected steps: {html.escape(str(selection.get('selected_steps', [])))}</p>"
        "<table><thead><tr><th>step</th><th>time</th><th>selected</th><th>waterline</th><th>potential</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></body></html>"
    )


__all__ = ["write_vgflow_exchange_outputs"]
