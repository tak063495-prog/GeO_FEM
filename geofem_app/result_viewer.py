"""Result-view index artifacts shared by GUI and CLI consumers."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .fem2d_types import Mesh2D, SolveResult2D, StageResult2D


def build_result_view_index(result: SolveResult2D) -> dict[str, Any]:
    stages = [_stage_view_record(result.mesh, stage, result.output_dir) for stage in result.stages]
    return {
        "schema": "geofem.result_view_index.v1",
        "output_dir": str(result.output_dir),
        "stage_count": len(stages),
        "node_count": len(result.mesh.node_ids),
        "element_count": len(result.mesh.elements),
        "stages": stages,
        "stage_comparison": _stage_comparison(stages),
    }


def write_result_view_index(result: SolveResult2D, output_dir: str | Path | None = None) -> dict[str, str]:
    out = Path(output_dir) if output_dir is not None else result.output_dir
    out.mkdir(parents=True, exist_ok=True)
    index = build_result_view_index(result)
    json_path = out / "result_view_index.json"
    csv_path = out / "result_view_index.csv"
    html_path = out / "result_view_index.html"
    json_path.write_text(json.dumps(index, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["stage", "time", "max_displacement", "max_settlement", "max_pore_pressure", "node_tables", "element_tables", "history_tables"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for stage in index["stages"]:
            writer.writerow({field: stage.get(field, "") for field in fields})
    html_path.write_text(_result_view_html(index), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _stage_view_record(mesh: Mesh2D, stage: StageResult2D, root: Path) -> dict[str, Any]:
    stage_dir = stage.output_dir or root
    solver = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
    large = solver.get("large_deformation", {}) if isinstance(solver.get("large_deformation", {}), Mapping) else {}
    files = {
        "displacements": stage_dir / "displacements.csv",
        "reactions": stage_dir / "reactions.csv",
        "element_stress": stage_dir / "element_stress.csv",
        "integration_point_stress": stage_dir / "integration_point_stress.csv",
        "pore_pressure": stage_dir / "pore_pressure.csv",
        "vtk": stage_dir / "results.vtk",
        "report": stage_dir / "report.html",
        "dynamic_history": stage_dir / "dynamic_history.csv",
        "riks_path": stage_dir / "riks_path.csv",
        "liquefaction_history": stage_dir / "liquefaction_history.csv",
    }
    node_tables = [name for name in ("displacements", "reactions", "pore_pressure") if files[name].exists()]
    element_tables = [name for name in ("element_stress", "integration_point_stress") if files[name].exists()]
    history_tables = [name for name in ("dynamic_history", "riks_path", "liquefaction_history") if files[name].exists()]
    return {
        "stage": stage.name,
        "time": stage.time,
        "output_dir": str(stage_dir),
        "active_element_count": len(stage.active_elements),
        "max_displacement": _max_displacement(stage, len(mesh.node_ids)),
        "max_settlement": _max_settlement(stage, len(mesh.node_ids)),
        "max_pore_pressure": None if stage.pore_pressure is None else float(np.max(stage.pore_pressure)),
        "node_tables": ",".join(node_tables),
        "element_tables": ",".join(element_tables),
        "history_tables": ",".join(history_tables),
        "components": _components(stage),
        "files": {name: _rel(path, root) for name, path in files.items() if path.exists()},
        "node_count": len(mesh.node_ids),
        "element_result_count": len(stage.element_results),
        "integration_point_result_count": len(stage.integration_point_results),
        "geometry_mode": solver.get("geometry_mode", ""),
        "element_type": solver.get("element_type", ""),
        "integration": solver.get("integration", ""),
        "material_model": solver.get("material_model", ""),
        "batched_elements": solver.get("batched_elements", 0),
        "fallback_count": solver.get("fallback_count", 0),
        "fallback_reasons": solver.get("fallback_reasons", []),
        "final_increment_only_postprocess": bool(large.get("skip_intermediate_postprocessing", False)),
        "postprocess_policy": large.get("postprocess_policy", solver.get("postprocess_results", "")),
    }


def _components(stage: StageResult2D) -> dict[str, list[str]]:
    node = ["ux", "uy", "u_norm", "settlement"]
    if stage.pore_pressure is not None:
        node.append("pore_pressure")
    element: list[str] = []
    if stage.element_results:
        element = [key for key in stage.element_results[0] if key not in {"element_id", "material", "active"}]
    history: list[str] = []
    dynamic = stage.solver_info.get("dynamic", {})
    if isinstance(dynamic, Mapping) and dynamic.get("history"):
        history.extend(["time", "dt", "residual_norm", "max_displacement"])
    if isinstance(stage.solver_info.get("riks"), Mapping):
        history.extend(["lambda", "load_factor", "residual_norm"])
    if isinstance(stage.solver_info.get("large_deformation"), Mapping):
        history.extend(["load_end", "adaptive_action", "residual_norm", "postprocessed"])
    if isinstance(stage.solver_info.get("consolidation"), Mapping):
        history.extend(["time", "pressure_residual_norm", "flow_balance", "max_pore_pressure"])
    return {"node": node, "element": element, "history": sorted(set(history))}


def _stage_comparison(stages: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: Mapping[str, Any] | None = None
    for stage in stages:
        row = {
            "stage": stage.get("stage"),
            "time": stage.get("time"),
            "max_displacement": stage.get("max_displacement"),
            "delta_max_displacement": None,
            "max_settlement": stage.get("max_settlement"),
            "active_element_count": stage.get("active_element_count"),
        }
        if previous is not None:
            row["delta_max_displacement"] = float(stage.get("max_displacement", 0.0) or 0.0) - float(previous.get("max_displacement", 0.0) or 0.0)
        previous = stage
        rows.append(row)
    return rows


def _result_view_html(index: Mapping[str, Any]) -> str:
    rows = []
    for stage in index.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        links = []
        files = stage.get("files", {})
        if isinstance(files, Mapping):
            links = [f'<a href="{html.escape(str(path))}">{html.escape(str(name))}</a>' for name, path in files.items()]
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(stage.get('stage', '')))}</td>"
            f"<td>{html.escape(str(stage.get('time', '')))}</td>"
            f"<td>{html.escape(str(stage.get('max_displacement', '')))}</td>"
            f"<td>{html.escape(str(stage.get('max_pore_pressure', '')))}</td>"
            f"<td>{', '.join(links)}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM result view index</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>結果ビュー索引</h1>
<p>stages={int(index.get('stage_count', 0) or 0)}, nodes={int(index.get('node_count', 0) or 0)}, elements={int(index.get('element_count', 0) or 0)}</p>
<table><thead><tr><th>stage</th><th>time</th><th>max displacement</th><th>max pore pressure</th><th>files</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


def _max_displacement(stage: StageResult2D, node_count: int) -> float:
    limit = min(stage.displacements.size, node_count * 2)
    if limit < 2:
        return 0.0
    values = [float(np.hypot(stage.displacements[i], stage.displacements[i + 1])) for i in range(0, limit - 1, 2)]
    return max(values, default=0.0)


def _max_settlement(stage: StageResult2D, node_count: int) -> float:
    limit = min(stage.displacements.size, node_count * 2)
    if limit < 2:
        return 0.0
    return float(max((-stage.displacements[1:limit:2]), default=0.0))


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


__all__ = ["build_result_view_index", "write_result_view_index"]
