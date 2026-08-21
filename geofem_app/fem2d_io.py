"""Output writers for 2D FEM stage and run results."""

from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import platform
import time
from typing import Any, Mapping

import numpy as np
import yaml

from .analysis_log import write_structured_analysis_log
from .fem2d_types import DOF_NAMES, Mesh2D, SolveResult2D, StageResult2D
from .geofeas_public import public_profile_summary, write_public_output_conditions, write_public_profile_summary
from .html_report_utils import format_value as _fmt
from .html_report_utils import html_escape as _h
from .html_report_utils import kv_table as _kv_table
from .html_report_utils import rel_link as _rel_link
from .html_report_utils import report_css as _report_css
from .html_report_utils import table as _table
from .large_model_operations import write_large_model_operation_artifacts
from .load_combinations import configured_load_combinations
from .material_models import write_material_reports
from .mesh_quality import write_mesh_quality_report
from .performance_kpis import write_performance_kpi_reports
from .performance_monitor import write_performance_summary
from .pdf_writer import write_text_pdf
from .reliability_summary import write_reliability_summary_reports
from .result_viewer import write_result_view_index
from .srm_reporting import (
    srm_fos_display,
    srm_fos_is_confirmed,
    srm_result_confidence,
    srm_result_status,
    srm_safety_verdict,
)
from .standard_report import write_standard_report_bundle


def _stage_output_formats(output_config: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(output_config, Mapping):
        return {"csv", "vtk", "html"}
    raw = output_config.get("formats", output_config.get("format", ("csv", "vtk", "html")))
    if raw is None:
        return {"csv", "html"}
    if isinstance(raw, str):
        values = [part.strip() for part in raw.replace(";", ",").split(",")]
    else:
        values = [str(part).strip() for part in raw if str(part).strip()] if isinstance(raw, (list, tuple, set)) else []
    normalized = {value.lower().replace("-", "_") for value in values if value}
    if not normalized:
        return {"csv", "html"}
    if "all" in normalized:
        return {"csv", "vtk", "html"}
    if "vtu" in normalized:
        normalized.add("vtk")
    if "report" in normalized:
        normalized.add("html")
    return normalized


def write_stage_outputs(mesh: Mesh2D, result: StageResult2D, output_dir: Path, *, output_config: Mapping[str, Any] | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = _stage_output_formats(output_config)
    if "csv" in formats:
        write_displacements_csv(mesh, result, output_dir / "displacements.csv")
        write_reactions_csv(mesh, result, output_dir / "reactions.csv")
        write_element_results_csv(result, output_dir / "element_stress.csv")
        write_integration_point_results_csv(result, output_dir / "integration_point_stress.csv")
        write_liquefaction_state_csv(result, output_dir / "liquefaction_state.csv")
        write_liquefaction_post_outputs(result, output_dir)
        if result.interface_results:
            write_interface_results_csv(result, output_dir / "interface_state.csv")
        if result.structural_results:
            write_structural_results_csv(result, output_dir / "structural_state.csv")
            write_structural_section_forces_csv(result, output_dir / "structural_section_forces.csv")
        if isinstance(result.solver_info.get("riks"), Mapping):
            write_riks_path_csv(result, output_dir / "riks_path.csv")
        if isinstance(result.solver_info.get("dynamic"), Mapping):
            write_dynamic_history_csv(result, output_dir / "dynamic_history.csv")
        if result.pore_pressure is not None:
            write_pore_pressure_csv(mesh, result, output_dir / "pore_pressure.csv")
    if "vtk" in formats:
        write_vtk(mesh, result, output_dir / "results.vtk")
    if "html" in formats:
        write_stage_report(mesh, result, output_dir / "report.html")


def write_displacements_csv(mesh: Mesh2D, result: StageResult2D, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "ux", "uy", "u_norm", "settlement"])
        for i, nid in enumerate(mesh.node_ids):
            ux = result.displacements[2 * i]
            uy = result.displacements[2 * i + 1]
            writer.writerow([nid, mesh.coords[i, 0], mesh.coords[i, 1], ux, uy, math.hypot(ux, uy), -uy])


def write_reactions_csv(mesh: Mesh2D, result: StageResult2D, path: Path) -> None:
    extra = result.solver_info.get("extra_dofs", {})
    if not isinstance(extra, Mapping):
        extra = {}
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "dof", "reaction", "constrained_value"])
        for dof, value in sorted(result.constrained_dofs.items()):
            node_i = dof // 2
            if node_i < len(mesh.node_ids) and dof < len(mesh.node_ids) * 2:
                writer.writerow([mesh.node_ids[node_i], DOF_NAMES[dof % 2], result.reactions[dof], value])
            else:
                label = str(extra.get(str(dof), extra.get(dof, f"extra:{dof}")))
                node_id, sep, dof_name = label.partition(":")
                writer.writerow([node_id if sep else "", dof_name if sep else label, result.reactions[dof], value])


def write_pore_pressure_csv(mesh: Mesh2D, result: StageResult2D, path: Path) -> None:
    if result.pore_pressure is None:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "x", "y", "pore_pressure", "time"])
        for i, nid in enumerate(mesh.node_ids):
            writer.writerow([nid, mesh.coords[i, 0], mesh.coords[i, 1], result.pore_pressure[i], result.time])


def write_dynamic_history_csv(result: StageResult2D, path: Path) -> None:
    dynamic = result.solver_info.get("dynamic", {})
    if not isinstance(dynamic, Mapping):
        return
    rows = [row for row in dynamic.get("history", []) if isinstance(row, Mapping)]
    if not rows:
        return
    fields = [
        "step",
        "time",
        "dt",
        "kh",
        "kv",
        "load_scale",
        "load_norm",
        "max_displacement",
        "max_velocity",
        "max_acceleration",
        "kinetic_energy",
        "strain_energy",
        "residual_norm",
        "max_pore_pressure",
        "min_pore_pressure",
        "cutbacks",
        "nonlinear_iterations",
        "converged",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_element_results_csv(result: StageResult2D, path: Path) -> None:
    if not result.element_results:
        return
    fieldnames = list(result.element_results[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.element_results)


def write_integration_point_results_csv(result: StageResult2D, path: Path) -> None:
    if not result.integration_point_results:
        return
    keys = list(result.integration_point_results[0].keys())
    fieldnames = ["stage", "time", *[key for key in keys if key not in {"stage", "time"}]]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in result.integration_point_results:
            writer.writerow({"stage": result.name, "time": result.time, **row})


def write_liquefaction_state_csv(result: StageResult2D, path: Path) -> None:
    fields = [
        "stage",
        "time",
        "element_id",
        "ip",
        "x",
        "y",
        "material",
        "advanced_model",
        "ru",
        "ru_generation_increment",
        "ru_dissipation_increment",
        "ru_dissipation_rate",
        "liquefaction_FL",
        "cycle_increment",
        "cycles",
        "cyclic_strain",
        "modulus_ratio",
        "effective_G",
        "effective_E",
        "dilatancy",
        "hardening_variable",
    ]
    rows = _liquefaction_rows(result, fields)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_liquefaction_post_outputs(result: StageResult2D, output_dir: Path) -> None:
    fields = [
        "stage",
        "time",
        "element_id",
        "ip",
        "x",
        "y",
        "material",
        "advanced_model",
        "ru",
        "liquefaction_FL",
        "cycle_increment",
        "cycles",
        "ru_generation_increment",
        "ru_dissipation_increment",
        "ru_dissipation_rate",
        "modulus_ratio",
    ]
    rows = _liquefaction_rows(result, fields)
    if not rows:
        return
    history = output_dir / "liquefaction_history.csv"
    with history.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    svg_path = output_dir / "liquefaction_ru_fl.svg"
    svg_path.write_text(_liquefaction_svg(rows), encoding="utf-8")
    html_path = output_dir / "liquefaction_post.html"
    html_path.write_text(_liquefaction_post_html(result, rows, history.name, svg_path.name), encoding="utf-8")


def _liquefaction_rows(result: StageResult2D, fields: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in result.integration_point_results:
        model = str(row.get("advanced_model", ""))
        if "liquefaction" not in model and float(row.get("ru", 0.0) or 0.0) <= 0.0:
            continue
        rows.append({"stage": result.name, "time": result.time, **{key: row.get(key, "") for key in fields if key not in {"stage", "time"}}})
    return rows


def _liquefaction_post_html(result: StageResult2D, rows: list[dict[str, Any]], history_name: str, svg_name: str) -> str:
    ru_values = [float(row.get("ru", 0.0) or 0.0) for row in rows]
    fl_values = [float(row.get("liquefaction_FL", math.inf) or math.inf) for row in rows]
    finite_fl = [value for value in fl_values if math.isfinite(value)]
    coupling = result.solver_info.get("consolidation", {}).get("liquefaction_coupling", {}) if isinstance(result.solver_info.get("consolidation", {}), Mapping) else {}
    metrics = [
        ("stage", result.name),
        ("max_ru", _fmt(max(ru_values) if ru_values else 0.0)),
        ("mean_ru", _fmt(sum(ru_values) / len(ru_values) if ru_values else 0.0)),
        ("min_FL", "-" if not finite_fl else _fmt(min(finite_fl))),
        ("points", len(rows)),
        ("coupled_generation_source", _fmt(float(coupling.get("generation_source", 0.0) or 0.0)) if isinstance(coupling, Mapping) else "-"),
        ("coupled_dissipation_matrix_sum", _fmt(float(coupling.get("dissipation_matrix_sum", 0.0) or 0.0)) if isinstance(coupling, Mapping) else "-"),
    ]
    top_rows = sorted(rows, key=lambda item: float(item.get("ru", 0.0) or 0.0), reverse=True)[:20]
    table_rows = "".join(
        "<tr>"
        f"<td>{_h(row.get('element_id', ''))}</td>"
        f"<td>{_h(row.get('ip', ''))}</td>"
        f"<td>{_fmt(float(row.get('x', 0.0) or 0.0))}</td>"
        f"<td>{_fmt(float(row.get('y', 0.0) or 0.0))}</td>"
        f"<td>{_fmt(float(row.get('ru', 0.0) or 0.0))}</td>"
        f"<td>{_fmt(float(row.get('liquefaction_FL', math.inf) or math.inf))}</td>"
        f"<td>{_fmt(float(row.get('cycles', 0.0) or 0.0))}</td>"
        "</tr>"
        for row in top_rows
    )
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Liquefaction post - {_h(result.name)}</title><style>{_report_css()}</style></head>
<body>
<h1>Liquefaction Post</h1>
{_kv_table(metrics)}
<p><a href="{_h(svg_name)}">ru/FL figure</a> | <a href="{_h(history_name)}">history CSV</a></p>
<img class="figure" src="{_h(svg_name)}" alt="Liquefaction ru and FL figure">
<h2>Critical integration points</h2>
<table><thead><tr><th>element</th><th>ip</th><th>x</th><th>y</th><th>ru</th><th>FL</th><th>cycles</th></tr></thead><tbody>{table_rows}</tbody></table>
</body></html>
"""


def _liquefaction_svg(rows: list[dict[str, Any]]) -> str:
    width, height = 720, 420
    pad = 42.0
    xs = [float(row.get("x", 0.0) or 0.0) for row in rows]
    ys = [float(row.get("y", 0.0) or 0.0) for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax <= xmin:
        xmax = xmin + 1.0
    if ymax <= ymin:
        ymax = ymin + 1.0

    def sx(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)

    def sy(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)

    circles = []
    for row in rows:
        ru = max(0.0, min(float(row.get("ru", 0.0) or 0.0), 1.0))
        fl = float(row.get("liquefaction_FL", math.inf) or math.inf)
        radius = 4.0 if not math.isfinite(fl) else max(3.0, min(10.0, 10.0 / max(fl, 0.2)))
        color = _contour_color(ru, 0.0, 1.0)
        title = f"element={row.get('element_id', '')}, ip={row.get('ip', '')}, ru={ru:.4g}, FL={fl:.4g}"
        circles.append(f'<circle cx="{sx(float(row.get("x", 0.0) or 0.0)):.2f}" cy="{sy(float(row.get("y", 0.0) or 0.0)):.2f}" r="{radius:.2f}" fill="{color}" stroke="#111827" stroke-width="0.6"><title>{_h(title)}</title></circle>')
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>
<text x="{pad}" y="24" class="svg-title">Liquefaction ru / FL integration-point map</text>
<rect x="{pad}" y="{pad}" width="{width - 2 * pad}" height="{height - 2 * pad}" fill="#f8fafc" stroke="#94a3b8"/>
{''.join(circles)}
<text x="{pad}" y="{height - 12}" fill="#334155" font-size="12">color: ru 0.0 to 1.0, radius: smaller FL is larger</text>
</svg>
"""


def write_interface_results_csv(result: StageResult2D, path: Path) -> None:
    if not result.interface_results:
        return
    fieldnames = list(result.interface_results[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.interface_results)


def write_structural_results_csv(result: StageResult2D, path: Path) -> None:
    if not result.structural_results:
        return
    fieldnames: list[str] = []
    for row in result.structural_results:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result.structural_results)


def write_structural_section_forces_csv(result: StageResult2D, path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for row in result.structural_results:
        samples = row.get("section_forces")
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            rows.append(
                {
                    "element_id": row.get("element_id", ""),
                    "type": row.get("type", ""),
                    "node_i": row.get("node_i", ""),
                    "node_j": row.get("node_j", ""),
                    **dict(sample),
                }
            )
    if not rows:
        return
    fieldnames = ["element_id", "type", "node_i", "node_j", "x", "ratio", "axial_force", "shear_force", "bending_moment"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_riks_path_csv(result: StageResult2D, path: Path) -> None:
    riks = result.solver_info.get("riks", {})
    if not isinstance(riks, Mapping):
        return
    rows = riks.get("path", [])
    if not isinstance(rows, list) or not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_vtk(mesh: Mesh2D, result: StageResult2D, path: Path) -> None:
    node_index = mesh.node_index
    vtk_type = {"TRI3": 5, "QUAD4": 9, "TRI6": 22, "QUAD8": 23}
    cell_size = sum(1 + len(e.nodes) for e in mesh.elements)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"GeoFEM 2D {result.name}\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        f.write(f"POINTS {len(mesh.node_ids)} float\n")
        for x, y in mesh.coords:
            f.write(f"{x:.16g} {y:.16g} 0\n")
        f.write(f"CELLS {len(mesh.elements)} {cell_size}\n")
        for e in mesh.elements:
            ids = " ".join(str(node_index[nid]) for nid in e.nodes)
            f.write(f"{len(e.nodes)} {ids}\n")
        f.write(f"CELL_TYPES {len(mesh.elements)}\n")
        for e in mesh.elements:
            f.write(f"{vtk_type[e.type]}\n")
        f.write(f"POINT_DATA {len(mesh.node_ids)}\n")
        f.write("VECTORS displacement float\n")
        for i in range(len(mesh.node_ids)):
            f.write(f"{result.displacements[2*i]:.16g} {result.displacements[2*i+1]:.16g} 0\n")
        f.write("SCALARS settlement float 1\nLOOKUP_TABLE default\n")
        for i in range(len(mesh.node_ids)):
            f.write(f"{-result.displacements[2*i+1]:.16g}\n")
        if result.pore_pressure is not None:
            f.write("SCALARS pore_pressure float 1\nLOOKUP_TABLE default\n")
            for value in result.pore_pressure:
                f.write(f"{value:.16g}\n")
        f.write(f"CELL_DATA {len(mesh.elements)}\n")
        for key in ["active", "plastic", "sigma_x", "sigma_y", "sigma_z", "tau_xy", "sigma_1", "sigma_3", "tau_max", "p", "q"]:
            f.write(f"SCALARS {key} float 1\nLOOKUP_TABLE default\n")
            for row in result.element_results:
                f.write(f"{row[key]:.16g}\n")


def write_stage_report(mesh: Mesh2D, result: StageResult2D, path: Path) -> None:
    max_settlement = max((-result.displacements[1::2]), default=0.0)
    max_unorm = max((math.hypot(result.displacements[2 * i], result.displacements[2 * i + 1]) for i in range(len(mesh.node_ids))), default=0.0)
    html = f"""<!doctype html>
<html lang="ja">
<meta charset="utf-8">
<title>GeoFEM 2D report - {result.name}</title>
<body>
<h1>GeoFEM 2D 解析レポート</h1>
<h2>{result.name}</h2>
<table border="1" cellspacing="0" cellpadding="4">
<tr><th>節点数</th><td>{len(mesh.node_ids)}</td></tr>
<tr><th>要素数</th><td>{len(mesh.elements)}</td></tr>
<tr><th>最大変位ノルム</th><td>{max_unorm:.8g}</td></tr>
<tr><th>最大沈下量 settlement=-uy</th><td>{max_settlement:.8g}</td></tr>
</table>
<p>内部応力符号は引張正です。2D平面ひずみとして eps_z=0, sigma_z を出力します。</p>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _run_io_profile_payload(result: SolveResult2D, entries: Mapping[str, float]) -> dict[str, Any]:
    stage_io = 0.0
    for stage in result.stages:
        solver_info = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
        performance = solver_info.get("performance", {}) if isinstance(solver_info.get("performance", {}), Mapping) else {}
        stage_io += float(performance.get("io_report_elapsed_seconds", 0.0) or 0.0)
    run_io = sum(float(value or 0.0) for value in entries.values())
    return {
        "schema": "geofem.run_io_profile.v1",
        "stage_io_report_elapsed_seconds": float(stage_io),
        "run_io_report_elapsed_seconds": float(run_io),
        "io_report_elapsed_seconds": float(stage_io + run_io),
        "entries": {str(key): float(value or 0.0) for key, value in entries.items()},
    }


def _run_output_policy(result: SolveResult2D) -> dict[str, Any]:
    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    output_cfg = cfg.get("output", cfg.get("outputs", {}))
    report_cfg = cfg.get("report", cfg.get("calculation_report", {}))
    output_map = output_cfg if isinstance(output_cfg, Mapping) else {}
    report_map = report_cfg if isinstance(report_cfg, Mapping) else {}
    lazy = bool(
        output_map.get("lazy_reports", output_map.get("defer_reports", False))
        or report_map.get("lazy", report_map.get("defer", report_map.get("defer_generation", False)))
    )
    return {
        "lazy_reports": lazy,
        "result_view": not bool(output_map.get("lazy_result_view", output_map.get("defer_result_view", lazy))),
        "standard_report": not bool(output_map.get("lazy_standard_report", output_map.get("defer_standard_report", lazy))),
        "calculation_report": not bool(report_map.get("lazy", report_map.get("defer", report_map.get("defer_generation", lazy)))),
        "formats": sorted(_stage_output_formats(output_map)),
        "write_log": bool(output_map.get("write_log", True)),
    }


def _deferred_artifact(paths: Mapping[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "generated": False,
        "deferred": True,
        "reason": reason,
        "paths": {str(key): str(value) for key, value in paths.items()},
    }


def _read_run_input_config(input_path: str | Path) -> dict[str, Any]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"input YAML for deferred report generation does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text) if text.strip() else {}
    if not isinstance(data, Mapping):
        raise ValueError(f"input config must be a mapping: {path}")
    return dict(data)


def _default_run_input_path(output_dir: Path) -> Path | None:
    candidates = (
        output_dir.parent / "input.yaml",
        output_dir.parent / "input.yml",
        output_dir / "input.yaml",
        output_dir / "input.yml",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _coerce_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    text = str(value)
    if text == "":
        return ""
    try:
        return float(text)
    except (TypeError, ValueError):
        return text


def _read_csv_dict_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{str(key): _coerce_csv_value(value) for key, value in row.items()} for row in reader]


def _stage_dir_from_summary(output_dir: Path, raw: Mapping[str, Any], index: int) -> Path:
    raw_dir = raw.get("output_dir")
    if raw_dir:
        path = Path(str(raw_dir))
        return path if path.is_absolute() else output_dir / path
    name = str(raw.get("name", f"Stage-{index}") or f"Stage-{index}")
    candidate = output_dir / name
    if candidate.exists():
        return candidate
    dirs = [path for path in output_dir.iterdir() if path.is_dir()]
    return dirs[index - 1] if index - 1 < len(dirs) else candidate


def _stage_displacements_from_csv(mesh: Mesh2D, stage_dir: Path) -> np.ndarray:
    values = np.zeros(len(mesh.node_ids) * 2, dtype=float)
    for row in _read_csv_dict_rows(stage_dir / "displacements.csv"):
        node_id = str(row.get("node_id", "") or "")
        if node_id not in mesh.node_index:
            continue
        idx = mesh.node_index[node_id]
        values[2 * idx] = float(row.get("ux", 0.0) or 0.0)
        values[2 * idx + 1] = float(row.get("uy", 0.0) or 0.0)
    return values


def _stage_pore_pressure_from_csv(mesh: Mesh2D, stage_dir: Path) -> np.ndarray | None:
    rows = _read_csv_dict_rows(stage_dir / "pore_pressure.csv")
    if not rows:
        return None
    values = np.zeros(len(mesh.node_ids), dtype=float)
    for row in rows:
        node_id = str(row.get("node_id", "") or "")
        if node_id in mesh.node_index:
            values[mesh.node_index[node_id]] = float(row.get("pore_pressure", 0.0) or 0.0)
    return values


def _stage_reactions_from_csv(mesh: Mesh2D, stage_dir: Path) -> tuple[np.ndarray, dict[int, float]]:
    rows = _read_csv_dict_rows(stage_dir / "reactions.csv")
    dof_name_to_offset = {name: index for index, name in enumerate(DOF_NAMES)}
    constrained: dict[int, float] = {}
    reactions: dict[int, float] = {}
    for row in rows:
        node_id = str(row.get("node_id", "") or "")
        dof_name = str(row.get("dof", "") or "")
        if node_id not in mesh.node_index or dof_name not in dof_name_to_offset:
            continue
        dof = 2 * mesh.node_index[node_id] + dof_name_to_offset[dof_name]
        constrained[dof] = float(row.get("constrained_value", 0.0) or 0.0)
        reactions[dof] = float(row.get("reaction", 0.0) or 0.0)
    ndof = max(len(mesh.node_ids) * 2, max(reactions.keys(), default=-1) + 1, max(constrained.keys(), default=-1) + 1)
    out = np.zeros(ndof, dtype=float)
    for dof, value in reactions.items():
        out[dof] = value
    return out, constrained


def load_run_result_from_artifacts(output_dir: str | Path, *, input_path: str | Path | None = None) -> SolveResult2D:
    """Reconstruct enough run state from CSV/summary artifacts for deferred reports."""

    out = Path(output_dir)
    summary_path = out / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json for deferred report generation does not exist: {summary_path}")
    resolved_input = Path(input_path) if input_path is not None else _default_run_input_path(out)
    if resolved_input is None:
        raise FileNotFoundError(f"input YAML for deferred report generation was not found near {out}")
    cfg = _read_run_input_config(resolved_input)

    from .fem2d_config import plane_strain_materials
    from .fem2d_mesh import interfaces_from_config, mesh_from_config
    from .fem2d_structural import structural_elements_from_config

    mesh = mesh_from_config(cfg)
    materials = plane_strain_materials(cfg)
    interfaces = interfaces_from_config(cfg, mesh)
    structural_elements = structural_elements_from_config(cfg, mesh)
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    raw_stages = summary_data.get("stages", [])
    stages = raw_stages if isinstance(raw_stages, list) else []
    stage_results: list[StageResult2D] = []
    for index, raw_stage in enumerate(stages, start=1):
        raw = raw_stage if isinstance(raw_stage, Mapping) else {}
        stage_dir = _stage_dir_from_summary(out, raw, index)
        displacements = _stage_displacements_from_csv(mesh, stage_dir)
        reactions, constrained = _stage_reactions_from_csv(mesh, stage_dir)
        element_rows = _read_csv_dict_rows(stage_dir / "element_stress.csv")
        active_elements = [
            str(row.get("element_id"))
            for row in element_rows
            if str(row.get("element_id", "") or "") and float(row.get("active", 1.0) or 0.0) > 0.0
        ]
        if not active_elements:
            active_elements = [element.id for element in mesh.elements if element.active]
        stage_results.append(
            StageResult2D(
                name=str(raw.get("name", f"Stage-{index}") or f"Stage-{index}"),
                displacements=displacements,
                reactions=reactions,
                element_results=element_rows,
                constrained_dofs=constrained,
                active_elements=active_elements,
                solver_info=dict(raw.get("solver", {})) if isinstance(raw.get("solver", {}), Mapping) else {},
                pore_pressure=_stage_pore_pressure_from_csv(mesh, stage_dir),
                time=float(raw.get("time", 0.0) or 0.0),
                interface_results=_read_csv_dict_rows(stage_dir / "interface_state.csv"),
                structural_results=_read_csv_dict_rows(stage_dir / "structural_state.csv"),
                integration_point_results=_read_csv_dict_rows(stage_dir / "integration_point_stress.csv"),
                output_dir=stage_dir,
            )
        )
    warnings = [str(item) for item in summary_data.get("warnings", [])] if isinstance(summary_data.get("warnings", []), list) else []
    result = SolveResult2D(
        mesh=mesh,
        materials=materials,
        stages=stage_results,
        output_dir=out,
        interfaces=interfaces,
        structural_elements=structural_elements,
        warnings=warnings,
        input_config=cfg,
    )
    return result


def write_run_summary(result: SolveResult2D) -> None:
    analysis_name = "axisymmetric_static" if any(stage.solver_info.get("geometry") == "axisymmetric" for stage in result.stages) else "plane_strain_static"
    run_io_entries: dict[str, float] = {}

    def _timed_output(key: str, func: Any) -> Any:
        started = time.perf_counter()
        value = func()
        run_io_entries[key] = max(time.perf_counter() - started, 0.0)
        return value

    data = _run_summary_data(result, analysis_name)
    log_lines = _run_log_lines(result, analysis_name)
    output_policy = _run_output_policy(result)
    data["output_generation"] = {
        "schema": "geofem.output_generation_policy.v1",
        **output_policy,
    }
    result.output_dir.mkdir(parents=True, exist_ok=True)
    report_html = result.output_dir / "calculation_report.html"
    report_pdf = result.output_dir / "calculation_report.pdf"
    report_manifest = result.output_dir / "calculation_report_manifest.json"
    input_snapshot = result.output_dir / "calculation_report_input_snapshot.json"
    data["report"] = str(report_html)
    data["report_pdf"] = str(report_pdf)
    data["report_manifest"] = str(report_manifest)
    data["report_input_snapshot"] = str(input_snapshot)
    public_profile_path = _timed_output("public_profile_summary_elapsed_seconds", lambda: write_public_profile_summary(result, result.output_dir / "geofeas_public_profile.json"))
    if public_profile_path is not None:
        data["geofeas_public_profile"] = str(public_profile_path)
    output_conditions_path = _timed_output("public_output_conditions_elapsed_seconds", lambda: write_public_output_conditions(result, result.output_dir / "geofeas_public_output_conditions.json"))
    if output_conditions_path is not None:
        data["geofeas_public_output_conditions"] = str(output_conditions_path)
    mesh_quality_paths = _timed_output("mesh_quality_report_elapsed_seconds", lambda: write_mesh_quality_report(result.mesh, result.output_dir, result.input_config if isinstance(result.input_config, Mapping) else {}))
    data["mesh_quality"] = mesh_quality_paths
    material_paths = _timed_output("material_report_elapsed_seconds", lambda: write_material_reports(result.materials, result.output_dir, input_config=result.input_config if isinstance(result.input_config, Mapping) else {}))
    data["material_models"] = material_paths
    analysis_log_paths = _timed_output("analysis_log_elapsed_seconds", lambda: write_structured_analysis_log(result, result.output_dir))
    data["analysis_log"] = analysis_log_paths
    performance_paths = _timed_output("performance_summary_elapsed_seconds", lambda: write_performance_summary(result, result.output_dir))
    data["performance"] = performance_paths
    reliability_paths = _timed_output("reliability_summary_elapsed_seconds", lambda: write_reliability_summary_reports(result, result.output_dir))
    data["reliability_summary"] = reliability_paths
    if output_policy["result_view"]:
        result_view_paths = _timed_output("result_view_elapsed_seconds", lambda: write_result_view_index(result, result.output_dir))
    else:
        result_view_paths = _deferred_artifact(
            {
                "json": result.output_dir / "result_view_index.json",
                "csv": result.output_dir / "result_view_index.csv",
                "html": result.output_dir / "result_view_index.html",
            },
            reason="lazy_report_generation",
        )
    data["result_view_index"] = result_view_paths
    large_model_paths = _timed_output("large_model_operations_elapsed_seconds", lambda: write_large_model_operation_artifacts(result, result.output_dir))
    data["large_model_operations"] = large_model_paths
    performance_kpi_paths = _timed_output(
        "performance_kpi_elapsed_seconds",
        lambda: write_performance_kpi_reports(
            result,
            result.output_dir,
            artifacts={
                "mesh_quality": mesh_quality_paths,
                "material_models": material_paths,
                "analysis_log": analysis_log_paths,
                "performance": performance_paths,
                "reliability_summary": reliability_paths,
                "result_view_index": result_view_paths,
                "large_model_operations": large_model_paths,
            },
        ),
    )
    data["performance_kpis"] = performance_kpi_paths
    standard_report_artifacts = {
        "mesh_quality": mesh_quality_paths,
        "material_models": material_paths,
        "analysis_log": analysis_log_paths,
        "performance": performance_paths,
        "performance_kpis": performance_kpi_paths,
        "reliability_summary": reliability_paths,
        "result_view_index": result_view_paths,
        "large_model_operations": large_model_paths,
    }
    if output_policy["standard_report"]:
        standard_report_paths = _timed_output(
            "standard_report_elapsed_seconds",
            lambda: write_standard_report_bundle(
                result,
                result.output_dir,
                artifacts=standard_report_artifacts,
            ),
        )
    else:
        standard_report_paths = _deferred_artifact(
            {
                "data_json": result.output_dir / "standard_report_data.json",
                "csv": result.output_dir / "standard_report_sections.csv",
                "html": result.output_dir / "standard_report.html",
                "pdf": result.output_dir / "standard_report.pdf",
            },
            reason="lazy_report_generation",
        )
    data["standard_report"] = standard_report_paths
    _timed_output("load_combinations_csv_elapsed_seconds", lambda: write_load_combination_csv(result, result.output_dir / "load_combinations.csv"))
    _timed_output("post_case_comparison_csv_elapsed_seconds", lambda: write_post_case_comparison_csv(result, result.output_dir / "post_case_comparison.csv"))
    if output_policy["calculation_report"]:
        _timed_output("calculation_report_html_elapsed_seconds", lambda: write_calculation_report(result, report_html, summary_data=data, log_lines=log_lines, analysis_name=analysis_name))
        _timed_output("calculation_report_pdf_elapsed_seconds", lambda: write_calculation_report_pdf(result, report_pdf, summary_data=data, log_lines=log_lines, analysis_name=analysis_name))
        _timed_output(
            "report_manifest_elapsed_seconds",
            lambda: write_report_reproducibility_manifest(
                result,
                report_manifest,
                summary_data=data,
                log_lines=log_lines,
                analysis_name=analysis_name,
                html_report=report_html,
                pdf_report=report_pdf,
                input_snapshot=input_snapshot,
            ),
        )
    else:
        data["calculation_report"] = _deferred_artifact(
            {
                "html": report_html,
                "pdf": report_pdf,
                "manifest": report_manifest,
                "input_snapshot": input_snapshot,
            },
            reason="lazy_report_generation",
        )
    data["io_profile"] = _run_io_profile_payload(result, run_io_entries)
    setattr(result, "run_io_profile", data["io_profile"])
    _timed_output("summary_json_elapsed_seconds", lambda: (result.output_dir / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"))
    if output_policy["write_log"]:
        _timed_output("run_log_elapsed_seconds", lambda: (result.output_dir / "run.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8"))
    data["io_profile"] = _run_io_profile_payload(result, run_io_entries)
    setattr(result, "run_io_profile", data["io_profile"])
    (result.output_dir / "summary.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_performance_summary(result, result.output_dir)


def write_deferred_run_artifacts(
    result: SolveResult2D,
    *,
    include: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Materialize heavy run-level artifacts that were deferred by lazy output."""

    requested = {str(item) for item in include} if include is not None else {"result_view_index", "standard_report", "calculation_report"}
    out = result.output_dir
    out.mkdir(parents=True, exist_ok=True)
    summary_path = out / "summary.json"
    if summary_path.exists():
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        analysis_name = "axisymmetric_static" if any(stage.solver_info.get("geometry") == "axisymmetric" for stage in result.stages) else "plane_strain_static"
        data = _run_summary_data(result, analysis_name)
    analysis_name = str(data.get("analysis", ""))
    if not analysis_name:
        analysis_name = "axisymmetric_static" if any(stage.solver_info.get("geometry") == "axisymmetric" for stage in result.stages) else "plane_strain_static"
    log_lines = _run_log_lines(result, analysis_name)
    generated: dict[str, Any] = {}
    timings: dict[str, float] = {}

    def _timed(key: str, func: Any) -> Any:
        started = time.perf_counter()
        value = func()
        timings[key] = max(time.perf_counter() - started, 0.0)
        return value

    if "result_view_index" in requested:
        paths = _timed("result_view_elapsed_seconds", lambda: write_result_view_index(result, out))
        data["result_view_index"] = paths
        generated["result_view_index"] = paths

    standard_artifacts = {
        key: data.get(key, {})
        for key in (
            "mesh_quality",
            "material_models",
            "analysis_log",
            "performance",
            "performance_kpis",
            "reliability_summary",
            "result_view_index",
            "large_model_operations",
        )
    }
    if "standard_report" in requested:
        paths = _timed("standard_report_elapsed_seconds", lambda: write_standard_report_bundle(result, out, artifacts=standard_artifacts))
        data["standard_report"] = paths
        generated["standard_report"] = paths

    if "calculation_report" in requested:
        report_html = Path(data.get("report", out / "calculation_report.html"))
        report_pdf = Path(data.get("report_pdf", out / "calculation_report.pdf"))
        report_manifest = Path(data.get("report_manifest", out / "calculation_report_manifest.json"))
        input_snapshot = Path(data.get("report_input_snapshot", out / "calculation_report_input_snapshot.json"))
        _timed("calculation_report_html_elapsed_seconds", lambda: write_calculation_report(result, report_html, summary_data=data, log_lines=log_lines, analysis_name=analysis_name))
        _timed("calculation_report_pdf_elapsed_seconds", lambda: write_calculation_report_pdf(result, report_pdf, summary_data=data, log_lines=log_lines, analysis_name=analysis_name))
        _timed(
            "report_manifest_elapsed_seconds",
            lambda: write_report_reproducibility_manifest(
                result,
                report_manifest,
                summary_data=data,
                log_lines=log_lines,
                analysis_name=analysis_name,
                html_report=report_html,
                pdf_report=report_pdf,
                input_snapshot=input_snapshot,
            ),
        )
        paths = {
            "generated": True,
            "deferred": False,
            "html": str(report_html),
            "pdf": str(report_pdf),
            "manifest": str(report_manifest),
            "input_snapshot": str(input_snapshot),
        }
        data["calculation_report"] = paths
        generated["calculation_report"] = paths

    output_generation = data.get("output_generation", {})
    if not isinstance(output_generation, dict):
        output_generation = {}
    output_generation.update(
        {
            "deferred_artifacts_materialized": sorted(generated),
            "deferred_generation_elapsed_seconds": float(sum(timings.values())),
        }
    )
    data["output_generation"] = output_generation
    io_profile = data.get("io_profile", {})
    if not isinstance(io_profile, dict):
        io_profile = {}
    entries = dict(io_profile.get("entries", {})) if isinstance(io_profile.get("entries", {}), Mapping) else {}
    entries.update({f"deferred_{key}": value for key, value in timings.items()})
    io_profile["entries"] = entries
    io_profile["deferred_generation_elapsed_seconds"] = float(sum(timings.values()))
    data["io_profile"] = io_profile
    summary_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"generated": generated, "timings": timings, "summary": str(summary_path)}


def write_deferred_run_artifacts_from_files(
    output_dir: str | Path,
    *,
    input_path: str | Path | None = None,
    include: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Materialize deferred run-level artifacts from persisted solver outputs."""

    result = load_run_result_from_artifacts(output_dir, input_path=input_path)
    return write_deferred_run_artifacts(result, include=include)


def write_calculation_report(
    result: SolveResult2D,
    path: Path,
    *,
    summary_data: Mapping[str, Any] | None = None,
    log_lines: list[str] | None = None,
    analysis_name: str | None = None,
) -> None:
    """Write a single printable HTML calculation report for the whole run."""

    analysis_name = analysis_name or ("axisymmetric_static" if any(stage.solver_info.get("geometry") == "axisymmetric" for stage in result.stages) else "plane_strain_static")
    summary_data = dict(summary_data or _run_summary_data(result, analysis_name))
    log_lines = list(log_lines or _run_log_lines(result, analysis_name))
    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    template = _report_template(cfg, analysis_name)
    stage_specs = _stage_specs_from_config(cfg, result)
    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    node_count = len(result.mesh.node_ids)
    element_count = len(result.mesh.elements)
    interface_count = len(result.interfaces)

    parts: list[str] = [
        "<!doctype html>",
        "<html lang=\"ja\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        f"<title>{_h(template['title'])}</title>",
        "<style>",
        _report_css(),
        "</style>",
        "</head>",
        "<body>",
        "<div class=\"report-feature-flags\" data-report-feature=\"direct-pdf title-block chapter-numbering corporate-template figure-numbering case-comparison judgement-table reproducibility-freeze\"></div>",
        "<header class=\"cover\">",
        "<div>",
        "<p class=\"eyebrow\">GeoFEM 2D Calculation Report</p>",
        f"<h1>{_h(template['title'])}</h1>",
        f"<p>{_h(template['subtitle'])}</p>",
        f"<p>解析種別: <strong>{_h(analysis_name)}</strong></p>",
        f"<p>作成日時: {_h(generated)}</p>",
        f"<p class=\"template-id\">Template: {_h(template['template_id'])} rev {_h(template['template_revision'])}</p>",
        "</div>",
        "<table class=\"cover-summary\"><tbody>",
        f"<tr><th>節点数</th><td>{node_count}</td></tr>",
        f"<tr><th>要素数</th><td>{element_count}</td></tr>",
        f"<tr><th>界面数</th><td>{interface_count}</td></tr>",
        f"<tr><th>ステージ数</th><td>{len(result.stages)}</td></tr>",
        "</tbody></table>",
        "</header>",
        _title_block_html(template, generated),
        "<nav class=\"toc\"><h2>目次</h2><ol>",
        "<li>入力条件</li><li>材料表</li><li>境界/荷重図</li><li>荷重組合せ/ケース比較</li><li>ステージ一覧</li><li>解析結果図</li><li>判定表</li><li>ログ</li><li>再現条件</li>",
        "</ol></nav>",
    ]

    parts.extend(_input_conditions_section(result, cfg, summary_data))
    parts.extend(_materials_section(result))
    parts.extend(_boundary_load_section(result, stage_specs))
    parts.extend(_load_combination_section(result, cfg))
    parts.extend(_stage_list_section(result, stage_specs))
    parts.extend(_results_section(result))
    parts.extend(_judgement_section(result))
    parts.extend(_srm_trials_section(result))
    parts.extend(_log_section(log_lines))
    parts.extend(_reproducibility_section(result, cfg, summary_data))
    parts.extend(["</body>", "</html>"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def write_calculation_report_pdf(
    result: SolveResult2D,
    path: Path,
    *,
    summary_data: Mapping[str, Any] | None = None,
    log_lines: list[str] | None = None,
    analysis_name: str | None = None,
) -> None:
    """Write a direct, dependency-free PDF calculation report for the run."""

    analysis_name = analysis_name or ("axisymmetric_static" if any(stage.solver_info.get("geometry") == "axisymmetric" for stage in result.stages) else "plane_strain_static")
    summary_data = dict(summary_data or _run_summary_data(result, analysis_name))
    log_lines = list(log_lines or _run_log_lines(result, analysis_name))
    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    template = _report_template(cfg, analysis_name)
    lines = _calculation_report_pdf_lines(result, cfg, summary_data, log_lines, analysis_name, template)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_pdf(path, lines, title=str(template["title"]))


def write_report_reproducibility_manifest(
    result: SolveResult2D,
    path: Path,
    *,
    summary_data: Mapping[str, Any],
    log_lines: list[str],
    analysis_name: str,
    html_report: Path,
    pdf_report: Path,
    input_snapshot: Path,
) -> None:
    """Freeze report inputs, template metadata, checksums, and output index."""

    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    safe_cfg = _json_safe(cfg)
    template = _report_template(cfg, analysis_name)
    geofeas_public = public_profile_summary(result)
    input_snapshot.parent.mkdir(parents=True, exist_ok=True)
    input_snapshot.write_text(json.dumps(safe_cfg, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    features = [
        "direct_pdf",
        "title_block",
        "chapter_numbering",
        "corporate_template",
        "figure_table_numbering",
        "case_comparison",
        "judgement_table",
        "reproducibility_freeze",
    ]
    if geofeas_public:
        features.append("geofeas_public_profile")
    if summary_data.get("geofeas_public_output_conditions"):
        features.append("geofeas_public_output_conditions")
    manifest = {
        "schema": "geofem.calculation_report_manifest.v1",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "software": {"name": "GeoFEM 2D", "python": platform.python_version(), "platform": platform.platform()},
        "features": features,
        "template": template,
        "analysis": analysis_name,
        "geofeas_public": geofeas_public,
        "reports": {
            "html": str(html_report),
            "pdf": str(pdf_report),
            "input_snapshot": str(input_snapshot),
            "geofeas_public_output_conditions": str(summary_data.get("geofeas_public_output_conditions", "")),
        },
        "reproducibility": {
            "frozen": True,
            "input_sha256": _stable_json_hash(safe_cfg),
            "summary_sha256": _stable_json_hash(summary_data),
            "log_sha256": hashlib.sha256("\n".join(log_lines).encode("utf-8")).hexdigest(),
            "html_sha256": _file_sha256(html_report),
            "pdf_sha256": _file_sha256(pdf_report),
        },
        "case_comparison": _post_case_comparison_rows(result),
        "judgement": _judgement_records(result),
        "output_index": _report_output_index(result.output_dir),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _report_template(cfg: Mapping[str, Any], analysis_name: str) -> dict[str, str]:
    report_cfg = cfg.get("report", cfg.get("calculation_report", {}))
    report = dict(report_cfg) if isinstance(report_cfg, Mapping) else {}
    raw_template = report.get("template", cfg.get("report_template", {}))
    template = dict(raw_template) if isinstance(raw_template, Mapping) else {"id": str(raw_template or "")}
    template_id = str(template.get("id") or report.get("template_id") or "corporate_a4_standard")
    defaults = _default_report_template(template_id)
    merged: dict[str, Any] = {**defaults, **template, **report}
    merged.setdefault("analysis_name", analysis_name)
    return {key: str(value if value is not None else "") for key, value in merged.items()}


def _default_report_template(template_id: str) -> dict[str, str]:
    templates = {
        "corporate_a4_standard": {
            "template_id": "corporate_a4_standard",
            "template_name": "GeoFEM corporate A4 standard",
            "template_revision": "2026.05",
            "title": "GeoFEM 2D 計算書",
            "subtitle": "解析条件・結果・判定・再現条件",
            "company": "Internal Engineering Department",
            "client": "-",
            "project": "-",
            "report_no": "-",
            "revision": "0",
            "prepared_by": "-",
            "checked_by": "-",
            "approved_by": "-",
            "page_style": "A4 portrait titleblock",
        },
        "geofeas_review": {
            "template_id": "geofeas_review",
            "template_name": "GeoFEAS-style review report",
            "template_revision": "2026.05",
            "title": "GeoFEAS風 2D 計算書",
            "subtitle": "入力条件・ケース比較・判定表",
            "company": "Internal Engineering Department",
            "client": "-",
            "project": "-",
            "report_no": "-",
            "revision": "0",
            "prepared_by": "-",
            "checked_by": "-",
            "approved_by": "-",
            "page_style": "A4 portrait titleblock",
        },
    }
    default = templates.get(template_id, templates["corporate_a4_standard"]).copy()
    default["template_id"] = template_id if template_id else default["template_id"]
    return default


def _title_block_html(template: Mapping[str, str], generated: str) -> str:
    rows = [
        ("Project", template.get("project", "")),
        ("Client", template.get("client", "")),
        ("Company", template.get("company", "")),
        ("Report No.", template.get("report_no", "")),
        ("Revision", template.get("revision", "")),
        ("Prepared by", template.get("prepared_by", "")),
        ("Checked by", template.get("checked_by", "")),
        ("Approved by", template.get("approved_by", "")),
        ("Generated at", generated),
        ("Template", f"{template.get('template_name', '')} ({template.get('template_id', '')}, rev {template.get('template_revision', '')})"),
    ]
    body = "".join(f"<tr><th>{_h(key)}</th><td>{_h(value)}</td></tr>" for key, value in rows)
    return (
        "<section class=\"title-block-section\" data-template-id=\""
        + _h(template.get("template_id", ""))
        + "\"><h2>0. Title Block</h2><table class=\"title-block\"><caption>Table 1. Title Block</caption><tbody>"
        + body
        + "</tbody></table></section>"
    )


def _calculation_report_pdf_lines(
    result: SolveResult2D,
    cfg: Mapping[str, Any],
    summary_data: Mapping[str, Any],
    log_lines: list[str],
    analysis_name: str,
    template: Mapping[str, str],
) -> list[str]:
    input_hash = _stable_json_hash(_json_safe(cfg))
    lines = [
        str(template.get("title", "GeoFEM 2D Calculation Report")),
        str(template.get("subtitle", "")),
        "",
        "Table 1. Title Block",
        f"Project: {template.get('project', '-')}",
        f"Client: {template.get('client', '-')}",
        f"Company: {template.get('company', '-')}",
        f"Report No.: {template.get('report_no', '-')}  Revision: {template.get('revision', '-')}",
        f"Prepared: {template.get('prepared_by', '-')}  Checked: {template.get('checked_by', '-')}  Approved: {template.get('approved_by', '-')}",
        f"Template: {template.get('template_id', '-')} rev {template.get('template_revision', '-')}",
        "",
        "1. Input Conditions",
        f"Dimension: {summary_data.get('dimension', '2D')}",
        f"Analysis: {analysis_name}",
        f"Nodes: {len(result.mesh.node_ids)}  Elements: {len(result.mesh.elements)}  Interfaces: {len(result.interfaces)}",
        "",
        "2. Materials",
    ]
    for name, mat in sorted(result.materials.items()):
        lines.append(f"Material {name}: model={mat.model}, E={_fmt(mat.E)}, nu={_fmt(mat.nu)}, gamma={_fmt(mat.gamma)}")
    lines.extend(["", "3. Boundary / Load Figures"])
    for idx, stage in enumerate(result.stages, start=1):
        lines.append(f"Figure {idx}. Boundary and load diagram - {stage.name}")
    lines.extend(["", "4. Load Combination / Case Comparison"])
    combo_rows = _load_combination_rows(cfg)
    if combo_rows:
        for row in combo_rows:
            lines.append(f"Combination {row.get('combination')}: case={row.get('case')} factor={_fmt(row.get('factor'))} type={row.get('case_type')} active={row.get('active')}")
    for row in _post_case_comparison_rows(result):
        lines.append(
            "Case comparison: "
            f"stage={row.get('stage')} combo={row.get('load_combination')} "
            f"max_u={_fmt(row.get('max_displacement'))} settlement={_fmt(row.get('max_settlement'))} "
            f"pmax={_fmt(row.get('max_pore_pressure'))}"
        )
    lines.extend(["", "5. Stage List"])
    for idx, stage in enumerate(result.stages, start=1):
        lines.append(f"{idx}. {stage.name}: active_elements={len(stage.active_elements)}, solver={stage.solver_info.get('method', '-')}, time={_fmt(stage.time)}")
    lines.extend(["", "6. Result Figures"])
    for idx, stage in enumerate(result.stages, start=len(result.stages) + 1):
        lines.append(f"Figure {idx}. Result contour and deformation overlay - {stage.name}")
    lines.extend(["", "7. Judgement Table"])
    for row in _judgement_records(result):
        lines.append(f"{row['status']}: {row['item']} / {row['target']} / {row['basis']}")
    srm_summaries = _srm_summary_records(result)
    if srm_summaries:
        lines.extend(["", "7a. SRM FOS Trial Results"])
        for row in srm_summaries:
            lines.append(
                "SRM summary: "
                f"stage={row['stage']} {row['factor_of_safety_display']} "
                f"stable={_fmt(row['stable_factor'])} failed={_fmt(row['failed_factor'])} "
                f"status={row['factor_of_safety_status']} confidence={row['factor_of_safety_confidence']} "
                f"trials={row['trial_count']} search={row['search_mode']}"
            )
        for row in _srm_trial_records(result):
            lines.append(
                "SRM trial: "
                f"stage={row['stage']} factor={_fmt(row['factor'])} "
                f"ok={row['ok']} converged={row['converged']} "
                f"plastic_ratio={_fmt(row['plastic_ratio'])} reason={row['failure_reason']} "
                f"status={row.get('trial_status', '')} last_load={_fmt(row.get('last_accepted_load_factor'))} "
                f"last_pr={_fmt(row.get('last_accepted_plastic_ratio'))} pr_delta={_fmt(row.get('plastic_ratio_delta'))} "
                f"max_eqp={_fmt(row.get('max_equivalent_plastic_strain'))} cluster={row.get('connected_plastic_cluster_size', '')} "
                f"cutbacks={row.get('cutback_count', '')}"
            )
    lines.extend(["", "8. Log"])
    lines.extend(log_lines[:40])
    lines.extend(
        [
            "",
            "9. Reproducibility Freeze",
            f"Input SHA256: {input_hash}",
            f"Summary SHA256: {_stable_json_hash(summary_data)}",
            "Frozen files: calculation_report.html, calculation_report.pdf, calculation_report_manifest.json, calculation_report_input_snapshot.json",
        ]
    )
    return _wrap_pdf_lines(lines)


def _wrap_pdf_lines(lines: list[str], width: int = 94) -> list[str]:
    wrapped: list[str] = []
    for line in lines:
        text = str(line)
        if not text:
            wrapped.append("")
            continue
        while len(text) > width:
            wrapped.append(text[:width])
            text = "  " + text[width:]
        wrapped.append(text)
    return wrapped


def _stable_json_hash(value: Any) -> str:
    payload = json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _report_output_index(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for path in sorted(root.rglob("*")):
        if path.is_file():
            try:
                label = str(path.relative_to(root)).replace("\\", "/")
            except ValueError:
                label = str(path)
            rows.append({"path": label, "bytes": path.stat().st_size, "sha256": _file_sha256(path)})
    return rows


def _judgement_records(result: SolveResult2D) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for stage in result.stages:
        solver = stage.solver_info
        converged = solver.get("converged", True)
        rows.append({"item": "convergence", "target": stage.name, "status": "OK" if converged is not False else "NG", "basis": f"method={solver.get('method', '-')}, iterations={solver.get('iterations', '-')}"})
        residual = solver.get("residual_norm")
        if residual is not None:
            ok = math.isfinite(float(residual))
            rows.append({"item": "residual", "target": stage.name, "status": "OK" if ok else "NG", "basis": f"residual_norm={_fmt(residual)}"})
        max_u = _max_displacement(result.mesh, stage)
        rows.append({"item": "displacement", "target": stage.name, "status": "OK" if math.isfinite(max_u) else "NG", "basis": f"max_displacement={_fmt(max_u)}"})
        if isinstance(solver.get("srm"), Mapping):
            srm = solver["srm"]
            rows.append(
                {
                    "item": "safety_factor",
                    "target": stage.name,
                    "status": srm_safety_verdict(srm),
                    "basis": srm_fos_display(srm, locale="en"),
                }
            )
        if stage.pore_pressure is not None:
            pmax = float(np.max(stage.pore_pressure))
            rows.append({"item": "pore_pressure", "target": stage.name, "status": "OK" if math.isfinite(pmax) else "NG", "basis": f"max_pore_pressure={_fmt(pmax)}"})
        if stage.interface_results:
            slip = max(float(row.get("slip_abs", 0.0)) for row in stage.interface_results)
            slip_points = sum(1 for row in stage.interface_results if row.get("state") == "slip")
            rows.append({"item": "interface", "target": stage.name, "status": "INFO", "basis": f"max_slip={_fmt(slip)}, slip_points={slip_points}"})
    for warning in result.warnings:
        rows.append({"item": "warning", "target": "run", "status": "WARN", "basis": str(warning)})
    if not rows:
        rows.append({"item": "overall", "target": "run", "status": "OK", "basis": "no diagnostics"})
    return rows


def write_load_combination_csv(result: SolveResult2D, path: Path) -> None:
    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    rows = _load_combination_rows(cfg)
    if not rows:
        return
    fieldnames = ["combination", "case", "factor", "case_type", "case_scale", "active", "description", "standard", "source", "revision", "coverage", "clause"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_post_case_comparison_csv(result: SolveResult2D, path: Path) -> None:
    rows = _post_case_comparison_rows(result)
    if not rows:
        return
    fieldnames = [
        "stage",
        "geofeas_workflow",
        "load_combination",
        "seismic_kh",
        "seismic_kv",
        "max_displacement",
        "max_settlement",
        "max_pore_pressure",
        "relative_stage",
        "liquefaction_min_fl",
        "liquefaction_max_ru",
        "structural_axial_force_max",
        "structural_spring_reaction_max",
        "interface_slip_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_summary_data(result: SolveResult2D, analysis_name: str) -> dict[str, Any]:
    return {
        "dimension": "2D",
        "analysis": analysis_name,
        "node_count": len(result.mesh.node_ids),
        "element_count": len(result.mesh.elements),
        "interface_count": len(result.interfaces),
        "structural_element_count": len(result.structural_elements),
        "report": str(result.output_dir / "calculation_report.html"),
        "stages": [
            {
                "name": stage.name,
                "output_dir": str(stage.output_dir) if stage.output_dir else None,
                "active_element_count": len(stage.active_elements),
                "solver": stage.solver_info,
                "time": stage.time,
                "max_pore_pressure": None if stage.pore_pressure is None else float(np.max(stage.pore_pressure)),
                "interface_slip_max": None if not stage.interface_results else float(max(row.get("slip_abs", 0.0) for row in stage.interface_results)),
                "interface_slip_points": 0 if not stage.interface_results else sum(1 for row in stage.interface_results if row.get("state") == "slip"),
                "active_structural_element_count": 0 if not stage.structural_results else sum(1 for row in stage.structural_results if float(row.get("active", 0.0) or 0.0) > 0.0),
                "structural_axial_force_max": None if not stage.structural_results else float(max(abs(float(row.get("axial_force", 0.0) or 0.0)) for row in stage.structural_results)),
                "structural_spring_reaction_max": None if not stage.structural_results else float(max(abs(float(row.get("spring_reaction", 0.0) or 0.0)) for row in stage.structural_results)),
                "max_displacement": _max_displacement(result.mesh, stage),
                "max_settlement": float(max((-stage.displacements[1::2]), default=0.0)),
            }
            for stage in result.stages
        ],
        "load_combinations": _load_combination_rows(result.input_config if isinstance(result.input_config, Mapping) else {}),
        "post_case_comparison": _post_case_comparison_rows(result),
        "geofeas_public": public_profile_summary(result),
        "warnings": result.warnings,
    }


def _stage_srm_info(stage: StageResult2D) -> Mapping[str, Any]:
    solver = stage.solver_info if isinstance(stage.solver_info, Mapping) else {}
    srm = solver.get("srm")
    return srm if isinstance(srm, Mapping) else {}


def _srm_summary_records(result: SolveResult2D) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in result.stages:
        srm = _stage_srm_info(stage)
        if not srm:
            continue
        trials = srm.get("trials", [])
        rows.append(
            {
                "stage": stage.name,
                "factor_of_safety": srm.get("factor_of_safety"),
                "stable_factor": srm.get("stable_factor"),
                "failed_factor": srm.get("failed_factor"),
                "search_mode": srm.get("search_mode", ""),
                "factor_of_safety_status": srm_result_status(srm),
                "factor_of_safety_confidence": srm_result_confidence(srm),
                "factor_of_safety_confirmed": srm_fos_is_confirmed(srm),
                "factor_of_safety_display": srm_fos_display(srm, locale="en"),
                "factor_of_safety_interval": srm.get("factor_of_safety_interval"),
                "factor_of_safety_boundary_certified": srm.get(
                    "factor_of_safety_boundary_certified", ""
                ),
                "factor_of_safety_tolerance_met": srm.get(
                    "factor_of_safety_tolerance_met", ""
                ),
                "factor_of_safety_certified": srm.get("factor_of_safety_certified", ""),
                "factor_of_safety_value_kind": srm.get("factor_of_safety_value_kind", ""),
                "boundary_quality": srm.get("boundary_quality", ""),
                "boundary_verified": srm.get("boundary_verified", ""),
                "material_fallback_verification_required": srm.get(
                    "material_fallback_verification_required", ""
                ),
                "trial_count": len(trials) if isinstance(trials, list) else 0,
            }
        )
    return rows


def _srm_trial_records(result: SolveResult2D) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in result.stages:
        srm = _stage_srm_info(stage)
        trials = srm.get("trials", []) if srm else []
        if not isinstance(trials, list):
            continue
        for index, raw in enumerate(trials, start=1):
            if not isinstance(raw, Mapping):
                continue
            rows.append(
                {
                    "stage": stage.name,
                    "index": index,
                    "factor": raw.get("factor"),
                    "converged": raw.get("converged", ""),
                    "plastic_ratio": raw.get("plastic_ratio"),
                    "ok": raw.get("ok", ""),
                    "failure_reason": raw.get("failure_reason") or raw.get("error") or "",
                    "auto_decision": raw.get("auto_decision", ""),
                    "auto_failure_class": raw.get("auto_failure_class", ""),
                    "auto_failure_score": raw.get("auto_failure_score", ""),
                    "auto_decision_reason": raw.get("auto_decision_reason", ""),
                    "auto_trial_action": raw.get("auto_trial_action", ""),
                    "auto_retry": raw.get("auto_retry", ""),
                    "auto_retry_of": raw.get("auto_retry_of", ""),
                    "auto_retry_index": raw.get("auto_retry_index", ""),
                    "auto_retry_planned": raw.get("auto_retry_planned", ""),
                    "auto_retry_reason": raw.get("auto_retry_reason", ""),
                    "auto_retry_result": raw.get("auto_retry_result", ""),
                    "auto_superseded_by_retry": raw.get("auto_superseded_by_retry", ""),
                    "auto_cluster_fraction": raw.get("auto_cluster_fraction", ""),
                    "auto_last_accepted_strength_factor_estimate": raw.get("auto_last_accepted_strength_factor_estimate", ""),
                    "estimated_fos_from_last_load": raw.get("estimated_fos_from_last_load", ""),
                    "srm_trial_state": raw.get("srm_trial_state", ""),
                    "boundary_verification": raw.get("boundary_verification", ""),
                    "boundary_verification_reason": raw.get("boundary_verification_reason", ""),
                    "boundary_verification_of": raw.get("boundary_verification_of", ""),
                    "boundary_verification_result": raw.get("boundary_verification_result", ""),
                    "boundary_verification_superseded": raw.get("boundary_verification_superseded", ""),
                    "boundary_checkpoint_continuation_requested": raw.get(
                        "boundary_checkpoint_continuation_requested", ""
                    ),
                    "boundary_checkpoint_continuation_used": raw.get(
                        "boundary_checkpoint_continuation_used", ""
                    ),
                    "boundary_checkpoint_fallback_reason": raw.get(
                        "boundary_checkpoint_fallback_reason", ""
                    ),
                    "factor_tol_numerical_failure_rejected": raw.get(
                        "factor_tol_numerical_failure_rejected", ""
                    ),
                    "factor_tol_physical_failure_evidence": raw.get(
                        "factor_tol_physical_failure_evidence", ""
                    ),
                    "factor_tol_enforcement_reason": raw.get(
                        "factor_tol_enforcement_reason", ""
                    ),
                    "checkpoint_residual_prediction_enabled": raw.get(
                        "checkpoint_residual_prediction_enabled", ""
                    ),
                    "checkpoint_residual_prediction_reason": raw.get(
                        "checkpoint_residual_prediction_reason", ""
                    ),
                    "checkpoint_residual_prediction_sample_count": raw.get(
                        "checkpoint_residual_prediction_sample_count", ""
                    ),
                    "checkpoint_residual_prediction_ratio": raw.get(
                        "checkpoint_residual_prediction_ratio", ""
                    ),
                    "checkpoint_residual_prediction_extra_cutbacks": raw.get(
                        "checkpoint_residual_prediction_extra_cutbacks", ""
                    ),
                    "early_failure_stop": raw.get("early_failure_stop", ""),
                    "early_failure_policy": raw.get("early_failure_policy", ""),
                    "early_failure_class": raw.get("early_failure_class", ""),
                    "early_failure_score": raw.get("early_failure_score", ""),
                    "early_failure_score_threshold": raw.get("early_failure_score_threshold", ""),
                    "early_failure_reason": raw.get("early_failure_reason", ""),
                    "early_failure_cutback_ratio": raw.get("early_failure_cutback_ratio", ""),
                    "early_failure_effective_cutbacks": raw.get("early_failure_effective_cutbacks", ""),
                    "elapsed_seconds": raw.get("elapsed_seconds", ""),
                    "solver_elapsed_seconds": raw.get("solver_elapsed_seconds", ""),
                    "overhead_elapsed_seconds": raw.get("overhead_elapsed_seconds", ""),
                    "accounted_elapsed_seconds": raw.get("accounted_elapsed_seconds", ""),
                    "unattributed_elapsed_seconds": raw.get("unattributed_elapsed_seconds", ""),
                    "timing_coverage_ratio": raw.get("timing_coverage_ratio", ""),
                    "assembly_elapsed_seconds": raw.get("assembly_elapsed_seconds", ""),
                    "linear_solve_elapsed_seconds": raw.get("linear_solve_elapsed_seconds", ""),
                    "line_search_elapsed_seconds": raw.get("line_search_elapsed_seconds", ""),
                    "postprocess_elapsed_seconds": raw.get("postprocess_elapsed_seconds", ""),
                    "solver_cancel_requested": raw.get("solver_cancel_requested", ""),
                    "solver_cancel_checkpoint": raw.get("solver_cancel_checkpoint", ""),
                    "solver_cancel_scope": raw.get("solver_cancel_scope", ""),
                    "increment_checkpoint_available": raw.get(
                        "increment_checkpoint_available", ""
                    ),
                    "increment_checkpoint_load_factor": raw.get(
                        "increment_checkpoint_load_factor", ""
                    ),
                    "increment_checkpoint_accepted_steps": raw.get(
                        "increment_checkpoint_accepted_steps", ""
                    ),
                    "increment_checkpoint_cutbacks": raw.get(
                        "increment_checkpoint_cutbacks", ""
                    ),
                    "increment_checkpoint_continuation_requested": raw.get(
                        "increment_checkpoint_continuation_requested", ""
                    ),
                    "increment_checkpoint_continuation_used": raw.get(
                        "increment_checkpoint_continuation_used", ""
                    ),
                    "increment_checkpoint_fallback_reason": raw.get(
                        "increment_checkpoint_fallback_reason", ""
                    ),
                    "increment_checkpoint_source_load_factor": raw.get(
                        "increment_checkpoint_source_load_factor", ""
                    ),
                    "increment_checkpoint_resumed_accepted_steps": raw.get(
                        "increment_checkpoint_resumed_accepted_steps", ""
                    ),
                    "increment_checkpoint_resumed_cutbacks": raw.get(
                        "increment_checkpoint_resumed_cutbacks", ""
                    ),
                    "increment_checkpoint_reused_history_rows": raw.get(
                        "increment_checkpoint_reused_history_rows", ""
                    ),
                    "trial_status": raw.get("trial_status", ""),
                    "attempted_load_factor": raw.get("attempted_load_factor", ""),
                    "last_accepted_load_factor": raw.get("last_accepted_load_factor", ""),
                    "last_accepted_plastic_ratio": raw.get("last_accepted_plastic_ratio", ""),
                    "last_accepted_residual_norm": raw.get("last_accepted_residual_norm", ""),
                    "accepted_increment_count": raw.get("accepted_increment_count", ""),
                    "cutback_count": raw.get("cutback_count", ""),
                    "failed_step_size": raw.get("failed_step_size", ""),
                    "next_step_size": raw.get("next_step_size", ""),
                    "diagnostic_summary": raw.get("diagnostic_summary", ""),
                    "plastic_ratio_delta": raw.get("plastic_ratio_delta", ""),
                    "max_equivalent_plastic_strain": raw.get("max_equivalent_plastic_strain", ""),
                    "mean_equivalent_plastic_strain": raw.get("mean_equivalent_plastic_strain", ""),
                    "top_percentile_equivalent_plastic_strain": raw.get("top_percentile_equivalent_plastic_strain", ""),
                    "yielded_element_count": raw.get("yielded_element_count", ""),
                    "connected_plastic_cluster_size": raw.get("connected_plastic_cluster_size", ""),
                    "plastic_cluster_spans_boundary": raw.get("plastic_cluster_spans_boundary", ""),
                    "final_step_size": raw.get("final_step_size", ""),
                    "newton_iterations_total": raw.get("newton_iterations_total", ""),
                    "newton_iterations_max": raw.get("newton_iterations_max", ""),
                    "line_search_reductions_total": raw.get("line_search_reductions_total", ""),
                    "line_search_batch_calls_total": raw.get(
                        "line_search_batch_calls_total", ""
                    ),
                    "line_search_batch_candidates_total": raw.get(
                        "line_search_batch_candidates_total", ""
                    ),
                    "residual_norm_final": raw.get("residual_norm_final", ""),
                    "min_det_j": raw.get("min_det_j", ""),
                    "max_displacement_norm": raw.get("max_displacement_norm", ""),
                    "displacement_increment_norm": raw.get("displacement_increment_norm", ""),
                    "internal_external_work_ratio": raw.get("internal_external_work_ratio", ""),
                    "mc_numba_to_python_fallback_count": raw.get("mc_numba_to_python_fallback_count", ""),
                    "mc_numba_regularized_projection_count": raw.get(
                        "mc_numba_regularized_projection_count", ""
                    ),
                    "mc_regularized_projection_count": raw.get("mc_regularized_projection_count", ""),
                    "mc_apex_regularization_count": raw.get(
                        "mc_apex_regularization_count", ""
                    ),
                    "mc_associated_apex_projection_count": raw.get(
                        "mc_associated_apex_projection_count", ""
                    ),
                    "mc_legacy_bounded_projection_count": raw.get(
                        "mc_legacy_bounded_projection_count", ""
                    ),
                    "mc_regularization_method": raw.get(
                        "mc_regularization_method", ""
                    ),
                    "mc_configured_apex_policy_verified": raw.get(
                        "mc_configured_apex_policy_verified", ""
                    ),
                    "mc_base_nonassociated_flow_rule_verified": raw.get(
                        "mc_base_nonassociated_flow_rule_verified", ""
                    ),
                    "mc_constitutive_model_fidelity": raw.get(
                        "mc_constitutive_model_fidelity", ""
                    ),
                    "mc_regularized_projection_above_relaxed_tolerance_count": raw.get(
                        "mc_regularized_projection_above_relaxed_tolerance_count", ""
                    ),
                    "mc_regularized_projection_max_yield_violation": raw.get(
                        "mc_regularized_projection_max_yield_violation", ""
                    ),
                    "mc_regularized_projection_max_relative_yield_violation": raw.get(
                        "mc_regularized_projection_max_relative_yield_violation", ""
                    ),
                    "mc_regularized_projection_samples": raw.get("mc_regularized_projection_samples", ""),
                }
            )
    return rows


def _srm_trial_log_details(raw: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key, label in (
        ("auto_decision", "auto"),
        ("srm_trial_state", "state"),
        ("auto_trial_action", "auto_action"),
        ("auto_retry_index", "retry"),
        ("auto_retry_result", "retry_result"),
        ("auto_failure_score", "auto_score"),
        ("boundary_verification", "boundary_verify"),
        ("boundary_verification_result", "boundary_result"),
        ("boundary_verification_deferred", "boundary_deferred"),
        ("boundary_verification_pending", "boundary_pending"),
        ("boundary_verification_trigger", "boundary_trigger"),
        ("boundary_checkpoint_continuation_requested", "checkpoint_requested"),
        ("boundary_checkpoint_continuation_used", "checkpoint_used"),
        ("boundary_checkpoint_fallback_reason", "checkpoint_fallback"),
        ("checkpoint_residual_prediction_reason", "checkpoint_predict"),
        ("checkpoint_residual_prediction_extra_cutbacks", "checkpoint_extra"),
        ("factor_tol_numerical_failure_rejected", "numeric_boundary_rejected"),
        ("factor_tol_enforcement_reason", "factor_tol_reason"),
        ("early_failure_stop", "early_stop"),
        ("early_failure_class", "early_class"),
        ("early_failure_score", "early_score"),
        ("early_failure_reason", "early_reason"),
        ("elapsed_seconds", "elapsed"),
        ("solver_elapsed_seconds", "solver_elapsed"),
        ("overhead_elapsed_seconds", "overhead"),
        ("unattributed_elapsed_seconds", "unattributed"),
        ("timing_coverage_ratio", "timing_coverage"),
        ("estimated_fos_from_last_load", "est_FOS"),
        ("trial_status", "status"),
        ("attempted_load_factor", "attempt"),
        ("last_accepted_load_factor", "last_load"),
        ("last_accepted_plastic_ratio", "last_pr"),
        ("line_search_batch_candidates_total", "ls_batch_candidates"),
        ("plastic_ratio_delta", "pr_delta"),
        ("max_equivalent_plastic_strain", "max_eqp"),
        ("yielded_element_count", "yielded"),
        ("connected_plastic_cluster_size", "cluster"),
        ("plastic_cluster_spans_boundary", "span_boundary"),
        ("final_step_size", "final_step"),
        ("last_accepted_residual_norm", "last_residual"),
        ("residual_norm_final", "residual"),
        ("newton_iterations_total", "newton_total"),
        ("line_search_reductions_total", "ls_total"),
        ("min_det_j", "min_detJ"),
        ("max_displacement_norm", "max_u"),
        ("displacement_increment_norm", "du_inc"),
        ("internal_external_work_ratio", "work_ratio"),
        ("accepted_increment_count", "accepted"),
        ("cutback_count", "cutbacks"),
        ("failed_step_size", "failed_step"),
        ("next_step_size", "next_step"),
        ("solver_cancel_scope", "cancel_scope"),
        ("increment_checkpoint_available", "checkpoint_available"),
        ("increment_checkpoint_load_factor", "checkpoint_load"),
        ("increment_checkpoint_reused_history_rows", "checkpoint_history"),
    ):
        value = raw.get(key)
        if value in (None, ""):
            continue
        parts.append(f"{label}={_fmt(value)}")
    return "" if not parts else ", " + ", ".join(parts)


def _run_log_lines(result: SolveResult2D, analysis_name: str) -> list[str]:
    lines = [f"[GeoFEM 2D] {analysis_name} analysis"]
    public = public_profile_summary(result)
    if public:
        workflows = ",".join(str(value) for value in public.get("workflows_used", []))
        lines.append(f"[geofeas_public] profile={public.get('profile')} workflows={workflows} blocked_proprietary={public.get('blocked_proprietary_count', 0)}")
    for warning in result.warnings:
        lines.append(f"[warning] {warning}")
    for stage in result.stages:
        solver = stage.solver_info.get("method", "unknown")
        active = len(stage.active_elements)
        active_structural = sum(1 for row in stage.structural_results if float(row.get("active", 0.0) or 0.0) > 0.0)
        residual = stage.solver_info.get("residual_norm")
        residual_text = "" if residual is None else f", residual={_fmt(residual)}"
        structural_text = "" if not stage.structural_results else f", active_structural={active_structural}"
        srm = _stage_srm_info(stage)
        srm_trials = srm.get("trials", []) if srm else []
        srm_trial_count = len(srm_trials) if isinstance(srm_trials, list) else 0
        srm_text = "" if not srm else f", FOS={_fmt(srm.get('factor_of_safety'))}, srm_trials={srm_trial_count}"
        lines.append(f"[stage] {stage.name}: completed, active_elements={active}{structural_text}, solver={solver}{residual_text}{srm_text}, output={stage.output_dir}")
        if srm:
            lines.append(
                f"[srm] {stage.name}: FOS={_fmt(srm.get('factor_of_safety'))}, "
                f"stable_factor={_fmt(srm.get('stable_factor'))}, failed_factor={_fmt(srm.get('failed_factor'))}, "
                f"trials={srm_trial_count}, search_mode={srm.get('search_mode', '')}"
            )
            auto_info = srm.get("auto", {}) if isinstance(srm.get("auto", {}), Mapping) else {}
            if auto_info:
                lines.append(
                    f"[srm-boundary] {stage.name}: strategy={auto_info.get('boundary_verification_strategy', '')}, "
                    f"deferred={auto_info.get('boundary_verification_deferred_count', '')}, "
                    f"executed={auto_info.get('boundary_verification_executed_count', '')}, "
                    f"recoveries={auto_info.get('boundary_verification_recovery_count', '')}, "
                    f"stable_reversals={auto_info.get('boundary_verification_stable_reversal_count', '')}, "
                    f"checkpoint_requested={auto_info.get('boundary_checkpoint_continuation_requested_count', '')}, "
                    f"checkpoint_used={auto_info.get('boundary_checkpoint_continuation_used_count', '')}, "
                    f"checkpoint_fallbacks={auto_info.get('boundary_checkpoint_fallback_count', '')}"
                )
            parallel = srm.get("parallel")
            if isinstance(parallel, Mapping):
                env = parallel.get("environment", {})
                if not isinstance(env, Mapping):
                    env = {}
                mesh_info = parallel.get("mesh", {})
                if not isinstance(mesh_info, Mapping):
                    mesh_info = {}
                thread_control = parallel.get("thread_control", {})
                if not isinstance(thread_control, Mapping):
                    thread_control = {}
                lines.append(
                    f"[srm-parallel] {stage.name}: enabled={parallel.get('enabled', '')}, "
                    f"strategy={parallel.get('strategy', '')}, context={parallel.get('context', env.get('context', ''))}, "
                    f"executor={parallel.get('executor', '')}, effective_executor={parallel.get('effective_executor', parallel.get('executor', ''))}, "
                    f"process_fallbacks={parallel.get('process_executor_fallback_count', '')}, "
                    f"policy={parallel.get('policy', env.get('policy', ''))}, workers={parallel.get('max_workers', '')}, "
                    f"requested_workers={parallel.get('requested_workers', '')}, "
                    f"threads_per_worker={parallel.get('selected_threads_per_worker', '')}, "
                    f"lookahead_depth={parallel.get('lookahead_depth', '')}, "
                    f"speculative_trials={parallel.get('speculative_trial_count', '')}, "
                    f"used_speculative_trials={parallel.get('used_speculative_trial_count', '')}, "
                    f"unused_speculative_trials={parallel.get('unused_speculative_trial_count', '')}, "
                    f"canceled_speculative_trials={parallel.get('canceled_speculative_trial_count', '')}, "
                    f"speculative_cancellation_requested={parallel.get('speculative_cancellation_requested', '')}, "
                    f"decision_cancel_enabled={parallel.get('decision_linked_cancellation_enabled', '')}, "
                    f"decision_cancel_requested={parallel.get('decision_linked_requested_count', '')}, "
                    f"decision_cancel_pending={parallel.get('decision_linked_pending_cancel_count', '')}, "
                    f"decision_safe_stops={parallel.get('decision_linked_safe_stop_count', '')}, "
                    f"decision_completed_after_request={parallel.get('decision_linked_completed_after_request_count', '')}, "
                    f"speculative_prefetch_calls={parallel.get('speculative_prefetch_call_count', '')}, "
                    f"speculative_prefetch_wall={_fmt(parallel.get('speculative_prefetch_wall_elapsed_seconds'))}, "
                    f"speculative_trial_elapsed={_fmt(parallel.get('speculative_trial_elapsed_seconds'))}, "
                    f"speculative_queue_wait={_fmt(parallel.get('speculative_queue_wait_elapsed_seconds'))}, "
                    f"speculative_estimated_saving={_fmt(parallel.get('speculative_estimated_wall_clock_saving_seconds'))}, "
                    f"bisection_speculation={parallel.get('bisection_speculation_enabled', '')}, "
                    f"bisection_speculative_trials={parallel.get('bisection_speculative_trial_count', '')}, "
                    f"bisection_used_speculative_trials={parallel.get('bisection_used_speculative_trial_count', '')}, "
                    f"bisection_unused_speculative_trials={parallel.get('bisection_unused_speculative_trial_count', '')}, "
                    f"thread_control_applied={thread_control.get('applied', '')}, "
                    f"thread_control_method={thread_control.get('apply_method', '')}, "
                    f"thread_env_restored={thread_control.get('environment_restored', '')}, "
                    f"threadpoolctl_available={thread_control.get('threadpoolctl_available', '')}, "
                    f"logical_cpus={parallel.get('logical_cpu_count', parallel.get('cpu_count', env.get('logical_cpu_count', '')))}, "
                    f"physical_cpus={parallel.get('physical_cpu_count', env.get('physical_cpu_count', ''))}, "
                    f"available_memory_mb={_fmt(parallel.get('available_memory_mb', env.get('available_memory_mb')))}, "
                    f"memory_limit_mb={_fmt(parallel.get('memory_limit_mb'))}, "
                    f"memory_per_worker_mb={_fmt(parallel.get('memory_per_worker_mb'))}, "
                    f"memory_limited={parallel.get('memory_limited', '')}, "
                    f"nodes={mesh_info.get('node_count', '')}, elements={mesh_info.get('element_count', '')}, "
                    f"active_elements={mesh_info.get('active_element_count', '')}, dof={mesh_info.get('dof_count', '')}"
                )
        if isinstance(srm_trials, list):
            for index, raw in enumerate(srm_trials, start=1):
                if not isinstance(raw, Mapping):
                    continue
                lines.append(
                    f"[srm-trial] {stage.name} #{index}: factor={_fmt(raw.get('factor'))}, "
                    f"ok={raw.get('ok', '')}, converged={raw.get('converged', '')}, "
                    f"plastic_ratio={_fmt(raw.get('plastic_ratio'))}, "
                    f"reason={raw.get('failure_reason') or raw.get('error') or ''}"
                    f"{_srm_trial_log_details(raw)}"
                )
    lines.append("[report] calculation_report.html and calculation_report.pdf generated")
    return lines


def _input_conditions_section(result: SolveResult2D, cfg: Mapping[str, Any], summary_data: Mapping[str, Any]) -> list[str]:
    analysis = cfg.get("analysis", {}) if isinstance(cfg.get("analysis", {}), Mapping) else {}
    mesh_cfg = cfg.get("mesh", {}) if isinstance(cfg.get("mesh", {}), Mapping) else {}
    solver_cfg = cfg.get("solver", {}) if isinstance(cfg.get("solver", {}), Mapping) else {}
    rows = [
        ("解析次元", summary_data.get("dimension", "2D")),
        ("解析種別", summary_data.get("analysis", "")),
        ("単位系", analysis.get("unit_system", "-") if isinstance(analysis, Mapping) else "-"),
        ("節点数", len(result.mesh.node_ids)),
        ("要素数", len(result.mesh.elements)),
        ("界面数", len(result.interfaces)),
        ("要素タイプ", _element_type_summary(result.mesh)),
        ("積分設定", _integration_summary(result.mesh)),
        ("メッシュ生成", mesh_cfg.get("generator", "explicit") if isinstance(mesh_cfg, Mapping) else "explicit"),
        ("出力先", result.output_dir),
    ]
    parts = ["<section id=\"input\"><h2>1. 入力条件</h2>", _kv_table(rows)]
    if solver_cfg:
        parts.append("<h3>ソルバ設定</h3>")
        parts.append(f"<pre>{_h(json.dumps(solver_cfg, ensure_ascii=False, indent=2, default=str))}</pre>")
    if analysis:
        parts.append("<h3>解析条件</h3>")
        parts.append(f"<pre>{_h(json.dumps(analysis, ensure_ascii=False, indent=2, default=str))}</pre>")
    parts.append("</section>")
    return parts


def _materials_section(result: SolveResult2D) -> list[str]:
    headers = ["材料", "モデル", "E", "ν", "γ", "t", "c", "φ", "ψ", "σy", "H", "No-Tension"]
    rows = []
    for name, mat in sorted(result.materials.items()):
        rows.append(
            [
                name,
                mat.model,
                _fmt(mat.E),
                _fmt(mat.nu),
                _fmt(mat.gamma),
                _fmt(mat.thickness),
                _fmt(mat.cohesion),
                _fmt(mat.friction_angle),
                _fmt(mat.dilation_angle),
                _fmt(mat.yield_stress),
                _fmt(mat.hardening),
                "ON" if mat.tension_cutoff else "OFF",
            ]
        )
    return ["<section id=\"materials\"><h2>2. 材料表</h2>", _table(headers, rows), "</section>"]


def _load_combination_section(result: SolveResult2D, cfg: Mapping[str, Any]) -> list[str]:
    combo_rows = _load_combination_rows(cfg)
    post_rows = _post_case_comparison_rows(result)
    if not combo_rows and not post_rows:
        return []
    parts = ["<section id=\"load-combinations\"><h2>4. 荷重組合せ照査</h2>"]
    if combo_rows:
        headers = ["組合せ", "ケース", "係数", "ケース種別", "ケース倍率", "有効", "説明"]
        rows = [[row["combination"], row["case"], row["factor"], row["case_type"], row["case_scale"], row["active"], row["description"]] for row in combo_rows]
        parts.extend(["<h3>荷重組合せ係数表</h3>", _table(headers, rows)])
    if post_rows:
        headers = ["ステージ", "組合せ", "kh", "kv", "最大変位", "最大沈下", "最大水圧", "構造軸力最大", "バネ反力最大", "界面すべり最大"]
        rows = [
            [
                row["stage"],
                row["load_combination"],
                _fmt(row["seismic_kh"]),
                _fmt(row["seismic_kv"]),
                _fmt(row["max_displacement"]),
                _fmt(row["max_settlement"]),
                _fmt(row["max_pore_pressure"]),
                _fmt(row["structural_axial_force_max"]),
                _fmt(row["structural_spring_reaction_max"]),
                _fmt(row["interface_slip_max"]),
            ]
            for row in post_rows
        ]
        parts.extend(["<h3>ケース別Post比較</h3>", _table(headers, rows)])
    parts.append("</section>")
    return parts


def _boundary_load_section(result: SolveResult2D, stage_specs: list[dict[str, Any]]) -> list[str]:
    parts = ["<section id=\"bc-load\"><h2>3. 境界/荷重図</h2>"]
    for idx, stage in enumerate(result.stages):
        spec = stage_specs[idx] if idx < len(stage_specs) else {"boundary_conditions": [], "loads": []}
        bcs = _ensure_list(spec.get("boundary_conditions", []))
        loads = _ensure_list(spec.get("loads", []))
        parts.append(f"<article class=\"stage-block\"><h3>{idx + 1}. {_h(stage.name)}</h3>")
        parts.append(
            f"<figure class=\"report-figure\"><figcaption>Figure {idx + 1}. Boundary and load diagram - {_h(stage.name)}</figcaption>"
            + _boundary_load_svg(result.mesh, stage, bcs, loads)
            + "</figure>"
        )
        parts.append(_kv_table([("境界条件数", len(bcs)), ("荷重数", len(loads)), ("拘束DOF数", len(stage.constrained_dofs))]))
        parts.append("</article>")
    parts.append("</section>")
    return parts


def _stage_list_section(result: SolveResult2D, stage_specs: list[dict[str, Any]]) -> list[str]:
    headers = ["No.", "ステージ", "タイプ", "Active要素", "時刻", "ソルバ", "収束", "出力"]
    rows = []
    for idx, stage in enumerate(result.stages):
        spec = stage_specs[idx] if idx < len(stage_specs) else {}
        solver = stage.solver_info
        rows.append(
            [
                idx + 1,
                stage.name,
                spec.get("type", "static"),
                len(stage.active_elements),
                _fmt(stage.time),
                solver.get("method", "-"),
                _convergence_text(solver),
                _rel_link(stage.output_dir, result.output_dir) if stage.output_dir else "-",
            ]
        )
    return ["<section id=\"stages\"><h2>5. ステージ一覧</h2>", _table(headers, rows, raw_columns={7}), "</section>"]


def _results_section(result: SolveResult2D) -> list[str]:
    parts = ["<section id=\"results\"><h2>6. 解析結果図</h2>"]
    for idx, stage in enumerate(result.stages):
        parts.append(f"<article class=\"stage-block\"><h3>{idx + 1}. {_h(stage.name)}</h3>")
        figure_no = len(result.stages) + idx + 1
        parts.append(
            f"<figure class=\"report-figure\"><figcaption>Figure {figure_no}. Result contour and deformation overlay - {_h(stage.name)}</figcaption>"
            + _result_svg(result.mesh, stage)
            + "</figure>"
        )
        parts.append(_kv_table(_stage_metric_rows(result.mesh, stage)))
        parts.append(_stage_file_links(result, stage))
        parts.append("</article>")
    parts.append("</section>")
    return parts


def _judgement_section(result: SolveResult2D) -> list[str]:
    headers = ["分類", "対象", "判定", "根拠"]
    rows = []
    for stage in result.stages:
        solver = stage.solver_info
        converged = solver.get("converged", True)
        rows.append(["収束", stage.name, "OK" if converged is not False else "NG", f"method={solver.get('method', '-')}, iterations={solver.get('iterations', '-')}"])
        residual = solver.get("residual_norm")
        if residual is not None:
            ok = math.isfinite(float(residual))
            rows.append(["残差", stage.name, "OK" if ok else "NG", f"residual_norm={_fmt(residual)}"])
        max_u = _max_displacement(result.mesh, stage)
        rows.append(["変位", stage.name, "OK" if math.isfinite(max_u) else "NG", f"max_displacement={_fmt(max_u)}"])
        if isinstance(solver.get("srm"), Mapping):
            srm = solver["srm"]
            rows.append(["安全率", stage.name, srm_safety_verdict(srm), srm_fos_display(srm, locale="ja")])
        if stage.pore_pressure is not None:
            pmax = float(np.max(stage.pore_pressure))
            rows.append(["水圧", stage.name, "OK" if math.isfinite(pmax) else "NG", f"max_pore_pressure={_fmt(pmax)}"])
        if stage.interface_results:
            slip = max(float(row.get("slip_abs", 0.0)) for row in stage.interface_results)
            rows.append(["界面", stage.name, "INFO", f"max_slip={_fmt(slip)}, slip_points={sum(1 for row in stage.interface_results if row.get('state') == 'slip')}"])
    for warning in result.warnings:
        rows.append(["警告", "run", "WARN", warning])
    if not rows:
        rows.append(["全体", "run", "OK", "診断対象の異常はありません。"])
    return ["<section id=\"judgement\"><h2>7. 判定表</h2>", _table(headers, rows), "</section>"]


def _srm_trials_section(result: SolveResult2D) -> list[str]:
    summary_rows = _srm_summary_records(result)
    if not summary_rows:
        return []
    trial_rows = _srm_trial_records(result)
    summary_table_rows = [
        [
            row["stage"],
            row["factor_of_safety"],
            row["stable_factor"],
            row["failed_factor"],
            row["factor_of_safety_status"],
            row["factor_of_safety_confidence"],
            row["trial_count"],
            row["search_mode"],
        ]
        for row in summary_rows
    ]
    trial_table_rows = [
        [
            row["stage"],
            row["index"],
            row["factor"],
            row["ok"],
            row["converged"],
            row["plastic_ratio"],
            row["auto_decision"],
            row["auto_trial_action"],
            row["early_failure_stop"],
            row["early_failure_score"],
            row["early_failure_reason"],
            row["auto_retry_index"],
            row["elapsed_seconds"],
            row["estimated_fos_from_last_load"],
            row["trial_status"],
            row["last_accepted_load_factor"],
            row["last_accepted_plastic_ratio"],
            row["plastic_ratio_delta"],
            row["max_equivalent_plastic_strain"],
            row["yielded_element_count"],
            row["connected_plastic_cluster_size"],
            row["plastic_cluster_spans_boundary"],
            row["newton_iterations_total"],
            row["line_search_reductions_total"],
            row["cutback_count"],
            row["mc_numba_to_python_fallback_count"],
            row["mc_numba_regularized_projection_count"],
            row["mc_regularized_projection_count"],
            row["mc_regularized_projection_max_relative_yield_violation"],
            row["failure_reason"],
        ]
        for row in trial_rows
    ]
    return [
        "<section id=\"srm-trials\"><h2>SRM FOS Trial Results</h2>",
        "<h3>Summary</h3>",
        _table(
            ["Stage", "FOS", "Stable factor", "Failed factor", "FOS status", "Confidence", "Trials", "Search mode"],
            summary_table_rows,
        ),
        "<h3>SRM Trial Results</h3>",
        _table(
            [
                "Stage",
                "#",
                "Factor",
                "OK",
                "Converged",
                "Plastic ratio",
                "Auto decision",
                "Auto action",
                "Early stop",
                "Early score",
                "Early reason",
                "Retry",
                "Elapsed",
                "Est. FOS",
                "Status",
                "Last load",
                "Last plastic ratio",
                "Plastic ratio delta",
                "Max eqp",
                "Yielded elements",
                "Cluster",
                "Spans boundary",
                "Newton total",
                "Line search total",
                "Cutbacks",
                "MC Python fallback",
                "MC Numba regularized projection",
                "MC regularized projection",
                "MC max relative violation",
                "Reason",
            ],
            trial_table_rows,
        ),
        "</section>",
    ]


def _log_section(log_lines: list[str]) -> list[str]:
    return ["<section id=\"log\"><h2>8. ログ</h2>", f"<pre>{_h(chr(10).join(log_lines))}</pre>", "</section>"]


def _reproducibility_section(result: SolveResult2D, cfg: Mapping[str, Any], summary_data: Mapping[str, Any]) -> list[str]:
    files = []
    for path in sorted(result.output_dir.rglob("*")):
        if path.is_file():
            files.append([_rel_link(path, result.output_dir), path.stat().st_size])
    parts = ["<section id=\"repro\"><h2>9. 再現条件</h2>"]
    parts.append(
        _kv_table(
            [
                ("出力ディレクトリ", result.output_dir),
                ("summary.json", result.output_dir / "summary.json"),
                ("run.log", result.output_dir / "run.log"),
                ("HTML計算書", result.output_dir / "calculation_report.html"),
                ("PDF計算書", result.output_dir / "calculation_report.pdf"),
                ("固定化manifest", result.output_dir / "calculation_report_manifest.json"),
                ("入力snapshot", result.output_dir / "calculation_report_input_snapshot.json"),
                ("入力SHA256", _stable_json_hash(_json_safe(cfg))),
                ("summary SHA256", _stable_json_hash(summary_data)),
            ]
        )
    )
    parts.append("<h3>出力ファイル一覧</h3>")
    parts.append(_table(["ファイル", "サイズ(byte)"], files, raw_columns={0}))
    parts.append("<h3>summary.json</h3>")
    parts.append(f"<pre>{_h(json.dumps(summary_data, ensure_ascii=False, indent=2, default=str))}</pre>")
    if cfg:
        parts.append("<h3>入力スナップショット</h3>")
        parts.append(f"<pre>{_h(json.dumps(_json_safe(cfg), ensure_ascii=False, indent=2, default=str))}</pre>")
    parts.append("</section>")
    return parts


def _stage_specs_from_config(cfg: Mapping[str, Any], result: SolveResult2D) -> list[dict[str, Any]]:
    global_bcs = _ensure_list(cfg.get("boundary_conditions", cfg.get("bc", [])))
    global_loads = _ensure_list(cfg.get("loads", []))
    global_mpc = _ensure_list(cfg.get("mpc_constraints", cfg.get("mpc", [])))
    raw_stages = cfg.get("stages", cfg.get("steps", []))
    stages = [dict(stage) for stage in raw_stages if isinstance(stage, Mapping)] if isinstance(raw_stages, list) else []
    if not stages:
        stages = [{"name": "Stage-1", "type": "static"} for _stage in result.stages]
    specs = []
    for idx, stage in enumerate(result.stages):
        spec = dict(stages[idx]) if idx < len(stages) else {"name": stage.name, "type": "static"}
        spec["boundary_conditions"] = global_bcs + _ensure_list(spec.get("boundary_conditions", spec.get("bc", [])))
        spec["loads"] = global_loads + _ensure_list(spec.get("loads", []))
        spec["mpc_constraints"] = global_mpc + _ensure_list(spec.get("mpc_constraints", spec.get("mpc", [])))
        specs.append(spec)
    return specs


def _load_combination_rows(cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    cases = _load_case_map_for_report(cfg)
    combos = configured_load_combinations(cfg)
    rows: list[dict[str, Any]] = []
    if combos:
        for combo in combos:
            if not isinstance(combo, Mapping):
                continue
            name = str(combo.get("name", combo.get("id", "")) or "")
            factors = combo.get("factors", {})
            if not isinstance(factors, Mapping):
                continue
            for case, factor in factors.items():
                case_info = cases.get(str(case), {})
                rows.append(
                    {
                        "combination": name,
                        "case": str(case),
                        "factor": float(factor),
                        "case_type": case_info.get("type", ""),
                        "case_scale": float(case_info.get("scale", 1.0) or 1.0),
                        "active": bool(case_info.get("active", True)),
                        "description": case_info.get("description", combo.get("description", "")),
                        "standard": combo.get("standard", ""),
                        "source": combo.get("source", ""),
                        "revision": combo.get("revision", ""),
                        "coverage": combo.get("coverage", ""),
                        "clause": combo.get("clause", ""),
                    }
                )
        return rows
    for name, case in cases.items():
        rows.append(
            {
                "combination": "service",
                "case": name,
                "factor": 1.0,
                "case_type": case.get("type", ""),
                "case_scale": float(case.get("scale", 1.0) or 1.0),
                "active": bool(case.get("active", True)),
                "description": case.get("description", ""),
            }
        )
    return rows


def _load_case_map_for_report(cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in _ensure_list(cfg.get("load_cases", [])):
        if isinstance(raw, Mapping):
            name = str(raw.get("name", raw.get("case", "")) or "")
            if name:
                out[name] = dict(raw)
    return out


def _post_case_comparison_rows(result: SolveResult2D) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage in result.stages:
        load_info = stage.solver_info.get("load_processing", {})
        seismic = stage.solver_info.get("seismic", {})
        public = stage.solver_info.get("geofeas_public", {})
        liquefaction = stage.solver_info.get("liquefaction", {})
        if not isinstance(load_info, Mapping):
            load_info = {}
        if not isinstance(seismic, Mapping):
            seismic = {}
        if not isinstance(public, Mapping):
            public = {}
        if not isinstance(liquefaction, Mapping):
            liquefaction = {}
        post_public = public.get("post", {})
        if not isinstance(post_public, Mapping):
            post_public = {}
        rows.append(
            {
                "stage": stage.name,
                "geofeas_workflow": str(public.get("workflow", "")),
                "load_combination": str(load_info.get("load_combination", "")),
                "seismic_kh": float(seismic.get("kh", 0.0) or 0.0),
                "seismic_kv": float(seismic.get("kv", 0.0) or 0.0),
                "max_displacement": _max_displacement(result.mesh, stage),
                "max_settlement": float(max((-stage.displacements[1::2]), default=0.0)),
                "max_pore_pressure": "" if stage.pore_pressure is None else float(np.max(stage.pore_pressure)),
                "relative_stage": "previous" if post_public.get("relative_displacement") else "",
                "liquefaction_min_fl": "" if not liquefaction else liquefaction.get("min_FL", ""),
                "liquefaction_max_ru": "" if not liquefaction else liquefaction.get("max_ru", ""),
                "structural_axial_force_max": "" if not stage.structural_results else float(max(abs(float(row.get("axial_force", 0.0) or 0.0)) for row in stage.structural_results)),
                "structural_spring_reaction_max": "" if not stage.structural_results else float(max(abs(float(row.get("spring_reaction", 0.0) or 0.0)) for row in stage.structural_results)),
                "interface_slip_max": "" if not stage.interface_results else float(max(row.get("slip_abs", 0.0) for row in stage.interface_results)),
            }
        )
    return rows


def _boundary_load_svg(mesh: Mesh2D, stage: StageResult2D, bcs: list[Any], loads: list[Any]) -> str:
    view, transform = _svg_view(mesh)
    active = set(stage.active_elements)
    parts = [_svg_open(view), "<g class=\"mesh\">"]
    for element in mesh.elements:
        points = " ".join(f"{transform(mesh.coords[mesh.node_index[nid]])[0]:.2f},{transform(mesh.coords[mesh.node_index[nid]])[1]:.2f}" for nid in element.nodes[: 3 if element.type.startswith("TRI") else 4])
        cls = "active-element" if element.id in active else "inactive-element"
        parts.append(f"<polygon class=\"{cls}\" points=\"{points}\"/>")
    parts.append("</g><g class=\"bc-symbols\">")
    for bc in bcs:
        if not isinstance(bc, Mapping):
            continue
        values = _bc_values(bc)
        label = ",".join(f"{dof}={_fmt(value)}" for dof, value in values.items()) or "BC"
        for nid in _target_nodes_for_report(mesh, bc):
            x, y = transform(mesh.coords[mesh.node_index[nid]])
            parts.append(f"<rect x=\"{x - 4:.2f}\" y=\"{y - 4:.2f}\" width=\"8\" height=\"8\" rx=\"1\"/>")
            parts.append(f"<text x=\"{x + 6:.2f}\" y=\"{y - 6:.2f}\">{_h(label)}</text>")
    parts.append("</g><g class=\"load-symbols\">")
    for load in loads:
        if not isinstance(load, Mapping):
            continue
        if "edge" in load or "edges" in load:
            tx = _float_or_zero(load.get("tx", load.get("qx", 0.0)))
            ty = _float_or_zero(load.get("ty", load.get("qy", 0.0)))
            for edge in _target_edges_for_report(mesh, load):
                p0 = mesh.coords[mesh.node_index[edge[0]]]
                p1 = mesh.coords[mesh.node_index[edge[1]]]
                mid = 0.5 * (p0 + p1)
                _append_arrow(parts, transform, mid, tx, ty, f"t=({_fmt(tx)},{_fmt(ty)})")
        else:
            fx = _float_or_zero(load.get("fx", load.get("px", load.get("ux", 0.0))))
            fy = _float_or_zero(load.get("fy", load.get("py", load.get("uy", 0.0))))
            for nid in _target_nodes_for_report(mesh, load):
                _append_arrow(parts, transform, mesh.coords[mesh.node_index[nid]], fx, fy, f"F=({_fmt(fx)},{_fmt(fy)})")
    parts.append("</g><text class=\"svg-title\" x=\"14\" y=\"22\">境界/荷重図</text></svg>")
    return "\n".join(parts)


def _result_svg(mesh: Mesh2D, stage: StageResult2D) -> str:
    view, transform = _svg_view(mesh)
    values_by_element = {str(row.get("element_id", row.get("id", ""))): float(row.get("q", row.get("sigma_y", 0.0)) or 0.0) for row in stage.element_results}
    values = list(values_by_element.values()) or [0.0]
    vmin = min(values)
    vmax = max(values)
    active = set(stage.active_elements)
    scale = _deformation_scale(mesh, stage)
    parts = [_svg_open(view), "<g class=\"result-elements\">"]
    for element in mesh.elements:
        corners = element.nodes[: 3 if element.type.startswith("TRI") else 4]
        fill = _contour_color(values_by_element.get(element.id, 0.0), vmin, vmax) if element.id in active else "#e5e7eb"
        points = " ".join(f"{transform(mesh.coords[mesh.node_index[nid]])[0]:.2f},{transform(mesh.coords[mesh.node_index[nid]])[1]:.2f}" for nid in corners)
        parts.append(f"<polygon class=\"result-cell\" fill=\"{fill}\" points=\"{points}\"/>")
    parts.append("</g><g class=\"deformed\">")
    for element in mesh.elements:
        if element.id not in active:
            continue
        corners = element.nodes[: 3 if element.type.startswith("TRI") else 4]
        dpoints = []
        for nid in corners:
            idx = mesh.node_index[nid]
            xy = mesh.coords[idx].copy()
            xy[0] += stage.displacements[2 * idx] * scale
            xy[1] += stage.displacements[2 * idx + 1] * scale
            x, y = transform(xy)
            dpoints.append(f"{x:.2f},{y:.2f}")
        parts.append(f"<polygon class=\"deformed-cell\" points=\"{' '.join(dpoints)}\"/>")
    parts.append("</g>")
    parts.append(f"<text class=\"svg-title\" x=\"14\" y=\"22\">解析結果図 q contour / 変形倍率 {_fmt(scale)}</text>")
    parts.append(_svg_legend(vmin, vmax))
    parts.append("</svg>")
    return "\n".join(parts)


def _svg_open(view: tuple[float, float, float, float]) -> str:
    x, y, w, h = view
    return (
        f"<svg class=\"figure\" viewBox=\"{x:.2f} {y:.2f} {w:.2f} {h:.2f}\" role=\"img\" xmlns=\"http://www.w3.org/2000/svg\">"
        "<defs><marker id=\"arrow\" markerWidth=\"8\" markerHeight=\"8\" refX=\"6\" refY=\"3\" orient=\"auto\" markerUnits=\"strokeWidth\">"
        "<path d=\"M0,0 L0,6 L6,3 z\" fill=\"#b91c1c\"/></marker></defs>"
    )


def _svg_view(mesh: Mesh2D) -> tuple[tuple[float, float, float, float], Any]:
    xmin = float(np.min(mesh.coords[:, 0])) if len(mesh.node_ids) else 0.0
    xmax = float(np.max(mesh.coords[:, 0])) if len(mesh.node_ids) else 1.0
    ymin = float(np.min(mesh.coords[:, 1])) if len(mesh.node_ids) else 0.0
    ymax = float(np.max(mesh.coords[:, 1])) if len(mesh.node_ids) else 1.0
    width = max(xmax - xmin, 1.0e-9)
    height = max(ymax - ymin, 1.0e-9)
    canvas_w, canvas_h, margin = 760.0, 420.0, 36.0
    scale = min((canvas_w - 2.0 * margin) / width, (canvas_h - 2.0 * margin) / height)
    ox = margin - xmin * scale + 0.5 * ((canvas_w - 2.0 * margin) - width * scale)
    oy = canvas_h - margin + ymin * scale - 0.5 * ((canvas_h - 2.0 * margin) - height * scale)

    def transform(xy: Any) -> tuple[float, float]:
        return ox + float(xy[0]) * scale, oy - float(xy[1]) * scale

    return (0.0, 0.0, canvas_w, canvas_h), transform


def _append_arrow(parts: list[str], transform: Any, point: Any, vx: float, vy: float, label: str) -> None:
    length = math.hypot(vx, vy)
    if length <= 1.0e-30:
        return
    x, y = transform(point)
    ux, uy = vx / length, vy / length
    arrow_len = 32.0
    x0 = x - ux * arrow_len
    y0 = y + uy * arrow_len
    parts.append(f"<line x1=\"{x0:.2f}\" y1=\"{y0:.2f}\" x2=\"{x:.2f}\" y2=\"{y:.2f}\" marker-end=\"url(#arrow)\"/>")
    parts.append(f"<text x=\"{x0 + 4:.2f}\" y=\"{y0 - 4:.2f}\">{_h(label)}</text>")


def _svg_legend(vmin: float, vmax: float) -> str:
    return (
        "<g class=\"legend\">"
        "<rect x=\"608\" y=\"310\" width=\"120\" height=\"14\" fill=\"#2563eb\"/>"
        "<rect x=\"608\" y=\"324\" width=\"120\" height=\"14\" fill=\"#f8fafc\"/>"
        "<rect x=\"608\" y=\"338\" width=\"120\" height=\"14\" fill=\"#dc2626\"/>"
        f"<text x=\"608\" y=\"304\">q: {_fmt(vmin)} ... {_fmt(vmax)}</text>"
        "</g>"
    )


def _deformation_scale(mesh: Mesh2D, stage: StageResult2D) -> float:
    width = float(np.ptp(mesh.coords[:, 0])) if len(mesh.node_ids) else 1.0
    height = float(np.ptp(mesh.coords[:, 1])) if len(mesh.node_ids) else 1.0
    diag = max(math.hypot(width, height), 1.0e-12)
    max_u = _max_displacement(mesh, stage)
    if max_u <= 1.0e-30:
        return 1.0
    return min(max(0.12 * diag / max_u, 1.0), 500.0)


def _contour_color(value: float, vmin: float, vmax: float) -> str:
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    if t < 0.5:
        a = t / 0.5
        r = int(37 + (248 - 37) * a)
        g = int(99 + (250 - 99) * a)
        b = int(235 + (252 - 235) * a)
    else:
        a = (t - 0.5) / 0.5
        r = int(248 + (220 - 248) * a)
        g = int(250 + (38 - 250) * a)
        b = int(252 + (38 - 252) * a)
    return f"#{r:02x}{g:02x}{b:02x}"


def _stage_file_links(result: SolveResult2D, stage: StageResult2D) -> str:
    if stage.output_dir is None:
        return ""
    names = [
        "displacements.csv",
        "reactions.csv",
        "element_stress.csv",
        "integration_point_stress.csv",
        "interface_state.csv",
        "structural_state.csv",
        "structural_section_forces.csv",
        "pore_pressure.csv",
        "liquefaction_state.csv",
        "liquefaction_history.csv",
        "liquefaction_ru_fl.svg",
        "liquefaction_post.html",
        "riks_path.csv",
        "results.vtk",
        "report.html",
    ]
    links = []
    for name in names:
        path = stage.output_dir / name
        if path.exists():
            links.append(f"<li>{_rel_link(path, result.output_dir)}</li>")
    return "<h4>出力ファイル</h4><ul class=\"file-list\">" + "".join(links) + "</ul>" if links else ""


def _stage_metric_rows(mesh: Mesh2D, stage: StageResult2D) -> list[tuple[str, Any]]:
    plastic_ratio = 0.0
    if stage.element_results:
        plastic_ratio = sum(1 for row in stage.element_results if float(row.get("plastic", 0.0) or 0.0) > 0.0) / len(stage.element_results)
    return [
        ("最大変位", _fmt(_max_displacement(mesh, stage))),
        ("最大沈下 settlement=-uy", _fmt(float(max((-stage.displacements[1::2]), default=0.0)))),
        ("塑性要素比", _fmt(plastic_ratio)),
        ("最大水圧", "-" if stage.pore_pressure is None else _fmt(float(np.max(stage.pore_pressure)))),
        ("界面すべり最大", "-" if not stage.interface_results else _fmt(max(float(row.get("slip_abs", 0.0)) for row in stage.interface_results))),
    ]


def _target_nodes_for_report(mesh: Mesh2D, spec: Mapping[str, Any]) -> list[str]:
    if "node" in spec:
        nid = str(spec["node"])
        return [nid] if nid in mesh.node_index else []
    if "nodes" in spec:
        raw = spec["nodes"]
        if isinstance(raw, str) and raw in mesh.node_sets:
            return list(mesh.node_sets[raw])
        return [str(value) for value in _ensure_list(raw) if str(value) in mesh.node_index]
    if "set" in spec:
        return list(mesh.node_sets.get(str(spec["set"]), []))
    if bool(spec.get("all", False)):
        return list(mesh.node_ids)
    return []


def _target_edges_for_report(mesh: Mesh2D, spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    raw_edges = spec.get("edges", spec.get("edge"))
    if raw_edges is not None:
        if isinstance(raw_edges, str):
            ids = mesh.node_sets.get(raw_edges, [])
            return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
        raw_list = _ensure_list(raw_edges)
        if len(raw_list) == 2 and not isinstance(raw_list[0], (list, tuple, Mapping)):
            a, b = str(raw_list[0]), str(raw_list[1])
            return [(a, b)] if a in mesh.node_index and b in mesh.node_index else []
        out = []
        for edge in raw_list:
            nodes = [str(value) for value in _ensure_list(edge)]
            if len(nodes) >= 2 and nodes[0] in mesh.node_index and nodes[1] in mesh.node_index:
                out.append((nodes[0], nodes[1]))
        return out
    if "set" in spec:
        ids = mesh.node_sets.get(str(spec["set"]), [])
        return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    return []


def _bc_values(bc: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    if "dof" in bc:
        values[str(bc["dof"]).lower()] = _float_or_zero(bc.get("value", 0.0))
    for dof in ("ux", "uy"):
        if dof in bc and bc[dof] is not None:
            values[dof] = _float_or_zero(bc[dof])
    if bool(bc.get("fixed", False)):
        values.setdefault("ux", 0.0)
        values.setdefault("uy", 0.0)
    return values


def _element_type_summary(mesh: Mesh2D) -> str:
    counts: dict[str, int] = {}
    for element in mesh.elements:
        counts[element.type] = counts.get(element.type, 0) + 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _integration_summary(mesh: Mesh2D) -> str:
    counts: dict[str, int] = {}
    for element in mesh.elements:
        counts[element.integration] = counts.get(element.integration, 0) + 1
    return ", ".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def _max_displacement(mesh: Mesh2D, stage: StageResult2D) -> float:
    return float(max((math.hypot(stage.displacements[2 * i], stage.displacements[2 * i + 1]) for i in range(len(mesh.node_ids))), default=0.0))


def _convergence_text(solver: Mapping[str, Any]) -> str:
    if solver.get("converged", True) is False:
        return "未収束"
    if "converged" in solver:
        return "収束"
    return "完了"


def _ensure_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _float_or_zero(value: Any) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return 0.0
    return out if math.isfinite(out) else 0.0


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except TypeError:
        if isinstance(value, Mapping):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return str(value)


__all__ = [
    "write_stage_outputs",
    "write_displacements_csv",
    "write_reactions_csv",
    "write_pore_pressure_csv",
    "write_element_results_csv",
    "write_integration_point_results_csv",
    "write_liquefaction_state_csv",
    "write_liquefaction_post_outputs",
    "write_interface_results_csv",
    "write_structural_results_csv",
    "write_structural_section_forces_csv",
    "write_riks_path_csv",
    "write_dynamic_history_csv",
    "write_vtk",
    "write_stage_report",
    "write_run_summary",
    "load_run_result_from_artifacts",
    "write_deferred_run_artifacts",
    "write_deferred_run_artifacts_from_files",
    "write_calculation_report",
    "write_calculation_report_pdf",
    "write_report_reproducibility_manifest",
    "write_load_combination_csv",
    "write_post_case_comparison_csv",
]

