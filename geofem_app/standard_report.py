"""Generalized report bundle generated from one canonical data source."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .analysis_log import build_structured_analysis_log
from .material_models import build_material_inventory
from .mesh_quality import evaluate_mesh_quality
from .messages import message
from .performance_kpis import build_result_performance_kpi_matrix
from .performance_monitor import build_performance_summary
from .pdf_writer import write_text_pdf
from .reliability_summary import build_reliability_summary
from .fem2d_types import SolveResult2D
from .srm_reporting import srm_fos_display, srm_fos_is_confirmed, srm_result_confidence, srm_result_status


def build_standard_report_data(result: SolveResult2D, artifacts: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    mesh_quality = evaluate_mesh_quality(result.mesh, cfg)
    materials = build_material_inventory(result.materials, cfg)
    analysis_log = build_structured_analysis_log(result)
    performance = build_performance_summary(result)
    performance_kpis = build_result_performance_kpi_matrix(result, artifacts=artifacts, performance_summary=performance)
    reliability = build_reliability_summary(result)
    return {
        "schema": "geofem.standard_report_data.v1",
        "title": message("reports.standard.title"),
        "artifacts": dict(artifacts or {}),
        "input": _input_section(result, cfg),
        "mesh_quality": mesh_quality["summary"],
        "mesh_quality_violations": mesh_quality["violations"],
        "boundary_conditions": _boundary_conditions_section(cfg),
        "loads": _loads_section(cfg),
        "materials": materials,
        "analysis_settings": cfg.get("analysis", {}),
        "stages": _stage_sections(result),
        "warnings": list(result.warnings),
        "analysis_log": {"stage_count": analysis_log["stage_count"], "event_count": len(analysis_log["events"])},
        "performance": {
            key: performance[key]
            for key in (
                "elapsed_seconds",
                "stage_elapsed_seconds",
                "cold_run_elapsed_seconds",
                "stage_count",
                "total_solver_iterations",
                "max_matrix_nnz",
                "estimated_memory_bytes",
                "assembly_elapsed_seconds",
                "nonlinear_iteration_elapsed_seconds",
                "linear_solve_elapsed_seconds",
                "postprocess_elapsed_seconds",
                "io_report_elapsed_seconds",
                "cache_build_elapsed_seconds",
                "cache_reuse_elapsed_seconds",
                "coupled_assembly_elapsed_seconds",
                "dominant_category",
                "dominant_stage",
                "cache_reuse_count",
                "cache_build_count",
                "cache_hit_count",
                "cache_miss_count",
                "iteration_profile_count",
            )
        },
        "performance_kpis": {key: performance_kpis[key] for key in ("measured_count", "not_measured_count", "warning_count", "passed", "area_coverage")},
        "performance_kpi_rows": performance_kpis["rows"],
        "reliability_summary": {key: reliability[key] for key in ("passed", "error_count", "warning_count", "input", "mesh_quality", "features")},
        "reliability_checks": reliability["checks"],
        "reliability_stages": reliability["stages"],
    }


def write_standard_report_bundle(
    result: SolveResult2D,
    output_dir: str | Path | None = None,
    *,
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    out = Path(output_dir) if output_dir is not None else result.output_dir
    out.mkdir(parents=True, exist_ok=True)
    data = build_standard_report_data(result, artifacts=artifacts)
    json_path = out / "standard_report_data.json"
    csv_path = out / "standard_report_sections.csv"
    html_path = out / "standard_report.html"
    pdf_path = out / "standard_report.pdf"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_standard_report_csv(data, csv_path)
    html_path.write_text(_standard_report_html(data), encoding="utf-8")
    write_text_pdf(pdf_path, _standard_report_lines(data), title=message("reports.standard.title"))
    return {"data_json": str(json_path), "csv": str(csv_path), "html": str(html_path), "pdf": str(pdf_path)}


def _input_section(result: SolveResult2D, cfg: Mapping[str, Any]) -> dict[str, Any]:
    mesh_cfg = cfg.get("mesh", {}) if isinstance(cfg.get("mesh", {}), Mapping) else {}
    return {
        "dimension": "2D",
        "node_count": len(result.mesh.node_ids),
        "element_count": len(result.mesh.elements),
        "interface_count": len(result.interfaces),
        "structural_element_count": len(result.structural_elements),
        "mesh_generator": mesh_cfg.get("generator", "explicit"),
        "output_dir": str(result.output_dir),
    }


def _boundary_conditions_section(cfg: Mapping[str, Any]) -> dict[str, Any]:
    rows = cfg.get("boundary_conditions", cfg.get("bc", []))
    rows = rows if isinstance(rows, list) else [rows] if rows else []
    return {"count": len(rows), "items": rows}


def _loads_section(cfg: Mapping[str, Any]) -> dict[str, Any]:
    rows = cfg.get("loads", [])
    rows = rows if isinstance(rows, list) else [rows] if rows else []
    return {"count": len(rows), "items": rows}


def _stage_srm_info(stage: Any) -> Mapping[str, Any]:
    solver = getattr(stage, "solver_info", {})
    solver = solver if isinstance(solver, Mapping) else {}
    srm = solver.get("srm")
    return srm if isinstance(srm, Mapping) else {}


def _stage_sections(result: SolveResult2D) -> list[dict[str, Any]]:
    out = []
    for stage in result.stages:
        max_u = max((float(np.hypot(stage.displacements[i], stage.displacements[i + 1])) for i in range(0, min(stage.displacements.size, len(result.mesh.node_ids) * 2) - 1, 2)), default=0.0)
        max_settlement = float(max((-stage.displacements[1 : min(stage.displacements.size, len(result.mesh.node_ids) * 2) : 2]), default=0.0))
        srm = _stage_srm_info(stage)
        srm_trials = srm.get("trials", []) if srm else []
        out.append(
            {
                "name": stage.name,
                "time": stage.time,
                "output_dir": str(stage.output_dir) if stage.output_dir else "",
                "active_element_count": len(stage.active_elements),
                "solver": stage.solver_info,
                "max_displacement": max_u,
                "max_settlement": max_settlement,
                "max_pore_pressure": None if stage.pore_pressure is None else float(np.max(stage.pore_pressure)),
                "element_result_count": len(stage.element_results),
                "integration_point_result_count": len(stage.integration_point_results),
                "srm_factor_of_safety": srm.get("factor_of_safety") if srm else None,
                "srm_stable_factor": srm.get("stable_factor") if srm else None,
                "srm_failed_factor": srm.get("failed_factor") if srm else None,
                "srm_search_mode": srm.get("search_mode", "") if srm else "",
                "srm_factor_of_safety_status": srm_result_status(srm) if srm else "",
                "srm_factor_of_safety_confidence": srm_result_confidence(srm) if srm else "",
                "srm_factor_of_safety_confirmed": srm_fos_is_confirmed(srm) if srm else False,
                "srm_factor_of_safety_display": srm_fos_display(srm, locale="ja") if srm else "",
                "srm_trial_count": len(srm_trials) if isinstance(srm_trials, list) else 0,
                "srm_trials": srm_trials if isinstance(srm_trials, list) else [],
            }
        )
    return out


def _write_standard_report_csv(data: Mapping[str, Any], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    rows.extend(_flatten_section("input", data.get("input", {})))
    rows.extend(_flatten_section("mesh_quality", data.get("mesh_quality", {})))
    rows.extend(_flatten_section("boundary_conditions", data.get("boundary_conditions", {})))
    rows.extend(_flatten_section("loads", data.get("loads", {})))
    rows.extend(_flatten_section("analysis_settings", data.get("analysis_settings", {})))
    rows.extend(_flatten_section("performance", data.get("performance", {})))
    rows.extend(_flatten_section("performance_kpis", data.get("performance_kpis", {})))
    rows.extend(_flatten_section("reliability_summary", data.get("reliability_summary", {})))
    for index, stage in enumerate(data.get("stages", []), start=1):
        rows.extend(_flatten_section(f"stage[{index}]", stage))
    for index, row in enumerate(data.get("performance_kpi_rows", []), start=1):
        rows.extend(_flatten_section(f"performance_kpi[{index}]", row))
    for index, row in enumerate(data.get("reliability_stages", []), start=1):
        rows.extend(_flatten_section(f"reliability_stage[{index}]", row))
    fields = ["section", "key", "value"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_section(section: str, value: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key, raw in value.items():
            if isinstance(raw, (Mapping, list, tuple)):
                rows.append({"section": section, "key": str(key), "value": json.dumps(raw, ensure_ascii=False, default=str)})
            else:
                rows.append({"section": section, "key": str(key), "value": raw})
    else:
        rows.append({"section": section, "key": "value", "value": value})
    return rows


def _standard_report_html(data: Mapping[str, Any]) -> str:
    input_heading = message("reports.standard.input")
    mesh_heading = message("reports.standard.mesh_quality")
    stages_heading = message("reports.standard.stages")
    performance_heading = message("reports.standard.performance")
    input_rows = _kv_rows(data.get("input", {}))
    quality_rows = _kv_rows(data.get("mesh_quality", {}))
    performance_rows = _kv_rows(data.get("performance", {}))
    performance_kpi_rows = _performance_kpi_table(data.get("performance_kpi_rows", []))
    reliability_rows = _reliability_table(data.get("reliability_stages", []), data.get("reliability_checks", []))
    srm_trial_rows = _srm_trial_report_table(data.get("stages", []))
    stage_rows = []
    for stage in data.get("stages", []):
        if not isinstance(stage, Mapping):
            continue
        stage_rows.append(
            "<tr>"
            f"<td>{html.escape(str(stage.get('name', '')))}</td>"
            f"<td>{html.escape(str(stage.get('active_element_count', '')))}</td>"
            f"<td>{html.escape(str(stage.get('max_displacement', '')))}</td>"
            f"<td>{html.escape(str(stage.get('max_settlement', '')))}</td>"
            f"<td>{html.escape(str((stage.get('solver') or {}).get('method', '') if isinstance(stage.get('solver'), Mapping) else ''))}</td>"
            f"<td>{html.escape(str(stage.get('srm_factor_of_safety_display', '')))}</td>"
            "</tr>"
        )
    violations = []
    for row in data.get("mesh_quality_violations", []):
        if not isinstance(row, Mapping):
            continue
        violations.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('element', '')))}</td>"
            f"<td>{html.escape(str(row.get('severity', '')))}</td>"
            f"<td>{html.escape(str(row.get('reason', '')))}</td>"
            f"<td>{html.escape(str(row.get('repair', '')))}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>{html.escape(str(data.get('title', 'GeoFEM Report')))}</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%;margin:8px 0 18px}}th,td{{border:1px solid #ccc;padding:6px;text-align:left;vertical-align:top}}th{{background:#f3f3f3}}h1,h2{{color:#111827}}</style></head>
<body><h1>{html.escape(str(data.get('title', 'GeoFEM Report')))}</h1>
<h2>{html.escape(input_heading)}</h2>{input_rows}
<h2>{html.escape(mesh_heading)}</h2>{quality_rows}
<table><thead><tr><th>element</th><th>severity</th><th>reason</th><th>repair</th></tr></thead><tbody>{''.join(violations)}</tbody></table>
<h2>{html.escape(stages_heading)}</h2><table><thead><tr><th>stage</th><th>active elements</th><th>max displacement</th><th>max settlement</th><th>method</th><th>FOS</th></tr></thead><tbody>{''.join(stage_rows)}</tbody></table>
{srm_trial_rows}
<h2>{html.escape(performance_heading)}</h2>{performance_rows}
<h2>Performance KPI Matrix</h2>{performance_kpi_rows}
<h2>信頼性サマリ</h2>{reliability_rows}
</body></html>
"""


def _srm_trial_report_table(stages: Any) -> str:
    if not isinstance(stages, list):
        return ""
    summary_rows: list[str] = []
    trial_rows: list[str] = []
    for stage in stages:
        if not isinstance(stage, Mapping) or stage.get("srm_factor_of_safety") is None:
            continue
        summary_rows.append(
            "<tr>"
            f"<td>{html.escape(str(stage.get('name', '')))}</td>"
            f"<td>{html.escape(str(stage.get('srm_factor_of_safety_display', '')))}</td>"
            f"<td>{html.escape(str(stage.get('srm_stable_factor', '')))}</td>"
            f"<td>{html.escape(str(stage.get('srm_failed_factor', '')))}</td>"
            f"<td>{html.escape(str(stage.get('srm_factor_of_safety_status', '')))}</td>"
            f"<td>{html.escape(str(stage.get('srm_factor_of_safety_confidence', '')))}</td>"
            f"<td>{html.escape(str(stage.get('srm_trial_count', '')))}</td>"
            f"<td>{html.escape(str(stage.get('srm_search_mode', '')))}</td>"
            "</tr>"
        )
        trials = stage.get("srm_trials", [])
        if not isinstance(trials, list):
            continue
        for index, raw in enumerate(trials, start=1):
            if not isinstance(raw, Mapping):
                continue
            trial_rows.append(
                "<tr>"
                f"<td>{html.escape(str(stage.get('name', '')))}</td>"
                f"<td>{index}</td>"
                f"<td>{html.escape(str(raw.get('factor', '')))}</td>"
                f"<td>{html.escape(str(raw.get('ok', '')))}</td>"
                f"<td>{html.escape(str(raw.get('converged', '')))}</td>"
                f"<td>{html.escape(str(raw.get('plastic_ratio', '')))}</td>"
                f"<td>{html.escape(str(raw.get('auto_decision', '')))}</td>"
                f"<td>{html.escape(str(raw.get('auto_trial_action', '')))}</td>"
                f"<td>{html.escape(str(raw.get('auto_retry_index', '')))}</td>"
                f"<td>{html.escape(str(raw.get('elapsed_seconds', '')))}</td>"
                f"<td>{html.escape(str(raw.get('estimated_fos_from_last_load', '')))}</td>"
                f"<td>{html.escape(str(raw.get('failure_reason') or raw.get('error') or ''))}</td>"
                "</tr>"
            )
    if not summary_rows:
        return ""
    return (
        "<h2>SRM Trial Results</h2>"
        "<h3>Summary</h3>"
        "<table><thead><tr><th>stage</th><th>FOS</th><th>stable factor</th><th>failed factor</th><th>status</th><th>confidence</th><th>trials</th><th>search mode</th></tr></thead><tbody>"
        + "".join(summary_rows)
        + "</tbody></table>"
        "<h3>Trials</h3>"
        "<table><thead><tr><th>stage</th><th>#</th><th>factor</th><th>OK</th><th>converged</th><th>plastic ratio</th><th>auto decision</th><th>auto action</th><th>retry</th><th>elapsed</th><th>est. FOS</th><th>reason</th></tr></thead><tbody>"
        + "".join(trial_rows)
        + "</tbody></table>"
    )


def _kv_rows(section: Any) -> str:
    rows = []
    if isinstance(section, Mapping):
        for key, value in section.items():
            if isinstance(value, (Mapping, list, tuple)):
                text = json.dumps(value, ensure_ascii=False, default=str)
            else:
                text = str(value)
            rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(text)}</td></tr>")
    return "<table><tbody>" + "".join(rows) + "</tbody></table>"


def _performance_kpi_table(rows_data: Any) -> str:
    rows = []
    if isinstance(rows_data, list):
        for row in rows_data:
            if not isinstance(row, Mapping):
                continue
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('area', '')))}</td>"
                f"<td>{html.escape(str(row.get('metric', '')))}</td>"
                f"<td>{html.escape(str(row.get('value', '')))}</td>"
                f"<td>{html.escape(str(row.get('unit', '')))}</td>"
                f"<td>{html.escape(str(row.get('budget', '')))}</td>"
                f"<td>{html.escape(str(row.get('status', '')))}</td>"
                "</tr>"
            )
    return "<table><thead><tr><th>area</th><th>metric</th><th>value</th><th>unit</th><th>budget</th><th>status</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def _reliability_table(stage_rows_data: Any, check_rows_data: Any) -> str:
    stage_rows = []
    if isinstance(stage_rows_data, list):
        for row in stage_rows_data:
            if not isinstance(row, Mapping):
                continue
            stage_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('stage', '')))}</td>"
                f"<td>{html.escape(str(row.get('converged', '')))}</td>"
                f"<td>{html.escape(str(row.get('residual_norm', '')))}</td>"
                f"<td>{html.escape(str(row.get('max_abs_reaction', '')))}</td>"
                f"<td>{html.escape(str(row.get('mass_balance', '')))}</td>"
                f"<td>{html.escape(str(row.get('total_energy', '')))}</td>"
                "</tr>"
            )
    check_rows = []
    if isinstance(check_rows_data, list):
        for row in check_rows_data:
            if not isinstance(row, Mapping):
                continue
            check_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('id', '')))}</td>"
                f"<td>{html.escape(str(row.get('severity', '')))}</td>"
                f"<td>{html.escape(str(row.get('passed', '')))}</td>"
                f"<td>{html.escape(str(row.get('detail', '')))}</td>"
                "</tr>"
            )
    return (
        "<table><thead><tr><th>stage</th><th>converged</th><th>residual</th><th>max reaction</th><th>mass balance</th><th>total energy</th></tr></thead><tbody>"
        + "".join(stage_rows)
        + "</tbody></table>"
        + "<table><thead><tr><th>check</th><th>severity</th><th>passed</th><th>detail</th></tr></thead><tbody>"
        + "".join(check_rows)
        + "</tbody></table>"
    )


def _standard_report_lines(data: Mapping[str, Any]) -> list[str]:
    lines = [str(data.get("title", "GeoFEM 2D Standard Report")), ""]
    for section_name in ("input", "mesh_quality", "performance", "performance_kpis", "reliability_summary", "analysis_log"):
        lines.append(section_name)
        section = data.get(section_name, {})
        if isinstance(section, Mapping):
            for key, value in section.items():
                if isinstance(value, (Mapping, list, tuple)):
                    value = json.dumps(value, ensure_ascii=False, default=str)
                lines.append(f"  {key}: {value}")
        lines.append("")
    lines.append("stages")
    for stage in data.get("stages", []):
        if isinstance(stage, Mapping):
            solver = stage.get("solver", {})
            method = solver.get("method", "") if isinstance(solver, Mapping) else ""
            fos = stage.get("srm_factor_of_safety")
            fos_text = "" if fos is None else f", {stage.get('srm_factor_of_safety_display', '')}, srm_trials={stage.get('srm_trial_count', 0)}"
            lines.append(f"  {stage.get('name')}: method={method}, max_u={stage.get('max_displacement')}, max_settlement={stage.get('max_settlement')}{fos_text}")
            trials = stage.get("srm_trials", [])
            if isinstance(trials, list):
                for index, raw in enumerate(trials, start=1):
                    if isinstance(raw, Mapping):
                        lines.append(
                            f"    SRM trial {index}: factor={raw.get('factor')}, ok={raw.get('ok')}, "
                            f"converged={raw.get('converged')}, plastic_ratio={raw.get('plastic_ratio')}, "
                            f"reason={raw.get('failure_reason') or raw.get('error') or ''}"
                        )
    warnings = data.get("warnings", [])
    if warnings:
        lines.append("")
        lines.append("warnings")
        for warning in warnings:
            lines.append(f"  {warning}")
    return _wrap_lines(lines)


def _wrap_lines(lines: list[str], width: int = 92) -> list[str]:
    out: list[str] = []
    for line in lines:
        text = str(line)
        while len(text) > width:
            out.append(text[:width])
            text = "  " + text[width:]
        out.append(text)
    return out


__all__ = ["build_standard_report_data", "write_standard_report_bundle"]
