"""Commercial performance KPI matrix for standard reports and quality gates."""

from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any, Mapping

from .fem2d_types import SolveResult2D
from .performance_monitor import build_performance_summary


KPI_AREAS = ("cold_run", "warm_solver", "solver_profile", "gui", "post", "report", "cad_mesh", "numba")


def build_result_performance_kpi_matrix(
    result: SolveResult2D,
    *,
    artifacts: Mapping[str, Any] | None = None,
    performance_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a cross-area performance KPI table for one solved case.

    The matrix intentionally accepts optional runtime measurements from the
    input config so GUI/CAD/Post/report timings can be attached without making
    the solver depend on GUI internals.
    """

    perf = dict(performance_summary or build_performance_summary(result))
    cfg = result.input_config if isinstance(result.input_config, Mapping) else {}
    runtime = _runtime_measurements(cfg)
    budgets = _mapping(cfg.get("performance_budgets", cfg.get("performance_targets", {})))
    out = result.output_dir
    rows: list[dict[str, Any]] = []

    _add_metric(
        rows,
        area="cold_run",
        metric="elapsed_seconds",
        label="End-to-end run elapsed seconds",
        value=_first_number(runtime, "cold_elapsed_seconds", "run_elapsed_seconds", fallback=perf.get("elapsed_seconds")),
        unit="s",
        source="runtime_measurements or solver stage sum",
        budget=_budget(budgets, "cold_run.elapsed_seconds", "run_elapsed_seconds"),
    )
    _add_metric(
        rows,
        area="warm_solver",
        metric="stage_elapsed_seconds",
        label="Warm solver stage elapsed seconds",
        value=perf.get("stage_elapsed_seconds", perf.get("elapsed_seconds")),
        unit="s",
        source="performance_summary.json",
        budget=_budget(budgets, "warm_solver.stage_elapsed_seconds", "solver_elapsed_seconds"),
    )
    _add_metric(
        rows,
        area="warm_solver",
        metric="total_solver_iterations",
        label="Total solver iterations",
        value=perf.get("total_solver_iterations"),
        unit="count",
        source="performance_summary.json",
        budget=_budget(budgets, "warm_solver.total_solver_iterations", "solver_iterations"),
    )
    _add_metric(
        rows,
        area="warm_solver",
        metric="max_matrix_nnz",
        label="Maximum sparse matrix nonzeros",
        value=perf.get("max_matrix_nnz"),
        unit="nnz",
        source="performance_summary.json",
        budget=_budget(budgets, "warm_solver.max_matrix_nnz", "matrix_nnz"),
    )
    _add_metric(
        rows,
        area="warm_solver",
        metric="estimated_memory_bytes",
        label="Estimated solver/result memory",
        value=perf.get("estimated_memory_bytes"),
        unit="bytes",
        source="performance_summary.json",
        budget=_budget(budgets, "warm_solver.estimated_memory_bytes", "estimated_memory_bytes"),
    )
    _add_metric(
        rows,
        area="solver_profile",
        metric="dominant_category_elapsed_seconds",
        label="Dominant solver performance category elapsed seconds",
        value=perf.get("dominant_category_elapsed_seconds"),
        unit="s",
        source="performance_summary.profile",
        budget=_budget(budgets, "solver_profile.dominant_category_elapsed_seconds"),
    )
    _add_metric(
        rows,
        area="solver_profile",
        metric="cache_reuse_count",
        label="Solver cache reuse observations",
        value=perf.get("cache_reuse_count"),
        unit="count",
        source="performance_summary.profile.cache",
        budget=None,
        larger_is_better=True,
    )
    _add_metric(
        rows,
        area="solver_profile",
        metric="cache_miss_count",
        label="Solver cache miss observations",
        value=perf.get("cache_miss_count"),
        unit="count",
        source="performance_summary.profile.cache",
        budget=_budget(budgets, "solver_profile.cache_miss_count"),
    )
    _add_metric(
        rows,
        area="solver_profile",
        metric="iteration_profile_count",
        label="Recorded nonlinear/time-step iteration profile rows",
        value=perf.get("iteration_profile_count"),
        unit="count",
        source="performance_summary.profile.iterations",
        budget=None,
        larger_is_better=True,
    )
    _add_metric(
        rows,
        area="gui",
        metric="max_response_ms",
        label="GUI observed maximum response time",
        value=_first_number(runtime, "gui_response_ms", "gui_max_response_ms"),
        unit="ms",
        source="runtime_measurements",
        budget=_budget(budgets, "gui.max_response_ms", "gui_response_ms", default=250.0),
    )
    _add_metric(
        rows,
        area="post",
        metric="output_elapsed_seconds",
        label="Post output elapsed seconds",
        value=_first_number(runtime, "post_elapsed_seconds", "post_output_elapsed_seconds"),
        unit="s",
        source="runtime_measurements",
        budget=_budget(budgets, "post.output_elapsed_seconds", "post_elapsed_seconds"),
    )
    _add_metric(
        rows,
        area="post",
        metric="artifact_count",
        label="Post/result view artifact count",
        value=_artifact_count(out, ("result_view_index.json", "result_view_index.html", "post_case_comparison.csv")),
        unit="count",
        source="output directory",
        budget=None,
        larger_is_better=True,
    )
    _add_metric(
        rows,
        area="report",
        metric="output_elapsed_seconds",
        label="Report output elapsed seconds",
        value=_first_number(runtime, "report_elapsed_seconds", "report_output_elapsed_seconds"),
        unit="s",
        source="runtime_measurements",
        budget=_budget(budgets, "report.output_elapsed_seconds", "report_elapsed_seconds"),
    )
    _add_metric(
        rows,
        area="report",
        metric="artifact_count",
        label="Report artifact count",
        value=_artifact_count(out, ("calculation_report.html", "calculation_report.pdf", "standard_report.html", "standard_report.pdf")),
        unit="count",
        source="output directory",
        budget=None,
        larger_is_better=True,
    )
    _add_metric(
        rows,
        area="cad_mesh",
        metric="mesh_generation_elapsed_seconds",
        label="CAD/mesh generation elapsed seconds",
        value=_first_number(runtime, "mesh_generation_elapsed_seconds", "cad_mesh_elapsed_seconds"),
        unit="s",
        source="runtime_measurements",
        budget=_budget(budgets, "cad_mesh.mesh_generation_elapsed_seconds", "mesh_generation_elapsed_seconds"),
    )
    _add_metric(
        rows,
        area="cad_mesh",
        metric="element_count",
        label="Mesh element count",
        value=perf.get("element_count"),
        unit="count",
        source="performance_summary.json",
        budget=None,
        larger_is_better=True,
    )
    _add_metric(
        rows,
        area="numba",
        metric="warmup_elapsed_seconds",
        label="Numba/kernel cold compile elapsed seconds",
        value=_first_number(runtime, "numba_warmup_elapsed_seconds", "kernel_warmup_elapsed_seconds"),
        unit="s",
        source="runtime_measurements or standard benchmark KPI",
        budget=_budget(budgets, "numba.warmup_elapsed_seconds", "numba_warmup_elapsed_seconds"),
    )

    covered = {row["area"] for row in rows}
    measured = [row for row in rows if row["status"] != "not_measured"]
    warnings = [row for row in rows if row["status"] == "warning"]
    return {
        "schema": "geofem.performance_kpi_matrix.v1",
        "case": str(perf.get("case", "")),
        "output_dir": str(out),
        "area_coverage": {area: area in covered for area in KPI_AREAS},
        "measured_count": len(measured),
        "not_measured_count": len(rows) - len(measured),
        "warning_count": len(warnings),
        "passed": len(warnings) == 0,
        "features": [
            "cold_warm_solver_kpi",
            "gui_response_kpi_slot",
            "post_report_cad_mesh_kpi_slots",
            "numba_compile_warm_kpi_slot",
            "solver_profile_breakdown",
            "standard_report_embedded_kpi_table",
        ],
        "artifacts": dict(artifacts or {}),
        "rows": rows,
    }


def write_performance_kpi_reports(
    result: SolveResult2D,
    output_dir: str | Path | None = None,
    *,
    artifacts: Mapping[str, Any] | None = None,
    performance_summary: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    out = Path(output_dir) if output_dir is not None else result.output_dir
    out.mkdir(parents=True, exist_ok=True)
    matrix = build_result_performance_kpi_matrix(result, artifacts=artifacts, performance_summary=performance_summary)
    json_path = out / "performance_kpi_matrix.json"
    csv_path = out / "performance_kpi_matrix.csv"
    html_path = out / "performance_kpi_matrix.html"
    json_path.write_text(json.dumps(matrix, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_kpi_csv(matrix, csv_path)
    html_path.write_text(_kpi_html(matrix), encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "html": str(html_path)}


def _add_metric(
    rows: list[dict[str, Any]],
    *,
    area: str,
    metric: str,
    label: str,
    value: Any,
    unit: str,
    source: str,
    budget: float | None,
    larger_is_better: bool = False,
) -> None:
    numeric = _number_or_none(value)
    status = "not_measured" if numeric is None else "measured"
    if numeric is not None and budget is not None:
        status = "passed" if (numeric >= budget if larger_is_better else numeric <= budget) else "warning"
    rows.append(
        {
            "area": area,
            "metric": metric,
            "label": label,
            "value": "" if numeric is None else numeric,
            "unit": unit,
            "budget": "" if budget is None else budget,
            "status": status,
            "source": source,
        }
    )


def _write_kpi_csv(matrix: Mapping[str, Any], path: Path) -> None:
    fields = ["area", "metric", "label", "value", "unit", "budget", "status", "source"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in matrix.get("rows", []):
            if isinstance(row, Mapping):
                writer.writerow({field: row.get(field, "") for field in fields})


def _kpi_html(matrix: Mapping[str, Any]) -> str:
    rows = []
    for row in matrix.get("rows", []):
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
            f"<td>{html.escape(str(row.get('source', '')))}</td>"
            "</tr>"
        )
    coverage = ", ".join(f"{area}={covered}" for area, covered in matrix.get("area_coverage", {}).items())
    return f"""<!doctype html>
<html lang="ja"><head><meta charset="utf-8"><title>GeoFEM Performance KPI Matrix</title>
<style>body{{font-family:Arial,'Yu Gothic',sans-serif;margin:24px;line-height:1.45}}table{{border-collapse:collapse;width:100%}}th,td{{border:1px solid #ccc;padding:6px;text-align:left}}th{{background:#f3f3f3}}</style></head>
<body><h1>GeoFEM Performance KPI Matrix</h1>
<p>passed={html.escape(str(matrix.get('passed', '')))}, measured={html.escape(str(matrix.get('measured_count', '')))}, not_measured={html.escape(str(matrix.get('not_measured_count', '')))}</p>
<p>{html.escape(coverage)}</p>
<table><thead><tr><th>area</th><th>metric</th><th>value</th><th>unit</th><th>budget</th><th>status</th><th>source</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
</body></html>
"""


def _runtime_measurements(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("runtime_measurements", "performance_kpi", "performance_measurements"):
        value = cfg.get(key, {})
        if isinstance(value, Mapping):
            return value
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _budget(budgets: Mapping[str, Any], *keys: str, default: float | None = None) -> float | None:
    for key in keys:
        value: Any = budgets
        for part in key.split("."):
            if isinstance(value, Mapping):
                value = value.get(part)
            else:
                value = None
                break
        number = _number_or_none(value)
        if number is not None:
            return number
    return default


def _first_number(source: Mapping[str, Any], *keys: str, fallback: Any = None) -> float | None:
    for key in keys:
        number = _number_or_none(source.get(key))
        if number is not None:
            return number
    return _number_or_none(fallback)


def _number_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _artifact_count(root: Path, names: tuple[str, ...]) -> int:
    return sum(1 for name in names if (root / name).exists())


__all__ = ["KPI_AREAS", "build_result_performance_kpi_matrix", "write_performance_kpi_reports"]
